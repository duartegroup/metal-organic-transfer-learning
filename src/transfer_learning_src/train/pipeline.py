from typing import Dict, Any, Tuple
import torch
from .config import TrainingConfig
from .phases import TrainingPhase, PretrainingPhase, CCSAPhase, FinetunePhase
from .results import ResultsManager
from model.multitask import MultiTaskPocket
from data.utils_data import (
    load_dataset,
    compute_source_scaling_stats,
    split_data_by_group_random,
    prepare_datasets,
    create_dataloaders,
    get_datasets,
)
from data.dataset import WrappedDataLoader
from utils.checkpoint_utils import save_phase_stats
import logging
import gc

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MultitaskTrainingPipeline:
    """
    Main pipeline orchestrator for multitask training with domain adaptation.

    This class manages the complete training pipeline including pretraining on source
    domain, CCSA (Contrastive Semantic Alignment) fine-tuning, and target domain fine-tuning.
    Supports optional consistent scaling across domains and various training strategies.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize the multitask training pipeline.

        Parameters
        ----------
        params : Dict[str, Any]
            Configuration dictionary containing training parameters including:
            - data_scaler: Scaling method for features (e.g., 'standard', 'minmax')
            - source_smote: Whether to apply SMOTE to source domain
            - target_smote: Whether to apply SMOTE to target domain
            - cross_domain_scaling: Whether to use consistent scaling across domains
            - fold: Cross-validation fold number
            - ccsa_followup_modes: List of follow-up training modes after CCSA
            - run_types: List of run types to execute
            - inhibitor_descriptor: Type of molecular descriptor
            - organic_percent_of_data: Fraction of source data to use
            - modality: Feature modalities to use (e.g., 'mol_protein_pocket')
            - affinity_type: Type of binding affinity ('pic50', 'pk', or 'pic50+pk')

        Returns
        -------
        None
        """
        self.params = params
        self.config = TrainingConfig()

        # Fix: Update the scaler based on the params
        self.config.data_scaler = params.get("data_scaler", None)
        # NEW: Update the new options
        self.config.source_smote = params.get("source_smote", False)
        self.config.target_smote = params.get("target_smote", False)
        self.config.cross_domain_scaling = params.get("cross_domain_scaling", False)

        # Initialize phases
        self.pretrain_phase = PretrainingPhase(self.config, params)
        self.ccsa_phase = CCSAPhase(self.config, params)
        self.finetune_phase = FinetunePhase(self.config, params)

        # Store original fold for restoration
        self.original_fold = params.get("fold", 1)

        # Parse follow-up modes
        self.ccsa_followup_modes = params.get(
            "ccsa_followup_modes", ["task", "encoder_task"]
        )

        # Cache for source data (optimization) - separate caches for different split ratios
        self._cached_source_data = {}  # Will store by affinity_type as key

        # Consistent scaling option
        self._global_scaling_stats = None

        # Add parameter to control which run types to execute
        self.run_types = params.get(
            "run_types", ["full_pipeline"]
        )  # Default to full pipeline

    def _compute_global_scaling_stats(self) -> None:
        """
        Compute global scaling statistics from source domain data.

        This method loads source data (both pIC50 and pKd), combines them, and computes
        scaling statistics that will be used consistently across all experiments.
        The computed statistics and loaded data are cached for reuse.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        logger.info("Computing global scaling statistics from source domain")

        # Load source data just for computing scaling stats
        pic50_data, pk_data = load_dataset(
            "organic",
            self.params.get("inhibitor_descriptor", "mace"),
            affinity_type="pic50+pk",
            organic_percent_of_data=self.params.get("organic_percent_of_data", 1.0),
        )

        # cache both datasets.
        self._cached_source_data["pic50"] = pic50_data
        self._cached_source_data["pk"] = pk_data

        # combine the two datasets
        source_raw = {
            "inhibitor_feats": {
                **pic50_data["inhibitor_feats"],
                **pk_data["inhibitor_feats"],
            },
            "protein_feats": {
                **pic50_data["protein_feats"],
                **pk_data["protein_feats"],
            },
            "pocket_feats": {**pic50_data["pocket_feats"], **pk_data["pocket_feats"]},
            "binding_affs": {**pic50_data["binding_affs"], **pk_data["binding_affs"]},
        }

        # Compute scaling stats from source data
        self._global_scaling_stats = compute_source_scaling_stats(
            source_raw,
            self.params.get("modality", "mol_protein_pocket"),
            scaling_method=self.params.get("data_scaler", None),
        )

    def _get_global_scaling_stats(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Retrieve cached global scaling statistics and source data.

        Parameters
        ----------
        None

        Returns
        -------
        Tuple[Any, Dict[str, Any]]
            A tuple containing:
            - Global scaling statistics object
            - Dictionary of raw source data with keys: 'inhibitor_feats', 'protein_feats',
              'pocket_feats', 'binding_affs'

        Raises
        ------
        ValueError
            If global scaling statistics have not been computed yet.
        """
        if self._global_scaling_stats is not None and hasattr(
            self, "_cached_source_raw"
        ):
            return self._global_scaling_stats, self._cached_source_raw
        else:
            raise ValueError("Global scaling stats not computed yet")

    def _load_source_data_pretraining(
        self, modality: str, split_ratio: float = 1.0
    ) -> Dict[str, Any]:
        """
        Load source data for pretraining with specified train/val split ratio.

        This method loads and caches source domain data (both pIC50 and pKd combined)
        for pretraining. It applies the pre-computed global scaling statistics and
        creates dataloaders for training and validation.

        Parameters
        ----------
        modality : str
            Feature modalities to use (e.g., 'mol_protein_pocket').
        split_ratio : float, optional
            Ratio for train/val split (0.9 for 90/10 split, 1.0 for no split),
            by default 1.0.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing 'src_train' and 'src_val' dataloaders.
        """
        # Check if already cached
        split_desc = (
            f"{int(split_ratio*100)}/{100-int(split_ratio*100)} split"
            if split_ratio < 1.0
            else "no split"
        )
        logger.info(f"Loading source data ({split_desc})")

        # Combine both cached dataset
        source_raw = {
            "inhibitor_feats": {
                **self._cached_source_data["pic50"]["inhibitor_feats"],
                **self._cached_source_data["pk"]["inhibitor_feats"],
            },
            "protein_feats": {
                **self._cached_source_data["pic50"]["protein_feats"],
                **self._cached_source_data["pk"]["protein_feats"],
            },
            "pocket_feats": {
                **self._cached_source_data["pic50"]["pocket_feats"],
                **self._cached_source_data["pk"]["pocket_feats"],
            },
            "binding_affs": {
                **self._cached_source_data["pic50"]["binding_affs"],
                **self._cached_source_data["pk"]["binding_affs"],
            },
        }

        # Create temporary params for source data loading
        temp_params = self.params.copy()
        temp_params.update({"train_on": "src"})

        # Split source data according to the specified ratio
        src_train, src_val = split_data_by_group_random(
            source_raw["inhibitor_feats"],
            source_raw["protein_feats"],
            source_raw["pocket_feats"],
            source_raw["binding_affs"],
            seed=temp_params.get("seed", 42),
            split_ratio=split_ratio,
        )

        # Build datasets using the pre-computed global stats and already-loaded data
        datasets, _ = prepare_datasets(
            src_train,
            src_val,
            tgt_train=None,
            tgt_val=None,  # Only source data
            modality=modality,
            scaling_method=self.params.get("data_scaler", None),
            scaling_stats=self._global_scaling_stats,
            target_smote=False,
            source_smote=False,
            cross_domain_scaling=True,  # does not matter because src is only used in pretrain
        )
        src_train_dataset, src_val_dataset, _, _ = datasets

        temp_params.update({"batch_size": self.config.pretrain_batch_size})

        # Create source dataloaders
        source_loaders = create_dataloaders(
            src_train_dataset, src_val_dataset, None, None, temp_params
        )

        logger.info(f"Source data loaded and cached ({split_desc})")

        return source_loaders

    def _load_source_data_domain_adaptation(
        self, modality: str, split_ratio: float = 1.0, pretrain: bool = False
    ) -> Tuple[Any, Any]:
        """
        Load source data for domain adaptation (CCSA) with specified split ratio.

        This method loads source domain data for a specific affinity type (pIC50 or pKd)
        for use in domain adaptation phases like CCSA training.

        Parameters
        ----------
        modality : str
            Feature modalities to use (e.g., 'mol_protein_pocket').
        split_ratio : float, optional
            Ratio for train/val split (0.9 for 90/10 split, 1.0 for no split),
            by default 1.0.
        pretrain : bool, optional
            Whether this is for pretraining phase, by default False.

        Returns
        -------
        Tuple[Any, Any]
            A tuple containing (src_train_dataset, src_val_dataset).
        """
        # Check if already cached
        # Get both scaling stats and source data in one call
        source_raw = self._cached_source_data[self.params.get("affinity_type", "pic50")]

        # Create temporary params for source data loading
        temp_params = self.params.copy()
        temp_params.update({"train_on": "src"})

        # Split source data according to the specified ratio
        src_train, src_val = split_data_by_group_random(
            source_raw["inhibitor_feats"],
            source_raw["protein_feats"],
            source_raw["pocket_feats"],
            source_raw["binding_affs"],
            seed=temp_params.get("seed", 1996),
            split_ratio=split_ratio,
        )

        # Build datasets using the pre-computed global stats and already-loaded data
        datasets, _ = prepare_datasets(
            src_train,
            src_val,
            tgt_train=None,
            tgt_val=None,  # Only source data
            modality=modality,
            scaling_method=self.params.get("data_scaler", None),
            scaling_stats=self._global_scaling_stats,
            target_smote=self.config.target_smote,
            source_smote=self.config.source_smote,
            cross_domain_scaling=self.config.cross_domain_scaling,
        )
        src_train_dataset, src_val_dataset, _, _ = datasets

        return src_train_dataset, src_val_dataset

    def run(self) -> Dict[str, Any]:
        """
        Execute the complete multitask training pipeline with all configured approaches.

        This is the main entry point for running the training pipeline. It orchestrates
        the execution of different training approaches including:
        - Full pipeline with pretraining on source domain
        - CCSA (Classification and Contrastive Semantic Alignment) fine-tuning
        - Standard fine-tuning on target domain
        - Various combinations and ablations based on configuration

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, Any]
            Dictionary containing results from all training approaches executed,
            with keys indicating the approach and values containing metrics and
            performance statistics.
        """
        logger.info("=" * 80)
        logger.info(f"      MULTITASK TRAINING PIPELINE")
        logger.info("=" * 80)

        all_results = {}

        try:
            # Always compute global scaling stats for consistency
            self._compute_global_scaling_stats()

            # Determine which run types to execute
            run_types = self.run_types

            if "full_pipeline" in run_types:
                # Run ALL training approaches for comprehensive comparison
                logger.info("Running FULL PIPELINE (all training approaches)")

                # 1. Original pipeline with pretraining
                logger.info("=" * 80)
                logger.info("       PHASE 1: WITH PRETRAINING")
                logger.info("=" * 80)
                pretrain_results = self._run_full_pipeline_with_pretraining()
                all_results.update(pretrain_results)

                # 2. All no-pretraining approaches
                logger.info("=" * 80)
                logger.info("       PHASE 2: WITHOUT PRETRAINING APPROACHES")
                logger.info("=" * 80)

                # No Pretrain CCSA
                no_pretrain_ccsa_results = self._run_no_pretrain_ccsa()
                all_results.update(no_pretrain_ccsa_results)

                # No Pretrain CCSA + Encoder Task
                no_pretrain_ccsa_encoder_task_results = (
                    self._run_no_pretrain_ccsa_with_followup("encoder_task")
                )
                all_results.update(no_pretrain_ccsa_encoder_task_results)

                # No Pretrain CCSA + Task
                no_pretrain_ccsa_task_results = (
                    self._run_no_pretrain_ccsa_with_followup("task")
                )
                all_results.update(no_pretrain_ccsa_task_results)

                # No Pretrain Encoder Task
                no_pretrain_encoder_task_results = self._run_no_pretrain_encoder_task()
                all_results.update(no_pretrain_encoder_task_results)

            # Individual run types (for backwards compatibility or specific testing)
            elif "no_pretrain_ccsa" in run_types:
                # No Pretrain CCSA: CCSA without pretraining, no followup
                no_pretrain_ccsa_results = self._run_no_pretrain_ccsa()
                all_results.update(no_pretrain_ccsa_results)

            elif "no_pretrain_ccsa_encoder_task" in run_types:
                # No Pretrain CCSA + Encoder Task: CCSA without pretraining + encoder_task followup
                no_pretrain_ccsa_encoder_task_results = (
                    self._run_no_pretrain_ccsa_with_followup("encoder_task")
                )
                all_results.update(no_pretrain_ccsa_encoder_task_results)

            elif "no_pretrain_ccsa_task" in run_types:
                # No Pretrain CCSA + Task: CCSA without pretraining + task followup
                no_pretrain_ccsa_task_results = (
                    self._run_no_pretrain_ccsa_with_followup("task")
                )
                all_results.update(no_pretrain_ccsa_task_results)

            elif "no_pretrain_encoder_task" in run_types:
                # No Pretrain Encoder Task: Direct fine-tuning on target domain only
                no_pretrain_encoder_task_results = self._run_no_pretrain_encoder_task()
                all_results.update(no_pretrain_encoder_task_results)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            # Restore original fold
            self.params["fold"] = self.original_fold

        # Generate results
        ResultsManager.generate_summary(all_results, self.params)
        ResultsManager.save_results(all_results, self.params)

        return all_results

    def _clear_unused_cached_data(self) -> None:
        """
        Remove unused source dataset from cache to free up memory after pre-training.

        This method determines which affinity type (pIC50 or pKd) is not required for the
        remaining training phases and removes it from the cache to reduce memory usage.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        required_affinity = self.params.get("affinity_type")

        # Determine which affinity type is NOT required for the rest of the run
        unused_affinity = None
        if required_affinity == "pic50" and "pk" in self._cached_source_data:
            unused_affinity = "pk"
        elif required_affinity == "pk" and "pic50" in self._cached_source_data:
            unused_affinity = "pic50"

        if unused_affinity:
            try:
                del self._cached_source_data[unused_affinity]
                gc.collect()  # Force garbage collection
                logger.info(
                    f"Cleared unused '{unused_affinity}' dataset from cache to free memory."
                )
            except KeyError:
                logger.warning(
                    f"Attempted to clear '{unused_affinity}' from cache, but it was not found."
                )

    def _run_full_pipeline_with_pretraining(self) -> Dict[str, Any]:
        """
        Run the complete pipeline with pretraining on source domain.

        This method executes the full training workflow including source pretraining,
        CCSA domain adaptation, and various fine-tuning approaches across all folds.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, Any]
            Dictionary containing results from all training phases with keys:
            - 'pretrain_src': Source pretraining metrics
            - 'pretrain_ccsa': CCSA training results per fold
            - 'pretrain_ccsa_then_{mode}': Follow-up training results per mode and fold
            - 'pretrain_{method}': Standard fine-tuning results per method and fold
        """
        all_results = {"pretrain_src": {}}

        # Phase 1: Source pretraining (once) - will use cached data
        source_model, pretrain_metrics = self._run_source_pretraining()
        all_results["pretrain_src"]["single_run"] = pretrain_metrics

        # Clear unused cached data
        self._clear_unused_cached_data()

        # Phase 2: Cross-fold training - will use cached data
        ccsa_results, followup_results = self._run_ccsa_fold_training(source_model)
        all_results["pretrain_ccsa"] = ccsa_results

        # Add followup results
        for mode, fold_results in followup_results.items():
            all_results[f"pretrain_ccsa_then_{mode}"] = fold_results

        # Phase 3: Standard fine-tuning methods - will use cached stats
        standard_finetune_results = self._run_standard_finetuning(source_model)
        all_results.update(standard_finetune_results)

        return all_results

    def _run_source_pretraining(self) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """
        Handle source domain pretraining or load pre-existing model.

        This method either loads a previously trained source model from disk or
        executes pretraining on the combined source domain data (pIC50 + pKd).

        Parameters
        ----------
        None

        Returns
        -------
        Tuple[torch.nn.Module, Dict[str, Any]]
            A tuple containing:
            - Pre-trained source model
            - Dictionary of pretraining metrics (or skip indicator if loaded from disk)
        """
        base_phase = TrainingPhase(self.config, self.params)
        base_results_dir = base_phase.create_results_base_dir()

        pretrained_model_path = base_results_dir / "pretrain_src"

        if pretrained_model_path.exists():
            logger.info(
                f"Loading pre-trained source model from {pretrained_model_path}"
            )
            source_model = MultiTaskPocket(
                domain="src", self_ssl=True, temperature=0.07, **self.params
            )
            try:
                load_status = self.pretrain_phase.component_manager.load_components_from_directory(
                    model=source_model, load_dir=pretrained_model_path, strict=False
                )
                if any(load_status.values()):
                    logger.info(
                        f"Successfully loaded pre-trained source model. Load status: {load_status}"
                    )
                    return source_model, {"pretraining_skipped": True}
                else:
                    raise RuntimeError("No components were successfully loaded")
            except Exception as e:
                logger.error(f"Failed to load pre-trained model: {e}")
                raise

        else:
            source_loaders = self._load_source_data_pretraining(
                self.params["modality"], split_ratio=0.9
            )
            return self.pretrain_phase.run(
                source_loaders=source_loaders,
            )

    def _run_ccsa_fold_training(
        self, source_model: torch.nn.Module
    ) -> Tuple[Dict[int, Any], Dict[str, Dict[int, Any]]]:
        """
        Run CCSA domain adaptation and follow-up training across all cross-validation folds.

        This method performs CCSA training for each fold using the pre-trained source model,
        then optionally runs follow-up fine-tuning methods on the CCSA-adapted models.

        Parameters
        ----------
        source_model : torch.nn.Module
            Pre-trained model from source domain pretraining phase.

        Returns
        -------
        Tuple[Dict[int, Any], Dict[str, Dict[int, Any]]]
            A tuple containing:
            - CCSA results dictionary with fold numbers as keys
            - Follow-up results dictionary with mode names as keys, each containing fold results
        """
        ccsa_results = {}
        followup_results = {mode: {} for mode in self.ccsa_followup_modes}

        for fold in range(1, self.params["n_folds"] + 1):
            logger.info("=" * 80)
            logger.info(f"      CCSA FOLD {fold}/{self.params['n_folds']}")
            logger.info("=" * 80)

            # Create clean model copy for this fold
            fold_model = MultiTaskPocket(
                domain="src", self_ssl=True, temperature=0.07, **self.params
            )
            fold_model.load_state_dict(source_model.state_dict(), strict=False)
            logger.info(f"Loaded source model for fold {fold}")

            # Load source data
            src_train_dataset, src_val_dataset = (
                self._load_source_data_domain_adaptation(
                    self.params["modality"], split_ratio=1.0, pretrain=True
                )
            )

            # Run CCSA with cached source data (full dataset)
            best_ccsa_model, ccsa_metrics = self.ccsa_phase.run_with_cached_source(
                src_train_dataset,
                src_val_dataset,
                fold,
                fold_model,
                self._global_scaling_stats,
            )
            ccsa_results[fold] = ccsa_metrics

            # Run followup methods
            for mode in self.ccsa_followup_modes:
                logger.info("=" * 80)
                logger.info(
                    f"      CCSA FOLLOW-UP: {mode.upper()} FOLD {fold}/{self.params['n_folds']}"
                )
                logger.info("=" * 80)

                followup_metrics = self._run_finetune(
                    fold=fold,
                    pretrained_model=best_ccsa_model,
                    finetune_mode=mode,
                    global_scaling_stats=self._global_scaling_stats,
                    training_path="pretrain_ccsa",
                )
                followup_results[mode][fold] = followup_metrics

        return ccsa_results, followup_results

    def _run_no_pretrain_ccsa(self) -> Dict[str, Any]:
        """
        Run CCSA domain adaptation without pretraining (random initialization).

        This method performs CCSA training across all folds starting from randomly
        initialized weights, with no follow-up fine-tuning.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, Any]
            Dictionary with key 'no_pretrain_ccsa' containing fold-wise CCSA results.
        """
        ccsa_results = {}
        global_stats = self._global_scaling_stats

        for fold in range(1, self.params["n_folds"] + 1):
            logger.info("=" * 80)
            logger.info(
                f"      NO PRETRAIN CCSA FOLD {fold}/{self.params['n_folds']} (NO PRETRAINING)"
            )
            logger.info("=" * 80)

            # Run CCSA without pretraining
            src_train_dataset, src_val_dataset = (
                self._load_source_data_domain_adaptation(
                    self.params["modality"], split_ratio=1.0, pretrain=False
                )
            )
            best_ccsa_model, ccsa_metrics = self.ccsa_phase.run_without_pretraining(
                src_train_dataset, src_val_dataset, fold, global_stats
            )
            ccsa_results[fold] = ccsa_metrics

        return {"no_pretrain_ccsa": ccsa_results}

    def _run_no_pretrain_ccsa_with_followup(self, followup_mode: str) -> Dict[str, Any]:
        """
        Run CCSA without pretraining followed by specific fine-tuning mode.

        This method performs CCSA training from random initialization across all folds,
        followed by a specific follow-up fine-tuning strategy.

        Parameters
        ----------
        followup_mode : str
            Fine-tuning mode to apply after CCSA ('encoder_task' or 'task').

        Returns
        -------
        Dict[str, Any]
            Dictionary with key 'no_pretrain_ccsa_then_{followup_mode}' containing
            fold-wise follow-up fine-tuning results.
        """
        ccsa_results = {}
        followup_results = {}
        global_stats = self._global_scaling_stats

        for fold in range(1, self.params["n_folds"] + 1):
            logger.info("=" * 80)
            logger.info(
                f"      NO PRETRAIN CCSA + {followup_mode.upper()} FOLD {fold}/{self.params['n_folds']} (NO PRETRAINING)"
            )
            logger.info("=" * 80)

            src_train_dataset, src_val_dataset = (
                self._load_source_data_domain_adaptation(
                    self.params["modality"], split_ratio=1.0, pretrain=False
                )
            )

            # Run CCSA without pretraining
            best_ccsa_model, ccsa_metrics = self.ccsa_phase.run_without_pretraining(
                src_train_dataset, src_val_dataset, fold, global_stats
            )
            ccsa_results[fold] = ccsa_metrics

            # Run followup method
            logger.info("=" * 80)
            logger.info(
                f"  FOLLOW-UP: {followup_mode.upper()} AFTER NO PRETRAIN CCSA FOLD {fold}/{self.params['n_folds']}"
            )
            logger.info("=" * 80)

            followup_metrics = self._run_finetune(
                fold=fold,
                pretrained_model=best_ccsa_model,
                finetune_mode=followup_mode,
                global_scaling_stats=global_stats,
                training_path="no_pretrain_ccsa",
            )
            followup_results[fold] = followup_metrics

        result_key = f"no_pretrain_ccsa_then_{followup_mode}"
        return {result_key: followup_results}

    def _run_no_pretrain_encoder_task(self) -> Dict[str, Any]:
        """
        Run direct fine-tuning on target domain only without pretraining.

        This method trains a model from scratch (random initialization) directly on
        the target domain data using the encoder_task fine-tuning mode.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, Any]
            Dictionary with key 'no_pretrain_encoder_task' containing fold-wise
            fine-tuning results on target domain only.
        """
        encoder_task_results = {}
        global_stats = self._global_scaling_stats

        for fold in range(1, self.params["n_folds"] + 1):
            logger.info("=" * 80)
            logger.info(
                f"      NO PRETRAIN ENCODER TASK FOLD {fold}/{self.params['n_folds']} (TARGET DOMAIN ONLY)"
            )
            logger.info("=" * 80)

            # Run encoder_task fine-tuning on target domain only
            metrics = self.finetune_phase.run_target_only(
                fold=fold,
                finetune_mode="encoder_task",
                global_scaling_stats=global_stats,
            )
            encoder_task_results[fold] = metrics

        return {"no_pretrain_encoder_task": encoder_task_results}

    def _run_standard_finetuning(
        self, source_model: torch.nn.Module
    ) -> Dict[str, Dict[int, Any]]:
        """
        Run standard supervised fine-tuning methods across all cross-validation folds.

        This method applies standard fine-tuning approaches (encoder_task and task)
        to the pre-trained source model across all folds without CCSA.

        Parameters
        ----------
        source_model : torch.nn.Module
            Pre-trained model from source domain pretraining phase.

        Returns
        -------
        Dict[str, Dict[int, Any]]
            Dictionary with keys 'pretrain_{method}' (method = 'encoder_task' or 'task'),
            each containing fold-wise fine-tuning results.
        """
        standard_methods = ["encoder_task", "task"]
        results = {f"pretrain_{method}": {} for method in standard_methods}

        # Use already computed global scaling stats (no need to recompute)
        global_stats = self._global_scaling_stats

        for method in standard_methods:
            for fold in range(1, self.params["n_folds"] + 1):
                logger.info("=" * 80)
                logger.info(
                    f"      PRETRAIN {method.upper()} FOLD {fold}/{self.params['n_folds']}"
                )
                logger.info("=" * 80)

                # Create clean model copy
                fold_model = MultiTaskPocket(
                    domain="src", self_ssl=True, temperature=0.07, **self.params
                )
                fold_model.load_state_dict(source_model.state_dict(), strict=False)

                # Run fine-tuning
                metrics = self._run_finetune(
                    fold=fold,
                    pretrained_model=fold_model,
                    finetune_mode=method,
                    global_scaling_stats=global_stats,
                    training_path="pretrain",
                )
                results[f"pretrain_{method}"][fold] = metrics

        return results

    def _run_finetune(
        self,
        fold: int,
        pretrained_model: torch.nn.Module,
        finetune_mode: str,
        global_scaling_stats: Dict[str, Any],
        domain: str = "tgt",
        phase_prefix: str = "",
        training_path: str = "",
    ) -> Dict[str, Any]:
        """
        Execute supervised fine-tuning on target domain with consistent scaling.

        This method performs supervised training on the target domain using a pre-trained
        model, with options for different freezing strategies and scaling configurations.

        Parameters
        ----------
        fold : int
            Current cross-validation fold number.
        pretrained_model : torch.nn.Module
            Model with pre-trained weights (from pretraining or CCSA).
        finetune_mode : str
            Training mode: 'encoder_task' (train full model) or 'task' (freeze encoders).
        global_scaling_stats : Dict[str, Any]
            Pre-computed scaling statistics for consistent feature normalization.
        domain : str, optional
            Domain to train on ('tgt' for target domain), by default "tgt".
        phase_prefix : str, optional
            Prefix for run naming in logging, by default "".
        training_path : str, optional
            Path identifier for distinguishing different training approaches, by default "".

        Returns
        -------
        Dict[str, Any]
            Dictionary of validation metrics including loss, accuracy, F1, AUROC, and MCC.

        Raises
        ------
        ValueError
            If finetune_mode is not 'encoder_task' or 'task'.
        """
        if finetune_mode not in {"encoder_task", "task"}:
            raise ValueError(f"Invalid finetune_mode: {finetune_mode}")

        logger.info(f"Fine-tuning {finetune_mode} for fold {fold}")

        # Update parameters
        temp_params = self.params.copy()
        temp_params.update(
            {
                "train_on": domain,
                "learning_rate": self.config.learning_rate_finetune,
                "dropout": self.config.dropout_finetune,
                "fold": fold,
            }
        )

        # Create model
        model = MultiTaskPocket(domain=domain, **temp_params)
        model.load_state_dict(pretrained_model.state_dict(), strict=False)

        # Apply freezing strategy
        if finetune_mode == "encoder_task":
            self.finetune_phase._freeze_nothing(model)
        elif finetune_mode == "task":
            self.finetune_phase._freeze_encoders_and_feature_extractor(model)

        # Get data with consistent scaling
        datasets, scaling_stats_dict = get_datasets(
            temp_params,
            fold=fold,
            global_scaling_stats=global_scaling_stats,
        )
        src_train_dataset, src_val_dataset, tgt_train_dataset, tgt_val_dataset = (
            datasets
        )

        # Setup output directory and save stats
        output_dir = self.finetune_phase.create_output_dir(
            fold, finetune_mode, training_path
        )

        # Save the appropriate scaling stats based on the mode
        if temp_params.get("cross_domain_scaling", False):
            stats_to_save = scaling_stats_dict.get("scaling_stats")
            logger.info(f"Saving cross-domain scaling stats for fold {fold}")
        else:
            stats_to_save = scaling_stats_dict.get("target_stats")
            logger.info(f"Saving target-specific scaling stats for fold {fold}")

        if stats_to_save:
            save_phase_stats(stats_to_save, output_dir)
        else:
            logger.warning(f"No scaling stats found to save for {output_dir}")

        dataloaders = create_dataloaders(
            src_train_dataset,
            src_val_dataset,
            tgt_train_dataset,
            tgt_val_dataset,
            temp_params,
        )

        train_loader = WrappedDataLoader(dataloaders[f"{domain}_train_loader"], domain)
        val_loader = WrappedDataLoader(dataloaders[f"{domain}_val_loader"], domain)

        # Create callbacks and trainer
        callbacks = self.finetune_phase.create_callbacks(
            output_dir,
            self.config.monitor_metric_finetune,
            self.config.monitor_mode_finetune,
        )
        run_name = f"{phase_prefix}{finetune_mode}_{model.combination_name}_{temp_params['inhibitor_descriptor']}_{temp_params['biological_descriptor']}_fold_{fold}_consistent_scaling"

        with self.finetune_phase.training_context():
            trainer = self.finetune_phase.create_trainer(
                callbacks, run_name, finetune_mode
            )
            trainer.fit(
                model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )
            metrics = self.finetune_phase._extract_metrics(trainer, val_loader)

        # Save model components
        self.finetune_phase.component_manager.save_components(
            model, output_dir, finetune_mode
        )

        return metrics
