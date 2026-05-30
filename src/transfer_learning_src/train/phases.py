import gc
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from prettytable import PrettyTable
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.utilities.combined_loader import CombinedLoader

from data.dataset import WrappedDataLoader
from data.utils_data import (
    get_datasets,
    create_dataloaders,
)
from .config import TrainingConfig
from .model_utils import ModelComponentManager
from model.multitask import MultiTaskPocket, CCSAFinetune
from utils.checkpoint_utils import PrintOnBestModelCheckpoint, save_phase_stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrainingPhase:
    """
    Base class for training phases with common functionality.

    This abstract base class provides shared methods for creating directory structures,
    managing wandb logging, and handling model checkpointing across different training
    phases (pretraining, CCSA, fine-tuning).
    """

    def __init__(self, config: TrainingConfig, params: Dict[str, Any]) -> None:
        """
        Initialize the training phase with configuration and parameters.

        Parameters
        ----------
        config : TrainingConfig
            Configuration object containing hyperparameters for all training phases.
        params : Dict[str, Any]
            Dictionary of runtime parameters including modality settings, dataset info,
            and training options.

        Returns
        -------
        None
        """
        self.config = config
        self.params = params
        self.component_manager = ModelComponentManager()

    def create_results_base_dir(self) -> Path:
        """
        Create hierarchical results directory based on experiment configuration.

        The directory structure is: inhibitor_descriptor/modality/scaler/affinity_type/scaling_config

        Parameters
        ----------
        None

        Returns
        -------
        Path
            Path object pointing to the created results directory.
        """
        # Get modality combination from params
        modality_parts = []
        if self.params.get("use_mol", False):
            modality_parts.append("molecule")
        if self.params.get("use_protein", False):
            modality_parts.append("protein")
        if self.params.get("use_pocket", False):
            modality_parts.append("pocket")

        modality_combination = "_".join(modality_parts) if modality_parts else "empty"

        # Get scaling method (default to minmax_scaler if not specified)
        scaling_method = self.params.get("data_scaler", "minmax")
        scaler_name = f"{scaling_method}_scaler"

        # Add cross-domain scaling and source SMOTE to directory structure
        scaling_suffix = ""
        if self.params.get("cross_domain_scaling", False):
            scaling_suffix += "_cross_domain"
        else:
            scaling_suffix += "_separate_domain"

        # Explicitly add SMOTE status to path for clear differentiation
        source_smote_status = (
            "source_smote"
            if self.params.get("source_smote", False)
            else "no_source_smote"
        )
        target_smote_status = (
            "target_smote"
            if self.params.get("target_smote", False)
            else "no_target_smote"
        )
        scaling_suffix += f"_{source_smote_status}_{target_smote_status}"

        # Create directory structure: inhibitor_descriptor/modality_combination/scaler/affinity_type/scaling_config/
        base_results_dir = (
            Path("results")
            / self.params["inhibitor_descriptor"]
            / modality_combination
            / scaler_name
            / self.params["affinity_type"]
            / scaling_suffix.lstrip("_")
        )
        base_results_dir.mkdir(parents=True, exist_ok=True)

        return base_results_dir

    def create_output_dir(
        self, fold: int, phase_name: str, training_path: str = ""
    ) -> Path:
        """
        Create standardized output directory structure with training path distinction.

        This method creates a hierarchical directory structure for saving training results,
        with optional training path prefixes to distinguish different training approaches.

        Parameters
        ----------
        fold : int
            Cross-validation fold number (0 for pretraining, 1+ for fold-specific training).
        phase_name : str
            Name of the training phase (e.g., 'pretrain_src', 'ccsa', 'encoder_task').
        training_path : str, optional
            Training path prefix to distinguish different approaches (e.g., 'pretrain', 'no_pretrain_ccsa'),
            by default "".

        Returns
        -------
        Path
            Path object pointing to the created output directory.
        """
        base_results_dir = self.create_results_base_dir()

        if fold == 0:
            # For pretraining, save at the same level as fold_x directories
            output_dir = base_results_dir / phase_name
        else:
            # Add training path prefix to distinguish different training approaches
            if training_path:
                phase_dir = f"{training_path}_{phase_name}"
            else:
                phase_dir = phase_name
            output_dir = base_results_dir / f"fold_{fold}" / phase_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def create_trainer(
        self, callbacks: List, run_name: str, finetune_mode: Optional[str] = None
    ) -> pl.Trainer:
        """
        Create a PyTorch Lightning trainer with standard configuration.

        This method instantiates a trainer with wandb logging, gradient clipping,
        and specified callbacks for model checkpointing and early stopping.

        Parameters
        ----------
        callbacks : List
            List of PyTorch Lightning callbacks (e.g., checkpointing, early stopping).
        run_name : str
            Name for the wandb run for experiment tracking.
        finetune_mode : Optional[str], optional
            Fine-tuning mode to log as hyperparameter, by default None.

        Returns
        -------
        pl.Trainer
            Configured PyTorch Lightning Trainer instance.
        """
        wandb_logger = WandbLogger(
            project="tl-final-runs",
            name=run_name,
            log_model=False,
            id=None,
            resume=False,
            reinit=True,
        )

        log_params = self.params.copy()
        if finetune_mode:
            log_params["finetune_mode"] = finetune_mode
        wandb_logger.log_hyperparams(log_params)

        return pl.Trainer(
            gradient_clip_val=1.0,
            max_epochs=self.config.max_epochs,
            callbacks=callbacks,
            logger=wandb_logger,
            enable_progress_bar=self.config.enable_progress_bar,
            log_every_n_steps=1,
            accelerator="gpu",
        )

    def create_callbacks(
        self, output_dir: Path, monitor_metric: str, monitor_mode: str
    ) -> List:
        """
        Create standard callbacks for model training.

        This method creates callbacks for model checkpointing, early stopping, and
        learning rate monitoring.

        Parameters
        ----------
        output_dir : Path
            Directory path for saving checkpoints.
        monitor_metric : str
            Metric name to monitor for checkpointing and early stopping (e.g., 'val/loss').
        monitor_mode : str
            Mode for monitoring ('min' or 'max').

        Returns
        -------
        List
            List of configured PyTorch Lightning callbacks.
        """
        return [
            PrintOnBestModelCheckpoint(
                print_on_train_epoch=(monitor_metric.startswith("train")),
                print_on_val_epoch=(monitor_metric.startswith("val")),
                print_at_end=True,
                monitor=monitor_metric,
                save_top_k=1,
                mode=monitor_mode,
                dirpath=output_dir / "checkpoints",
            ),
            EarlyStopping(
                monitor=monitor_metric, patience=self.config.patience, mode=monitor_mode
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

    @contextmanager
    def training_context(self):
        """
        Context manager for training with automatic cleanup.

        This context manager ensures proper cleanup of wandb runs and GPU cache
        after training, even if errors occur.

        Parameters
        ----------
        None

        Yields
        ------
        None

        Notes
        -----
        Automatically calls wandb.finish() and torch.cuda.empty_cache() on exit.
        """
        try:
            yield
        finally:
            # Cleanup
            if wandb.run is not None:
                wandb.finish()
            torch.cuda.empty_cache()


class PretrainingPhase(TrainingPhase):
    """
    Manages self-supervised pretraining phase on source domain data.

    This phase trains a model using contrastive self-supervised learning (SimCLR-style)
    on the source domain before transfer learning to the target domain. Uses k-NN
    probing to evaluate embedding quality during training.
    """

    def run(
        self, source_loaders: Dict, pretrained_model: Optional[torch.nn.Module] = None
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """
        Execute the pretraining phase on source domain data.

        Parameters
        ----------
        source_loaders : Dict
            Dictionary containing 'src_train' and 'src_val' dataloaders.
        pretrained_model : Optional[torch.nn.Module], optional
            Pre-existing model to use (skips training if provided), by default None.

        Returns
        -------
        Tuple[torch.nn.Module, Dict[str, Any]]
            A tuple containing:
            - Trained model with learned representations
            - Dictionary of training metrics and statistics
        """
        logger.info(f"Starting pre-training phase on all source dataset")

        # Update parameters for pretraining
        temp_params = self.params.copy()
        temp_params.update(
            {
                "learning_rate": self.config.learning_rate_pretrain,
                "dropout": self.config.dropout_pretrain,
                "ssl_temperature": self.config.ssl_temperature,
                "train_on": "src",
            }
        )

        # Create model
        model = MultiTaskPocket(
            domain="src",
            self_ssl=True,
            temperature=self.config.ssl_temperature,
            **temp_params,
        )

        if pretrained_model:
            model.load_state_dict(pretrained_model.state_dict(), strict=True)
            logger.info("Loaded pre-trained weights from previous phase")

        # Setup output directory - use the new structure
        output_dir = self.create_output_dir("0", f"pretrain_src")
        train_loader = WrappedDataLoader(source_loaders[f"src_train_loader"], "src")
        val_loader = WrappedDataLoader(source_loaders[f"src_val_loader"], "src")

        # Create callbacks and trainer
        callbacks = self.create_callbacks(
            output_dir,
            self.config.monitor_metric_pretrain,
            self.config.monitor_mode_pretrain,
        )
        run_name = f"pretrain_src_{model.combination_name}_{temp_params['inhibitor_descriptor']}_{temp_params['biological_descriptor']}"

        with self.training_context():
            trainer = self.create_trainer(callbacks, run_name)
            trainer.fit(
                model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )
            metrics = self._extract_metrics(trainer, val_loader)

        # Save model components
        self.component_manager.save_components(model, output_dir, f"pretrain_src")

        logger.info(f"Pre-training on source completed. Saved to {output_dir}")
        return model, metrics

    def _extract_metrics(self, trainer: pl.Trainer, dataloader: Any) -> Dict[str, Any]:
        """
        Extract validation metrics from trainer after pretraining.

        This method retrieves k-NN probe metrics from the best checkpoint or logged metrics.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer instance.
        dataloader : Any
            Validation dataloader (not used in current implementation).

        Returns
        -------
        Dict[str, Any]
            Dictionary of validation metrics including k-NN AUROC, AUPRC, and SSL loss.
        """
        # Get the checkpoint callback to access the best model score
        checkpoint_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, PrintOnBestModelCheckpoint):
                checkpoint_callback = callback
                break

        # The checkpoint callback tracks the best value of the monitored metric (now val/knn_auroc)
        if checkpoint_callback and checkpoint_callback.best_model_score is not None:
            best_score = float(checkpoint_callback.best_model_score)
            # Get the actual metrics from trainer
            logged_metrics = trainer.logged_metrics

            return {
                "val/knn_auroc": best_score,  # Best k-NN probe AUROC score
                "val/knn_auprc": float(logged_metrics.get("val/knn_auprc", 0.0)),
                "val/ssl_loss_debug": float(
                    logged_metrics.get("val/ssl_loss_debug", float("inf"))
                ),
                "val/loss": best_score,  # For compatibility with downstream code (use AUROC as proxy)
                "val/accuracy": best_score,  # Use AUROC as proxy for accuracy
                "val/f1": float(
                    logged_metrics.get("val/knn_auprc", 0.0)
                ),  # Use AUPRC as proxy for F1
                "val/auroc": best_score,
                "val/mcc": 0.0,
            }

        # Fallback to logged metrics if checkpoint callback not available
        if trainer.logged_metrics:
            logged_metrics = trainer.logged_metrics
            knn_auroc = float(logged_metrics.get("val/knn_auroc", 0.0))
            knn_auprc = float(logged_metrics.get("val/knn_auprc", 0.0))
            ssl_loss = float(logged_metrics.get("val/ssl_loss_debug", float("inf")))

            return {
                "val/knn_auroc": knn_auroc,
                "val/knn_auprc": knn_auprc,
                "val/ssl_loss_debug": ssl_loss,
                "val/loss": knn_auroc,  # For compatibility
                "val/accuracy": knn_auroc,  # Use AUROC as proxy
                "val/f1": knn_auprc,  # Use AUPRC as proxy
                "val/auroc": knn_auroc,
                "val/mcc": 0.0,
            }

        return {
            "val/knn_auroc": 0.0,
            "val/knn_auprc": 0.0,
            "val/ssl_loss_debug": float("inf"),
            "val/loss": 0.0,
            "val/accuracy": 0.0,
            "val/f1": 0.0,
            "val/auroc": 0.0,
            "val/mcc": 0.0,
        }


class CCSAPhase(TrainingPhase):
    """
    Manages CCSA (Classification and Contrastive Semantic Alignment) domain adaptation phase.

    This phase performs domain adaptation using contrastive learning to align
    source and target domain representations while maintaining semantic structure.
    Combines source classification loss with cross-domain contrastive alignment loss.
    """

    def run_with_cached_source(
        self,
        src_train_dataset: Dict[str, Any],
        src_val_dataset: Dict[str, Any],
        fold: int,
        pretrained_model: torch.nn.Module,
        global_scaling_stats: Dict[str, Any],
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """
        Execute CCSA training using pre-loaded source domain data.

        Parameters
        ----------
        src_train_dataset : Dict[str, Any]
            Pre-loaded source training dataset.
        src_val_dataset : Dict[str, Any]
            Pre-loaded source validation dataset.
        fold : int
            Current cross-validation fold number.
        pretrained_model : torch.nn.Module
            Pre-trained model from pretraining phase (weights are loaded into CCSA model).
        global_scaling_stats : Dict[str, Any]
            Scaling statistics computed from source domain.

        Returns
        -------
        Tuple[torch.nn.Module, Dict[str, Any]]
            A tuple containing:
            - CCSA-trained model
            - Dictionary of training metrics and statistics for this fold
        """
        logger.info(
            f"Starting CCSA fine-tuning for fold {fold} (using cached source data)"
        )

        # Update parameters for CCSA
        temp_params = self.params.copy()
        temp_params.update(
            {
                "train_on": "tgt",  # Only load target data
                "learning_rate": self.config.learning_rate_ccsa,
                "dropout": self.config.dropout_ccsa,
                "fold": fold,
            }
        )

        # Create base model and CCSA wrapper
        base_model = MultiTaskPocket(domain="tgt", **temp_params)
        base_model.load_state_dict(pretrained_model.state_dict(), strict=False)

        ccsa_model = CCSAFinetune(
            base_model=base_model,
            gamma=self.config.gamma,
            margin=self.config.margin,
        )

        # Setup output directory with training path
        output_dir = self.create_output_dir(
            fold, "ccsa", "pretrain"
        )  # Add "pretrain" prefix

        # Load target data - always use global scaling for consistency
        datasets, scaling_stats_dict = get_datasets(
            temp_params,
            fold=fold,
            global_scaling_stats=global_scaling_stats,
        )
        _, _, tgt_train_dataset, tgt_val_dataset = datasets

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

        # Create target dataloaders
        dataloaders = create_dataloaders(
            src_train_dataset,
            src_val_dataset,
            tgt_train_dataset,
            tgt_val_dataset,
            temp_params,
        )

        # Combine cached source loaders with new target loaders
        train_loader = CombinedLoader(
            {
                "src": dataloaders["src_train_loader"],
                "tgt": dataloaders["tgt_train_loader"],
            }
        )
        val_loader = dataloaders["tgt_val_loader"]

        # Create callbacks and trainer
        callbacks = self.create_callbacks(
            output_dir, self.config.monitor_metric_ccsa, self.config.monitor_mode_ccsa
        )
        run_name = f"ccsa_{ccsa_model.base_model.combination_name}_{temp_params['inhibitor_descriptor']}_{temp_params['biological_descriptor']}_fold_{fold}"

        with self.training_context():
            trainer = self.create_trainer(callbacks, run_name, "ccsa")
            trainer.fit(
                ccsa_model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )
            metrics = self._extract_metrics(trainer, val_loader)

        # Save model components
        self.component_manager.save_components(ccsa_model, output_dir, "ccsa")

        # Return the best base model for chaining
        best_base_model = MultiTaskPocket(domain="tgt", **temp_params)
        best_base_model.load_state_dict(
            ccsa_model.base_model.state_dict(), strict=False
        )

        logger.info(
            f"CCSA fine-tuning completed for fold {fold}. Saved to {output_dir}"
        )
        return best_base_model, metrics

    def run_without_pretraining(
        self,
        src_train_dataset: Dict[str, Any],
        src_val_dataset: Dict[str, Any],
        fold: int,
        global_scaling_stats: Dict[str, Any],
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """
        Execute CCSA training without pretraining (random initialization).

        This method performs CCSA domain adaptation starting from randomly initialized
        weights rather than pre-trained weights.

        Parameters
        ----------
        src_train_dataset : Dict[str, Any]
            Pre-loaded source training dataset.
        src_val_dataset : Dict[str, Any]
            Pre-loaded source validation dataset.
        fold : int
            Current cross-validation fold number.
        global_scaling_stats : Dict[str, Any]
            Scaling statistics computed from source domain.

        Returns
        -------
        Tuple[torch.nn.Module, Dict[str, Any]]
            A tuple containing:
            - CCSA-trained model (base model extracted from CCSA wrapper)
            - Dictionary of training metrics and statistics for this fold
        """
        logger.info(
            f"Starting CCSA training for fold {fold} (no pretraining, random initialization)"
        )

        # Update parameters for CCSA
        temp_params = self.params.copy()
        temp_params.update(
            {
                "learning_rate": self.config.learning_rate_ccsa,
                "dropout": self.config.dropout_ccsa,
                "fold": fold,
            }
        )

        # Create fresh base model with random initialization
        base_model = MultiTaskPocket(domain="tgt", **temp_params)

        ccsa_model = CCSAFinetune(
            base_model=base_model,
            gamma=self.config.gamma,
            margin=self.config.margin,
        )

        # Setup output directory
        output_dir = self.create_output_dir(fold, "no_pretrain_ccsa")

        # Use cached source data if available, otherwise load fresh
        temp_params_tgt = temp_params.copy()
        temp_params_tgt.update({"train_on": "tgt"})

        datasets, scaling_stats_dict = get_datasets(
            temp_params_tgt, fold=fold, global_scaling_stats=global_scaling_stats
        )
        _, _, tgt_train_dataset, tgt_val_dataset = datasets

        # Save the appropriate scaling stats based on the mode
        if scaling_stats_dict.get("cross_domain"):
            stats_to_save = scaling_stats_dict.get("scaling_stats")
            logger.info(f"Saving cross-domain scaling stats for fold {fold}")
        else:
            stats_to_save = scaling_stats_dict.get("target_stats")
            logger.info(f"Saving target-specific scaling stats for fold {fold}")

        if stats_to_save:
            save_phase_stats(stats_to_save, output_dir)
        else:
            logger.warning(f"No scaling stats found to save for {output_dir}")

        # Create target dataloaders
        dataloaders = create_dataloaders(
            src_train_dataset,
            src_val_dataset,
            tgt_train_dataset,
            tgt_val_dataset,
            temp_params,
        )

        # Combine cached source loaders with new target loaders
        train_loader = CombinedLoader(
            {
                "src": dataloaders["src_train_loader"],  # Reuse cached source
                "tgt": dataloaders["tgt_train_loader"],  # New target for this fold
            }
        )
        val_loader = dataloaders["tgt_val_loader"]

        # Create callbacks and trainer
        callbacks = self.create_callbacks(
            output_dir, self.config.monitor_metric_ccsa, self.config.monitor_mode_ccsa
        )
        run_name = f"no_pretrain_ccsa_{ccsa_model.base_model.combination_name}_{temp_params['inhibitor_descriptor']}_{temp_params['biological_descriptor']}_fold_{fold}_target_scaling"

        with self.training_context():
            trainer = self.create_trainer(callbacks, run_name, "no_pretrain_ccsa")
            trainer.fit(
                ccsa_model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )
            metrics = self._extract_metrics(trainer, val_loader)

        # Save model components
        self.component_manager.save_components(
            ccsa_model, output_dir, "no_pretrain_ccsa"
        )

        # Return the best base model for chaining
        best_base_model = MultiTaskPocket(domain="tgt", **temp_params)
        best_base_model.load_state_dict(
            ccsa_model.base_model.state_dict(), strict=False
        )

        logger.info(
            f"CCSA training (no pretraining) completed for fold {fold}. Saved to {output_dir}"
        )
        return best_base_model, metrics

    def run(
        self,
        fold: int,
        pretrained_model: torch.nn.Module,
        global_scaling_stats: Dict[str, Any],
    ) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """
        Execute CCSA training using pre-trained model (original method).

        This is the original CCSA method provided for backward compatibility.
        It loads both source and target data and performs domain adaptation.

        Parameters
        ----------
        fold : int
            Current cross-validation fold number.
        pretrained_model : torch.nn.Module
            Pre-trained model from pretraining phase (weights are loaded into CCSA model).
        global_scaling_stats : Dict[str, Any]
            Scaling statistics computed from source domain.

        Returns
        -------
        Tuple[torch.nn.Module, Dict[str, Any]]
            A tuple containing:
            - CCSA-trained model (base model extracted from CCSA wrapper)
            - Dictionary of training metrics and statistics for this fold
        """
        logger.info(f"Starting CCSA fine-tuning for fold {fold}")

        # Update parameters for CCSA
        temp_params = self.params.copy()
        temp_params.update(
            {
                "train_on": "src+tgt",
                "learning_rate": self.config.learning_rate_ccsa,
                "dropout": self.config.dropout_ccsa,
                "fold": fold,
            }
        )

        # Create base model and CCSA wrapper
        base_model = MultiTaskPocket(domain="tgt", **temp_params)
        base_model.load_state_dict(pretrained_model.state_dict(), strict=False)

        ccsa_model = CCSAFinetune(
            base_model=base_model,
            gamma=self.config.gamma,
            margin=self.config.margin,
        )

        # Setup output directory
        output_dir = self.create_output_dir(fold, "ccsa")

        # Get data for joint training - always use global scaling for consistency
        datasets, scaling_stats_dict = get_datasets(
            temp_params, fold=fold, global_scaling_stats=global_scaling_stats
        )
        src_train_dataset, src_val_dataset, tgt_train_dataset, tgt_val_dataset = (
            datasets
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

        train_loader = CombinedLoader(
            {
                "src": dataloaders["src_train_loader"],
                "tgt": dataloaders["tgt_train_loader"],
            }
        )
        val_loader = dataloaders["tgt_val_loader"]

        # Create callbacks and trainer
        callbacks = self.create_callbacks(
            output_dir, self.config.monitor_metric_ccsa, self.config.monitor_mode_ccsa
        )
        run_name = f"ccsa_{ccsa_model.base_model.combination_name}_{temp_params['inhibitor_descriptor']}_{temp_params['biological_descriptor']}_fold_{fold}"

        with self.training_context():
            trainer = self.create_trainer(callbacks, run_name, "ccsa")
            trainer.fit(
                ccsa_model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )
            metrics = self._extract_metrics(trainer, val_loader)

        # Save model components
        self.component_manager.save_components(ccsa_model, output_dir, "ccsa")

        # Return the best base model for chaining
        best_base_model = MultiTaskPocket(domain="tgt", **temp_params)
        best_base_model.load_state_dict(
            ccsa_model.base_model.state_dict(), strict=False
        )

        logger.info(
            f"CCSA fine-tuning completed for fold {fold}. Saved to {output_dir}"
        )
        return best_base_model, metrics

    def _extract_metrics(self, trainer: pl.Trainer, val_loader: Any) -> Dict[str, Any]:
        """
        Extract validation metrics from best model checkpoint.

        This method loads the best checkpoint and evaluates it on the validation set
        to retrieve final metrics.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer instance.
        val_loader : Any
            Validation dataloader for evaluation.

        Returns
        -------
        Dict[str, Any]
            Dictionary of validation metrics including loss, accuracy, F1, AUROC, and MCC.
        """
        checkpoint_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, PrintOnBestModelCheckpoint):
                checkpoint_callback = callback
                break

        if checkpoint_callback and checkpoint_callback.best_model_path:
            try:
                model = trainer.lightning_module
                checkpoint = torch.load(
                    checkpoint_callback.best_model_path,
                    map_location="cpu",
                    weights_only=True,
                )
                model.load_state_dict(checkpoint["state_dict"])

                val_results = trainer.validate(
                    model, dataloaders=val_loader, verbose=False
                )
                result = (
                    val_results[-1]
                    if isinstance(val_results, list) and val_results
                    else {}
                )

                return {
                    "val/loss": float(result.get("val/loss", float("inf"))),
                    "val/accuracy": float(result.get("val/accuracy", 0.0)),
                    "val/f1": float(result.get("val/f1", 0.0)),
                    "val/auroc": float(result.get("val/auroc", 0.0)),
                    "val/mcc": float(result.get("val/mcc", 0.0)),
                }
            except Exception as e:
                logger.warning(f"Could not load best checkpoint: {e}")

        # Fallback
        if trainer.logged_metrics:
            return {
                "val/loss": float(trainer.logged_metrics.get("val/loss", float("inf"))),
                "val/accuracy": float(trainer.logged_metrics.get("val/accuracy", 0.0)),
                "val/f1": float(trainer.logged_metrics.get("val/f1", 0.0)),
                "val/auroc": float(trainer.logged_metrics.get("val/auroc", 0.0)),
                "val/mcc": float(trainer.logged_metrics.get("val/mcc", 0.0)),
            }

        return {
            "val/loss": float("inf"),
            "val/accuracy": 0.0,
            "val/f1": 0.0,
            "val/auroc": 0.0,
            "val/mcc": 0.0,
        }


class FinetunePhase(TrainingPhase):
    """
    Manages standard supervised fine-tuning phase on target domain.

    This phase performs supervised training on the target domain, either from scratch
    or using weights from previous phases (pretraining/CCSA). Supports two modes:
    - 'task': Train only classification head (freeze encoders)
    - 'encoder_task': Train full model including encoders
    """

    def run(
        self,
        fold: int,
        pretrained_model: torch.nn.Module,
        global_scaling_stats: Dict[str, Any],
        finetune_mode: str,
        domain: str = "tgt",
        training_path: str = "",
    ) -> Dict[str, Any]:
        """
        Execute fine-tuning on target domain data.

        Parameters
        ----------
        fold : int
            Current cross-validation fold number.
        pretrained_model : torch.nn.Module
            Model with pre-trained weights (from pretraining or CCSA).
        global_scaling_stats : Dict[str, Any]
            Scaling statistics computed from source domain.
        finetune_mode : str
            Training mode: 'task' (freeze encoders) or 'encoder_task' (train all).
        domain : str, optional
            Domain to train on ('tgt' for target), by default "tgt".
        training_path : str, optional
            Description of the training path (for logging/results), by default "".

        Returns
        -------
        Dict[str, Any]
            Dictionary of fine-tuning metrics and statistics for this fold.

        Raises
        ------
        ValueError
            If finetune_mode is not 'encoder_task' or 'task'.
        """
        if finetune_mode not in {"encoder_task", "task"}:
            raise ValueError(f"Invalid finetune_mode: {finetune_mode}")

        logger.info(f"Starting fine-tuning phase: {finetune_mode} for fold {fold}")

        # Update parameters for fine-tuning
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
            self._freeze_nothing(model)
        elif finetune_mode == "task":
            self._freeze_encoders_and_feature_extractor(model)

        # Setup output directory with training path
        output_dir = self.create_output_dir(fold, finetune_mode, training_path)

        # Get data - always pass global_scaling_stats for consistency
        datasets, scaling_stats_dict = get_datasets(
            temp_params, fold=fold, global_scaling_stats=global_scaling_stats
        )
        src_train_dataset, src_val_dataset, tgt_train_dataset, tgt_val_dataset = (
            datasets
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
        loaders = create_dataloaders(
            src_train_dataset,
            src_val_dataset,
            tgt_train_dataset,
            tgt_val_dataset,
            temp_params,
        )

        train_loader = WrappedDataLoader(loaders[f"{domain}_train_loader"], domain)
        val_loader = WrappedDataLoader(loaders[f"{domain}_val_loader"], domain)

        # Create callbacks and trainer
        callbacks = self.create_callbacks(
            output_dir,
            self.config.monitor_metric_finetune,
            self.config.monitor_mode_finetune,
        )
        run_name = f"{finetune_mode}_{model.combination_name}_{temp_params['inhibitor_descriptor']}_{temp_params['biological_descriptor']}_fold_{fold}"

        with self.training_context():
            trainer = self.create_trainer(callbacks, run_name, finetune_mode)
            trainer.fit(
                model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )
            metrics = self._extract_metrics(trainer, val_loader)

        # Save model components
        self.component_manager.save_components(model, output_dir, finetune_mode)

        logger.info(
            f"Fine-tuning phase '{finetune_mode}' completed for fold {fold}. Saved to {output_dir}"
        )
        return metrics

    def run_target_only(
        self,
        fold: int,
        finetune_mode: str,
        global_scaling_stats: Dict[str, Any] = None,
        target_smote: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute fine-tuning on target domain only with target-specific scaling.

        This method trains a model from scratch on the target domain without using
        source domain data or pre-trained weights. Scaling statistics are computed
        from the target training set only.

        Parameters
        ----------
        fold : int
            Current cross-validation fold number.
        finetune_mode : str
            Training mode: 'task' (freeze encoders) or 'encoder_task' (train all).
        global_scaling_stats : Dict[str, Any], optional
            Global scaling statistics (not used in this method), by default None.
        target_smote : bool, optional
            Whether to apply SMOTE to target domain (deprecated parameter), by default False.

        Returns
        -------
        Dict[str, Any]
            Dictionary of fine-tuning metrics and statistics for this fold.

        Raises
        ------
        ValueError
            If finetune_mode is not 'encoder_task' or 'task'.
        """
        if finetune_mode not in {"encoder_task", "task"}:
            raise ValueError(f"Invalid finetune_mode: {finetune_mode}")

        logger.info(
            f"Starting target-only fine-tuning phase: {finetune_mode} for fold {fold}"
        )

        # Update parameters for fine-tuning
        temp_params = self.params.copy()
        temp_params.update(
            {
                "train_on": "tgt",
                "learning_rate": self.config.learning_rate_finetune,
                "dropout": self.config.dropout_finetune,
                "fold": fold,
            }
        )

        # For target-only training, we explicitly want stats computed from the target training set.
        # We achieve this by setting cross_domain_scaling to False and passing no global stats.
        temp_params["cross_domain_scaling"] = False

        # Create fresh model with random initialization
        model = MultiTaskPocket(domain="tgt", **temp_params)

        # Apply freezing strategy
        if finetune_mode == "encoder_task":
            self._freeze_nothing(model)
        elif finetune_mode == "task":
            self._freeze_encoders_and_feature_extractor(model)

        # Setup output directory
        output_dir = self.create_output_dir(fold, f"no_pretrain_{finetune_mode}")

        # IMPORTANT: Pass global_scaling_stats=None to force computation from target data for this fold.
        datasets, scaling_stats_dict = get_datasets(
            temp_params, fold=fold, global_scaling_stats=None
        )
        _, _, tgt_train_dataset, tgt_val_dataset = datasets

        # Save the correctly computed target-specific scaling stats.
        stats_to_save = scaling_stats_dict.get("target_stats")
        if stats_to_save:
            save_phase_stats(stats_to_save, output_dir)
        else:
            logger.warning(f"No target scaling stats found to save for {output_dir}")

        loaders = create_dataloaders(
            None, None, tgt_train_dataset, tgt_val_dataset, temp_params
        )

        train_loader = WrappedDataLoader(loaders["tgt_train_loader"], "tgt")
        val_loader = WrappedDataLoader(loaders["tgt_val_loader"], "tgt")

        # Create callbacks and trainer
        callbacks = self.create_callbacks(
            output_dir,
            self.config.monitor_metric_finetune,
            self.config.monitor_mode_finetune,
        )
        run_name = f"no_pretrain_{finetune_mode}_{model.combination_name}_{temp_params['inhibitor_descriptor']}_{temp_params['biological_descriptor']}_fold_{fold}_target_scaling"

        with self.training_context():
            trainer = self.create_trainer(
                callbacks, run_name, f"no_pretrain_{finetune_mode}"
            )
            trainer.fit(
                model, train_dataloaders=train_loader, val_dataloaders=val_loader
            )
            metrics = self._extract_metrics(trainer, val_loader)

        # Save model components
        self.component_manager.save_components(
            model, output_dir, f"no_pretrain_{finetune_mode}"
        )

        logger.info(
            f"Target-only fine-tuning phase '{finetune_mode}' completed for fold {fold}. Saved to {output_dir}"
        )
        return metrics

    def _freeze_nothing(self, model: torch.nn.Module) -> None:
        """
        Unfreeze all model parameters for full fine-tuning.

        This method sets all model parameters to trainable, allowing the entire
        model to be updated during training.

        Parameters
        ----------
        model : torch.nn.Module
            Model to unfreeze.

        Returns
        -------
        None
        """
        for param in model.parameters():
            param.requires_grad = True
        logger.info("Fine-tuning entire model (no parameters frozen)")

    def _freeze_encoders_and_feature_extractor(self, model: torch.nn.Module) -> None:
        """
        Freeze encoders and feature extractor, train only classification head.

        This method freezes all encoder and feature extractor parameters, allowing
        only the classification head to be updated during training. This is useful
        for transfer learning when only task-specific adaptation is needed.

        Parameters
        ----------
        model : torch.nn.Module
            Model to apply freezing strategy to.

        Returns
        -------
        None
        """
        # Freeze encoders
        for encoder_name in ["mol_encoder", "protein_encoder", "pocket_encoder"]:
            encoder = getattr(model, encoder_name, None)
            if encoder is not None:
                for param in encoder.parameters():
                    param.requires_grad = False

        # Freeze feature extractor
        if hasattr(model, "feature_extractor"):
            for param in model.feature_extractor.parameters():
                param.requires_grad = False

        # Ensure classification head is trainable
        if hasattr(model, "classification_head"):
            for param in model.classification_head.parameters():
                param.requires_grad = True

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Frozen encoders and feature extractor. Training only classification head."
        )
        logger.info(
            f"Trainable parameters: {trainable_params}/{total_params} ({100*trainable_params/total_params:.2f}%)"
        )

    def _extract_metrics(self, trainer: pl.Trainer, val_loader: Any) -> Dict[str, Any]:
        """
        Extract validation metrics from best model checkpoint.

        This method loads the best checkpoint and evaluates it on the validation set
        to retrieve final metrics.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer instance.
        val_loader : Any
            Validation dataloader for evaluation.

        Returns
        -------
        Dict[str, Any]
            Dictionary of validation metrics including loss, accuracy, F1, AUROC, and MCC.
        """
        # Same implementation as CCSAPhase._extract_metrics
        checkpoint_callback = None
        for callback in trainer.callbacks:
            if isinstance(callback, PrintOnBestModelCheckpoint):
                checkpoint_callback = callback
                break

        if checkpoint_callback and checkpoint_callback.best_model_path:
            try:
                model = trainer.lightning_module
                checkpoint = torch.load(
                    checkpoint_callback.best_model_path,
                    map_location="cpu",
                    weights_only=True,
                )
                model.load_state_dict(checkpoint["state_dict"])

                val_results = trainer.validate(
                    model, dataloaders=val_loader, verbose=False
                )
                result = (
                    val_results[-1]
                    if isinstance(val_results, list) and val_results
                    else {}
                )

                return {
                    "val/loss": float(result.get("val/loss", float("inf"))),
                    "val/accuracy": float(result.get("val/accuracy", 0.0)),
                    "val/f1": float(result.get("val/f1", 0.0)),
                    "val/auroc": float(result.get("val/auroc", 0.0)),
                    "val/mcc": float(result.get("val/mcc", 0.0)),
                }
            except Exception as e:
                logger.warning(f"Could not load best checkpoint: {e}")

        # Fallback
        if trainer.logged_metrics:
            return {
                "val/loss": float(trainer.logged_metrics.get("val/loss", float("inf"))),
                "val/accuracy": float(trainer.logged_metrics.get("val/accuracy", 0.0)),
                "val/f1": float(trainer.logged_metrics.get("val/f1", 0.0)),
                "val/auroc": float(trainer.logged_metrics.get("val/auroc", 0.0)),
                "val/mcc": float(trainer.logged_metrics.get("val/mcc", 0.0)),
            }

        return {
            "val/loss": float("inf"),
            "val/accuracy": 0.0,
            "val/f1": 0.0,
            "val/auroc": 0.0,
            "val/mcc": 0.0,
        }
