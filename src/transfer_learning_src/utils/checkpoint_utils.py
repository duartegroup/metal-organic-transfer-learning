import os
import pickle
import logging
import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
from typing import List, Optional, Dict, Any
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryMatthewsCorrCoef,
    BinaryConfusionMatrix,
)

logger = logging.getLogger(__name__)


# == STATS == #
def build_stats_path(output_dir: str) -> str:
    """
    Build the path for normalization statistics file.

    Parameters
    ----------
    output_dir : str
        Output directory for saving statistics.

    Returns
    -------
    str
        Full path to the normalization statistics file.
    """
    return os.path.join(output_dir, f"normalization_stats.pkl")


def save_phase_stats(stats: dict, output_dir: str) -> None:
    """
    Save normalization statistics to a pickle file.

    Parameters
    ----------
    stats : dict
        Dictionary containing normalization statistics (e.g., mean, std).
    output_dir : str
        Output directory for saving the statistics file.

    Returns
    -------
    None
    """
    path = build_stats_path(output_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(stats, f)
    logging.info(f"Saved stats to {path}")


def load_phase_stats(output_dir: str) -> dict:
    """
    Load normalization statistics from a pickle file.

    Parameters
    ----------
    output_dir : str
        Output directory where statistics file is located.

    Returns
    -------
    dict
        Dictionary containing normalization statistics.

    Raises
    ------
    FileNotFoundError
        If the statistics file does not exist.
    """
    path = build_stats_path(output_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Stats not found: {path}")
    with open(path, "rb") as f:
        stats = pickle.load(f)
    logging.info(f"Loaded stats from {path}")
    return stats


# == COMPONENTS == #
def save_model_components(
    model: torch.nn.Module, output_dir: str, components_map: Dict[str, str]
) -> List[str]:
    """
    Save specific PyTorch modules based on a provided map.

    This function saves individual model components (e.g., encoders, heads) to separate
    files, along with a manifest tracking what was saved.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model containing components to save.
    output_dir : str
        Directory where component files will be saved.
    components_map : Dict[str, str]
        Mapping from component names to their attribute paths in the model
        (e.g., {'mol_encoder': 'mol_encoder', 'classification_head': 'classification_head'}).

    Returns
    -------
    List[str]
        List of successfully saved component names.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_components = []

    for comp_name, module_path in components_map.items():
        module_obj = _get_nested_attr(model, module_path)

        if module_obj is None:
            logging.warning(
                f"Component '{comp_name}' at path '{module_path}' not found in model; skipping save."
            )
            continue

        try:
            save_path = os.path.join(output_dir, f"{comp_name}.pt")
            torch.save(module_obj.state_dict(), save_path)
            saved_components.append(comp_name)
            logging.info(f"Saved {comp_name} to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save {comp_name}: {str(e)}")

    # Save a manifest of what was saved and the map used
    manifest = {"saved_components": saved_components, "components_map": components_map}
    manifest_path = os.path.join(output_dir, "manifest.pt")
    torch.save(manifest, manifest_path)

    logging.info(f"Saved {len(saved_components)} model components to {output_dir}")
    logging.info(f"Saved components: {saved_components}")
    return saved_components


def load_model_component(
    load_dir: str,
    model: torch.nn.Module,
    components_map: Optional[Dict[str, str]] = None,
    strict: bool = True,
) -> List[str]:
    """
    Load specific PyTorch modules from saved files.

    This function loads saved model components and restores them into the provided model.
    If no components_map is provided, it attempts to load from a manifest file.

    Parameters
    ----------
    load_dir : str
        Directory containing saved component files.
    model : torch.nn.Module
        Target model to load components into.
    components_map : Optional[Dict[str, str]], optional
        Mapping from component names to their attribute paths in the model.
        If None, loads from manifest.pt, by default None.
    strict : bool, optional
        If True, raises errors on missing files/components. If False, logs warnings
        and continues, by default True.

    Returns
    -------
    List[str]
        List of successfully loaded component names.

    Raises
    ------
    FileNotFoundError
        If load_dir doesn't exist (when strict=True).
    ValueError
        If manifest doesn't contain components_map (when strict=True).
    RuntimeError
        If component loading fails (when strict=True).
    """
    if not os.path.exists(load_dir):
        msg = f"Load directory {load_dir} does not exist"
        if strict:
            raise FileNotFoundError(msg)
        else:
            logging.warning(msg)
            return []

    # If no map is provided, try to get it from the manifest
    if components_map is None:
        manifest_path = os.path.join(load_dir, "manifest.pt")
        if os.path.exists(manifest_path):
            manifest = torch.load(manifest_path, map_location="cpu")
            components_map = manifest.get("components_map")
            if components_map is None:
                msg = (
                    f"Manifest at {manifest_path} does not contain a 'components_map'."
                )
                if strict:
                    raise ValueError(msg)
                else:
                    logging.warning(msg)
                    return []
            logging.info(
                f"Loaded components map from manifest: {list(components_map.keys())}"
            )
        else:
            msg = f"Load directory {load_dir} does not contain a manifest.pt and no components_map was provided."
            if strict:
                raise FileNotFoundError(msg)
            else:
                logging.warning(msg)
                return []

    loaded_components = []
    for comp_name, module_path in components_map.items():
        # Check if the component exists in the model first
        module_obj = _get_nested_attr(model, module_path)
        if module_obj is None:
            logging.debug(
                f"Component '{comp_name}' at path '{module_path}' not found in the target model; skipping load."
            )
            continue

        load_path = os.path.join(load_dir, f"{comp_name}.pt")
        if not os.path.exists(load_path):
            msg = f"Component file for '{comp_name}' not found at {load_path}"
            if strict:
                raise FileNotFoundError(msg)
            else:
                logging.warning(msg + "; skipping.")
                continue

        try:
            state_dict = torch.load(load_path, map_location="cpu")
            module_obj.load_state_dict(state_dict)
            loaded_components.append(comp_name)
            logging.info(f"Loaded '{comp_name}' from {load_path}")
        except Exception as e:
            msg = f"Failed to load state dict for '{comp_name}' from {load_path}: {str(e)}"
            if strict:
                raise RuntimeError(msg)
            else:
                logging.error(msg)

    logging.info(
        f"Successfully loaded {len(loaded_components)} components: {loaded_components}"
    )
    return loaded_components


def _get_nested_attr(obj: Any, attr_path: str) -> Any:
    """
    Get nested attribute from an object using dot notation.

    Parameters
    ----------
    obj : Any
        Object from which to extract the nested attribute.
    attr_path : str
        Attribute path using dot notation (e.g., 'model.encoder.layer').

    Returns
    -------
    Any
        The nested attribute object, or None if the attribute doesn't exist.
    """
    try:
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        return obj
    except AttributeError:
        return None


def list_available_components(output_dir: str, fold: int, mode: str) -> List[str]:
    """
    List all saved model components for a given fold and mode.

    This function checks a directory for saved model components and returns a list
    of available component names, either from the manifest or by scanning for .pt files.

    Parameters
    ----------
    output_dir : str
        Base output directory containing saved components.
    fold : int
        Cross-validation fold number.
    mode : str
        Training mode (e.g., 'pretrain', 'ccsa', 'encoder_task').

    Returns
    -------
    List[str]
        List of available component names, or empty list if directory doesn't exist.
    """
    load_dir = os.path.join(output_dir, f"fold_{fold}", mode)

    if not os.path.exists(load_dir):
        return []

    # Try manifest first
    manifest_path = os.path.join(load_dir, "manifest.pt")
    if os.path.exists(manifest_path):
        manifest = torch.load(manifest_path, map_location="cpu")
        return manifest.get("saved_components", [])

    # Fallback: list .pt files
    components = []
    for file in os.listdir(load_dir):
        if file.endswith(".pt") and file != "manifest.pt":
            components.append(file[:-3])

    return components


# === MultiTaskPocket combination-specific functions === #
def get_combination_configs() -> List[Dict[str, bool]]:
    """
    Get all possible modality combinations for multi-task training.

    This function returns all 7 possible combinations of using molecular,
    protein, and pocket features (excluding the combination where all are False).

    Parameters
    ----------
    None

    Returns
    -------
    List[Dict[str, bool]]
        List of dictionaries, each specifying a unique combination of
        use_mol, use_protein, and use_pocket flags.
    """
    return [
        {"use_mol": True, "use_protein": False, "use_pocket": False},  # mol only
        {"use_mol": False, "use_protein": True, "use_pocket": False},  # protein only
        {"use_mol": False, "use_protein": False, "use_pocket": True},  # pocket only
        {"use_mol": True, "use_protein": True, "use_pocket": False},  # mol + protein
        {"use_mol": True, "use_protein": False, "use_pocket": True},  # mol + pocket
        {"use_mol": False, "use_protein": True, "use_pocket": True},  # protein + pocket
        {"use_mol": True, "use_protein": True, "use_pocket": True},  # all three
    ]


def train_all_combinations(
    base_params: Dict[str, Any], output_dir: str, fold: int, **training_kwargs: Any
) -> Dict[str, Any]:
    """
    Train models for all modality combinations.

    This function trains separate models for each possible combination of feature
    modalities (molecular, protein, pocket) and saves them with appropriate naming.

    Parameters
    ----------
    base_params : Dict[str, Any]
        Base parameters for model initialization.
    output_dir : str
        Directory where trained models will be saved.
    fold : int
        Cross-validation fold number.
    **training_kwargs : Any
        Additional keyword arguments for training (currently unused).

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping combination names to their results, containing:
        - 'model': The trained model
        - 'combination_config': The modality configuration used
        - 'save_dir': Directory where the model was saved
    """
    combinations = get_combination_configs()
    results = {}

    for combo_config in combinations:
        # Create model with this combination
        model_params = {**base_params, **combo_config}
        model = MultiTaskPocket(**model_params)

        logging.info(f"Training combination: {model.combination_name}")

        # Define components to save for this combination
        components_map = {}
        if model.use_mol:
            components_map["mol_encoder"] = "mol_encoder"
        if model.use_protein:
            components_map["protein_encoder"] = "protein_encoder"
        if model.use_pocket:
            components_map["pocket_encoder"] = "pocket_encoder"
        if hasattr(model, "feature_extractor"):
            components_map["feature_extractor"] = "feature_extractor"
        if hasattr(model, "classification_head"):
            components_map["classification_head"] = "classification_head"
        if hasattr(model, "projection_head"):
            components_map["projection_head"] = "projection_head"

        # Save the trained model
        save_dir = model.get_combination_save_dir(output_dir, fold)
        save_model_components(model, save_dir, components_map=components_map)

        # Store results
        results[model.combination_name] = {
            "model": model,
            "combination_config": combo_config,
            "save_dir": save_dir,
        }

    return results


class PrintOnBestModelCheckpoint(ModelCheckpoint):
    """
    Custom ModelCheckpoint that logs detailed information when a new best model is found.

    This callback extends PyTorch Lightning's ModelCheckpoint to provide enhanced logging
    of model performance metrics, including MCC scores and confusion matrices, whenever
    a new best model is detected.

    Parameters
    ----------
    print_at_end : bool, optional
        If True, print a summary of the best model at the end of training, by default False.
    print_on_train_epoch : Optional[int], optional
        If True, log best model info at the end of each training epoch, by default None.
    print_on_val_epoch : Optional[int], optional
        If True, log best model info at the end of each validation epoch, by default None.
    *args : Any
        Positional arguments passed to parent ModelCheckpoint.
    **kwargs : Any
        Keyword arguments passed to parent ModelCheckpoint.

    Attributes
    ----------
    best_model_message : str
        Formatted message describing the best model found.
    best_confusion : Optional[Tuple[int, int, int, int]]
        Confusion matrix (tn, fp, fn, tp) for the best model.
    """

    def __init__(
        self,
        print_at_end: bool = False,
        print_on_train_epoch: Optional[int] = None,
        print_on_val_epoch: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._last_best_score = self.best_model_score
        self.print_at_end = print_at_end
        self.best_model_message = ""
        self.best_confusion = None  # (tn, fp, fn, tp)

        # Control when to print: if both are None, default to validation only for backward compatibility
        if print_on_train_epoch is None and print_on_val_epoch is None:
            self.print_on_train_epoch = False
            self.print_on_val_epoch = True
        else:
            self.print_on_train_epoch = (
                print_on_train_epoch if print_on_train_epoch is not None else False
            )
            self.print_on_val_epoch = (
                print_on_val_epoch if print_on_val_epoch is not None else False
            )

    def _log_new_best_model(self, trainer: pl.Trainer) -> None:
        """
        Log detailed information about a newly detected best model.

        This method extracts and logs the monitored metric, MCC score, and confusion
        matrix from the trainer's callback metrics whenever a new best model is found.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer instance containing callback metrics.

        Returns
        -------
        None
        """
        if (
            self.best_model_score is not None
            and self.best_model_score != self._last_best_score
        ):
            self._last_best_score = self.best_model_score

            msg = (
                f"New best model found at epoch {trainer.current_epoch} "
                f"with {self.monitor}: {self.best_model_score:.4f}"
            )

            # accept both slash and underscore variants (progress bar uses slashes)
            def _to_num(v: Any) -> Optional[float]:
                if isinstance(v, torch.Tensor):
                    return float(v.detach().cpu().item())
                if isinstance(v, (int, float)):
                    return float(v)
                return None

            # MCC key options
            for k in (
                "val/mcc",
                "tgt_val/mcc",
                "val_mcc",
                "tgt_val_mcc",
                "mcc",
                "val_mcc_epoch",
            ):
                if k in trainer.callback_metrics:
                    mcc_score = _to_num(trainer.callback_metrics[k])
                    if mcc_score is not None:
                        msg += f" | MCC: {mcc_score:.4f}"
                        break

            # Confusion matrix key options (prefer slash)
            cm_key_sets = [
                ("val/tn", "val/fp", "val/fn", "val/tp"),
                ("tgt_val/tn", "tgt_val/fp", "tgt_val/fn", "tgt_val/tp"),
                ("val_tn", "val_fp", "val_fn", "val_tp"),
                ("tgt_val_tn", "tgt_val_fp", "tgt_val_fn", "tgt_val_tp"),
            ]
            for tn_k, fp_k, fn_k, tp_k in cm_key_sets:
                tn = _to_num(trainer.callback_metrics.get(tn_k, None))
                fp = _to_num(trainer.callback_metrics.get(fp_k, None))
                fn = _to_num(trainer.callback_metrics.get(fn_k, None))
                tp = _to_num(trainer.callback_metrics.get(tp_k, None))
                if None not in (tn, fp, fn, tp):
                    # store ints
                    self.best_confusion = (int(tn), int(fp), int(fn), int(tp))
                    break

            logger.info(msg)
            self.best_model_message = msg

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Log detailed information about the best model at the end of each validation epoch.

        This method logs the best model's performance metrics and confusion matrix whenever
        a new best model is detected during validation.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer instance.
        pl_module : pl.LightningModule
            PyTorch Lightning module instance.

        Returns
        -------
        None
        """

        super().on_validation_epoch_end(trainer, pl_module)
        if self.print_on_val_epoch:
            self._log_new_best_model(trainer)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        Log detailed information about the best model at the end of each training epoch.

        This method logs the best model's performance metrics and confusion matrix whenever
        a new best model is detected during training.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer instance.
        pl_module : pl.LightningModule
            PyTorch Lightning module instance.

        Returns
        -------
        None
        """
        super().on_train_epoch_end(trainer, pl_module)
        # Check both the new flag and the existing Lightning flag for backward compatibility
        if self.print_on_train_epoch or getattr(
            self, "check_on_train_epoch_end", False
        ):
            self._log_new_best_model(trainer)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Log detailed information about the best model at the end of training.

        This method logs the best model's performance metrics and confusion matrix whenever
        a new best model is detected during training.

        Parameters
        ----------
        trainer : pl.Trainer
            PyTorch Lightning trainer instance.
        pl_module : pl.LightningModule
            PyTorch Lightning module instance.

        Returns
        -------
        None
        """
        super().on_train_end(trainer, pl_module)
        if self.print_at_end and self.best_model_path:
            logger.info("=" * 60)
            logger.info("Final Best Model Summary:")
            if self.best_model_message:
                logger.info(self.best_model_message)
            else:
                logger.info(
                    f"Best model score ({self.monitor}): {self.best_model_score:.4f}"
                )
            logger.info(f"Best model path: {self.best_model_path}")

            if self.best_confusion is not None:
                tn, fp, fn, tp = self.best_confusion
                logger.info("=" * 60)
                logger.info("Best Confusion Matrix (at best score):")
                logger.info("                 Predicted")
                logger.info("                 0      1")
                logger.info(f"Actual    0   {tn:4d}   {fp:4d}")
                logger.info(f"          1   {fn:4d}   {tp:4d}")
            logger.info("=" * 60)
