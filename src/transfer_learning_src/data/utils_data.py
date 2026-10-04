import logging
import torch
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
from typing import Optional, Dict, Tuple, Union, Any

from .split_data import split_data_by_group, split_data_by_group_random
from .load_data import load_data_hdf5
from .dataset import BindingAffinitiesDataset, BindingAffinitiesDatasetWithKeys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_scaling_stats(
    data_dict: Dict[str, dict], modality: str, scaling_method: str = "standard"
) -> Optional[Dict[str, Any]]:
    """
    Compute scaling statistics from dataset dictionary using unified scaling approach.

    Parameters
    ----------
    data_dict : dict[str, dict]
        Dictionary with 'inhibitor_feats', 'protein_feats', 'pocket_feats' keys.
    modality : str
        Modality string indicating which features to include (e.g., 'mol', 'protein', 'pocket').
    scaling_method : str, optional
        Scaling method ('standard', 'minmax', or 'none'). Default is 'standard'.

    Returns
    -------
    dict[str, Any] or None
        Scaling statistics for the combined feature vector, or None if scaling_method is 'none'.

    Raises
    ------
    ValueError
        If invalid modality string or unknown scaling method.
    """
    if scaling_method is None or scaling_method == "none":
        return None

    features_to_combine = []
    mol_dim = protein_dim = pocket_dim = None

    if "mol" in modality:
        mol_vecs = torch.stack(
            [
                torch.tensor(mol).mean(dim=0)
                for mol in data_dict["inhibitor_feats"].values()
            ]
        )
        features_to_combine.append(mol_vecs)
        mol_dim = mol_vecs.shape[1]

    if "protein" in modality:
        protein_vecs = torch.stack(
            [torch.tensor(prot)[0, 0] for prot in data_dict["protein_feats"].values()]
        )
        features_to_combine.append(protein_vecs)
        protein_dim = protein_vecs.shape[1]

    if "pocket" in modality:
        pocket_vecs = torch.stack(
            [
                torch.tensor(pocket).mean(dim=0)
                for pocket in data_dict["pocket_feats"].values()
            ]
        )
        features_to_combine.append(pocket_vecs)
        pocket_dim = pocket_vecs.shape[1]

    if not features_to_combine:
        raise ValueError(
            "Invalid modality string. At least one of 'mol', 'protein', or 'pocket' must be present."
        )

    combined_features = torch.cat(features_to_combine, dim=1)

    # Store dimensions for later splitting
    stats = {
        "scaling_method": scaling_method,
        "mol_dim": mol_dim,
        "protein_dim": protein_dim,
        "pocket_dim": pocket_dim,
    }

    if scaling_method == "standard":
        # Z-score normalization for combined features
        combined_mean = combined_features.mean(dim=0).numpy()
        combined_std = combined_features.std(dim=0).numpy()

        # Add epsilon to prevent division by zero
        combined_std = np.where(combined_std == 0, 1e-8, combined_std)

        stats.update({"combined_mean": combined_mean, "combined_std": combined_std})

    elif scaling_method == "minmax":
        # Min-max normalization for combined features
        combined_min = combined_features.min(dim=0)[0].numpy()
        combined_max = combined_features.max(dim=0)[0].numpy()

        stats.update({"combined_min": combined_min, "combined_max": combined_max})

    else:
        raise ValueError(f"Unknown scaling method: {scaling_method}")

    return stats


def compute_source_scaling_stats(
    source_raw: Dict[str, dict], modality: str, scaling_method: str = "standard"
) -> Optional[Dict[str, Any]]:
    """
    Compute scaling statistics from source domain data only.

    These statistics will be used for both source and target data in cross-domain scaling.

    Parameters
    ----------
    source_raw : dict[str, dict]
        Source domain data dictionary with 'inhibitor_feats', 'protein_feats', 'pocket_feats' keys.
    modality : str
        Modality string indicating which features to include.
    scaling_method : str, optional
        Scaling method ('standard', 'minmax', or 'none'). Default is 'standard'.

    Returns
    -------
    dict[str, Any] or None
        Scaling statistics computed from source data, or None if scaling_method is 'none'.
    """
    if scaling_method is None or scaling_method == "none":
        return None

    logger.info(
        f"Computing scaling statistics from source domain using {scaling_method} method"
    )
    return compute_scaling_stats(source_raw, modality, scaling_method)


def get_datasets(
    params: Dict[str, Any],
    fold: Optional[int] = None,
    global_scaling_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[
    Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset], Optional[Dataset]],
    Dict[str, Any],
]:
    """
    Unified dataset loading function with proper parameter handling.

    Parameters
    ----------
    params : dict[str, Any]
        Parameters including train_on, data_scaler, source_smote, cross_domain_scaling, etc.
    fold : int or None, optional
        Fold number for target data splitting. Default is None.
    global_scaling_stats : dict[str, Any] or None, optional
        Pre-computed scaling statistics for consistent scaling. Default is None.

    Returns
    -------
    tuple[tuple[Dataset or None, ...], dict[str, Any]]
        A tuple containing:
        - datasets_tuple: (src_train_ds, src_val_ds, tgt_train_ds, tgt_val_ds)
        - scaling_stats_dict: Dictionary with scaling statistics and metadata

    Raises
    ------
    ValueError
        If fold is required but not provided, or invalid train_on value.
    """
    train_on = params["train_on"]
    use_source = "src" in train_on
    use_target = "tgt" in train_on
    scaling_method = params.get("data_scaler", None)
    source_smote = params.get("source_smote", False)
    target_smote = params.get("target_smote", False)
    cross_domain_scaling = params.get("cross_domain_scaling", False)

    # Normalize scaling method
    if scaling_method == "none":
        scaling_method = None

    # Load raw data
    source_raw = load_dataset("organic", params) if use_source else None
    target_raw = load_dataset("caged", params) if use_target else None

    # Split data based on train_on parameter
    src_train = src_val = tgt_train = tgt_val = None

    if train_on == "src":
        src_train, src_val = split_data_by_group_random(
            source_raw["inhibitor_feats"],
            source_raw["protein_feats"],
            source_raw["pocket_feats"],
            source_raw["binding_affs"],
            seed=params.get("seed", 42),
            split_ratio=params.get("split_ratio", 1.0),
        )
    elif train_on == "tgt":
        if fold is None:
            raise ValueError("Fold number required for target data splitting")
        tgt_train, tgt_val = split_data_by_group(
            target_raw["inhibitor_feats"],
            target_raw["protein_feats"],
            target_raw["pocket_feats"],
            target_raw["binding_affs"],
            "caged",
            params["affinity_type"],
            fold=fold,
            n_folds=params["n_folds"],
        )
    elif train_on == "src+tgt":
        if fold is None:
            raise ValueError("Fold number required for target data splitting")
        # Use all source data for training, split target
        src_train = source_raw
        src_val = None
        tgt_train, tgt_val = split_data_by_group(
            target_raw["inhibitor_feats"],
            target_raw["protein_feats"],
            target_raw["pocket_feats"],
            target_raw["binding_affs"],
            "caged",
            params["affinity_type"],
            fold=fold,
            n_folds=params["n_folds"],
        )
    else:
        raise ValueError(f"Invalid train_on value: {train_on}")

    if params.get("visualize", False):
        with_keys = True
    else:
        with_keys = False

    # Build datasets using unified function with consistent parameter passing
    return prepare_datasets(
        src_train,
        src_val,
        tgt_train,
        tgt_val,
        params["modality"],
        scaling_method=scaling_method,
        scaling_stats=global_scaling_stats,  # Use consistent parameter name
        source_smote=source_smote,
        target_smote=target_smote,
        cross_domain_scaling=cross_domain_scaling,
        with_keys=with_keys,
    )


def load_dataset(
    dataset_name: str,
    inhibitor_descriptor_or_params: Union[str, Dict[str, Any]],
    affinity_type: Optional[str] = None,
    organic_percent_of_data: float = 1.0,
) -> Dict[str, dict]:
    """
    Load dataset with support for both individual parameters and params dict.

    Supports two calling patterns:
    1. load_dataset("organic", params_dict)
    2. load_dataset("organic", "mace", "pic50")

    Parameters
    ----------
    dataset_name : str
        Dataset name ('organic' or 'caged').
    inhibitor_descriptor_or_params : str or dict[str, Any]
        Either inhibitor descriptor string (e.g., 'mace') or full params dictionary.
    affinity_type : str or None, optional
        Affinity type string (e.g., 'pic50', 'pk'). Required if second arg is not params dict. Default is None.
    organic_percent_of_data : float, optional
        Percentage of data to load from organic dataset (0.0-1.0). Default is 1.0.

    Returns
    -------
    dict[str, dict]
        Dictionary containing 'inhibitor_feats', 'protein_feats', 'pocket_feats', 'binding_affs'.

    Raises
    ------
    ValueError
        If affinity_type is not provided when using individual parameter calling pattern.
    """
    # Handle both calling patterns
    if isinstance(inhibitor_descriptor_or_params, dict):
        # Called with params dict: load_dataset("organic", params)
        params = inhibitor_descriptor_or_params
        inhibitor_descriptor = params.get("inhibitor_descriptor", "mace")
        affinity_type = params.get("affinity_type", "pic50")
        organic_percent_of_data = params.get("organic_percent_of_data", 1.0)
    else:
        # Called with individual args: load_dataset("organic", "mace", "pic50")
        inhibitor_descriptor = inhibitor_descriptor_or_params
        if affinity_type is None:
            raise ValueError(
                "affinity_type must be provided when not using params dict"
            )

    if dataset_name == "organic":
        # Load organic compounds dataset (source domain)
        data = load_data_hdf5(
            dataset_name,
            inhibitor_descriptor,
            affinity_type,
            percent_of_data=organic_percent_of_data,
        )
        return data  # Return the 5-tuple directly
    elif dataset_name == "caged":
        # Load caged compounds dataset (target domain)
        data = load_data_hdf5(
            dataset_name,
            inhibitor_descriptor,
            affinity_type,
        )
        return data  # Return the 5-tuple directly


def prepare_datasets(
    src_train: Optional[Dict[str, dict]],
    src_val: Optional[Dict[str, dict]],
    tgt_train: Optional[Dict[str, dict]],
    tgt_val: Optional[Dict[str, dict]],
    modality: str,
    scaling_method: Optional[str] = None,
    scaling_stats: Optional[Dict[str, Any]] = None,
    source_smote: bool = False,
    target_smote: bool = False,
    cross_domain_scaling: bool = False,
    with_keys: bool = False,
) -> Tuple[
    Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset], Optional[Dataset]],
    Dict[str, Any],
]:
    """
    Unified function to prepare datasets with proper scaling control.

    The scaling strategy is controlled by cross_domain_scaling:
    - True: Use source statistics for both domains (consistent cross-domain scaling)
    - False: Compute separate statistics for each domain (domain-specific scaling)

    Parameters
    ----------
    src_train : dict[str, dict] or None
        Source training data dictionary.
    src_val : dict[str, dict] or None
        Source validation data dictionary.
    tgt_train : dict[str, dict] or None
        Target training data dictionary.
    tgt_val : dict[str, dict] or None
        Target validation data dictionary.
    modality : str
        Modality string indicating which features to use.
    scaling_method : str or None, optional
        Scaling method ('standard', 'minmax', or None). Default is None.
    scaling_stats : dict[str, Any] or None, optional
        Pre-computed scaling statistics. Default is None.
    source_smote : bool, optional
        Whether to apply SMOTE to source training data. Default is False.
    target_smote : bool, optional
        Whether to apply SMOTE to target training data. Default is False.
    cross_domain_scaling : bool, optional
        Whether to use cross-domain scaling (source stats for both). Default is False.
    with_keys : bool, optional
        Whether to use BindingAffinitiesDatasetWithKeys. Default is False.

    Returns
    -------
    tuple[tuple[Dataset or None, ...], dict[str, Any]]
        A tuple containing:
        - datasets: (src_train_ds, src_val_ds, tgt_train_ds, tgt_val_ds)
        - scaling_stats_return: Dictionary with scaling statistics and metadata

    Raises
    ------
    ValueError
        If cross-domain scaling is requested but no source data or stats available.
    """
    logger.info(f"Scaling method: {scaling_method}")
    logger.info(f"Apply SMOTE to target: {target_smote}")
    logger.info(f"Apply SMOTE to source: {source_smote}")
    logger.info(f"Cross-domain scaling: {cross_domain_scaling}")

    # Enhanced scaling strategy based on cross_domain_scaling parameter
    src_scaling_stats = tgt_scaling_stats = None

    if scaling_method is not None:
        if cross_domain_scaling:
            # CROSS-DOMAIN SCALING: Use source statistics for both domains
            if scaling_stats is not None:
                # Use provided global scaling stats (should be source-based)
                src_scaling_stats = tgt_scaling_stats = scaling_stats
                logger.info(
                    "Using pre-computed cross-domain scaling statistics (source-based)"
                )
            elif src_train is not None:
                # Compute from source and apply to both
                src_scaling_stats = compute_scaling_stats(
                    src_train, modality, scaling_method
                )
                tgt_scaling_stats = src_scaling_stats
                logger.info(
                    "Computed cross-domain scaling statistics from source training data"
                )
            else:
                raise ValueError(
                    "Cross-domain scaling requested but no source data or global stats available"
                )
        else:
            # DOMAIN-SPECIFIC SCALING: Compute separate statistics for each domain
            if src_train is not None:
                src_scaling_stats = compute_scaling_stats(
                    src_train, modality, scaling_method
                )
                logger.info(
                    "Computed domain-specific scaling statistics from source training data"
                )

            if tgt_train is not None:
                tgt_scaling_stats = compute_scaling_stats(
                    tgt_train, modality, scaling_method
                )
                logger.info(
                    "Computed domain-specific scaling statistics from target training data"
                )

            # If global stats provided but cross_domain_scaling=False, respect the setting
            if scaling_stats is not None:
                logger.warning(
                    "Global scaling stats provided but cross_domain_scaling=False. Using domain-specific scaling."
                )
    else:
        logger.info("No scaling will be applied")

    def build_dataset(
        inhibitor: dict,
        protein: dict,
        pocket: dict,
        binding_affinities: dict,
        modality: str,
        apply_smote: bool = False,
        scaler_stats: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dataset]:
        """
        Helper function to build dataset with consistent parameters.

        Parameters
        ----------
        inhibitor : dict
            Inhibitor features dictionary.
        protein : dict
            Protein features dictionary.
        pocket : dict
            Pocket features dictionary.
        binding_affinities : dict
            Binding affinities dictionary.
        modality : str
            Modality string.
        apply_smote : bool, optional
            Whether to apply SMOTE. Default is False.
        scaler_stats : dict[str, Any] or None, optional
            Scaling statistics. Default is None.

        Returns
        -------
        Dataset or None
            Dataset instance or None if data is empty.
        """
        if not inhibitor or not protein or not pocket or not binding_affinities:
            return None
        if with_keys:
            return BindingAffinitiesDatasetWithKeys(
                inhibitor,
                protein,
                pocket,
                binding_affinities,
                modality,
                apply_smote=apply_smote,
                scaler_type=scaling_method,
                scaler_stats=scaler_stats,
            )
        else:
            return BindingAffinitiesDataset(
                inhibitor,
                protein,
                pocket,
                binding_affinities,
                modality,
                apply_smote=apply_smote,
                scaler_type=scaling_method,
                scaler_stats=scaler_stats,
            )

    # Build datasets with appropriate scaling and SMOTE settings
    src_train_ds = src_val_ds = tgt_train_ds = tgt_val_ds = None

    if src_train is not None:
        src_train_ds = build_dataset(
            src_train["inhibitor_feats"],
            src_train["protein_feats"],
            src_train["pocket_feats"],
            src_train["binding_affs"],
            modality,
            apply_smote=source_smote,
            scaler_stats=src_scaling_stats,
        )
    if src_val is not None:
        src_val_ds = build_dataset(
            src_val["inhibitor_feats"],
            src_val["protein_feats"],
            src_val["pocket_feats"],
            src_val["binding_affs"],
            modality,
            apply_smote=False,
            scaler_stats=src_scaling_stats,
        )
    if tgt_train is not None:
        tgt_train_ds = build_dataset(
            tgt_train["inhibitor_feats"],
            tgt_train["protein_feats"],
            tgt_train["pocket_feats"],
            tgt_train["binding_affs"],
            modality,
            apply_smote=target_smote,  # Apply SMOTE to target training
            scaler_stats=tgt_scaling_stats,
        )
    if tgt_val is not None:
        tgt_val_ds = build_dataset(
            tgt_val["inhibitor_feats"],
            tgt_val["protein_feats"],
            tgt_val["pocket_feats"],
            tgt_val["binding_affs"],
            modality,
            apply_smote=False,
            scaler_stats=tgt_scaling_stats,
        )

    # Return scaling stats in a structured way
    if cross_domain_scaling:
        scaling_stats_return = {
            "cross_domain": True,
            "scaling_stats": src_scaling_stats,  # Source stats used for both
            "scaling_method": scaling_method,
        }
    else:
        scaling_stats_return = {
            "cross_domain": False,
            "source_stats": src_scaling_stats,
            "target_stats": tgt_scaling_stats,
            "scaling_method": scaling_method,
        }

    return (src_train_ds, src_val_ds, tgt_train_ds, tgt_val_ds), scaling_stats_return


def create_dataloaders(
    src_train_dataset: Optional[Dataset],
    src_val_dataset: Optional[Dataset],
    tgt_train_dataset: Optional[Dataset],
    tgt_val_dataset: Optional[Dataset],
    params: Dict[str, Any],
) -> Dict[str, Optional[TorchDataLoader]]:
    """
    Create dataloaders from datasets.

    Parameters
    ----------
    src_train_dataset : Dataset or None
        Source training dataset.
    src_val_dataset : Dataset or None
        Source validation dataset.
    tgt_train_dataset : Dataset or None
        Target training dataset.
    tgt_val_dataset : Dataset or None
        Target validation dataset.
    params : dict[str, Any]
        Parameters dictionary containing 'batch_size'.

    Returns
    -------
    dict[str, DataLoader or None]
        Dictionary with keys: 'src_train_loader', 'src_val_loader', 'tgt_train_loader', 'tgt_val_loader'.
    """
    batch_size = params["batch_size"]
    src_train_loader = src_val_loader = tgt_train_loader = tgt_val_loader = None

    if src_train_dataset is not None:
        src_train_loader = TorchDataLoader(
            src_train_dataset, batch_size=batch_size, shuffle=True
        )

    if src_val_dataset is not None:
        src_val_loader = TorchDataLoader(
            src_val_dataset, batch_size=batch_size, shuffle=False
        )

    if tgt_train_dataset is not None:
        tgt_train_loader = TorchDataLoader(
            tgt_train_dataset, batch_size=batch_size, shuffle=True
        )

    if tgt_val_dataset is not None:
        tgt_val_loader = TorchDataLoader(
            tgt_val_dataset, batch_size=batch_size, shuffle=False
        )

    loaders = {
        "src_train_loader": src_train_loader,
        "src_val_loader": src_val_loader,
        "tgt_train_loader": tgt_train_loader,
        "tgt_val_loader": tgt_val_loader,
    }

    # Print dataset sizes
    logger.info("Dataset sizes:")
    if src_train_dataset:
        logger.info(f"Source train: {len(src_train_dataset)}")
        logger.info(
            f" (-) {src_train_dataset.neg_count} : (+) {src_train_dataset.pos_count}"
        )
    if src_val_dataset:
        logger.info(f"Source val: {len(src_val_dataset)}")
        logger.info(
            f" (-) {src_val_dataset.neg_count} : (+) {src_val_dataset.pos_count}"
        )
    if tgt_train_dataset:
        logger.info(f"Target train: {len(tgt_train_dataset)}")
        logger.info(
            f" (-) {tgt_train_dataset.neg_count} : (+) {tgt_train_dataset.pos_count}"
        )
    if tgt_val_dataset:
        logger.info(f"Target val: {len(tgt_val_dataset)}")
        logger.info(
            f" (-) {tgt_val_dataset.neg_count} : (+) {tgt_val_dataset.pos_count}"
        )
    return loaders
