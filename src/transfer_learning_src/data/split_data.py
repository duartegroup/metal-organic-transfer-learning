import random
import pickle
from pathlib import Path
import logging
from typing import Dict, List, Tuple


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# == SPLIT DATSET INTO TRAIN AND TEST == #
def split_data_by_group(
    inhibitor_feats: dict,
    protein_feats: dict,
    pocket_feats: dict,
    binding_affs: dict,
    dataset: str,
    affinity_type: str,
    fold: int = 1,
    n_folds: int = 4,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    Split data into train and test sets using pre-computed fold indices from optimization.

    Parameters
    ----------
    inhibitor_feats : dict
        The inhibitor features.
    pocket_feats : dict
        The pocket embeddings.
    protein_feats : dict
        The protein embeddings.
    binding_affs : dict
        The binding affinities.
    dataset : str
        The type of data (e.g., 'caged', 'organic').
    affinity_type : str
        The type of affinity (e.g., 'pic50', 'pk').
    fold : int
        The split to return (1-indexed).
    n_folds : int
        The number of folds.

    Returns
    -------
    tuple[dict[str, dict], dict[str, dict]]
        A tuple containing:
        - train_data: Dictionary with keys 'inhibitor_feats', 'protein_feats', 'pocket_feats', 'binding_affs'
        - test_data: Dictionary with keys 'inhibitor_feats', 'protein_feats', 'pocket_feats', 'binding_affs'
    """
    # Load the pre-computed fold indices
    fold_indices = load_fold_indices(dataset, affinity_type, n_folds)

    # Convert from 1-indexed to 0-indexed for accessing fold_indices
    fold_idx = fold - 1

    if fold_idx not in fold_indices:
        raise ValueError(
            f"fold {fold} (index {fold_idx}) not found in fold_indices. Available: {list(fold_indices.keys())}"
        )

    # Get the train and test names for this fold
    train_names = fold_indices[fold_idx]["train_names"]
    test_names = fold_indices[fold_idx]["test_names"]

    # Extract data for this fold
    train_inhibitor_features = {
        name: inhibitor_feats[name] for name in train_names if name in inhibitor_feats
    }
    train_protein_embeddings = {
        name: protein_feats[name] for name in train_names if name in protein_feats
    }
    train_pocket_embeddings = {
        name: pocket_feats[name] for name in train_names if name in pocket_feats
    }
    train_binding_affs = {
        name: binding_affs[name] for name in train_names if name in binding_affs
    }

    test_inhibitor_features = {
        name: inhibitor_feats[name] for name in test_names if name in inhibitor_feats
    }
    test_protein_embeddings = {
        name: protein_feats[name] for name in test_names if name in protein_feats
    }
    test_pocket_embeddings = {
        name: pocket_feats[name] for name in test_names if name in pocket_feats
    }
    test_binding_affs = {
        name: binding_affs[name] for name in test_names if name in binding_affs
    }

    return (
        {
            "inhibitor_feats": train_inhibitor_features,
            "protein_feats": train_protein_embeddings,
            "pocket_feats": train_pocket_embeddings,
            "binding_affs": train_binding_affs,
        },
        {
            "inhibitor_feats": test_inhibitor_features,
            "protein_feats": test_protein_embeddings,
            "pocket_feats": test_pocket_embeddings,
            "binding_affs": test_binding_affs,
        },
    )


def split_data_by_group_random(
    inhibitor_feats: dict,
    protein_feats: dict,
    pocket_feats: dict,
    binding_affs: dict,
    seed: int = 42,
    split_ratio: float = 0.9,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    Deterministically split data into train and test sets using random sampling.

    Creates a random train/test split based on the specified ratio. Uses seeded random
    number generation for reproducibility.

    Parameters
    ----------
    inhibitor_feats : dict
        Dictionary mapping compound keys to inhibitor feature arrays.
    protein_feats : dict
        Dictionary mapping compound keys to protein embedding arrays.
    pocket_feats : dict
        Dictionary mapping compound keys to pocket embedding arrays.
    binding_affs : dict
        Dictionary mapping compound keys to binding affinity values.
    seed : int, optional
        Random seed for reproducibility. Default is 42.
    split_ratio : float, optional
        Fraction of data to use for training (e.g., 0.9 for 90/10 split). Default is 0.9.

    Returns
    -------
    tuple[dict[str, dict], dict[str, dict]]
        A tuple containing:
        - train_data: Dictionary with keys 'inhibitor_feats', 'protein_feats', 'pocket_feats', 'binding_affs'
        - test_data: Dictionary with keys 'inhibitor_feats', 'protein_feats', 'pocket_feats', 'binding_affs'
    """

    # Use the intersection of available keys in all dicts as valid entries
    all_keys = set(binding_affs.keys())
    valid_keys = sorted(
        all_keys & inhibitor_feats.keys() & pocket_feats.keys() & protein_feats.keys()
    )  # sorted ensures deterministic order

    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(valid_keys)

    split_index = int(len(valid_keys) * split_ratio)
    train_names = valid_keys[:split_index]
    test_names = valid_keys[split_index:]

    def subset_dict(data: dict, keys: List[str]) -> dict:
        return {k: data[k] for k in keys}

    return (
        {
            "inhibitor_feats": subset_dict(inhibitor_feats, train_names),
            "protein_feats": subset_dict(protein_feats, train_names),
            "pocket_feats": subset_dict(pocket_feats, train_names),
            "binding_affs": subset_dict(binding_affs, train_names),
        },
        {
            "inhibitor_feats": subset_dict(inhibitor_feats, test_names),
            "protein_feats": subset_dict(protein_feats, test_names),
            "pocket_feats": subset_dict(pocket_feats, test_names),
            "binding_affs": subset_dict(binding_affs, test_names),
        },
    )


def load_fold_indices(dataset: str, affinity_type: str, n_folds: int) -> Dict[int, Dict[str, List[str]]]:
    """
    Load pre-computed fold indices from spectral clustering optimization.

    Parameters
    ----------
    dataset : str
        The type of data (e.g., 'caged', 'organic').
    affinity_type : str
        The type of affinity (e.g., 'pic50', 'pk').
    n_folds : int
        The number of folds (used for file path construction).

    Returns
    -------
    dict[int, dict[str, list[str]]]
        Dictionary mapping fold index to a dict with 'train_names' and 'test_names' lists.
    """
    # Get the directory where this file is located
    current_file_dir = Path(__file__).resolve().parent
    # Navigate to the project root and then to the data directory
    data_dir = current_file_dir.parent.parent.parent / "data"

    # Load the fold indices
    fold_indices_path = (
        data_dir
        / "fold_split"
        / "spectral_clustering_analysis"
        / f"{dataset}_{affinity_type}_spectral_clustering_fold_indices.pkl"
    )

    with open(fold_indices_path, "rb") as f:
        fold_indices = pickle.load(f)

    return fold_indices
