import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
import h5py
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_descriptors_from_hdf5(
    dataset_name: str = "caged",
    affinity_type: str = "pic50",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    """
    Load MACE and protein descriptors from HDF5 file.

    Parameters
    ----------
    dataset_name : str, default='caged'
        The type of data (e.g., 'caged', 'organic').
    affinity_type : str, default='pic50'
        The type of affinity ('pic50' or 'pk').

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]
        A tuple containing:
        - MACE descriptors dictionary (compound name -> descriptor array)
        - Protein descriptors dictionary (compound name -> descriptor array)
        - Binding affinities dictionary (compound name -> affinity value)
    """
    h5_dir = Path("./../../data/hf5_dataset/caged")
    h5_path = h5_dir / f"{dataset_name}_{affinity_type}_dataset.h5"

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 dataset not found: {h5_path}")

    logger.info(f"Loading descriptors from {h5_path}")

    mace_descriptors = {}
    protein_descriptors = {}
    binding_affinities = {}

    with h5py.File(h5_path, "r") as h5_file:
        for compound_name in h5_file.keys():
            try:
                if "mol_mace_embedding" in h5_file[compound_name]:
                    mace_desc = h5_file[compound_name]["mol_mace_embedding"][...]
                    protein_desc = h5_file[compound_name]["protein_embeddings"][...]
                    target_val = h5_file[compound_name]["target_value"][()]

                    if pd.isna(target_val) or np.isnan(target_val):
                        continue

                    mace_descriptors[compound_name] = mace_desc
                    protein_descriptors[compound_name] = protein_desc
                    binding_affinities[compound_name] = float(target_val)
            except Exception as e:
                logger.warning(f"Error loading data for {compound_name}: {e}")
                continue

    logger.info(f"Loaded {len(mace_descriptors)} compounds from {h5_path.name}")
    return mace_descriptors, protein_descriptors, binding_affinities


def aggregate_descriptors_by_identifier(
    mace_descriptors: Dict[str, np.ndarray],
    protein_descriptors: Dict[str, np.ndarray],
    binding_affinities: Dict[str, float],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float], List[str]]:
    """
    Aggregate descriptors by compound identifier.

    Parameters
    ----------
    mace_descriptors : Dict[str, np.ndarray]
        MACE descriptors dictionary (compound name -> descriptor array).
    protein_descriptors : Dict[str, np.ndarray]
        Protein descriptors dictionary (compound name -> descriptor array).
    binding_affinities : Dict[str, float]
        Binding affinities dictionary (compound name -> affinity value).

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float], List[str]]
        A tuple containing:
        - Aggregated MACE descriptors
        - Aggregated protein descriptors
        - Aggregated binding affinities
        - List of unique compound identifiers
    """
    aggregated_mace = {}
    aggregated_protein = {}
    aggregated_affinities = {}

    # Create a map from unique identifier to all its full compound names
    identifier_map = {}
    for name in mace_descriptors.keys():
        parts = name.split("_", 2)
        if len(parts) >= 2:
            compound_identifier = f"{parts[0]}_{parts[1]}"
            base_identifier = compound_identifier

            if len(parts) > 2:
                protein_part = parts[2]
                # Check for and remove replicate markers (e.g., 'A_', 'B_')
                if (
                    len(protein_part) > 1
                    and protein_part[1] == "_"
                    and "A" <= protein_part[0] <= "Z"
                ):
                    protein_part = protein_part[2:]
                base_identifier = f"{compound_identifier}_{protein_part}"

            if base_identifier not in identifier_map:
                identifier_map[base_identifier] = []
            identifier_map[base_identifier].append(name)

    # Aggregate the data by averaging
    for identifier, full_names in identifier_map.items():
        # Average MACE descriptors over atoms, then over occurrences
        all_descriptors_for_id = [
            np.mean(mace_descriptors[name], axis=0) for name in full_names
        ]
        aggregated_mace[identifier] = np.mean(all_descriptors_for_id, axis=0)

        # Protein: Take the <CLS> token (first element) from each, then average
        all_protein_for_id = [protein_descriptors[name][0, 0] for name in full_names]
        aggregated_protein[identifier] = np.mean(all_protein_for_id, axis=0)

        # Average binding affinities
        all_affinities_for_id = [binding_affinities[name] for name in full_names]
        aggregated_affinities[identifier] = np.mean(all_affinities_for_id)

    unique_identifiers = sorted(list(aggregated_mace.keys()))

    return (
        aggregated_mace,
        aggregated_protein,
        aggregated_affinities,
        unique_identifiers,
    )


def apply_pca_and_concatenate(
    mace_descriptors: Dict[str, np.ndarray],
    protein_descriptors: Dict[str, np.ndarray],
    compound_names: List[str],
    n_components: int = 200,
) -> np.ndarray:
    """
    Apply PCA to reduce both descriptor types to n_components and concatenate.

    Parameters
    ----------
    mace_descriptors : Dict[str, np.ndarray]
        MACE descriptors dictionary.
    protein_descriptors : Dict[str, np.ndarray]
        Protein descriptors dictionary.
    compound_names : List[str]
        List of compound names.
    n_components : int, default=200
        Number of PCA components to use for each descriptor type.

    Returns
    -------
    np.ndarray
        Combined feature array with PCA-reduced descriptors concatenated.
        Returns None if there are issues with the input data.
    """

    # Convert to arrays
    mace_array = np.array([mace_descriptors[name] for name in compound_names])
    protein_array = np.array([protein_descriptors[name] for name in compound_names])

    logger.info(f"Original MACE shape: {mace_array.shape}")
    logger.info(f"Original protein shape: {protein_array.shape}")

    # Check for NaN or infinite values in input data
    if np.isnan(mace_array).any() or np.isinf(mace_array).any():
        logger.error("MACE array contains NaN or infinite values")
        nan_indices = np.where(np.isnan(mace_array).any(axis=1))[0]
        inf_indices = np.where(np.isinf(mace_array).any(axis=1))[0]
        logger.error(f"NaN indices: {nan_indices}")
        logger.error(f"Inf indices: {inf_indices}")
        return None

    if np.isnan(protein_array).any() or np.isinf(protein_array).any():
        logger.error("Protein array contains NaN or infinite values")
        nan_indices = np.where(np.isnan(protein_array).any(axis=1))[0]
        inf_indices = np.where(np.isinf(protein_array).any(axis=1))[0]
        logger.error(f"NaN indices: {nan_indices}")
        logger.error(f"Inf indices: {inf_indices}")
        return None

    # Check for zero variance features before standardization
    mace_var = np.var(mace_array, axis=0)
    protein_var = np.var(protein_array, axis=0)

    zero_var_mace = np.sum(mace_var == 0)
    zero_var_protein = np.sum(protein_var == 0)

    if zero_var_mace > 0:
        logger.warning(f"Found {zero_var_mace} zero-variance MACE features")
    if zero_var_protein > 0:
        logger.warning(f"Found {zero_var_protein} zero-variance protein features")

    # Standardize features
    scaler_mace = StandardScaler()
    scaler_protein = StandardScaler()

    mace_scaled = scaler_mace.fit_transform(mace_array)
    protein_scaled = scaler_protein.fit_transform(protein_array)

    # Check for NaN after standardization
    if np.isnan(mace_scaled).any():
        logger.error("MACE scaled array contains NaN values after standardization")
        return None
    if np.isnan(protein_scaled).any():
        logger.error("Protein scaled array contains NaN values after standardization")
        return None

    # Apply PCA with appropriate number of components
    # Make sure n_components doesn't exceed the number of samples
    n_samples = mace_array.shape[0]
    n_components_actual = min(n_components, n_samples - 1, mace_array.shape[1])

    logger.info(
        f"Using {n_components_actual} PCA components (requested: {n_components})"
    )

    pca_mace = PCA(n_components=n_components_actual, random_state=42)
    pca_protein = PCA(n_components=n_components_actual, random_state=42)

    mace_reduced = pca_mace.fit_transform(mace_scaled)
    protein_reduced = pca_protein.fit_transform(protein_scaled)

    logger.info(
        f"MACE PCA explained variance ratio: {pca_mace.explained_variance_ratio_.sum():.3f}"
    )
    logger.info(
        f"Protein PCA explained variance ratio: {pca_protein.explained_variance_ratio_.sum():.3f}"
    )

    # Check for NaN after PCA
    if np.isnan(mace_reduced).any():
        logger.error("MACE reduced array contains NaN values after PCA")
        return None
    if np.isnan(protein_reduced).any():
        logger.error("Protein reduced array contains NaN values after PCA")
        return None

    # Concatenate
    combined_features = np.concatenate([mace_reduced, protein_reduced], axis=1)
    logger.info(f"Combined feature shape: {combined_features.shape}")

    # Final check for NaN
    if np.isnan(combined_features).any():
        logger.error("Combined features contain NaN values")
        return None

    return combined_features


def perform_spectral_clustering(
    features: np.ndarray, n_clusters: int = 4, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform spectral clustering using cosine similarity.

    Parameters
    ----------
    features : np.ndarray
        Feature array to cluster.
    n_clusters : int, default=4
        Number of clusters to create.
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - Cluster labels for each sample
        - Cosine similarity matrix
        Returns (None, None) if clustering fails.
    """

    if features is None:
        logger.error("Features array is None")
        return None, None

    # Check for NaN or infinite values in features
    if np.isnan(features).any():
        logger.error("Features contain NaN values")
        return None, None
    if np.isinf(features).any():
        logger.error("Features contain infinite values")
        return None, None

    # Check for zero-norm vectors
    norms = np.linalg.norm(features, axis=1)
    zero_norm_indices = np.where(norms == 0)[0]
    if len(zero_norm_indices) > 0:
        logger.error(
            f"Found {len(zero_norm_indices)} zero-norm feature vectors at indices: {zero_norm_indices}"
        )
        return None, None

    # Calculate cosine similarity matrix
    original_similarity_matrix = cosine_similarity(features)
    logger.info(f"Original similarity matrix shape: {original_similarity_matrix.shape}")
    logger.info(
        f"Original similarity matrix range: [{original_similarity_matrix.min():.3f}, {original_similarity_matrix.max():.3f}]"
    )

    # Check for NaN in similarity matrix
    if np.isnan(original_similarity_matrix).any():
        logger.error("Similarity matrix contains NaN values")
        nan_count = np.isnan(original_similarity_matrix).sum()
        logger.error(f"Number of NaN values: {nan_count}")
        return None, None

    # Transform cosine similarity to make it suitable for spectral clustering
    # Convert cosine similarity to cosine distance, then to RBF similarity
    cosine_distance = 1 - original_similarity_matrix
    # Use RBF kernel: exp(-gamma * distance^2)
    gamma = 1.0  # You can tune this parameter
    similarity_matrix = np.exp(-gamma * cosine_distance**2)

    logger.info(
        f"Transformed similarity matrix range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]"
    )

    # Ensure diagonal elements are 1 (self-similarity)
    np.fill_diagonal(similarity_matrix, 1.0)

    # Add small epsilon for numerical stability
    epsilon = 1e-10
    similarity_matrix = similarity_matrix + epsilon

    # Apply spectral clustering
    try:
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=random_state,
            n_init=10,
            assign_labels="discretize",  # Use discretize instead of kmeans for stability
        )

        cluster_labels = spectral.fit_predict(similarity_matrix)
        logger.info(
            f"Spectral clustering complete. Cluster distribution: {np.bincount(cluster_labels)}"
        )

        # Return both cluster labels and the ORIGINAL cosine similarity matrix for visualization
        return cluster_labels, original_similarity_matrix

    except Exception as e:
        logger.error(f"Spectral clustering failed: {e}")

        # Fallback: Try with different parameters
        logger.info("Trying spectral clustering with different parameters...")
        try:
            spectral_fallback = SpectralClustering(
                n_clusters=n_clusters,
                affinity="precomputed",
                random_state=random_state,
                n_init=5,
                assign_labels="kmeans",
                eigen_solver="arpack",
            )

            cluster_labels = spectral_fallback.fit_predict(similarity_matrix)
            logger.info(
                f"Fallback spectral clustering complete. Cluster distribution: {np.bincount(cluster_labels)}"
            )
            return cluster_labels, original_similarity_matrix

        except Exception as e2:
            logger.error(f"Fallback spectral clustering also failed: {e2}")

            # Final fallback: Use KMeans clustering directly on features
            logger.info("Using KMeans as final fallback...")
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            logger.info(
                f"KMeans clustering complete. Cluster distribution: {np.bincount(cluster_labels)}"
            )
            return cluster_labels, original_similarity_matrix


def create_cross_validation_folds(
    compound_names: List[str],
    cluster_labels: np.ndarray,
    binding_affinities: np.ndarray,
    n_folds: int = 4,
) -> List[Tuple[List[int], List[int]]]:
    """
    Create cross-validation folds where each fold uses one cluster as test set.

    Parameters
    ----------
    compound_names : List[str]
        List of compound names.
    cluster_labels : np.ndarray
        Cluster labels for each compound.
    binding_affinities : np.ndarray
        Binding affinity values for each compound.
    n_folds : int, default=4
        Number of folds to create.

    Returns
    -------
    List[Tuple[List[int], List[int]]]
        List of (train_indices, test_indices) tuples for each fold.
    """

    folds = []
    unique_clusters = np.unique(cluster_labels)

    logger.info("Creating cross-validation folds:")

    for test_cluster in unique_clusters:
        test_indices = np.where(cluster_labels == test_cluster)[0].tolist()
        train_indices = np.where(cluster_labels != test_cluster)[0].tolist()

        folds.append((train_indices, test_indices))

        # Log fold statistics
        test_affinities = binding_affinities[test_indices]
        train_affinities = binding_affinities[train_indices]

        logger.info(f"Fold {test_cluster + 1}:")
        logger.info(
            f"  Train: {len(train_indices)} samples, Test: {len(test_indices)} samples"
        )
        logger.info(
            f"  Train affinity range: [{train_affinities.min():.2f}, {train_affinities.max():.2f}]"
        )
        logger.info(
            f"  Test affinity range: [{test_affinities.min():.2f}, {test_affinities.max():.2f}]"
        )

    return folds


def analyze_fold_distributions(
    folds: List[Tuple[List[int], List[int]]],
    binding_affinities: np.ndarray,
    compound_names: List[str],
    affinity_type: str,
    thresholds: List[float] = [5.0, 6.0, 7.0],
) -> pd.DataFrame:
    """
    Analyze the distribution of good/bad compounds across folds for different thresholds.

    Parameters
    ----------
    folds : List[Tuple[List[int], List[int]]]
        List of (train_indices, test_indices) tuples for each fold.
    binding_affinities : np.ndarray
        Binding affinity values for each compound.
    compound_names : List[str]
        List of compound names.
    affinity_type : str
        Type of affinity ('pic50' or 'pk').
    thresholds : List[float], default=[5.0, 6.0, 7.0]
        Activity thresholds to analyze.

    Returns
    -------
    pd.DataFrame
        DataFrame containing fold distribution analysis results.
    """

    results = []

    for threshold in thresholds:
        logger.info(
            f"\n=== THRESHOLD ANALYSIS: {threshold} ({affinity_type.upper()}) ==="
        )

        fold_stats = []
        total_good = 0
        total_bad = 0

        for fold_idx, (train_indices, test_indices) in enumerate(folds):
            test_affinities = binding_affinities[test_indices]

            # Count good (>= threshold) and bad (< threshold) compounds
            good_count = np.sum(test_affinities >= threshold)
            bad_count = np.sum(test_affinities < threshold)
            total_count = len(test_indices)

            fold_stats.append(
                {
                    "Fold": fold_idx + 1,
                    "Total": total_count,
                    "Bad": bad_count,
                    "Good": good_count,
                    "Threshold": threshold,
                    "Affinity_Type": affinity_type.upper(),
                }
            )

            total_good += good_count
            total_bad += bad_count

            logger.info(
                f"Fold {fold_idx + 1}: Total={total_count}, Bad={bad_count}, Good={good_count}"
            )

        # Add total row
        fold_stats.append(
            {
                "Fold": "Total",
                "Total": total_good + total_bad,
                "Bad": total_bad,
                "Good": total_good,
                "Threshold": threshold,
                "Affinity_Type": affinity_type.upper(),
            }
        )

        logger.info(
            f"Total: Total={total_good + total_bad}, Bad={total_bad}, Good={total_good}"
        )

        results.extend(fold_stats)

    return pd.DataFrame(results)


def save_fold_analysis(
    results_pic50: pd.DataFrame, results_pki: pd.DataFrame, output_dir: str
) -> None:
    """
    Save fold analysis results to CSV files.

    Parameters
    ----------
    results_pic50 : pd.DataFrame
        Fold analysis results for pIC50 data.
    results_pki : pd.DataFrame
        Fold analysis results for pKi data.
    output_dir : str
        Directory to save the analysis results.

    Returns
    -------
    None
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save individual results
    results_pic50.to_csv(output_path / "fold_analysis_pic50.csv", index=False)
    results_pki.to_csv(output_path / "fold_analysis_pki.csv", index=False)

    # Create combined summary table for each threshold
    thresholds = results_pic50["Threshold"].unique()

    for threshold in thresholds:
        pic50_data = results_pic50[results_pic50["Threshold"] == threshold]
        pki_data = results_pki[results_pki["Threshold"] == threshold]

        # Create combined table
        combined_data = []
        for _, pic50_row in pic50_data.iterrows():
            fold = pic50_row["Fold"]
            pki_row = pki_data[pki_data["Fold"] == fold].iloc[0]

            combined_data.append(
                {
                    "Fold": fold,
                    "pIC50_Total": pic50_row["Total"],
                    "pIC50_Bad": pic50_row["Bad"],
                    "pIC50_Good": pic50_row["Good"],
                    "pKi_Total": pki_row["Total"],
                    "pKi_Bad": pki_row["Bad"],
                    "pKi_Good": pki_row["Good"],
                }
            )

        combined_df = pd.DataFrame(combined_data)
        combined_df.to_csv(
            output_path / f"combined_fold_analysis_threshold_{threshold}.csv",
            index=False,
        )

        # Print formatted table
        logger.info(f"\n=== COMBINED FOLD ANALYSIS (Threshold = {threshold}) ===")
        logger.info("Fold  pIC50        pKi")
        logger.info("      Total Bad Good  Total Bad Good")
        for _, row in combined_df.iterrows():
            # Convert fold to string and handle both int and string cases
            fold_str = str(row["Fold"])
            logger.info(
                f"{fold_str:>4s}  {row['pIC50_Total']:5d} {row['pIC50_Bad']:3d} {row['pIC50_Good']:4d}  {row['pKi_Total']:5d} {row['pKi_Bad']:3d} {row['pKi_Good']:4d}"
            )


def save_fold_indices(
    folds: List[Tuple[List[int], List[int]]],
    compound_names: List[str],
    output_dir: str,
    dataset_name: str,
    affinity_type: str,
    n_clusters: int,
) -> Dict[int, Dict[str, List[str]]]:
    """
    Save fold indices in the same format as optimize_stratified_split.py.

    Parameters
    ----------
    folds : List[Tuple[List[int], List[int]]]
        List of (train_indices, test_indices) tuples for each fold.
    compound_names : List[str]
        List of compound names.
    output_dir : str
        Directory to save the fold indices.
    dataset_name : str
        Name of the dataset.
    affinity_type : str
        Type of affinity ('pic50' or 'pk').
    n_clusters : int
        Number of clusters used.

    Returns
    -------
    Dict[int, Dict[str, List[str]]]
        Dictionary mapping fold index to train/test compound names.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create fold indices with compound names
    fold_indices = {}
    for fold_idx, (train_indices, test_indices) in enumerate(folds):
        train_names = [compound_names[i] for i in train_indices]
        test_names = [compound_names[i] for i in test_indices]

        fold_indices[fold_idx] = {
            "train_names": train_names,
            "test_names": test_names,
        }

    # Save fold indices
    fold_indices_file = (
        output_path
        / f"{dataset_name}_{affinity_type}_spectral_clustering_fold_indices.pkl"
    )

    with open(fold_indices_file, "wb") as f:
        pickle.dump(fold_indices, f)

    logger.info(f"Saved fold indices to {fold_indices_file}")

    return fold_indices


def save_optimization_results(
    fold_indices_pic50: Dict[int, Dict[str, List[str]]],
    fold_indices_pki: Dict[int, Dict[str, List[str]]],
    cluster_labels_pic50: np.ndarray,
    cluster_labels_pki: np.ndarray,
    similarity_matrix_pic50: np.ndarray,
    similarity_matrix_pki: np.ndarray,
    comp_names_pic50: List[str],
    comp_names_pki: List[str],
    binding_affinities_pic50: Dict[str, float],
    binding_affinities_pki: Dict[str, float],
    combined_features_pic50: np.ndarray,
    combined_features_pki: np.ndarray,
    output_dir: str,
    dataset_name: str,
    args: argparse.Namespace,
) -> None:
    """
    Save complete optimization results in the same format as optimize_stratified_split.py.

    Parameters
    ----------
    fold_indices_pic50 : Dict[int, Dict[str, List[str]]]
        Fold indices for pIC50 data.
    fold_indices_pki : Dict[int, Dict[str, List[str]]]
        Fold indices for pKi data.
    cluster_labels_pic50 : np.ndarray
        Cluster labels for pIC50 data.
    cluster_labels_pki : np.ndarray
        Cluster labels for pKi data.
    similarity_matrix_pic50 : np.ndarray
        Similarity matrix for pIC50 data.
    similarity_matrix_pki : np.ndarray
        Similarity matrix for pKi data.
    comp_names_pic50 : List[str]
        Compound names for pIC50 data.
    comp_names_pki : List[str]
        Compound names for pKi data.
    binding_affinities_pic50 : Dict[str, float]
        Binding affinities for pIC50 data.
    binding_affinities_pki : Dict[str, float]
        Binding affinities for pKi data.
    combined_features_pic50 : np.ndarray
        Combined features for pIC50 data.
    combined_features_pki : np.ndarray
        Combined features for pKi data.
    output_dir : str
        Directory to save the optimization results.
    dataset_name : str
        Name of the dataset.
    args : argparse.Namespace
        Command-line arguments.

    Returns
    -------
    None
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert binding affinities to arrays in the same order as comp_names
    targets_pic50 = [binding_affinities_pic50[name] for name in comp_names_pic50]
    targets_pki = [binding_affinities_pki[name] for name in comp_names_pki]

    # Create results for pIC50
    results_pic50 = {
        "data_type": dataset_name,
        "affinity_type": "pic50",
        "descriptor_types": ["inhibitor", "protein"],  # MACE + protein
        "combination_method": "pca_concatenate",
        "use_pca": True,
        "pca_components": args.pca_components,
        "n_folds": args.n_clusters,
        "fold_indices": fold_indices_pic50,
        "cluster_labels": cluster_labels_pic50,
        "similarity_matrix": similarity_matrix_pic50,  # Add similarity matrix
        "keys": comp_names_pic50,
        "targets": targets_pic50,  # Add targets
        "features_shape": combined_features_pic50.shape,
        "random_state": 42,
        "clustering_method": "spectral",
        "n_clusters": args.n_clusters,
    }

    # Create results for pKi
    results_pki = {
        "data_type": dataset_name,
        "affinity_type": "pk",
        "descriptor_types": ["inhibitor", "protein"],  # MACE + protein
        "combination_method": "pca_concatenate",
        "use_pca": True,
        "pca_components": args.pca_components,
        "n_folds": args.n_clusters,
        "fold_indices": fold_indices_pki,
        "cluster_labels": cluster_labels_pki,
        "similarity_matrix": similarity_matrix_pki,  # Add similarity matrix
        "keys": comp_names_pki,
        "targets": targets_pki,  # Add targets
        "features_shape": combined_features_pki.shape,
        "random_state": 42,
        "clustering_method": "spectral",
        "n_clusters": args.n_clusters,
    }

    # Save optimization results
    results_file_pic50 = (
        output_path
        / f"{dataset_name}_pic50_spectral_clustering_optimization_results.pkl"
    )
    results_file_pki = (
        output_path / f"{dataset_name}_pk_spectral_clustering_optimization_results.pkl"
    )

    with open(results_file_pic50, "wb") as f:
        pickle.dump(results_pic50, f)
    logger.info(f"Saved pIC50 optimization results to {results_file_pic50}")

    with open(results_file_pki, "wb") as f:
        pickle.dump(results_pki, f)
    logger.info(f"Saved pKi optimization results to {results_file_pki}")


def main() -> None:
    """
    Main function to perform spectral clustering for dataset splitting.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Perform spectral clustering for dataset splitting."
    )
    parser.add_argument("--dataset", default="caged", help="Dataset name.")
    parser.add_argument(
        "--output_dir",
        default="./spectral_clustering_analysis",
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--pca_components",
        default=200,
        type=int,
        help="Number of PCA components for each descriptor type.",
    )
    parser.add_argument(
        "--n_clusters",
        default=5,
        type=int,
        help="Number of clusters for spectral clustering.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[5.0, 6.0, 7.0],
        help="Activity thresholds to analyze.",
    )

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load both datasets
    logger.info("Loading pIC50 dataset...")
    mace_desc_pic50, protein_desc_pic50, binding_affinities_pic50 = (
        load_descriptors_from_hdf5(dataset_name=args.dataset, affinity_type="pic50")
    )

    logger.info("Loading pKi dataset...")
    mace_desc_pki, protein_desc_pki, binding_affinities_pki = (
        load_descriptors_from_hdf5(dataset_name=args.dataset, affinity_type="pk")
    )

    # 2. Aggregate data by compound identifier
    logger.info("Aggregating pIC50 data...")
    agg_mace_pic50, agg_protein_pic50, agg_aff_pic50, comp_names_pic50 = (
        aggregate_descriptors_by_identifier(
            mace_desc_pic50, protein_desc_pic50, binding_affinities_pic50
        )
    )

    logger.info("Aggregating pKi data...")
    agg_mace_pki, agg_protein_pki, agg_aff_pki, comp_names_pki = (
        aggregate_descriptors_by_identifier(
            mace_desc_pki, protein_desc_pki, binding_affinities_pki
        )
    )

    logger.info(
        f"After aggregation: {len(comp_names_pic50)} unique compounds for pIC50, {len(comp_names_pki)} for pKi"
    )

    # 3. Apply PCA and concatenate features
    logger.info("Applying PCA and concatenating features for pIC50...")
    combined_features_pic50 = apply_pca_and_concatenate(
        agg_mace_pic50,
        agg_protein_pic50,
        comp_names_pic50,
        n_components=args.pca_components,
    )

    if combined_features_pic50 is None:
        logger.error("Failed to process pIC50 features. Exiting.")
        return

    logger.info("Applying PCA and concatenating features for pKi...")
    combined_features_pki = apply_pca_and_concatenate(
        agg_mace_pki, agg_protein_pki, comp_names_pki, n_components=args.pca_components
    )

    if combined_features_pki is None:
        logger.error("Failed to process pKi features. Exiting.")
        return

    # 4. Perform spectral clustering
    logger.info("Performing spectral clustering for pIC50...")
    cluster_labels_pic50, similarity_matrix_pic50 = perform_spectral_clustering(
        combined_features_pic50, n_clusters=args.n_clusters
    )

    if cluster_labels_pic50 is None:
        logger.error("Failed to cluster pIC50 data. Exiting.")
        return

    logger.info("Performing spectral clustering for pKi...")
    cluster_labels_pki, similarity_matrix_pki = perform_spectral_clustering(
        combined_features_pki, n_clusters=args.n_clusters
    )

    if cluster_labels_pki is None:
        logger.error("Failed to cluster pKi data. Exiting.")
        return

    # 5. Create cross-validation folds
    binding_array_pic50 = np.array([agg_aff_pic50[name] for name in comp_names_pic50])
    binding_array_pki = np.array([agg_aff_pki[name] for name in comp_names_pki])

    logger.info("Creating cross-validation folds for pIC50...")
    folds_pic50 = create_cross_validation_folds(
        comp_names_pic50,
        cluster_labels_pic50,
        binding_array_pic50,
        n_folds=args.n_clusters,
    )

    logger.info("Creating cross-validation folds for pKi...")
    folds_pki = create_cross_validation_folds(
        comp_names_pki, cluster_labels_pki, binding_array_pki, n_folds=args.n_clusters
    )

    # 6. Save fold indices (NEW - following optimize_stratified_split.py structure)
    logger.info("Saving fold indices...")
    fold_indices_pic50 = save_fold_indices(
        folds_pic50,
        comp_names_pic50,
        args.output_dir,
        args.dataset,
        "pic50",
        args.n_clusters,
    )
    fold_indices_pki = save_fold_indices(
        folds_pki, comp_names_pki, args.output_dir, args.dataset, "pk", args.n_clusters
    )

    # 7. Save complete optimization results (NEW - following optimize_stratified_split.py structure)
    logger.info("Saving optimization results...")
    save_optimization_results(
        fold_indices_pic50,
        fold_indices_pki,
        cluster_labels_pic50,
        cluster_labels_pki,
        similarity_matrix_pic50,
        similarity_matrix_pki,  # Add similarity matrices
        comp_names_pic50,
        comp_names_pki,
        agg_aff_pic50,
        agg_aff_pki,  # Add binding affinities
        combined_features_pic50,
        combined_features_pki,
        args.output_dir,
        args.dataset,
        args,
    )

    # 8. Analyze fold distributions for different thresholds
    logger.info("Analyzing fold distributions...")
    results_pic50 = analyze_fold_distributions(
        folds_pic50,
        binding_array_pic50,
        comp_names_pic50,
        "pic50",
        thresholds=args.thresholds,
    )

    results_pki = analyze_fold_distributions(
        folds_pki, binding_array_pki, comp_names_pki, "pki", thresholds=args.thresholds
    )

    # 9. Save fold analysis results
    save_fold_analysis(results_pic50, results_pki, args.output_dir)

    logger.info(f"Analysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
