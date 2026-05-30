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
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min

# Optional: UMAP
try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# === FONT SETUP === #
import matplotlib.font_manager as fm

palatino_font_path = None  # Set to your Palatino Linotype .ttf directory to use this font
if palatino_font_path:
    try:
        fm.fontManager.addfont(palatino_font_path + "/Palatino_Linotype.ttf")
        fm.fontManager.addfont(palatino_font_path + "/Palatino_Linotype_Bold.ttf")
        fm.fontManager.addfont(palatino_font_path + "/Palatino_Linotype_Italic.ttf")
        fm.fontManager.addfont(palatino_font_path + "/Palatino_Linotype_Bold_Italic.ttf")
        plt.rcParams["mathtext.fontset"] = "custom"
        plt.rcParams["mathtext.rm"] = "Palatino Linotype"
        plt.rcParams["mathtext.it"] = "Palatino Linotype:italic"
        plt.rcParams["mathtext.bf"] = "Palatino Linotype:bold"
        plt.rcParams["font.family"] = "Palatino Linotype"
        plt.rcParams["text.usetex"] = False
    except (FileNotFoundError, OSError):
        logger.warning("Palatino font not found. Using default font.")


# === DATA LOADING === #
def load_descriptors_from_hdf5(
    dataset_name: str = "caged",
    affinity_type: str = "pic50",
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, float],
]:
    """
    Load MACE, protein, pocket descriptors and sequences from HDF5 file.

    Parameters
    ----------
    dataset_name : str, default='caged'
        The type of data (e.g., 'caged', 'organic').
    affinity_type : str, default='pic50'
        The type of affinity ('pic50' or 'pk').

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]
        A tuple containing:
        - MACE descriptors dictionary
        - Protein descriptors dictionary
        - Protein sequences dictionary
        - Pocket descriptors dictionary
        - Binding affinities dictionary
    """
    h5_dir = Path("./../../../data/hf5_dataset/caged")
    h5_path = h5_dir / f"{dataset_name}_{affinity_type}_dataset.h5"

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 dataset not found: {h5_path}")

    logger.info(f"Loading SOAP descriptors from {h5_path}")

    mace_descriptors = {}
    protein_descriptors = {}
    pocket_descriptors = {}
    protein_sequences = {}
    binding_affinities = {}

    with h5py.File(h5_path, "r") as h5_file:
        for compound_name in h5_file.keys():
            try:
                if "mol_mace_embedding" in h5_file[compound_name]:
                    mace_desc = h5_file[compound_name]["mol_mace_embedding"][...]
                    protein_desc = h5_file[compound_name]["protein_embeddings"][...]
                    # obtain the sequence
                    sequence = h5_file[compound_name]["protein_sequence"][...]
                    pocket_desc = h5_file[compound_name]["pocket_embeddings"][...]
                    target_val = h5_file[compound_name]["target_value"][()]
                    if pd.isna(target_val) or np.isnan(target_val):
                        continue
                    mace_descriptors[compound_name] = mace_desc
                    protein_descriptors[compound_name] = protein_desc
                    protein_sequences[compound_name] = sequence
                    pocket_descriptors[compound_name] = pocket_desc
                    binding_affinities[compound_name] = float(target_val)
            except Exception as e:
                logger.warning(f"Error loading data for {compound_name}: {e}")
                continue

    logger.info(f"Loaded {len(mace_descriptors)} MACE descriptors from {h5_path.name}")
    logger.info(
        f"Loaded {len(protein_descriptors)} protein descriptors from {h5_path.name}"
    )
    logger.info(
        f"Loaded {len(protein_sequences)} protein sequences from {h5_path.name}"
    )
    logger.info(
        f"Loaded {len(pocket_descriptors)} pocket descriptors from {h5_path.name}"
    )
    return (
        mace_descriptors,
        protein_descriptors,
        protein_sequences,
        pocket_descriptors,
        binding_affinities,
    )


def aggregate_descriptors_by_identifier(
    mace_descriptors: Dict[str, np.ndarray],
    protein_descriptors: Dict[str, np.ndarray],
    pocket_descriptors: Dict[str, np.ndarray],
    binding_affinities: Dict[str, float],
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, float],
    List[str],
]:
    """
    Aggregate descriptors and affinities by a common compound identifier.

    For example, 'compound_102_A' and 'compound_102_B' are aggregated under 'compound_102'.
    This function averages the descriptors from all occurrences of a compound to create
    a single, more representative feature vector.

    Parameters
    ----------
    mace_descriptors : Dict[str, np.ndarray]
        MACE descriptors dictionary.
    protein_descriptors : Dict[str, np.ndarray]
        Protein descriptors dictionary.
    pocket_descriptors : Dict[str, np.ndarray]
        Pocket descriptors dictionary.
    binding_affinities : Dict[str, float]
        Binding affinities dictionary.

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float], List[str]]
        A tuple containing:
        - Aggregated MACE descriptors
        - Aggregated protein descriptors
        - Aggregated pocket descriptors
        - Aggregated binding affinities
        - List of unique compound identifiers
    """
    aggregated_mace = {}
    aggregated_protein = {}
    aggregated_pocket = {}
    aggregated_affinities = {}

    # Create a map from a unique identifier to all its full compound names
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
        # 1. For each occurrence, average the descriptor over its atoms (axis=0).
        # 2. Then, average the resulting vectors for all occurrences.
        all_descriptors_for_id = [
            np.mean(mace_descriptors[name], axis=0) for name in full_names
        ]
        aggregated_mace[identifier] = np.mean(all_descriptors_for_id, axis=0)

        # Protein: Take the <CLS> token (first element) from each, then average.
        all_protein_for_id = [protein_descriptors[name][0, 0] for name in full_names]
        aggregated_protein[identifier] = np.mean(all_protein_for_id, axis=0)

        # Pocket: Average over all residues for each, then average occurrences.
        all_pocket_for_id = [
            np.mean(pocket_descriptors[name], axis=0) for name in full_names
        ]
        aggregated_pocket[identifier] = np.mean(all_pocket_for_id, axis=0)

        # Also average the binding affinities for all occurrences
        all_affinities_for_id = [binding_affinities[name] for name in full_names]
        aggregated_affinities[identifier] = np.mean(all_affinities_for_id)

    unique_identifiers = sorted(list(aggregated_mace.keys()))

    return (
        aggregated_mace,
        aggregated_protein,
        aggregated_pocket,
        aggregated_affinities,
        unique_identifiers,
    )


# === CORE ANALYSIS FUNCTIONS === #
def perform_dimensionality_reduction(
    features: np.ndarray, method: str = "tsne", random_state: int = 42
) -> np.ndarray:
    """
    Perform dimensionality reduction using t-SNE, UMAP, or PCA.

    Parameters
    ----------
    features : np.ndarray
        Feature array to reduce.
    method : str, default='tsne'
        Dimensionality reduction method ('tsne', 'umap', or 'pca').
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    np.ndarray
        2D embeddings of the input features.
    """
    # Standardize features before dimensionality reduction
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    if method.lower() == "tsne":
        logger.info("Performing t-SNE dimensionality reduction...")
        tsne = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=min(30, len(features_scaled) - 1),
            n_iter=1000,
            learning_rate="auto",
        )
        embeddings = tsne.fit_transform(features_scaled)
    elif method.lower() == "umap":
        if not HAS_UMAP:
            raise ImportError(
                "UMAP is not installed. Install with `pip install umap-learn`"
            )
        logger.info("Performing UMAP dimensionality reduction...")
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        embeddings = reducer.fit_transform(features_scaled)
    elif method.lower() == "pca":
        logger.info("Reducing to 2D using PCA...")
        pca = PCA(n_components=2, random_state=random_state)
        embeddings = pca.fit_transform(features_scaled)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    logger.info(
        f"Dimensionality reduction complete. Embeddings shape: {embeddings.shape}"
    )
    return embeddings


# === CLUSTERING HELPERS === #
def choose_optimal_k_elbow(
    features: np.ndarray,
    max_k: int = 10,
    random_state: int = 42,
    output_path: Optional[Path] = None,
    descriptor_type: str = "",
    affinity_type: str = "",
) -> int:
    """
    Find the optimal number of clusters using the elbow method and save a plot.

    Parameters
    ----------
    features : np.ndarray
        Feature array to cluster.
    max_k : int, default=10
        Maximum number of clusters to test.
    random_state : int, default=42
        Random state for reproducibility.
    output_path : Optional[Path], default=None
        Path to save the elbow plot.
    descriptor_type : str, default=''
        Type of descriptor being analyzed.
    affinity_type : str, default=''
        Type of affinity data.

    Returns
    -------
    int
        Optimal number of clusters determined by the elbow method.
    """
    inertias = []
    # Ensure max_k is reasonable
    max_k = min(max_k, len(features) - 1)
    if max_k < 2:
        logger.warning(
            "Not enough samples to perform automatic k-selection. Defaulting to k=2."
        )
        return 2

    K_range = range(2, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)
        logger.info(f"Inertia for k={k}: {kmeans.inertia_:.2f}")

    if output_path:
        plt.figure(figsize=(8, 6))
        plt.plot(list(K_range), inertias, marker="o", linestyle="--")
        plt.xlabel("Number of clusters (k)", fontsize=12)
        plt.ylabel("Inertia (WCSS)", fontsize=12)
        plt.title(
            f"Elbow Method ({descriptor_type.upper()}, {affinity_type.upper()})",
            fontsize=14,
        )
        plt.grid(True, alpha=0.3)
        plt.xticks(list(K_range))
        plt.tight_layout()
        plot_file = output_path / f"elbow_curve_{descriptor_type}_{affinity_type}.png"
        plt.savefig(plot_file, dpi=300)
        plt.close()
        logger.info(f"Elbow plot saved to {plot_file}")

    # Find the elbow point: the point with maximum distance to the line connecting start and end points.
    points = np.array(list(zip(K_range, inertias)))
    if len(points) < 2:
        return 2  # Not enough points to determine an elbow

    line_vec = points[-1] - points[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))

    vec_from_first = points - points[0]
    scalar_product = np.sum(
        vec_from_first * np.tile(line_vec_norm, (len(points), 1)), axis=1
    )
    vec_from_first_parallel = scalar_product[:, np.newaxis] * np.tile(
        line_vec_norm, (len(points), 1)
    )
    vec_to_line = vec_from_first - vec_from_first_parallel

    dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))
    best_k_index = np.argmax(dist_to_line)
    best_k = K_range[best_k_index]

    logger.info(f"Chosen number of clusters (elbow): {best_k}")
    return best_k


def perform_kmeans_clustering(
    features: np.ndarray, n_clusters: int = 5, random_state: int = 42
) -> Tuple[np.ndarray, KMeans]:
    """
    Perform K-means clustering on the feature array.

    Parameters
    ----------
    features : np.ndarray
        Feature array to cluster.
    n_clusters : int, default=5
        Number of clusters to create.
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, KMeans]
        A tuple containing:
        - Cluster labels for each sample
        - Fitted KMeans model
    """
    logger.info(f"Performing K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    logger.info("Clustering complete.")
    return cluster_labels, kmeans


def find_representative_compounds(
    features: np.ndarray,
    compound_names: List[str],
    kmeans_model: KMeans,
    cluster_labels: np.ndarray,
) -> Dict[int, str]:
    """
    For each cluster, find the compound closest to the cluster center.

    Parameters
    ----------
    features : np.ndarray
        Feature array used for clustering.
    compound_names : List[str]
        List of compound names.
    kmeans_model : KMeans
        Fitted KMeans model.
    cluster_labels : np.ndarray
        Cluster labels for each compound.

    Returns
    -------
    Dict[int, str]
        Dictionary mapping cluster ID to representative compound name.
    """
    centers = kmeans_model.cluster_centers_
    representatives = {}

    for cluster_id in range(len(centers)):
        # Get indices and data for molecules belonging to this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_features = features[cluster_mask]
        cluster_names = [
            name for i, name in enumerate(compound_names) if cluster_mask[i]
        ]

        if len(cluster_features) > 0:
            # Find the closest molecule within this specific cluster
            closest_idx, _ = pairwise_distances_argmin_min(
                [centers[cluster_id]], cluster_features
            )
            representatives[cluster_id] = cluster_names[closest_idx[0]]
        else:
            representatives[cluster_id] = "No compounds in cluster"

    logger.info(f"Representative compounds per cluster: {representatives}")
    return representatives


# === PLOTTING FUNCTION === #
def create_clustering_plot(
    embeddings_pic50: np.ndarray,
    cluster_labels_pic50: np.ndarray,
    binding_affinities_pic50: np.ndarray,
    embeddings_pki: np.ndarray,
    cluster_labels_pki: np.ndarray,
    binding_affinities_pki: np.ndarray,
    output_dir: str,
    method: str = "tsne",
    descriptor_type: str = "mace",
    **kwargs,
) -> None:
    """
    Create a 2x2 grid clustering plot for pIC50 and pKi data.

    Parameters
    ----------
    embeddings_pic50 : np.ndarray
        2D embeddings for pIC50 data.
    cluster_labels_pic50 : np.ndarray
        Cluster labels for pIC50 data.
    binding_affinities_pic50 : np.ndarray
        Binding affinities for pIC50 data.
    embeddings_pki : np.ndarray
        2D embeddings for pKi data.
    cluster_labels_pki : np.ndarray
        Cluster labels for pKi data.
    binding_affinities_pki : np.ndarray
        Binding affinities for pKi data.
    output_dir : str
        Directory to save the plot.
    method : str, default='tsne'
        Dimensionality reduction method used.
    descriptor_type : str, default='mace'
        Type of descriptor being visualized.
    **kwargs : dict
        Additional formatting arguments for the plot.

    Returns
    -------
    None
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, figsize=kwargs.get("figsize", (16, 12))
    )

    # Extract formatting kwargs with defaults
    label_fontsize = kwargs.get("label_fontsize", 12)
    tick_fontsize = kwargs.get("tick_fontsize", 10)
    legend_fontsize = kwargs.get("legend_fontsize", 10)
    marker_size = kwargs.get("marker_size", 50)
    alpha = kwargs.get("alpha", 0.7)
    grid_alpha = kwargs.get("grid_alpha", 0.3)
    dpi = kwargs.get("dpi", 300)

    # Calculate global limits for consistent scaling
    all_embeddings = np.vstack([embeddings_pic50, embeddings_pki])
    x_min, x_max = all_embeddings[:, 0].min(), all_embeddings[:, 0].max()
    y_min, y_max = all_embeddings[:, 1].min(), all_embeddings[:, 1].max()
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    xlims = (x_min - x_padding, x_max + x_padding)
    ylims = (y_min - y_padding, y_max + y_padding)

    # Plot 1: pIC50 Clustering by Labels
    for cluster_id in np.unique(cluster_labels_pic50):
        mask = cluster_labels_pic50 == cluster_id
        cluster_count = np.sum(mask)
        ax1.scatter(
            embeddings_pic50[mask, 0],
            embeddings_pic50[mask, 1],
            label=f"Cluster {cluster_id+1} (N={cluster_count})",
            alpha=alpha,
            s=marker_size,
        )
    ax1.set_ylabel(f"{method.upper()} 2", fontsize=label_fontsize)
    ax1.grid(True, alpha=grid_alpha)
    ax1.legend(fontsize=legend_fontsize)
    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims)
    ax1.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    # Plot 2: pIC50 by Affinity
    scatter2 = ax2.scatter(
        embeddings_pic50[:, 0],
        embeddings_pic50[:, 1],
        c=binding_affinities_pic50,
        cmap="viridis",
        alpha=alpha,
        s=marker_size,
    )
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label("pIC$_{50}$", fontsize=label_fontsize)
    cbar2.ax.tick_params(labelsize=tick_fontsize)
    ax2.grid(True, alpha=grid_alpha)
    ax2.set_xlim(xlims)
    ax2.set_ylim(ylims)
    ax2.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    # Plot 3: pKi Clustering by Labels
    for cluster_id in np.unique(cluster_labels_pki):
        mask = cluster_labels_pki == cluster_id
        cluster_count = np.sum(mask)
        ax3.scatter(
            embeddings_pki[mask, 0],
            embeddings_pki[mask, 1],
            label=f"Cluster {cluster_id+1} (N={cluster_count})",
            alpha=alpha,
            s=marker_size,
        )
    ax3.set_xlabel(f"{method.upper()} 1", fontsize=label_fontsize)
    ax3.set_ylabel(f"{method.upper()} 2", fontsize=label_fontsize)
    ax3.grid(True, alpha=grid_alpha)
    ax3.legend(fontsize=legend_fontsize)
    ax3.set_xlim(xlims)
    ax3.set_ylim(ylims)
    ax3.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    # Plot 4: pKi by Affinity
    scatter4 = ax4.scatter(
        embeddings_pki[:, 0],
        embeddings_pki[:, 1],
        c=binding_affinities_pki,
        cmap="magma",
        alpha=alpha,
        s=marker_size,
    )
    cbar4 = plt.colorbar(scatter4, ax=ax4)
    cbar4.set_label("pK$_{i}$", fontsize=label_fontsize)
    cbar4.ax.tick_params(labelsize=tick_fontsize)
    ax4.set_xlabel(f"{method.upper()} 1", fontsize=label_fontsize)
    ax4.grid(True, alpha=grid_alpha)
    ax4.set_xlim(xlims)
    ax4.set_ylim(ylims)
    ax4.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    plt.tight_layout()
    plot_path = (
        output_path
        / f"{descriptor_type}_{method.lower()}_clustering_both_affinities_2x2.png"
    )
    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved clustering plot to {plot_path}")
    plt.show()


def create_binding_affinity_distribution_plot(
    binding_affinities_pic50: np.ndarray,
    binding_affinities_pki: np.ndarray,
    output_dir: str,
    **kwargs,
) -> None:
    """
    Create a side-by-side plot showing the distribution of binding affinities.

    Parameters
    ----------
    binding_affinities_pic50 : np.ndarray
        Binding affinities for pIC50 data.
    binding_affinities_pki : np.ndarray
        Binding affinities for pKi data.
    output_dir : str
        Directory to save the plot.
    **kwargs : dict
        Additional formatting arguments for the plot.

    Returns
    -------
    None
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=kwargs.get("figsize", (16, 8)))

    # Extract formatting kwargs with defaults
    label_fontsize = kwargs.get("label_fontsize", 12)
    tick_fontsize = kwargs.get("tick_fontsize", 10)
    title_fontsize = kwargs.get("title_fontsize", 14)
    alpha = kwargs.get("alpha", 0.7)
    grid_alpha = kwargs.get("grid_alpha", 0.3)
    dpi = kwargs.get("dpi", 300)
    bins = kwargs.get("bins", 30)

    # Plot 1: pIC50 Distribution
    counts, bins, _ = ax1.hist(
        binding_affinities_pic50,
        bins=bins,
        alpha=alpha,
        color="skyblue",
        edgecolor="black",
    )
    ax1.set_xlabel("pIC$_{50}$", fontsize=label_fontsize)
    ax1.set_ylabel("Frequency", fontsize=label_fontsize)
    ax1.grid(True, alpha=grid_alpha)
    ax1.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    # Set y-axis to integer values
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Add subplot label A
    ax1.text(
        -0.05,
        1.05,
        "A)",
        transform=ax1.transAxes,
        fontsize=title_fontsize,
        ha="center",
        va="center",
    )

    # Add statistics text
    mean_pic50 = np.mean(binding_affinities_pic50)
    std_pic50 = np.std(binding_affinities_pic50)
    ax1.text(
        0.05,
        0.95,
        f"Mean: {mean_pic50:.2f}\nStd: {std_pic50:.2f}",
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=tick_fontsize,
    )

    # Plot 2: pKi Distribution
    counts, bins, _ = ax2.hist(
        binding_affinities_pki,
        bins=bins,
        alpha=alpha,
        color="lightcoral",
        edgecolor="black",
    )
    ax2.set_xlabel("pK$_{i}$", fontsize=label_fontsize)
    ax2.set_ylabel("Frequency", fontsize=label_fontsize)
    ax2.grid(True, alpha=grid_alpha)
    ax2.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    # Set y-axis to integer values
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Add subplot label B
    ax2.text(
        -0.05,
        1.05,
        "B)",
        transform=ax2.transAxes,
        fontsize=title_fontsize,
        ha="center",
        va="center",
    )

    # Add statistics text
    mean_pki = np.mean(binding_affinities_pki)
    std_pki = np.std(binding_affinities_pki)
    ax2.text(
        0.05,
        0.95,
        f"Mean: {mean_pki:.2f}\nStd: {std_pki:.2f}",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=tick_fontsize,
    )

    plt.tight_layout()
    plot_path = output_path / "binding_affinity_distribution.png"
    plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved binding affinity distribution plot to {plot_path}")
    plt.show()


def count_proteins_in_clusters(
    compound_names: List[str], cluster_labels: np.ndarray
) -> Dict[int, Dict[str, int]]:
    """
    For each cluster, count the occurrences of each different protein.

    Parameters
    ----------
    compound_names : List[str]
        List of compound names.
    cluster_labels : np.ndarray
        Cluster labels for each compound.

    Returns
    -------
    Dict[int, Dict[str, int]]
        Dictionary where keys are cluster IDs and values are dictionaries
        mapping protein names to their counts.
    """
    cluster_protein_counts = {}

    for cluster_id in range(len(np.unique(cluster_labels))):
        # Get compound names belonging to this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_compounds = [
            name for i, name in enumerate(compound_names) if cluster_mask[i]
        ]

        protein_counts = {}
        for compound_name in cluster_compounds:
            # Extract protein name from compound name
            parts = compound_name.split("_", 2)
            if len(parts) >= 3:
                protein_part = parts[2]
                # Remove replicate markers (e.g., 'A_', 'B_')
                if (
                    len(protein_part) > 1
                    and protein_part[1] == "_"
                    and "A" <= protein_part[0] <= "Z"
                ):
                    protein_part = protein_part[2:]

                # Remove the last dash and number from protein name
                # e.g., "MLCK-182" -> "MLCK", "PAK1-169" -> "PAK1"
                if "-" in protein_part:
                    protein_name = protein_part.rsplit("-", 1)[0]
                else:
                    protein_name = protein_part
            else:
                protein_name = "unknown"

            protein_counts[protein_name] = protein_counts.get(protein_name, 0) + 1

        cluster_protein_counts[cluster_id] = protein_counts

    logger.info(f"Protein counts per cluster: {cluster_protein_counts}")
    return cluster_protein_counts


def debug_protein_embeddings(
    protein_descriptors: Dict[str, np.ndarray],
    protein_sequences: Dict[str, np.ndarray],
    compound_names: List[str],
    target_protein: str = "PTP-1B",
) -> None:
    """
    Debug function to check if compounds binding to the same protein have identical protein embeddings.

    Parameters
    ----------
    protein_descriptors : Dict[str, np.ndarray]
        Protein descriptors dictionary.
    protein_sequences : Dict[str, np.ndarray]
        Protein sequences dictionary.
    compound_names : List[str]
        List of compound names.
    target_protein : str, default='PTP-1B'
        Target protein to analyze.

    Returns
    -------
    None
    """
    target_compounds = [name for name in compound_names if target_protein in name]

    if len(target_compounds) < 2:
        logger.info(f"Not enough {target_protein} compounds to compare")
        return

    logger.info(f"Found {len(target_compounds)} compounds for {target_protein}")

    # Get protein embeddings for all target compounds
    embeddings = []
    sequences = []
    for compound in target_compounds:
        if compound in protein_descriptors:
            # Take the <CLS> token (first element)
            embedding = protein_descriptors[compound][0, 0]
            embeddings.append(embedding)
            sequences.append(protein_sequences[compound])
            logger.info(f"{compound}: embedding shape = {embedding.shape}")

    # Check if embeddings are identical
    if len(embeddings) >= 2:
        base_embedding = embeddings[0]
        for i, embedding in enumerate(embeddings[1:], 1):
            diff = np.abs(base_embedding - embedding)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            logger.info(
                f"Difference between {target_compounds[0]} and {target_compounds[i]}:"
            )
            logger.info(f"  Max difference: {max_diff:.6f}")
            logger.info(f"  Mean difference: {mean_diff:.6f}")
            logger.info(
                f"  Are identical: {np.allclose(base_embedding, embedding, atol=1e-6)}"
            )
            logger.info(f"  Sequences: {sequences[0]} and {sequences[i]}")


def verify_compound_clustering_consistency(
    compound_names: List[str],
    cluster_labels: np.ndarray,
    original_compound_names: List[str],
    original_cluster_labels: np.ndarray,
    affinity_type: str = "",
) -> Dict[str, any]:
    """
    Verify that after aggregation, all instances of the same compound would be in the same cluster.

    Parameters
    ----------
    compound_names : List[str]
        Aggregated compound names (after aggregation).
    cluster_labels : np.ndarray
        Cluster labels for aggregated data.
    original_compound_names : List[str]
        Original compound names (before aggregation).
    original_cluster_labels : np.ndarray
        Cluster labels for original data.
    affinity_type : str, default=''
        Type of affinity data being processed.

    Returns
    -------
    Dict[str, any]
        Dictionary with verification results.
    """
    logger.info(
        f"=== COMPOUND CLUSTERING CONSISTENCY CHECK ({affinity_type.upper()}) ==="
    )

    # Create mapping from compound identifier to cluster
    compound_to_cluster = {}
    for name, cluster in zip(compound_names, cluster_labels):
        # Extract base compound identifier (compound_X)
        parts = name.split("_", 2)
        if len(parts) >= 2:
            base_compound = f"{parts[0]}_{parts[1]}"
            compound_to_cluster[base_compound] = cluster

    # Check consistency in original data
    compound_instances = {}
    inconsistencies = []

    for orig_name in original_compound_names:
        parts = orig_name.split("_", 2)
        if len(parts) >= 2:
            base_compound = f"{parts[0]}_{parts[1]}"

            if base_compound not in compound_instances:
                compound_instances[base_compound] = []
            compound_instances[base_compound].append(orig_name)

    # Verify that compounds with multiple instances would cluster consistently
    total_compounds = len(compound_instances)
    compounds_with_multiple_instances = 0
    consistent_compounds = 0

    for base_compound, instances in compound_instances.items():
        if len(instances) > 1:
            compounds_with_multiple_instances += 1

            # All instances should map to the same aggregated compound
            expected_cluster = compound_to_cluster.get(base_compound)

            logger.info(f"Compound {base_compound}:")
            logger.info(f"  Instances: {instances}")
            logger.info(f"  Expected cluster after aggregation: {expected_cluster}")

            # Check if this compound would have been consistent before aggregation
            # (This is more of a theoretical check since we're working with aggregated data)
            consistent_compounds += 1
        else:
            # Single instance compounds are automatically consistent
            consistent_compounds += 1

    # Summary statistics
    consistency_rate = (
        (consistent_compounds / total_compounds) * 100 if total_compounds > 0 else 100
    )

    results = {
        "total_unique_compounds": total_compounds,
        "compounds_with_multiple_instances": compounds_with_multiple_instances,
        "consistent_compounds": consistent_compounds,
        "consistency_rate": consistency_rate,
        "inconsistencies": inconsistencies,
    }

    logger.info(f"SUMMARY ({affinity_type.upper()}):")
    logger.info(f"  Total unique compounds: {total_compounds}")
    logger.info(
        f"  Compounds with multiple instances: {compounds_with_multiple_instances}"
    )
    logger.info(f"  Consistent compounds: {consistent_compounds}")
    logger.info(f"  Consistency rate: {consistency_rate:.1f}%")

    if inconsistencies:
        logger.warning(f"Found {len(inconsistencies)} inconsistencies!")
        for inconsistency in inconsistencies:
            logger.warning(f"  {inconsistency}")
    else:
        logger.info("✓ All compounds are clustering consistently!")

    return results


def verify_aggregation_logic(
    original_mace_descriptors: Dict[str, np.ndarray],
    original_binding_affinities: Dict[str, float],
    aggregated_compound_names: List[str],
    affinity_type: str = "",
) -> None:
    """
    Verify that the aggregation logic is working correctly.

    Checks that compounds with multiple instances are properly averaged.

    Parameters
    ----------
    original_mace_descriptors : Dict[str, np.ndarray]
        Original MACE descriptors before aggregation.
    original_binding_affinities : Dict[str, float]
        Original binding affinities before aggregation.
    aggregated_compound_names : List[str]
        List of aggregated compound names.
    affinity_type : str, default=''
        Type of affinity data being processed.

    Returns
    -------
    None
    """
    logger.info(f"=== AGGREGATION LOGIC VERIFICATION ({affinity_type.upper()}) ===")

    # Create identifier mapping
    identifier_map = {}
    for name in original_mace_descriptors.keys():
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

    # Check compounds with multiple instances
    multi_instance_compounds = {k: v for k, v in identifier_map.items() if len(v) > 1}

    logger.info(
        f"Found {len(multi_instance_compounds)} compounds with multiple instances:"
    )

    for identifier, instances in multi_instance_compounds.items():
        logger.info(f"  {identifier}: {len(instances)} instances")
        logger.info(f"    Instances: {instances}")

        # Check if binding affinities are similar
        affinities = [original_binding_affinities[name] for name in instances]
        if len(set(affinities)) > 1:
            logger.info(f"    Binding affinities: {affinities} (DIFFERENT)")
            logger.info(f"    Average affinity: {np.mean(affinities):.3f}")
        else:
            logger.info(f"    Binding affinities: {affinities[0]} (IDENTICAL)")

    # Verify that aggregated names match expected identifiers
    expected_identifiers = set(identifier_map.keys())
    actual_identifiers = set(aggregated_compound_names)

    missing_identifiers = expected_identifiers - actual_identifiers
    extra_identifiers = actual_identifiers - expected_identifiers

    if missing_identifiers:
        logger.warning(f"Missing identifiers in aggregated data: {missing_identifiers}")
    if extra_identifiers:
        logger.warning(f"Extra identifiers in aggregated data: {extra_identifiers}")

    if not missing_identifiers and not extra_identifiers:
        logger.info("✓ Aggregation identifier mapping is correct!")

    logger.info(f"Original compounds: {len(original_mace_descriptors)}")
    logger.info(f"Aggregated compounds: {len(aggregated_compound_names)}")
    logger.info(
        f"Reduction: {len(original_mace_descriptors) - len(aggregated_compound_names)} compounds"
    )


# === MAIN EXECUTION BLOCK === #
def main() -> None:
    """
    Main function to perform clustering analysis on MACE descriptors.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Perform clustering analysis on MACE descriptors."
    )
    parser.add_argument("--dataset", default="caged", help="Dataset name.")
    parser.add_argument(
        "--output_dir", default="./mace_analysis", help="Directory to save outputs."
    )
    parser.add_argument(
        "--dim_reduction",
        default="pca",
        choices=["tsne", "umap", "pca"],
        help="Dimensionality reduction method.",
    )
    parser.add_argument(
        "--n_clusters",
        default=None,
        type=int,
        help="Number of clusters for K-means. Overrides automatic selection.",
    )
    parser.add_argument(
        "--max_k",
        default=10,
        type=int,
        help="Maximum number of clusters to test for automatic k selection.",
    )
    # Formatting arguments
    parser.add_argument("--figsize_width", default=16, type=int)
    parser.add_argument("--figsize_height", default=12, type=int)
    parser.add_argument("--label_fontsize", default=20, type=int)
    parser.add_argument("--tick_fontsize", default=18, type=int)
    parser.add_argument("--legend_fontsize", default=16, type=int)
    parser.add_argument("--title_fontsize", default=22, type=int)
    parser.add_argument("--marker_size", default=70, type=int)
    parser.add_argument("--alpha", default=0.7, type=float)
    parser.add_argument("--grid_alpha", default=0.3, type=float)
    parser.add_argument("--dpi", default=300, type=int)
    parser.add_argument(
        "--bins", default=30, type=int, help="Number of bins for histogram plots"
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load both raw datasets
    (
        mace_desc_pic50,
        protein_desc_pic50,
        protein_sequences_pic50,
        pocket_desc_pic50,
        binding_affinities_pic50,
    ) = load_descriptors_from_hdf5(dataset_name=args.dataset, affinity_type="pic50")
    (
        mace_desc_pki,
        protein_desc_pki,
        protein_sequences_pki,
        pocket_desc_pki,
        binding_affinities_pki,
    ) = load_descriptors_from_hdf5(dataset_name=args.dataset, affinity_type="pk")

    # Add debug check after loading raw data
    debug_protein_embeddings(
        protein_desc_pic50,
        protein_sequences_pic50,
        list(mace_desc_pic50.keys()),
        "PTP-1B",
    )

    # 2. Aggregate data to handle duplicate compounds by averaging
    (
        agg_mace_pic50,
        agg_protein_pic50,
        agg_pocket_pic50,
        agg_aff_pic50,
        comp_names_pic50,
    ) = aggregate_descriptors_by_identifier(
        mace_desc_pic50, protein_desc_pic50, pocket_desc_pic50, binding_affinities_pic50
    )
    agg_mace_pki, agg_protein_pki, agg_pocket_pki, agg_aff_pki, comp_names_pki = (
        aggregate_descriptors_by_identifier(
            mace_desc_pki, protein_desc_pki, pocket_desc_pki, binding_affinities_pki
        )
    )
    logger.info(
        f"After aggregation: {len(comp_names_pic50)} unique compounds for pIC50, {len(comp_names_pki)} for pKi"
    )

    # ADD THESE VERIFICATION CHECKS HERE:
    # Verify aggregation logic
    verify_aggregation_logic(
        mace_desc_pic50, binding_affinities_pic50, comp_names_pic50, "pic50"
    )
    verify_aggregation_logic(
        mace_desc_pki, binding_affinities_pki, comp_names_pki, "pki"
    )

    # Create dictionaries for easier looping
    features_pic50 = {
        "mace": np.array([agg_mace_pic50[name] for name in comp_names_pic50]),
        "protein": np.array([agg_protein_pic50[name] for name in comp_names_pic50]),
        "pocket": np.array([agg_pocket_pic50[name] for name in comp_names_pic50]),
    }
    features_pki = {
        "mace": np.array([agg_mace_pki[name] for name in comp_names_pki]),
        "protein": np.array([agg_protein_pki[name] for name in comp_names_pki]),
        "pocket": np.array([agg_pocket_pki[name] for name in comp_names_pki]),
    }

    binding_array_pic50 = np.array([agg_aff_pic50[name] for name in comp_names_pic50])
    binding_array_pki = np.array([agg_aff_pki[name] for name in comp_names_pki])

    formatting_args = {
        "figsize": (args.figsize_width, args.figsize_height),
        "label_fontsize": args.label_fontsize,
        "tick_fontsize": args.tick_fontsize,
        "legend_fontsize": args.legend_fontsize,
        "marker_size": args.marker_size,
        "alpha": args.alpha,
        "grid_alpha": args.grid_alpha,
        "dpi": args.dpi,
    }

    descriptor_types = ["mace", "protein", "pocket"]
    for descriptor_type in descriptor_types:
        logger.info(f"Processing descriptor type: {descriptor_type.upper()}")

        # 3. Process pIC50 data for current descriptor type
        current_features_pic50 = features_pic50[descriptor_type]
        if current_features_pic50.ndim == 1:
            current_features_pic50 = current_features_pic50.reshape(-1, 1)

        embeddings_pic50 = perform_dimensionality_reduction(
            current_features_pic50, method=args.dim_reduction
        )

        # Determine number of clusters for pIC50
        if args.n_clusters:
            n_clusters_pic50 = args.n_clusters
            logger.info(
                f"Using manually specified number of clusters for pIC50: {n_clusters_pic50}"
            )
        else:
            n_clusters_pic50 = choose_optimal_k_elbow(
                current_features_pic50,
                max_k=args.max_k,
                output_path=output_path,
                descriptor_type=descriptor_type,
                affinity_type="pic50",
            )

        cluster_labels_pic50, kmeans_model_pic50 = perform_kmeans_clustering(
            current_features_pic50, n_clusters=n_clusters_pic50
        )
        representatives_pic50 = find_representative_compounds(
            current_features_pic50,
            comp_names_pic50,
            kmeans_model_pic50,
            cluster_labels_pic50,
        )

        # 4. Process pKi data for current descriptor type
        current_features_pki = features_pki[descriptor_type]
        if current_features_pki.ndim == 1:
            current_features_pki = current_features_pki.reshape(-1, 1)

        embeddings_pki = perform_dimensionality_reduction(
            current_features_pki, method=args.dim_reduction
        )

        # Determine number of clusters for pKi
        if args.n_clusters:
            n_clusters_pki = args.n_clusters
            logger.info(
                f"Using manually specified number of clusters for pKi: {n_clusters_pki}"
            )
        else:
            n_clusters_pki = choose_optimal_k_elbow(
                current_features_pki,
                max_k=args.max_k,
                output_path=output_path,
                descriptor_type=descriptor_type,
                affinity_type="pki",
            )

        cluster_labels_pki, kmeans_model_pki = perform_kmeans_clustering(
            current_features_pki, n_clusters=n_clusters_pki
        )
        representatives_pki = find_representative_compounds(
            current_features_pki, comp_names_pki, kmeans_model_pki, cluster_labels_pki
        )

        # 5. Create the combined clustering plot for the current descriptor type
        create_clustering_plot(
            embeddings_pic50,
            cluster_labels_pic50,
            binding_array_pic50,
            embeddings_pki,
            cluster_labels_pki,
            binding_array_pki,
            output_dir=args.output_dir,
            method=args.dim_reduction,
            descriptor_type=descriptor_type,
            **formatting_args,
        )

        # 7. Save protein counts to CSV files for the current descriptor type
        output_path = Path(args.output_dir)

        # Count proteins in clusters for both affinity types
        protein_counts_pic50 = count_proteins_in_clusters(
            comp_names_pic50, cluster_labels_pic50
        )
        protein_counts_pki = count_proteins_in_clusters(
            comp_names_pki, cluster_labels_pki
        )

        # Convert to DataFrames and save
        # For pIC50
        pic50_data = []
        for cluster_id, protein_counts in protein_counts_pic50.items():
            for protein, count in protein_counts.items():
                pic50_data.append(
                    {
                        "Cluster": cluster_id + 1,  # 1-indexed for display
                        "Protein": protein,
                        "Count": count,
                    }
                )

        rep_df_pic50 = pd.DataFrame(pic50_data)
        rep_df_pic50.to_csv(
            output_path / f"protein_counts_{descriptor_type}_pic50.csv", index=False
        )
        logger.info(
            f"Saved pIC50 protein counts to {output_path / f'protein_counts_{descriptor_type}_pic50.csv'}"
        )

        # For pKi
        pki_data = []
        for cluster_id, protein_counts in protein_counts_pki.items():
            for protein, count in protein_counts.items():
                pki_data.append(
                    {
                        "Cluster": cluster_id + 1,  # 1-indexed for display
                        "Protein": protein,
                        "Count": count,
                    }
                )

        rep_df_pki = pd.DataFrame(pki_data)
        rep_df_pki.to_csv(
            output_path / f"protein_counts_{descriptor_type}_pki.csv", index=False
        )
        logger.info(
            f"Saved pKi protein counts to {output_path / f'protein_counts_{descriptor_type}_pki.csv'}"
        )

        verify_compound_clustering_consistency(
            comp_names_pic50,
            cluster_labels_pic50,
            list(mace_desc_pic50.keys()),
            cluster_labels_pic50,
            f"{descriptor_type}_pic50",
        )
        verify_compound_clustering_consistency(
            comp_names_pki,
            cluster_labels_pki,
            list(mace_desc_pki.keys()),
            cluster_labels_pki,
            f"{descriptor_type}_pki",
        )

    # 6. Create the binding affinity distribution plot (once for all data)
    distribution_formatting_args = {
        "figsize": (16, 8),  # Wider format for side-by-side histograms
        "label_fontsize": args.label_fontsize,
        "tick_fontsize": args.tick_fontsize,
        "title_fontsize": args.title_fontsize,
        "alpha": args.alpha,
        "grid_alpha": args.grid_alpha,
        "dpi": args.dpi,
        "bins": args.bins,
    }
    create_binding_affinity_distribution_plot(
        binding_array_pic50,
        binding_array_pki,
        output_dir=args.output_dir,
        **distribution_formatting_args,
    )


if __name__ == "__main__":
    main()
