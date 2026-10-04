import os
import sys
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

# === FONT === #
import matplotlib.font_manager as fm

palatino_font_path = os.path.expanduser("~/fonts/Palatino.ttf")
if os.path.isfile(palatino_font_path):
    fm.fontManager.addfont(palatino_font_path)
    palatino_name = fm.FontProperties(fname=palatino_font_path).get_name()
    plt.rcParams["font.family"] = palatino_name
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = palatino_name
    plt.rcParams["mathtext.it"] = f"{palatino_name}:italic"
    plt.rcParams["mathtext.bf"] = f"{palatino_name}:bold"
    plt.rcParams["text.usetex"] = False
    plt.rcParams["axes.unicode_minus"] = False  # Palatino.ttf lacks the U+2212 minus glyph
else:
    print("Palatino font not found. Using default font.")

# Add the src directory to the path to import modules
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
src_path = project_root / "src" / "transfer_learning_src"
sys.path.insert(0, str(src_path))

from model.multitask import MultiTaskPocket, CCSAFinetune
from data.utils_data import get_datasets, create_dataloaders, compute_scaling_stats
from data.dataset import WrappedDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def compute_domain_alignment_metrics_pca(
    X_source: np.ndarray,
    X_target: np.ndarray,
    y_source: np.ndarray,
    y_target: np.ndarray,
    metrics: List[str] = ["jsd", "mmd", "coral", "cosine"],
    n_components: int = 50,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Compute domain alignment metrics in PCA space.

    Parameters
    ----------
    X_source : np.ndarray
        Source domain embeddings (N_src x feature_dim).
    X_target : np.ndarray
        Target domain embeddings (N_tgt x feature_dim).
    y_source : np.ndarray
        Source domain labels (N_src,).
    y_target : np.ndarray
        Target domain labels (N_tgt,).
    metrics : List[str]
        List of metrics to compute: 'jsd', 'mmd', 'coral', 'cosine'.
    n_components : int
        Number of PCA components to use.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, float]
        Dictionary containing computed metrics.
    """
    results = {}

    # Combine embeddings and apply PCA
    all_embeddings = np.vstack([X_source, X_target])
    pca = PCA(
        n_components=min(n_components, all_embeddings.shape[1]),
        random_state=random_state,
    )
    all_pca = pca.fit_transform(all_embeddings)

    # Split back into source and target
    src_size = len(X_source)
    X_source_pca = all_pca[:src_size]
    X_target_pca = all_pca[src_size:]

    # Separate by class
    src_active = X_source_pca[y_source == 1]
    src_inactive = X_source_pca[y_source == 0]
    tgt_active = X_target_pca[y_target == 1]
    tgt_inactive = X_target_pca[y_target == 0]

    # Jensen-Shannon Divergence
    if "jsd" in metrics:
        jsd_metrics = {}

        # Within-source separability
        if len(src_active) >= 2 and len(src_inactive) >= 2:
            try:
                kde_active = gaussian_kde(src_active.T)
                kde_inactive = gaussian_kde(src_inactive.T)
                all_points = np.vstack([src_active, src_inactive]).T
                p = kde_active(all_points)
                q = kde_inactive(all_points)
                p_norm = p / (np.sum(p) + 1e-10)
                q_norm = q / (np.sum(q) + 1e-10)
                jsd_metrics["source_active_vs_inactive"] = jensenshannon(
                    p_norm, q_norm, base=2
                )
            except (np.linalg.LinAlgError, ValueError):
                jsd_metrics["source_active_vs_inactive"] = float("nan")

        # Within-target separability
        if len(tgt_active) >= 2 and len(tgt_inactive) >= 2:
            try:
                kde_active = gaussian_kde(tgt_active.T)
                kde_inactive = gaussian_kde(tgt_inactive.T)
                all_points = np.vstack([tgt_active, tgt_inactive]).T
                p = kde_active(all_points)
                q = kde_inactive(all_points)
                p_norm = p / (np.sum(p) + 1e-10)
                q_norm = q / (np.sum(q) + 1e-10)
                jsd_metrics["target_active_vs_inactive"] = jensenshannon(
                    p_norm, q_norm, base=2
                )
            except (np.linalg.LinAlgError, ValueError):
                jsd_metrics["target_active_vs_inactive"] = float("nan")

        # Cross-domain alignment: Same class across domains (should be LOW for good alignment)
        if len(src_active) >= 2 and len(tgt_active) >= 2:
            try:
                kde_src_active = gaussian_kde(src_active.T)
                kde_tgt_active = gaussian_kde(tgt_active.T)
                all_points = np.vstack([src_active, tgt_active]).T
                p = kde_src_active(all_points)
                q = kde_tgt_active(all_points)
                p_norm = p / (np.sum(p) + 1e-10)
                q_norm = q / (np.sum(q) + 1e-10)
                jsd_metrics["cross_domain_active_alignment"] = jensenshannon(
                    p_norm, q_norm, base=2
                )
            except (np.linalg.LinAlgError, ValueError):
                jsd_metrics["cross_domain_active_alignment"] = float("nan")

        if len(src_inactive) >= 2 and len(tgt_inactive) >= 2:
            try:
                kde_src_inactive = gaussian_kde(src_inactive.T)
                kde_tgt_inactive = gaussian_kde(tgt_inactive.T)
                all_points = np.vstack([src_inactive, tgt_inactive]).T
                p = kde_src_inactive(all_points)
                q = kde_tgt_inactive(all_points)
                p_norm = p / (np.sum(p) + 1e-10)
                q_norm = q / (np.sum(q) + 1e-10)
                jsd_metrics["cross_domain_inactive_alignment"] = jensenshannon(
                    p_norm, q_norm, base=2
                )
            except (np.linalg.LinAlgError, ValueError):
                jsd_metrics["cross_domain_inactive_alignment"] = float("nan")

        # Cross-domain misalignment: Different classes across domains (should be HIGH for good separation)
        if len(src_active) >= 2 and len(tgt_inactive) >= 2:
            try:
                kde_src_active = gaussian_kde(src_active.T)
                kde_tgt_inactive = gaussian_kde(tgt_inactive.T)
                all_points = np.vstack([src_active, tgt_inactive]).T
                p = kde_src_active(all_points)
                q = kde_tgt_inactive(all_points)
                p_norm = p / (np.sum(p) + 1e-10)
                q_norm = q / (np.sum(q) + 1e-10)
                jsd_metrics["cross_domain_active_vs_inactive_divergence"] = (
                    jensenshannon(p_norm, q_norm, base=2)
                )
            except (np.linalg.LinAlgError, ValueError):
                jsd_metrics["cross_domain_active_vs_inactive_divergence"] = float("nan")

        if len(src_inactive) >= 2 and len(tgt_active) >= 2:
            try:
                kde_src_inactive = gaussian_kde(src_inactive.T)
                kde_tgt_active = gaussian_kde(tgt_active.T)
                all_points = np.vstack([src_inactive, tgt_active]).T
                p = kde_src_inactive(all_points)
                q = kde_tgt_active(all_points)
                p_norm = p / (np.sum(p) + 1e-10)
                q_norm = q / (np.sum(q) + 1e-10)
                jsd_metrics["cross_domain_inactive_vs_active_divergence"] = (
                    jensenshannon(p_norm, q_norm, base=2)
                )
            except (np.linalg.LinAlgError, ValueError):
                jsd_metrics["cross_domain_inactive_vs_active_divergence"] = float("nan")

        results.update({f"jsd_{k}": v for k, v in jsd_metrics.items()})

    # Maximum Mean Discrepancy (MMD)
    if "mmd" in metrics:

        def compute_mmd(X, Y):
            """Compute MMD between two distributions."""
            if len(X) == 0 or len(Y) == 0:
                return float("nan")

            # Use Gaussian kernel with median heuristic
            def gaussian_kernel(X, Y, sigma):
                pairwise_dists = (
                    np.sum(X**2, axis=1)[:, np.newaxis]
                    + np.sum(Y**2, axis=1)
                    - 2 * np.dot(X, Y.T)
                )
                return np.exp(-pairwise_dists / (2 * sigma**2))

            # Estimate sigma using median heuristic
            X_sample = X[np.random.choice(len(X), min(100, len(X)), replace=False)]
            Y_sample = Y[np.random.choice(len(Y), min(100, len(Y)), replace=False)]
            pairwise_dists = (
                np.sum(X_sample**2, axis=1)[:, np.newaxis]
                + np.sum(Y_sample**2, axis=1)
                - 2 * np.dot(X_sample, Y_sample.T)
            )
            sigma = np.sqrt(np.median(pairwise_dists))

            if sigma == 0:
                return float("nan")

            K_XX = gaussian_kernel(X, X, sigma)
            K_YY = gaussian_kernel(Y, Y, sigma)
            K_XY = gaussian_kernel(X, Y, sigma)

            mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
            return max(0, mmd)  # MMD is always non-negative

        mmd_metrics = {}

        # Cross-domain MMD: Same class across domains (should be LOW for good alignment)
        if len(src_active) >= 1 and len(tgt_active) >= 1:
            mmd_metrics["cross_domain_active_mmd"] = compute_mmd(src_active, tgt_active)

        if len(src_inactive) >= 1 and len(tgt_inactive) >= 1:
            mmd_metrics["cross_domain_inactive_mmd"] = compute_mmd(
                src_inactive, tgt_inactive
            )

        # Cross-domain MMD: Different classes across domains (should be HIGH for good separation)
        if len(src_active) >= 1 and len(tgt_inactive) >= 1:
            mmd_metrics["cross_domain_active_vs_inactive_mmd"] = compute_mmd(
                src_active, tgt_inactive
            )

        if len(src_inactive) >= 1 and len(tgt_active) >= 1:
            mmd_metrics["cross_domain_inactive_vs_active_mmd"] = compute_mmd(
                src_inactive, tgt_active
            )

        # Within-domain MMD (for reference)
        if len(src_active) >= 1 and len(src_inactive) >= 1:
            mmd_metrics["source_active_vs_inactive_mmd"] = compute_mmd(
                src_active, src_inactive
            )

        if len(tgt_active) >= 1 and len(tgt_inactive) >= 1:
            mmd_metrics["target_active_vs_inactive_mmd"] = compute_mmd(
                tgt_active, tgt_inactive
            )

        results.update(mmd_metrics)

    # CORAL distance (covariance alignment loss)
    if "coral" in metrics:

        def compute_coral_distance(X, Y):
            """Compute CORAL distance between two distributions."""
            if len(X) == 0 or len(Y) == 0:
                return float("nan")

            # Center the data
            X_centered = X - np.mean(X, axis=0)
            Y_centered = Y - np.mean(Y, axis=0)

            # Compute covariance matrices
            C_X = np.cov(X_centered.T)
            C_Y = np.cov(Y_centered.T)

            # CORAL distance
            coral_dist = np.linalg.norm(C_X - C_Y, "fro") ** 2 / (4 * X.shape[1] ** 2)
            return coral_dist

        coral_metrics = {}

        # Cross-domain CORAL: Same class across domains (should be LOW for good alignment)
        if len(src_active) >= 2 and len(tgt_active) >= 2:
            coral_metrics["cross_domain_active_coral"] = compute_coral_distance(
                src_active, tgt_active
            )

        if len(src_inactive) >= 2 and len(tgt_inactive) >= 2:
            coral_metrics["cross_domain_inactive_coral"] = compute_coral_distance(
                src_inactive, tgt_inactive
            )

        # Cross-domain CORAL: Different classes across domains (should be HIGH for good separation)
        if len(src_active) >= 2 and len(tgt_inactive) >= 2:
            coral_metrics["cross_domain_active_vs_inactive_coral"] = (
                compute_coral_distance(src_active, tgt_inactive)
            )

        if len(src_inactive) >= 2 and len(tgt_active) >= 2:
            coral_metrics["cross_domain_inactive_vs_active_coral"] = (
                compute_coral_distance(src_inactive, tgt_active)
            )

        # Within-domain CORAL (for reference)
        if len(src_active) >= 2 and len(src_inactive) >= 2:
            coral_metrics["source_active_vs_inactive_coral"] = compute_coral_distance(
                src_active, src_inactive
            )

        if len(tgt_active) >= 2 and len(tgt_inactive) >= 2:
            coral_metrics["target_active_vs_inactive_coral"] = compute_coral_distance(
                tgt_active, tgt_inactive
            )

        results.update(coral_metrics)

    # Cosine centroid similarity
    if "cosine" in metrics:

        def compute_centroid_similarity(X, Y):
            """Compute cosine similarity between centroids."""
            if len(X) == 0 or len(Y) == 0:
                return float("nan")

            centroid_X = np.mean(X, axis=0)
            centroid_Y = np.mean(Y, axis=0)

            # Compute cosine similarity
            similarity = cosine_similarity([centroid_X], [centroid_Y])[0, 0]
            return similarity

        cosine_metrics = {}

        # Cross-domain cosine similarity: Same class across domains (should be HIGH for good alignment)
        if len(src_active) >= 1 and len(tgt_active) >= 1:
            cosine_metrics["cross_domain_active_cosine"] = compute_centroid_similarity(
                src_active, tgt_active
            )

        if len(src_inactive) >= 1 and len(tgt_inactive) >= 1:
            cosine_metrics["cross_domain_inactive_cosine"] = (
                compute_centroid_similarity(src_inactive, tgt_inactive)
            )

        # Cross-domain cosine similarity: Different classes across domains (should be LOW for good separation)
        if len(src_active) >= 1 and len(tgt_inactive) >= 1:
            cosine_metrics["cross_domain_active_vs_inactive_cosine"] = (
                compute_centroid_similarity(src_active, tgt_inactive)
            )

        if len(src_inactive) >= 1 and len(tgt_active) >= 1:
            cosine_metrics["cross_domain_inactive_vs_active_cosine"] = (
                compute_centroid_similarity(src_inactive, tgt_active)
            )

        # Within-domain cosine similarity (for reference)
        if len(src_active) >= 1 and len(src_inactive) >= 1:
            cosine_metrics["source_active_vs_inactive_cosine"] = (
                compute_centroid_similarity(src_active, src_inactive)
            )

        if len(tgt_active) >= 1 and len(tgt_inactive) >= 1:
            cosine_metrics["target_active_vs_inactive_cosine"] = (
                compute_centroid_similarity(tgt_active, tgt_inactive)
            )

        results.update(cosine_metrics)

    return results


def build_train_config(
    cross_domain: bool, separate_domains: bool, source_smote: bool, target_smote: bool
) -> str:
    """
    Build training configuration directory name from boolean flags.

    This function constructs a standardized directory name that encodes the
    training configuration for domain scaling and SMOTE application.

    Parameters
    ----------
    cross_domain : bool
        If True, cross-domain scaling was used (parameter for compatibility).
    separate_domains : bool
        If True, separate domain scaling was used; if False, cross-domain scaling was used.
    source_smote : bool
        If True, SMOTE was applied to source domain.
    target_smote : bool
        If True, SMOTE was applied to target domain.

    Returns
    -------
    str
        Training configuration string (e.g., 'cross_domain_no_source_smote_no_target_smote').
    """
    # Determine domain type
    if separate_domains:
        domain_type = "separate_domain"
    else:
        domain_type = "cross_domain"

    # Build SMOTE configuration
    smote_parts = []
    if source_smote:
        smote_parts.append("source_smote")
    else:
        smote_parts.append("no_source_smote")

    if target_smote:
        smote_parts.append("target_smote")
    else:
        smote_parts.append("no_target_smote")

    # Combine domain type and SMOTE configuration
    train_config = f"{domain_type}_{smote_parts[0]}_{smote_parts[1]}"

    return train_config


class EmbeddingExtractor:
    """
    Extracts embeddings from trained models.

    This class provides functionality to extract feature embeddings, labels, and
    predictions from a trained MultiTaskPocket model given input data.

    Parameters
    ----------
    model : MultiTaskPocket
        Trained model to extract embeddings from.

    Attributes
    ----------
    model : MultiTaskPocket
        The model instance in evaluation mode.
    """

    def __init__(self, model: MultiTaskPocket) -> None:
        self.model = model
        self.model.eval()

    def extract_embeddings(
        self, dataloader: DataLoader, domain: str = "tgt"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Extract embeddings, labels, predictions, and targets from a dataloader.

        This method processes batches from the dataloader and extracts feature embeddings,
        binary labels, model predictions, continuous affinity values, and sample keys.

        Parameters
        ----------
        dataloader : DataLoader
            PyTorch DataLoader containing the data to process.
        domain : str, optional
            Domain identifier ('tgt' for target, 'src' for source), by default "tgt".

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]
            A tuple containing:
            - Feature embeddings (N x embedding_dim)
            - Binary labels (N,)
            - Model predictions (N,)
            - Continuous affinity targets (N,)
            - List of sample keys

        Raises
        ------
        RuntimeError
            If no embeddings were extracted from the dataloader.
        """
        embeddings, binary_labels, predictions, continuous_targets, keys = (
            [],
            [],
            [],
            [],
            [],
        )
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    if domain not in batch:
                        logger.error(
                            f"Domain '{domain}' not found in batch keys: {list(batch.keys())}"
                        )
                        continue
                    domain_batch = batch[domain]
                else:
                    domain_batch = batch

                batch_keys = domain_batch[0]
                data_tensors = domain_batch[1:]

                if data_tensors is None:
                    logger.error(f"Data tensors are None for domain '{domain}'")
                    continue

                emb = self.model.encode(data_tensors)
                if emb is None:
                    logger.error(f"Model.encode() returned None for domain '{domain}'")
                    continue

                pred_logits = self.model(data_tensors)
                pred_labels = (torch.sigmoid(pred_logits).squeeze(-1) > 0.5).float()

                labels_tensor = data_tensors[-1]
                if labels_tensor is None:
                    logger.error("Batch labels are None")
                    continue

                binary_labels_batch = (labels_tensor > self.model.threshold).float()

                embeddings.append(emb.cpu().numpy())
                binary_labels.append(binary_labels_batch.cpu().numpy())
                predictions.append(pred_labels.cpu().numpy())
                continuous_targets.append(labels_tensor.cpu().numpy())
                keys.extend(batch_keys)

        if not embeddings:
            raise RuntimeError(f"No embeddings were extracted for domain '{domain}'")

        return (
            np.vstack(embeddings),
            np.concatenate(binary_labels),
            np.concatenate(predictions),
            np.concatenate(continuous_targets),
            keys,
        )


class TSNEVisualizer:
    """
    Creates t-SNE visualizations of high-dimensional embeddings.

    This class wraps scikit-learn's t-SNE implementation for dimensionality reduction
    of feature embeddings for visualization purposes. Uses PCA preprocessing for
    quantitative analysis and t-SNE for visualization.

    Parameters
    ----------
    n_components : int, optional
        Number of dimensions for t-SNE output, by default 2.
    perplexity : float, optional
        Perplexity parameter for t-SNE (related to number of nearest neighbors),
        by default 30.0.
    random_state : int, optional
        Random seed for reproducibility, by default 42.
    n_iter : int, optional
        Maximum number of iterations for optimization, by default 1000.
    pca_components : int, optional
        Number of PCA components to use before t-SNE, by default 50.

    Attributes
    ----------
    tsne : TSNE
        Scikit-learn t-SNE instance with configured parameters.
    pca : PCA
        Scikit-learn PCA instance for preprocessing.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        random_state: int = 42,
        n_iter: int = 1000,
        pca_components: int = 50,
    ) -> None:
        self.random_state = random_state
        self.pca_components = pca_components
        self.tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=n_iter,
            verbose=1,
        )
        self.pca = None

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit PCA and t-SNE and transform embeddings to lower-dimensional space.

        This method first applies PCA dimensionality reduction to the input embeddings,
        then applies t-SNE for visualization. PCA is used for quantitative analysis
        while t-SNE is used for visualization.

        Parameters
        ----------
        embeddings : np.ndarray
            High-dimensional embeddings to reduce (N x feature_dim).

        Returns
        -------
        np.ndarray
            Reduced embeddings (N x n_components).
        """
        logger.info(
            f"Applying PCA (n_components={self.pca_components}) followed by t-SNE (perplexity={self.tsne.perplexity})"
        )

        # First apply PCA
        self.pca = PCA(
            n_components=min(self.pca_components, embeddings.shape[1]),
            random_state=self.random_state,
        )
        pca_embeddings = self.pca.fit_transform(embeddings)

        logger.info(
            f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}"
        )

        # Adjust perplexity if it's too high for the number of samples
        if self.tsne.perplexity >= len(pca_embeddings):
            self.tsne.perplexity = len(pca_embeddings) - 1
            logger.warning(
                f"Perplexity adjusted to {self.tsne.perplexity} due to small sample size."
            )

        # Apply t-SNE to PCA-reduced embeddings
        tsne_embeddings = self.tsne.fit_transform(pca_embeddings)

        return tsne_embeddings


class EmbeddingVisualizer:
    """
    Main class for orchestrating embedding visualization and analysis.

    Parameters
    ----------
    params : Dict[str, Any]
        Dictionary of model and training parameters.
    model_path : str
        Path to the directory containing trained model checkpoints.
    output_dir : str
        Directory path where visualization outputs will be saved.
    descriptor_type : str, optional
        Type of molecular descriptor used (e.g., 'SOAP', 'MACE', 'ACSF'), by default "SOAP".
    modality : str, optional
        Feature modality used by the model, by default "molecule".
    scaler_type : str, optional
        Type of data scaler used ('minmax', 'standard', 'none'), by default "minmax".
    affinity_type : str, optional
        Type of binding affinity ('pic50' or 'pk'), by default "pic50".
    cross_domain : bool, optional
        Whether cross-domain scaling was used, by default True.
    separate_domains : bool, optional
        Whether separate domain scaling was used, by default False.
    source_smote : bool, optional
        Whether SMOTE was applied to source domain, by default False.
    target_smote : bool, optional
        Whether SMOTE was applied to target domain, by default False.
    use_simple_structure : bool, optional
        If True, use simple directory structure (model_path/fold_X/phase/checkpoints).
        If False, use complex nested structure with descriptor/modality/scaler layers.
        By default False.
    metrics : List[str], optional
        List of metrics to compute ('jsd', 'mmd', 'coral', 'cosine'), by default ["jsd", "mmd", "coral", "cosine"].
    pca_components : int, optional
        Number of PCA components to use for quantitative analysis, by default 50.
    random_state : int, optional
        Random seed for reproducibility, by default 42.
    recompute_stats : bool, optional
        If True, recompute scaling statistics following training logic.
        If False, load from saved pickle files, by default False.

    Attributes
    ----------
    params : Dict[str, Any]
        Stored model and training parameters.
    model_path : Path
        Path object to model directory.
    output_dir : Path
        Path object to output directory (organized by threshold and affinity type).
    tsne_visualizer : TSNEVisualizer
        Instance for performing t-SNE dimensionality reduction.
    descriptor_type : str
        Molecular descriptor type.
    modality : str
        Feature modality.
    scaler_type : str
        Data scaler type.
    affinity_type : str
        Binding affinity type.
    train_config : str
        Training configuration string derived from domain and SMOTE settings.
    use_simple_structure : bool
        Whether to use simple directory structure for model paths.
    recompute_stats : bool
        Whether to recompute scaling statistics instead of loading from files.
    _cached_raw_source_data : Dict[str, dict] or None
        Cached raw source data to avoid redundant loading when computing statistics.
    """

    def __init__(
        self,
        params: Dict[str, Any],
        model_path: str,
        output_dir: str,
        descriptor_type: str = "SOAP",
        modality: str = "molecule",
        scaler_type: str = "minmax",
        affinity_type: str = "pic50",
        cross_domain: bool = True,
        separate_domains: bool = False,
        source_smote: bool = False,
        target_smote: bool = False,
        use_simple_structure: bool = False,
        metrics: List[str] = ["jsd", "mmd", "coral", "cosine"],
        pca_components: int = 50,
        random_state: int = 42,
        recompute_stats: bool = False,
    ) -> None:
        self.params = params
        self.model_path = Path(model_path).expanduser()
        # Organize outputs under: output_dir/threshold_x/affinity_type
        self.base_output_dir = Path(output_dir).expanduser()

        # Check if threshold directory is already in the path
        if f"threshold_{int(params.get('threshold', 0))}" in str(self.base_output_dir):
            # Threshold directory already exists in path, use it directly
            self.threshold_dir = self.base_output_dir
        else:
            # Add threshold directory
            threshold_tag = f"threshold_{int(params.get('threshold', 0))}"
            self.threshold_dir = self.base_output_dir / threshold_tag

        self.threshold_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.threshold_dir / affinity_type
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tsne_visualizer = TSNEVisualizer(
            pca_components=pca_components, random_state=random_state
        )
        self._cached_source_dataloaders = None
        self._cached_normalization_stats = {}
        self._cached_source_scaling_stats = None
        self._cached_raw_source_data = None

        # Model configuration parameters
        self.descriptor_type = descriptor_type
        self.modality = modality
        self.scaler_type = scaler_type
        self.affinity_type = affinity_type
        self.cross_domain = cross_domain
        self.separate_domains = separate_domains
        self.source_smote = source_smote
        self.target_smote = target_smote
        self.use_simple_structure = use_simple_structure
        self.metrics = metrics
        self.pca_components = pca_components
        self.random_state = random_state
        self.recompute_stats = recompute_stats

        # Build training configuration
        self.train_config = build_train_config(
            cross_domain=cross_domain,
            separate_domains=separate_domains,
            source_smote=source_smote,
            target_smote=target_smote,
        )

        # Font sizes for plotting
        self.label_fontsize = 30
        self.tick_fontsize = 30
        self.legend_fontsize = 20

        # Colors for plotting - scalar colors
        self.scatter_colors = {
            "src_active":        "#2E7D32",  # Forest green (warm, distinct from blue)
            "src_inactive":      "#81C784",  # Soft mint green
            "tgt_train_active":  "#1565C0",  # Vivid royal blue (not navy)
            "tgt_train_inactive":"#90CAF9",  # Clear sky blue
            "tgt_test_active":   "#C62828",  # Crimson (not as dark/muddy as darkred)
            "tgt_test_inactive": "#F48FB1",  # Warm pink (distinct from the blue inactive)
        }

        # Source-only plot colors: highlight actives against a neutral background
        self.source_only_colors = {
            "src_active":   "#00FF3C",  # Fluorescent green
            "src_inactive": "#B0B0B0",  # Light neutral gray
        }

        # KDE-specific colors
        self.kde_colors = {
            "src_active":   "#1B5E20",  # Deep forest green (solid contour)
            "src_inactive": "#F9A825",  # Golden amber (dashed contour — fully distinct from green)
        }

        # Prediction plot colors
        self.prediction_colors = {
            "train_active":  "#1565C0",  # Royal blue
            "train_inactive":"#E53935",  # Vivid red (replaces tomato)
            "tp":            "#00B0FF",  # Bright cyan-blue
            "tn":            "#F48FB1",  # Warm pink
            "fp_bg":         "#F48FB1",  # Warm pink
            "fn_bg":         "#00B0FF",  # Bright cyan-blue
        }

        # Marker sizes for plotting
        self.marker_sizes = {
            "src_only": 50,              # Source-only plot scatter size
            "src_combined": 50,          # Source scatter in combined plots
            "tgt_train_combined": 80,    # Target train scatter in combined plots
            "tgt_test_combined": 80,     # Target test scatter in combined plots
            "tgt_train_prediction": 100, # Target train scatter in prediction plot
            "tp": 175,                   # True positive scatter size
            "tn": 175,                   # True negative scatter size
            "fp": 200,                   # False positive scatter size
            "fn": 200,                   # False negative scatter size
        }

        # Legend marker size (for Line2D objects in legends)
        self.legend_marker_size = 12

        # Line widths for plotting
        self.line_widths = {
            "kde_contour": 1.8,        # KDE contour lines
            "spine": 1.5,              # Axis spines
            "scatter_edge": 0.5,       # Scatter plot edge colors
            "axis_tick": 2,            # Axis tick marks
            "legend_edge": 1.5,        # Legend marker edge (mew for X markers)
        }

        # Alpha (transparency) values for plotting
        self.alpha_values = {
            "src_scatter": 0.7,        # Source scatter plot transparency
            "tgt_train_combined": 0.9, # Target train in combined plots transparency
            "tgt_test_combined": 0.9,  # Target test in combined plots transparency
        }

    def _load_raw_source_data_once(self) -> Dict[str, dict]:
        """
        Load raw source data once and cache it for statistics computation.

        This method ensures raw source data is loaded only once to improve performance
        and avoid redundant loading when computing scaling statistics.

        Returns
        -------
        Dict[str, dict]
            Raw source data dictionary with 'inhibitor_feats', 'protein_feats',
            'pocket_feats', 'binding_affs' keys.
        """
        if self._cached_raw_source_data is not None:
            return self._cached_raw_source_data

        from data.load_data import load_data_hdf5

        logger.info("Loading raw source data once for statistics computation...")
        source_raw = load_data_hdf5(
            "organic",
            self.descriptor_type,
            self.affinity_type,
            percent_of_data=self.params.get("organic_percent_of_data", 1.0),
        )
        self._cached_raw_source_data = source_raw
        return source_raw

    def _load_source_data_once(
        self,
        use_cached_raw_data: bool = False,
        source_scaling_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Load source data once and cache it for reuse across folds.

        This method ensures source data is loaded only once to improve performance
        when processing multiple folds. The loaded data is stored in the instance's
        cached source dataloaders attribute.

        Parameters
        ----------
        use_cached_raw_data : bool, optional
            If True and raw source data is cached, create datasets directly from it
            instead of loading from disk again. Default is False.
        source_scaling_stats : Dict[str, Any] or None, optional
            Pre-computed source scaling statistics to use when creating datasets.

        Returns
        -------
        None
        """
        if self._cached_source_dataloaders:
            return

        logger.info("Loading source data once...")
        temp_params = {
            **self.params,
            "train_on": "src",
            "fold": 1,
            "source_smote": False,
            "split_ratio": 1.0,
        }

        # If we have cached raw data and should use it, create datasets directly
        if use_cached_raw_data and self._cached_raw_source_data is not None:
            logger.info("Using cached raw source data to create dataloaders...")
            from data.dataset import BindingAffinitiesDatasetWithKeys

            # Create dataset directly from cached raw data
            src_train_ds = BindingAffinitiesDatasetWithKeys(
                self._cached_raw_source_data["inhibitor_feats"],
                self._cached_raw_source_data["protein_feats"],
                self._cached_raw_source_data["pocket_feats"],
                self._cached_raw_source_data["binding_affs"],
                self.modality,
                apply_smote=False,
                scaler_type=self.scaler_type,
                scaler_stats=source_scaling_stats,
            )
            loaders = create_dataloaders(src_train_ds, None, None, None, temp_params)
            self._cached_source_dataloaders = loaders
        else:
            # Standard loading path using get_datasets
            if source_scaling_stats is not None:
                temp_params["cross_domain_scaling"] = True
                datasets, _ = get_datasets(
                    temp_params, fold=None, global_scaling_stats=source_scaling_stats
                )
            else:
                datasets, _ = get_datasets(temp_params, fold=None)

            src_train, _, _, _ = datasets
            loaders = create_dataloaders(src_train, None, None, None, temp_params)
            self._cached_source_dataloaders = loaders

    def _compute_scaling_statistics(self, fold: int) -> Optional[Dict[str, Any]]:
        """
        Compute scaling statistics following training logic.

        This method computes scaling statistics using the same logic as during training:
        - For cross-domain scaling: Use source statistics for both domains
        - For separate-domain scaling: Compute separate statistics per domain

        Parameters
        ----------
        fold : int
            Cross-validation fold number (used for loading target training data).

        Returns
        -------
        Dict[str, Any] or None
            Scaling statistics dictionary or None if no scaling is applied.
        """
        scaling_method = self.scaler_type

        # No scaling case
        if scaling_method is None or scaling_method == "none":
            logger.info("No scaling will be applied")
            return None

        # Determine cross-domain scaling mode
        cross_domain_scaling = not self.separate_domains

        if cross_domain_scaling:
            # CROSS-DOMAIN SCALING: Use source statistics for both domains
            if self._cached_source_scaling_stats is not None:
                logger.info(
                    "Using cached source scaling statistics for cross-domain scaling"
                )
                return self._cached_source_scaling_stats

            # Load raw source data (cached to avoid redundant loading)
            logger.info(
                "Computing source scaling statistics for cross-domain scaling..."
            )
            source_raw = self._load_raw_source_data_once()

            # Compute statistics from source data
            stats = compute_scaling_stats(source_raw, self.modality, scaling_method)
            self._cached_source_scaling_stats = stats
            logger.info(
                "Computed cross-domain scaling statistics from source training data"
            )
            return stats

        else:
            # DOMAIN-SPECIFIC SCALING: Compute separate statistics for each domain
            # For visualization, we need target stats per fold
            from data.load_data import load_data_hdf5
            from data.split_data import split_data_by_group

            # Compute source stats once (cached)
            if self._cached_source_scaling_stats is None:
                logger.info(
                    "Computing source scaling statistics for separate-domain scaling..."
                )
                source_raw = self._load_raw_source_data_once()
                self._cached_source_scaling_stats = compute_scaling_stats(
                    source_raw, self.modality, scaling_method
                )
                logger.info("Computed domain-specific source scaling statistics")

            # Load and split target data for this fold
            logger.info(f"Computing target scaling statistics for fold {fold}...")
            target_raw = load_data_hdf5(
                "caged",
                self.descriptor_type,
                self.affinity_type,
            )

            # Split target data by fold
            tgt_train, tgt_val = split_data_by_group(
                target_raw["inhibitor_feats"],
                target_raw["protein_feats"],
                target_raw["pocket_feats"],
                target_raw["binding_affs"],
                "caged",
                self.affinity_type,
                fold=fold,
                n_folds=self.params["n_folds"],
            )

            # Compute target statistics from target training data
            tgt_stats = compute_scaling_stats(tgt_train, self.modality, scaling_method)
            logger.info(
                f"Computed domain-specific target scaling statistics for fold {fold}"
            )

            # Return structure similar to prepare_datasets
            return tgt_stats

    def load_data(self, fold: int, phase: str) -> Dict[str, Any]:
        """
        Load data for a specific fold, reusing cached source data.

        This method loads target domain data for the specified fold and phase,
        while reusing the cached source domain data. Normalization statistics
        are also loaded if available.

        Parameters
        ----------
        fold : int
            Cross-validation fold number to load.
        phase : str
            Training phase identifier (e.g., 'ccsa', 'finetune').

        Returns
        -------
        Dict[str, Any]
            Dictionary containing data loaders for source and target domains.
        """
        logger.info(f"Loading data for fold {fold}, phase {phase}")

        # Determine whether to compute or load statistics
        # Compute statistics before loading source data to avoid double-loading
        if self.recompute_stats:
            # Compute statistics following training logic (this loads raw data and caches it)
            logger.info(f"Recomputing scaling statistics for fold {fold}...")
            norm_stats = self._compute_scaling_statistics(fold)
            use_cross_scaling = not self.separate_domains
            # Now create source dataloaders using cached raw data (avoids reloading from disk)
            self._load_source_data_once(
                use_cached_raw_data=True, source_scaling_stats=norm_stats
            )
        else:
            # Load source data first with default behavior
            self._load_source_data_once()
            # Load from pickle file (current behavior)
            if fold not in self._cached_normalization_stats:
                # Use the appropriate path structure based on configuration
                if self.use_simple_structure:
                    # Simple structure: model_path/fold_X/phase/
                    stats_file = (
                        self.model_path
                        / f"fold_{fold}"
                        / phase
                        / "normalization_stats.pkl"
                    )
                else:
                    # Complex structure: descriptor/modality/scaler/affinity/train_config/fold_X/phase/
                    stats_file = (
                        self.model_path
                        / self.descriptor_type
                        / self.modality
                        / f"{self.scaler_type}_scaler"
                        / self.affinity_type
                        / self.train_config
                        / f"fold_{fold}"
                        / phase
                        / "normalization_stats.pkl"
                    )
                if stats_file.exists():
                    import pickle

                    with open(stats_file, "rb") as f:
                        self._cached_normalization_stats[fold] = pickle.load(f)
                        logger.info(f"Loaded normalization stats from {stats_file}")
                else:
                    self._cached_normalization_stats[fold] = None
                    logger.warning(f"Normalization stats file not found: {stats_file}")
                    logger.info(
                        f"Falling back to computing statistics for fold {fold}..."
                    )
                    self._cached_normalization_stats[fold] = (
                        self._compute_scaling_statistics(fold)
                    )

            norm_stats = self._cached_normalization_stats[fold]
            use_cross_scaling = norm_stats is not None

        temp_params = {
            **self.params,
            "fold": fold,
            "train_on": "tgt",
            "cross_domain_scaling": use_cross_scaling,
        }
        datasets, _ = get_datasets(
            temp_params, fold=fold, global_scaling_stats=norm_stats
        )
        _, _, tgt_train, tgt_val = datasets

        tgt_loaders = create_dataloaders(None, None, tgt_train, tgt_val, temp_params)
        tgt_loaders["src_train_loader"] = self._cached_source_dataloaders[
            "src_train_loader"
        ]
        return {**tgt_loaders}

    def load_model_checkpoint(self, fold: int, phase: str) -> MultiTaskPocket:
        """
        Load a trained model checkpoint for a specific fold and phase.

        This method constructs the checkpoint path based on the model configuration
        and loads the trained model weights.

        Parameters
        ----------
        fold : int
            Cross-validation fold number.
        phase : str
            Training phase identifier.

        Returns
        -------
        MultiTaskPocket
            Loaded model with trained weights.

        Raises
        ------
        FileNotFoundError
            If the checkpoint directory or checkpoint files are not found.
        """
        logger.info(f"Loading model for fold {fold}, phase {phase}")

        # Build the path based on configuration
        if self.use_simple_structure:
            # Simple structure: model_path/fold_X/phase/checkpoints
            checkpoint_dir = self.model_path / f"fold_{fold}" / phase / "checkpoints"
        else:
            # Complex structure: descriptor/modality/scaler/affinity/train_config/fold_X/phase/checkpoints
            checkpoint_dir = (
                self.model_path
                / self.descriptor_type
                / self.modality
                / f"{self.scaler_type}_scaler"
                / self.affinity_type
                / self.train_config
                / f"fold_{fold}"
                / phase
                / "checkpoints"
            )

        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        # Find the best checkpoint file
        checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

        # Use the first checkpoint (assuming they're named consistently)
        checkpoint_path = checkpoint_files[0]
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Always load as MultiTaskPocket - no need for CCSAFinetune wrapper
        model = MultiTaskPocket(domain="tgt", **self.params)
        model.load_state_dict(checkpoint["state_dict"], strict=False)

        return model

    def discover_available_folds_and_phases(
        self,
    ) -> Tuple[List[int], Dict[int, List[str]]]:
        """
        Discover available folds and phases automatically from the model directory.

        This method scans the model directory structure to identify which folds
        and training phases have available checkpoints.

        Returns
        -------
        Tuple[List[int], Dict[int, List[str]]]
            A tuple containing:
            - List of available fold numbers
            - Dictionary mapping fold numbers to lists of available phase names

        Raises
        ------
        FileNotFoundError
            If the base model path does not exist.
        """
        # Build the base path based on configuration
        if self.use_simple_structure:
            # Simple structure: model_path is already the base
            base_path = self.model_path
        else:
            # Complex structure: build nested path
            base_path = (
                self.model_path
                / self.descriptor_type
                / self.modality
                / f"{self.scaler_type}_scaler"
                / self.affinity_type
                / self.train_config
            )

        if not base_path.exists():
            raise FileNotFoundError(f"Base model path not found: {base_path}")

        # Discover available folds
        available_folds = []
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith("fold_"):
                try:
                    fold_num = int(item.name.split("_")[1])
                    available_folds.append(fold_num)
                except ValueError:
                    continue

        available_folds.sort()
        logger.info(f"Found available folds: {available_folds}")

        # Discover available phases for each fold
        fold_phases = {}
        for fold in available_folds:
            fold_path = base_path / f"fold_{fold}"
            if not fold_path.exists():
                continue

            available_phases = []
            for item in fold_path.iterdir():
                if item.is_dir() and item.name != "pretrain_src":  # Skip pretrain_src
                    available_phases.append(item.name)

            fold_phases[fold] = available_phases
            logger.info(f"Found available phases for fold {fold}: {available_phases}")

        return available_folds, fold_phases

    def extract_all_embeddings(self, fold: int, phase: str) -> Dict[str, Any]:
        """
        Extract all embeddings from source, target train, and target test sets for a fold.

        This method loads the model and data for the specified fold and phase,
        then extracts embeddings from all available data splits.

        Parameters
        ----------
        fold : int
            Cross-validation fold number.
        phase : str
            Training phase identifier.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys for each data split ('src_train', 'tgt_train', 'tgt_test')
            and values containing embeddings, labels, predictions, continuous targets, and keys.
        """
        logger.info(f"Extracting all embeddings for fold {fold}, phase {phase}")
        model = self.load_model_checkpoint(fold, phase)
        loaders = self.load_data(fold, phase)
        extractor = EmbeddingExtractor(model)

        results = {}
        data_splits = {
            "src_train": ("src_train_loader", "src"),
            "tgt_train": ("tgt_train_loader", "tgt"),
            "tgt_test": ("tgt_val_loader", "tgt"),
        }

        for name, (loader_key, domain) in data_splits.items():
            if loader_key in loaders:
                try:
                    logger.info(f"Extracting embeddings for {name} using {loader_key}")
                    loader = WrappedDataLoader(loaders[loader_key], domain)
                    emb, labels, preds, cont, keys = extractor.extract_embeddings(
                        loader, domain
                    )
                    results[name] = {
                        "embeddings": emb,
                        "labels": labels,
                        "predictions": preds,
                        "continuous": cont,
                        "keys": keys,
                    }
                    logger.info(
                        f"Successfully extracted {len(emb)} embeddings for {name}"
                    )
                except RuntimeError as e:
                    logger.warning(f"Could not extract embeddings for {name}: {e}")
            else:
                logger.warning(f"Dataloader key '{loader_key}' not found for {name}.")

        logger.info(f"Final results for fold {fold}: {list(results.keys())}")
        return results

    def _analyze_predictions(
        self, true_labels: np.ndarray, pred_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze prediction performance and calculate classification metrics.

        This method computes accuracy, precision, recall, F1 score, and confusion
        matrix values from predicted and true labels.

        Parameters
        ----------
        true_labels : np.ndarray
            Ground truth binary labels.
        pred_labels : np.ndarray
            Predicted binary labels.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - accuracy: Overall accuracy
            - precision: Precision score
            - recall: Recall score
            - f1_score: F1 score
            - true_positives, true_negatives, false_positives, false_negatives: Confusion matrix values
            - correct_indices: Indices of correctly predicted samples
            - incorrect_indices: Indices of misclassified samples
        """
        correct_mask = true_labels == pred_labels
        cm_values = np.bincount(
            true_labels.astype(int) * 2 + pred_labels.astype(int), minlength=4
        )
        tn, fp, fn, tp = cm_values

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        analysis = {
            "accuracy": np.mean(correct_mask),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "correct_indices": np.where(correct_mask)[0],
            "incorrect_indices": np.where(~correct_mask)[0],
        }
        logger.info(
            f"Prediction Analysis (Fold): Accuracy={analysis['accuracy']:.4f}, F1={analysis['f1_score']:.4f}, FP={fp}, FN={fn}"
        )
        return analysis

    def compute_kde_density(
        self, tsne_embs: np.ndarray, labels: np.ndarray, grid_size: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Kernel Density Estimation for active and inactive classes.

        This method calculates KDE over a 2D grid for visualization of class
        distributions in t-SNE space.

        Parameters
        ----------
        tsne_embs : np.ndarray
            2D t-SNE embeddings (N x 2).
        labels : np.ndarray
            Binary labels (N,) indicating class membership.
        grid_size : int, optional
            Number of grid points along each axis, by default 100.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing:
            - xx: X-coordinates of grid
            - yy: Y-coordinates of grid
            - active_density: KDE density values for active class
            - inactive_density: KDE density values for inactive class
        """
        active_embs = tsne_embs[labels == 1]
        inactive_embs = tsne_embs[labels == 0]

        xmin, ymin = tsne_embs.min(axis=0)
        xmax, ymax = tsne_embs.max(axis=0)

        xx, yy = np.mgrid[
            xmin : xmax : complex(0, grid_size), ymin : ymax : complex(0, grid_size)
        ]
        grid_coords = np.vstack([xx.ravel(), yy.ravel()])

        active_density = np.zeros(xx.shape)
        if active_embs.shape[0] > 1:
            try:
                kde_active = gaussian_kde(active_embs.T)
                active_density = kde_active(grid_coords).reshape(xx.shape)
            except np.linalg.LinAlgError:
                logger.warning(
                    "Could not compute KDE for active class (singular matrix)."
                )

        inactive_density = np.zeros(xx.shape)
        if inactive_embs.shape[0] > 1:
            try:
                kde_inactive = gaussian_kde(inactive_embs.T)
                inactive_density = kde_inactive(grid_coords).reshape(xx.shape)
            except np.linalg.LinAlgError:
                logger.warning(
                    "Could not compute KDE for inactive class (singular matrix)."
                )

        return xx, yy, active_density, inactive_density

    def separability_metrics(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate separability metrics between two classes using PCA-based analysis.

        This method computes Jensen-Shannon Divergence and Bhattacharyya distance
        in PCA space to quantify the separation between active and inactive classes.

        Parameters
        ----------
        embeddings : np.ndarray
            Feature embeddings (N x feature_dim).
        labels : np.ndarray
            Binary labels (N,) indicating class membership.

        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            - JSD: Jensen-Shannon Divergence
            - Bhattacharyya: Bhattacharyya distance
        """
        active_embs = embeddings[labels == 1]
        inactive_embs = embeddings[labels == 0]

        if active_embs.shape[0] < 2 or inactive_embs.shape[0] < 2:
            logger.warning(
                "Not enough samples in one of the classes for KDE separability. Need at least 2."
            )
            return {"JSD": float("nan"), "Bhattacharyya": float("nan")}

        # Apply PCA for quantitative analysis
        pca = PCA(
            n_components=min(self.pca_components, embeddings.shape[1]),
            random_state=self.random_state,
        )
        pca_embeddings = pca.fit_transform(embeddings)

        active_pca = pca_embeddings[labels == 1]
        inactive_pca = pca_embeddings[labels == 0]

        # Check if the number of samples is sufficient for the given dimension
        if active_pca.shape[0] <= active_pca.shape[1]:
            logger.warning(
                f"The number of active samples ({active_pca.shape[0]}) is not greater than the PCA dimension ({active_pca.shape[1]}). KDE may be unstable."
            )
        if inactive_pca.shape[0] <= inactive_pca.shape[1]:
            logger.warning(
                f"The number of inactive samples ({inactive_pca.shape[0]}) is not greater than the PCA dimension ({inactive_pca.shape[1]}). KDE may be unstable."
            )

        try:
            kde_active = gaussian_kde(active_pca.T)
            kde_inactive = gaussian_kde(inactive_pca.T)
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(
                f"Could not compute KDE for separability metrics due to: {e}"
            )
            return {"JSD": float("nan"), "Bhattacharyya": float("nan")}

        # Evaluate the KDEs on the points from both distributions
        all_points = np.vstack([active_pca, inactive_pca]).T
        p = kde_active(all_points)
        q = kde_inactive(all_points)

        # Normalize to get probability distributions
        p_norm = p / (np.sum(p) + 1e-10)
        q_norm = q / (np.sum(q) + 1e-10)

        jsd = jensenshannon(p_norm, q_norm, base=2)
        bc_coefficient = np.sum(np.sqrt(p_norm * q_norm))
        b_dist = -np.log(bc_coefficient + 1e-10)

        return {"JSD": jsd, "Bhattacharyya": b_dist}

    def compute_comprehensive_separability_metrics(
        self, all_fold_data: Dict[int, Dict[str, Any]], phase: str
    ) -> Dict[str, Any]:
        """
        Compute comprehensive separability metrics across all folds and domains using PCA-based analysis.

        This method calculates JSD, MMD, CORAL, and cosine similarity metrics for:
        - Organic domain: active vs inactive (per fold)
        - Metal-Organic domain: active vs inactive (per fold)
        - Organic vs Metal-Organic: cross-domain class alignment (per fold)
            * Organic Active vs Metal-Organic Active (should be low if well-aligned)
            * Organic Active vs Metal-Organic Inactive (should be high)
            * Organic Inactive vs Metal-Organic Active (should be high)
            * Organic Inactive vs Metal-Organic Inactive (should be low if well-aligned)

        Parameters
        ----------
        all_fold_data : Dict[int, Dict[str, Any]]
            Dictionary mapping fold numbers to their extracted data.
        phase : str
            Training phase identifier.

        Returns
        -------
        Dict[str, Any]
            Comprehensive dictionary of separability metrics organized by plot type.
        """
        logger.info(
            "Computing comprehensive separability metrics across all folds using PCA-based analysis..."
        )

        comprehensive_metrics = {
            "phase": phase,
            "metrics_used": self.metrics,
            "pca_components": self.pca_components,
            "random_state": self.random_state,
            "source_only": {},
            "target_only": {},
            "source_target_combined": {},
        }

        for fold, fold_data in all_fold_data.items():
            logger.info(f"Computing metrics for fold {fold}...")

            # === SOURCE ONLY METRICS ===
            if "src_train" in fold_data and fold_data["src_train"] is not None:
                src_data = fold_data["src_train"]
                if len(np.unique(src_data["labels"])) >= 2:
                    # Use PCA-based separability metrics for source domain
                    src_separability = self.separability_metrics(
                        src_data["embeddings"], src_data["labels"]
                    )
                    comprehensive_metrics["source_only"][f"fold_{fold}"] = {
                        "active_vs_inactive": src_separability
                    }
                    logger.info(
                        f"Fold {fold} - Source separability: {src_separability}"
                    )

            # === TARGET ONLY METRICS ===
            # Combine target train and test for target-only analysis
            if "tgt_train" in fold_data and "tgt_test" in fold_data:
                tgt_train = fold_data["tgt_train"]
                tgt_test = fold_data["tgt_test"]

                if tgt_train is not None and tgt_test is not None:
                    # Combine target train and test
                    tgt_combined_embs = np.vstack(
                        [tgt_train["embeddings"], tgt_test["embeddings"]]
                    )
                    tgt_combined_labels = np.concatenate(
                        [tgt_train["labels"], tgt_test["labels"]]
                    )

                    if len(np.unique(tgt_combined_labels)) >= 2:
                        # Use PCA-based separability metrics for target domain
                        tgt_separability = self.separability_metrics(
                            tgt_combined_embs, tgt_combined_labels
                        )
                        comprehensive_metrics["target_only"][f"fold_{fold}"] = {
                            "active_vs_inactive": tgt_separability
                        }
                        logger.info(
                            f"Fold {fold} - Target separability: {tgt_separability}"
                        )

            # === SOURCE AND TARGET COMBINED METRICS ===
            # Check cross-domain class alignment using PCA-based metrics
            if (
                "src_train" in fold_data
                and "tgt_train" in fold_data
                and "tgt_test" in fold_data
            ):
                src_data = fold_data["src_train"]
                tgt_train = fold_data["tgt_train"]
                tgt_test = fold_data["tgt_test"]

                if (
                    src_data is not None
                    and tgt_train is not None
                    and tgt_test is not None
                ):
                    # Combine target train and test
                    tgt_combined_embs = np.vstack(
                        [tgt_train["embeddings"], tgt_test["embeddings"]]
                    )
                    tgt_combined_labels = np.concatenate(
                        [tgt_train["labels"], tgt_test["labels"]]
                    )

                    # Compute comprehensive PCA-based metrics
                    pca_metrics = compute_domain_alignment_metrics_pca(
                        X_source=src_data["embeddings"],
                        X_target=tgt_combined_embs,
                        y_source=src_data["labels"],
                        y_target=tgt_combined_labels,
                        metrics=self.metrics,
                        n_components=self.pca_components,
                        random_state=self.random_state,
                    )

                    comprehensive_metrics["source_target_combined"][
                        f"fold_{fold}"
                    ] = pca_metrics
                    logger.info(
                        f"Fold {fold} - PCA-based metrics computed: {list(pca_metrics.keys())}"
                    )

        return comprehensive_metrics

    def _calculate_knn_distances(
        self, query_emb: np.ndarray, data_subset: Dict[str, Any], k: int
    ) -> Tuple[float, List[Dict]]:
        """
        Calculate k-nearest neighbors distances from query embedding to data subset.

        This helper method computes distances to k nearest neighbors, handling
        cases where the data subset is smaller than k.

        Parameters
        ----------
        query_emb : np.ndarray
            Query embedding (1 x feature_dim).
        data_subset : Dict[str, Any]
            Dictionary containing 'embeddings' and 'keys' for the reference data.
        k : int
            Number of nearest neighbors to find.

        Returns
        -------
        Tuple[float, List[Dict]]
            A tuple containing:
            - Average distance to k nearest neighbors (or -1.0 if no data)
            - List of dictionaries with 'key' and 'distance' for each neighbor
        """
        if len(data_subset["embeddings"]) == 0:
            return -1.0, []

        n_neighbors = min(k, len(data_subset["embeddings"]))
        if n_neighbors == 0:
            return -1.0, []

        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(
            data_subset["embeddings"]
        )
        distances, indices = nn.kneighbors(query_emb)

        avg_dist = float(np.mean(distances))
        neighbor_info = [
            {"key": data_subset["keys"][idx], "distance": float(dist)}
            for dist, idx in zip(distances[0], indices[0])
        ]

        return avg_dist, neighbor_info

    def run_advanced_misclassification_analysis(
        self,
        fold_data: Dict[str, Any],
        pred_analysis: Dict[str, Any],
        fold: int,
        phase: str,
        k: int = 5,
    ) -> None:
        """
        Run comprehensive analysis on misclassified samples.

        This method analyzes why samples were misclassified by examining their
        neighborhoods in the embedding space, computing similarity to correct
        and incorrect classes, and identifying potential outliers.

        Parameters
        ----------
        fold_data : Dict[str, Any]
            Dictionary containing embeddings and labels for all data splits.
        pred_analysis : Dict[str, Any]
            Dictionary containing prediction analysis results including indices
            of correctly and incorrectly classified samples.
        fold : int
            Cross-validation fold number.
        phase : str
            Training phase identifier.
        k : int, optional
            Number of nearest neighbors to consider, by default 5.

        Returns
        -------
        None
            Results are saved to a JSON file in the output directory.
        """
        logger.info(f"Running advanced misclassification analysis for fold {fold}...")

        if "tgt_train" not in fold_data or "tgt_test" not in fold_data:
            logger.warning(
                "Missing target train or test data. Skipping advanced analysis."
            )
            return

        train_data = fold_data["tgt_train"]
        test_data = fold_data["tgt_test"]

        active_train_mask = train_data["labels"] == 1
        train_subsets = {
            "active": {
                "embeddings": train_data["embeddings"][active_train_mask],
                "keys": [
                    k for i, k in enumerate(train_data["keys"]) if active_train_mask[i]
                ],
            },
            "inactive": {
                "embeddings": train_data["embeddings"][~active_train_mask],
                "keys": [
                    k
                    for i, k in enumerate(train_data["keys"])
                    if not active_train_mask[i]
                ],
            },
        }

        correct_indices = pred_analysis["correct_indices"]
        avg_dist_correct_baseline = -1.0
        if correct_indices.size > 0:
            correct_test_embeddings = test_data["embeddings"][correct_indices]
            avg_dist_correct_baseline, _ = self._calculate_knn_distances(
                correct_test_embeddings, train_data, k
            )

        results = {}
        for idx in pred_analysis["incorrect_indices"]:
            query_emb = test_data["embeddings"][idx : idx + 1]
            true_label = test_data["labels"][idx]

            correct_class_key = "active" if true_label == 1 else "inactive"
            incorrect_class_key = "inactive" if true_label == 1 else "active"

            avg_dist_correct_class, neighbors_correct = self._calculate_knn_distances(
                query_emb, train_subsets[correct_class_key], k
            )
            avg_dist_incorrect_class, neighbors_incorrect = (
                self._calculate_knn_distances(
                    query_emb, train_subsets[incorrect_class_key], k
                )
            )

            avg_dist_overall, purity_neighbors_info = self._calculate_knn_distances(
                query_emb, train_data, k
            )
            correct_class_neighbor_keys = set(train_subsets[correct_class_key]["keys"])
            purity_count = sum(
                1
                for n in purity_neighbors_info
                if n["key"] in correct_class_neighbor_keys
            )
            purity_score = purity_count / k if k > 0 else 0

            outlier_score = (
                avg_dist_overall / avg_dist_correct_baseline
                if avg_dist_correct_baseline > 0
                else None
            )

            results[test_data["keys"][idx]] = {
                "true_label": int(true_label),
                "predicted_label": int(1 - true_label),
                "avg_dist_to_correct_class": avg_dist_correct_class,
                "avg_dist_to_counterpart_class": avg_dist_incorrect_class,
                "similarity_ratio": (
                    (avg_dist_correct_class / avg_dist_incorrect_class)
                    if avg_dist_incorrect_class > 0
                    else None
                ),
                "neighborhood_purity_correct_class": purity_score,
                "outlier_score": outlier_score,
                "neighbors_in_correct_class": neighbors_correct,
                "neighbors_in_counterpart_class": neighbors_incorrect,
            }

        final_output = {
            "baseline_avg_knn_dist_for_correct_samples": avg_dist_correct_baseline,
            "misclassified_sample_analysis": results,
        }
        output_path = (
            self.output_dir
            / f"fold_{fold}_{phase}_advanced_misclassification_analysis.json"
        )
        with open(output_path, "w") as f:
            json.dump(final_output, f, indent=2)
        logger.info(f"Saved advanced misclassification analysis to {output_path}")

    # --- PLOTTING FUNCTIONS ---
    def create_source_only_plot(
        self, all_fold_data: Dict[int, Dict[str, Any]], phase: str
    ) -> None:
        """
        Create a 3x2 scatter plot for source domain data across folds.

        This method generates t-SNE visualizations of source domain embeddings
        for up to 5 folds, showing active and inactive classes.

        Parameters
        ----------
        all_fold_data : Dict[int, Dict[str, Any]]
            Dictionary mapping fold numbers to their extracted data.
        phase : str
            Training phase identifier for file naming.

        Returns
        -------
        None
            Plot is saved to the output directory.
        """
        logger.info("Generating scatter plot for source domain only...")
        rows, cols = 3, 2
        fig, axes = plt.subplots(rows, cols, figsize=(18, 24), dpi=600, squeeze=False)
        axes = axes.flatten()

        label_fontsize = self.label_fontsize
        tick_fontsize = self.tick_fontsize
        legend_fontsize = self.label_fontsize  # For source only plot, use same as label
        plot_labels = ["A", "B", "C", "D", "E"]  # Labels for 5 subplots

        colors = self.source_only_colors

        # Reorder legend elements: Active column first, then Inactive column
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Organic Active",
                mfc=self.source_only_colors["src_active"],
                mec="none",
                alpha=0.8,
                ms=self.legend_marker_size,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Organic Inactive",
                mfc=self.source_only_colors["src_inactive"],
                mec="k",
                ms=self.legend_marker_size,
            ),
        ]

        for idx, (fold, fold_data) in enumerate(all_fold_data.items()):
            if idx >= 5:  # Only process first 5 folds
                break

            ax = axes[idx]
            ax.text(
                -0.085,
                1.07,
                f"{plot_labels[idx]})",
                transform=ax.transAxes,
                fontsize=label_fontsize + 2,
                va="top",
                ha="right",
            )

            if "src_train" not in fold_data or not fold_data["src_train"]:
                ax.text(
                    0.5,
                    0.5,
                    "No source data",
                    ha="center",
                    va="center",
                    fontsize=label_fontsize,
                )
                continue

            src_data = fold_data["src_train"]
            tsne_embs = self.tsne_visualizer.fit_transform(src_data["embeddings"])

            # Draw inactive (gray) first so the active (green) points sit on top
            for class_val, status in [(0, "inactive"), (1, "active")]:
                mask = src_data["labels"] == class_val
                if np.any(mask):
                    # Actives slightly transparent with no dark edge so the fluorescent green stays bright
                    is_active = status == "active"
                    ax.scatter(
                        tsne_embs[mask, 0],
                        tsne_embs[mask, 1],
                        c=colors[f"src_{status}"],
                        s=self.marker_sizes["src_only"],
                        marker="o",
                        alpha=0.8 if is_active else self.alpha_values["src_scatter"],
                        edgecolors="none" if is_active else "k",
                        linewidth=self.line_widths["scatter_edge"],
                    )

            ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, width=self.line_widths["axis_tick"])
            ax.set_xlabel("t-SNE 1", fontsize=tick_fontsize)
            ax.set_ylabel("t-SNE 2", fontsize=tick_fontsize)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(self.line_widths["spine"])
            ax.spines["bottom"].set_linewidth(self.line_widths["spine"])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        # Set up the sixth subplot for legend
        legend_ax = axes[5]
        legend_ax.spines["top"].set_visible(False)
        legend_ax.spines["right"].set_visible(False)
        legend_ax.spines["left"].set_visible(False)
        legend_ax.spines["bottom"].set_visible(False)
        # Hide ticks and tick labels
        legend_ax.tick_params(axis="both", which="both", length=0)
        legend_ax.set_xticklabels([])
        legend_ax.set_yticklabels([])
        legend_ax.legend(
            legend_elements,
            [elem.get_label() for elem in legend_elements],
            fontsize=legend_fontsize,
            loc="center",
            ncol=2,
        )

        # Add note about PCA/t-SNE usage
        plt.tight_layout()

        threshold = int(self.params.get('threshold', 0))
        save_path = self.output_dir / f"source_only_{self.affinity_type}_thresh_{threshold}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved source-only scatter plot to {save_path}")

    def create_source_target_combined_plot(
        self, all_fold_data: Dict[int, Dict[str, Any]], phase: str
    ) -> Dict[str, float]:
        """
        Create a 3x2 grid plot showing combined source and target domain data.

        This method generates t-SNE visualizations that display source train,
        target train, and target test data in the same embedding space across
        multiple folds.

        Parameters
        ----------
        all_fold_data : Dict[int, Dict[str, Any]]
            Dictionary mapping fold numbers to their extracted data.
        phase : str
            Training phase identifier for file naming.

        Returns
        -------
        Dict[str, float]
            Source domain separability metrics (JSD and Bhattacharyya distance).
        """
        logger.info("Generating plot for combined source and target domains...")
        rows, cols = 3, 2
        fig, axes = plt.subplots(rows, cols, figsize=(18, 24), dpi=600, squeeze=False)
        axes = axes.flatten()

        label_fontsize = self.label_fontsize
        tick_fontsize = self.tick_fontsize
        legend_fontsize = self.legend_fontsize
        plot_labels = ["A", "B", "C", "D", "E"]  # Labels for 5 subplots

        # Reorder legend elements: Active column first, then Inactive column
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Organic Active",
                mfc=self.scatter_colors["src_active"],
                mec="k",
                ms=self.legend_marker_size,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="Metal-Organic Train Active",
                mfc=self.scatter_colors["tgt_train_active"],
                mec="k",
                ms=self.legend_marker_size,
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                label="Metal-Organic Test Active",
                mfc=self.scatter_colors["tgt_test_active"],
                mec="k",
                ms=self.legend_marker_size,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Organic Inactive",
                mfc=self.scatter_colors["src_inactive"],
                mec="k",
                ms=self.legend_marker_size,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="Metal-Organic Train Inactive",
                mfc=self.scatter_colors["tgt_train_inactive"],
                mec="k",
                ms=self.legend_marker_size,
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                label="Metal-Organic Test Inactive",
                mfc=self.scatter_colors["tgt_test_inactive"],
                mec="k",
                ms=self.legend_marker_size,
            ),
        ]

        # Calculate separability metrics for source domain once (same across all folds)
        source_separability_metrics = {}
        if all_fold_data:
            # Get source data from any fold (it's the same across folds)
            first_fold_data = next(iter(all_fold_data.values()))
            if (
                "src_train" in first_fold_data
                and first_fold_data["src_train"] is not None
            ):
                src_data = first_fold_data["src_train"]
                if len(np.unique(src_data["labels"])) >= 2:
                    # Apply t-SNE to source domain data for separability calculation
                    source_tsne = self.tsne_visualizer.fit_transform(
                        src_data["embeddings"]
                    )
                    source_separability_metrics = self.separability_metrics(
                        source_tsne, src_data["labels"]
                    )
                    logger.info(
                        f"Source domain separability metrics: {source_separability_metrics}"
                    )
                else:
                    logger.warning(
                        "Source domain has only one class, cannot calculate separability metrics"
                    )

        for idx, (fold, fold_data) in enumerate(all_fold_data.items()):
            if idx >= 5:  # Only process first 5 folds
                break

            ax = axes[idx]
            ax.text(
                -0.085,
                1.07,
                f"{plot_labels[idx]})",
                transform=ax.transAxes,
                fontsize=label_fontsize + 2,
                va="top",
                ha="right",
            )

            data_parts, embeddings_list = {}, []
            for key in ["src_train", "tgt_train", "tgt_test"]:
                if key in fold_data and fold_data[key] is not None:
                    data_parts[key] = fold_data[key]
                    embeddings_list.append(fold_data[key]["embeddings"])

            if not embeddings_list:
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    ha="center",
                    va="center",
                    fontsize=label_fontsize,
                )
                continue

            all_embs = np.vstack(embeddings_list)
            tsne_embs = self.tsne_visualizer.fit_transform(all_embs)

            start = 0
            for key in data_parts:
                end = start + len(data_parts[key]["embeddings"])
                data_parts[key]["tsne"] = tsne_embs[start:end]
                start = end

            plot_configs = {
                "src_train": {"marker": "o", "size": self.marker_sizes["src_combined"], "prefix": "src", "alpha": self.alpha_values["src_scatter"]},
                "tgt_train": {
                    "marker": "s",
                    "size": self.marker_sizes["tgt_train_combined"],
                    "prefix": "tgt_train",
                    "alpha": self.alpha_values["tgt_train_combined"],
                },
                "tgt_test": {
                    "marker": "^",
                    "size": self.marker_sizes["tgt_test_combined"],
                    "prefix": "tgt_test",
                    "alpha": self.alpha_values["tgt_test_combined"],
                },
            }

            for key, config in plot_configs.items():
                if key in data_parts:
                    data = data_parts[key]
                    for class_val, status in [(1, "active"), (0, "inactive")]:
                        mask = data["labels"] == class_val
                        if np.any(mask):
                            ax.scatter(
                                data["tsne"][mask, 0],
                                data["tsne"][mask, 1],
                                c=self.scatter_colors[f"{config['prefix']}_{status}"],
                                s=config["size"],
                                marker=config["marker"],
                                alpha=config["alpha"],
                                edgecolors="k",
                                linewidth=self.line_widths["scatter_edge"],
                            )

            ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, width=self.line_widths["axis_tick"])
            if idx >= 4:  # Bottom row (indices 4, 5)
                ax.set_xlabel("t-SNE 1", fontsize=tick_fontsize)
            if idx % cols == 0:  # Left column (indices 0, 2, 4)
                ax.set_ylabel("t-SNE 2", fontsize=tick_fontsize)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(self.line_widths["spine"])
            ax.spines["bottom"].set_linewidth(self.line_widths["spine"])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        # Set up the sixth subplot for legend
        legend_ax = axes[5]
        legend_ax.spines["top"].set_visible(False)
        legend_ax.spines["right"].set_visible(False)
        legend_ax.spines["left"].set_visible(False)
        legend_ax.spines["bottom"].set_visible(False)
        # Hide ticks and tick labels
        legend_ax.tick_params(axis="both", which="both", length=0)
        legend_ax.set_xticklabels([])
        legend_ax.set_yticklabels([])
        legend_ax.legend(
            legend_elements,
            [elem.get_label() for elem in legend_elements],
            fontsize=legend_fontsize,
            loc="center",
            ncol=2,
        )

        # Add note about PCA/t-SNE usage
        plt.tight_layout()

        threshold = int(self.params.get('threshold', 0))
        save_path = self.output_dir / f"source_and_target_{self.affinity_type}_thresh_{threshold}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved combined source-target plot to {save_path}")

        # Return only the source domain separability metrics
        return source_separability_metrics

    def create_source_kde_target_scatter_plot(
        self,
        all_fold_data: Dict[int, Dict[str, Any]],
        phase: str,
        bw_method: float = 0.3,
        n_contour_levels: int = 4,
        density_threshold_pct: float = 0.2,
    ) -> None:
        """
        Create a 3x2 grid plot with source domain as KDE contours and target as scatter.

        Source active and inactive classes are shown as contour outlines computed from
        a 2-D KDE over the joint t-SNE space (source + target reduced together).
        Target train and test points are overlaid as scatter markers.

        Parameters
        ----------
        all_fold_data : Dict[int, Dict[str, Any]]
            Dictionary mapping fold numbers to their extracted data.
        phase : str
            Training phase identifier for file naming.
        bw_method : float, optional
            Bandwidth for ``scipy.stats.gaussian_kde``. Smaller values give sharper,
            less-smoothed contours which can improve separation visibility, by default 0.3.
        n_contour_levels : int, optional
            Number of contour iso-lines drawn per class, by default 4.
        density_threshold_pct : float, optional
            Only draw contours above this fraction of the maximum density value,
            focusing the outlines on the core of each distribution, by default 0.2.

        Returns
        -------
        None
            Plot is saved to the output directory.
        """
        logger.info("Generating KDE (source) + scatter (target) combined plot...")
        rows, cols = 3, 2
        fig, axes = plt.subplots(rows, cols, figsize=(18, 24), dpi=600, squeeze=False)
        axes = axes.flatten()

        label_fontsize = self.label_fontsize
        tick_fontsize = self.tick_fontsize
        legend_fontsize = self.legend_fontsize
        plot_labels = ["A", "B", "C", "D", "E"]

        legend_elements = [
            Line2D([0], [0], color=self.kde_colors["src_active"], linewidth=2,
                   linestyle="-", label="Organic Active (KDE)"),
            Line2D([0], [0], color=self.kde_colors["src_inactive"], linewidth=2,
                   linestyle="--", label="Organic Inactive (KDE)"),
            Line2D([0], [0], marker="s", color="w", label="Metal-Organic Train Active",
                   mfc=self.scatter_colors["tgt_train_active"], mec="k", ms=self.legend_marker_size),
            Line2D([0], [0], marker="^", color="w", label="Metal-Organic Test Active",
                   mfc=self.scatter_colors["tgt_test_active"], mec="k", ms=self.legend_marker_size),
            Line2D([0], [0], marker="s", color="w", label="Metal-Organic Train Inactive",
                   mfc=self.scatter_colors["tgt_train_inactive"], mec="k", ms=self.legend_marker_size),
            Line2D([0], [0], marker="^", color="w", label="Metal-Organic Test Inactive",
                   mfc=self.scatter_colors["tgt_test_inactive"], mec="k", ms=self.legend_marker_size),
        ]

        for idx, (fold, fold_data) in enumerate(all_fold_data.items()):
            if idx >= 5:
                break

            ax = axes[idx]
            ax.text(
                -0.085, 1.07, f"{plot_labels[idx]})",
                transform=ax.transAxes,
                fontsize=label_fontsize + 2,
                va="top", ha="right",
            )

            # Collect all parts for joint t-SNE reduction
            data_parts, embeddings_list = {}, []
            for key in ["src_train", "tgt_train", "tgt_test"]:
                if key in fold_data and fold_data[key] is not None:
                    data_parts[key] = fold_data[key]
                    embeddings_list.append(fold_data[key]["embeddings"])

            if not embeddings_list:
                ax.text(0.5, 0.5, "No data available",
                        ha="center", va="center", fontsize=label_fontsize)
                continue

            # Joint t-SNE (same convention as create_source_target_combined_plot)
            all_embs = np.vstack(embeddings_list)
            tsne_embs = self.tsne_visualizer.fit_transform(all_embs)

            start = 0
            for key in data_parts:
                end = start + len(data_parts[key]["embeddings"])
                data_parts[key]["tsne"] = tsne_embs[start:end]
                start = end

            # --- Source domain: KDE contour outlines ---
            if "src_train" in data_parts:
                src = data_parts["src_train"]
                src_tsne = src["tsne"]
                src_labels = src["labels"]

                xmin, ymin = tsne_embs.min(axis=0)
                xmax, ymax = tsne_embs.max(axis=0)
                grid_size = 120
                xx, yy = np.mgrid[
                    xmin:xmax:complex(0, grid_size),
                    ymin:ymax:complex(0, grid_size),
                ]
                grid_coords = np.vstack([xx.ravel(), yy.ravel()])

                for class_val, status, linestyle in [
                    (1, "src_active", "-"),
                    (0, "src_inactive", "--"),
                ]:
                    mask = src_labels == class_val
                    class_embs = src_tsne[mask]
                    if class_embs.shape[0] < 2:
                        continue
                    try:
                        kde = gaussian_kde(class_embs.T, bw_method=bw_method)
                        density = kde(grid_coords).reshape(xx.shape)
                        # Only draw contours above the density threshold
                        level_min = density.max() * density_threshold_pct
                        level_max = density.max()
                        levels = np.linspace(level_min, level_max, n_contour_levels + 1)[1:]
                        ax.contour(
                            xx, yy, density,
                            levels=levels,
                            colors=self.kde_colors[status],
                            linestyles=linestyle,
                            linewidths=self.line_widths["kde_contour"],
                        )
                    except np.linalg.LinAlgError:
                        logger.warning(
                            f"KDE singular matrix for {status} in fold {fold}. Skipping."
                        )

            # --- Target domain: scatter ---
            scatter_configs = {
                "tgt_train": {"marker": "s", "size": self.marker_sizes["tgt_train_combined"], "prefix": "tgt_train", "alpha": self.alpha_values["tgt_train_combined"]},
                "tgt_test":  {"marker": "^", "size": self.marker_sizes["tgt_test_combined"], "prefix": "tgt_test",  "alpha": self.alpha_values["tgt_test_combined"]},
            }
            for key, config in scatter_configs.items():
                if key in data_parts:
                    data = data_parts[key]
                    for class_val, status in [(1, "active"), (0, "inactive")]:
                        mask = data["labels"] == class_val
                        if np.any(mask):
                            ax.scatter(
                                data["tsne"][mask, 0],
                                data["tsne"][mask, 1],
                                c=self.scatter_colors[f"{config['prefix']}_{status}"],
                                s=config["size"],
                                marker=config["marker"],
                                alpha=config["alpha"],
                                edgecolors="k",
                                linewidth=self.line_widths["scatter_edge"],
                            )

            ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, width=self.line_widths["axis_tick"])
            if idx >= 4:
                ax.set_xlabel("t-SNE 1", fontsize=tick_fontsize)
            if idx % cols == 0:
                ax.set_ylabel("t-SNE 2", fontsize=tick_fontsize)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(self.line_widths["spine"])
            ax.spines["bottom"].set_linewidth(self.line_widths["spine"])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        # Sixth subplot: legend only
        legend_ax = axes[5]
        for spine in legend_ax.spines.values():
            spine.set_visible(False)
        legend_ax.tick_params(axis="both", which="both", length=0)
        legend_ax.set_xticklabels([])
        legend_ax.set_yticklabels([])
        legend_ax.legend(
            legend_elements,
            [e.get_label() for e in legend_elements],
            fontsize=legend_fontsize,
            loc="center",
            ncol=2,
        )

        plt.tight_layout()
        threshold = int(self.params.get('threshold', 0))
        save_path = self.output_dir / f"source_and_target_kde_{self.affinity_type}_thresh_{threshold}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved KDE+scatter plot to {save_path}")

    def create_target_predictions_plot(
        self, all_fold_data: Dict[int, Dict[str, Any]], phase: str
    ) -> None:
        """
        Create a 3x2 plot showing target domain train data and test predictions.

        This method visualizes target domain train data along with test set
        predictions, highlighting true positives, true negatives, false positives,
        and false negatives.

        Parameters
        ----------
        all_fold_data : Dict[int, Dict[str, Any]]
            Dictionary mapping fold numbers to their extracted data.
        phase : str
            Training phase identifier for file naming.

        Returns
        -------
        None
            Plot is saved to the output directory.
        """
        logger.info("Generating plot for target domain with test predictions...")
        rows, cols = 3, 2
        fig, axes = plt.subplots(rows, cols, figsize=(18, 24), dpi=600, squeeze=False)
        axes = axes.flatten()

        label_fontsize = self.label_fontsize
        tick_fontsize = self.tick_fontsize
        legend_fontsize = self.legend_fontsize
        plot_labels = ["A", "B", "C", "D", "E"]  # Labels for 5 subplots

        # Reorder legend elements: Active column first, then Inactive column
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Train Active",
                mfc=self.prediction_colors["train_active"],
                mec="k",
                ms=self.legend_marker_size,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="True Positive",
                mfc=self.prediction_colors["tp"],
                mec="k",
                ms=self.legend_marker_size,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="True Negative",
                mfc=self.prediction_colors["tn"],
                mec="k",
                ms=self.legend_marker_size,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Train Inactive",
                mfc=self.prediction_colors["train_inactive"],
                mec="k",
                ms=self.legend_marker_size,
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color=self.prediction_colors["fp_bg"],
                label="False Positive",
                mec="k",
                mew=self.line_widths["legend_edge"],
                ms=self.legend_marker_size,
                ls="None",
            ),
            Line2D(
                [0],
                [0],
                marker="X",
                color=self.prediction_colors["fn_bg"],
                label="False Negative",
                mec="k",
                mew=self.line_widths["legend_edge"],
                ms=self.legend_marker_size,
                ls="None",
            ),
        ]

        for idx, (fold, fold_data) in enumerate(all_fold_data.items()):
            if idx >= 5:  # Only process first 5 folds
                break

            ax = axes[idx]
            ax.text(
                -0.085,
                1.07,
                f"{plot_labels[idx]})",
                transform=ax.transAxes,
                fontsize=label_fontsize + 2,
                va="top",
                ha="right",
            )

            target_data, embeddings_list = {}, []
            for key in ["tgt_train", "tgt_test"]:
                if key in fold_data and fold_data[key] is not None:
                    target_data[key] = fold_data[key]
                    embeddings_list.append(fold_data[key]["embeddings"])

            if not embeddings_list:
                ax.text(
                    0.5,
                    0.5,
                    "No target data",
                    ha="center",
                    va="center",
                    fontsize=label_fontsize,
                )
                continue

            all_target_embs = np.vstack(embeddings_list)
            tsne_embs = self.tsne_visualizer.fit_transform(all_target_embs)

            start = 0
            for key in target_data:
                end = start + len(target_data[key]["embeddings"])
                target_data[key]["tsne"] = tsne_embs[start:end]
                start = end

            if "tgt_train" in target_data:
                train_data = target_data["tgt_train"]
                for class_val, status in [(1, "active"), (0, "inactive")]:
                    mask = train_data["labels"] == class_val
                    if np.any(mask):
                        ax.scatter(
                            train_data["tsne"][mask, 0],
                            train_data["tsne"][mask, 1],
                            c=self.prediction_colors[f"train_{status}"],
                            s=self.marker_sizes["tgt_train_prediction"],
                            marker="o",
                            alpha=1.0,
                            edgecolors="k",
                            linewidth=self.line_widths["scatter_edge"],
                        )

            if "tgt_test" in target_data:
                test_data = target_data["tgt_test"]
                labels, preds = test_data["labels"], test_data["predictions"]
                masks = {
                    "tp": (labels == 1) & (preds == 1),
                    "tn": (labels == 0) & (preds == 0),
                    "fp": (labels == 0) & (preds == 1),
                    "fn": (labels == 1) & (preds == 0),
                }
                plot_params = {
                    "tp": {"c": self.prediction_colors["tp"], "s": self.marker_sizes["tp"], "marker": "s"},
                    "tn": {"c": self.prediction_colors["tn"], "s": self.marker_sizes["tn"], "marker": "s"},
                    "fp": {
                        "c": self.prediction_colors["fp_bg"],
                        "s": self.marker_sizes["fp"],
                        "marker": "X",
                        "linewidth": self.line_widths["legend_edge"],
                    },
                    "fn": {
                        "c": self.prediction_colors["fn_bg"],
                        "s": self.marker_sizes["fn"],
                        "marker": "X",
                        "linewidth": self.line_widths["legend_edge"],
                    },
                }
                for key, mask in masks.items():
                    if np.any(mask):
                        ax.scatter(
                            test_data["tsne"][mask, 0],
                            test_data["tsne"][mask, 1],
                            edgecolors="k",
                            **plot_params[key],
                        )

            ax.tick_params(axis="both", which="major", labelsize=tick_fontsize, width=self.line_widths["axis_tick"])
            if idx >= 4:  # Bottom row (indices 4, 5)
                ax.set_xlabel("t-SNE 1", fontsize=tick_fontsize)
            if idx % cols == 0:  # Left column (indices 0, 2, 4)
                ax.set_ylabel("t-SNE 2", fontsize=tick_fontsize)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(self.line_widths["spine"])
            ax.spines["bottom"].set_linewidth(self.line_widths["spine"])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        # Set up the sixth subplot for legend
        legend_ax = axes[5]
        legend_ax.spines["top"].set_visible(False)
        legend_ax.spines["right"].set_visible(False)
        legend_ax.spines["left"].set_visible(False)
        legend_ax.spines["bottom"].set_visible(False)
        # Hide ticks and tick labels
        legend_ax.tick_params(axis="both", which="both", length=0)
        legend_ax.set_xticklabels([])
        legend_ax.set_yticklabels([])
        legend_ax.legend(
            legend_elements,
            [elem.get_label() for elem in legend_elements],
            fontsize=legend_fontsize,
            loc="center",
            ncol=2,
        )

        # Add note about PCA/t-SNE usage
        plt.tight_layout()

        threshold = int(self.params.get('threshold', 0))
        save_path = self.output_dir / f"target_only_{self.affinity_type}_thresh_{threshold}.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved target prediction plot to {save_path}")

    def run_analysis_for_all_folds(
        self, n_folds: int, phase: str, with_predictions: bool, source_only: bool = False
    ) -> None:
        """
        Run the full analysis pipeline for all folds and create summary plots.

        This method orchestrates the complete analysis workflow: extracting embeddings,
        analyzing predictions (if enabled), and generating visualizations across
        all specified folds.

        Parameters
        ----------
        n_folds : int
            Number of cross-validation folds to process.
        phase : str
            Training phase identifier.
        with_predictions : bool
            If True, perform prediction analysis and misclassification analysis.
        source_only : bool, optional
            If True, only create the source-only plot and skip all other analysis.

        Returns
        -------
        None
            Results and plots are saved to the output directory.
        """
        logger.info(f"--- Running analysis for all {n_folds} folds, Phase {phase} ---")
        all_fold_data = {}
        for fold in range(1, n_folds + 1):
            try:
                logger.info(f"--- Processing Fold {fold}/{n_folds} ---")
                fold_data = self.extract_all_embeddings(fold, phase)
                all_fold_data[fold] = fold_data

                if with_predictions and not source_only and "tgt_test" in fold_data:
                    logger.info(f"--- Analyzing predictions for Fold {fold} ---")
                    test_data = fold_data["tgt_test"]
                    pred_analysis = self._analyze_predictions(
                        test_data["labels"], test_data["predictions"]
                    )

                    if pred_analysis["incorrect_indices"].size > 0:
                        self.run_advanced_misclassification_analysis(
                            fold_data, pred_analysis, fold, phase
                        )
                    else:
                        logger.info(
                            f"No misclassified samples found for Fold {fold}. Skipping advanced analysis."
                        )

            except Exception as e:
                logger.error(f"Failed to process fold {fold}: {e}", exc_info=True)

        if len(all_fold_data) > 1:
            logger.info("--- Creating final combined plots for all folds ---")
            try:
                self.create_source_only_plot(all_fold_data, phase)
                if source_only:
                    return

                # Compute comprehensive separability metrics for all folds
                comprehensive_metrics = self.compute_comprehensive_separability_metrics(
                    all_fold_data, phase
                )

                if comprehensive_metrics:
                    logger.info(
                        f"Comprehensive separability metrics computed for {len(all_fold_data)} folds"
                    )
                    # Save comprehensive metrics to JSON file
                    try:
                        metrics_path = self.output_dir / "separability_metrics.json"

                        # Convert NumPy types and NaN/Inf to JSON-serializable format
                        def sanitize_for_json(obj):
                            """Recursively convert NumPy types, NaN, and Inf to JSON-serializable types."""
                            import numpy as np

                            if isinstance(obj, dict):
                                return {k: sanitize_for_json(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [sanitize_for_json(item) for item in obj]
                            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                                return int(obj)
                            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                                if np.isnan(obj) or np.isinf(obj):
                                    return None
                                return float(obj)
                            elif isinstance(obj, np.ndarray):
                                return sanitize_for_json(obj.tolist())
                            elif isinstance(obj, float):
                                if np.isnan(obj) or np.isinf(obj):
                                    return None
                                return obj
                            return obj

                        sanitized_metrics = sanitize_for_json(comprehensive_metrics)

                        with open(metrics_path, "w") as f:
                            json.dump(sanitized_metrics, f, indent=2)
                        logger.info(
                            f"Saved comprehensive separability metrics to {metrics_path}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to save separability metrics JSON: {e}")

                # Still create the combined plot (for visualization purposes)
                self.create_source_target_combined_plot(all_fold_data, phase)
                self.create_source_kde_target_scatter_plot(all_fold_data, phase)

                if with_predictions:
                    self.create_target_predictions_plot(all_fold_data, phase)
                else:
                    logger.info(
                        "Skipping prediction plot because --with_predictions was not set."
                    )
            except Exception as e:
                logger.error(
                    f"Failed to create the final combined plots: {e}", exc_info=True
                )


def main() -> None:
    """
    Main execution function for embedding visualization and analysis.

    This function orchestrates the complete pipeline for extracting embeddings from
    trained models, creating t-SNE visualizations, and performing advanced analysis
    on the separation between classes and misclassified samples.

    Parameters
    ----------
    None
        Command-line arguments are parsed from sys.argv.

    Returns
    -------
    None

    Notes
    -----
    The function expects command-line arguments for:
    - model_path: Path to trained model directory
    - output_dir: Directory to save plots and analyses
    - phase: Training phase to visualize
    - inhibitor_descriptor: Type of molecular descriptor
    - biological_descriptor: Type of biological descriptor
    - affinity_type: Type of binding affinity
    - threshold: Binding affinity threshold
    - modality: Feature modalities to use
    - data_scaler: Scaling method
    - n_folds: Number of cross-validation folds
    - fold: Specific fold to visualize (optional)
    - cross_domain/separate_domains: Domain scaling configuration
    - source_smote/target_smote: SMOTE application flags
    - with_predictions: Whether to include prediction visualizations
    - skip_analysis: Whether to skip detailed misclassification analysis
    """
    parser = argparse.ArgumentParser(
        description="Embedding Visualization and Analysis Script"
    )

    # Basic configuration
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save plots and analyses",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="pretrain_ccsa_encoder_task",
        help="Training phase to visualize (e.g., 'ccsa')",
    )
    parser.add_argument(
        "--n_folds", type=int, default=5, help="Total number of folds to process"
    )
    parser.add_argument(
        "--with_predictions",
        action="store_true",
        help="Enable detailed prediction and misclassification analysis",
    )
    parser.add_argument(
        "--recompute_stats",
        action="store_true",
        help="Recompute scaling statistics instead of loading from saved files",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="jsd,mmd,coral,cosine",
        help="Comma-separated list of metrics to compute: jsd,mmd,coral,cosine",
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=30,
        help="Number of PCA components to use for quantitative analysis",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Model hyperparameters
    parser.add_argument(
        "--embed_dim", type=int, default=128, help="Embedding dimension"
    )
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--folds", type=int, default=5, help="Number of cross-validation folds"
    )

    # Model configuration parameters
    parser.add_argument(
        "--threshold", type=float, default=6.0, help="Threshold for prediction"
    )
    parser.add_argument(
        "--inhibitor_descriptor",
        type=str,
        default="SOAP",
        choices=["SOAP", "MACE", "ACSF"],
        help="Molecular descriptor type",
    )
    parser.add_argument(
        "--biological_descriptor",
        type=str,
        default="ESM-C",
        choices=["ESM-C"],
        help="Biological descriptor type",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="molecule",
        choices=[
            "molecule",
            "protein",
            "pocket",
            "molecule_protein",
            "molecule_pocket",
            "protein_pocket",
            "molecule_protein_pocket",
        ],
        help="Model modality",
    )
    parser.add_argument(
        "--affinity_type",
        type=str,
        default="pic50",
        choices=["pic50", "pk"],
        help="Affinity type",
    )
    parser.add_argument(
        "--data_scaler",
        type=str,
        default="minmax",
        choices=["minmax", "standard", "none"],
        help="Data scaling method",
    )
    parser.add_argument(
        "--domain_scaling",
        action="store_true",
        default=True,
        help="Use domain scaling (default: True)",
    )
    parser.add_argument(
        "--cross_domain",
        action="store_true",
        default=True,
        help="Use cross-domain training (default: True)",
    )
    parser.add_argument(
        "--separate_domains",
        action="store_true",
        help="Use separate domain training (overrides --cross_domain)",
    )
    parser.add_argument(
        "--source_smote", action="store_true", help="Use SMOTE for source data"
    )
    parser.add_argument(
        "--target_smote", action="store_true", help="Use SMOTE for target data"
    )
    parser.add_argument(
        "--use_simple_structure",
        action="store_true",
        help="Use simple directory structure (model_path/fold_X/phase/checkpoints) instead of complex nested structure",
    )

    parser.add_argument(
        "--source_only_plot",
        action="store_true",
        help="Only regenerate the source-domain-only plot (skips metrics and other plots)",
    )

    args = parser.parse_args()

    features_dim_dict = {"MACE": 256, "SOAP": 5376, "ACSF": 360, "ESM-C": 1152}

    params = {
        "threshold": args.threshold,
        "inhibitor_descriptor": args.inhibitor_descriptor,
        "biological_descriptor": args.biological_descriptor,
        "affinity_type": args.affinity_type,
        "modality": args.modality,
        "use_mol": "molecule" in args.modality or "mol" in args.modality,
        "use_protein": "protein" in args.modality,
        "use_pocket": "pocket" in args.modality,
        "inhibitor_input_dim": features_dim_dict.get(args.inhibitor_descriptor, 256),
        "biological_input_dim": features_dim_dict.get(args.biological_descriptor, 1152),
        "embed_dim": args.embed_dim,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "n_folds": args.folds,
        "seed": args.seed,
        "data_scaler": args.data_scaler,
        "source_smote": args.source_smote,
        "target_smote": args.target_smote,
        "cross_domain_scaling": args.domain_scaling,
        "visualize": True,
    }

    logger.info(f"Configuration: {params}")
    logger.info(
        f"Training configuration: {build_train_config(args.cross_domain, args.separate_domains, args.source_smote, args.target_smote)}"
    )

    # Parse metrics
    metrics = [m.strip() for m in args.metrics.split(",")]
    logger.info(f"Metrics to compute: {metrics}")

    # Create visualizer with all configuration parameters
    visualizer = EmbeddingVisualizer(
        params=params,
        model_path=args.model_path,
        output_dir=args.output_dir,
        descriptor_type=args.inhibitor_descriptor,
        modality=args.modality,
        scaler_type=args.data_scaler,
        affinity_type=args.affinity_type,
        cross_domain=args.cross_domain,
        separate_domains=args.separate_domains,
        source_smote=args.source_smote,
        target_smote=args.target_smote,
        use_simple_structure=args.use_simple_structure,
        metrics=metrics,
        pca_components=args.pca_components,
        random_state=args.random_state,
        recompute_stats=args.recompute_stats,
    )

    # Run analysis
    visualizer.run_analysis_for_all_folds(
        args.n_folds, args.phase, args.with_predictions, args.source_only_plot
    )

    logger.info("Script finished.")


if __name__ == "__main__":
    main()
