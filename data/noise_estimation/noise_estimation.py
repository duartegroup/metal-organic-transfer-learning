import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold

from tabulate import tabulate
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import seaborn as sns
import h5py
import warnings

warnings.filterwarnings("ignore")


# Import CompoundH5Reader with safe path handling
try:
    from compound_h5_reader import CompoundH5Reader
except ModuleNotFoundError:
    import sys

    _root_dir = Path(__file__).resolve().parents[2]
    _reader_path = _root_dir / "src" / "data_src"
    sys.path.append(str(_reader_path))
    from compound_h5_reader import CompoundH5Reader


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# === FONT === #
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

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    matthews_corrcoef,
)
from scipy.stats import pearsonr


def load_descriptors_from_hdf5(
    dataset_name: str = "caged",
    affinity_type: str = "pic50",
    base_data_dir: Path = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]:
    """
    Load MACE and protein descriptors from HDF5 file.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., 'caged', 'organic') (default: 'caged').
    affinity_type : str
        Type of affinity data to load (e.g., 'pic50', 'pk') (default: 'pic50').
    base_data_dir : Path
        Base directory for data files. If None, uses relative path from current file.

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float]]
        Tuple containing (mace_descriptors, protein_descriptors, binding_affinities).
    """
    if base_data_dir is None:
        current_file_dir = Path(__file__).parent
        base_data_dir = current_file_dir.parent.parent / "data"

    h5_dir = base_data_dir / "hf5_dataset" / dataset_name
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
    Aggregate descriptors by compound identifier, averaging over replicates.

    Parameters
    ----------
    mace_descriptors : Dict[str, np.ndarray]
        Dictionary mapping compound names to MACE descriptor arrays.
    protein_descriptors : Dict[str, np.ndarray]
        Dictionary mapping compound names to protein descriptor arrays.
    binding_affinities : Dict[str, float]
        Dictionary mapping compound names to binding affinity values.

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float], List[str]]
        Tuple containing (aggregated_mace, aggregated_protein, aggregated_affinities, unique_identifiers).
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


class NoiseEstimator:
    """
    Estimate noise levels in dataset and calculate performance bounds.

    Parameters
    ----------
    dataset : pd.Series or np.ndarray
        The dataset to analyze.
    noise_levels : float, int, or list
        Noise level(s) to simulate.
    predictor_noise_level : float, optional
        Noise level for predictor (default: None, which sets to 0).
    noise_type : str
        Type of noise: 'gaussian', 'uniform', or 'asymmetric' (default: 'gaussian').
    n_bootstrap : int
        Number of bootstrap samples (default: 1000).
    classifier : bool
        Whether to treat as classification task (default: False).
    asym_bound : float, optional
        Boundary for asymmetric noise (default: None).
    asym_noise_up : float, optional
        Noise level above boundary for asymmetric noise (default: None).
    asym_noise_low : float, optional
        Noise level below boundary for asymmetric noise (default: None).
    class_barrier : float, optional
        Classification threshold (default: None, uses median).
    class_labels : np.ndarray, optional
        Pre-computed class labels (default: None).
    """

    def __init__(
        self,
        dataset,
        noise_levels,
        predictor_noise_level=None,
        noise_type="gaussian",
        n_bootstrap=1000,
        classifier=False,
        asym_bound=None,
        asym_noise_up=None,
        asym_noise_low=None,
        class_barrier=None,
        class_labels=None,
    ):
        self.dataset = dataset
        self.noise_levels = noise_levels
        self.noise_type = noise_type
        self.n_bootstrap = n_bootstrap
        self.classifier = classifier
        if self.noise_levels is None:
            raise ValueError("Noise levels must be non-empty")
        elif (
            type(self.noise_levels) == int
            or type(self.noise_levels) == float
            or type(self.noise_levels) == np.float64
        ):
            self.noise_levels = [self.noise_levels]

        self.asym_bound = asym_bound
        self.asym_noise_up = asym_noise_up
        self.asym_noise_low = asym_noise_low

        self.predictor_noise_level = predictor_noise_level
        if predictor_noise_level is None:
            self.predictor_noise_level = 0

        if self.classifier:
            if class_barrier is None:
                self.barrier = np.median(self.dataset)
            else:
                self.barrier = class_barrier
            if class_labels is None:
                self.dataset_labels = np.where(self.dataset >= self.barrier, 1, 0)
            else:
                self.dataset_labels = class_labels

        self.noise_estimates = self.estimate()
        if self.classifier:
            self.noisy_labels = self.noise_estimates.map(
                lambda x: 1 if x >= self.barrier else 0
            )

        self.noise_bootstraps = self.estimate_multi_bootstrap(
            n_bootstrap=self.n_bootstrap
        )

    def estimate(self, noise_levels: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Estimate noisy datasets for given noise levels.

        Parameters
        ----------
        noise_levels : Optional[List[float]]
            List of noise levels to apply. If None, uses self.noise_levels.

        Returns
        -------
        pd.DataFrame
            DataFrame with original data and noisy versions for each noise level.
        """
        output = [self.dataset]
        if self.noise_type == "gaussian":
            estimator = self._estimate_gaussian
        elif self.noise_type == "uniform":
            estimator = self._estimate_uniform
        elif self.noise_type == "asymmetric":
            estimator = self._estimate_asymmetric

        else:
            raise ValueError("Unknown noise type")

        if noise_levels is None:
            noise_levels = self.noise_levels

        for noise_level in noise_levels:
            output.append(estimator(noise_level))

        output_df = pd.DataFrame(output).T
        output_df.columns = ["original"] + [f"noise_{i}" for i in (noise_levels)]
        return output_df

    def estimate_multi_bootstrap(
        self, noise_levels: Optional[List[float]] = None, n_bootstrap: int = 1000
    ) -> pd.DataFrame:
        """
        Estimate noise using multiple bootstrap iterations and calculate performance metrics.

        Parameters
        ----------
        noise_levels : Optional[List[float]]
            List of noise levels to apply. If None, uses self.noise_levels.
        n_bootstrap : int
            Number of bootstrap iterations (default: 1000).

        Returns
        -------
        pd.DataFrame
            DataFrame containing performance metrics for each bootstrap iteration.
        """
        output = []
        if noise_levels is None:
            noise_levels = self.noise_levels

        for noise_level in noise_levels:
            for i in range(n_bootstrap):
                noisy_data = self.estimate(noise_levels=[noise_level])
                # reorder the index to match the original dataset
                noisy_data = noisy_data.reindex(self.dataset.index)
                # for a realistic bound, we need to consider predictor noise
                if self.predictor_noise_level > 0:
                    predictor_data = self.estimate(
                        noise_levels=[self.predictor_noise_level]
                    ).iloc[:, 1]
                # if no predictor noise, use the original dataset for max bound
                else:
                    predictor_data = self.dataset
                if not self.classifier:
                    output.append(
                        [
                            mean_absolute_error(predictor_data, noisy_data.iloc[:, 1]),
                            mean_squared_error(predictor_data, noisy_data.iloc[:, 1]),
                            np.sqrt(
                                mean_squared_error(
                                    predictor_data, noisy_data.iloc[:, 1]
                                )
                            ),
                            r2_score(predictor_data, noisy_data.iloc[:, 1]),
                            pearsonr(predictor_data, noisy_data.iloc[:, 1])[0],
                            f"noise_{noise_level}",
                        ]
                    )
                else:
                    labels_noisy = np.where(noisy_data.iloc[:, 1] > self.barrier, 1, 0)

                    # Calculate additional metrics
                    from sklearn.metrics import accuracy_score, f1_score

                    # Calculate accuracy
                    accuracy = accuracy_score(self.dataset_labels, labels_noisy)

                    # Calculate F1 score (handle cases where there might be only one class)
                    try:
                        f1 = f1_score(self.dataset_labels, labels_noisy)
                    except ValueError:
                        # If there's only one class in either set, F1 is undefined
                        f1 = np.nan

                    output.append(
                        [
                            matthews_corrcoef(self.dataset_labels, labels_noisy),
                            roc_auc_score(self.dataset_labels, labels_noisy),
                            accuracy,
                            f1,
                            f"noise_{noise_level}",
                        ]
                    )

        if self.classifier:
            error_df = pd.DataFrame(
                output,
                columns=["matthews_corrcoef", "roc_auc", "accuracy", "f1", "noise"],
            )
        else:
            error_df = pd.DataFrame(
                output, columns=["mae", "mse", "rmse", "r2", "pearsonr", "noise"]
            )
        return error_df

    def plot(self, noise_df: Optional[pd.DataFrame] = None):
        """
        Plot noise estimation results.

        Parameters
        ----------
        noise_df : Optional[pd.DataFrame]
            DataFrame containing noise estimates. If None, uses self.noise_estimates.

        Returns
        -------
        seaborn.axisgrid.FacetGrid
            Seaborn plot object showing noise effects.
        """
        if not self.classifier:
            if noise_df is None:
                noise_df = self.noise_estimates

            df_melt = noise_df.melt(
                id_vars="original", var_name="noise", value_name="value"
            )
            # add column of 1s for the hue
            df_melt["hue"] = 1
            g = sns.lmplot(
                x="original",
                y="value",
                col="noise",
                data=df_melt,
                aspect=1.0,
                fit_reg=False,
                palette=["black", "red", "blue"],
                hue="hue",
                legend=False,
            )
            for i in range(len(noise_df.columns) - 1):
                g.axes[0, i].set_title(f"Noise: {noise_df.columns[i+1]}")
                g.axes[0, i].set_xlabel("Original")
                g.axes[0, i].set_ylabel("Noisy")

            return g
        else:
            if noise_df is None:
                noise_df = self.noise_estimates

            df_melt = noise_df.melt(
                id_vars="original", var_name="noise", value_name="value"
            )

            noise_melt = self.noisy_labels.melt(
                id_vars="original", var_name="noise", value_name="value"
            )
            # now change the value column to either 'tp/tn' or 'fp' or 'fn' depending on the original label
            noise_melt["value"] = noise_melt.apply(
                lambda row: (
                    "tp/tn"
                    if row["original"] == row["value"]
                    else "fp" if row["original"] == 1 else "fn"
                ),
                axis=1,
            )
            df_melt["classifier"] = noise_melt["value"]
            g = sns.lmplot(
                x="original",
                y="value",
                col="noise",
                hue="classifier",
                data=df_melt,
                aspect=1.0,
                palette=["black", "red", "blue"],
                fit_reg=False,
            )
            for i in range(len(noise_df.columns) - 1):
                g.axes[0, i].set_title(f"Noise: {noise_df.columns[i+1]}")
                g.axes[0, i].set_xlabel("Original")
                g.axes[0, i].set_ylabel("Noisy")
                g.axes[0, i].axvline(self.barrier, color="black", linestyle="--")

            return g

    def plot_bootstrap(
        self, metric: Optional[str] = None, noise_df: Optional[pd.DataFrame] = None
    ):
        """
        Plot bootstrap results for a specific metric.

        Parameters
        ----------
        metric : Optional[str]
            Metric to plot. If None, defaults to 'r2' (default: None).
        noise_df : Optional[pd.DataFrame]
            DataFrame containing bootstrap results. If None, uses self.noise_bootstraps.

        Returns
        -------
        matplotlib.axes.Axes
            Matplotlib axes object with the boxplot.
        """
        if noise_df is None:
            noise_df = self.noise_bootstraps
        if metric is None:
            metric = "r2"

        ax = sns.boxplot(x="noise", y=metric, data=noise_df)
        ax.set_title(f"{metric} for different noise levels")
        ax.set_xlabel("Noise level")
        ax.set_ylabel(metric)

        return ax

    def _estimate_gaussian(self, noise_level: float) -> pd.Series:
        """
        Estimate Gaussian noise.

        Parameters
        ----------
        noise_level : float
            Standard deviation of Gaussian noise to apply.

        Returns
        -------
        pd.Series
            Dataset with added Gaussian noise.
        """
        error = np.random.normal(0, noise_level, len(self.dataset))
        output = self.dataset + error
        return output

    def _estimate_uniform(self, noise_level: float) -> pd.Series:
        """
        Estimate uniform noise (not yet implemented).

        Parameters
        ----------
        noise_level : float
            Range of uniform noise to apply.

        Returns
        -------
        pd.Series
            Dataset with added uniform noise.

        Raises
        ------
        NotImplementedError
            This method is not yet implemented.
        """
        raise NotImplementedError("Uniform noise not implemented yet")

    def _estimate_asymmetric(self, noise_level: float) -> pd.Series:
        """
        Estimate asymmetric noise with different noise levels above and below a boundary.

        Parameters
        ----------
        noise_level : float
            Base noise level for asymmetric noise calculation.

        Returns
        -------
        pd.Series
            Dataset with added asymmetric noise.
        """
        if self.asym_bound is None:
            self.asym_bound = np.median(self.dataset)
        below_boundary = self.dataset[self.dataset <= self.asym_bound]
        above_boundary = self.dataset[self.dataset > self.asym_bound]

        if self.asym_noise_low is None:
            self.asym_noise_low = noise_level * 2
        if self.asym_noise_up is None:
            self.asym_noise_up = noise_level / 2

        error_below = np.random.normal(0, self.asym_noise_low, len(below_boundary))
        error_above = np.random.normal(0, self.asym_noise_up, len(above_boundary))

        below_boundary = below_boundary + error_below
        above_boundary = above_boundary + error_above
        output = pd.concat([below_boundary, above_boundary], axis=0).sort_index()
        return output


class SimpleFoldAnalysis:
    """
    Analyze folds and calculate noise estimation using optimized stratified splits.

    Parameters
    ----------
    n_folds : int
        Number of folds for cross-validation (default: 5).
    base_data_dir : Path
        Base directory for data. If None, uses relative path from current file.
    """

    def __init__(self, n_folds: int = 5, base_data_dir: Optional[Path] = None):
        self.n_folds = n_folds

        if base_data_dir is None:
            current_file_dir = Path(__file__).parent
            self.base_data_dir = current_file_dir.parent.parent / "data"
        else:
            self.base_data_dir = Path(base_data_dir)

    def load_embeddings_and_targets(
        self,
        data_type: str,
        affinity_type: str,
        descriptor_types: List[str] = ["inhibitor", "protein"],
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
        """
        Load embeddings and target values using the same approach as spectral_clustering_split.

        Parameters
        ----------
        data_type : str
            Type of dataset (e.g., 'caged', 'organic').
        affinity_type : str
            Type of affinity data (e.g., 'pic50', 'pk').
        descriptor_types : List[str]
            List of descriptor types to load (default: ['inhibitor', 'protein']).

        Returns
        -------
        Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]
            Tuple containing (features_dict, targets_array, compound_names).
        """
        logger.info(f"Loading data for {data_type} {affinity_type}")

        # Load raw descriptors from HDF5
        mace_descriptors, protein_descriptors, binding_affinities = (
            load_descriptors_from_hdf5(
                dataset_name=data_type,
                affinity_type=affinity_type,
                base_data_dir=self.base_data_dir,
            )
        )

        # Aggregate descriptors by compound identifier (same as spectral clustering)
        agg_mace, agg_protein, agg_affinities, compound_names = (
            aggregate_descriptors_by_identifier(
                mace_descriptors, protein_descriptors, binding_affinities
            )
        )

        logger.info(f"After aggregation: {len(compound_names)} unique compounds")

        # Convert to the format expected by the rest of the pipeline
        features = {}
        if "inhibitor" in descriptor_types:
            # Convert aggregated MACE descriptors to array
            inhibitor_features = []
            for name in compound_names:
                inhibitor_features.append(agg_mace[name])
            features["inhibitor"] = np.array(inhibitor_features)

        if "protein" in descriptor_types:
            # Convert aggregated protein descriptors to array
            protein_features = []
            for name in compound_names:
                protein_features.append(agg_protein[name])
            features["protein"] = np.array(protein_features)

        # Convert targets to array
        targets = np.array([agg_affinities[name] for name in compound_names])

        logging.info(
            f"Loaded {len(compound_names)} samples with {len(descriptor_types)} descriptor types"
        )
        return features, targets, compound_names

    def combine_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine multiple feature types into a single feature matrix by concatenation.

        Parameters
        ----------
        features : Dict[str, np.ndarray]
            Dictionary of feature matrices.

        Returns
        -------
        np.ndarray
            Combined feature matrix.
        """
        feature_arrays = list(features.values())

        if not feature_arrays:
            return np.array([])

        if len(feature_arrays) == 1:
            return feature_arrays[0]

        # Concatenate features
        combined = np.concatenate(feature_arrays, axis=1)
        logger.info(
            f"Combined features with shapes {[f.shape for f in feature_arrays]} into shape {combined.shape}"
        )
        return combined

    def load_optimized_fold_indices(
        self,
        data_type: str,
        affinity_type: str,
        class_barrier: float = 7.0,
    ) -> Dict[int, Dict[str, List[str]]]:
        """
        Load previously saved optimized fold indices.

        Parameters
        ----------
        data_type : str
            Type of dataset (e.g., 'caged', 'organic').
        affinity_type : str
            Type of affinity data (e.g., 'pic50', 'pk').
        class_barrier : float
            Classification threshold (default: 7.0).

        Returns
        -------
        Dict[int, Dict[str, List[str]]]
            Dictionary mapping fold indices to train/test compound names.
        """
        fold_split_dir = (
            self.base_data_dir / "fold_split" / data_type / f"n_folds_{self.n_folds}"
        )

        # Use the correct file naming pattern that matches the actual files
        results_file = (
            fold_split_dir / f"{data_type}_{affinity_type}_optimized_fold_indices.pkl"
        )

        if not results_file.exists():
            raise FileNotFoundError(f"Optimization results not found: {results_file}")

        with open(results_file, "rb") as f:
            fold_indices = pickle.load(f)

        return fold_indices

    def load_spectral_clustering_fold_indices(
        self,
        data_type: str,
        affinity_type: str,
    ) -> Dict[int, Dict[str, List[str]]]:
        """
        Load spectral clustering-based 5-fold indices.

        Parameters
        ----------
        data_type : str
            Type of dataset (e.g., 'caged', 'organic').
        affinity_type : str
            Type of affinity data (e.g., 'pic50', 'pk').

        Returns
        -------
        Dict[int, Dict[str, List[str]]]
            Dictionary mapping fold indices to train/test compound names.

        Notes
        -----
        Expected path: data/fold_split/spectral_clustering_analysis/{data_type}_{affinity_type}_spectral_clustering_fold_indices.pkl
        """
        spectral_dir = (
            self.base_data_dir / "fold_split" / "spectral_clustering_analysis"
        )
        results_file = (
            spectral_dir
            / f"{data_type}_{affinity_type}_spectral_clustering_fold_indices.pkl"
        )

        if not results_file.exists():
            raise FileNotFoundError(
                f"Spectral clustering fold indices not found: {results_file}"
            )

        with open(results_file, "rb") as f:
            fold_indices = pickle.load(f)

        logger.info(
            f"Loaded spectral clustering fold indices from {results_file} with {len(fold_indices)} folds"
        )
        return fold_indices

    @staticmethod
    def _coerce_fold_to_index_lists(
        fold_info: Dict[str, Any], key_to_idx: Dict[str, int], keys: List[str]
    ) -> Tuple[List[int], List[int]]:
        """
        Convert various possible fold formats into index lists over keys.

        Parameters
        ----------
        fold_info : Dict[str, Any]
            Dictionary containing fold information in various formats.
        key_to_idx : Dict[str, int]
            Mapping from compound names to indices.
        keys : List[str]
            List of all compound names.

        Returns
        -------
        Tuple[List[int], List[int]]
            Tuple containing (train_indices, test_indices).

        Notes
        -----
        Supported formats per fold_info:
        - {'train_names': [str], 'test_names': [str]}
        - {'train': [str|int], 'test': [str|int]}
        - {'train_idx'|'train_indices': [int], 'test_idx'|'test_indices': [int]}
        - {'train_names'|'test_names': [int]} assumed to be indices
        """

        def names_to_indices(names: List[Any]) -> List[int]:
            if len(names) == 0:
                return []
            # If already ints, assume they are indices
            if isinstance(names[0], (int, np.integer)):
                return [int(i) for i in names]
            # Else assume strings/names mapping needed
            return [key_to_idx[name] for name in names if name in key_to_idx]

        # Prefer explicit name keys
        if "train_names" in fold_info and "test_names" in fold_info:
            return names_to_indices(fold_info["train_names"]), names_to_indices(
                fold_info["test_names"]
            )

        # Common alternatives
        for t_key, v_key in (
            ("train", "test"),
            ("train_idx", "test_idx"),
            ("train_indices", "test_indices"),
        ):
            if t_key in fold_info and v_key in fold_info:
                return names_to_indices(fold_info[t_key]), names_to_indices(
                    fold_info[v_key]
                )

        # Fallback: try to detect single split with 'train' only, treat rest as test
        if "train" in fold_info and "test" not in fold_info:
            train_idx = names_to_indices(fold_info["train"])
            all_idx = set(range(len(keys)))
            test_idx = sorted(list(all_idx - set(train_idx)))
            return train_idx, test_idx

        raise ValueError(
            "Unsupported fold format; expected keys like train_names/test_names or train/test."
        )

    def run_analysis_with_optimized_splits(
        self,
        data_type: str = "caged",
        affinity_type: str = "pic50",
        descriptor_types: List[str] = ["inhibitor", "protein"],
        class_barrier: float = 7.0,
        classifier: bool = True,
        n_bootstrap: int = 100,
        do_plotting: bool = True,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Run noise analysis using the optimized stratified splits.

        Parameters
        ----------
        data_type : str
            Type of dataset (e.g., 'caged', 'organic') (default: 'caged').
        affinity_type : str
            Type of affinity data (e.g., 'pic50', 'pk') (default: 'pic50').
        descriptor_types : List[str]
            List of descriptor types to use (default: ['inhibitor', 'protein']).
        class_barrier : float
            Classification threshold (default: 7.0).
        classifier : bool
            Whether to run as classifier (default: True).
        n_bootstrap : int
            Number of bootstrap iterations (default: 100).
        do_plotting : bool
            Whether to generate plots (default: True).

        Returns
        -------
        Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
            Tuple containing (train_results, test_results) dictionaries.
        """
        logging.info(f"Running noise analysis for {data_type} {affinity_type}")
        logging.info("=" * 50)

        # Load data using the same method as optimize_stratified_split
        features, targets, keys = self.load_embeddings_and_targets(
            data_type, affinity_type, descriptor_types
        )

        # Combine features if multiple descriptor types
        X = self.combine_features(features)
        logging.info(f"Final features shape: {X.shape}")

        # Load spectral clustering fold indices first; fallback to optimized; then to temporary stratified splits
        try:
            fold_indices = self.load_spectral_clustering_fold_indices(
                data_type, affinity_type
            )
        except FileNotFoundError:
            logging.info(
                "No spectral clustering fold indices found; trying optimized folds..."
            )
            try:
                fold_indices = self.load_optimized_fold_indices(
                    data_type, affinity_type, class_barrier
                )
                logging.info(
                    f"Loaded optimized fold indices with {len(fold_indices)} folds"
                )
            except FileNotFoundError:
                logging.info(
                    "No optimized fold indices found. Creating temporary folds using StratifiedGroupKFold..."
                )

                # Fallback: create temporary groups based on target quantiles
                n_groups = max(self.n_folds, 5)
                temp_groups = pd.qcut(
                    targets, q=n_groups, labels=False, duplicates="drop"
                )

                sgkf = StratifiedGroupKFold(
                    n_splits=self.n_folds, shuffle=True, random_state=1996
                )
                fold_splits = list(
                    sgkf.split(X, (targets > class_barrier).astype(int), temp_groups)
                )

                fold_indices = {}
                for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
                    fold_indices[fold_idx] = {
                        "train_names": [keys[i] for i in train_idx],
                        "test_names": [keys[i] for i in test_idx],
                    }

        # Create key to index mapping
        key_to_idx = {key: idx for idx, key in enumerate(keys)}

        # Process each fold for noise estimation
        all_train_results = {}
        all_test_results = {}

        for fold_idx, fold_info in fold_indices.items():
            logger.info("=" * 50)
            logger.info(f"FOLD {fold_idx+1}")
            logger.info("=" * 50)

            # Convert fold info to index lists
            train_indices, test_indices = self._coerce_fold_to_index_lists(
                fold_info, key_to_idx, keys
            )

            logging.info(f"Train samples: {len(train_indices)}")
            logging.info(f"Test samples: {len(test_indices)}")

            # Create dataframes
            fold_df_train = pd.DataFrame(targets[train_indices], columns=["vals"])
            fold_df_test = pd.DataFrame(targets[test_indices], columns=["vals"])

            logging.info("TRAIN SUMMARY")
            logging.info(f"Count: {len(fold_df_train)}")
            logging.info(f"Mean: {fold_df_train['vals'].mean():.3f}")
            logging.info(f"Std: {fold_df_train['vals'].std():.3f}")
            logging.info(f"Min: {fold_df_train['vals'].min():.3f}")
            logging.info(f"Max: {fold_df_train['vals'].max():.3f}")
            logging.info("TEST SUMMARY")
            logging.info(f"Count: {len(fold_df_test)}")
            logging.info(f"Mean: {fold_df_test['vals'].mean():.3f}")
            logging.info(f"Std: {fold_df_test['vals'].std():.3f}")
            logging.info(f"Min: {fold_df_test['vals'].min():.3f}")
            logging.info(f"Max: {fold_df_test['vals'].max():.3f}")

            # Calculate intrinsic error for training set
            try:
                results_train = self._calculate_and_plot_noise(
                    name=f"Fold_{fold_idx+1}",
                    vals=fold_df_train.vals,
                    affinity_type=affinity_type,
                    average_noise=0.69,
                    n_bootstrap=n_bootstrap,
                    classifier=classifier,
                    class_barrier=class_barrier,
                    fold_type="train",
                    do_plotting=do_plotting,
                )
                all_train_results[f"Fold {fold_idx+1}"] = results_train

            except Exception as e:
                logging.info(f"Error in training fold {fold_idx+1}: {e}")
                all_train_results[f"Fold {fold_idx+1}"] = pd.DataFrame(
                    {"error": [str(e)]}
                )

            # Calculate intrinsic error for test set
            try:
                results_test = self._calculate_and_plot_noise(
                    name=f"Fold_{fold_idx+1}",
                    vals=fold_df_test.vals,
                    affinity_type=affinity_type,
                    average_noise=0.69,
                    n_bootstrap=n_bootstrap,
                    classifier=classifier,
                    class_barrier=class_barrier,
                    fold_type="test",
                    do_plotting=do_plotting,
                )
                all_test_results[f"Fold {fold_idx+1}"] = results_test

            except Exception as e:
                logging.info(f"Error in test fold {fold_idx+1}: {e}")
                all_test_results[f"Fold {fold_idx+1}"] = pd.DataFrame(
                    {"error": [str(e)]}
                )

        logging.info("=" * 50)
        logging.info("TRAIN RESULTS SUMMARY")
        train_summary = []
        for fold, df in all_train_results.items():
            if "error" not in df.columns:
                summary_dict = {"fold": fold}
                summary_dict.update(df.drop(columns="noise").mean().to_dict())
                train_summary.append(summary_dict)
            else:
                train_summary.append({"fold": fold, "error": df["error"].iloc[0]})

        logging.info("\n" + tabulate(train_summary, headers="keys", tablefmt="psql"))

        logging.info("TEST RESULTS SUMMARY")
        test_summary = []
        for fold, df in all_test_results.items():
            if "error" not in df.columns:
                summary_dict = {"fold": fold}
                summary_dict.update(df.drop(columns="noise").mean().to_dict())
                test_summary.append(summary_dict)
            else:
                test_summary.append({"fold": fold, "error": df["error"].iloc[0]})
        logging.info("\n" + tabulate(test_summary, headers="keys", tablefmt="psql"))

        return all_train_results, all_test_results

    def _calculate_and_plot_noise(
        self,
        name: str,
        vals: np.ndarray,
        affinity_type: str,
        average_noise: float = 0.69,
        n_bootstrap: int = 10000,
        classifier: bool = True,
        class_barrier: float = 7.0,
        fold_type: str = "train",
        do_plotting: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate noise estimation and generate plots.

        Parameters
        ----------
        name : str
            Name for the analysis (used in filenames).
        vals : np.ndarray
            Values to analyze.
        affinity_type : str
            Type of affinity data (e.g., 'pic50', 'pk').
        average_noise : float
            Average noise level to apply (default: 0.69).
        n_bootstrap : int
            Number of bootstrap iterations (default: 10000).
        classifier : bool
            Whether to run as classifier (default: True).
        class_barrier : float
            Classification threshold (default: 7.0).
        fold_type : str
            Type of fold ('train' or 'test') (default: 'train').
        do_plotting : bool
            Whether to generate plots (default: True).

        Returns
        -------
        pd.DataFrame
            DataFrame containing bootstrap results with performance metrics.
        """
        if average_noise < 0.69:
            average_noise = 0.69

        # Create a directory for the plots
        base_filename = f"{name.lower()}_{fold_type}"
        plot_dir = Path(
            f"./threshold_{class_barrier}/{affinity_type}/{base_filename}_plots"
        )
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Calculate the noise level
        NE = NoiseEstimator(
            vals,
            average_noise,
            average_noise,
            n_bootstrap=n_bootstrap,
            classifier=classifier,
            class_barrier=class_barrier,
        )

        if do_plotting:
            # Generate and save the noise estimation plot
            init_plot = NE.plot()
            plot_name = f"{base_filename}_noise_estimation.png"
            init_plot.savefig(plot_dir / plot_name, dpi=300)
            logging.info(f"Noise estimation plot saved as {plot_dir / plot_name}")
            plt.close()

            # Generate and save classifier-specific plots
            if classifier:
                mcc_plot = NE.plot_bootstrap(metric="matthews_corrcoef")
                plot_name = f"{base_filename}_mcc.png"
                mcc_plot.figure.savefig(plot_dir / plot_name, dpi=300)
                logging.info(f"MCC plot saved as {plot_dir / plot_name}")
                plt.close()

                # also do the accuracy
                acc_plot = NE.plot_bootstrap(metric="accuracy")
                plot_name = f"{base_filename}_accuracy.png"
                acc_plot.figure.savefig(plot_dir / plot_name, dpi=300)
                logging.info(f"Accuracy plot saved as {plot_dir / plot_name}")
                plt.close()

                f1_plot = NE.plot_bootstrap(metric="f1")
                plot_name = f"{base_filename}_f1.png"
                f1_plot.figure.savefig(plot_dir / plot_name, dpi=300)
                logging.info(f"F1 plot saved as {plot_dir / plot_name}")
                plt.close()

                roc_plot = NE.plot_bootstrap(metric="roc_auc")
                plot_name = f"{base_filename}_roc.png"
                roc_plot.figure.savefig(plot_dir / plot_name, dpi=300)
                logging.info(f"ROC AUC plot saved as {plot_dir / plot_name}")
                plt.close()
            else:
                r2_plot = NE.plot_bootstrap(metric="r2")
                plot_name = f"{base_filename}_r2.png"
                r2_plot.figure.savefig(plot_dir / plot_name, dpi=300)
                logging.info(f"R2 plot saved as {plot_dir / plot_name}")
                plt.close()

                mse_plot = NE.plot_bootstrap(metric="mse")
                plot_name = f"{base_filename}_mse.png"
                mse_plot.figure.savefig(plot_dir / plot_name, dpi=300)
                logging.info(f"MSE plot saved as {plot_dir / plot_name}")
                plt.close()

        # Return the full bootstrap results
        return NE.noise_bootstraps

    def plot_results(
        self,
        train_results: Dict[str, pd.DataFrame],
        test_results: Dict[str, pd.DataFrame],
    ) -> None:
        """
        Plot aggregated results from noise analysis.

        This function creates subplots with MCC, F1, and Accuracy boxplots for train and test sets.

        Parameters
        ----------
        train_results : Dict[str, pd.DataFrame]
            Dictionary mapping fold names to training results DataFrames.
        test_results : Dict[str, pd.DataFrame]
            Dictionary mapping fold names to test results DataFrames.

        Returns
        -------
        None
        """
        logging.info("Generating combined performance metrics plot for all folds.")

        # Combine train results
        all_train_dfs = []
        for fold, df in train_results.items():
            if "error" not in df.columns:
                df_copy = df.copy()
                df_copy["fold"] = fold.split("_")[-1]  # Extract just the fold number
                all_train_dfs.append(df_copy)

        if not all_train_dfs:
            logging.warning("No valid training results to plot.")
            return

        df_train_all = pd.concat(all_train_dfs, ignore_index=True)

        # Combine test results
        all_test_dfs = []
        for fold, df in test_results.items():
            if "error" not in df.columns:
                df_copy = df.copy()
                df_copy["fold"] = fold.split("_")[-1]  # Extract just the fold number
                all_test_dfs.append(df_copy)

        if not all_test_dfs:
            logging.warning("No valid test results to plot.")
            return

        df_test_all = pd.concat(all_test_dfs, ignore_index=True)

        # Set style
        # plt.style.use("seaborn")

        # Create figure with 3 rows (MCC, F1, Accuracy) and 2 columns (Train, Test)
        fig, axes = plt.subplots(3, 2, figsize=(15, 18), dpi=600)

        # Custom color palette
        colors = sns.color_palette("husl", n_colors=len(train_results))

        # Define metrics to plot
        metrics = ["matthews_corrcoef", "f1", "accuracy"]
        metric_labels = ["MCC", "F1", "Accuracy"]

        # Plot each metric
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            # Training set
            sns.boxplot(
                x="fold",
                y=metric,
                data=df_train_all,
                ax=axes[i, 0],
                palette=colors,
                width=0.7,
                linewidth=1.5,
            )
            axes[i, 0].set_ylabel(label, fontsize=16)
            axes[i, 0].set_xlabel("")
            axes[i, 0].grid(True, linestyle="--", alpha=0.7)
            axes[i, 0].text(
                -0.10,
                1.00,
                f"{chr(65 + i*2)})",
                transform=axes[i, 0].transAxes,
                fontsize=18,
            )

            # Add mean values as text for training plot
            for j, fold in enumerate(df_train_all["fold"].unique()):
                mean_val = df_train_all[df_train_all["fold"] == fold][metric].mean()
                # Adjust text position based on metric type
                if metric == "matthews_corrcoef":
                    text_y = 0.97
                elif metric == "f1":
                    text_y = 0.97
                else:  # accuracy
                    text_y = 0.97
                axes[i, 0].text(
                    j,
                    text_y,
                    f"μ={mean_val:.3f}",
                    horizontalalignment="center",
                    verticalalignment="top",
                    fontsize=16,
                )

            # Test set
            sns.boxplot(
                x="fold",
                y=metric,
                data=df_test_all,
                ax=axes[i, 1],
                palette=colors,
                width=0.7,
                linewidth=1.5,
            )
            axes[i, 1].set_ylabel(label, fontsize=16)
            axes[i, 1].set_xlabel("")
            axes[i, 1].grid(True, linestyle="--", alpha=0.7)
            axes[i, 1].text(
                -0.10,
                1.00,
                f"{chr(66 + i*2)})",
                transform=axes[i, 1].transAxes,
                fontsize=18,
            )

            # Add mean values as text for test plot
            for j, fold in enumerate(df_test_all["fold"].unique()):
                mean_val = df_test_all[df_test_all["fold"] == fold][metric].mean()
                # Adjust text position based on metric type
                if metric == "matthews_corrcoef":
                    text_y = 1.09
                elif metric == "f1":
                    text_y = 1.09
                else:  # accuracy
                    text_y = 1.09
                axes[i, 1].text(
                    j,
                    text_y,
                    f"μ={mean_val:.3f}",
                    horizontalalignment="center",
                    verticalalignment="top",
                    fontsize=16,
                )

        # Add column headers
        axes[0, 0].set_title("Training Set", fontsize=18, fontweight="bold")
        axes[0, 1].set_title("Test Set", fontsize=18, fontweight="bold")

        # Adjust layout and style
        plt.tight_layout(pad=3.0)

        # Add subtle border to the figure
        for ax_row in axes:
            for ax in ax_row:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_linewidth(1.5)
                ax.spines["bottom"].set_linewidth(1.5)
                ax.tick_params(labelsize=16)

        plot_filename = "aggregated_performance_metrics_plot.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
        logging.info(f"Aggregated performance metrics plot saved to {plot_filename}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run noise analysis or plot results.")
    subparsers = parser.add_subparsers(
        dest="action", required=True, help="Action to perform"
    )

    # --- Run Parser ---
    run_parser = subparsers.add_parser(
        "run", help="Run the noise analysis and save results."
    )
    run_parser.add_argument(
        "--data_type",
        type=str,
        default="caged",
        help="Dataset type (e.g., caged, organic).",
    )
    run_parser.add_argument(
        "--affinity_type",
        type=str,
        default="pic50",
        help="Affinity type (e.g., pic50, pk).",
    )
    run_parser.add_argument(
        "--descriptor_types",
        type=str,
        default="inhibitor,pocket",
        help="Comma-separated list of descriptor types.",
    )
    run_parser.add_argument(
        "--n_folds", type=int, default=5, help="Number of folds for cross-validation."
    )
    run_parser.add_argument(
        "--threshold", type=float, default=7.0, help="Class barrier for classification."
    )
    run_parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=None,
        help="Multiple thresholds to evaluate (overrides --threshold if provided).",
    )
    run_parser.add_argument(
        "--classifier",
        type=bool,
        default=True,
        help="Whether to run as a classifier or regressor.",
    )
    run_parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap samples for noise estimation.",
    )
    run_parser.add_argument(
        "--output_file",
        type=str,
        default="noise_analysis_results.pkl",
        help="File to save the results.",
    )

    # --- Plot Parser ---
    plot_parser = subparsers.add_parser(
        "plot", help="Plot results from a previous analysis run."
    )
    plot_parser.add_argument(
        "--input_file",
        type=str,
        default="noise_analysis_results.pkl",
        help="File to load results from.",
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = SimpleFoldAnalysis(n_folds=getattr(args, "n_folds", 5))

    if args.action == "run":
        # Convert comma-separated descriptor_types string to list
        descriptor_types_list = [
            desc.strip() for desc in args.descriptor_types.split(",")
        ]

        # Determine thresholds to evaluate
        thresholds_to_evaluate = (
            args.thresholds if args.thresholds is not None else [args.threshold]
        )

        logging.info(f"Evaluating thresholds: {thresholds_to_evaluate}")

        # Run analysis for each threshold
        for threshold in thresholds_to_evaluate:
            logging.info(f"=" * 60)
            logging.info(f"RUNNING ANALYSIS FOR THRESHOLD: {threshold}")
            logging.info(f"=" * 60)

            # Run analysis for current threshold
            train_results, test_results = analyzer.run_analysis_with_optimized_splits(
                data_type=args.data_type,
                affinity_type=args.affinity_type,
                descriptor_types=descriptor_types_list,
                class_barrier=threshold,
                classifier=args.classifier,
                n_bootstrap=args.n_bootstrap,
                do_plotting=True,
            )

            # Save results to a file specific to this threshold
            results_to_save = {
                "train_results": train_results,
                "test_results": test_results,
                "threshold": threshold,
                "args": vars(args),
            }

            # Create output directory for this threshold
            output_dir = Path(f"./threshold_{threshold}/{args.affinity_type}")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / args.output_file
            with open(output_file, "wb") as f:
                pickle.dump(results_to_save, f)
            logging.info(
                f"Analysis complete for threshold {threshold}. Results saved to {output_file}"
            )

        logging.info(f"All analyses complete for thresholds: {thresholds_to_evaluate}")

    elif args.action == "plot":
        # Load results from a file
        with open(
            f"./threshold_{args.threshold}/{args.affinity_type}/{args.input_file}", "rb"
        ) as f:
            saved_results = pickle.load(f)

        logging.info(f"Loaded results from {args.input_file}")
        train_results = saved_results["train_results"]
        test_results = saved_results["test_results"]

        # Display results in the console
        logging.info("--- SAVED TRAIN RESULTS ---")
        logging.info("\n" + tabulate(train_results, headers="keys", tablefmt="psql"))
        logging.info("--- SAVED TEST RESULTS ---")
        logging.info("\n" + tabulate(test_results, headers="keys", tablefmt="psql"))

        # Call the plotting function
        analyzer.plot_results(train_results, test_results)
