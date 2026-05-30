#!/usr/bin/env python3
"""
Script to plot spectral clustering group splits for caged dataset.
"""

import logging
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
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


class SpectralClusteringPlotter:
    """
    Class to plot spectral clustering group splits for caged dataset.
    """

    def __init__(
        self,
        spectral_analysis_dir: str = "./../../data/fold_split/spectral_clustering_analysis",
    ) -> None:
        """
        Initialize the plotter.

        Parameters
        ----------
        spectral_analysis_dir : str, optional
            Path to the spectral clustering analysis directory
            (default: './../../data/fold_split/spectral_clustering_analysis').

        Returns
        -------
        None
        """
        # Set the spectral clustering analysis directory
        self.spectral_analysis_dir = Path(spectral_analysis_dir)

    def _get_spectral_clustering_file_path(
        self, affinity_type: str, file_type: str
    ) -> Path:
        """
        Get the path to spectral clustering files for caged dataset.

        Parameters
        ----------
        affinity_type : str
            Affinity type ('pic50' or 'pk')
        file_type : str
            Type of file ('fold_indices' or 'optimization_results')

        Returns
        -------
        Path
            Path to the requested file
        """
        filename = f"caged_{affinity_type}_spectral_clustering_{file_type}.pkl"
        return self.spectral_analysis_dir / filename

    # ===== DATA LOADING FUNCTIONS =====#
    def load_optimization_results(self, affinity_type: str) -> Dict[str, Any]:
        """
        Load optimization results from pickle files.

        Parameters
        ----------
        affinity_type : str
            Affinity type ('pic50' or 'pk').

        Returns
        -------
        Dict[str, Any]
            Dictionary containing optimization results.
        """
        results_file = self._get_spectral_clustering_file_path(
            affinity_type, "optimization_results"
        )

        if not results_file.exists():
            raise FileNotFoundError(
                f"Optimization results file not found: {results_file}"
            )

        with open(results_file, "rb") as f:
            results = pickle.load(f)
        logger.info(f"Loaded optimization results from {results_file}")
        return results

    def load_fold_indices(self, affinity_type: str) -> Dict:
        """
        Load fold indices from pickle files.

        Parameters
        ----------
        affinity_type : str
            Affinity type ('pic50' or 'pk').

        Returns
        -------
        Dict
            Dictionary containing fold indices.
        """
        fold_file = self._get_spectral_clustering_file_path(
            affinity_type, "fold_indices"
        )

        if not fold_file.exists():
            raise FileNotFoundError(f"Fold indices file not found: {fold_file}")

        with open(fold_file, "rb") as f:
            fold_indices = pickle.load(f)
        logger.info(f"Loaded fold indices from {fold_file}")
        return fold_indices

    def prepare_results_for_plotting(
        self,
        affinity_types: List[str] = ["pic50", "pk"],
    ) -> Tuple[Dict, Dict]:
        """
        Prepare optimization results in the format expected by plotting functions.

        Parameters
        ----------
        affinity_types : List[str], optional
            List of affinity types to prepare (default: ['pic50', 'pk']).

        Returns
        -------
        Tuple[Dict, Dict]
            A tuple containing results dictionary and targets dictionary.
        """
        results = {}
        targets = {}

        for affinity_type in affinity_types:
            # Load optimization results
            opt_results = self.load_optimization_results(affinity_type)

            # Extract data from optimization results
            cluster_labels = opt_results.get("cluster_labels")
            similarity_matrix = opt_results.get(
                "similarity_matrix"
            )  # Get real similarity matrix
            keys = opt_results["keys"]
            target_values = opt_results["targets"]

            # Prepare results in the format expected by plotting functions
            results[affinity_type] = {
                "similarity_matrices": (
                    {"combined": similarity_matrix}
                    if similarity_matrix is not None
                    else {}
                ),
                "labels": cluster_labels,
                "targets": target_values,
                "keys": keys,
            }

            # Prepare targets dictionary
            targets[affinity_type] = {
                key: target_values[i] for i, key in enumerate(keys)
            }

            logger.info(
                f"Prepared {affinity_type} data: {len(keys)} samples, "
                f"{len(np.unique(cluster_labels)) if cluster_labels is not None else 'N/A'} groups"
            )

        return results, targets

    def get_dataset_output_dir(self) -> Path:
        """
        Get the output directory for caged dataset.

        Returns
        -------
        Path
            Path to the output directory.
        """
        dataset_dir = self.spectral_analysis_dir
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    # ===== PLOTTING FUNCTIONS =====
    def plot_group_distributions_and_similarity_matrices(
        self,
        results: Dict,
        targets: Dict,
        n_folds: int = 5,
        save_fig: bool = True,
    ) -> Tuple:
        """
        Create a 2x2 subplot figure visualizing the distributions of target values
        and similarity matrices with cluster boundaries.

        Parameters
        ----------
        results : Dict
            Dictionary containing clustering results for each affinity type.
        targets : Dict
            Dictionary containing target values for each affinity type.
        n_folds : int, optional
            Number of folds (default: 5).
        save_fig : bool, optional
            Whether to save the figure (default: True).

        Returns
        -------
        Tuple
            A tuple containing the figure and axes objects.
        """
        # Font size configuration variables
        LABEL_SIZE = 24
        TICK_SIZE = 24
        MATRIX_ANNOTATION_SIZE = 14
        COLORBAR_SIZE = 24

        # Figure size and DPI
        FIG_WIDTH = 18
        FIG_HEIGHT = 14
        FIG_DPI = 600

        # Extract data for both affinity types
        pic50_data = results["pic50"]
        pk_data = results["pk"]

        # Get target dictionaries
        target_dict_pic50 = targets["pic50"]
        target_dict_pk = targets["pk"]

        # Create cluster assignments dictionaries
        cluster_assignments_pic50 = {
            key: label for key, label in zip(pic50_data["keys"], pic50_data["labels"])
        }
        cluster_assignments_pk = {
            key: label for key, label in zip(pk_data["keys"], pk_data["labels"])
        }

        # Use real similarity matrices instead of creating dummy ones
        similarity_matrix_pic50 = results["pic50"]["similarity_matrices"].get(
            "combined"
        )
        similarity_matrix_pk = results["pk"]["similarity_matrices"].get("combined")

        # If no similarity matrices available, create dummy ones as fallback
        if similarity_matrix_pic50 is None:
            n_samples_pic50 = len(pic50_data["keys"])
            similarity_matrix_pic50 = np.eye(n_samples_pic50) + 0.1 * np.random.rand(
                n_samples_pic50, n_samples_pic50
            )
            similarity_matrix_pic50 = (
                similarity_matrix_pic50 + similarity_matrix_pic50.T
            ) / 2
            logger.warning("Using dummy similarity matrix for pIC50")

        if similarity_matrix_pk is None:
            n_samples_pk = len(pk_data["keys"])
            similarity_matrix_pk = np.eye(n_samples_pk) + 0.1 * np.random.rand(
                n_samples_pk, n_samples_pk
            )
            similarity_matrix_pk = (similarity_matrix_pk + similarity_matrix_pk.T) / 2
            logger.warning("Using dummy similarity matrix for pKi")

        cluster_labels_pic50 = pic50_data["labels"]
        cluster_labels_pk = pk_data["labels"]

        # Create a 2x2 subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=FIG_DPI)

        # Prepare fold data for box plots
        def prepare_fold_data_for_boxplot(target_dict, cluster_assignments):
            unique_clusters = sorted(set(cluster_assignments.values()))
            fold_data = [[] for _ in range(len(unique_clusters))]
            cluster_to_index = {
                cluster: idx for idx, cluster in enumerate(unique_clusters)
            }

            for key, value in target_dict.items():
                if key in cluster_assignments:
                    cluster = cluster_assignments[key]
                    if cluster in cluster_to_index:
                        fold_data[cluster_to_index[cluster]].append(value)
            return fold_data, unique_clusters

        # Get data by cluster/fold
        pic50_fold_data, pic50_unique_clusters = prepare_fold_data_for_boxplot(
            target_dict_pic50, cluster_assignments_pic50
        )
        pk_fold_data, pk_unique_clusters = prepare_fold_data_for_boxplot(
            target_dict_pk, cluster_assignments_pk
        )

        # Use the actual number of groups instead of n_folds
        actual_n_folds_pic50 = len(pic50_unique_clusters)
        actual_n_folds_pk = len(pk_unique_clusters)

        # Reorder matrices by cluster
        def reorder_matrix_by_clusters(
            similarity_matrix, cluster_labels, unique_clusters
        ):
            order = []
            for cluster in unique_clusters:
                order.extend(np.where(cluster_labels == cluster)[0])
            reordered_clusters = cluster_labels[order]
            reordered_matrix = similarity_matrix[order][:, order]
            return reordered_matrix, reordered_clusters

        reordered_matrix_pic50, reordered_clusters_pic50 = reorder_matrix_by_clusters(
            similarity_matrix_pic50, cluster_labels_pic50, pic50_unique_clusters
        )
        reordered_matrix_pk, reordered_clusters_pk = reorder_matrix_by_clusters(
            similarity_matrix_pk, cluster_labels_pk, pk_unique_clusters
        )

        # A) Plot 1: pIC50 distribution
        sns.violinplot(data=pic50_fold_data, ax=axes[0, 0], palette="viridis")
        axes[0, 0].set_ylabel(r"pIC$_{50}$", fontsize=LABEL_SIZE)
        axes[0, 0].set_xticks(range(actual_n_folds_pic50))
        axes[0, 0].set_xticklabels(
            [f"{cluster+1}" for cluster in pic50_unique_clusters],
            fontsize=TICK_SIZE,
        )
        axes[0, 0].tick_params(axis="y", labelsize=TICK_SIZE)
        axes[0, 0].grid(alpha=0.3, axis="y")
        axes[0, 0].text(
            -0.12, 1.0, "A)", transform=axes[0, 0].transAxes, fontsize=LABEL_SIZE + 2
        )

        # B) Plot 2: pIC50 similarity matrix (dummy)
        im0 = axes[0, 1].imshow(reordered_matrix_pic50, cmap="viridis", origin="lower")
        axes[0, 1].set_ylabel("Data Index", fontsize=LABEL_SIZE)
        axes[0, 1].invert_yaxis()
        axes[0, 1].text(
            -0.12, 1.0, "B)", transform=axes[0, 1].transAxes, fontsize=LABEL_SIZE + 2
        )

        cbar0 = fig.colorbar(im0, ax=axes[0, 1], fraction=0.046, pad=0.04)
        cbar0.ax.tick_params(labelsize=COLORBAR_SIZE)

        # Add cluster boundaries for pIC50
        cluster_boundaries_pic50 = np.where(
            reordered_clusters_pic50[:-1] != reordered_clusters_pic50[1:]
        )[0]
        for boundary in cluster_boundaries_pic50:
            axes[0, 1].axhline(y=boundary + 0.5, color="red", linewidth=2)
            axes[0, 1].axvline(x=boundary + 0.5, color="red", linewidth=2)

        axes[0, 1].tick_params(labelsize=TICK_SIZE)

        # C) Plot 3: pK distribution
        sns.violinplot(data=pk_fold_data, ax=axes[1, 0], palette="magma")
        axes[1, 0].set_xlabel("Fold", fontsize=LABEL_SIZE)
        axes[1, 0].set_ylabel(r"pK$_{i}$", fontsize=LABEL_SIZE)
        axes[1, 0].set_xticks(range(actual_n_folds_pk))
        axes[1, 0].set_xticklabels(
            [f"{cluster+1}" for cluster in pk_unique_clusters], fontsize=TICK_SIZE
        )
        axes[1, 0].tick_params(axis="y", labelsize=TICK_SIZE)
        axes[1, 0].grid(alpha=0.3, axis="y")
        axes[1, 0].text(
            -0.12, 1.0, "C)", transform=axes[1, 0].transAxes, fontsize=LABEL_SIZE + 2
        )

        # D) Plot 4: pK similarity matrix (dummy)
        im1 = axes[1, 1].imshow(reordered_matrix_pk, cmap="magma", origin="lower")
        axes[1, 1].set_xlabel("Data Index", fontsize=LABEL_SIZE)
        axes[1, 1].set_ylabel("Data Index", fontsize=LABEL_SIZE)
        axes[1, 1].invert_yaxis()
        axes[1, 1].text(
            -0.12, 1.0, "D)", transform=axes[1, 1].transAxes, fontsize=LABEL_SIZE + 2
        )

        cbar1 = fig.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=COLORBAR_SIZE)

        # Add cluster boundaries for pK
        cluster_boundaries_pk = np.where(
            reordered_clusters_pk[:-1] != reordered_clusters_pk[1:]
        )[0]
        for boundary in cluster_boundaries_pk:
            axes[1, 1].axhline(y=boundary + 0.5, color="red", linewidth=2)
            axes[1, 1].axvline(x=boundary + 0.5, color="red", linewidth=2)

        axes[1, 1].tick_params(labelsize=TICK_SIZE)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_fig:
            output_dir = self.get_dataset_output_dir()
            filename = "caged_spectral_clustering_fold_dist_and_sim.png"
            output_path = output_dir / filename
            plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
            logger.info(f"Saved clustering visualization to: {output_path}")

        return fig, axes

    def plot_train_test_distributions(
        self,
        affinity_type: str,
        results: Dict,
        targets: Dict,
        fold_indices: List,
        save_fig: bool = True,
    ) -> None:
        """
        Create train/test distribution plots.

        Parameters
        ----------
        affinity_type : str
            Affinity type ('pic50' or 'pk').
        results : Dict
            Dictionary containing clustering results.
        targets : Dict
            Dictionary containing target values.
        fold_indices : List
            List of fold index dictionaries.
        save_fig : bool, optional
            Whether to save the figure (default: True).

        Returns
        -------
        None
        """
        # Get the data for this affinity type
        affinity_results = results[affinity_type]
        affinity_fold_indices = fold_indices

        # Extract data
        keys = affinity_results["keys"]
        target_values = affinity_results["targets"]

        # Determine the number of folds
        n_folds = len(affinity_fold_indices)

        # Set plot parameters
        figsize = (16, 12)
        dpi = 300
        palette = "viridis"
        stat_x = 0.0

        # Set appropriate y_label based on affinity type
        y_label = r"pIC$_{50}$" if affinity_type == "pic50" else r"pK$_{i}$"

        # Create dataframes for train and test sets
        train_data = []
        test_data = []

        # Use the actual fold assignments from the optimization
        for fold_idx in range(n_folds):
            fold_data = affinity_fold_indices[fold_idx]
            train_names = set(fold_data["train_names"])
            test_names = set(fold_data["test_names"])

            # Create train/test target lists for this fold
            for i, key in enumerate(keys):
                if key in train_names:
                    train_data.append(
                        {"Fold": f"Fold {fold_idx+1}", "Target Value": target_values[i]}
                    )
                elif key in test_names:
                    test_data.append(
                        {"Fold": f"Fold {fold_idx+1}", "Target Value": target_values[i]}
                    )

        # Convert to DataFrames
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True)

        # Function to create violin plot
        def create_violin_plot(ax, df, split_type, color_offset=0):
            if df.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"No {split_type.lower()} data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=16,
                )
                return

            sns.violinplot(
                x="Fold",
                y="Target Value",
                data=df,
                palette=[
                    plt.cm.get_cmap(palette)(i / n_folds + color_offset)
                    for i in range(n_folds)
                ],
                inner="quartile",
                ax=ax,
            )

            sns.boxplot(
                x="Fold",
                y="Target Value",
                data=df,
                width=0.15,
                color="white",
                boxprops={"zorder": 2},
                ax=ax,
            )

            ax.text(
                0.01,
                0.95,
                split_type,
                transform=ax.transAxes,
                fontsize=20,
                fontweight="bold",
                va="top",
                ha="left",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3),
            )

            # Add statistical annotations
            # Calculate dynamic annotation position based on data range
            y_min, y_max = ax.get_ylim()
            # Extend y-axis slightly to make room for annotations
            ax.set_ylim(y_min, y_max + 0.15 * (y_max - y_min))
            # Update y_max after extending the range
            _, y_max_extended = ax.get_ylim()
            annotation_y = y_max + 0.05 * (
                y_max - y_min
            )  # Position in the extended space

            for i, fold_name in enumerate(sorted(df["Fold"].unique())):
                fold_values = df[df["Fold"] == fold_name]["Target Value"]
                if len(fold_values) > 0:  # Check if fold has data
                    fold_mean = fold_values.mean()
                    fold_std = fold_values.std()
                    count = len(fold_values)

                    ax.annotate(
                        f"μ={fold_mean:.2f}\nσ={fold_std:.2f}\nn={count}",
                        xy=(i + stat_x, annotation_y),
                        ha="center",
                        va="bottom",
                        fontsize=14,
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            edgecolor="gray",
                            alpha=0.9,
                        ),
                        zorder=10,
                    )  # Ensure annotations are on top

            ax.set_ylabel(f"{y_label}", fontsize=18, labelpad=10)
            ax.tick_params(axis="both", labelsize=18)
            ax.grid(alpha=0.3, axis="y")

        # Create the two plots
        ax1.text(-0.05, 1.0, "A)", transform=ax1.transAxes, fontsize=20)
        ax2.text(-0.05, 1.0, "B)", transform=ax2.transAxes, fontsize=20)
        create_violin_plot(ax1, train_df, "Train", color_offset=0)
        create_violin_plot(ax2, test_df, "Test", color_offset=0.5)

        # do not show the top bar and the right bar of both subplots
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        ax2.set_xlabel("Cross-Validation Fold", fontsize=18, labelpad=10)
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        if save_fig:
            output_dir = self.get_dataset_output_dir()
            filename = (
                f"caged_{affinity_type}_spectral_clustering_train_test_distribution.png"
            )
            output_path = output_dir / filename
            plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved train/test distribution plot to: {output_path}")

    # ===== RUN PLOT FUNCTIONS =====#
    def run_plot_group_distributions(
        self,
        affinity_types: List[str] = ["pic50", "pk"],
        save_plots: bool = True,
    ) -> None:
        """
        Create cluster distribution plots using spectral clustering splits.

        Parameters
        ----------
        affinity_types : List[str], optional
            List of affinity types to plot (default: ['pic50', 'pk']).
        save_plots : bool, optional
            Whether to save the plots (default: True).

        Returns
        -------
        None
        """
        logger.info(
            "Creating spectral clustering group distribution plots for caged dataset"
        )

        # Prepare data
        results, targets = self.prepare_results_for_plotting(affinity_types)

        # Create the visualization
        if len(affinity_types) == 2:
            try:
                fig, axes = self.plot_group_distributions_and_similarity_matrices(
                    results=results,
                    targets=targets,
                    n_folds=len(np.unique(results[affinity_types[0]]["labels"])),
                    save_fig=save_plots,
                )

            except Exception as e:
                logger.error(f"Error creating cluster distribution plot: {str(e)}")
        else:
            logger.warning(
                "Cluster distribution plot requires exactly 2 affinity types"
            )

    def run_plot_train_test_distributions(
        self,
        affinity_types: List[str] = ["pic50", "pk"],
        save_plots: bool = True,
    ) -> None:
        """
        Create train/test distribution plots using spectral clustering splits.

        Parameters
        ----------
        affinity_types : List[str], optional
            List of affinity types to plot (default: ['pic50', 'pk']).
        save_plots : bool, optional
            Whether to save the plots (default: True).

        Returns
        -------
        None
        """
        logger.info(
            "Creating spectral clustering train/test distribution plots for caged dataset"
        )

        # Prepare data
        results, targets = self.prepare_results_for_plotting(affinity_types)

        # Load fold indices
        fold_indices = {}
        for affinity_type in affinity_types:
            try:
                fold_indices[affinity_type] = self.load_fold_indices(affinity_type)
            except Exception as e:
                logger.error(
                    f"Error loading fold indices for {affinity_type}: {str(e)}"
                )
                return

        for affinity_type in affinity_types:
            self.plot_train_test_distributions(
                affinity_type=affinity_type,
                results=results,
                targets=targets,
                fold_indices=fold_indices[affinity_type],
                save_fig=save_plots,
            )

    # ===== MAIN FUNCTION =====#
    def create_all_plots(
        self,
        affinity_types: List[str] = ["pic50", "pk"],
        save_plots: bool = True,
    ) -> None:
        """
        Create all available plots for the spectral clustering splits.

        Parameters
        ----------
        affinity_types : List[str], optional
            List of affinity types to plot (default: ['pic50', 'pk']).
        save_plots : bool, optional
            Whether to save the plots (default: True).

        Returns
        -------
        None
        """
        logger.info("Creating all plots for caged dataset")

        # 1. Group distribution and similarity plots
        logger.info("=" * 70)
        logger.info("1. Creating group distribution and similarity plots...")
        logger.info("=" * 70)
        self.run_plot_group_distributions(affinity_types, save_plots)

        # 2. Train/test distribution plots
        logger.info("=" * 70)
        logger.info("2. Creating train/test distribution plots...")
        logger.info("=" * 70)
        self.run_plot_train_test_distributions(affinity_types, save_plots)

        logger.info("=" * 50)
        logger.info("Completed all plots for caged dataset")


def main() -> None:
    """
    Main function to create plots for spectral clustering group splits.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Create plots for spectral clustering group splits for caged dataset"
    )

    parser.add_argument(
        "--affinity_type",
        type=str,
        choices=["pic50", "pk", "both"],
        default="both",
        help="Affinity type to plot (default: both)",
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=["all", "distributions", "train_test"],
        default="all",
        help="Type of plots to create (default: all)",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        default=True,
        help="Save plots to files (default: True)",
    )
    parser.add_argument(
        "--no_save", action="store_true", help="Don't save plots to files"
    )
    parser.add_argument(
        "--spectral_analysis_dir",
        type=str,
        default="./../../data/fold_split/spectral_clustering_analysis",
        help="Path to the spectral clustering analysis directory",
    )

    args = parser.parse_args()

    # Handle save_plots argument
    if args.no_save:
        args.save_plots = False

    # Determine affinity types to process
    affinity_types = (
        ["pic50", "pk"] if args.affinity_type == "both" else [args.affinity_type]
    )

    # Create plotter
    plotter = SpectralClusteringPlotter(
        spectral_analysis_dir=args.spectral_analysis_dir
    )

    logger.info("Processing caged dataset...")

    try:
        if args.plot_type == "all":
            plotter.create_all_plots(affinity_types, args.save_plots)
        elif args.plot_type == "distributions":
            plotter.run_plot_group_distributions(affinity_types, args.save_plots)
        elif args.plot_type == "train_test":
            plotter.run_plot_train_test_distributions(affinity_types, args.save_plots)

    except Exception as e:
        logger.error(f"Error processing caged dataset: {str(e)}")

    logger.info("Plotting completed!")


if __name__ == "__main__":
    main()
