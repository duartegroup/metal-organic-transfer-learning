from console import printer
# plot.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib import font_manager as fm
from typing import Union, List, Tuple
from constants import STRATEGY_MAP, MODALITY_MAP, BASELINE_STRATEGY
from io_utils import atomic_savefig

# Palatino Linotype is the manuscript body font, so figure text matches the paper.
# Override with STATS_PALATINO_DIR when the fonts live elsewhere.
palatino_font_path = os.path.expanduser(
    os.environ.get("STATS_PALATINO_DIR", "~/fonts/font/Serif/Palatino")
)
if not os.path.isdir(palatino_font_path):
    printer.warning(f"Palatino directory not found at {palatino_font_path} — using default font.")
    palatino_font_path = None
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
        # fontset="custom" leaves cal/sf/tt on their defaults, and the default
        # cal family (cursive) is absent here, which logs a findfont warning.
        plt.rcParams["mathtext.cal"] = "Palatino Linotype:italic"
        plt.rcParams["mathtext.sf"] = "Palatino Linotype"
        plt.rcParams["font.family"] = "Palatino Linotype"
        plt.rcParams["text.usetex"] = False
        plt.rcParams["text.latex.preamble"] = r"\\usepackage{mathpazo}"
    except (FileNotFoundError, OSError):
        printer.warning("Palatino font not found. Using default font.")


class PlotGenerator:
    """
    A class to generate various plots for model performance analysis.

    Parameters
    ----------
    threshold : Union[int, str], optional
        The threshold value used to create a unique output directory.
        Use 'global' for cross-threshold analysis, by default 6.
    scoring_function : str, optional
        Scoring function to use for model ranking, by default "mean_mcc_minus_std".
        Options are:
        - 'mean_mcc': Use mean performance
        - 'mean_mcc_minus_std': Use mean minus std (penalizes variance)

    Attributes
    ----------
    threshold : Union[int, str]
        Threshold value for analysis.
    scoring_function : str
        Scoring function used for rankings.
    output_dir : str
        Directory path where plots will be saved.
    STRATEGY_MAP : Dict[str, str]
        Mapping of training strategies to short codes.
    MODALITY_MAP : Dict[str, str]
        Mapping of modality codes to display names.
    axis_label_size : int
        Font size for axis labels.
    tick_label_size : int
        Font size for tick labels.
    title_size : int
        Font size for titles.
    legend_text_size : int
        Font size for legend text.
    legend_title_size : int
        Font size for legend titles.
    """

    def __init__(
        self,
        output_dir: str,
        scoring_function: str = "mean_mcc_minus_std",
    ) -> None:
        self.output_dir = output_dir
        self.scoring_function = scoring_function
        os.makedirs(output_dir, exist_ok=True)

        self.STRATEGY_MAP = STRATEGY_MAP
        self.MODALITY_MAP = MODALITY_MAP

        self.axis_label_size = 26
        self.tick_label_size = 26
        self.title_size = 26
        self.legend_text_size = 26
        self.legend_title_size = 26

    def create_supervised_scatter_plot(
        self,
        df: pd.DataFrame,
        affinity_type: str = "pic50",
        metric: str = "val/mcc",
        figsize: Tuple[int, int] = (24, 22),
        show_all_modalities: bool = True,
    ) -> None:
        """
        Create scatter plots showing mean MCC vs standard deviation across folds.

        This method generates a 3x2 grid of scatter plots, one for each modality,
        showing the trade-off between mean performance and variance across folds
        for supervised learning models.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with supervised learning performance data.
        affinity_type : str, optional
            Type of affinity data ('pic50' or 'pk'), by default "pic50".
        metric : str, optional
            Metric to analyze, by default "val/mcc".
        figsize : Tuple[int, int], optional
            Figure size (width, height), by default (24, 22).
        show_all_modalities : bool, optional
            If True, shows all 6 modalities; if False, shows only 3, by default True.

        Returns
        -------
        None
            Plot is saved to the output directory.
        """
        # === FONT SIZE CONFIGURATION === #
        printer.info(
            f"Creating supervised learning scatter plot for metric: {metric}..."
        )

        if metric not in df.columns:
            printer.warning(f"Warning: Metric column '{metric}' not found in dataframe.")
            return

        # I want to know the number of experiments in the dataframe , so not the folds but the experiments
        df_filtered = df.dropna(subset=[metric]).copy()
        if df_filtered.empty:
            printer.warning(
                f"Warning: No data available for metric '{metric}' after filtering."
            )
            return

        df_stats = (
            df_filtered.groupby(
                [
                    "inhibitor",
                    "modality",
                    "scaler",
                    "method",
                    "target_smote",
                    "pca_status",
                ]
            )[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        printer.info(
            f"Number of experiments in the dataframe: {len(df_filtered['exp_id'].unique())}"
        )

        df_stats = df_stats.rename(columns={"mean": "mean_score", "std": "std_score"})
        df_stats = df_stats[df_stats["count"] >= 2]

        if df_stats.empty:
            printer.warning(
                "Warning: No data with at least 2 folds available for the plot."
            )
            return

        # Define modalities and layout to match transfer learning plot
        # Order: [0,0], [0,1], [0,2], [1,0], [1,1], [1,2]
        modalities = [
            "inhibitor",
            "protein",
            "pocket",
            "inhibitor_protein",
            "inhibitor_pocket",
            "inhibitor_protein_pocket",
        ]
        scalers = ["none_scaler", "standard_scaler", "minmax_scaler"]

        # Add modality mapping for supervised learning (similar to MODALITY_MAP)
        supervised_modality_map = {
            "inhibitor": "Molecule",
            "protein": "Protein",
            "pocket": "Pocket",
            "inhibitor_protein": "Molecule+Protein",
            "inhibitor_pocket": "Molecule+Pocket",
            "inhibitor_protein_pocket": "Molecule+Protein+Pocket",
        }

        n_rows = 3
        n_cols = 2
        panel_labels = ["A)", "B)", "C)", "D)", "E)", "F)"]

        # Create the figure with a 2x3 subplot layout
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize, dpi=600, sharex=True, sharey=True
        )

        # Flatten axes array for easier indexing
        axes_flat = axes.flatten()

        methods = sorted(df_stats["method"].unique())
        method_colors = sns.color_palette("colorblind", n_colors=len(methods))
        method_color_map = dict(zip(methods, method_colors))

        inhibitors = ["ACSF", "MACE", "SOAP"]
        marker_shapes = ["o", "s", "^"]
        inhibitor_marker_map = dict(zip(inhibitors, marker_shapes))

        # Set consistent axis limits similar to transfer learning plot
        x_min, x_max = 0, df_stats["std_score"].max() * 1.1
        y_min, y_max = -0.2, 0.8

        # Create one subplot per modality (following transfer learning pattern)
        for i, modality in enumerate(modalities):
            ax = axes_flat[i]
            ax.text(-0.12, 1.05, panel_labels[i], transform=ax.transAxes, fontsize=26)

            df_subplot = df_stats[df_stats["modality"] == modality]

            if df_subplot.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=16,
                )
            else:
                # Plot data for each scaler, method, and inhibitor combination
                # Arrow from right edge pointing left (toward lower std deviation)
                # Draw arrows first so they appear behind the data points
                # Arrow from right edge pointing left (toward lower std deviation)
                ax.annotate(
                    "",
                    xy=(x_min + 0.05 * (x_max - x_min), y_max - 0.05 * (y_max - y_min)),
                    xytext=(
                        x_max - 0.05 * (x_max - x_min),
                        y_max - 0.05 * (y_max - y_min),
                    ),
                    arrowprops=dict(
                        arrowstyle="-|>,head_length=1.2,head_width=0.5",
                        lw=2,
                        color="black",
                        alpha=0.6,
                    ),
                    zorder=-1,
                )

                # Arrow from bottom edge pointing up (toward higher mean MCC)
                ax.annotate(
                    "",
                    xy=(x_min + 0.05 * (x_max - x_min), y_max - 0.05 * (y_max - y_min)),
                    xytext=(
                        x_min + 0.05 * (x_max - x_min),
                        y_min + 0.05 * (y_max - y_min),
                    ),
                    arrowprops=dict(
                        arrowstyle="-|>,head_length=1.2,head_width=0.5",
                        lw=2,
                        color="black",
                        alpha=0.6,
                    ),
                    zorder=-1,
                )
                for method in methods:
                    for inhibitor in inhibitors:
                        for scaler in scalers:
                            # Plot PCA points (filled)
                            df_pca = df_subplot[
                                (df_subplot["method"] == method)
                                & (df_subplot["inhibitor"] == inhibitor)
                                & (df_subplot["scaler"] == scaler)
                                & (df_subplot["pca_status"].str.contains("with_pca"))
                            ]
                            if not df_pca.empty:
                                ax.scatter(
                                    df_pca["std_score"],
                                    df_pca["mean_score"],
                                    c=[method_color_map[method]],
                                    marker=inhibitor_marker_map[inhibitor],
                                    s=150,
                                    alpha=0.8,
                                    edgecolors="black",
                                    linewidth=0.5,
                                )

                            # Plot no-PCA points (hollow)
                            df_no_pca = df_subplot[
                                (df_subplot["method"] == method)
                                & (df_subplot["inhibitor"] == inhibitor)
                                & (df_subplot["scaler"] == scaler)
                                & (df_subplot["pca_status"] == "no_pca")
                            ]
                            if not df_no_pca.empty:
                                ax.scatter(
                                    df_no_pca["std_score"],
                                    df_no_pca["mean_score"],
                                    c="none",
                                    marker=inhibitor_marker_map[inhibitor],
                                    s=150,
                                    edgecolors=method_color_map[method],
                                    linewidth=1.5,
                                )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            # Use mapped title names like transfer learning plot
            ax.set_title(
                supervised_modality_map.get(modality, modality),
                fontsize=self.title_size,
            )
            ax.set_xlabel(rf"MCC (± $\sigma$)", fontsize=self.axis_label_size)
            if i % n_cols == 0:  # Only label y-axis for leftmost subplots
                ax.set_ylabel(rf"MCC ($\mu$)", fontsize=self.axis_label_size)

            ax.tick_params(labelsize=self.tick_label_size, width=2, length=6)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1.5)
            ax.spines["bottom"].set_linewidth(1.5)

        # Hide unused subplots if showing only 3 modalities
        if not show_all_modalities:
            for i in range(len(modalities), len(axes_flat)):
                axes_flat[i].set_visible(False)

        # Create custom legends with consistent formatting
        method_legend_elements = [
            plt.Line2D([0], [0], color=color, lw=4, label=method)
            for method, color in method_color_map.items()
        ]

        inhibitor_legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="gray",
                linestyle="None",
                markersize=12,
                label=inhibitor,
            )
            for inhibitor, marker in inhibitor_marker_map.items()
        ]

        pca_legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=12,
                label="With PCA",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor="black",
                markersize=12,
                label="No PCA",
                linewidth=1.5,
            ),
        ]

        fig.legend(
            handles=method_legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=len(methods),
            fontsize=self.legend_text_size,
            title_fontsize=self.legend_title_size,
        )

        fig.legend(
            handles=inhibitor_legend_elements,
            loc="lower left",
            bbox_to_anchor=(0.125, 0.05),
            ncol=len(inhibitors),
            fontsize=self.legend_text_size,
            title_fontsize=self.legend_title_size,
        )

        fig.legend(
            handles=pca_legend_elements,
            loc="lower right",
            bbox_to_anchor=(0.9125, 0.05),
            fontsize=self.legend_text_size,
            title_fontsize=self.legend_title_size,
            ncol=2,
        )

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])

        filename = f"supervised_{affinity_type}_{metric.replace('/', '_')}_scatter.png"
        output_path = os.path.join(self.output_dir, filename)
        atomic_savefig(plt, output_path, dpi=600, bbox_inches="tight")
        plt.close()

        printer.info(
            f"Successfully created supervised learning scatter plot: {filename}"
        )

    def create_transfer_learning_scatter_plot(
        self,
        df: pd.DataFrame,
        affinity_type: str = "pic50",
        metric: str = "val/mcc",
        figsize: Tuple[int, int] = (24, 22),
        show_all_modalities: bool = True,
    ) -> None:
        """
        Create a comprehensive scatter plot for transfer learning results.

        This method generates scatter plots showing the trade-off between mean
        performance and variance across folds for transfer learning models,
        organized by modality.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with transfer learning results.
        affinity_type : str, optional
            Type of affinity data ('pic50' or 'pk'), by default "pic50".
        metric : str, optional
            Metric to plot, by default "val/mcc".
        figsize : Tuple[int, int], optional
            Figure size (width, height), by default (24, 22).
        show_all_modalities : bool, optional
            If True, shows all 6 modalities (molecule, protein, pocket,
            molecule+protein, molecule+pocket, molecule+protein+pocket).
            If False, shows only the 3 combined modalities, by default True.

        Returns
        -------
        None
            Plot is saved to the output directory.
        """
        printer.info(f"Creating transfer learning scatter plot for metric: {metric}...")

        if metric not in df.columns:
            printer.warning(f"Warning: Metric column '{metric}' not found.")
            return

        df_filtered = df[df["scaler"] == "minmax_scaler"].dropna(subset=[metric]).copy()
        if df_filtered.empty:
            printer.warning("Warning: No data available for minmax_scaler or metric.")
            return

        # Corrected groupby to include all experimental conditions
        id_vars = [
            "inhibitor",
            "modality",
            "method",
            "pretrain_status",
            "source_smote",
            "target_smote",
            "cross_domain_scaling",
        ]
        present_id_vars = [v for v in id_vars if v in df_filtered.columns]

        df_stats = (
            df_filtered.groupby(present_id_vars)[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        df_stats.rename(
            columns={"mean": "mean_score", "std": "std_score"}, inplace=True
        )
        # Enhanced debugging
        printer.debug(f"=== TL SCATTER DEBUG ({affinity_type}) ===")
        printer.debug(f"Total rows in original df: {len(df)}")
        printer.debug(f"After minmax_scaler filter: {len(df_filtered)}")
        printer.debug(f"Unique experimental combinations: {len(df_stats)}")
        printer.debug(f"COUNT DISTRIBUTION: {df_stats['count'].value_counts().sort_index().to_dict()}")

        # Show which experiments have only 1 fold
        single_fold_experiments = df_stats[df_stats["count"] == 1]
        printer.debug(f"Experiments with only 1 fold: {len(single_fold_experiments)}")
        if len(single_fold_experiments) > 0:
            printer.debug("First 5 single-fold experiments:")
            printer.debug(str(single_fold_experiments[present_id_vars + ["count"]].head()))

        printer.debug(f'After count >= 2 filter: {len(df_stats[df_stats["count"] >= 2])}')
        printer.debug("=" * 50)

        df_stats = df_stats[df_stats["count"] >= 2]

        printer.info(
            f"Number of experiments in the dataframe: {len(df_filtered['exp_id'].unique())}"
        )

        if df_stats.empty:
            printer.warning(
                "Warning: No data with at least 2 folds available after grouping."
            )
            return

        df_stats["strategy"] = df_stats["method"].map(self.STRATEGY_MAP)
        df_stats.dropna(subset=["strategy"], inplace=True)
        printer.debug(f"After strategy mapping: {len(df_stats)} (dropped: {1296 - len(df_stats)})")

        # Check actual coordinate ranges for molecule modality
        df_molecule = df_stats[df_stats["modality"] == "molecule"]
        printer.debug(f"COORDINATE RANGES ({affinity_type}):")
        printer.debug(f'  mean_score range: {df_molecule["mean_score"].min():.3f} to {df_molecule["mean_score"].max():.3f}')
        printer.debug(f'  std_score range: {df_molecule["std_score"].min():.3f} to {df_molecule["std_score"].max():.3f}')
        printer.debug(f'  Zero std_score count: {(df_molecule["std_score"] == 0).sum()}')
        printer.debug(f'  Very small std_score (<0.001): {(df_molecule["std_score"] < 0.001).sum()}')
        printer.debug(f'  Very small std_score (<0.01): {(df_molecule["std_score"] < 0.01).sum()}')

        # printer.info("DataFrame after grouping and processing:")
        # printer.info(df_stats)

        # Define modalities based on the parameter
        if show_all_modalities:
            # All 6 modalities
            modalities = [
                "molecule",
                "protein",
                "pocket",
                "molecule_protein",
                "molecule_pocket",
                "molecule_protein_pocket",
            ]
            n_cols = 2  # 2 rows x 3 columns
            n_rows = 3
            panel_labels = ["A)", "B)", "C)", "D)", "E)", "F)"]
            figsize = (24, 22)  # Larger figure for 6 subplots
        else:
            # Only the 3 combined modalities (current behavior)
            modalities = [
                "molecule_protein",
                "molecule_pocket",
                "molecule_protein_pocket",
            ]
            n_cols = 3
            n_rows = 1
            panel_labels = ["A)", "B)", "C)"]
            figsize = (26, 10)  # Original figure size

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize, dpi=600, sharex=True, sharey=True
        )

        # Flatten axes array for easier indexing
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()

        strategies = sorted(df_stats["strategy"].unique())
        strategy_colors = sns.color_palette("colorblind", n_colors=len(strategies))
        strategy_color_map = dict(zip(strategies, strategy_colors))

        inhibitors = ["ACSF", "MACE", "SOAP"]
        marker_shapes = ["o", "s", "^"]
        inhibitor_marker_map = dict(zip(inhibitors, marker_shapes))

        # Check if std_score has non-NA values before calculating max
        x_min, x_max = 0, df_stats["std_score"].max() * 1.1
        y_min, y_max = -0.2, 0.8

        for i, modality in enumerate(modalities):
            ax = axes_flat[i]
            ax.text(-0.12, 1.05, panel_labels[i], transform=ax.transAxes, fontsize=30)

            df_subplot = df_stats[df_stats["modality"] == modality]

            printer.debug(f"MODALITY {modality}: {len(df_subplot)} experiments")
            if modality == "molecule":  # Debug just the first modality
                points_plotted = 0
                for strategy in strategies:
                    for inhibitor in inhibitors:
                        # Count pretrained points
                        df_pretrain = df_subplot[
                            (df_subplot["strategy"] == strategy)
                            & (df_subplot["inhibitor"] == inhibitor)
                            & (df_subplot["pretrain_status"] == "True")
                        ]
                        if not df_pretrain.empty:
                            points_plotted += len(df_pretrain)
                            printer.debug(f"  Plotting {len(df_pretrain)} pretrained points for {strategy}/{inhibitor}")

                        # Count non-pretrained points
                        df_no_pretrain = df_subplot[
                            (df_subplot["strategy"] == strategy)
                            & (df_subplot["inhibitor"] == inhibitor)
                            & (df_subplot["pretrain_status"] == "False")
                        ]
                        if not df_no_pretrain.empty:
                            points_plotted += len(df_no_pretrain)
                            printer.debug(f"  Plotting {len(df_no_pretrain)} non-pretrained points for {strategy}/{inhibitor}")

                printer.debug(f"  TOTAL POINTS PLOTTED for molecule: {points_plotted}")

            if df_subplot.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=16,
                )
            else:
                ax.annotate(
                    "",
                    xy=(x_min + 0.05 * (x_max - x_min), y_max - 0.05 * (y_max - y_min)),
                    xytext=(
                        x_max - 0.05 * (x_max - x_min),
                        y_max - 0.05 * (y_max - y_min),
                    ),
                    arrowprops=dict(
                        arrowstyle="-|>,head_length=1.2,head_width=0.5",
                        lw=2,
                        color="black",
                        alpha=0.6,
                    ),
                    zorder=-1,
                )

                # Arrow from bottom edge pointing up (toward higher mean MCC)
                ax.annotate(
                    "",
                    xy=(x_min + 0.05 * (x_max - x_min), y_max - 0.05 * (y_max - y_min)),
                    xytext=(
                        x_min + 0.05 * (x_max - x_min),
                        y_min + 0.05 * (y_max - y_min),
                    ),
                    arrowprops=dict(
                        arrowstyle="-|>,head_length=1.2,head_width=0.5",
                        lw=2,
                        color="black",
                        alpha=0.6,
                    ),
                    zorder=-1,
                )

                for strategy in strategies:
                    for inhibitor in inhibitors:
                        # Pretrained (filled)
                        df_pretrain = df_subplot[
                            (df_subplot["strategy"] == strategy)
                            & (df_subplot["inhibitor"] == inhibitor)
                            & (df_subplot["pretrain_status"] == "True")
                        ]
                        if not df_pretrain.empty:
                            ax.scatter(
                                df_pretrain["std_score"],
                                df_pretrain["mean_score"],
                                c=[strategy_color_map[strategy]],
                                marker=inhibitor_marker_map[inhibitor],
                                s=150,
                                alpha=0.8,
                                edgecolors="black",
                                linewidth=0.5,
                            )

                        # Not pretrained (hollow)
                        df_no_pretrain = df_subplot[
                            (df_subplot["strategy"] == strategy)
                            & (df_subplot["inhibitor"] == inhibitor)
                            & (df_subplot["pretrain_status"] == "False")
                        ]
                        if not df_no_pretrain.empty:
                            ax.scatter(
                                df_no_pretrain["std_score"],
                                df_no_pretrain["mean_score"],
                                c="none",
                                marker=inhibitor_marker_map[inhibitor],
                                s=150,
                                edgecolors=strategy_color_map[strategy],
                                linewidth=1.5,
                            )

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(
                self.MODALITY_MAP.get(modality, modality), fontsize=self.title_size
            )
            ax.set_xlabel(rf"MCC (± $\sigma$)", fontsize=self.axis_label_size)
            if i % n_cols == 0:  # Only label y-axis for leftmost subplots
                ax.set_ylabel(rf"MCC ($\mu$)", fontsize=self.axis_label_size)

            ax.tick_params(labelsize=self.tick_label_size, width=2, length=6)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.tick_params(labelsize=self.tick_label_size)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1.5)
            ax.spines["bottom"].set_linewidth(1.5)

        # Hide unused subplots if showing only 3 modalities
        if not show_all_modalities:
            for i in range(len(modalities), len(axes_flat)):
                axes_flat[i].set_visible(False)

        strategy_legend = [
            plt.Line2D([0], [0], color=c, lw=4) for c in strategy_color_map.values()
        ]
        inhibitor_legend = [
            plt.Line2D(
                [0], [0], marker=m, color="gray", linestyle="None", markersize=20
            )
            for m in inhibitor_marker_map.values()
        ]
        pretrain_legend = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=20,
                label="Pretrained",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="none",
                markeredgecolor="gray",
                markersize=20,
                label="No Pretrain",
                mew=1.5,
            ),
        ]

        fig.legend(
            strategy_legend,
            strategy_color_map.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=len(strategies),
            fontsize=self.legend_text_size,
            title_fontsize=self.legend_title_size,
        )  # title="Training strategy",

        fig.legend(
            inhibitor_legend,
            inhibitor_marker_map.keys(),
            loc="lower left",
            bbox_to_anchor=(0.125, 0.05),
            ncol=len(inhibitors),
            fontsize=self.legend_text_size,
            title_fontsize=self.legend_title_size,
        )  #  title="Inhibitor",

        fig.legend(
            handles=pretrain_legend,
            loc="lower right",
            bbox_to_anchor=(0.9125, 0.05),
            ncol=2,
            fontsize=self.legend_text_size,
            title_fontsize=self.legend_title_size,
        )  # title="Pretrain Status",

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])

        # Update filename to indicate modality selection
        modality_suffix = (
            "all_modalities" if show_all_modalities else "combined_modalities"
        )
        filename = f"transfer_learning_{affinity_type}_{metric.replace('/', '_')}_scatter_{modality_suffix}.png"
        output_path = os.path.join(self.output_dir, filename)
        atomic_savefig(plt, output_path, dpi=600, bbox_inches="tight")
        plt.close()

    # ──────────────────────────────────────────────────────────────────────
    # Champion comparison plot  (cross-threshold best SL vs best TL vs Max)
    # ──────────────────────────────────────────────────────────────────────

    def create_champion_comparison_plot(
        self,
        champion_data: dict,
        affinity_types: List[str],
        threshold: str = "",
        metric: str = "val/mcc",
    ) -> None:
        """
        One figure per threshold per metric: two panels (one per affinity type)
        showing per-fold values for the cross-threshold SL champion (B), TL
        champion (TL), and the per-fold maximum across all models (Max).

        Produces separate figures for MCC, Accuracy, and F1.

        Parameters
        ----------
        champion_data : dict
            Keyed by affinity_type; each value contains metric arrays keyed as
            ``sl_mcc``, ``tl_mcc``, ``max_mcc``, ``sl_accuracy``, etc., plus
            ``fold_labels``, ``sl_exp_id``, ``tl_exp_id``. When a baseline arm
            was collected the ``bl_*`` arrays are present too and a fourth bar
            is drawn; without them the chart keeps its original three bars.
        affinity_types : list[str]
            Ordered list of affinity types to plot as panels.
        threshold : str
            Threshold label used in the filename (e.g. ``"5"``).
        metric : str
            Unused — kept for backward-compatibility. All three metrics are
            always plotted.
        """
        import matplotlib.ticker as mticker

        n_panels = sum(1 for a in affinity_types if a in champion_data)
        if n_panels == 0:
            printer.warning("No champion data available — skipping champion comparison plot.")
            return

        panel_labels = ["A)", "B)", "C)"]
        affinity_display = {"pic50": r"pIC$_{50}$", "pk": r"p$K$"}

        metrics_to_plot = [
            ("val/mcc",      "sl_mcc",      "tl_mcc",      "bl_mcc",      "max_mcc",      "MCC",      "mcc"),
            ("val/accuracy", "sl_accuracy", "tl_accuracy", "bl_accuracy", "max_accuracy", "Accuracy", "accuracy"),
            ("val/f1",       "sl_f1",       "tl_f1",       "bl_f1",       "max_f1",       "F1",       "f1"),
        ]

        def _arm(data: dict, key: str, n_folds: int):
            """Return an arm's values, or None when it was never collected."""
            vals = data.get(key)
            if vals is None:
                return None
            vals = np.asarray(vals, dtype=float)
            if vals.size == 0 or np.all(np.isnan(vals)):
                return None
            return vals

        for _metric_key, sl_key, tl_key, bl_key, max_key, metric_label, metric_suffix in metrics_to_plot:
            fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5.5), sharey=True)
            if n_panels == 1:
                axes = [axes]

            panel_idx = 0
            for affinity_type in affinity_types:
                if affinity_type not in champion_data:
                    continue
                data = champion_data[affinity_type]
                n_folds = len(data["fold_labels"])

                sl_vals  = data.get(sl_key,  data.get("sl_vals",  np.full(n_folds, np.nan)))
                tl_vals  = data.get(tl_key,  data.get("tl_vals",  np.full(n_folds, np.nan)))
                max_vals = data.get(max_key, data.get("max_vals", np.full(n_folds, np.nan)))
                bl_vals  = _arm(data, bl_key, n_folds)

                self._plot_champion_panel(
                    ax=axes[panel_idx],
                    sl_vals=sl_vals,
                    tl_vals=tl_vals,
                    bl_vals=bl_vals,
                    max_vals=max_vals,
                    fold_labels=data["fold_labels"],
                    panel_label=panel_labels[panel_idx],
                    title=affinity_display.get(affinity_type, affinity_type),
                    metric_label=metric_label,
                )
                panel_idx += 1

            for ax in axes[1:]:
                ax.set_ylabel("")

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles, labels,
                loc="upper center", ncol=len(labels),
                fontsize=16, frameon=True, fancybox=False, edgecolor="#ccc",
                bbox_to_anchor=(0.5, 1.02),
            )

            thresh_suffix = f"_threshold_{threshold}" if threshold else ""
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            out_png = os.path.join(
                self.output_dir,
                f"fig_champion_comparison{thresh_suffix}_{metric_suffix}.png",
            )
            atomic_savefig(plt, out_png, dpi=300, bbox_inches="tight")
            plt.close()
            printer.info(f"Champion comparison plot saved to: {out_png}")

    def _plot_champion_panel(
        self,
        ax,
        sl_vals: np.ndarray,
        tl_vals: np.ndarray,
        max_vals: np.ndarray,
        fold_labels: List[str],
        panel_label: str,
        title: str,
        metric_label: str = "MCC",
        bl_vals: np.ndarray = None,
    ) -> None:
        """Render one panel of the champion comparison bar chart.

        Draws one bar per arm per fold, plus a Mean column. ``bl_vals`` adds the
        transfer learning baseline strategy as a fourth arm; when it is None the
        panel falls back to the original three bars, at the original width, so
        previously generated figures stay reproducible.
        """
        import matplotlib.ticker as mticker

        n = len(fold_labels)
        x = np.arange(n + 1)

        def _stats(vals):
            vals = np.asarray(vals, dtype=float)
            return float(np.nanmean(vals)), float(np.nanstd(vals))

        # (label, per-fold values, face colour, edge colour, hatch, error bar?)
        arms = [
            ("B", sl_vals, "#5B7FA5", "white", None, True),
        ]
        if bl_vals is not None:
            arms.append(
                (BASELINE_STRATEGY, bl_vals, "#B08A3E", "white", None, True)
            )
        arms.append(("TL", tl_vals, "#6BA368", "white", None, True))
        arms.append(
            ("Max (upper bound)", max_vals, "#E8C8C3", "#C0796E", "//", False)
        )

        # Keeps the three-bar layout at its original 0.25 width and shrinks the
        # bars only when a fourth arm is actually present.
        width = 0.75 / len(arms)
        offsets = (np.arange(len(arms)) - (len(arms) - 1) / 2) * width

        mean_x = x[-1]
        err_kw = dict(
            fmt="none", color="black", lw=1.2, capsize=3, capthick=1.2, zorder=5
        )
        label_specs = []
        # MCC is bounded at -1, not 0. A hard ylim of 0 clipped negative bars off
        # the axis entirely and dropped their labels on top of the tick text.
        observed = []

        for (label, vals, face, edge, hatch, show_err), offset in zip(arms, offsets):
            mean, std = _stats(vals)
            heights = np.append(np.asarray(vals, dtype=float), mean)
            bars = ax.bar(
                x + offset, heights, width,
                color=face, edgecolor=edge,
                linewidth=1.0 if hatch else 0.5,
                hatch=hatch, label=label, zorder=3,
            )
            if show_err:
                ax.errorbar(mean_x + offset, mean, yerr=std, **err_kw)
            # Only the Mean bar carries an error bar, so only it needs clearance
            label_specs.append((bars, [0.0] * n + [std if show_err else 0.0]))
            # Lowest ink this arm puts on the axis: the shortest bar, or the
            # bottom of the Mean bar's error whisker if that reaches lower.
            # Subtracting the error bar from the *shortest* bar instead would
            # drag the axis below zero whenever any bar happened to be short.
            lowest_here = float(np.nanmin(heights))
            if show_err:
                lowest_here = min(lowest_here, mean - std)
            observed.append(lowest_here)

        # Dashed separator between individual folds and the Mean column
        ax.axvline(x=n - 0.5, color="grey", lw=0.8, ls="--", zorder=1)

        # Value labels: above positive bars, below negative ones, so a label
        # never lands on the axis or on the fold tick text.
        fs = 12 if len(arms) == 3 else 10
        for bars, stds in label_specs:
            for bar, std in zip(bars, stds):
                height = bar.get_height()
                if height >= 0:
                    y, va = height + std + 0.012, "bottom"
                else:
                    y, va = height - std - 0.012, "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y,
                    f"{height:.2f}",
                    ha="center", va=va,
                    fontsize=fs, rotation=90,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(fold_labels + ["Mean"], fontsize=14)
        ax.set_ylabel(metric_label, fontsize=17)
        lowest = min([0.0] + [v for v in observed if np.isfinite(v)])
        y_min = 0.0 if lowest >= 0 else float(np.floor((lowest - 0.10) * 10) / 10)
        ax.set_ylim(y_min, 1.1)
        if y_min < 0:
            ax.axhline(0, color="#888", lw=0.8, zorder=2)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.tick_params(axis="y", labelsize=16)
        ax.grid(axis="y", which="major", color="#ddd", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.08, 1.07, panel_label,
            transform=ax.transAxes,
            fontsize=18, va="top",
        )
        ax.set_title(title, fontsize=17, pad=12)
