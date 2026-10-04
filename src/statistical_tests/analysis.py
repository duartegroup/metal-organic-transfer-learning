# analysis.py
import json
import os
import pandas as pd
from typing import Dict, Any, List, Optional
from statistical_tests import ModelRanker
from statistical_tests import (
    perform_wilcoxon_test,
    perform_friedman_test,
    perform_pairwise_wilcoxon,
)
from reporting import Reporter
from plot import PlotGenerator
from console import (
    printer,
    print_phase_banner,
    print_analysis_context,
    print_two_group_result,
    print_multi_group_result,
)
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import numpy as np

from constants import BASELINE_METHOD, BASELINE_STRATEGY




def calculate_ranking_score(mean: float, std: float, scoring_function: str) -> float:
    """
    Calculate ranking score for model performance based on specified scoring function.

    Parameters
    ----------
    mean : float
        Mean performance metric value across folds.
    std : float
        Standard deviation of performance metric across folds.
    scoring_function : str
        Scoring function to use. Options:
        - 'mean_mcc': Use only the mean value
        - 'mean_mcc_minus_std': Use mean minus standard deviation (penalizes variance)

    Returns
    -------
    float
        Computed ranking score.

    Raises
    ------
    ValueError
        If an unknown scoring function is specified.
    """
    if scoring_function == "mean_mcc":
        return mean
    elif scoring_function == "mean_mcc_minus_std":
        return mean - std
    else:
        raise ValueError(f"Unknown scoring function: {scoring_function}")


# ---------------------------------------------------------------------------
# exp_id parsers
# ---------------------------------------------------------------------------

_TL_INHIBITORS   = ("ACSF", "MACE", "SOAP")
_TL_SCALERS      = ("minmax_scaler", "standard_scaler", "none_scaler")
_TL_CROSS_DOMAIN = ("separate_domain", "cross_domain")
_TL_SOURCE_SMOTE = ("no_source_smote", "source_smote")
_TL_TARGET_SMOTE = ("no_target_smote", "target_smote")

_SL_INHIBITORS   = ("ACSF", "MACE", "SOAP")
_SL_SCALERS      = ("minmax_scaler", "standard_scaler", "none_scaler")
_SL_PCA          = ("with_pca_50components", "no_pca")
_SL_TARGET_SMOTE = ("no_target_smote", "target_smote")


def _strip_suffix(s: str, candidates: tuple) -> tuple[str, str]:
    """Remove the first matching suffix from s and return (remainder, matched)."""
    for c in candidates:
        if s.endswith("_" + c):
            return s[:-(len(c) + 1)], c
    raise ValueError(f"No known suffix from {candidates} found in '{s}'")


def _strip_prefix(s: str, candidates: tuple) -> tuple[str, str]:
    """Remove the first matching prefix from s and return (remainder, matched)."""
    for c in candidates:
        if s.startswith(c + "_"):
            return s[len(c) + 1:], c
    raise ValueError(f"No known prefix from {candidates} found in '{s}'")


def _split_on_scaler(s: str, candidates: tuple, exp_id: str) -> tuple[str, str, str]:
    """Split an exp_id remainder into (modality, method, scaler).

    Matching is case-insensitive because the scaler column is normalised with
    ``str()``, which turns a missing scaler into "None" -- so the token in the
    exp_id reads "None_scaler" while the candidate list is lowercase. A
    case-sensitive test silently fails to parse every unscaled model.
    """
    lowered = s.lower()
    for sc in candidates:
        token = f"_{sc}_"
        if token in lowered:
            idx = lowered.index(token)
            return s[:idx], s[idx + len(token):], sc
        if lowered.endswith(f"_{sc}"):
            return s[:len(s) - len(sc) - 1], "", sc
    raise ValueError(f"Could not find scaler token in '{s}' for exp_id '{exp_id}'")


def parse_tl_exp_id(exp_id: str) -> Dict[str, Any]:
    """Parse a TL exp_id string into a structured configuration dictionary.

    The exp_id encodes fields in order:
    ``{inhibitor}_{modality}_{scaler}_{method}_{cross_domain}_{source_smote}_{target_smote}``

    Returns a dict with keys: inhibitor, modality, scaler, method, phase,
    cross_domain, source_smote, target_smote, and derived shell-script values
    (scaler_flag, cross_domain_flag, source_smote_flag, target_smote_flag).
    """
    s = exp_id
    s, target_smote  = _strip_suffix(s, _TL_TARGET_SMOTE)
    s, source_smote  = _strip_suffix(s, _TL_SOURCE_SMOTE)
    s, cross_domain  = _strip_suffix(s, _TL_CROSS_DOMAIN)
    s, inhibitor     = _strip_prefix(s, _TL_INHIBITORS)

    # Find scaler — it always ends with "_scaler"
    modality, method, scaler = _split_on_scaler(s, _TL_SCALERS, exp_id)

    # phase = method name as used in checkpoint directories ("then_" is dropped)
    phase = method.replace("_then_", "_")

    return {
        "exp_id":            exp_id,
        "inhibitor":         inhibitor,
        "modality":          modality,
        "scaler":            scaler.replace("_scaler", ""),   # minmax / standard / none
        "method":            method,
        "phase":             phase,
        "cross_domain":      cross_domain == "cross_domain",
        "source_smote":      source_smote == "source_smote",
        "target_smote":      target_smote == "target_smote",
        # Ready-to-use shell flags
        "cross_domain_flag": "--cross_domain" if cross_domain == "cross_domain" else "--separate_domains",
        "source_smote_flag": "--source_smote" if source_smote == "source_smote" else "",
        "target_smote_flag": "--target_smote" if target_smote == "target_smote" else "",
    }


def parse_sl_exp_id(exp_id: str) -> Dict[str, Any]:
    """Parse an SL exp_id string into a structured configuration dictionary.

    The exp_id encodes fields in order:
    ``{inhibitor}_{modality}_{scaler}_{method}_{target_smote}_{pca_status}``
    """
    s = exp_id
    s, pca_status   = _strip_suffix(s, _SL_PCA)
    s, target_smote = _strip_suffix(s, _SL_TARGET_SMOTE)
    s, inhibitor    = _strip_prefix(s, _SL_INHIBITORS)

    modality, method, scaler = _split_on_scaler(s, _SL_SCALERS, exp_id)

    return {
        "exp_id":       exp_id,
        "inhibitor":    inhibitor,
        "modality":     modality,
        "scaler":       scaler.replace("_scaler", ""),
        "method":       method,
        "target_smote": target_smote == "target_smote",
        "pca":          pca_status != "no_pca",
    }


def resolve_baseline_exp_id(
    champion_exp_id: str,
    champion_method: str,
    available_exp_ids: set,
) -> Optional[str]:
    """Return the champion's own configuration trained with the baseline strategy.

    The TL champion and this sibling differ in exactly one factor -- the training
    strategy -- so the gap between them is attributable to pretraining and domain
    adaptation rather than to a different descriptor, modality or scaler. That is
    what makes it the right reference; the *best* baseline model would be a
    different question, since it is free to change every other factor too.

    Parameters
    ----------
    champion_exp_id : str
        exp_id of the top-ranked transfer learning model.
    champion_method : str
        The champion's training strategy -- the token swapped out for the baseline.
    available_exp_ids : set
        exp_ids present in the data, used to verify the sibling was actually run.

    Returns
    -------
    str or None
        The sibling exp_id, or None if the champion already is the baseline or
        the sibling is absent from the results.
    """
    if champion_method == BASELINE_METHOD:
        printer.info(
            f"TL champion already uses the {BASELINE_STRATEGY} baseline strategy "
            "-- no separate baseline arm."
        )
        return None

    token = f"_{champion_method}_"
    if champion_exp_id.count(token) != 1:
        printer.warning(
            f"Could not locate a unique '{champion_method}' token in "
            f"'{champion_exp_id}' -- skipping the {BASELINE_STRATEGY} baseline arm."
        )
        return None

    baseline_exp_id = champion_exp_id.replace(token, f"_{BASELINE_METHOD}_")
    if baseline_exp_id not in available_exp_ids:
        printer.warning(
            f"{BASELINE_STRATEGY} sibling '{baseline_exp_id}' is not present in the "
            "results -- skipping the baseline arm."
        )
        return None

    return baseline_exp_id


class BaseAnalyzer:
    """
    Base class providing common data manipulation and analysis orchestration.

    This class provides shared functionality for analyzing experimental data,
    including methods for breaking down data by variables and performing
    statistical comparisons.

    Parameters
    ----------
    noise_results_dir : str, optional
        Directory path for noise estimation results, by default "noise_estimation".
    scoring_function : str, optional
        Scoring function to use for model ranking ('mean_mcc' or 'mean_mcc_minus_std'),
        by default "mean_mcc_minus_std".
    verbose_analysis : bool, optional
        If True, output detailed analysis logging, by default True.

    Attributes
    ----------
    noise_results_dir : str
        Directory path for noise estimation results.
    scoring_function : str
        Scoring function used for rankings.
    verbose_analysis : bool
        Flag for verbose logging.
    """

    def __init__(
        self,
        noise_results_dir: str = "noise_estimation",
        scoring_function: str = "mean_mcc_minus_std",
        verbose_analysis: bool = True,
    ) -> None:
        self.noise_results_dir = noise_results_dir
        self.scoring_function = scoring_function
        self.verbose_analysis = verbose_analysis

    def analyze_by_variable(
        self,
        df: pd.DataFrame,
        analysis_variable: str,
        grouping_variable: str,
        metrics: List[str],
        verbose: bool,
    ) -> Dict[str, Any]:
        """
        Analyze performance by breaking down data by a specific variable for multiple metrics.

        This method groups data by a specified variable and performs statistical tests
        to compare performance across different values of the analysis variable.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing experimental results.
        analysis_variable : str
            Variable to analyze (e.g., 'pca_status', 'scaler').
        grouping_variable : str
            Variable to group by (e.g., 'modality', 'inhibitor').
        metrics : List[str]
            List of metric names to analyze.
        verbose : bool
            If True, output detailed logging.

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping metric names to analysis results for each grouping value.
        """
        results = {}

        unique_values = df[grouping_variable].unique()
        if self.verbose_analysis:
            printer.info(
                f"Analyzing effect of '{analysis_variable}' across each '{grouping_variable}'."
            )
            printer.info(f"Unique values for {grouping_variable}: {unique_values}")

        for value in unique_values:
            if self.verbose_analysis:
                printer.info(
                    "=" * 20
                    + f" Analyzing for {grouping_variable} = {value} "
                    + "=" * 20
                )
            df_subset = df[df[grouping_variable] == value]
            if self.verbose_analysis:
                printer.info(f"Subset size: {len(df_subset)}")

            for metric in metrics:
                # Average across folds for each experiment within the subset
                df_pivot = df_subset.pivot_table(
                    index="exp_id",
                    columns=analysis_variable,
                    values=metric,
                    aggfunc="mean",
                ).dropna()

                if self.verbose_analysis:
                    printer.info(f"Pivot table shape for {metric}: {df_pivot.shape}")
                    printer.info(f"Pivot table columns: {list(df_pivot.columns)}")

                if df_pivot.shape[0] < 2 or df_pivot.shape[1] < 2:
                    if self.verbose_analysis:
                        printer.warning(
                            f"Insufficient data for {metric} analysis: shape {df_pivot.shape}"
                        )
                    continue

                if metric not in results:
                    results[metric] = {}
                n_groups = df_pivot.shape[1]
                if n_groups == 2:
                    results[metric][value] = self._perform_two_group_analysis(
                        df_pivot, metric, verbose
                    )
                else:
                    results[metric][value] = self._perform_multi_group_analysis(
                        df_pivot, metric, verbose
                    )
        return results

    def _perform_two_group_analysis(
        self, df_pivot: pd.DataFrame, metric: str, verbose: bool
    ) -> Dict[str, Any]:
        """
        Perform statistical analysis for two groups using Wilcoxon signed-rank test.

        This method compares two groups using a paired Wilcoxon test and computes
        summary statistics for each group.

        Parameters
        ----------
        df_pivot : pd.DataFrame
            Pivot table with experiments as rows and two groups as columns.
        metric : str
            Name of the metric being analyzed.
        verbose : bool
            If True, output detailed test results.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - test_type: 'wilcoxon'
            - wilcoxon_test: Test results
            - group_stats: Mean and std for each group
            - group_means: Mean values for each group
        """
        sorted_columns = sorted(df_pivot.columns, key=lambda x: str(x))
        col1_name, col2_name = sorted_columns[0], sorted_columns[1]
        data1, data2 = df_pivot[col1_name], df_pivot[col2_name]

        wilcoxon_results = perform_wilcoxon_test(
            data1, data2, str(col1_name), str(col2_name), metric, verbose
        )

        group_stats = {
            str(col1_name): {"mean": data1.mean(), "std": data1.std()},
            str(col2_name): {"mean": data2.mean(), "std": data2.std()},
        }

        return {
            "test_type": "wilcoxon",
            "wilcoxon_test": wilcoxon_results,
            "group_stats": group_stats,
            "group_means": {str(col1_name): data1.mean(), str(col2_name): data2.mean()},
        }

    def _perform_multi_group_analysis(
        self, df_pivot: pd.DataFrame, metric: str, verbose: bool, alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform statistical analysis for multiple groups using Friedman and post-hoc tests.

        This method uses the Friedman test to detect overall differences, followed
        by pairwise Wilcoxon tests with Holm correction if significant differences
        are found.

        Parameters
        ----------
        df_pivot : pd.DataFrame
            Pivot table with experiments as rows and groups as columns.
        metric : str
            Name of the metric being analyzed.
        verbose : bool
            If True, output detailed test results.
        alpha : float, optional
            Significance level for hypothesis testing, by default 0.05.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - test_type: 'friedman'
            - friedman_test: Friedman test results
            - pairwise_tests: Post-hoc pairwise test results (if significant)
            - group_stats: Mean and std for each group
            - best_groups_by_stat: List of best-performing groups
        """
        friedman_results = perform_friedman_test(df_pivot, metric, verbose)
        pairwise_results = None
        group_stats = {
            str(col): {"mean": df_pivot[col].mean(), "std": df_pivot[col].std()}
            for col in df_pivot.columns
        }
        if not group_stats:
            return {"error": "No group statistics calculated."}

        is_loss_metric = "loss" in metric.lower()
        # Use centralized scoring function
        group_scores = {
            g: calculate_ranking_score(
                stats["mean"], stats["std"], self.scoring_function
            )
            for g, stats in group_stats.items()
        }
        top_group_by_score = (
            min(group_scores, key=group_scores.get)
            if is_loss_metric
            else max(group_scores, key=group_scores.get)
        )
        best_groups_by_stat = [top_group_by_score]

        if friedman_results.get("significant", False):
            pairwise_results = perform_pairwise_wilcoxon(
                df_pivot, metric, "holm", verbose
            )
            for group in df_pivot.columns:
                group_str = str(group)
                if group_str == top_group_by_score:
                    continue

                key1, key2 = (
                    f"{top_group_by_score}_vs_{group_str}",
                    f"{group_str}_vs_{top_group_by_score}",
                )
                p_val = pairwise_results.get(key1, {}).get(
                    "p_value_corrected"
                ) or pairwise_results.get(key2, {}).get("p_value_corrected")

                if p_val is not None and p_val > alpha:
                    best_groups_by_stat.append(group_str)
        else:
            best_groups_by_stat = list(map(str, df_pivot.columns))

        return {
            "test_type": "friedman",
            "friedman_test": friedman_results,
            "pairwise_tests": pairwise_results,
            "group_stats": group_stats,
            "best_groups_by_stat": sorted(best_groups_by_stat),
        }


class ExperimentAnalyzer(BaseAnalyzer):
    """
    A configurable analyzer for running standardized experiments.

    This class extends BaseAnalyzer to provide a structured framework for
    running predefined experimental analyses with consistent reporting.

    Parameters
    ----------
    noise_results_dir : str, optional
        Directory path for noise estimation results, by default "noise_estimation".
    scoring_function : str, optional
        Scoring function to use for model ranking, by default "mean_mcc_minus_std".
    verbose_analysis : bool, optional
        If True, output detailed analysis logging, by default True.
    log_experiment_headers : bool, optional
        If True, log section headers for each experiment, by default True.

    Attributes
    ----------
    log_experiment_headers : bool
        Flag for logging experiment headers.
    """

    def __init__(
        self,
        noise_results_dir: str = "noise_estimation",
        scoring_function: str = "mean_mcc_minus_std",
        verbose_analysis: bool = True,
        log_experiment_headers: bool = True,
    ) -> None:
        super().__init__(noise_results_dir, scoring_function, verbose_analysis)
        self.log_experiment_headers = log_experiment_headers

    def run(
        self,
        df: pd.DataFrame,
        config: Dict[str, Any],
        reporter: Reporter,
        metrics: List[str],
        affinity_type: str,
        verbose: bool = False,
        learning_type: str = "supervised",
    ) -> None:
        """
        Run a configured analysis, generating summary tables and optional verbose breakdowns.

        This method executes a predefined analysis based on the configuration,
        performing statistical tests and generating reports.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing experimental results.
        config : Dict[str, Any]
            Configuration dictionary specifying the analysis parameters:
            - name: Analysis name
            - analysis_variable: Variable to analyze
            - id_variables: Variables defining unique experiments
            - required_groups: Expected number of groups (optional)
        reporter : Reporter
            Reporter instance for generating output tables.
        metrics : List[str]
            List of metric names to analyze.
        affinity_type : str
            Type of affinity data ('pic50' or 'pk').
        verbose : bool, optional
            If True, perform detailed breakdown analyses, by default False.
        learning_type : str, optional
            Type of learning ('supervised' or 'transfer'), by default "supervised".

        Returns
        -------
        None
            Results are reported via the reporter instance.
        """
        name, analysis_var = config["name"], config["analysis_variable"]

        df_temp = df.copy()
        df_temp["exp_id"] = df_temp[config["id_variables"]].agg("_".join, axis=1)

        required_groups = config.get("required_groups") or df[analysis_var].nunique()
        exp_group_counts = df_temp.groupby("exp_id")[analysis_var].nunique()
        common_exps = exp_group_counts[exp_group_counts == required_groups].index

        if len(common_exps) == 0:
            if self.verbose_analysis:
                printer.warning(
                    f"No experiments have all {required_groups} required groups "
                    f"for {analysis_var}. Skipping."
                )
            return

        df_filtered = df_temp[df_temp["exp_id"].isin(common_exps)]
        if df_filtered.empty:
            if self.verbose_analysis:
                printer.warning(f"No overlapping experiments for {name}. Skipping.")
            return

        overall_results_by_metric = {}
        first_pivot = None  # Cache the first pivot to avoid recomputing it for n_pairs
        for metric in metrics:
            df_pivot = df_temp.pivot_table(
                index="exp_id", columns=analysis_var, values=metric, aggfunc="mean"
            ).dropna()
            if df_pivot.shape[0] < 2 or df_pivot.shape[1] < 2:
                if self.verbose_analysis:
                    printer.warning(f"Not enough data for {name} / {metric}.")
                continue

            if first_pivot is None:
                first_pivot = df_pivot

            n_groups = df_pivot.shape[1]
            analysis_func = (
                self._perform_two_group_analysis
                if n_groups == 2
                else self._perform_multi_group_analysis
            )
            overall_results_by_metric[metric] = analysis_func(
                df_pivot, metric, verbose=verbose
            )

        if not overall_results_by_metric:
            return

        # --- Rich output: context block + per-metric result tables ---
        first_result = overall_results_by_metric[next(iter(overall_results_by_metric))]
        groups  = list(first_result.get("group_stats", {}).keys())
        n_pairs = first_pivot.shape[0] if first_pivot is not None else 0

        lt_label = "SL" if learning_type == "supervised" else "TL"
        print_analysis_context(
            name=name,
            factor=analysis_var,
            groups=groups,
            n_pairs=n_pairs,
            learning_type=lt_label,
        )

        for metric, result in overall_results_by_metric.items():
            group_stats = result.get("group_stats", {})
            if result.get("test_type") == "wilcoxon":
                w_result    = result.get("wilcoxon_test", {})
                better      = w_result.get("better_group")
                print_two_group_result(metric, group_stats, w_result, better)
            else:
                print_multi_group_result(
                    metric=metric,
                    group_stats=group_stats,
                    friedman_result=result.get("friedman_test", {}),
                    pairwise_results=result.get("pairwise_tests"),
                    best_groups=result.get("best_groups_by_stat", []),
                )

        reporter.report_overall_comparison(
            overall_results_by_metric, config, affinity_type, metrics, learning_type
        )

        if verbose:
            breakdown_results = {}
            for b_var in config["id_variables"]:
                breakdown_results[f"by_{b_var}"] = self.analyze_by_variable(
                    df_temp.copy(), analysis_var, b_var, metrics, verbose
                )


class GlobalAnalysis(BaseAnalyzer):
    """
    Perform global analysis across all experiments for each metric.

    This class conducts comprehensive analysis across all experimental
    configurations, ranking models and performing statistical comparisons.

    Parameters
    ----------
    noise_results_dir : str, optional
        Directory path for noise estimation results, by default "noise_estimation".
    scoring_function : str, optional
        Scoring function to use for model ranking, by default "mean_mcc_minus_std".
    verbose_analysis : bool, optional
        If True, output detailed analysis logging, by default True.
    log_global_headers : bool, optional
        If True, log section headers for global analyses, by default True.
    include_pvalues_in_tables : bool, optional
        If True, include p-values in output tables, by default True.

    Attributes
    ----------
    log_global_headers : bool
        Flag for logging global analysis headers.
    include_pvalues_in_tables : bool
        Flag for including p-values in tables.
    """

    def __init__(
        self,
        noise_results_dir: str = "noise_estimation",
        scoring_function: str = "mean_mcc_minus_std",
        verbose_analysis: bool = True,
        log_global_headers: bool = True,
        include_pvalues_in_tables: bool = True,
    ) -> None:
        super().__init__(noise_results_dir, scoring_function, verbose_analysis)
        self.log_global_headers = log_global_headers
        self.include_pvalues_in_tables = include_pvalues_in_tables

    def run(
        self,
        df: pd.DataFrame,
        plot_generator: PlotGenerator,
        metrics: List[str],
        verbose: bool = False,
        alpha: float = 0.05,
        method: str = "supervised_learning",
        affinity_type: str = "pic50",
        generate_plots: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a global analysis across all experiments.

        This method ranks all experimental configurations based on their
        performance across metrics and generates visualization plots.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing experimental results with a pre-built 'exp_id' column.
        plot_generator : PlotGenerator
            PlotGenerator instance for creating visualizations.
        metrics : List[str]
            List of metric names to analyze.
        verbose : bool, optional
            If True, output detailed test results, by default False.
        alpha : float, optional
            Significance level for hypothesis testing, by default 0.05.
        method : str, optional
            Learning method type ('supervised_learning' or 'transfer_learning'),
            by default "supervised_learning".
        affinity_type : str, optional
            Type of affinity data ('pic50' or 'pk'), by default "pic50".

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping metric names to analysis results including
            rankings, statistics, and best experiments.

        Raises
        ------
        ValueError
            If 'exp_id' column is not present in the DataFrame.
        """
        if self.log_global_headers:
            printer.info("=" * 60)
            printer.info(f"RUNNING GLOBAL ANALYSIS for {method.upper()}")
            printer.info("=" * 60)

        df_temp = df.copy()  # Always work with a copy

        if "exp_id" not in df_temp.columns:
            raise ValueError(
                "DataFrame must contain a pre-built 'exp_id' column for global analysis."
            )

        results: Dict[str, Any] = {}
        for metric in metrics:
            if metric not in df_temp.columns:
                if self.verbose_analysis:
                    printer.warning(
                        f"Metric '{metric}' not found in dataframe — skipping."
                    )
                continue

            if self.verbose_analysis:
                printer.info(f"=== Global Analysis for metric: {metric} ===")
            df_pivot = df_temp.pivot_table(
                index="fold", columns="exp_id", values=metric, aggfunc="mean"
            ).dropna(axis=1)

            n_folds, n_exps = df_pivot.shape
            if self.verbose_analysis:
                printer.info(
                    f"Folds found: {n_folds}; experiments with complete fold results: {n_exps}"
                )

            if n_exps < 2 or n_folds < 2:
                if self.verbose_analysis:
                    printer.warning(
                        f"Not enough complete data for metric '{metric}' (need >=2 experiments and folds). Skipping."
                    )
                continue

            exp_stats_df = pd.DataFrame(
                {"mean": df_pivot.mean(axis=0), "std": df_pivot.std(axis=0)}
            )
            is_loss = "loss" in metric.lower()

            # Use centralized scoring function
            exp_stats_df["ranking_score"] = exp_stats_df.apply(
                lambda row: calculate_ranking_score(
                    row["mean"], row["std"], self.scoring_function
                ),
                axis=1,
            )
            sorted_exp_df = exp_stats_df.sort_values("ranking_score", ascending=is_loss)

            # Reuse the already-computed pivot — no need to rebuild for Friedman
            friedman_results = perform_friedman_test(df_pivot, metric, verbose)
            pairwise_results = None
            best_exps: List[str] = []

            top_exp_by_score = sorted_exp_df.index[0]
            best_exps.append(top_exp_by_score)

            best_exps = list(sorted_exp_df.index[:5])
            composite_scores = exp_stats_df.loc[best_exps, "ranking_score"].to_dict()

            results[metric] = {
                "friedman": friedman_results,
                "pairwise": pairwise_results,
                "exp_stats": exp_stats_df.to_dict(orient="index"),
                "top_by_score": list(sorted_exp_df.index[:5]),
                "best_experiments": best_exps,
                "composite_scores": composite_scores,
            }

            if self.verbose_analysis:
                printer.info(
                    f"Selected best (statistically tied or top-5) for {metric}: {len(best_exps)} experiments."
                )
                max_len = max(len(exp) for exp in best_exps) if best_exps else 0
                for exp in best_exps:
                    printer.info(
                        f'{exp:{max_len}}: μ={exp_stats_df.loc[exp,"mean"]:.3f} ± {exp_stats_df.loc[exp,"std"]:.3f}, score={composite_scores.get(exp, 0):.3f}'
                    )
                printer.info("\n")

        if generate_plots:
            if method == "transfer_learning":
                plot_generator.create_transfer_learning_scatter_plot(
                    df_temp, metric="val/mcc", affinity_type=affinity_type
                )
            else:
                plot_generator.create_supervised_scatter_plot(
                    df_temp, metric="val/mcc", affinity_type=affinity_type
                )
        return results


class MainAnalysisController:
    """
    Main controller to orchestrate the entire analysis pipeline.

    This class coordinates all analyses across supervised learning and transfer
    learning approaches, managing data processing, statistical tests, and reporting.

    Parameters
    ----------
    df_sl : pd.DataFrame
        DataFrame containing supervised learning results.
    df_tl : pd.DataFrame
        DataFrame containing transfer learning results.
    noise_results_dir : str, optional
        Directory path for noise estimation results, by default "noise_estimation".
    scoring_function : str, optional
        Scoring function to use for model ranking, by default "mean_mcc_minus_std".
    verbose_analysis : bool, optional
        If True, output detailed analysis logging, by default False.
    log_experiment_headers : bool, optional
        If True, log headers for individual experiments, by default False.
    log_global_headers : bool, optional
        If True, log headers for global analyses, by default True.
    log_main_headers : bool, optional
        If True, log main section headers, by default True.

    Attributes
    ----------
    _df_sl_original : pd.DataFrame
        Original supervised learning DataFrame (immutable).
    _df_tl_original : pd.DataFrame
        Original transfer learning DataFrame (immutable).
    df_sl : pd.DataFrame
        Processed supervised learning DataFrame.
    df_tl : pd.DataFrame
        Processed transfer learning DataFrame.
    analyzer : ExperimentAnalyzer
        Instance for running experiment-level analyses.
    global_analysis : GlobalAnalysis
        Instance for running global analyses.
    supervised_configs : List[Dict[str, Any]]
        Analysis configurations for supervised learning.
    tl_configs : List[Dict[str, Any]]
        Analysis configurations for transfer learning.
    """

    def __init__(
        self,
        df_sl: pd.DataFrame,
        df_tl: pd.DataFrame,
        noise_results_dir: str = "noise_estimation",
        scoring_function: str = "mean_mcc_minus_std",
        rank_method: str = "dense",
        metric_decimals: Optional[int] = 3,
        verbose_analysis: bool = False,
        log_experiment_headers: bool = False,
        log_global_headers: bool = True,
        log_main_headers: bool = True,
    ) -> None:
        # Store original dataframes (never modify these)
        self._df_sl_original = df_sl.copy()
        self._df_tl_original = df_tl.copy()

        self.scoring_function = scoring_function
        self.rank_method = rank_method
        self.metric_decimals = metric_decimals
        self.noise_results_dir = noise_results_dir

        # Logging control flags
        self.verbose_analysis = verbose_analysis
        self.log_experiment_headers = log_experiment_headers
        self.log_global_headers = log_global_headers
        self.log_main_headers = log_main_headers

        # Stores champion fold data keyed by (affinity_type, threshold) for the combined plot
        self._champion_data: dict = {}

        # Initialize analyzers with logging controls
        self.analyzer = ExperimentAnalyzer(
            noise_results_dir=self.noise_results_dir,
            scoring_function=self.scoring_function,
            verbose_analysis=self.verbose_analysis,
            log_experiment_headers=self.log_experiment_headers,
        )
        self.global_analysis = GlobalAnalysis(
            noise_results_dir=self.noise_results_dir,
            scoring_function=self.scoring_function,
            verbose_analysis=self.verbose_analysis,
            log_global_headers=self.log_global_headers,
        )
        self.all_sl_metrics = ["val/mcc", "val/accuracy", "val/f1"]
        self.all_tl_metrics = ["val/loss", "val/mcc", "val/accuracy", "val/f1"]

        self.sl_id_vars = [
            "inhibitor",
            "modality",
            "scaler",
            "method",
            "target_smote",
            "pca_status",
        ]
        self.tl_id_vars = [
            "inhibitor",
            "modality",
            "scaler",
            "method",
            "cross_domain_scaling",
            "source_smote",
            "target_smote",
        ]

        # Initialize processed dataframes once
        self.df_sl = self._prepare_dataframe(self._df_sl_original, self.sl_id_vars)
        self.df_tl = self._prepare_dataframe(self._df_tl_original, self.tl_id_vars)

        # Analysis configurations
        self.supervised_configs = [
            {
                "name": "PCA",
                "analysis_variable": "pca_status",
                "id_variables": [
                    "inhibitor",
                    "modality",
                    "scaler",
                    "method",
                    "target_smote",
                ],
                "required_groups": 2,
            },
            {
                "name": "Scaler",
                "analysis_variable": "scaler",
                "id_variables": [
                    "inhibitor",
                    "modality",
                    "pca_status",
                    "method",
                    "target_smote",
                ],
                "required_groups": 3,
            },
            {
                "name": "Descriptor",
                "analysis_variable": "modality",
                "id_variables": [
                    "inhibitor",
                    "scaler",
                    "pca_status",
                    "method",
                    "target_smote",
                ],
                "required_groups": 6,
            },
            {
                "name": "Inhibitor",
                "analysis_variable": "inhibitor",
                "id_variables": [
                    "modality",
                    "scaler",
                    "pca_status",
                    "method",
                    "target_smote",
                ],
                "required_groups": 3,
            },
            {
                "name": "Method",
                "analysis_variable": "method",
                "id_variables": [
                    "inhibitor",
                    "modality",
                    "scaler",
                    "pca_status",
                    "target_smote",
                ],
            },
            {
                "name": "Target SMOTE",
                "analysis_variable": "target_smote",
                "id_variables": [
                    "inhibitor",
                    "modality",
                    "scaler",
                    "pca_status",
                    "method",
                ],
            },
        ]
        self.tl_configs = [
            {
                "name": "TL Scaler",
                "analysis_variable": "scaler",
                "id_variables": [
                    "inhibitor",
                    "modality",
                    "method",
                    "cross_domain_scaling",
                    "source_smote",
                    "target_smote",
                ],
                "required_groups": 3,
            },
            {
                "name": "TL Cross-Domain Scaling",
                "analysis_variable": "cross_domain_scaling",
                "id_variables": [
                    "inhibitor",
                    "modality",
                    "scaler",
                    "method",
                    "source_smote",
                    "target_smote",
                ],
                "required_groups": 2,
            },
            {
                "name": "TL Source SMOTE",
                "analysis_variable": "source_smote",
                "id_variables": [
                    "inhibitor",
                    "modality",
                    "scaler",
                    "method",
                    "cross_domain_scaling",
                    "target_smote",
                ],
                "required_groups": 2,
            },
            {
                "name": "TL Target SMOTE",
                "analysis_variable": "target_smote",
                "id_variables": [
                    "inhibitor",
                    "modality",
                    "scaler",
                    "method",
                    "cross_domain_scaling",
                    "source_smote",
                ],
                "required_groups": 2,
            },
            {
                "name": "TL Descriptor",
                "analysis_variable": "modality",
                "id_variables": [
                    "inhibitor",
                    "scaler",
                    "method",
                    "cross_domain_scaling",
                    "source_smote",
                    "target_smote",
                ],
                "required_groups": 6,
            },
            {
                "name": "TL Inhibitor",
                "analysis_variable": "inhibitor",
                "id_variables": [
                    "modality",
                    "scaler",
                    "method",
                    "cross_domain_scaling",
                    "source_smote",
                    "target_smote",
                ],
                "required_groups": 3,
            },
            {
                "name": "Pretrain",
                "analysis_variable": "pretrain_status",
                "id_variables": [
                    "inhibitor",
                    "modality",
                    "scaler",
                    "cross_domain_scaling",
                    "source_smote",
                    "target_smote",
                ],
            },
            {
                "name": "CCSA",
                "analysis_variable": "ccsa_status",
                "id_variables": [
                    "inhibitor",
                    "modality",
                    "scaler",
                    "cross_domain_scaling",
                    "source_smote",
                    "target_smote",
                ],
                "required_groups": 2,
            },
            {
                "name": "TL Method",
                "analysis_variable": "method",
                "id_variables": [
                    "inhibitor",
                    "modality",
                    "scaler",
                    "cross_domain_scaling",
                    "source_smote",
                    "target_smote",
                ],
            },
        ]

    def _make_reporter(self, output_dir: str) -> Reporter:
        return Reporter(
            output_dir=output_dir,
            noise_results_dir=self.noise_results_dir,
            scoring_function=self.scoring_function,
            verbose_reporting=self.verbose_analysis,
            log_table_generation=self.log_experiment_headers,
        )

    def _make_plot_generator(self, output_dir: str) -> PlotGenerator:
        return PlotGenerator(output_dir=output_dir, scoring_function=self.scoring_function)

    def _prepare_dataframe(self, df: pd.DataFrame, id_vars: List[str]) -> pd.DataFrame:
        """
        Prepare a DataFrame for cross-threshold analysis.

        This method normalizes column values and creates unique experiment IDs
        for consistent analysis across thresholds.

        Parameters
        ----------
        df : pd.DataFrame
            Raw experimental results DataFrame.
        id_vars : List[str]
            List of variable names that define unique experiments.

        Returns
        -------
        pd.DataFrame
            Processed DataFrame with normalized columns and 'exp_id' field.
        """
        if df.empty:
            return df

        df = df.copy()

        # # Normalize ALL columns that will be used in any analysis
        present_vars = [v for v in id_vars if v in df.columns]

        for var in present_vars:
            df = self._normalize_column_values(df, var)

        # Normalize threshold separately (keep it as a column for filtering)
        if "threshold" in df.columns:
            df = self._normalize_column_values(df, "threshold")

        # Create exp_id WITHOUT threshold - this is the unique experiment signature
        id_vars_present = [v for v in id_vars if v in df.columns]
        df["exp_id"] = df[id_vars_present].agg("_".join, axis=1)

        return df

    def _normalize_column_values(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Normalize column values to handle boolean and other types consistently.

        This method converts boolean values to meaningful strings and handles
        special cases for domain scaling and SMOTE variables.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to process.
        column : str
            Name of the column to normalize.

        Returns
        -------
        pd.DataFrame
            DataFrame with normalized column values.
        """
        df = df.copy()

        # Special handling for cross_domain_scaling and source_smote
        if column == "cross_domain_scaling":
            df[column] = df[column].map(
                {True: "cross_domain", False: "separate_domain"}
            )
        elif column == "source_smote":
            df[column] = df[column].map(
                {True: "source_smote", False: "no_source_smote"}
            )
        elif column == "target_smote":
            df[column] = df[column].map(
                {True: "target_smote", False: "no_target_smote"}
            )
        elif df[column].dtype == "bool":
            # Convert boolean to string explicitly for other boolean columns
            df[column] = df[column].map({True: "True", False: "False"})
        else:
            # Ensure all values are strings for consistency, handle NaN values
            df[column] = df[column].fillna("unknown").astype(str)

        return df

    def _filter_data(
        self,
        df: pd.DataFrame,
        affinity_type: str,
        threshold: int = None,
    ) -> pd.DataFrame:
        """
        Filter a prepared DataFrame by affinity type and optionally by threshold.

        Parameters
        ----------
        df : pd.DataFrame
            Prepared DataFrame (output of _prepare_dataframe).
        affinity_type : str
            Affinity type to keep ('pic50' or 'pk').
        threshold : int, optional
            If provided, also filter to this threshold value.

        Returns
        -------
        pd.DataFrame
            Filtered copy of the input DataFrame.
        """
        mask = df["affinity_type"] == affinity_type
        if threshold is not None:
            mask &= df["threshold"] == str(threshold)
        return df[mask].copy()

    def _run_approach_analyses(
        self,
        df: pd.DataFrame,
        configs: List[Dict[str, Any]],
        metrics: List[str],
        learning_type: str,
        reporter: "Reporter",
        plot_generator: "PlotGenerator",
        affinity_type: str,
        verbose: bool,
    ):
        """
        Run factor analyses and global ranking for one learning approach (SL or TL).

        Encapsulates the shared logic that was previously duplicated for the
        supervised and transfer learning blocks inside both
        run_single_threshold_analysis and run_global_cross_threshold_analysis.

        Parameters
        ----------
        df : pd.DataFrame
            Already-filtered DataFrame for this approach/affinity/threshold.
        configs : list
            Analysis configs to run (filtered by _filter_configs).
        metrics : list[str]
            Metrics to analyse.
        learning_type : str
            ``"supervised"`` or ``"transfer"`` — used for display labels and
            to build the ``method`` string passed to GlobalAnalysis.run().
        reporter : Reporter
        plot_generator : PlotGenerator
        affinity_type : str
        verbose : bool

        Returns
        -------
        dict or None
            GlobalAnalysis results dict, or None if the DataFrame was empty.
        """
        if df.empty:
            if self.verbose_analysis:
                printer.warning(
                    f"Skipping {learning_type} analysis for {affinity_type} "
                    "due to missing data."
                )
            return None

        available = [m for m in metrics if m in df.columns]
        if available:
            df.dropna(subset=available, inplace=True)

        for config in configs:
            self.analyzer.run(
                df, config, reporter, metrics, affinity_type, verbose,
                learning_type=learning_type,
            )

        return self.global_analysis.run(
            df,
            plot_generator,
            metrics,
            verbose,
            method=f"{learning_type}_learning",
            affinity_type=affinity_type,
            generate_plots=False,
        )

    def _collect_champion_data(
        self,
        affinity_type: str,
        df_sl: pd.DataFrame,
        df_tl: pd.DataFrame,
        sl_avg_ranks: pd.DataFrame,
        tl_avg_ranks: pd.DataFrame,
        tl_baseline_exp_id: Optional[str] = None,
    ) -> None:
        """
        Extract per-fold values for MCC, accuracy and F1 for the top-ranked SL
        and TL models, split by threshold.

        When ``tl_baseline_exp_id`` is given, a third arm is collected under the
        ``bl_`` prefix: the champion's own configuration trained with the
        baseline strategy. Everything downstream treats it as optional, so
        omitting it reproduces the two-arm output exactly.

        The Max column is the noise ceiling (mean of noise distribution) loaded
        from the noise estimation pkl files via MaxValueExtractor — not the best
        observed model score, which can reach 1.0 on easy folds.

        Stores results in ``self._champion_data[(affinity_type, threshold)]``.
        """
        from max_value_extractor import MaxValueExtractor

        sl_exp_id = sl_avg_ranks.iloc[0]["exp_id"]
        tl_exp_id = tl_avg_ranks.iloc[0]["exp_id"]

        all_metrics = ["val/mcc", "val/accuracy", "val/f1"]
        short = {"val/mcc": "mcc", "val/accuracy": "accuracy", "val/f1": "f1"}

        def _fold_values(df: pd.DataFrame, exp_id: str, cols: list) -> pd.DataFrame:
            keep = ["threshold", "fold"] + [c for c in cols if c in df.columns]
            rows = df[df["exp_id"] == exp_id][keep].copy()
            rows["threshold"] = rows["threshold"].astype(str)
            rows["fold"] = rows["fold"].astype(str)
            return rows.sort_values(["threshold", "fold"]).reset_index(drop=True)

        sl_rows = _fold_values(df_sl, sl_exp_id, all_metrics)
        tl_rows = _fold_values(df_tl, tl_exp_id, all_metrics)

        # Inner-join on (threshold, fold) to ensure alignment
        merge_cols = ["threshold", "fold"]
        sl_rename = {m: f"sl_{short[m]}" for m in all_metrics if m in sl_rows.columns}
        tl_rename = {m: f"tl_{short[m]}" for m in all_metrics if m in tl_rows.columns}
        paired_all = pd.merge(
            sl_rows.rename(columns=sl_rename),
            tl_rows.rename(columns=tl_rename),
            on=merge_cols,
            how="inner",
        )

        # Optional third arm: the baseline sibling, joined on the same folds so
        # all three stay paired. An inner join would silently drop folds from the
        # SL/TL comparison if the baseline were missing any, so join left and let
        # the gaps surface as NaN.
        if tl_baseline_exp_id:
            bl_rows = _fold_values(df_tl, tl_baseline_exp_id, all_metrics)
            bl_rename = {
                m: f"bl_{short[m]}" for m in all_metrics if m in bl_rows.columns
            }
            paired_all = pd.merge(
                paired_all,
                bl_rows.rename(columns=bl_rename),
                on=merge_cols,
                how="left",
            )
            n_missing = int(paired_all["bl_mcc"].isna().sum()) if "bl_mcc" in paired_all else len(paired_all)
            if n_missing:
                printer.warning(
                    f"{BASELINE_STRATEGY} baseline '{tl_baseline_exp_id}' is missing "
                    f"{n_missing} of {len(paired_all)} (threshold, fold) pairs."
                )

        noise_extractor = MaxValueExtractor(self.noise_results_dir)

        for threshold, grp in paired_all.groupby("threshold"):
            grp = grp.sort_values("fold").reset_index(drop=True)
            n_folds = len(grp)

            # Load noise ceiling for this threshold
            try:
                threshold_int = int(float(threshold))
                noise_max = noise_extractor.extract_max_values_from_noise_results(
                    threshold_int, affinity_type
                )
            except Exception as e:
                printer.warning(f"Could not load noise ceiling for threshold {threshold}: {e}")
                noise_max = {}

            def _noise_vals(key: str) -> np.ndarray:
                vals = noise_max.get(key, [])
                if len(vals) >= n_folds:
                    return np.array(vals[:n_folds], dtype=float)
                printer.warning(
                    f"Noise ceiling '{key}' has {len(vals)} folds, expected {n_folds} "
                    f"(threshold={threshold}, {affinity_type}) — filling with NaN"
                )
                out = np.full(n_folds, np.nan)
                out[:len(vals)] = vals
                return out

            def _arm(col: str) -> np.ndarray:
                """Per-fold values for one arm, or all-NaN when the arm is absent."""
                return grp[col].values if col in grp else np.full(n_folds, np.nan)

            fold_labels = [f"Fold {i+1}" for i in range(n_folds)]
            key = (affinity_type, threshold)
            self._champion_data[key] = {
                # MCC (used by the bar-chart plot)
                "sl_vals":    grp["sl_mcc"].values if "sl_mcc" in grp else np.full(n_folds, np.nan),
                "tl_vals":    grp["tl_mcc"].values if "tl_mcc" in grp else np.full(n_folds, np.nan),
                "max_vals":   _noise_vals("mcc"),
                # All metrics for the summary table
                "sl_mcc":     grp["sl_mcc"].values if "sl_mcc" in grp else np.full(n_folds, np.nan),
                "tl_mcc":     grp["tl_mcc"].values if "tl_mcc" in grp else np.full(n_folds, np.nan),
                "max_mcc":    _noise_vals("mcc"),
                "sl_accuracy": grp["sl_accuracy"].values if "sl_accuracy" in grp else np.full(n_folds, np.nan),
                "tl_accuracy": grp["tl_accuracy"].values if "tl_accuracy" in grp else np.full(n_folds, np.nan),
                "max_accuracy": _noise_vals("accuracy"),
                "sl_f1":      grp["sl_f1"].values if "sl_f1" in grp else np.full(n_folds, np.nan),
                "tl_f1":      grp["tl_f1"].values if "tl_f1" in grp else np.full(n_folds, np.nan),
                "max_f1":     _noise_vals("f1"),
                # Baseline arm — populated only when tl_baseline_exp_id was given
                "bl_vals":     _arm("bl_mcc"),
                "bl_mcc":      _arm("bl_mcc"),
                "bl_accuracy": _arm("bl_accuracy"),
                "bl_f1":       _arm("bl_f1"),
                "fold_labels": fold_labels,
                "sl_exp_id":  sl_exp_id,
                "tl_exp_id":  tl_exp_id,
                "bl_exp_id":  tl_baseline_exp_id,
                "threshold":  threshold,
            }
        baseline_note = (
            f"  {BASELINE_STRATEGY}={tl_baseline_exp_id}" if tl_baseline_exp_id else ""
        )
        printer.info(
            f"Champion data collected for {affinity_type}: "
            f"SL={sl_exp_id}  TL={tl_exp_id}{baseline_note}"
        )

    def _compute_global_performance(self, affinity_type: str) -> Dict[str, Any]:
        """
        Compute the global mean performance of the champion SL and TL models.

        For each metric, the global mean is computed as:
        1. Mean over the five cross-validation folds at each threshold.
        2. Mean of those per-threshold means across all thresholds.

        This two-level averaging ensures each threshold contributes equally
        regardless of how many folds it has.

        Parameters
        ----------
        affinity_type : str
            Affinity type ('pic50' or 'pk').

        Returns
        -------
        Dict[str, Any]
            Nested dict: source → metric → {mean_t<N>, …, global_mean, global_std}.
        """
        sources = ["sl", "tl", "bl", "max"]
        metrics = ["mcc", "accuracy", "f1"]

        thresholds = sorted(
            t for (aff, t) in self._champion_data if aff == affinity_type
        )

        result: Dict[str, Any] = {}
        for src in sources:
            result[src] = {}
            for metric in metrics:
                key_name = f"{src}_{metric}"
                per_threshold: Dict[str, float] = {}
                for t in thresholds:
                    data = self._champion_data.get((affinity_type, t))
                    if data is None or key_name not in data:
                        continue
                    vals_t = np.asarray(data[key_name], dtype=float)
                    # An absent arm (e.g. no baseline sibling) is all-NaN; skip it
                    # rather than writing NaN into the JSON.
                    if vals_t.size == 0 or np.all(np.isnan(vals_t)):
                        continue
                    per_threshold[f"mean_t{t}"] = round(float(np.nanmean(vals_t)), 4)

                if not per_threshold:
                    continue

                vals = list(per_threshold.values())
                result[src][metric] = {
                    **per_threshold,
                    "global_mean": round(float(np.mean(vals)), 4),
                    "global_std":  round(float(np.std(vals)), 4),
                }

            if not result[src]:
                del result[src]

        return result

    @staticmethod
    def _wilcoxon_raw(a: np.ndarray, b: np.ndarray) -> Optional[float]:
        """Two-sided paired Wilcoxon p-value, or None when it cannot be computed.

        Pairs where either arm is NaN are dropped first, so a baseline that is
        missing a few folds is still tested on the folds it does cover rather
        than collapsing the whole comparison.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        if a.size == 0 or np.all(a - b == 0):
            return None
        try:
            _, p = wilcoxon(a, b, alternative="two-sided")
            return float(p)
        except Exception as e:
            printer.warning(f"Wilcoxon test failed: {e}")
            return None

    @staticmethod
    def _holm(p_values: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """Holm-correct the pre-specified champion comparisons.

        The family is the two hypotheses this comparison actually poses: TL
        against the supervised champion, and TL against the baseline strategy.
        The baseline-vs-supervised contrast is reported as a descriptive delta
        instead — it is determined by the other two, so testing it as well would
        dress two degrees of freedom up as three.
        """
        keys = [k for k, p in p_values.items() if p is not None]
        out: Dict[str, Optional[float]] = {k: None for k in p_values}
        if not keys:
            return out
        _, corrected, _, _ = multipletests(
            [p_values[k] for k in keys], alpha=0.05, method="holm"
        )
        for k, p in zip(keys, corrected):
            out[k] = float(p)
        return out

    @staticmethod
    def _format_p(p: Optional[float]) -> str:
        """Render a p-value with a significance marker and colour."""
        if p is None:
            return "n/a"
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        colour = "green" if p < 0.05 else "yellow"
        shown = "<0.001" if p < 0.001 else f"{p:.3f}"
        return f"[{colour}]{shown} {sig}[/{colour}]"

    @staticmethod
    def _format_pct(reference: float, value: float) -> str:
        """Percentage change of ``value`` relative to ``reference``."""
        if not np.isfinite(reference) or not np.isfinite(value) or reference == 0:
            return "n/a"
        pct = (value - reference) / abs(reference) * 100
        colour = "green" if pct >= 0 else "red"
        sign = "+" if pct >= 0 else ""
        return f"[{colour}]{sign}{pct:.1f}%[/{colour}]"

    @staticmethod
    def _format_delta(delta: float) -> str:
        """Render an absolute difference with sign colouring."""
        if not np.isfinite(delta):
            return "n/a"
        return (
            f"[green]+{delta:.3f}[/green]" if delta >= 0 else f"[red]{delta:.3f}[/red]"
        )

    @staticmethod
    def _has_baseline(data: dict) -> bool:
        """True when a baseline arm was collected and is not entirely missing."""
        vals = np.asarray(data.get("bl_mcc", []), dtype=float)
        return vals.size > 0 and not np.all(np.isnan(vals))

    def _print_champion_summary(
        self, affinity_type: str, threshold: str, data: dict
    ) -> None:
        """
        Print two rich tables summarising the champion comparison for one
        (affinity_type, threshold) combination:

        1. Per-fold MCC table  — B / S1 / TL / Max per fold, with Δ TL−B and
                                 Δ TL−S1, plus a mean row.
        2. Metric summary      — mean ± std, Max mean, Δ%, and per-threshold
                                 Wilcoxon p-values for MCC / accuracy / F1.

        The S1 columns appear only when a baseline arm was collected. Note that
        at n=5 the smallest two-sided Wilcoxon p is 0.0625, so no per-threshold
        test can reach α=0.05 — these tables are descriptive, and the pooled
        cross-threshold test is where the comparison is decided. For that reason
        no multiplicity correction is applied here.
        """
        from console import _console
        from rich.table import Table
        from rich.rule import Rule
        from rich import box

        affinity_display = {"pic50": "pIC50", "pk": "pKi"}
        label = f"{affinity_display.get(affinity_type, affinity_type)} · θ = {threshold}"
        _console.print(Rule(f"[bold]Champion summary — {label}[/bold]"))

        fold_labels = data["fold_labels"]
        n = len(fold_labels)
        has_bl = self._has_baseline(data)

        # ── Table 1: per-fold MCC ────────────────────────────────────────
        t1 = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
        t1.add_column("Fold", style="dim")
        t1.add_column("B (MCC)",   justify="right")
        if has_bl:
            t1.add_column(f"{BASELINE_STRATEGY} (MCC)", justify="right")
        t1.add_column("TL (MCC)",  justify="right")
        t1.add_column("Max (MCC)", justify="right")
        t1.add_column("Δ TL−B",   justify="right")
        if has_bl:
            t1.add_column(f"Δ TL−{BASELINE_STRATEGY}", justify="right")

        sl_mcc  = np.asarray(data["sl_mcc"],  dtype=float)
        tl_mcc  = np.asarray(data["tl_mcc"],  dtype=float)
        max_mcc = np.asarray(data["max_mcc"], dtype=float)
        bl_mcc  = np.asarray(
            data.get("bl_mcc", np.full(n, np.nan)), dtype=float
        )

        for i, lbl in enumerate(fold_labels):
            row = [lbl, f"{sl_mcc[i]:.3f}"]
            if has_bl:
                row.append(f"{bl_mcc[i]:.3f}" if np.isfinite(bl_mcc[i]) else "n/a")
            row.append(f"{tl_mcc[i]:.3f}")
            row.append(f"{max_mcc[i]:.3f}" if np.isfinite(max_mcc[i]) else "n/a")
            row.append(self._format_delta(tl_mcc[i] - sl_mcc[i]))
            if has_bl:
                row.append(self._format_delta(tl_mcc[i] - bl_mcc[i]))
            t1.add_row(*row)

        sl_mean_mcc  = float(np.nanmean(sl_mcc))
        tl_mean_mcc  = float(np.nanmean(tl_mcc))
        max_mean_mcc = float(np.nanmean(max_mcc))
        sl_std_mcc   = float(np.nanstd(sl_mcc))
        tl_std_mcc   = float(np.nanstd(tl_mcc))

        mean_row = [
            "[bold]Mean ± std[/bold]",
            f"[bold]{sl_mean_mcc:.3f} ± {sl_std_mcc:.3f}[/bold]",
        ]
        if has_bl:
            mean_row.append(
                f"[bold]{np.nanmean(bl_mcc):.3f} ± {np.nanstd(bl_mcc):.3f}[/bold]"
            )
        mean_row.append(f"[bold]{tl_mean_mcc:.3f} ± {tl_std_mcc:.3f}[/bold]")
        mean_row.append(
            f"[bold]{max_mean_mcc:.3f}[/bold]"
            if np.isfinite(max_mean_mcc)
            else "[bold]n/a[/bold]"
        )
        mean_row.append(self._format_delta(tl_mean_mcc - sl_mean_mcc))
        if has_bl:
            mean_row.append(
                self._format_delta(tl_mean_mcc - float(np.nanmean(bl_mcc)))
            )
        t1.add_row(*mean_row)
        _console.print(t1)

        if has_bl:
            _console.print(
                f"  [dim]B  = {data.get('sl_exp_id')}\n"
                f"  TL = {data.get('tl_exp_id')}\n"
                f"  {BASELINE_STRATEGY} = {data.get('bl_exp_id')}[/dim]"
            )

        # ── Table 2: metric summary + per-threshold Wilcoxon ─────────────
        metrics_info = [
            ("MCC",      "sl_mcc",      "tl_mcc",      "bl_mcc",      "max_mcc"),
            ("Accuracy", "sl_accuracy", "tl_accuracy", "bl_accuracy", "max_accuracy"),
            ("F1",       "sl_f1",       "tl_f1",       "bl_f1",       "max_f1"),
        ]

        t2 = Table(
            box=box.SIMPLE_HEAVY, show_header=True, header_style="bold",
            title=f"Metric summary  [dim](per-threshold Wilcoxon, n={n}; "
                  f"min achievable p = {1/2**(n-1):.4f} — descriptive only, "
                  f"uncorrected)[/dim]",
        )
        t2.add_column("Metric")
        t2.add_column("B  mean ± std",  justify="right")
        if has_bl:
            t2.add_column(f"{BASELINE_STRATEGY} mean ± std", justify="right")
        t2.add_column("TL mean ± std",  justify="right")
        t2.add_column("Max mean",        justify="right")
        t2.add_column("Δ% TL−B",        justify="right")
        if has_bl:
            t2.add_column(f"Δ% TL−{BASELINE_STRATEGY}", justify="right")
        t2.add_column("p TL−B",          justify="right")
        if has_bl:
            t2.add_column(f"p TL−{BASELINE_STRATEGY}", justify="right")

        for metric_label, sl_key, tl_key, bl_key, max_key in metrics_info:
            sl_v  = np.asarray(data.get(sl_key,  [np.nan] * n), dtype=float)
            tl_v  = np.asarray(data.get(tl_key,  [np.nan] * n), dtype=float)
            bl_v  = np.asarray(data.get(bl_key,  [np.nan] * n), dtype=float)
            max_v = np.asarray(data.get(max_key, [np.nan] * n), dtype=float)

            sl_m, sl_s = float(np.nanmean(sl_v)), float(np.nanstd(sl_v))
            tl_m, tl_s = float(np.nanmean(tl_v)), float(np.nanstd(tl_v))
            max_m      = float(np.nanmean(max_v))

            row = [metric_label, f"{sl_m:.3f} ± {sl_s:.3f}"]
            if has_bl:
                row.append(f"{np.nanmean(bl_v):.3f} ± {np.nanstd(bl_v):.3f}")
            row.append(f"{tl_m:.3f} ± {tl_s:.3f}")
            row.append(f"{max_m:.3f}" if np.isfinite(max_m) else "n/a")
            row.append(self._format_pct(sl_m, tl_m))
            if has_bl:
                row.append(self._format_pct(float(np.nanmean(bl_v)), tl_m))
            row.append(self._format_p(self._wilcoxon_raw(tl_v, sl_v)))
            if has_bl:
                row.append(self._format_p(self._wilcoxon_raw(tl_v, bl_v)))
            t2.add_row(*row)
        _console.print(t2)

    def _print_pooled_champion_summary(
        self, affinity_type: str, thresholds: list
    ) -> None:
        """
        Print a pooled Wilcoxon signed-rank test (n = n_folds × n_thresholds)
        for each metric, comparing the cross-threshold TL champion against the
        supervised champion and, when collected, against the baseline strategy.

        This is the test that decides both comparisons: at n=5 a per-threshold
        test cannot reach α=0.05, whereas pooling across thresholds gives enough
        pairs for a real verdict. The two comparisons form the pre-specified
        family and are Holm-corrected within each metric; the baseline-vs-
        supervised gap is shown as a descriptive delta only.
        """
        from console import _console
        from rich.table import Table
        from rich.rule import Rule
        from rich import box

        affinity_display = {"pic50": "pIC50", "pk": "pKi"}
        label = affinity_display.get(affinity_type, affinity_type)
        n_total = sum(
            len(self._champion_data[(affinity_type, t)]["fold_labels"])
            for t in thresholds
            if (affinity_type, t) in self._champion_data
        )
        has_bl = any(
            self._has_baseline(self._champion_data[(affinity_type, t)])
            for t in thresholds
            if (affinity_type, t) in self._champion_data
        )

        _console.print(Rule(
            f"[bold]Pooled cross-threshold Wilcoxon — {label} "
            f"(n={n_total} fold pairs)[/bold]"
        ))

        metrics_info = [
            ("MCC",      "sl_mcc",      "tl_mcc",      "bl_mcc"),
            ("Accuracy", "sl_accuracy", "tl_accuracy", "bl_accuracy"),
            ("F1",       "sl_f1",       "tl_f1",       "bl_f1"),
        ]

        t = Table(
            box=box.SIMPLE_HEAVY, show_header=True, header_style="bold",
            title=(
                f"[dim]p raw (Holm-corrected over the {'two' if has_bl else 'one'} "
                f"pre-specified comparison{'s' if has_bl else ''} per metric)[/dim]"
                if has_bl else None
            ),
        )
        t.add_column("Metric")
        t.add_column("B  mean ± std",  justify="right")
        if has_bl:
            t.add_column(f"{BASELINE_STRATEGY} mean ± std", justify="right")
        t.add_column("TL mean ± std",  justify="right")
        t.add_column("Δ% TL−B",       justify="right")
        if has_bl:
            t.add_column(f"Δ% TL−{BASELINE_STRATEGY}", justify="right")
            t.add_column(f"Δ {BASELINE_STRATEGY}−B [dim](descr.)[/dim]", justify="right")
        t.add_column("p TL−B",         justify="right")
        if has_bl:
            t.add_column(f"p TL−{BASELINE_STRATEGY}", justify="right")

        for metric_label, sl_key, tl_key, bl_key in metrics_info:
            sl_all, tl_all, bl_all = [], [], []
            for thresh in thresholds:
                key = (affinity_type, thresh)
                if key not in self._champion_data:
                    continue
                d = self._champion_data[key]
                n_d = len(d["fold_labels"])
                sl_all.extend(d.get(sl_key, [np.nan] * n_d))
                tl_all.extend(d.get(tl_key, [np.nan] * n_d))
                bl_all.extend(d.get(bl_key, [np.nan] * n_d))

            sl_v = np.asarray(sl_all, dtype=float)
            tl_v = np.asarray(tl_all, dtype=float)
            bl_v = np.asarray(bl_all, dtype=float)

            sl_m, sl_s = float(np.nanmean(sl_v)), float(np.nanstd(sl_v))
            tl_m, tl_s = float(np.nanmean(tl_v)), float(np.nanstd(tl_v))

            raw = {"tl_vs_sl": self._wilcoxon_raw(tl_v, sl_v)}
            if has_bl:
                raw["tl_vs_bl"] = self._wilcoxon_raw(tl_v, bl_v)
            corrected = self._holm(raw)

            row = [metric_label, f"{sl_m:.3f} ± {sl_s:.3f}"]
            if has_bl:
                row.append(f"{np.nanmean(bl_v):.3f} ± {np.nanstd(bl_v):.3f}")
            row.append(f"{tl_m:.3f} ± {tl_s:.3f}")
            row.append(self._format_pct(sl_m, tl_m))
            if has_bl:
                bl_m = float(np.nanmean(bl_v))
                row.append(self._format_pct(bl_m, tl_m))
                row.append(self._format_delta(bl_m - sl_m))
            row.append(
                f"{self._format_p(raw['tl_vs_sl'])}"
                + (f" [dim]({self._format_p(corrected['tl_vs_sl'])})[/dim]" if has_bl else "")
            )
            if has_bl:
                row.append(
                    f"{self._format_p(raw['tl_vs_bl'])} "
                    f"[dim]({self._format_p(corrected['tl_vs_bl'])})[/dim]"
                )
            t.add_row(*row)
        _console.print(t)

    def generate_champion_comparison_plot(
        self, affinity_types: List[str], output_dir: str
    ) -> None:
        """
        Generate one per-threshold plot (with one panel per affinity type) using
        data collected during ``run_champion``.

        Saves ``fig_champion_comparison_threshold_<N>.png`` to ``output_dir``.
        """
        if not self._champion_data:
            printer.warning("No champion data collected — run run_champion first.")
            return

        thresholds = sorted({t for (_, t) in self._champion_data.keys()})
        plot_gen = self._make_plot_generator(output_dir)

        for threshold in thresholds:
            panel_data = {
                aff: self._champion_data[(aff, threshold)]
                for aff in affinity_types
                if (aff, threshold) in self._champion_data
            }
            if not panel_data:
                continue
            plot_gen.create_champion_comparison_plot(
                champion_data=panel_data,
                affinity_types=affinity_types,
                threshold=threshold,
            )
            for aff in affinity_types:
                if (aff, threshold) in self._champion_data:
                    self._print_champion_summary(
                        aff, threshold, self._champion_data[(aff, threshold)]
                    )

        for aff in affinity_types:
            aff_thresholds = [t for t in thresholds if (aff, t) in self._champion_data]
            if aff_thresholds:
                self._print_pooled_champion_summary(aff, aff_thresholds)

    def run_scatter(
        self, affinity_type: str, threshold: int, output_dir: str
    ) -> None:
        """Generate scatter plots for one threshold / affinity-type combination."""
        plot_gen = self._make_plot_generator(output_dir)
        df_sl_f = self._filter_data(self.df_sl, affinity_type, threshold)
        df_tl_f = self._filter_data(self.df_tl, affinity_type, threshold)
        if not df_sl_f.empty:
            plot_gen.create_supervised_scatter_plot(
                df_sl_f, metric="val/mcc", affinity_type=affinity_type
            )
        if not df_tl_f.empty:
            plot_gen.create_transfer_learning_scatter_plot(
                df_tl_f, metric="val/mcc", affinity_type=affinity_type
            )

    def run_tables(
        self,
        affinity_type: str,
        output_dir: str,
        threshold: int = None,
        verbose: bool = False,
    ) -> None:
        """Generate LaTeX factor-analysis tables.

        If ``threshold`` is None, all thresholds are used together (cross-threshold).
        """
        reporter = self._make_reporter(output_dir)
        plot_gen = self._make_plot_generator(output_dir)

        if threshold is not None:
            df_sl = self._filter_data(self.df_sl, affinity_type, threshold)
            df_tl = self._filter_data(self.df_tl, affinity_type, threshold)
            metrics_sl = self.all_sl_metrics
            metrics_tl = self.all_tl_metrics
        else:
            df_sl = self._filter_data(self.df_sl, affinity_type)
            df_tl = self._filter_data(self.df_tl, affinity_type)
            metrics_sl = [
                m for m in self.all_sl_metrics
                if "loss" not in m and m in df_sl.columns
            ] or self.all_sl_metrics
            metrics_tl = [
                m for m in self.all_tl_metrics if m in df_tl.columns
            ] or self.all_tl_metrics

        self._run_approach_analyses(
            df_sl, self.supervised_configs, metrics_sl,
            "supervised", reporter, plot_gen, affinity_type, verbose,
        )
        self._run_approach_analyses(
            df_tl, self.tl_configs, metrics_tl,
            "transfer", reporter, plot_gen, affinity_type, verbose,
        )

    def run_champion(
        self, affinity_types: List[str], output_dir: str
    ) -> None:
        """Rank cross-threshold champions and generate the champion comparison plot."""
        n_thresholds = self.df_sl["threshold"].nunique()
        if n_thresholds < 2:
            printer.warning(
                f"--champion requires data from at least 2 thresholds for meaningful "
                f"cross-threshold ranking, but only {n_thresholds} threshold(s) are loaded. "
                "Re-run without --thresholds or include all desired thresholds."
            )
            return

        ranker = ModelRanker(
            scoring_function=self.scoring_function,
            rank_method=self.rank_method,
            metric_decimals=self.metric_decimals,
        )

        champion_configs: Dict[str, Any] = {}

        for affinity_type in affinity_types:
            df_sl_aff = self._filter_data(self.df_sl, affinity_type)
            df_tl_aff = self._filter_data(self.df_tl, affinity_type)

            sl_avg_ranks, _ = ranker.rank_models(
                df_sl_aff, affinity_type=affinity_type, label="Supervised Learning"
            )
            tl_avg_ranks, _ = ranker.rank_models(
                df_tl_aff, affinity_type=affinity_type, label="Transfer Learning"
            )

            # The baseline arm is the champion's own configuration retrained with
            # the S1 strategy, so the gap between them isolates pretraining and
            # domain adaptation from every other design choice.
            tl_baseline_exp_id = None
            if not tl_avg_ranks.empty:
                tl_baseline_exp_id = resolve_baseline_exp_id(
                    champion_exp_id=tl_avg_ranks.iloc[0]["exp_id"],
                    champion_method=tl_avg_ranks.iloc[0]["method"],
                    available_exp_ids=set(df_tl_aff["exp_id"]),
                )

            ranker.compare_robust_champions(
                df_sl_aff,
                df_tl_aff,
                sl_avg_ranks,
                tl_avg_ranks,
                tl_baseline_exp_id=tl_baseline_exp_id,
            )

            if not sl_avg_ranks.empty and not tl_avg_ranks.empty:
                self._collect_champion_data(
                    affinity_type,
                    df_sl_aff,
                    df_tl_aff,
                    sl_avg_ranks,
                    tl_avg_ranks,
                    tl_baseline_exp_id=tl_baseline_exp_id,
                )
                try:
                    config: Dict[str, Any] = {
                        "tl": parse_tl_exp_id(tl_avg_ranks.iloc[0]["exp_id"]),
                        "sl": parse_sl_exp_id(sl_avg_ranks.iloc[0]["exp_id"]),
                    }
                    if tl_baseline_exp_id:
                        # Parsed separately: the baseline is auxiliary, so a parse
                        # failure here must not discard the champion config itself.
                        try:
                            config["tl_baseline"] = parse_tl_exp_id(tl_baseline_exp_id)
                        except ValueError as e:
                            printer.warning(
                                f"Could not parse {BASELINE_STRATEGY} baseline exp_id "
                                f"'{tl_baseline_exp_id}': {e}"
                            )
                    config["performance"] = self._compute_global_performance(
                        affinity_type
                    )
                    champion_configs[affinity_type] = config
                except ValueError as e:
                    printer.warning(f"Could not parse champion exp_id for {affinity_type}: {e}")

        self.generate_champion_comparison_plot(affinity_types, output_dir)

        if champion_configs:
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, "champion_config.json")
            with open(json_path, "w") as f:
                json.dump(champion_configs, f, indent=2)
            printer.info(f"Champion config written to: {json_path}")
