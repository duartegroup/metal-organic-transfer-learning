# statistical_tests.py
from itertools import combinations
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from typing import Dict, Any, Tuple, List, Optional
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from constants import BASELINE_STRATEGY
from io_utils import atomic_write
from console import (
    printer,
    print_champion_showdown,
    print_ranking_table,
    print_best_per_threshold,
    print_per_threshold_scores,  # TEMPORARY — investigative only
    print_per_fold_scores,       # TEMPORARY — investigative only
)



def aggregate_folds(
    data: pd.DataFrame, level_cols: List[str], metric: str
) -> pd.DataFrame:
    """
    Aggregate fold-level results to one value per experiment ID and level combination.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing fold-level performance metrics.
    level_cols : list[str]
        List of column names defining the grouping levels (e.g., ['exp_id', 'threshold']).
    metric : str
        The metric column name to aggregate (e.g., 'val/mcc').

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with mean values per experiment and level, unstacked by the last level.
    """
    agg = data.groupby(level_cols)[metric].mean().unstack(level=-1)
    return agg


def perform_friedman_test(
    data: pd.DataFrame, metric: str, verbose: bool = False
) -> Dict[str, Any]:
    """
    Perform Friedman test on fold-aggregated data to compare multiple related groups.

    The Friedman test is a non-parametric statistical test for comparing more than two related
    samples. It tests the null hypothesis that the distributions of all groups are identical.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame where rows are blocks (e.g., exp_id) and columns are groups to compare.
    metric : str
        Name of the metric being tested (used for logging only).
    verbose : bool, optional
        If True, log detailed information about the test. Default is False.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'statistic': Friedman chi-squared statistic
        - 'p_value': p-value of the test
        - 'significant': Boolean indicating if p < 0.05
        - 'mean_ranks': Dictionary of mean ranks for each group
        - 'error': Error message if test failed (only present on error)
    """
    if data.shape[0] < 2:
        if verbose:
            printer.warning(
                "Friedman test requires at least 2 blocks (exp_id). Skipping."
            )
        return {"error": "Not enough data for Friedman test."}

    groups = list(data.columns)
    if verbose:
        printer.info(
            f"Performing Friedman test on metric '{metric}' for groups: {groups}"
        )

    try:
        friedman_stat, friedman_p = friedmanchisquare(*[data[col] for col in groups])
        ranks = data.rank(axis=1, method="average", ascending=False)
        mean_ranks = ranks.mean(axis=0).sort_values(ascending=False)
        is_significant = friedman_p < 0.05

        if verbose:
            printer.info(f"   Friedman χ² = {friedman_stat:.4f}, p = {friedman_p:.6f}")
            conclusion = "REJECT" if is_significant else "FAIL TO REJECT"
            printer.info(
                f"   Conclusion: We {conclusion} the null hypothesis at α=0.05"
            )

        return {
            "statistic": friedman_stat,
            "p_value": friedman_p,
            "significant": is_significant,
            "mean_ranks": mean_ranks.to_dict(),
        }
    except Exception as e:
        if verbose:
            printer.error(f"Friedman test error: {e}")
        return {"error": str(e)}


def perform_pairwise_wilcoxon(
    data: pd.DataFrame, metric: str, correction: str = "holm", verbose: bool = False
) -> Dict[str, Any]:
    """
    Perform pairwise Wilcoxon signed-rank tests with multiple-testing correction.

    Compares all pairs of groups using Wilcoxon signed-rank tests and applies
    multiple-testing correction to control for family-wise error rate.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame where rows are blocks (e.g., exp_id) and columns are groups to compare.
    metric : str
        Name of the metric being tested (used to determine if lower is better).
    correction : str, optional
        Multiple-testing correction method ('holm', 'bonferroni', 'fdr_bh', etc.). Default is 'holm'.
    verbose : bool, optional
        If True, log detailed information about the tests. Default is False.

    Returns
    -------
    dict[str, Any]
        Dictionary where keys are comparison names (e.g., 'group1_vs_group2') and values are dicts containing:
        - 'statistic': Wilcoxon test statistic
        - 'p_value': Raw p-value
        - 'p_value_corrected': Corrected p-value
        - 'significant_corrected': Boolean indicating if corrected p < 0.05
        - 'col1_mean': Mean value for first group
        - 'col2_mean': Mean value for second group
        - 'better': Name of the better-performing group
    """
    groups = list(data.columns)
    comparisons = list(combinations(groups, 2))
    if verbose:
        printer.info(
            f"Performing {len(comparisons)} pairwise Wilcoxon tests with '{correction}' correction"
        )

    raw_pvals, results_tmp = [], []
    for col1, col2 in comparisons:
        try:
            stat, p_val = wilcoxon(data[col1], data[col2])
            results_tmp.append(
                {
                    "key": f"{col1}_vs_{col2}",
                    "stat": stat,
                    "p_val": p_val,
                    "col1": col1,
                    "col2": col2,
                }
            )
            raw_pvals.append(p_val)
        except Exception as e:
            if verbose:
                printer.warning(f"Wilcoxon test failed for {col1} vs {col2}: {e}")
            raw_pvals.append(1.0)

    if not raw_pvals:
        return {}

    reject, pvals_corrected, _, _ = multipletests(
        raw_pvals, alpha=0.05, method=correction
    )
    pairwise_results = {}
    for i, res in enumerate(results_tmp):
        is_significant = reject[i]
        p_val_corr = pvals_corrected[i]
        mean1, mean2 = data[res["col1"]].mean(), data[res["col2"]].mean()
        is_loss = "loss" in metric.lower()
        better_group = (
            res["col1"]
            if (mean1 < mean2 if is_loss else mean1 > mean2)
            else res["col2"]
        )

        pairwise_results[res["key"]] = {
            "statistic": res["stat"],
            "p_value": res["p_val"],
            "p_value_corrected": p_val_corr,
            "significant_corrected": is_significant,
            "col1_mean": mean1,
            "col2_mean": mean2,
            "better": better_group,
        }
    return pairwise_results


def perform_wilcoxon_test(
    data1: pd.Series,
    data2: pd.Series,
    group1_name: str,
    group2_name: str,
    metric: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Perform Wilcoxon signed-rank test for two related (paired) groups.

    The Wilcoxon signed-rank test is a non-parametric test that compares two related samples
    to assess whether their population mean ranks differ.

    Parameters
    ----------
    data1 : pd.Series
        Performance values for the first group.
    data2 : pd.Series
        Performance values for the second group (must be paired with data1).
    group1_name : str
        Name of the first group for reporting.
    group2_name : str
        Name of the second group for reporting.
    metric : str
        Name of the metric being tested (used to determine if lower is better).
    verbose : bool, optional
        If True, log detailed information about the test. Default is False.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'statistic': Wilcoxon test statistic
        - 'p_value': p-value of the test
        - 'significant': Boolean indicating if p < 0.05
        - 'group1_mean': Mean value for group 1
        - 'group2_mean': Mean value for group 2
        - 'better_group': Name of the better-performing group
        - 'error': Error message if test failed (only present on error)
    """
    from scipy.stats import wilcoxon

    if verbose:
        printer.info(
            f"Performing Wilcoxon test on '{metric}' between '{group1_name}' and '{group2_name}'"
        )
        printer.info(f"   H₀: Median difference between paired values is zero")

    try:
        stat, p_val = wilcoxon(data1, data2)
        is_significant = p_val < 0.05

        if verbose:
            printer.info(f"   Wilcoxon W = {stat:.4f}, p = {p_val:.6f}")
            conclusion = "REJECT" if is_significant else "FAIL TO REJECT"
            printer.info(
                f"   Conclusion: We {conclusion} the null hypothesis at α=0.05"
            )

        mean1, mean2 = data1.mean(), data2.mean()
        is_loss_metric = "loss" in metric.lower()
        better_group = (
            group1_name
            if (mean1 < mean2 if is_loss_metric else mean1 > mean2)
            else group2_name
        )

        return {
            "statistic": stat,
            "p_value": p_val,
            "significant": is_significant,
            "group1_mean": mean1,
            "group2_mean": mean2,
            "better_group": better_group,
        }
    except Exception as e:
        printer.error(f"Wilcoxon test failed for {group1_name} vs {group2_name}: {e}")
        return {"error": str(e)}


class ModelRanker:
    """
    Ranks and compares machine learning models across different experimental configurations.

    This class provides utilities for ranking models within thresholds, comparing
    models across configurations, and generating reports on model performance.
    """

    def __init__(
        self,
        scoring_function: str = "mean_mcc_minus_std",
        rank_method: str = "dense",
        metric_decimals: Optional[int] = 3,
    ) -> None:
        self.scoring_function = scoring_function
        self.rank_method = rank_method
        self.metric_decimals = metric_decimals

    def rank_models(
        self, df: pd.DataFrame, affinity_type: str, label: str, metric: str = "val/mcc"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Rank models within each threshold and calculate average ranks across thresholds.

        Averages performance across folds first, then ranks within each threshold.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing model performance results with columns: threshold, exp_id, method, and metric.
        affinity_type : str
            Type of affinity data (e.g., 'pic50', 'pk').
        label : str
            Label for the analysis (e.g., 'Supervised Learning', 'Transfer Learning').
        metric : str, optional
            The performance metric to rank by. Default is 'val/mcc'.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing:
            - avg_ranks_df: DataFrame with average ranks across all thresholds
            - per_threshold_df: DataFrame with ranks per threshold
        """
        from analysis import calculate_ranking_score  # Import the centralized function

        if df.empty:
            printer.warning(f"No data found for {label}, skipping.")
            return pd.DataFrame(), pd.DataFrame()

        df = df.copy()

        # 1. Average over folds to get a single performance score per model per threshold
        df_avg = (
            df.groupby(["threshold", "exp_id", "method"])
            .agg({metric: ["mean", "std"]})
            .reset_index()
        )
        df_avg.columns = ["threshold", "exp_id", "method", "mean", "std"]

        # 2. Calculate the ranking metric
        df_avg["ranking_metric"] = df_avg.apply(
            lambda row: calculate_ranking_score(
                row["mean"], row["std"], self.scoring_function
            ),
            axis=1,
        )

        # Warn about NaN metrics — models excluded from some thresholds get inflated ranks
        nan_mask = df_avg["ranking_metric"].isna()
        if nan_mask.any():
            nan_rows = df_avg[nan_mask][["threshold", "exp_id"]]
            printer.warning(
                f"[{label}] {nan_mask.sum()} (threshold, model) combination(s) have NaN "
                f"metrics and will be excluded from those thresholds. "
                f"This can artificially inflate their average rank."
            )
            for _, row in nan_rows.iterrows():
                printer.warning(f"  NaN at threshold={row['threshold']}: {row['exp_id']}")

        # 3. Rank within each threshold
        if self.metric_decimals is not None:
            df_avg["ranking_metric"] = df_avg["ranking_metric"].round(self.metric_decimals)

        df_avg["rank"] = df_avg.groupby("threshold")["ranking_metric"].rank(
            ascending=False, method=self.rank_method
        )

        # Show best model per threshold
        best_per_threshold = df_avg.loc[df_avg["rank"] == 1].sort_values("threshold")
        print_best_per_threshold(best_per_threshold, label)

        # 4. Calculate average ranks across all thresholds
        n_thresholds = df_avg["threshold"].nunique()
        avg_ranks = (
            df_avg.groupby("exp_id")
            .agg(
                avg_rank=("rank", "mean"),
                avg_score=("ranking_metric", "mean"),
                method=("method", "first"),
                n_ranked=("rank", "count"),
            )
            .sort_values("avg_rank")
            .reset_index()
        )

        partial = avg_ranks[avg_ranks["n_ranked"] < n_thresholds]
        if not partial.empty:
            printer.warning(
                f"[{label}] {len(partial)} model(s) ranked on fewer than "
                f"{n_thresholds} threshold(s) due to missing data — "
                f"their avg rank may be misleading:"
            )
            for _, row in partial.iterrows():
                printer.warning(
                    f"  {row['exp_id']}  "
                    f"(ranked on {int(row['n_ranked'])}/{n_thresholds} thresholds)"
                )

        avg_ranks = avg_ranks.drop(columns=["n_ranked"])

        top = avg_ranks.head(10).copy()

        # Rename columns
        top = top.rename(
            columns={
                "exp_id": "Model",
                "avg_rank": "Rank",
                "avg_score": "Score",
                "method": "Method",
            }
        )

        # Round numbers
        top["Rank"] = top["Rank"].round(2)
        top["Score"] = top["Score"].round(4)

        # Render top-10 ranking via rich table
        print_ranking_table(top, label)

        print_per_threshold_scores(list(top["Model"]), df_avg, label)

        # Generate parameter breakdown table for top 5 models
        self.generate_parameter_breakdown_table(top, label, df)

        return avg_ranks, df_avg

    def generate_parameter_breakdown_table(
        self, top_models_df: pd.DataFrame, label: str, source_df: pd.DataFrame
    ) -> None:
        """
        Generate a parameter breakdown table for the top 5 models.

        Looks up parameter values directly from the source DataFrame using the
        exp_id, avoiding fragile string parsing of underscore-joined model names.
        Also generates a LaTeX version of the table for publication.

        Parameters
        ----------
        top_models_df : pd.DataFrame
            DataFrame containing top models with columns: Model, Rank, Score, Method.
        label : str
            Label for the analysis (used for table headers and filenames).
        source_df : pd.DataFrame
            The original DataFrame used for ranking, containing all parameter columns.
        """
        # Define parameter mappings for both learning types
        sl_params = [
            "inhibitor",
            "modality",
            "scaler",
            "method",
            "target_smote",
            "pca_status",
        ]
        tl_params = [
            "inhibitor",
            "modality",
            "scaler",
            "method",
            "cross_domain_scaling",
            "source_smote",
            "target_smote",
        ]

        is_transfer_learning = "Transfer" in label
        param_names = tl_params if is_transfer_learning else sl_params

        top_5 = top_models_df.head(5).copy()
        if top_5.empty:
            printer.warning(f"No models found for parameter breakdown table - {label}")
            return

        # Look up parameter values directly from source_df by exp_id
        available_params = [p for p in param_names if p in source_df.columns]
        breakdown_data = []
        for _, row in top_5.iterrows():
            exp_id = row["Model"]
            match = source_df[source_df["exp_id"] == exp_id]
            if match.empty:
                printer.warning(f"exp_id '{exp_id}' not found in source data — skipping.")
                continue
            first = match.iloc[0]
            param_values = {p: first[p] for p in available_params}
            # Fill any missing params
            for p in param_names:
                param_values.setdefault(p, "n/a")
            param_values["Rank"] = row["Rank"]
            param_values["Score"] = row["Score"]
            breakdown_data.append(param_values)

        if not breakdown_data:
            printer.warning(f"Could not build parameter breakdown for {label}")
            return

        breakdown_df = pd.DataFrame(breakdown_data)
        cols = ["Rank", "Score"] + param_names
        breakdown_df = breakdown_df[cols]

        # Render as a rich table
        from console import _console
        from rich.rule import Rule
        from rich.table import Table as _Table
        from rich import box as _box
        _console.print(Rule(f"[bold]Top 5 parameter breakdown — {label}[/bold]"))
        bt = _Table(box=_box.SIMPLE_HEAVY, show_header=True, header_style="bold")
        for col in breakdown_df.columns:
            bt.add_column(str(col), no_wrap=False)
        for _, brow in breakdown_df.iterrows():
            bt.add_row(*[str(v) for v in brow])
        _console.print(bt)

        # Generate LaTeX table
        self._generate_latex_parameter_breakdown_table(breakdown_df, label, param_names)

    def _parse_model_name_to_params(
        self, parts: List[str], param_names: List[str], is_transfer_learning: bool
    ) -> Dict[str, str]:
        """
        Parse model name components into structured parameter values.

        Parameters
        ----------
        parts : list[str]
            Model name split into component parts.
        param_names : list[str]
            Expected parameter names for this model type.
        is_transfer_learning : bool
            Whether the model is a transfer learning model (affects parsing structure).

        Returns
        -------
        dict[str, str]
            Dictionary mapping parameter names to their values.
        """
        param_values = {}

        if is_transfer_learning:
            # Transfer learning model name structure:
            # [inhibitor]_[modality_parts...]_[scaler]_[method_parts...]_[cross_domain]_[source_smote]_[target_smote]
            try:
                param_values["inhibitor"] = parts[0] if len(parts) > 0 else "unknown"

                # Find scaler position first
                scaler_idx = None
                for i, part in enumerate(parts[1:], 1):
                    if part in [
                        "none_scaler",
                        "minmax_scaler",
                        "standard_scaler",
                        "None",
                    ]:
                        scaler_idx = i
                        break

                if scaler_idx is not None:
                    # Modality is everything between inhibitor and scaler
                    modality_parts = parts[1:scaler_idx]
                    param_values["modality"] = (
                        "_".join(modality_parts) if modality_parts else "unknown"
                    )
                    param_values["scaler"] = parts[scaler_idx]

                    # Method is everything between scaler and last 3 parts
                    method_start = scaler_idx + 1
                    method_end = len(parts) - 3
                    if method_start < method_end:
                        method_parts = parts[method_start:method_end]
                        param_values["method"] = "_".join(method_parts)
                    else:
                        param_values["method"] = "unknown"

                    # Last 3 parts are cross_domain_scaling, source_smote, target_smote
                    if len(parts) >= 3:
                        param_values["cross_domain_scaling"] = parts[-3]
                        param_values["source_smote"] = parts[-2]
                        param_values["target_smote"] = parts[-1]
                    else:
                        param_values["cross_domain_scaling"] = "unknown"
                        param_values["source_smote"] = "unknown"
                        param_values["target_smote"] = "unknown"
                else:
                    # Fallback parsing
                    for param in param_names:
                        param_values[param] = "parse_error"

            except Exception as e:
                printer.warning(f"Error parsing TL model name {parts}: {e}")
                for param in param_names:
                    param_values[param] = "parse_error"
        else:
            # Supervised learning model name structure:
            # [inhibitor]_[modality_parts...]_[scaler]_[method]_[target_smote]_[pca_status]
            try:
                param_values["inhibitor"] = parts[0] if len(parts) > 0 else "unknown"

                # Find scaler position
                scaler_idx = None
                for i, part in enumerate(parts[1:], 1):
                    if part in ["none_scaler", "minmax_scaler", "standard_scaler"]:
                        scaler_idx = i
                        break

                if scaler_idx is not None:
                    # Modality is everything between inhibitor and scaler
                    modality_parts = parts[1:scaler_idx]
                    param_values["modality"] = (
                        "_".join(modality_parts) if modality_parts else "unknown"
                    )
                    param_values["scaler"] = parts[scaler_idx]

                    # Method is right after scaler
                    if scaler_idx + 1 < len(parts):
                        param_values["method"] = parts[scaler_idx + 1]
                    else:
                        param_values["method"] = "unknown"

                    # Last 2 parts are target_smote and pca_status
                    if len(parts) >= 2:
                        param_values["target_smote"] = parts[-2]
                        param_values["pca_status"] = parts[-1]
                    else:
                        param_values["target_smote"] = "unknown"
                        param_values["pca_status"] = "unknown"
                else:
                    # Fallback parsing
                    for param in param_names:
                        param_values[param] = "parse_error"

            except Exception as e:
                printer.warning(f"Error parsing SL model name {parts}: {e}")
                for param in param_names:
                    param_values[param] = "parse_error"

        return param_values

    def _generate_latex_parameter_breakdown_table(
        self, breakdown_df: pd.DataFrame, label: str, param_names: List[str]
    ) -> None:
        """
        Generate a LaTeX table for the parameter breakdown and save it to file.

        Parameters
        ----------
        breakdown_df : pd.DataFrame
            DataFrame containing parameter breakdown for top models.
        label : str
            Label for the analysis (used in filename and caption).
        param_names : list[str]
            List of parameter names to include in the table.
        """
        import os

        # Determine output directory and filename
        is_transfer_learning = "Transfer" in label
        learning_type = (
            "transfer_learning" if is_transfer_learning else "supervised_learning"
        )

        # Create results directory if it doesn't exist
        import os as _os
        results_dir = _os.path.join(
            _os.path.dirname(__file__), "..", "..", "results", "statistical_tests", "global_analysis"
        )
        os.makedirs(results_dir, exist_ok=True)

        # Generate filename
        filename = f"table_top5_parameter_breakdown_{learning_type}.tex"
        filepath = os.path.join(results_dir, filename)

        # Create LaTeX table content
        latex_content = self._create_latex_parameter_table(
            breakdown_df, label, param_names, is_transfer_learning
        )

        # Write to file
        try:
            atomic_write(filepath, latex_content)
            printer.info(f"LaTeX parameter breakdown table saved to: {filepath}")
        except Exception as e:
            printer.error(f"Failed to save LaTeX parameter breakdown table: {e}")

    def _create_latex_parameter_table(
        self,
        breakdown_df: pd.DataFrame,
        label: str,
        param_names: List[str],
        is_transfer_learning: bool,
    ) -> str:
        """
        Create the LaTeX table content for parameter breakdown.

        Parameters
        ----------
        breakdown_df : pd.DataFrame
            DataFrame containing parameter breakdown data.
        label : str
            Label for the table caption.
        param_names : list[str]
            List of parameter names to include.
        is_transfer_learning : bool
            Whether this is for transfer learning models.

        Returns
        -------
        str
            Complete LaTeX table code as a string.
        """
        # Clean up parameter names for LaTeX headers
        header_map = {
            "Rank": "Rank",
            "Score": "Score",
            "inhibitor": "Descriptor",
            "modality": "Modality",
            "scaler": "Scaler",
            "method": "Method",
            "target_smote": "Target SMOTE",
            "pca_status": "PCA",
            "cross_domain_scaling": "Cross-Domain",
            "source_smote": "Source SMOTE",
        }

        # Create column headers
        columns = ["Rank", "Score"] + param_names
        headers = [
            header_map.get(col, col.replace("_", " ").title()) for col in columns
        ]

        # Determine column alignment - center for most, left for longer text
        if is_transfer_learning:
            alignment = "c" * len(columns)  # All centered for TL
        else:
            alignment = "c" * len(columns)  # All centered for SL too

        # Start LaTeX table
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Top 5 Models Parameter Breakdown - {label}}}",
            f"\\begin{{tabular}}{{{alignment}}}",
            "\\toprule",
        ]

        # Add headers
        header_line = " & ".join([f"\\textbf{{{h}}}" for h in headers]) + " \\\\"
        latex_lines.append(header_line)
        latex_lines.append("\\midrule")

        # Add data rows
        for idx, row in breakdown_df.iterrows():
            row_data = []
            for col in columns:
                value = str(row[col])
                # Clean up values for LaTeX
                value = self._clean_latex_value(value, col)
                row_data.append(value)

            row_line = " & ".join(row_data) + " \\\\"
            latex_lines.append(row_line)

        # End table
        latex_lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                f"\\label{{tab:parameter_breakdown_{label.lower().replace(' ', '_')}}}",
                "\\end{table}",
            ]
        )

        return "\n".join(latex_lines)

    def _clean_latex_value(self, value: str, column: str) -> str:
        """
        Clean and format values for LaTeX output.

        Handles numeric formatting, escapes special characters, and maps common
        values to more readable forms.

        Parameters
        ----------
        value : str
            The value to clean and format.
        column : str
            The column name (affects formatting strategy).

        Returns
        -------
        str
            Cleaned and formatted value ready for LaTeX.
        """
        # Handle numeric values
        if column in ["Rank", "Score"]:
            try:
                float_val = float(value)
                if column == "Rank":
                    return f"{float_val:.2f}"
                else:  # Score
                    return f"{float_val:.4f}"
            except ValueError:
                pass

        # Clean up text values
        value = value.replace("_", "\\_")  # Escape underscores
        value = value.replace("&", "\\&")  # Escape ampersands
        value = value.replace("%", "\\%")  # Escape percent signs

        # Map common values to more readable forms
        value_map = {
            "no\\_target\\_smote": "No",
            "target\\_smote": "Yes",
            "no\\_source\\_smote": "No",
            "source\\_smote": "Yes",
            "cross\\_domain": "Yes",
            "separate\\_domain": "No",
            "with\\_pca\\_50components": "Yes (50)",
            "no\\_pca": "No",
            "none\\_scaler": "None",
            "minmax\\_scaler": "MinMax",
            "standard\\_scaler": "Standard",
        }

        return value_map.get(value, value)

    def compare_robust_champions(
        self,
        df_sl: pd.DataFrame,
        df_tl: pd.DataFrame,
        sl_avg_ranks_df: pd.DataFrame,
        tl_avg_ranks_df: pd.DataFrame,
        metric: str = "val/mcc",
        tl_baseline_exp_id: Optional[str] = None,
    ) -> None:
        """
        Compare the most robust supervised learning and transfer learning models.

        Implements the "Analysis of Ranks" methodology:
        1. Identifies the single most robust model (lowest average rank) for each approach
        2. Aligns their fold-level scores by (threshold, fold) to form matched pairs
        3. Compares the paired scores using the Wilcoxon signed-rank test

        The Wilcoxon signed-rank test is appropriate here because each pair shares
        the same cross-validation fold and classification threshold, making the
        observations paired rather than independent.

        Parameters
        ----------
        df_sl : pd.DataFrame
            DataFrame containing all supervised learning results.
        df_tl : pd.DataFrame
            DataFrame containing all transfer learning results.
        sl_avg_ranks_df : pd.DataFrame
            DataFrame with average ranks for supervised learning models.
        tl_avg_ranks_df : pd.DataFrame
            DataFrame with average ranks for transfer learning models.
        metric : str, optional
            The performance metric to compare. Default is 'val/mcc'.
        tl_baseline_exp_id : str, optional
            exp_id of the transfer learning baseline: the champion's own
            configuration trained with the baseline strategy. When given it is
            joined onto the same (threshold, fold) pairs and tested against the
            TL champion as well, so the gap attributable to the training
            strategy alone can be read off next to the SL comparison.
        """
        from scipy.stats import wilcoxon as wilcoxon_test

        if sl_avg_ranks_df.empty or tl_avg_ranks_df.empty:
            printer.warning(
                "Average ranks data missing for SL or TL. Cannot perform comparison."
            )
            return

        df_sl = df_sl.copy()
        df_tl = df_tl.copy()

        # --- Step 1: Identify the Single Most Robust Model for Each Approach ---
        sl_champion_id = sl_avg_ranks_df.iloc[0]["exp_id"]
        tl_champion_id = tl_avg_ranks_df.iloc[0]["exp_id"]

        sl_champion_avg_rank = sl_avg_ranks_df.iloc[0]["avg_rank"]
        tl_champion_avg_rank = tl_avg_ranks_df.iloc[0]["avg_rank"]

        printer.info(
            f"SL champion: '{sl_champion_id}' (avg rank {sl_champion_avg_rank:.2f})"
        )
        printer.info(
            f"TL champion: '{tl_champion_id}' (avg rank {tl_champion_avg_rank:.2f})"
        )

        # --- Step 2: Build paired arrays aligned on (threshold, fold) ---
        sl_raw = df_sl[df_sl["exp_id"] == sl_champion_id][
            ["threshold", "fold", metric]
        ].dropna()
        tl_raw = df_tl[df_tl["exp_id"] == tl_champion_id][
            ["threshold", "fold", metric]
        ].dropna()

        sl_raw = sl_raw.copy()
        tl_raw = tl_raw.copy()
        sl_raw["threshold"] = sl_raw["threshold"].astype(str)
        tl_raw["threshold"] = tl_raw["threshold"].astype(str)
        sl_raw["fold"] = sl_raw["fold"].astype(str)
        tl_raw["fold"] = tl_raw["fold"].astype(str)

        paired = pd.merge(
            sl_raw.rename(columns={metric: "sl"}),
            tl_raw.rename(columns={metric: "tl"}),
            on=["threshold", "fold"],
            how="inner",
        )

        # Optional third arm: joined left so a missing baseline can never shrink
        # the SL/TL pair set that the primary comparison depends on.
        if tl_baseline_exp_id:
            bl_raw = df_tl[df_tl["exp_id"] == tl_baseline_exp_id][
                ["threshold", "fold", metric]
            ].dropna().copy()
            bl_raw["threshold"] = bl_raw["threshold"].astype(str)
            bl_raw["fold"] = bl_raw["fold"].astype(str)
            paired = pd.merge(
                paired,
                bl_raw.rename(columns={metric: "bl"}),
                on=["threshold", "fold"],
                how="left",
            )

        printer.info(
            f"Paired scores: {len(paired)} (SL raw: {len(sl_raw)}, TL raw: {len(tl_raw)})"
        )

        if paired.empty:
            printer.error(
                "No (threshold, fold) pairs overlap between champions. Cannot perform test."
            )
            return

        # --- Step 3 & 4: Rich champion showdown (breakdown + test) ---
        try:
            statistic, p_value = wilcoxon_test(
                paired["sl"], paired["tl"], alternative="two-sided"
            )

            bl_statistic = bl_p_value = None
            if "bl" in paired.columns:
                usable = paired.dropna(subset=["bl", "tl"])
                if usable.empty:
                    printer.warning(
                        f"Baseline '{tl_baseline_exp_id}' overlaps no folds with the "
                        "TL champion — skipping the baseline test."
                    )
                elif (usable["tl"] - usable["bl"]).abs().sum() == 0:
                    printer.warning(
                        "TL champion and baseline scores are identical on every "
                        "fold — no baseline test to run."
                    )
                else:
                    bl_statistic, bl_p_value = wilcoxon_test(
                        usable["bl"], usable["tl"], alternative="two-sided"
                    )

            print_champion_showdown(
                paired=paired,
                sl_champion_id=sl_champion_id,
                tl_champion_id=tl_champion_id,
                sl_avg_rank=sl_champion_avg_rank,
                tl_avg_rank=tl_champion_avg_rank,
                statistic=statistic,
                p_value=p_value,
                metric=metric,
                bl_champion_id=tl_baseline_exp_id,
                bl_label=BASELINE_STRATEGY,
                bl_statistic=bl_statistic,
                bl_p_value=bl_p_value,
            )
        except Exception as e:
            printer.error(f"Error during Wilcoxon signed-rank test: {e}")
