from console import printer
# reporting.py
import os
from typing import Dict, Any, List, Set, Union
import pandas as pd
import numpy as np
from max_value_extractor import MaxValueExtractor
from constants import (
    STRATEGY_MAP,
    MODALITY_MAP,
    SCALER_MAP,
    SORT_ORDER_MAP,
    FALLBACK_MAX_VALUES,
)


class Reporter:
    """Handles the generation of reports, summaries, and LaTeX tables."""
    def __init__(self, output_dir: str, noise_results_dir: str = "noise_estimation",
                 scoring_function: str = "mean_mcc_minus_std", verbose_reporting: bool = True,
                 log_table_generation: bool = True, include_pvalues_in_tables: bool = True):
        """Initializes the Reporter and creates the output directory."""
        self.scoring_function = scoring_function
        self.verbose_reporting = verbose_reporting
        self.log_table_generation = log_table_generation
        self.include_pvalues_in_tables = include_pvalues_in_tables

        self.sl_output_dir = os.path.join(output_dir, "supervised")
        self.tl_output_dir = os.path.join(output_dir, "transfer")
        os.makedirs(self.sl_output_dir, exist_ok=True)
        os.makedirs(self.tl_output_dir, exist_ok=True)

        self.max_value_extractor = MaxValueExtractor(noise_results_dir)

    # --- Mappings for consistent name formatting (imported from constants.py) ---
    STRATEGY_MAP = STRATEGY_MAP
    MODALITY_MAP = MODALITY_MAP
    SCALER_MAP = SCALER_MAP
    SORT_ORDER_MAP = SORT_ORDER_MAP

    MAX_MCC_VALUES_PK    = FALLBACK_MAX_VALUES["pk"]["mcc"]
    MAX_ACC_VALUES_PK    = FALLBACK_MAX_VALUES["pk"]["accuracy"]
    MAX_F1_VALUES_PK     = FALLBACK_MAX_VALUES["pk"]["f1"]
    MAX_MCC_VALUES_PIC50 = FALLBACK_MAX_VALUES["pic50"]["mcc"]
    MAX_ACC_VALUES_PIC50 = FALLBACK_MAX_VALUES["pic50"]["accuracy"]
    MAX_F1_VALUES_PIC50  = FALLBACK_MAX_VALUES["pic50"]["f1"]


    def report_overall_comparison(self, results_by_metric: Dict[str, Any], config: Dict[str, Any], affinity_type: str, metrics: List[str], learning_type: str = "supervised"):
        """Generates a single LaTeX table summarizing the overall comparison across multiple metrics."""
        name = config['name']
        analysis_var = config['analysis_variable']

        title = f"Overall Effect of {name} ({learning_type.title()})"
        if self.log_table_generation:
            printer.info("="*10 + f" {title} Summary & LaTeX Table " + "="*10)

        # Collect all variables from both two-group and multi-group analyses
        all_vars = set()
        for metric_res in results_by_metric.values():
            # Handle multi-group analysis results (Friedman test)
            if 'group_stats' in metric_res:
                all_vars.update(metric_res['group_stats'].keys())
            # Handle two-group analysis results (Wilcoxon test)
            elif 'group_means' in metric_res:
                all_vars.update(metric_res['group_means'].keys())

        if self.verbose_reporting:
            printer.info(f"Collected variables: {all_vars}")

        if not all_vars:
            if self.verbose_reporting:
                printer.warning(f"No data to generate LaTeX table for {title}.")
            return

        # Get the specific sort order for the current analysis variable
        sort_order = self.SORT_ORDER_MAP.get(analysis_var, [])

        # Sort the collected variables based on the defined order
        if sort_order:
            order_map = {val: i for i, val in enumerate(sort_order)}
            sorted_vars = sorted(list(all_vars), key=lambda x: order_map.get(x, float('inf')))
        else:
            sorted_vars = sorted(list(all_vars)) # Fallback to alphabetical

        if self.verbose_reporting:
            printer.info(f"Sorted variables: {sorted_vars}")

        metric_order = [m for m in ['val/loss', 'val/mcc', 'val/accuracy', 'val/f1'] if m in results_by_metric]

        caption = (f"Overall performance comparison for different {name.lower()} settings within each {learning_type} experiment on the {self._format_affinity_type(affinity_type)} dataset. "
                   "Values are mean $\\pm$ standard deviation. For each metric, statistically equivalent best performing "
                   "variables are denoted with the same superscript letter.")

        label = f"tab:overall_{name.lower().replace(' ', '_')}_{learning_type}_{affinity_type.lower()}"

        latex_table = self._generate_multi_metric_latex_table(
            results_by_metric=results_by_metric,
            sorted_vars=sorted_vars,
            metric_order=metric_order,
            caption=caption,
            label=label,
            include_pvalues=self.include_pvalues_in_tables
        )

        # Choose the appropriate output directory based on learning type
        output_dir = self.sl_output_dir if learning_type.lower() == "supervised" else self.tl_output_dir
        filename = f"table_overall_{name.lower().replace(' ', '_').replace('/', '_')}_{affinity_type.lower()}.tex"
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            f.write(latex_table)

        if self.log_table_generation:
            printer.info(f"Saved LaTeX table for {title} to {output_path}")

        self._save_overall_comparison_csv(
            results_by_metric=results_by_metric,
            metric_order=metric_order,
            sorted_vars=sorted_vars,
            name=name,
            affinity_type=affinity_type,
            learning_type=learning_type,
            output_dir=output_dir,
        )

    def _save_overall_comparison_csv(
        self,
        results_by_metric: Dict[str, Any],
        metric_order: List[str],
        sorted_vars: List[str],
        name: str,
        affinity_type: str,
        learning_type: str,
        output_dir: str,
    ) -> None:
        """Save statistical test results (group stats + p-values) to a CSV file."""
        rows: List[Dict] = []
        base = {
            "analysis_name": name,
            "affinity_type": affinity_type,
            "learning_type": learning_type,
        }

        for metric in metric_order:
            if metric not in results_by_metric:
                continue
            result = results_by_metric[metric]
            group_stats = result.get("group_stats", {})
            test_type   = result.get("test_type", "")

            # Determine which variables are "best" (matching LaTeX superscript logic)
            is_loss = "loss" in metric.lower()
            if group_stats:
                group_means = {v: s["mean"] for v, s in group_stats.items()}
                top = min(group_means, key=group_means.get) if is_loss else max(group_means, key=group_means.get)
                if test_type == "wilcoxon":
                    w = result.get("wilcoxon_test", {})
                    best_set = {top} if w.get("p_value", 1.0) < 0.05 else set(group_means)
                else:
                    best_set = set(result.get("best_groups_by_stat", []))
            else:
                best_set = set()

            # Group-stat rows
            for var in sorted_vars:
                if var not in group_stats:
                    continue
                stats = group_stats[var]
                rows.append({
                    **base,
                    "row_type":           "group_stat",
                    "variable":           var,
                    "variable_2":         "",
                    "metric":             metric,
                    "mean":               round(stats.get("mean", float("nan")), 6),
                    "std":                round(stats.get("std",  float("nan")), 6),
                    "p_value":            "",
                    "p_value_corrected":  "",
                    "is_significant":     "",
                    "is_best":            var in best_set,
                })

            # Overall test row
            if test_type == "wilcoxon":
                w     = result.get("wilcoxon_test", {})
                p_val = w.get("p_value", "")
                is_sig = w.get("significant", "")
            else:
                f     = result.get("friedman_test", {})
                p_val = f.get("p_value", "")
                is_sig = f.get("significant", "")

            rows.append({
                **base,
                "row_type":          "overall_test",
                "variable":          "",
                "variable_2":        "",
                "metric":            metric,
                "mean":              "",
                "std":               "",
                "p_value":           round(p_val, 6) if isinstance(p_val, float) else p_val,
                "p_value_corrected": "",
                "is_significant":    is_sig,
                "is_best":           "",
            })

            # Pairwise rows (only present for Friedman follow-up)
            for pair_key, pair_res in (result.get("pairwise_tests") or {}).items():
                parts = pair_key.split("_vs_", 1)
                var1, var2 = (parts[0], parts[1]) if len(parts) == 2 else (pair_key, "")
                p_raw  = pair_res.get("p_value", "")
                p_corr = pair_res.get("p_value_corrected", "")
                rows.append({
                    **base,
                    "row_type":          "pairwise",
                    "variable":          var1,
                    "variable_2":        var2,
                    "metric":            metric,
                    "mean":              "",
                    "std":               "",
                    "p_value":           round(p_raw,  6) if isinstance(p_raw,  float) else p_raw,
                    "p_value_corrected": round(p_corr, 6) if isinstance(p_corr, float) else p_corr,
                    "is_significant":    p_corr < 0.05 if isinstance(p_corr, float) else "",
                    "is_best":           "",
                })

        if not rows:
            return

        df_csv = pd.DataFrame(rows)
        safe_name = name.lower().replace(" ", "_").replace("/", "_")
        csv_path  = os.path.join(output_dir, f"stats_{safe_name}_{affinity_type.lower()}.csv")
        df_csv.to_csv(csv_path, index=False)
        if self.log_table_generation:
            printer.info(f"Saved CSV for {name} to {csv_path}")

    def _generate_multi_metric_latex_table(self, results_by_metric: Dict[str, Any], sorted_vars: List[str],
                                     metric_order: List[str], caption: str, label: str,
                                     include_pvalues: bool = True) -> str:
        """
        Generates a consolidated LaTeX table comparing variables across multiple metrics,
        using superscripts to denote statistically equivalent best groups.
        """
        stat_groups: Dict[str, Set[str]] = {}
        metric_letters: Dict[str, str] = {}
        next_letter_code = ord('A')

        for metric in metric_order:
            if metric in results_by_metric:
                result = results_by_metric[metric]
                best_groups = []

                # Both two-group and multi-group analyses now have group_stats
                if 'group_stats' in result:
                    # Determine if this is a loss metric (lower is better) or performance metric (higher is better)
                    is_loss_metric = 'loss' in metric.lower()

                    # Get the best group based on mean performance
                    group_means = {var: stats['mean'] for var, stats in result['group_stats'].items()}
                    if is_loss_metric:
                        best_group = min(group_means, key=group_means.get)
                    else:
                        best_group = max(group_means, key=group_means.get)

                    # Check if the difference is statistically significant
                    if result.get('test_type') == 'wilcoxon':
                        # For two-group comparisons, check Wilcoxon significance
                        wilcoxon_test = result.get('wilcoxon_test', {})
                        is_significant = wilcoxon_test.get('p_value', 1.0) < 0.05

                        if is_significant:
                            best_groups = [best_group]
                        else:
                            # If not significant, both groups are equivalent
                            best_groups = list(group_means.keys())
                    else:
                        # For multi-group comparisons, use the pre-calculated best groups
                        best_groups = result.get('best_groups_by_stat', [])

                if best_groups:
                    stat_groups[metric] = set(best_groups)
                    metric_letters[metric] = chr(next_letter_code)
                    next_letter_code += 1

        metric_headers = {m: m.replace('val/', '').upper() for m in metric_order}
        column_format = "l" * (len(metric_order) + 1)  # +1 for the Variable column
        column_titles = " & ".join(
            [f"\\multicolumn{{1}}{{c}}{{\\textbf{{{header}}}}}" for header in metric_headers.values()]
        )

        header = (
            "\\begin{table}[H]\n"
            "    \\centering\n"
            "    \\small\n"
            f"    \\caption{{{caption}}}\n"
            f"    \\label{{{label}}}\n"
            "    \\resizebox{\\textwidth}{!}{%\n"
            f"    \\begin{{tabular}}{{{column_format}}}\n"
            "        \\toprule\n"
            f"        \\multicolumn{{1}}{{c}}{{\\textbf{{Variable}}}} & {column_titles} \\\\\n"
            "        \\midrule"
        )

        rows = []
        for var_name in sorted_vars:
            row_cells = [f"\\textbf{{{self._format_name(var_name)}}}"]
            for metric in metric_order:
                cell_text = "N/A"
                if metric in results_by_metric:
                    result = results_by_metric[metric]

                    # Now we can always use group_stats since both analysis types provide it
                    if 'group_stats' in result and var_name in result['group_stats']:
                        stats = result['group_stats'][var_name]
                        mean, std = stats.get('mean', 0), stats.get('std', 0)

                        superscript = ""
                        if metric in stat_groups and var_name in stat_groups[metric]:
                            letter = metric_letters[metric]
                            superscript = r"$^{\textit{\textbf{" + letter + r"}}}$"

                        cell_text = f"{mean:.3f} $\\pm$ {std:.3f}{superscript}"

                row_cells.append(cell_text)
            rows.append(" & ".join(row_cells) + " \\\\")

        # Add p-values section at the end if requested
        if include_pvalues:
            p_value_rows = []
            p_value_rows.append("        \\midrule")
            p_value_rows.append("        \\multicolumn{" + str(len(metric_order) + 1) + "}{c}{\\textbf{Statistical Test Results}} \\\\")
            p_value_rows.append("        \\midrule")
            
            # First row: Overall test p-values (Friedman or Wilcoxon)
            overall_row_cells = ["\\textbf{Overall p-value}"]
            for metric in metric_order:
                p_val_text = "N/A"
                if metric in results_by_metric:
                    result = results_by_metric[metric]
                    
                    if result.get('test_type') == 'wilcoxon':
                        # Two-group comparison - show Wilcoxon p-value
                        wilcoxon_test = result.get('wilcoxon_test', {})
                        p_val = wilcoxon_test.get('p_value')
                        if p_val is not None:
                            if p_val < 0.001:
                                p_val_text = "$< 0.001$"
                            else:
                                p_val_text = f"{p_val:.3f}"
                    elif 'friedman_test' in result:
                        # Multi-group comparison - show Friedman p-value
                        friedman_test = result.get('friedman_test', {})
                        p_val = friedman_test.get('p_value')
                        if p_val is not None:
                            if p_val < 0.001:
                                p_val_text = "$< 0.001$"
                            else:
                                p_val_text = f"{p_val:.3f}"
                
                overall_row_cells.append(p_val_text)
            
            p_value_rows.append("        " + " & ".join(overall_row_cells) + " \\\\")
            
            # Add pairwise Wilcoxon p-values for multi-group comparisons
            if len(sorted_vars) > 2:
                # Check if we have pairwise results
                has_pairwise = False
                for metric in metric_order:
                    if metric in results_by_metric:
                        result = results_by_metric[metric]
                        if result.get('pairwise_tests'):
                            has_pairwise = True
                            break
                
                if has_pairwise:
                    from itertools import combinations
                    pairwise_comparisons = list(combinations(sorted_vars, 2))
                    
                    for comp1, comp2 in pairwise_comparisons:
                        pairwise_row_cells = [f"\\textbf{{{self._format_name(comp1)} vs {self._format_name(comp2)}}}"]
                        
                        for metric in metric_order:
                            p_val_text = "N/A"
                            if metric in results_by_metric:
                                result = results_by_metric[metric]
                                pairwise_tests = result.get('pairwise_tests', {})
                                
                                # Try both possible key formats
                                key1 = f"{comp1}_vs_{comp2}"
                                key2 = f"{comp2}_vs_{comp1}"
                                
                                pairwise_result = pairwise_tests.get(key1) or pairwise_tests.get(key2)
                                
                                if pairwise_result:
                                    # Use corrected p-value (Holm correction)
                                    p_val = pairwise_result.get('p_value_corrected')
                                    if p_val is not None:
                                        if p_val < 0.001:
                                            p_val_text = "$< 0.001$"
                                        else:
                                            p_val_text = f"{p_val:.3f}"
                            
                            pairwise_row_cells.append(p_val_text)
                        
                        p_value_rows.append("        " + " & ".join(pairwise_row_cells) + " \\\\")
            
            # Best performer interpretation row
            interpretation_row_cells = ["\\textbf{Best performer}"]
            for metric in metric_order:
                interpretation_text = "N/A"
                if metric in results_by_metric:
                    result = results_by_metric[metric]
                    
                    if 'group_stats' in result:
                        # Get the best groups (those with superscripts)
                        best_groups = []
                        if metric in stat_groups:
                            best_groups = list(stat_groups[metric])
                        
                        if len(best_groups) == 1:
                            # Only one significantly best group
                            best_name = self._format_name(best_groups[0])
                            interpretation_text = f"{best_name} sig. better"
                        elif len(best_groups) > 1:
                            # Multiple equivalent best groups
                            if len(best_groups) == len(sorted_vars):
                                interpretation_text = "No sig. difference"
                            else:
                                best_names = [self._format_name(g) for g in best_groups]
                                if len(best_names) <= 2:
                                    interpretation_text = " \\& ".join(best_names) + " equiv."
                                else:
                                    interpretation_text = f"{len(best_names)} groups equiv."
                        else:
                            # No best groups identified
                            interpretation_text = "No sig. difference"
                
                interpretation_row_cells.append(interpretation_text)
            
            p_value_rows.append("        " + " & ".join(interpretation_row_cells) + " \\\\")
            
            footer = "        \\bottomrule\n    \\end{tabular}}\n\\end{table}"
            return "\n".join([header] + ["        " + row for row in rows] + p_value_rows + [footer])
        else:
            footer = "        \\bottomrule\n    \\end{tabular}}\n\\end{table}"
            return "\n".join([header] + ["        " + row for row in rows] + [footer])

    def _format_parameter_value(self, value: Any) -> str:
        """Formats a parameter value for display in LaTeX tables."""
        s = str(value)
        
        # Check against existing maps first for full-string matches
        if s in self.STRATEGY_MAP: return self.STRATEGY_MAP.get(s, s)
        if s in self.MODALITY_MAP: return self.MODALITY_MAP.get(s, s)
        if s in self.SCALER_MAP: return self.SCALER_MAP.get(s, s)

        # General formatting for values that are not in maps
        s = s.replace('_', ' ').title()
        
        # Acronyms and specific formatting updates
        s = s.replace('Pca', 'PCA')
        s = s.replace('Smote', 'SMOTE')
        s = s.replace('Knn', 'KNN')
        s = s.replace('Mlp', 'MLP')
        s = s.replace('Ac Sf', 'ACSF')
        s = s.replace('So Ap', 'SOAP')
        s = s.replace('Ma Ce', 'MACE')
        s = s.replace('With PCA 50Components', 'PCA (50)')
        s = s.replace('Logisticregression', 'Logistic Regression')

        return s

    def report_top_models_table(self, top_models_df: pd.DataFrame, id_vars: List[str], learning_type: str, affinity_type: str, n_top: int = 5):
        """Generates a LaTeX table for the top N ranked models with their parameters."""
        
        title = f"Top {n_top} {learning_type.title()} Models by Average Rank"
        if self.log_table_generation:
            printer.info("="*10 + f" {title} LaTeX Table " + "="*10)

        df = top_models_df.head(n_top).copy()

        # Prepare columns for the table
        param_cols = [var for var in id_vars if var in df.columns]
        data_cols = param_cols + ['avg_rank', 'avg_score']
        df_table = df[data_cols].copy()

        # Format parameter columns
        for col in param_cols:
            df_table[col] = df_table[col].apply(self._format_parameter_value)

        # Format numeric columns
        df_table['avg_rank'] = df_table['avg_rank'].map('{:.2f}'.format)
        df_table['avg_score'] = df_table['avg_score'].map('{:.4f}'.format)
        
        col_map = {
            'inhibitor': 'Descriptor',
            'modality': 'Modality',
            'scaler': 'Scaler',
            'method': 'Method',
            'target_smote': 'Target SMOTE',
            'pca_status': 'PCA',
            'cross_domain_scaling': 'Cross-Domain Scaling',
            'source_smote': 'Source SMOTE',
            'avg_rank': 'Rank',
            'avg_score': 'Score'
        }
        
        display_cols = [col_map.get(c, c.replace('_', ' ').title()) for c in data_cols]
        df_table.columns = display_cols

        # LaTeX table generation
        column_format = "l" * len(display_cols)
        header = " & ".join([f"\\textbf{{{name}}}" for name in display_cols]) + " \\\\"
        
        rows = []
        for _, row in df_table.iterrows():
            rows.append(" & ".join(row.astype(str)) + " \\\\")

        caption = (f"Top {n_top} most robust {learning_type.lower()} models on the {self._format_affinity_type(affinity_type)} dataset, "
                   "ranked by their average rank across all thresholds. The table displays the hyperparameters for each of the top models.")
        label = f"tab:top_{n_top}_robust_{learning_type.lower()}_{affinity_type.lower()}"

        latex_table = f"""\\begin{{table}}[H]
    \\centering
    \\small
    \\caption{{{caption}}}
    \\label{{{label}}}
    \\resizebox{{\\textwidth}}{{!}}{{
    \\begin{{tabular}}{{{column_format}}}
        \\toprule
        {header}
        \\midrule
        {''.join(rows)}
        \\bottomrule
    \\end{{tabular}}
    }}
\\end{{table}}"""

        output_dir = self.base_output_dir
        filename = f"table_top_{n_top}_robust_{learning_type.lower()}_{affinity_type.lower()}.tex"
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'w') as f:
            f.write(latex_table)
        
        if self.log_table_generation:
            printer.info(f"Saved top {n_top} models LaTeX table to {output_path}")

    def _format_name(self, name: Any) -> str:
        """Formats a variable name for display in LaTeX tables."""
        name_str = str(name)
        if name_str in self.STRATEGY_MAP: return self.STRATEGY_MAP[name_str]
        if name_str in self.MODALITY_MAP: return self.MODALITY_MAP[name_str]
        if name_str in self.SCALER_MAP: return self.SCALER_MAP[name_str]
        if name_str.lower() in ['true', 'false']: return name_str.title()
        return name_str.replace('_', ' ').title()

    def _format_affinity_type(self, affinity_type: str) -> str:
        """Formats the affinity type for LaTeX captions."""
        return {'pk': 'pK', 'pic50': 'pIC\\textsubscript{50}'}.get(affinity_type.lower(), affinity_type.upper())