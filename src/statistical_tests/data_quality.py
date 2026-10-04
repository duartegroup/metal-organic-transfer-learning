from console import printer
import pandas as pd
import itertools
from functools import reduce
from operator import mul
from typing import List, Dict, Any



class DataQuality:
    """
    Analyze the quality and completeness of experimental data.

    This class identifies missing experiment submissions on a per-threshold basis and
    reports on overall data integrity, fold completeness, and cross-threshold consistency.
    """

    def __init__(
        self, df_sl: pd.DataFrame, df_tl: pd.DataFrame, affinity_types: List[str]
    ) -> None:
        """
        Initialize the DataQuality analyzer.

        Parameters
        ----------
        df_sl : pd.DataFrame
            DataFrame containing supervised learning experimental results.
        df_tl : pd.DataFrame
            DataFrame containing transfer learning experimental results.
        affinity_types : list[str]
            List of affinity types to analyze (e.g., ['pic50', 'pk']).
        """
        self.df_sl = df_sl
        self.df_tl = df_tl
        self.affinity_types = affinity_types

        # Centralized definitions for a single experiment row (includes method)
        self.sl_exp_vars = [
            "inhibitor",
            "modality",
            "scaler",
            "method",
            "target_smote",
            "pca_status",
        ]
        self.tl_exp_vars = [
            "inhibitor",
            "modality",
            "scaler",
            "base_method",
            "cross_domain_scaling",
            "source_smote",
            "target_smote",
        ]

        # Definitions for a "submission" (excludes method)
        self.sl_submission_vars = [
            "inhibitor",
            "modality",
            "scaler",
            "target_smote",
            "pca_status",
        ]
        self.tl_submission_vars = [
            "inhibitor",
            "modality",
            "scaler",
            "cross_domain_scaling",
            "source_smote",
            "target_smote",
        ]

    def init_check(self) -> None:
        """
        Perform initial data loading checks and log summary statistics.

        Displays counts of experiments per threshold and affinity type, along with
        unique parameter value counts for both supervised and transfer learning.
        """
        printer.info("=" * 60)
        printer.info("DATA LOADING COMPLETE: SUMMARY")
        printer.info("=" * 60)

        printer.info("Supervised Learning Data Counts:")
        printer.info("-" * 40)
        sl_counts = self.df_sl.groupby(["threshold", "affinity_type"]).size()
        for (threshold, affinity), count in sl_counts.items():
            printer.info(f"  • Threshold {threshold}, {affinity:5}: {count} entries")

        printer.info("\n")
        printer.info("Transfer Learning Data Counts:")
        printer.info("-" * 40)
        tl_counts = self.df_tl.groupby(["threshold", "affinity_type"]).size()
        for (threshold, affinity), count in tl_counts.items():
            printer.info(f"  • Threshold {threshold}, {affinity:5}: {count} entries")
        printer.info("=" * 60)

        # Supervised Learning unique combinations
        # Supervised Learning unique combinations
        printer.info("=" * 60)
        printer.info("SUPERVISED LEARNING EXPERIMENT COUNTS")
        printer.info("=" * 60)

        sl_columns = [
            "inhibitor",
            "modality",
            "scaler",
            "method",
            "target_smote",
            "pca_status",
        ]
        unique_counts_sl = self.df_sl[sl_columns].nunique()

        for col in sl_columns:
            count = unique_counts_sl[col]
            printer.info(f"  • {col:20}: {count:3} unique values")

        total_sl_experiments = unique_counts_sl.product()
        printer.info("-" * 60)
        printer.info(f"Total possible SL experiments: {total_sl_experiments}")
        printer.info("\n")

        # Transfer Learning unique combinations
        printer.info("=" * 60)
        printer.info("TRANSFER LEARNING EXPERIMENT COUNTS")
        printer.info("=" * 60)

        tl_columns = [
            "inhibitor",
            "modality",
            "scaler",
            "method",
            "cross_domain_scaling",
            "source_smote",
            "target_smote",
        ]
        unique_counts_tl = self.df_tl[tl_columns].nunique()

        for col in tl_columns:
            count = unique_counts_tl[col]
            printer.info(f"  • {col:20}: {count:3} unique values")

        total_tl_experiments = unique_counts_tl.product()
        printer.info("-" * 60)
        printer.info(f"Total possible TL experiments: {total_tl_experiments}")

    def _prepare_dataframe(self, df: pd.DataFrame, method_type: str) -> pd.DataFrame:
        """
        Prepare a dataframe by adding identifier columns for quality analysis.

        Normalizes column values and adds three types of ID columns:
        1. 'exp_id': Full experiment signature, including method and threshold
        2. 'unique_experiment_id': Excludes threshold, for cross-threshold consistency checks
        3. 'submission_id': Excludes method, for identifying missing submissions

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing experimental results.
        method_type : str
            Type of method ('supervised_learning' or 'transfer_learning').

        Returns
        -------
        pd.DataFrame
            Prepared DataFrame with added identifier columns.
        """
        if df.empty:
            return df

        df_copy = df.copy()
        exp_vars = (
            self.sl_exp_vars
            if method_type == "supervised_learning"
            else self.tl_exp_vars
        )
        submission_vars = (
            self.sl_submission_vars
            if method_type == "supervised_learning"
            else self.tl_submission_vars
        )

        cols_to_normalize = list(set(exp_vars + submission_vars + ["threshold"]))
        for col in [c for c in cols_to_normalize if c in df_copy.columns]:
            df_copy = self.normalize_column_values(df_copy, col)

        # ID for cross-threshold consistency (excludes threshold, includes method)
        available_exp_vars = [var for var in exp_vars if var in df_copy.columns]
        df_copy["unique_experiment_id"] = df_copy[available_exp_vars].agg(
            "_".join, axis=1
        )

        # ID for fold counting (includes method and threshold)
        exp_vars_with_threshold = [
            var for var in available_exp_vars + ["threshold"] if var in df_copy.columns
        ]
        df_copy["exp_id"] = df_copy[exp_vars_with_threshold].agg("_".join, axis=1)

        # ID for identifying missing submissions (excludes method, includes threshold)
        available_submission_vars = [
            var for var in submission_vars if var in df_copy.columns
        ]
        submission_vars_with_threshold = [
            var
            for var in available_submission_vars + ["threshold"]
            if var in df_copy.columns
        ]
        df_copy["submission_id"] = df_copy[submission_vars_with_threshold].agg(
            "_".join, axis=1
        )

        return df_copy

    def _find_and_report_missing_submissions(
        self, prepared_df: pd.DataFrame, method_type: str
    ) -> None:
        """
        Identify and report missing experiment submissions on a per-threshold basis.

        A "submission" is a unique combination of parameters, excluding the method.
        Generates bash array format output for easy resubmission of missing experiments.

        Parameters
        ----------
        prepared_df : pd.DataFrame
            Prepared DataFrame with identifier columns.
        method_type : str
            Type of method ('supervised_learning' or 'transfer_learning').
        """
        if prepared_df.empty:
            return

        printer.info("--- CHECKING FOR MISSING SUBMISSIONS PER THRESHOLD ---")
        submission_vars = (
            self.sl_submission_vars
            if method_type == "supervised_learning"
            else self.tl_submission_vars
        )

        any_missing = False

        for threshold in sorted(prepared_df["threshold"].unique()):
            df_for_threshold = prepared_df[prepared_df["threshold"] == threshold]

            # 1. Get all possible submission configurations for this threshold
            param_cols = [
                col for col in submission_vars if col in df_for_threshold.columns
            ]
            unique_values_lists = [
                prepared_df[col].unique().tolist() for col in param_cols
            ]

            all_param_combos = itertools.product(*unique_values_lists)
            possible_submission_ids = {
                f"{'_'.join(combo)}_{threshold}" for combo in all_param_combos
            }

            # 2. Get the submissions actually found in the data for this threshold
            found_submission_ids = set(df_for_threshold["submission_id"].unique())

            # 3. Find the difference and report
            missing_ids = sorted(list(possible_submission_ids - found_submission_ids))

            if missing_ids:
                any_missing = True
                printer.warning(
                    f"❗️ Found {len(missing_ids)} missing submissions for Threshold {threshold}:"
                )

                # Collect missing runs for this threshold
                missing_runs_this_threshold = []
                limit = 500
                for sub_id in missing_ids[:limit]:
                    missing_run = sub_id.rsplit("_", 1)[0]  # Remove threshold from end
                    missing_runs_this_threshold.append(missing_run)
                    printer.warning(f"  • {missing_run}")
                if len(missing_ids) > limit:
                    printer.warning(f"  • ... and {len(missing_ids) - limit} more.")

                # Generate bash array format for this threshold
                printer.info("=" * 60)
                printer.info(
                    f"MISSING RUNS FOR THRESHOLD {threshold} - BASH ARRAY FORMAT"
                )
                printer.info("=" * 60)
                print(f"\n# Missing runs for threshold {threshold}")
                print(f"missing_runs_threshold_{threshold}=(")
                for run in sorted(
                    set(missing_runs_this_threshold)
                ):  # Remove duplicates and sort
                    print(f"    '{run}'")
                print(");")
                printer.info("=" * 60)

        if not any_missing:
            printer.info("No missing submissions found for any threshold.")

    def _analyze_dataframe_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate aggregate statistics about fold completeness.

        Counts experiments with complete vs. incomplete folds and logs warnings
        for experiments with missing folds.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing experimental results with 'exp_id' and 'fold' columns.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - 'found': Total number of experiments found
            - 'with_complete_folds': Number of experiments with all 5 folds
            - 'with_missing_folds': Number of experiments with incomplete folds
        """
        fold_counts = df.groupby("exp_id")["fold"].nunique()
        found_experiments = len(fold_counts)

        expected_folds = 5
        complete = fold_counts[fold_counts == expected_folds]
        missing_folds = fold_counts[fold_counts < expected_folds]

        quality_report = {
            "found": found_experiments,
            "with_complete_folds": len(complete),
            "with_missing_folds": len(missing_folds),
        }

        printer.info(f"Found: {quality_report['found']} total experiments")
        printer.info(
            f"Complete Folds: {quality_report['with_complete_folds']} | Incomplete Folds: {quality_report['with_missing_folds']}"
        )

        if quality_report["with_missing_folds"] > 0:
            printer.warning("Experiments with incomplete folds:")
            for exp_id, num_folds in missing_folds.items():
                printer.warning(f"  - {exp_id} has {num_folds}/{expected_folds} folds")

        return quality_report

    def check_experiment_ids_across_thresholds(self, df: pd.DataFrame) -> None:
        """
        Check if each threshold has the same set of unique experiment IDs.

        Verifies that all thresholds have consistent experiment coverage, which is
        important for fair cross-threshold comparisons.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing experimental results with 'threshold' and 'unique_experiment_id' columns.
        """
        printer.info("--- CHECKING EXPERIMENT CONSISTENCY ACROSS THRESHOLDS ---")
        thresholds = df["threshold"].unique()
        if len(thresholds) < 2:
            printer.info("Only one threshold found, skipping consistency check.")
            return

        threshold_ids = {
            t: set(df[df["threshold"] == t]["unique_experiment_id"]) for t in thresholds
        }
        all_ids = set().union(*threshold_ids.values())

        is_consistent = True
        for t in thresholds:
            missing_ids = all_ids - threshold_ids[t]
            if missing_ids:
                is_consistent = False
                printer.warning(
                    f"Threshold {t} is missing {len(missing_ids)} experiment IDs compared to others."
                )

        if is_consistent:
            printer.info("All thresholds have a consistent set of experiments.")

    def _get_expected_experiment_count(
        self, df: pd.DataFrame, id_vars: List[str]
    ) -> int:
        """
        Calculate expected number of experiments per threshold based on parameter combinations.

        Computes the Cartesian product size of all unique parameter values.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing experimental results.
        id_vars : list[str]
            List of parameter column names to consider.

        Returns
        -------
        int
            Expected number of unique experiments (product of unique values per parameter).
        """
        if df.empty:
            return 0
        param_cols = [col for col in id_vars if col in df.columns]
        num_uniques = [df[col].nunique() for col in param_cols]
        return reduce(mul, num_uniques, 1)

    def normalize_column_values(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Normalize column values to consistent string representations.

        Converts boolean columns and special parameter columns to standardized string values.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to normalize.
        column : str
            Column name to normalize.

        Returns
        -------
        pd.DataFrame
            DataFrame with normalized column values.
        """
        df_copy = df.copy()
        if column in df_copy.columns:
            if column == "cross_domain_scaling":
                df_copy[column] = df_copy[column].map(
                    {True: "cross_domain", False: "separate_domain"}
                )
            elif column == "source_smote":
                df_copy[column] = df_copy[column].map(
                    {True: "source_smote", False: "no_source_smote"}
                )
            elif column == "target_smote":
                df_copy[column] = df_copy[column].map(
                    {True: "target_smote", False: "no_target_smote"}
                )
            elif df_copy[column].dtype == "bool":
                df_copy[column] = df_copy[column].map({True: "True", False: "False"})
            else:
                df_copy[column] = df_copy[column].astype(str)
        return df_copy

    def _generate_data_quality_summary(
        self,
        sl_quality: Dict[str, Any],
        tl_quality: Dict[str, Any],
        sl_expected: int,
        tl_expected: int,
        affinity_type: str,
    ) -> None:
        """
        Generate final summary report with actionable recommendations.

        Calculates completeness percentages and provides recommendations for addressing
        missing or incomplete experiments.

        Parameters
        ----------
        sl_quality : dict[str, Any]
            Quality metrics for supervised learning experiments.
        tl_quality : dict[str, Any]
            Quality metrics for transfer learning experiments.
        sl_expected : int
            Expected number of supervised learning experiments.
        tl_expected : int
            Expected number of transfer learning experiments.
        affinity_type : str
            The affinity type being analyzed.
        """
        printer.info("=" * 60)
        printer.info(f"DATA QUALITY SUMMARY FOR {affinity_type.upper()}")
        printer.info("=" * 60)

        complete_sl = sl_quality.get("with_complete_folds", 0)
        sl_completeness = (complete_sl / sl_expected * 100) if sl_expected > 0 else 0
        printer.info(
            f"Supervised Learning: {complete_sl}/{sl_expected} complete ({sl_completeness:.1f}%)"
        )

        complete_tl = tl_quality.get("with_complete_folds", 0)
        tl_completeness = (complete_tl / tl_expected * 100) if tl_expected > 0 else 0
        printer.info(
            f"Transfer Learning: {complete_tl}/{tl_expected} complete ({tl_completeness:.1f}%)"
        )

        printer.info("RECOMMENDATIONS:")
        missing_sl = sl_expected - complete_sl
        if missing_sl > 0:
            printer.error(
                f"Address the {missing_sl} missing or incomplete supervised experiments."
            )

        missing_tl = tl_expected - complete_tl
        if missing_tl > 0:
            printer.error(
                f"Address the {missing_tl} missing or incomplete transfer learning experiments."
            )

        if sl_completeness >= 100 and tl_completeness >= 100:
            printer.info(
                "All expected experiments are present and have the correct number of folds."
            )

    def analyze_data_quality(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run the complete data quality analysis pipeline.

        Orchestrates the entire analysis workflow: initial checks, per-affinity-type analysis,
        cross-threshold consistency checks, fold completeness analysis, and missing submission detection.

        Returns
        -------
        tuple[dict[str, Any], dict[str, Any]]
            A tuple containing:
            - sl_quality: Quality metrics for supervised learning
            - tl_quality: Quality metrics for transfer learning
        """
        printer.info("=" * 60)
        printer.info("DATA QUALITY ANALYSIS")
        printer.info("=" * 60)

        self.init_check()

        for affinity_type in self.affinity_types:
            printer.info("\n")
            printer.info(f"{'='*60}")
            printer.info(f" ANALYZING {affinity_type.upper()}")
            printer.info(f"{'='*60}")

            df_sl_filtered = self.df_sl[self.df_sl["affinity_type"] == affinity_type]
            df_tl_filtered = self.df_tl[self.df_tl["affinity_type"] == affinity_type]

            sl_prepared = self._prepare_dataframe(df_sl_filtered, "supervised_learning")
            tl_prepared = self._prepare_dataframe(df_tl_filtered, "transfer_learning")

            # Calculate Expected Totals
            num_thresholds_sl = (
                sl_prepared["threshold"].nunique() if not sl_prepared.empty else 0
            )
            expected_sl_per_threshold = self._get_expected_experiment_count(
                sl_prepared, self.sl_exp_vars
            )
            total_expected_sl = expected_sl_per_threshold * num_thresholds_sl

            num_thresholds_tl = (
                tl_prepared["threshold"].nunique() if not tl_prepared.empty else 0
            )
            expected_tl_per_threshold = self._get_expected_experiment_count(
                tl_prepared, self.tl_exp_vars
            )
            total_expected_tl = expected_tl_per_threshold * num_thresholds_tl

            sl_quality, tl_quality = {}, {}
            if not sl_prepared.empty:
                printer.info("=" * 60)
                printer.info("SUPERVISED LEARNING ANALYSIS")
                printer.info("=" * 60)
                self.check_experiment_ids_across_thresholds(sl_prepared)
                sl_quality = self._analyze_dataframe_quality(sl_prepared)
                self._find_and_report_missing_submissions(
                    sl_prepared, "supervised_learning"
                )

            if not tl_prepared.empty:
                printer.info("=" * 60)
                printer.info("TRANSFER LEARNING ANALYSIS")
                printer.info("=" * 60)
                self.check_experiment_ids_across_thresholds(tl_prepared)
                tl_quality = self._analyze_dataframe_quality(tl_prepared)
                self._find_and_report_missing_submissions(
                    tl_prepared, "transfer_learning"
                )

            self._generate_data_quality_summary(
                sl_quality,
                tl_quality,
                total_expected_sl,
                total_expected_tl,
                affinity_type,
            )

        return sl_quality, tl_quality
