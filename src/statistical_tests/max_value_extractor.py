from console import printer
# max_value_extractor.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from constants import FALLBACK_MAX_VALUES



class MaxValueExtractor:
    """
    Extract maximum values from noise estimation results for fold comparison tables.

    This class provides methods to extract theoretical maximum performance values
    from noise estimation experiments, which are used as baselines in fold comparisons.
    """

    def __init__(self, noise_results_dir: str = "noise_estimation") -> None:
        """
        Initialize the extractor with the noise results directory.

        Parameters
        ----------
        noise_results_dir : str, optional
            Path to the directory containing noise estimation results.
            Default is 'noise_estimation'.

        Returns
        -------
        None
        """
        self.noise_results_dir = Path(noise_results_dir)

    def extract_max_values_from_noise_results(
        self, threshold: int, affinity_type: str
    ) -> Dict[str, List[float]]:
        """
        Extract maximum values from noise estimation results for a specific threshold and affinity type.

        Parameters
        ----------
        threshold : int
            The threshold value (e.g., 5, 6, 7).
        affinity_type : str
            The affinity type ('pic50' or 'pk').

        Returns
        -------
        dict[str, list[float]]
            Dictionary with keys 'mcc', 'accuracy', 'f1' containing lists of maximum values per fold.
        """
        try:
            # Look for noise analysis results in the threshold-specific directory
            threshold_dir = (
                self.noise_results_dir / f"threshold_{threshold}" / affinity_type
            )

            if not threshold_dir.exists():
                printer.warning(f"Noise results directory not found: {threshold_dir}")
                return self._get_fallback_max_values(affinity_type)

            # Find all noise_analysis_results.pkl files
            pkl_files = list(threshold_dir.glob("**/noise_analysis_results.pkl"))

            if not pkl_files:
                printer.warning(f"No noise analysis results found in {threshold_dir}")
                return self._get_fallback_max_values(affinity_type)

            # Load the first available results file
            results_file = pkl_files[0]
            printer.info(f"Loading noise results from: {results_file}")

            data = pd.read_pickle(results_file)

            # Extract maximum values from test results (which represent the best achievable performance)
            max_values = self._extract_max_from_test_results(data)

            if max_values:
                printer.info(
                    f"Successfully extracted max values for threshold {threshold}, {affinity_type}"
                )
                return max_values
            else:
                printer.warning(f"Could not extract max values from {results_file}")
                return self._get_fallback_max_values(affinity_type)

        except Exception as e:
            printer.error(f"Error extracting max values: {e}")
            return self._get_fallback_max_values(affinity_type)

    def _extract_max_from_test_results(
        self, data: Dict
    ) -> Optional[Dict[str, List[float]]]:
        """
        Extract maximum values from test results in the noise analysis data.

        Parameters
        ----------
        data : dict
            The loaded noise analysis results dictionary.

        Returns
        -------
        dict[str, list[float]] or None
            Dictionary with maximum values per fold, or None if extraction fails.
        """
        try:
            if "test_results" not in data:
                printer.warning("No test_results found in noise analysis data")
                return None

            test_results = data["test_results"]

            # Initialize lists to store maximum values for each fold
            mean_mcc_values = []
            std_mcc_values = []

            mean_accuracy_values = []
            std_accuracy_values = []

            mean_f1_values = []
            std_f1_values = []

            # Process each fold
            for fold_name in sorted(test_results.keys()):
                fold_data = test_results[fold_name]

                if not isinstance(fold_data, pd.DataFrame):
                    printer.warning(
                        f"Fold {fold_name} data is not a DataFrame: {type(fold_data)}"
                    )
                    continue

                # Check if required columns exist
                required_columns = ["matthews_corrcoef", "accuracy", "f1"]
                missing_columns = [
                    col for col in required_columns if col not in fold_data.columns
                ]

                if missing_columns:
                    printer.warning(
                        f"Missing columns in fold {fold_name}: {missing_columns}"
                    )
                    continue

                # Calculate maximum values for this fold
                # For noise estimation, we want the maximum achievable performance
                # This could be the 95th percentile or maximum value from bootstrap results
                # fold_max_mcc = fold_data['matthews_corrcoef'].quantile(0.95)  # 95th percentile
                # fold_max_accuracy = fold_data['accuracy'].quantile(0.95)
                # fold_max_f1 = fold_data['f1'].quantile(0.95)

                # Alternative: use actual maximum values
                fold_mean_mcc = fold_data["matthews_corrcoef"].mean()
                fold_std_mcc = fold_data["matthews_corrcoef"].std()

                fold_mean_accuracy = fold_data["accuracy"].mean()
                fold_std_accuracy = fold_data["accuracy"].std()

                fold_mean_f1 = fold_data["f1"].mean()
                fold_std_f1 = fold_data["f1"].std()

                mean_mcc_values.append(fold_mean_mcc)
                std_mcc_values.append(fold_std_mcc)

                mean_accuracy_values.append(fold_mean_accuracy)
                std_accuracy_values.append(fold_std_accuracy)

                mean_f1_values.append(fold_mean_f1)
                std_f1_values.append(fold_std_f1)

                printer.debug(
                    f"Fold {fold_name}: MCC={fold_mean_mcc:.3f}, Acc={fold_mean_accuracy:.3f}, F1={fold_mean_f1:.3f}"
                )

            if not mean_mcc_values:
                printer.warning("No valid fold data found for extracting maximum values")
                return None

            return {
                "mcc": mean_mcc_values,
                "accuracy": mean_accuracy_values,
                "f1": mean_f1_values,
            }

        except Exception as e:
            printer.error(f"Error extracting max values from test results: {e}")
            return None

    def _get_fallback_max_values(self, affinity_type: str) -> Dict[str, List[float]]:
        """
        Get fallback maximum values when noise estimation results are not available.

        Values are sourced from constants.FALLBACK_MAX_VALUES.

        Parameters
        ----------
        affinity_type : str
            The affinity type ('pic50' or 'pk').

        Returns
        -------
        dict[str, list[float]]
            Dictionary containing fallback maximum values for mcc, accuracy, and f1.
        """
        printer.warning(f"Using fallback maximum values for {affinity_type}")
        key = affinity_type if affinity_type in FALLBACK_MAX_VALUES else "pk"
        return FALLBACK_MAX_VALUES[key]

    def get_max_values_for_threshold(
        self, threshold: int, affinity_type: str
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Get maximum values for a specific threshold and affinity type.

        Parameters
        ----------
        threshold : int
            The threshold value (e.g., 5, 6, 7).
        affinity_type : str
            The affinity type ('pic50' or 'pk').

        Returns
        -------
        tuple[list[float], list[float], list[float]]
            Tuple of (mcc_values, accuracy_values, f1_values) for all folds.
        """
        max_values = self.extract_max_values_from_noise_results(
            threshold, affinity_type
        )

        return (max_values["mcc"], max_values["accuracy"], max_values["f1"])


def extract_max_values_from_noise_results(
    threshold: int, affinity_type: str, noise_results_dir: str = "noise_estimation"
) -> Tuple[List[float], List[float], List[float]]:
    """
    Convenience function to extract maximum values from noise estimation results.

    Parameters
    ----------
    threshold : int
        The threshold value (e.g., 5, 6, 7).
    affinity_type : str
        The affinity type ('pic50' or 'pk').
    noise_results_dir : str, optional
        Path to the noise results directory. Default is 'noise_estimation'.

    Returns
    -------
    tuple[list[float], list[float], list[float]]
        Tuple of (mcc_values, accuracy_values, f1_values) for all folds.
    """
    extractor = MaxValueExtractor(noise_results_dir)
    return extractor.get_max_values_for_threshold(threshold, affinity_type)
