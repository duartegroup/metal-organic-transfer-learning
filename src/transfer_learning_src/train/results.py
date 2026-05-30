from typing import Dict, Any, List
import numpy as np
import pandas as pd
import json
import logging
from prettytable import PrettyTable
from .config import TrainingConfig
from .phases import TrainingPhase
from model.multitask import MultiTaskPocket

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


#
class ResultsManager:
    """
    Handle results aggregation, reporting, and persistence for training experiments.

    This class provides static methods for generating summaries, formatting results,
    and saving experiment outcomes to disk.
    """

    @staticmethod
    def generate_summary(all_results: Dict[str, Any], params: Dict[str, Any]) -> None:
        """
        Generate comprehensive summary of training results.

        Parameters
        ----------
        all_results : dict[str, Any]
            Dictionary containing results from all training methods and folds.
        params : dict[str, Any]
            Dictionary containing experiment parameters and configuration.

        Returns
        -------
        None
        """
        logger.info("=" * 100)
        logger.info("COMPREHENSIVE MULTITASK TRAINING RESULTS SUMMARY")
        logger.info("=" * 100)

        # Show pre-training results
        ResultsManager._show_pretraining_results(all_results)

        # Show fine-tuning results
        methods = [
            k
            for k, v in all_results.items()
            if k not in ("pretrain_src")
            and isinstance(v, dict)
            and any(isinstance(fold, int) for fold in v.keys())
        ]

        if methods:
            ResultsManager._show_finetuning_results(all_results, methods, params)

    @staticmethod
    def _show_pretraining_results(all_results: Dict[str, Any]) -> None:
        """
        Display pre-training results in a formatted table.

        Parameters
        ----------
        all_results : dict[str, Any]
            Dictionary containing all training results including pretraining metrics.

        Returns
        -------
        None
        """
        logger.info("PRE-TRAINING RESULTS:")
        pretrain_metrics = all_results.get("pretrain_src", {}).get("single_run", {})

        if pretrain_metrics.get("pretraining_skipped"):
            logger.info("Source pre-training was skipped (loaded from checkpoint).")
        else:
            table = PrettyTable()
            table.field_names = ["Phase", "Loss", "Accuracy", "F1", "AUROC", "MCC"]
            table.add_row(
                [
                    "Source Pretrain",
                    f"{pretrain_metrics.get('val/loss', 0.0):.3f}",
                    f"{pretrain_metrics.get('val/accuracy', 0.0):.3f}",
                    f"{pretrain_metrics.get('val/f1', 0.0):.3f}",
                    f"{pretrain_metrics.get('val/auroc', 0.0):.3f}",
                    f"{pretrain_metrics.get('val/mcc', 0.0):.3f}",
                ]
            )
            logger.info(f"\n{table}")

    @staticmethod
    def _show_finetuning_results(
        all_results: Dict[str, Any], methods: List[str], params: Dict[str, Any]
    ) -> None:
        """
        Display fine-tuning results across all folds in formatted tables.

        Parameters
        ----------
        all_results : dict[str, Any]
            Dictionary containing all training results.
        methods : list[str]
            List of training method names to display.
        params : dict[str, Any]
            Dictionary containing experiment parameters including n_folds.

        Returns
        -------
        None
        """
        metrics = ["val/loss", "val/accuracy", "val/f1", "val/auroc", "val/mcc"]

        # Individual results table
        table = PrettyTable()
        table.field_names = ["Method", "Fold"] + [
            m.replace("val/", "").upper() for m in metrics
        ]

        method_stats = {}

        for method in methods:
            method_stats[method] = {metric: [] for metric in metrics}

            for fold in range(1, params["n_folds"] + 1):
                if fold in all_results[method]:
                    fold_results = all_results[method][fold]

                    row = [method.replace("_", " ").title(), fold]
                    for metric in metrics:
                        value = fold_results.get(metric, 0.0)
                        if metric == "val/loss":
                            row.append(f"{value:.3f}")
                        else:
                            row.append(f"{value:.3f}")
                        method_stats[method][metric].append(value)

                    table.add_row(row)
                else:
                    row = [method.replace("_", " ").title(), fold] + ["ERROR"] * len(
                        metrics
                    )
                    table.add_row(row)

        logger.info("FINE-TUNING RESULTS ACROSS FOLDS:")
        logger.info(f"\n{table}")

        # Summary statistics
        ResultsManager._show_summary_statistics(method_stats, methods, metrics)

    @staticmethod
    def _show_summary_statistics(
        method_stats: Dict, methods: List[str], metrics: List[str]
    ) -> None:
        """
        Display summary statistics across all folds for each method and metric.

        Parameters
        ----------
        method_stats : dict
            Dictionary mapping methods to their metric statistics.
        methods : list[str]
            List of method names to summarize.
        metrics : list[str]
            List of metric names to include in summary.

        Returns
        -------
        None
        """
        logger.info("=" * 80)
        logger.info("       SUMMARY STATISTICS ACROSS ALL FOLDS")
        logger.info("=" * 80)

        summary_table = PrettyTable()
        summary_table.field_names = ["Method", "Metric", "Mean ± Std", "Best Fold"]

        best_methods = {}

        for method in methods:
            for metric in metrics:
                if method_stats[method][metric]:
                    values = method_stats[method][metric]
                    mean_val = np.mean(values)
                    std_val = np.std(values)

                    # Find best fold
                    if metric == "val/loss":
                        best_idx = np.argmin(values)
                        best_val = np.min(values)
                    else:
                        best_idx = np.argmax(values)
                        best_val = np.max(values)

                    best_fold = best_idx + 1

                    # Track best method for each metric
                    if metric not in best_methods or (
                        (
                            metric == "val/loss"
                            and mean_val < best_methods[metric]["mean"]
                        )
                        or (
                            metric != "val/loss"
                            and mean_val > best_methods[metric]["mean"]
                        )
                    ):
                        best_methods[metric] = {
                            "method": method,
                            "mean": mean_val,
                            "std": std_val,
                        }

                    # Format display
                    if metric == "val/loss":
                        mean_std_str = f"{mean_val:.3f} ± {std_val:.3f}"
                        best_str = f"Fold {best_fold}: {best_val:.3f}"
                    else:
                        mean_std_str = f"{mean_val:.3f} ± {std_val:.3f}"
                        best_str = f"Fold {best_fold}: {best_val:.3f}"

                    summary_table.add_row(
                        [
                            method.replace("_", " ").title(),
                            metric.replace("val/", "").upper(),
                            mean_std_str,
                            best_str,
                        ]
                    )

        logger.info(f"\n{summary_table}")

        # Best methods summary
        ResultsManager._show_best_methods(best_methods)

    @staticmethod
    def _show_best_methods(best_methods: Dict) -> None:
        """
        Display the best performing method for each metric.

        Parameters
        ----------
        best_methods : dict
            Dictionary mapping metrics to their best performing methods and scores.

        Returns
        -------
        None
        """
        logger.info("=" * 80)
        logger.info("       BEST PERFORMING METHODS BY METRIC")
        logger.info("=" * 80)

        best_table = PrettyTable()
        best_table.field_names = ["Metric", "Best Method", "Performance"]

        for metric, best_info in best_methods.items():
            if metric == "val/loss":
                perf_str = f"{best_info['mean']:.3f} ± {best_info['std']:.3f}"
            else:
                perf_str = f"{best_info['mean']:.3f} ± {best_info['std']:.3f}"

            best_table.add_row(
                [
                    metric.replace("val/", "").upper(),
                    best_info["method"].replace("_", " ").title(),
                    perf_str,
                ]
            )

        logger.info(f"\n{best_table}")

    @staticmethod
    def save_results(all_results: Dict[str, Any], params: Dict[str, Any]) -> None:
        """
        Save experimental results to JSON and CSV files.

        Parameters
        ----------
        all_results : dict[str, Any]
            Dictionary containing all training results from various methods and folds.
        params : dict[str, Any]
            Dictionary containing experiment parameters and configuration.

        Returns
        -------
        None
        """
        # Use the same directory structure as training phases
        base_phase = TrainingPhase(TrainingConfig(), params)
        base_results_dir = base_phase.create_results_base_dir()

        # Save results in the same base directory as training results
        # This keeps everything organized together
        results_dir = base_results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        # Create run identifier
        lightning_module = MultiTaskPocket(**params)
        run_id = f"{lightning_module.combination_name}_{params['inhibitor_descriptor']}_{params['biological_descriptor']}"

        # Clean results (convert tensors to floats)
        clean_results = ResultsManager._convert_tensors_to_float(all_results)

        # Save JSON
        json_path = results_dir / f"{run_id}_raw_results.json"
        with open(json_path, "w") as f:
            json.dump(clean_results, f, indent=2)

        # Save CSV with more robust logic
        summary_data = []
        logger.info(
            f"Processing results for CSV generation. Results structure: {list(clean_results.keys())}"
        )

        for method, folds in clean_results.items():
            logger.info(f"Processing method: {method}, type: {type(folds)}")

            if isinstance(folds, dict):
                logger.info(f"  Folds keys: {list(folds.keys())}")
                for fold, metrics in folds.items():
                    logger.info(f"    Fold: {fold}, type: {type(metrics)}")
                    if isinstance(metrics, dict):
                        row = {"method": method, "fold": fold}
                        row.update(metrics)
                        summary_data.append(row)
                        logger.info(f"      Added row with {len(metrics)} metrics")
                    else:
                        logger.warning(f"      Metrics is not a dict: {type(metrics)}")
            else:
                logger.warning(f"  Folds is not a dict: {type(folds)}")

        logger.info(f"Total rows collected for CSV: {len(summary_data)}")

        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_path = results_dir / f"{run_id}_summary.csv"
            df.to_csv(csv_path, index=False)

            logger.info(f"Results saved to:")
            logger.info(f"  Raw results: {json_path}")
            logger.info(f"  Summary CSV: {csv_path}")
            logger.info(f"  CSV shape: {df.shape}")
        else:
            logger.error("No data collected for CSV generation!")
            logger.error(f"Results structure: {clean_results}")

    @staticmethod
    def _convert_tensors_to_float(obj: Any) -> Any:
        """
        Recursively convert PyTorch tensors to Python floats for JSON serialization.

        Parameters
        ----------
        obj : Any
            Object to convert. Can be a dict, tensor, or other type.

        Returns
        -------
        Any
            Converted object with all tensors replaced by floats.
        """
        if isinstance(obj, dict):
            return {
                key: ResultsManager._convert_tensors_to_float(value)
                for key, value in obj.items()
            }
        elif hasattr(obj, "item"):
            return float(obj.item())
        elif hasattr(obj, "cpu"):
            return float(obj.cpu().numpy())
        else:
            return obj
