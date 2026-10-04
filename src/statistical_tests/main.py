# main.py
import argparse
import pandas as pd
from typing import List

from console import printer, print_phase_banner
from analysis import MainAnalysisController
from data_loader import DataLoader
from data_quality import DataQuality

# All paths are derived from a single root so the pipeline can read results
# that live anywhere and write its own outputs somewhere else entirely -- useful
# when the results tree is on a filesystem you would rather not write to.
DEFAULT_RESULTS_BASE = "./../../results"
SUPERVISED_DATA_DIR_TEMPLATE = (
    "{results_base}/supervised_learning_results/threshold_{threshold}"
)
TRANSFER_LEARNING_CSV_TEMPLATE = (
    "{results_base}/transfer_learning_results/"
    "final_runs_threshold_{threshold}/all_transfer_learning_results.csv"
)


def load_all_data(
    affinity_types: List[str],
    thresholds: List[int],
    results_base: str = DEFAULT_RESULTS_BASE,
) -> tuple:
    print_phase_banner("LOADING DATA", subtitle="Supervised learning + Transfer learning")

    all_sl_data, all_tl_data = [], []

    for threshold in thresholds:
        supervised_dir = SUPERVISED_DATA_DIR_TEMPLATE.format(
            results_base=results_base, threshold=threshold
        )
        tl_csv_path = TRANSFER_LEARNING_CSV_TEMPLATE.format(
            results_base=results_base, threshold=threshold
        )

        temp_loader = DataLoader(
            supervised_base_dir=supervised_dir, tl_csv_path=tl_csv_path
        )

        for affinity_type in affinity_types:
            df_sl = temp_loader.load_supervised_learning_results(affinity_type)
            if not df_sl.empty:
                df_sl = df_sl.copy()
                df_sl["threshold"]     = threshold
                df_sl["affinity_type"] = affinity_type
                all_sl_data.append(df_sl)

            df_tl = temp_loader.load_transfer_learning_results(
                csv_path=tl_csv_path, affinity_type=affinity_type
            )
            if not df_tl.empty:
                df_tl = df_tl.copy()
                df_tl["threshold"]     = threshold
                df_tl["affinity_type"] = affinity_type
                all_tl_data.append(df_tl)

    if not all_sl_data or not all_tl_data:
        printer.error("No data loaded for at least one learning approach. Aborting.")
        return None, None

    df_sl_global = pd.concat(all_sl_data, ignore_index=True)
    df_tl_global = pd.concat(all_tl_data, ignore_index=True)
    printer.info(
        f"Loaded  SL: {len(df_sl_global):,} rows  │  "
        f"TL: {len(df_tl_global):,} rows  │  "
        f"thresholds: {thresholds}  │  affinity types: {affinity_types}"
    )
    return df_sl_global, df_tl_global


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical analysis pipeline for SL vs TL experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
outputs
  --scatter   Scatter plots of individual SL / TL results, per threshold.
              Saved to: results/statistical_tests/scatter/threshold_<N>/
  --champion  Champion comparison bar chart selected by cross-threshold
              mean_mcc_minus_std ranking across θ ∈ {5, 6, 7}.
              Saved to: results/statistical_tests/champion/
  --tables    LaTeX cross-threshold parameter comparison tables.
              Saved to: results/statistical_tests/tables/threshold_<N>/
                        results/statistical_tests/tables/global/
        """,
    )
    parser.add_argument(
        "--affinity-types", nargs="+", default=["pic50", "pk"],
        choices=["pic50", "pk"],
        help="Affinity types to analyse.",
    )
    parser.add_argument(
        "--thresholds", nargs="+", type=int, default=[5, 6, 7],
        help="Threshold values to analyse.",
    )
    parser.add_argument(
        "--results-base", default=DEFAULT_RESULTS_BASE,
        help=(
            "Root directory holding the supervised, transfer learning and noise "
            "estimation results that are read in."
        ),
    )
    parser.add_argument(
        "--output-base", default=None,
        help=(
            "Root directory for everything this pipeline writes (plots, LaTeX "
            "tables, CSVs, champion_config.json). Defaults to "
            "<results-base>/statistical_tests."
        ),
    )
    parser.add_argument(
        "--noise-results-dir", default=None,
        help=(
            "Directory containing noise estimation results. "
            "Defaults to <results-base>/noise_estimation."
        ),
    )
    parser.add_argument(
        "--scatter", action="store_true", default=False,
        help="Generate scatter plots of individual SL/TL results.",
    )
    parser.add_argument(
        "--champion", action="store_true", default=False,
        help="Generate champion comparison bar chart.",
    )
    parser.add_argument(
        "--tables", action="store_true", default=False,
        help="Generate LaTeX parameter comparison tables.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed debug output.",
    )
    args = parser.parse_args()

    results_base    = args.results_base
    output_base     = args.output_base or f"{results_base}/statistical_tests"
    noise_results_dir = args.noise_results_dir or f"{results_base}/noise_estimation"

    printer.set_verbose(args.verbose)
    printer.legend()

    if not (args.scatter or args.champion or args.tables):
        printer.error("Specify at least one of --scatter, --champion, --tables.")
        return

    printer.info(f"Reading results from : {results_base}")
    printer.info(f"Writing outputs to   : {output_base}")

    df_sl_global, df_tl_global = load_all_data(
        args.affinity_types, args.thresholds, results_base
    )
    if df_sl_global is None:
        return

    data_quality = DataQuality(
        df_sl=df_sl_global,
        df_tl=df_tl_global,
        affinity_types=args.affinity_types,
    )
    data_quality.analyze_data_quality()

    controller = MainAnalysisController(
        df_sl=df_sl_global,
        df_tl=df_tl_global,
        noise_results_dir=noise_results_dir,
        scoring_function="mean_mcc_minus_std",
        rank_method="dense",
        metric_decimals=3,
    )

    if args.scatter:
        print_phase_banner("SCATTER PLOTS")
        for threshold in args.thresholds:
            for affinity_type in args.affinity_types:
                out = f"{output_base}/scatter/threshold_{threshold}"
                controller.run_scatter(affinity_type, threshold, out)

    if args.tables:
        print_phase_banner("LATEX TABLES")
        for affinity_type in args.affinity_types:
            for threshold in args.thresholds:
                out = f"{output_base}/tables/threshold_{threshold}"
                controller.run_tables(
                    affinity_type, out, threshold=threshold, verbose=args.verbose
                )
            out = f"{output_base}/tables/global"
            controller.run_tables(
                affinity_type, out, threshold=None, verbose=args.verbose
            )

    if args.champion:
        if len(args.thresholds) < 2:
            printer.warning(
                "--champion requires at least 2 thresholds for cross-threshold ranking. "
                f"Only {args.thresholds} provided — skipping."
            )
        else:
            print_phase_banner("CHAMPION COMPARISON")
            out = f"{output_base}/champion"
            controller.run_champion(args.affinity_types, out)

    print_phase_banner("ALL ANALYSES COMPLETE")


if __name__ == "__main__":
    main()
