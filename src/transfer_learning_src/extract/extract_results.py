#!/usr/bin/env python3
"""
Extract and analyze transfer learning results from multiple experimental setups.

This script processes results from various combinations of:
- Inhibitors: ACSF, MACE, SOAP
- Biological targets: molecule, pocket, protein
- Scalers: minmax_scaler, None_scaler, standard_scaler
- Affinity types: pic50, pk
- Transfer learning methods: various pretraining and fine-tuning approaches
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple, Optional
import warnings
import multiprocessing
from tqdm import tqdm
import subprocess
from functools import partial

warnings.filterwarnings("ignore")


# --- HELPER FUNCTION FOR FAST FILE SEARCH ---
def _parse_path(base_path: Path, file_path_str: str) -> Optional[Dict]:
    """
    Parse a single file path string into a metadata dictionary.

    Parameters
    ----------
    base_path : Path
        The base directory path for relative path calculations.
    file_path_str : str
        The file path string to parse.

    Returns
    -------
    Optional[Dict]
        Dictionary containing parsed metadata (inhibitor, biological_target, scaler, etc.)
        or None if parsing fails.
    """
    try:
        csv_file = Path(file_path_str)
        parts = csv_file.relative_to(base_path).parts

        if len(parts) >= 7:
            inhibitor = parts[1]
            biological_target = parts[2]
            scaler = parts[3]
            affinity_type = parts[4]
            scaling_config = parts[5]

            cross_domain_scaling = "cross_domain" in scaling_config
            source_smote = (
                "source_smote" in scaling_config
                and "no_source_smote" not in scaling_config
            )
            target_smote = (
                "target_smote" in scaling_config
                and "no_target_smote" not in scaling_config
            )

            return {
                "file_path": csv_file,
                "inhibitor": inhibitor,
                "biological_target": biological_target,
                "scaler": scaler,
                "affinity_type": affinity_type,
                "scaling_config": scaling_config,
                "cross_domain_scaling": cross_domain_scaling,
                "source_smote": source_smote,
                "target_smote": target_smote,
                "filename": csv_file.name,
                "relative_path": str(csv_file.relative_to(base_path)),
            }
    except Exception as e:
        print(f"Warning: Could not parse path {file_path_str}: {e}")

    return None


# --- SLOW, FALLBACK FILE SEARCH FUNCTION ---
def find_summary_csv_files(base_dir: str) -> List[Dict]:
    """
    Find all summary CSV files using Python's rglob (fallback method).

    Parameters
    ----------
    base_dir : str
        The base directory to search for summary CSV files.

    Returns
    -------
    List[Dict]
        List of dictionaries containing metadata for each CSV file found.
    """
    base_path = Path(base_dir)
    csv_files_metadata = []

    for csv_file in base_path.rglob("*summary*.csv"):
        metadata = _parse_path(base_path, str(csv_file))
        if metadata:
            csv_files_metadata.append(metadata)

    return csv_files_metadata


# --- OPTIMIZED FAST FILE SEARCH FUNCTION ---
def find_summary_csv_files_fast(base_dir: str) -> List[Dict]:
    """
    Find all summary CSV files using the fast 'find' command and process results in parallel.

    Falls back to the slower Python method if 'find' command fails.

    Parameters
    ----------
    base_dir : str
        The base directory to search for summary CSV files.

    Returns
    -------
    List[Dict]
        List of dictionaries containing metadata for each CSV file found.
    """
    print("🚀 Using fast file search with the 'find' command...")
    base_path = Path(base_dir)

    command = ["find", base_dir, "-type", "f", "-name", "*summary*.csv"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        file_paths = result.stdout.strip().split("\n")

        if not file_paths or file_paths == [""]:
            return []

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"⚠️ Warning: 'find' command failed ({e}).")
        print("Falling back to slower Python-based file search...")
        return find_summary_csv_files(base_dir)

    print(f"⚙️ Found {len(file_paths)} files. Parsing paths in parallel...")
    num_processes = os.cpu_count()
    parse_func = partial(_parse_path, base_path)

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(parse_func, file_paths)

    csv_files_metadata = [res for res in results if res is not None]

    return csv_files_metadata


# --- PARALLEL CSV LOADING FUNCTION ---
def load_and_process_csv(csv_info: Dict) -> Optional[pd.DataFrame]:
    """
    Load a CSV file and add metadata columns.

    This function is called by each worker process in parallel processing.

    Parameters
    ----------
    csv_info : Dict
        Dictionary containing file path and metadata for the CSV file.

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with loaded data and added metadata columns, or None if loading fails.
    """
    try:
        df = pd.read_csv(csv_info["file_path"])

        df["inhibitor"] = csv_info["inhibitor"]
        df["biological_target"] = csv_info["biological_target"]
        df["scaler"] = csv_info["scaler"]
        df["affinity_type"] = csv_info["affinity_type"]
        df["scaling_config"] = csv_info["scaling_config"]
        df["cross_domain_scaling"] = csv_info["cross_domain_scaling"]
        df["source_smote"] = csv_info["source_smote"]
        df["filename"] = csv_info["filename"]
        df["file_path"] = str(csv_info["file_path"])

        return df

    except Exception as e:
        print(f"Error loading {csv_info['file_path']}: {e}")
        return None


# --- MAIN DATA EXTRACTION ORCHESTRATOR ---
def extract_all_results(base_dir: str) -> pd.DataFrame:
    """
    Extract all results from CSV files using optimized parallel processing.

    Finds files quickly using system commands, then loads them in parallel.

    Parameters
    ----------
    base_dir : str
        The base directory containing the results CSV files.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing all extracted results with metadata.
    """
    # Step 1: Find all files using the fast, parallel method.
    csv_files = find_summary_csv_files_fast(base_dir)
    print(f"✅ Found and processed metadata for {len(csv_files)} CSV files.")

    if not csv_files:
        print("No CSV files found. Check the base directory path.")
        return pd.DataFrame()

    # Step 2: Load the CSV data in parallel.
    num_processes = os.cpu_count()
    print(
        f"\n📚 Loading data from {len(csv_files)} files in parallel using {num_processes} processes..."
    )

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(pool.imap(load_and_process_csv, csv_files), total=len(csv_files))
        )

    all_dataframes = [df for df in results if df is not None and not df.empty]

    if not all_dataframes:
        print("No valid data found in CSV files.")
        return pd.DataFrame()

    print("\n🔗 Combining all dataframes...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")

    return combined_df


# --- ANALYSIS FUNCTIONS (UNCHANGED) ---
def analyze_transfer_learning_methods(df: pd.DataFrame) -> Dict:
    """
    Analyze the different transfer learning methods and their performance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing transfer learning experiment results.

    Returns
    -------
    Dict
        Dictionary containing analysis results including methods, performance statistics.
    """
    analysis = {}
    methods = df["method"].unique()
    analysis["methods"] = sorted(methods)
    analysis["num_methods"] = len(methods)

    numeric_cols = ["val/accuracy", "val/f1", "val/auroc", "val/mcc"]
    method_performance = {}

    for method in methods:
        method_df = df[df["method"] == method]
        method_stats = {}
        for col in numeric_cols:
            if col in method_df.columns:
                values = pd.to_numeric(method_df[col], errors="coerce").dropna()
                if len(values) > 0:
                    method_stats[col] = {
                        "mean": values.mean(),
                        "std": values.std(),
                        "min": values.min(),
                        "max": values.max(),
                        "count": len(values),
                    }
        method_performance[method] = method_stats

    analysis["method_performance"] = method_performance
    return analysis


def generate_summary_statistics(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive summary statistics from experimental results.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing experimental results.

    Returns
    -------
    Dict
        Dictionary containing summary statistics including counts, means, standard deviations.
    """
    stats = {
        "total_experiments": len(df),
        "unique_inhibitors": sorted(df["inhibitor"].unique()),
        "unique_biological_targets": sorted(df["biological_target"].unique()),
        "unique_scalers": sorted(df["scaler"].unique()),
        "unique_affinity_types": sorted(df["affinity_type"].unique()),
        "unique_methods": sorted(df["method"].unique()),
    }

    numeric_cols = ["val/accuracy", "val/f1", "val/auroc", "val/mcc"]
    for col in numeric_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(values) > 0:
                stats[f"{col}_stats"] = {
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                    "median": values.median(),
                    "count": len(values),
                }
    return stats


def find_best_combinations(
    df: pd.DataFrame, metric: str = "val/mcc", top_n: int = 20
) -> pd.DataFrame:
    """
    Find the best performing combinations for a given metric.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing experimental results.
    metric : str
        The metric to optimize for (default: 'val/mcc').
    top_n : int
        Number of top combinations to return (default: 20).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the top N best combinations sorted by metric performance.
    """
    df_filtered = df[df["method"] != "pre_train"].copy()

    if metric not in df_filtered.columns:
        print(f"Warning: Metric {metric} not found in data")
        return pd.DataFrame()

    df_filtered[metric] = pd.to_numeric(df_filtered[metric], errors="coerce")
    df_filtered.dropna(subset=[metric], inplace=True)

    if df_filtered.empty:
        print(f"No valid data for metric {metric}")
        return pd.DataFrame()

    groupby_cols = [
        "inhibitor",
        "biological_target",
        "scaler",
        "affinity_type",
        "method",
    ]
    best_combinations = (
        df_filtered.groupby(groupby_cols)
        .agg({metric: ["mean", "std", "count", "min", "max"], "fold": "nunique"})
        .round(4)
    )

    best_combinations.columns = [
        "_".join(col).strip() for col in best_combinations.columns
    ]
    best_combinations = best_combinations.reset_index()

    sort_col = f"{metric}_mean"
    best_combinations = best_combinations.sort_values(sort_col, ascending=False)

    return best_combinations.head(top_n)


def create_comprehensive_comparison_table(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create comprehensive comparison tables for different metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing experimental results.
    output_dir : str
        Directory where comparison tables will be saved.

    Returns
    -------
    None
    """
    metrics = ["val/mcc", "val/accuracy", "val/f1", "val/auroc"]
    os.makedirs(output_dir, exist_ok=True)

    for metric in metrics:
        if metric in df.columns:
            print(f"\nGenerating comparison table for {metric}...")
            best_combos = find_best_combinations(df, metric, top_n=50)
            if not best_combos.empty:
                csv_filename = os.path.join(
                    output_dir, f'best_combinations_{metric.replace("/", "_")}.csv'
                )
                best_combos.to_csv(csv_filename, index=False)
                print(f"Saved: {csv_filename}")

                print(f"\nTop 10 combinations for {metric}:")
                print("=" * 100)
                display_cols = [
                    "inhibitor",
                    "biological_target",
                    "scaler",
                    "affinity_type",
                    "method",
                    f"{metric}_mean",
                    f"{metric}_std",
                ]
                if all(col in best_combos.columns for col in display_cols):
                    for i, (_, row) in enumerate(best_combos.head(10).iterrows(), 1):
                        print(
                            f"{i:2d}. {row['inhibitor']:4s} + {row['biological_target']:8s} + "
                            f"{row['scaler']:12s} + {row['affinity_type']:5s} + "
                            f"{row['method']:25s} = {row[f'{metric}_mean']:6.3f} ± {row[f'{metric}_std']:6.3f}"
                        )


def generate_method_comparison_latex_table(
    df: pd.DataFrame, output_dir: str, metric: str = "val/mcc"
) -> None:
    """
    Generate LaTeX table comparing different transfer learning methods.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing experimental results.
    output_dir : str
        Directory where LaTeX table will be saved.
    metric : str
        The metric to compare (default: 'val/mcc').

    Returns
    -------
    None
    """
    df_filtered = df[df["method"] != "pre_train"].copy()
    if metric not in df_filtered.columns:
        print(f"Warning: Metric {metric} not found in data")
        return

    df_filtered[metric] = pd.to_numeric(df_filtered[metric], errors="coerce")
    df_filtered.dropna(subset=[metric], inplace=True)

    method_stats = (
        df_filtered.groupby("method")[metric]
        .agg(["mean", "std", "count", "min", "max"])
        .round(4)
    )
    method_stats = method_stats.sort_values("mean", ascending=False)

    latex_filename = os.path.join(
        output_dir,
        f'transfer_learning_methods_comparison_{metric.replace("/", "_")}.tex',
    )
    with open(latex_filename, "w") as f:
        f.write(
            "\\documentclass{article}\n\\usepackage{booktabs}\n\\usepackage{geometry}\n"
        )
        f.write(
            "\\geometry{margin=1in}\n\\begin{document}\n\n\\begin{table}[h!]\n\\centering\n"
        )
        f.write(f"\\caption{{Transfer Learning Methods Comparison ({metric})}}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("Method & Mean & Std & Min & Max & Count \\\\\n\\midrule\n")

        for method, row in method_stats.iterrows():
            method_clean = method.replace("_", "\\_")
            f.write(
                f"{method_clean} & {row['mean']:.3f} & {row['std']:.3f} & "
                f"{row['min']:.3f} & {row['max']:.3f} & {int(row['count'])} \\\\\n"
            )

        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write(
            f"\\label{{tab:transfer_learning_methods_{metric.replace('/', '_')}}}\n"
        )
        f.write("\\end{table}\n\n\\end{document}\n")

    print(f"📄 LaTeX table saved to: {latex_filename}")


# --- MAIN EXECUTION BLOCK ---
def main() -> None:
    """
    Main function to extract and analyze transfer learning results.

    Parses command-line arguments, extracts results from CSV files,
    generates summary statistics, and creates comparison tables.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Extract and analyze transfer learning experimental results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory containing the results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="wojtek_suggestions_transfer_learning_analysis",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="val/mcc",
        choices=["val/mcc", "val/accuracy", "val/f1", "val/auroc"],
        help="Primary metric for analysis",
    )

    args = parser.parse_args()

    print("🔬 Transfer Learning Results Extraction and Analysis")
    print("=" * 60)
    print(f"Base directory: {args.base_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Primary metric: {args.metric}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n📊 Extracting results from CSV files...")
    df = extract_all_results(args.base_dir)

    if df.empty:
        print("❌ No data extracted. Exiting.")
        return

    raw_data_file = os.path.join(args.output_dir, "all_transfer_learning_results.csv")
    df.to_csv(raw_data_file, index=False)
    print(f"💾 Raw combined data saved to: {raw_data_file}")

    print("\n📈 Generating summary statistics...")
    summary_stats = generate_summary_statistics(df)
    summary_file = os.path.join(args.output_dir, "summary_statistics.json")
    with open(summary_file, "w") as f:
        json.dump(summary_stats, f, indent=2, default=str)
    print(f"💾 Summary statistics saved to: {summary_file}")

    print(f"\n📋 Key Statistics:")
    print(f"  Total experiments: {summary_stats['total_experiments']}")
    print(f"  Inhibitors: {summary_stats['unique_inhibitors']}")
    print(f"  Transfer learning methods: {len(summary_stats['unique_methods'])}")

    print("\n🧠 Analyzing transfer learning methods...")
    tl_analysis = analyze_transfer_learning_methods(df)
    tl_analysis_file = os.path.join(args.output_dir, "transfer_learning_analysis.json")
    with open(tl_analysis_file, "w") as f:
        json.dump(tl_analysis, f, indent=2, default=str)
    print(f"💾 Transfer learning analysis saved to: {tl_analysis_file}")

    print(f"\n🏆 Finding best combinations for {args.metric}...")
    create_comprehensive_comparison_table(df, args.output_dir)

    print(f"\n📄 Generating LaTeX comparison table...")
    generate_method_comparison_latex_table(df, args.output_dir, args.metric)

    print(f"\n✅ Analysis complete! Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
