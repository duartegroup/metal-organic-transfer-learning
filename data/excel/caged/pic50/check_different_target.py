#!/usr/bin/env python3
"""
Script to compare IC50/pIC50 values between caged_pic50.csv and new_caged_pic50.csv datasets.
Identifies differences in IC50 values for the same compound-protein combinations.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both CSV datasets.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Original dataset and new dataset.
    """
    try:
        # Define file paths
        original_file = "caged_pic50_updated.csv"
        new_file = "new_caged_pic50.csv"

        # Load datasets
        print(f"Loading {original_file}...")
        df_original = pd.read_csv(original_file)
        print(f"Loaded {len(df_original)} rows from {original_file}")

        print(f"Loading {new_file}...")
        df_new = pd.read_csv(new_file)
        print(f"Loaded {len(df_new)} rows from {new_file}")

        return df_original, df_new

    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        sys.exit(1)


def compare_ic50_values(
    df_original: pd.DataFrame,
    df_new: pd.DataFrame,
    tolerance: float = 0.1,
) -> pd.DataFrame:
    """
    Compare IC50/pIC50 values between datasets for matching compound-protein pairs.

    Parameters
    ----------
    df_original : pd.DataFrame
        Original dataset.
    df_new : pd.DataFrame
        New dataset.
    tolerance : float, default=0.1
        Tolerance for considering values as "different" (log units).

    Returns
    -------
    pd.DataFrame
        DataFrame with comparison results.
    """

    # Create comparison keys
    df_original["comparison_key"] = (
        df_original["compound_n"].astype(str) + "_" + df_original["protein"].astype(str)
    )
    df_new["comparison_key"] = (
        df_new["compound_n"].astype(str) + "_" + df_new["protein"].astype(str)
    )

    # Merge datasets on compound_n and protein
    merged = pd.merge(
        df_original[["compound_n", "protein", "ic50_(nm)", "pic50", "comparison_key"]],
        df_new[["compound_n", "protein", "ic50_(nm)", "pic50", "comparison_key"]],
        on=["compound_n", "protein"],
        how="inner",
        suffixes=("_original", "_new"),
    )

    print(f"\nFound {len(merged)} matching compound-protein pairs between datasets")

    if len(merged) == 0:
        print("No matching pairs found. Checking for overlap...")
        original_keys = set(df_original["comparison_key"])
        new_keys = set(df_new["comparison_key"])
        overlap = original_keys.intersection(new_keys)
        print(f"Key overlap: {len(overlap)} pairs")
        return pd.DataFrame()

    # Calculate differences
    merged["ic50_diff"] = merged["ic50_(nm)_new"] - merged["ic50_(nm)_original"]
    merged["pic50_diff"] = merged["pic50_new"] - merged["pic50_original"]
    merged["pic50_abs_diff"] = abs(merged["pic50_diff"])

    # Identify significant differences
    merged["significant_diff"] = merged["pic50_abs_diff"] > tolerance

    # Calculate percentage change for IC50
    merged["ic50_pct_change"] = (
        (merged["ic50_(nm)_new"] - merged["ic50_(nm)_original"])
        / merged["ic50_(nm)_original"]
        * 100
    )

    return merged


def print_summary_statistics(comparison_df: pd.DataFrame, tolerance: float = 0.1) -> None:
    """Print summary statistics of the comparison."""

    if len(comparison_df) == 0:
        print("No data to analyze.")
        return

    total_pairs = len(comparison_df)
    significant_diffs = sum(comparison_df["significant_diff"])

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Total compound-protein pairs compared: {total_pairs}")
    print(
        f"Pairs with significant pIC50 differences (>{tolerance}): {significant_diffs}"
    )
    print(
        f"Percentage with significant differences: {significant_diffs/total_pairs*100:.1f}%"
    )

    # IC50 statistics
    print(f"\nIC50 Statistics (nM):")
    print(
        f"  Original IC50 range: {comparison_df['ic50_(nm)_original'].min():.1f} - {comparison_df['ic50_(nm)_original'].max():.1f}"
    )
    print(
        f"  New IC50 range: {comparison_df['ic50_(nm)_new'].min():.1f} - {comparison_df['ic50_(nm)_new'].max():.1f}"
    )
    print(f"  Mean IC50 difference: {comparison_df['ic50_diff'].mean():.1f} nM")
    print(
        f"  Median IC50 percentage change: {comparison_df['ic50_pct_change'].median():.1f}%"
    )

    # pIC50 statistics
    print(f"\npIC50 Statistics:")
    print(
        f"  Original pIC50 range: {comparison_df['pic50_original'].min():.2f} - {comparison_df['pic50_original'].max():.2f}"
    )
    print(
        f"  New pIC50 range: {comparison_df['pic50_new'].min():.2f} - {comparison_df['pic50_new'].max():.2f}"
    )
    print(f"  Mean pIC50 difference: {comparison_df['pic50_diff'].mean():.3f}")
    print(
        f"  Max absolute pIC50 difference: {comparison_df['pic50_abs_diff'].max():.3f}"
    )


def print_significant_differences(
    comparison_df: pd.DataFrame, tolerance: float = 0.01, max_display: int = 20
) -> None:
    """Print details of compounds with significant differences."""

    significant = comparison_df[comparison_df["significant_diff"]].copy()
    significant = significant.sort_values("pic50_abs_diff", ascending=False)

    if len(significant) == 0:
        print(
            f"\nNo significant differences found (threshold: {tolerance} pIC50 units)"
        )
        return

    print(f"\n{'='*80}")
    print(f"SIGNIFICANT DIFFERENCES (showing top {min(max_display, len(significant))})")
    print(f"{'='*80}")

    print(
        f"{'Compound':<10} {'Protein':<10} {'IC50_orig':<12} {'IC50_new':<12} {'pIC50_orig':<12} {'pIC50_new':<12} {'Diff':<8}"
    )
    print(f"{'-'*80}")

    for idx, row in significant.head(max_display).iterrows():
        print(
            f"{row['compound_n']:<10} {row['protein']:<10} "
            f"{row['ic50_(nm)_original']:<12.1f} {row['ic50_(nm)_new']:<12.1f} "
            f"{row['pic50_original']:<12.3f} {row['pic50_new']:<12.3f} "
            f"{row['pic50_diff']:<8.3f}"
        )


def print_full_row_differences(
    df_original: pd.DataFrame,
    df_new: pd.DataFrame,
    diff_df: pd.DataFrame,
    max_display: int = 5,
) -> None:
    """Show the complete rows for entries that have significant pIC50 differences."""
    sig = diff_df[diff_df["significant_diff"]]
    if sig.empty:
        return

    print(f"\n{'='*80}")
    print(f"FULL ROW COMPARISONS (first {min(max_display, len(sig))})")
    print(f"{'='*80}")

    shown = 0
    for _, r in sig.iterrows():
        if shown >= max_display:
            break

        mask_orig = (df_original["compound_n"] == r["compound_n"]) & (
            df_original["protein"] == r["protein"]
        )
        mask_new = (df_new["compound_n"] == r["compound_n"]) & (
            df_new["protein"] == r["protein"]
        )

        orig_row = df_original[mask_orig].iloc[0]
        new_row = df_new[mask_new].iloc[0]

        print(f"\n--- Compound {r['compound_n']}  /  Protein {r['protein']} ---")
        print("ORIGINAL:")
        print(orig_row.to_json(indent=2))
        print("NEW:")
        print(new_row.to_json(indent=2))
        shown += 1


def update_original_dataset(
    df_original: pd.DataFrame,
    df_new: pd.DataFrame,
    diff_df: pd.DataFrame,
    columns_to_update: Optional[List[str]] = None,
    output_file: str = "caged_pic50_updated.csv",
) -> pd.DataFrame:
    """
    Replace values in df_original with values from df_new for every compound-protein
    pair that shows a significant pIC50 difference.
    """
    if columns_to_update is None:
        columns_to_update = ["ic50_(nm)", "pic50"]

    updated = df_original.copy()
    sig = diff_df[diff_df["significant_diff"]]

    for _, r in sig.iterrows():
        orig_mask = (updated["compound_n"] == r["compound_n"]) & (
            updated["protein"] == r["protein"]
        )
        new_mask = (df_new["compound_n"] == r["compound_n"]) & (
            df_new["protein"] == r["protein"]
        )
        if not new_mask.any():
            continue

        new_vals = df_new[new_mask].iloc[0]
        for col in columns_to_update:
            if col in updated.columns and col in new_vals:
                updated.loc[orig_mask, col] = new_vals[col]

    updated.to_csv(output_file, index=False)
    print(f"\nUpdated original dataset written to '{output_file}'")
    return updated


def save_results(
    comparison_df: pd.DataFrame, output_file: str = "ic50_comparison_results.csv"
) -> None:
    """Save comparison results to CSV file."""

    if len(comparison_df) == 0:
        print("No results to save.")
        return

    # Select relevant columns for output
    output_cols = [
        "compound_n",
        "protein",
        "ic50_(nm)_original",
        "ic50_(nm)_new",
        "ic50_diff",
        "ic50_pct_change",
        "pic50_original",
        "pic50_new",
        "pic50_diff",
        "pic50_abs_diff",
        "significant_diff",
    ]

    output_df = comparison_df[output_cols].copy()
    output_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


def main() -> None:
    """Main function to run the comparison analysis."""

    print("IC50/pIC50 Comparison Analysis")
    print("=" * 40)

    # Load datasets
    df_original, df_new = load_datasets()

    # Show basic info about datasets
    print(f"\nDataset Information:")
    print(
        f"Original dataset: {len(df_original)} rows, {len(df_original.columns)} columns"
    )
    print(f"New dataset: {len(df_new)} rows, {len(df_new.columns)} columns")

    # Check for required columns
    required_cols = ["compound_n", "protein", "ic50_(nm)", "pic50"]

    for col in required_cols:
        if col not in df_original.columns:
            print(f"Error: Column '{col}' not found in original dataset")
            return
        if col not in df_new.columns:
            print(f"Error: Column '{col}' not found in new dataset")
            return

    # Perform comparison
    tolerance = 0.01  # pIC50 units
    comparison_results = compare_ic50_values(df_original, df_new, tolerance)

    if len(comparison_results) > 0:
        # Print analysis results
        print_summary_statistics(comparison_results, tolerance)
        print_significant_differences(comparison_results, tolerance)

        # NEW: show all columns for the differing pairs
        print_full_row_differences(df_original, df_new, comparison_results)

        # columns that should be replaced if a significant pIC50 difference is found
        cols_to_patch = [
            "ic50_(nm)",
            "ic50_error_(nm)",
            "ki_(nm)",
            "ki_error_(nm)",
            "pic50",
            "pki_",
            "kd/ki_converted",
            "km_(nm)",
            "km_error(nm)",
            "subs._conc.(nm)",
            "reference_km",
        ]

        update_original_dataset(
            df_original,
            df_new,
            comparison_results,
            columns_to_update=cols_to_patch,
        )

        # Save results
        save_results(comparison_results)

        # Additional analysis: Check for compounds only in one dataset
        original_keys = set(
            df_original["compound_n"].astype(str)
            + "_"
            + df_original["protein"].astype(str)
        )
        new_keys = set(
            df_new["compound_n"].astype(str) + "_" + df_new["protein"].astype(str)
        )

        only_original = original_keys - new_keys
        only_new = new_keys - original_keys

        print(f"\n{'='*60}")
        print("DATASET COMPLETENESS")
        print(f"{'='*60}")
        print(f"Compound-protein pairs only in original: {len(only_original)}")
        print(f"Compound-protein pairs only in new: {len(only_new)}")

        if len(only_original) > 0:
            print(f"\nSample pairs only in original (first 5):")
            for key in list(only_original)[:5]:
                print(f"  {key}")

        if len(only_new) > 0:
            print(f"\nSample pairs only in new dataset (first 5):")
            for key in list(only_new)[:5]:
                print(f"  {key}")


if __name__ == "__main__":
    main()
