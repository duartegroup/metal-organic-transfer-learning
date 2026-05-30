import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, DefaultDict
import argparse
import re


def find_and_parse_results(base_dir: str) -> DefaultDict:
    """
    Find and parse all noise_analysis_results.pkl files to extract MCC and ROC AUC scores.

    Parameters
    ----------
    base_dir : str
        Base directory where threshold folders are located.

    Returns
    -------
    DefaultDict
        Nested dictionary containing results organized by threshold, affinity type,
        metric (MCC/ROC AUC), and train/test split.
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    base_path = Path(base_dir)
    pkl_files = list(base_path.glob("threshold_*/*/noise_analysis_results.pkl"))

    for pkl_file in pkl_files:
        path_str = str(pkl_file)

        threshold_match = re.search(r"threshold_([\d\.]+)", path_str)
        affinity_match = re.search(r"(pic50|pk)", path_str)

        if not threshold_match or not affinity_match:
            continue

        threshold = threshold_match.group(1)
        affinity = affinity_match.group(1)

        try:
            data = pd.read_pickle(pkl_file)
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            continue

        train_mcc = []
        train_roc_auc = []
        test_mcc = []
        test_roc_auc = []

        # Check if required keys exist
        if "train_results" not in data or "test_results" not in data:
            print(f"Missing train_results or test_results in {pkl_file}")
            continue

        train_folds = sorted(data["train_results"].keys())
        test_folds = sorted(data["test_results"].keys())

        # Process training folds
        for fold in train_folds:
            df_train = data["train_results"][fold]
            if not isinstance(df_train, pd.DataFrame):
                print(
                    f"Train fold {fold} in {pkl_file} is not a DataFrame: {type(df_train)}"
                )
                continue

            # Check if required columns exist
            if "matthews_corrcoef" not in df_train.columns:
                print(
                    f"Missing 'matthews_corrcoef' column in train fold {fold} of {pkl_file}"
                )
                print(f"Available columns: {list(df_train.columns)}")
                continue
            if "roc_auc" not in df_train.columns:
                print(f"Missing 'roc_auc' column in train fold {fold} of {pkl_file}")
                print(f"Available columns: {list(df_train.columns)}")
                continue

            train_mcc.append(df_train["matthews_corrcoef"].mean())
            train_roc_auc.append(df_train["roc_auc"].mean())

        # Process test folds
        for fold in test_folds:
            df_test = data["test_results"][fold]
            if not isinstance(df_test, pd.DataFrame):
                print(
                    f"Test fold {fold} in {pkl_file} is not a DataFrame: {type(df_test)}"
                )
                continue

            # Check if required columns exist
            if "matthews_corrcoef" not in df_test.columns:
                print(
                    f"Missing 'matthews_corrcoef' column in test fold {fold} of {pkl_file}"
                )
                print(f"Available columns: {list(df_test.columns)}")
                continue
            if "roc_auc" not in df_test.columns:
                print(f"Missing 'roc_auc' column in test fold {fold} of {pkl_file}")
                print(f"Available columns: {list(df_test.columns)}")
                continue

            test_mcc.append(df_test["matthews_corrcoef"].mean())
            test_roc_auc.append(df_test["roc_auc"].mean())

        # Only add results if we have valid data
        if train_mcc or test_mcc:
            results[threshold][affinity]["MCC"]["Train"] = train_mcc
            results[threshold][affinity]["MCC"]["Test"] = test_mcc
            results[threshold][affinity]["ROC AUC"]["Train"] = train_roc_auc
            results[threshold][affinity]["ROC AUC"]["Test"] = test_roc_auc
        else:
            print(f"No valid data found in {pkl_file}")

    return results


def format_and_print_results(results: Dict) -> None:
    """
    Format and print the parsed results as markdown tables.

    Parameters
    ----------
    results : Dict
        Nested dictionary containing parsed results from find_and_parse_results.

    Returns
    -------
    None
    """
    for threshold in sorted(results.keys(), key=float):
        if not results[threshold]:
            continue

        for metric in ["MCC", "ROC AUC"]:
            has_data = any(
                metric in results[threshold].get(affinity, {})
                for affinity in ["pic50", "pk"]
            )
            if not has_data:
                continue

            print(f"## Threshold {threshold} — {metric}\n")

            for affinity in ["pic50", "pk"]:
                if (
                    affinity in results[threshold]
                    and metric in results[threshold][affinity]
                ):
                    affinity_name = "pIC50" if affinity == "pic50" else "pK"
                    print(f"**{affinity_name}**\n")

                    table_data = {}

                    if (
                        "Train" in results[threshold][affinity][metric]
                        and results[threshold][affinity][metric]["Train"]
                    ):
                        train_data = results[threshold][affinity][metric]["Train"]
                        table_data["Train"] = [f"{v:.3f}" for v in train_data]

                    if (
                        "Test" in results[threshold][affinity][metric]
                        and results[threshold][affinity][metric]["Test"]
                    ):
                        test_data = results[threshold][affinity][metric]["Test"]
                        table_data["Test"] = [f"{v:.3f}" for v in test_data]

                    if not table_data:
                        continue

                    # Check if Train and Test have the same number of folds
                    fold_counts = [len(v) for v in table_data.values()]
                    if len(set(fold_counts)) > 1:
                        print(
                            f"Warning: Mismatched fold counts for {affinity_name} {metric} at threshold {threshold}"
                        )
                        print(
                            f"Fold counts: {dict(zip(table_data.keys(), fold_counts))}"
                        )
                        # Use the minimum count to avoid index errors
                        min_folds = min(fold_counts)
                        for key in table_data:
                            table_data[key] = table_data[key][:min_folds]

                    num_folds = len(next(iter(table_data.values())))
                    columns = [f"Fold {i+1}" for i in range(num_folds)]

                    # Create DataFrame and transpose so Train/Test become rows
                    df = pd.DataFrame(table_data).T  # .T transposes the DataFrame
                    df.columns = columns

                    # Ensure index is in Train, Test order
                    if "Train" in df.index and "Test" in df.index:
                        df = df.reindex(["Train", "Test"])

                    print(df.to_markdown(index=True))
                    print("\n")


def main() -> None:
    """
    Main function to parse command-line arguments and run the script.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Extract MCC and ROC AUC from noise analysis pickle files."
    )
    parser.add_argument(
        "base_dir",
        nargs="?",
        default=".",
        help="Base directory where threshold folders are located (default: results/noise_estimation).",
    )
    args = parser.parse_args()

    parsed_data = find_and_parse_results(args.base_dir)
    format_and_print_results(parsed_data)


if __name__ == "__main__":
    main()
