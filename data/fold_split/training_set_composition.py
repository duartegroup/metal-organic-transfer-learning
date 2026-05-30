"""
training_set_composition.py

Print the per-protein-series breakdown of the training set for any fold,
dataset, and affinity type.

Intended use: fill in the "only X hsBChE vs Y for others" statement in the
paper by running this for fold 5, caged, pic50, threshold 6.

Usage (run from the repo root or any directory — paths are absolute):
    python data/fold_split/training_set_composition.py
    python data/fold_split/training_set_composition.py --fold 5 --dataset caged --affinity pic50
"""

import argparse
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict

# Fold indices live here (relative to repo root)
FOLD_INDICES_DIR = Path(__file__).resolve().parent / "spectral_clustering_analysis"


def load_fold_indices(dataset: str, affinity_type: str, n_folds: int = 5) -> Dict[int, Any]:
    """
    Load pre-computed spectral clustering fold indices from disk.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g. 'caged', 'organic').
    affinity_type : str
        Affinity type ('pic50' or 'pk').
    n_folds : int, default=5
        Total number of folds (used only to locate the file by convention).

    Returns
    -------
    Dict[int, Any]
        Mapping from 0-based fold index to dict with 'train_names' and 'test_names'.
    """
    pkl_path = (
        FOLD_INDICES_DIR
        / f"{dataset}_{affinity_type}_spectral_clustering_fold_indices.pkl"
    )
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def parse_protein(compound_name: str) -> str:
    """
    Extract the protein/inhibitor series name from a compound key.

    Expected formats:
        compound_108_hsBChE-157   → hsBChE
        compound_42_hsAChE-23     → hsAChE
        compound_5_MMP-12-7       → MMP-12
    """
    # Everything between the second underscore and the last hyphen+digits block
    match = re.match(r"compound_\d+_(.+)-\d+$", compound_name)
    if match:
        return match.group(1)
    # Fallback: return the whole name so nothing is silently dropped
    return compound_name


def analyse(fold: int, dataset: str, affinity_type: str, n_folds: int = 5) -> None:
    """
    Print a per-protein-series breakdown of the training and test sets for a given fold.

    Parameters
    ----------
    fold : int
        1-based fold number.
    dataset : str
        Dataset name (e.g. 'caged', 'organic').
    affinity_type : str
        Affinity type ('pic50' or 'pk').
    n_folds : int, default=5
        Total number of folds.
    """
    fold_indices = load_fold_indices(dataset, affinity_type, n_folds)

    # fold_indices is 0-indexed internally
    fold_idx = fold - 1
    if fold_idx not in fold_indices:
        raise KeyError(
            f"Fold {fold} (index {fold_idx}) not found. "
            f"Available indices: {sorted(fold_indices.keys())}"
        )

    train_names = fold_indices[fold_idx]["train_names"]
    test_names = fold_indices[fold_idx]["test_names"]

    train_counts = Counter(parse_protein(n) for n in train_names)
    test_counts = Counter(parse_protein(n) for n in test_names)
    all_proteins = sorted(set(train_counts) | set(test_counts))

    print(f"\nDataset : {dataset}  |  Affinity : {affinity_type}  |  Fold : {fold}")
    print(f"Training compounds : {len(train_names)}  |  Test compounds : {len(test_names)}")
    print()
    print(f"{'Protein series':<25}  {'Train':>7}  {'Test':>6}")
    print("-" * 42)
    for protein in all_proteins:
        print(f"{protein:<25}  {train_counts.get(protein, 0):>7}  {test_counts.get(protein, 0):>6}")
    print("-" * 42)
    print(f"{'TOTAL':<25}  {len(train_names):>7}  {len(test_names):>6}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Training set composition per fold.")
    parser.add_argument("--fold", type=int, default=5, help="Fold number (1-indexed).")
    parser.add_argument("--dataset", type=str, default="caged", help="Dataset name (e.g. caged, organic).")
    parser.add_argument("--affinity", type=str, default="pic50", help="Affinity type (pic50 or pk).")
    parser.add_argument("--n_folds", type=int, default=5, help="Total number of folds.")
    args = parser.parse_args()

    analyse(
        fold=args.fold,
        dataset=args.dataset,
        affinity_type=args.affinity,
        n_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()
