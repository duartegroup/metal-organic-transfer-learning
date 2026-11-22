#!/usr/bin/env python3
"""Rebuild the fold-index pickles from the published fold assignment.

`data/fold_split/spectral_clustering_analysis/caged_{pic50,pk}_spectral_clustering_fold_indices.pkl`
is gitignored and was in neither the repository nor the handover. Re-running
`spectral_clustering_split.py` does NOT reproduce the published folds: with the
scikit-learn and scipy versions available here it agrees on only 129/214 pKi and
86/201 pIC50 compounds.

The authoritative assignment, however, is committed: `data/excel/metadata.xlsx`
carries a `fold` column per measurement, and its `entry_name` values cover every
compound the published runs used — 214/214 for pKi and 201/201 for pIC50, with one
extra entry in each HDF5 that the published runs evidently excluded.

So the pickles are rebuilt from the spreadsheet rather than reconstructed by
clustering. This is exact rather than approximate, and needs nothing from Leiden.

Usage, from the repository root:
    python fold_indices_from_metadata.py --repo . [--write]

Without --write it only reports what it would do.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

ENDPOINTS = {"pKi": "pk", "pIC50": "pic50"}


def build(repo: Path) -> dict[str, dict]:
    metadata = pd.read_excel(repo / "data" / "excel" / "metadata.xlsx",
                             sheet_name="measurements")
    built = {}

    for activity_type, tag in ENDPOINTS.items():
        subset = metadata[metadata.activity_type == activity_type]
        names = subset.entry_name.astype(str)
        folds = subset.fold.astype(int)

        if sorted(folds.unique()) != list(range(folds.nunique())):
            raise SystemExit(
                f"{activity_type}: fold column is not 0..n-1, got "
                f"{sorted(folds.unique())}"
            )

        everything = list(names)
        if len(set(everything)) != len(everything):
            raise SystemExit(f"{activity_type}: duplicate entry_name values")

        fold_indices = {}
        for fold in sorted(folds.unique()):
            test = [n for n, f in zip(names, folds) if f == fold]
            train = [n for n, f in zip(names, folds) if f != fold]
            assert not set(test) & set(train)
            assert len(test) + len(train) == len(everything)
            fold_indices[int(fold)] = {"train_names": train, "test_names": test}

        built[tag] = fold_indices

    return built


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    out_dir = args.repo / "data" / "fold_split" / "spectral_clustering_analysis"
    built = build(args.repo)

    for tag, fold_indices in built.items():
        sizes = [len(fold_indices[f]["test_names"]) for f in sorted(fold_indices)]
        total = sum(sizes)
        path = out_dir / f"caged_{tag}_spectral_clustering_fold_indices.pkl"
        print(f"caged_{tag}: {len(fold_indices)} folds, test sizes {sizes}, {total} compounds")

        if args.write:
            out_dir.mkdir(parents=True, exist_ok=True)
            if path.exists():
                backup = path.with_suffix(".pkl.regenerated")
                if not backup.exists():
                    path.rename(backup)
                    print(f"  previous (clustering-derived) file kept as {backup.name}")
            with open(path, "wb") as handle:
                pickle.dump(fold_indices, handle)
            print(f"  wrote {path}")
        else:
            print(f"  would write {path}  (pass --write)")


if __name__ == "__main__":
    main()
