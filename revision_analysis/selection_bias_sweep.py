#!/usr/bin/env python3
"""Estimate the winner's curse in selecting a configuration on the reported folds.

The published numbers are the maximum, over 432 configurations, of a mean across
the same 5 folds the number is reported on. Taking the maximum of 432 noisy
estimates is upward biased, and the bias is separate from, and additional to, the
checkpoint-selection bias already measured (+0.100 MCC).

The estimate here is split-half and uses only the sweep results, no re-training.
For every way of splitting the 5 folds into a selection set and a held-out set:

  * pick the configuration with the highest mean on the selection folds,
  * read that same configuration's mean on the held-out folds.

The gap between the two is the part of the winning score that does not
generalise. Averaged over all splits it estimates how much of the published
maximum is selection noise.

Usage:
    python selection_bias_sweep.py SWEEP_ROOT [--out results.csv]

where SWEEP_ROOT contains final_runs_threshold_{5,6,7}/.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_KEYS = [
    "inhibitor",
    "biological_target",
    "scaler",
    "scaling_config",
    "cross_domain_scaling",
    "source_smote",
]
METRIC = "val/mcc"


def _pivot(df: pd.DataFrame) -> pd.DataFrame:
    """configs x folds matrix of the metric, dropping configs with missing folds."""
    wide = df.pivot_table(
        index=CONFIG_KEYS, columns="fold", values=METRIC, aggfunc="mean"
    )
    return wide.dropna()


def split_half_bias(wide: pd.DataFrame, n_select: int) -> dict:
    folds = list(wide.columns)
    selected_scores, heldout_scores = [], []

    for selection in combinations(folds, n_select):
        heldout = [f for f in folds if f not in selection]
        sel_mean = wide[list(selection)].mean(axis=1)
        winner = sel_mean.idxmax()
        selected_scores.append(sel_mean.loc[winner])
        heldout_scores.append(wide.loc[winner, heldout].mean())

    selected = np.array(selected_scores)
    heldout = np.array(heldout_scores)
    return {
        "n_configs": len(wide),
        "n_splits": len(selected),
        "mean_selection_score": selected.mean(),
        "mean_heldout_score": heldout.mean(),
        "mean_optimism": (selected - heldout).mean(),
        "optimism_sd": (selected - heldout).std(ddof=1),
        "reported_max_all_folds": wide.mean(axis=1).max(),
        "median_config_all_folds": wide.mean(axis=1).median(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_root", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--n-select",
        type=int,
        default=3,
        help="folds used for selection; the rest are held out (default 3 of 5)",
    )
    parser.add_argument(
        "--methods",
        default="no_pretrain_encoder_task,pretrain_ccsa_then_encoder_task",
        help="comma-separated methods to analyse (default: S1 and the TL champion)",
    )
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows = []

    for threshold in (5, 6, 7):
        path = (
            args.sweep_root
            / f"final_runs_threshold_{threshold}"
            / "all_transfer_learning_results.csv"
        )
        if not path.exists():
            raise SystemExit(f"missing {path}")
        data = pd.read_csv(path)

        for method in methods:
            for affinity in ("pk", "pic50"):
                subset = data[
                    (data["method"] == method) & (data["affinity_type"] == affinity)
                ]
                if subset.empty:
                    continue
                wide = _pivot(subset)
                if wide.empty:
                    continue
                stats = split_half_bias(wide, args.n_select)
                stats.update(
                    threshold=float(threshold), method=method, affinity_type=affinity
                )
                rows.append(stats)

    out = pd.DataFrame(rows)[
        [
            "method",
            "affinity_type",
            "threshold",
            "n_configs",
            "n_splits",
            "reported_max_all_folds",
            "median_config_all_folds",
            "mean_selection_score",
            "mean_heldout_score",
            "mean_optimism",
            "optimism_sd",
        ]
    ].round(4)

    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print()
    print(
        f"mean optimism of selecting on {args.n_select} folds and reading off the "
        f"other {5 - args.n_select}: {out['mean_optimism'].mean():.4f} MCC"
    )
    print(
        "reported_max_all_folds is the published-style number: the best of "
        f"{out['n_configs'].iloc[0]} configurations, scored on the same folds."
    )

    if args.out:
        out.to_csv(args.out, index=False)
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
