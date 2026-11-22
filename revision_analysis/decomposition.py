#!/usr/bin/env python3
"""The champion decomposition (REVISION_GUIDE §0.2l and §0.2m), under either MCC column.

Two MCC values are logged for every run:

    val/mcc              computed from raw logits. torchmetrics applies a sigmoid only
                         when some value falls outside [0, 1], so a batch whose logits
                         all land inside that range is thresholded at 0.5 on the logit
                         rather than at 0 - a different decision boundary. See §0.3a.
    val/mcc_from_probs   computed from predicted probabilities. Unambiguous.

Which of the two the paper reports is still open (§0.3a), so this script computes the
whole decomposition either way and nothing downstream has to be re-run when it is
settled.

`mcc_from_probs` does not exist for `pretrain_ccsa` and `no_pretrain_ccsa`, which are
representation-only phases that never emit class probabilities. The pooled figure is
therefore restricted to the rows where BOTH metrics are present, so that the two columns
are like for like.

Usage:
    python decomposition.py --metric val_mcc
    python decomposition.py --metric mcc_from_probs
    python decomposition.py --metric both        # default
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

CFG = ["affinity", "threshold", "inhibitor", "biological_target",
       "scaler", "scaling_config"]
S1 = "no_pretrain_encoder_task"
S4C = "pretrain_ccsa_then_encoder_task"
COLUMN = {"val_mcc": "mcc_mean", "mcc_from_probs": "mcc_probs_mean"}


def load(results: Path):
    get = lambda n: pd.read_csv(results / n)
    champions = get("champ_nested_report.csv").merge(
        get("champ_control_report.csv"), on=CFG + ["method"], suffixes=("_n", "_c"))
    allm = get("champ_nested_report_allmethods.csv").merge(
        get("champ_control_report_allmethods.csv"),
        on=CFG + ["method"], suffixes=("_n", "_c"))
    return champions, allm


def report(champions: pd.DataFrame, allm: pd.DataFrame, metric: str) -> None:
    col = COLUMN[metric]
    # Like for like: only rows where both metrics exist, so the pooled bias of the two
    # metrics is computed over the same runs.
    common = allm.dropna(subset=[f"{c}_{s}" for c in COLUMN.values() for s in "nc"])

    print("=" * 66)
    print(f"METRIC: {metric}")
    print(f"  reproduction gap (12 champions): "
          f"{(champions[f'{col}_c'] - champions['published_mcc_c']).mean():+.3f}")
    for method, label in [(S1, "S1"), (S4C, "S4C")]:
        s = common[common.method == method]
        print(f"  selection bias {label:4s}: "
              f"{(s[f'{col}_c'] - s[f'{col}_n']).mean():+.3f}  (n={len(s)})")
    print(f"  selection bias pooled over {common.method.nunique()} methods "
          f"(n={len(common)}): "
          f"{(common[f'{col}_c'] - common[f'{col}_n']).mean():+.3f}")

    piv = common.pivot_table(index=CFG, columns="method",
                             values=[f"{col}_n", f"{col}_c"])
    published = piv[(f"{col}_c", S4C)] - piv[(f"{col}_c", S1)]
    nested = piv[(f"{col}_n", S4C)] - piv[(f"{col}_n", S1)]
    print(f"  S4C-S1 published protocol: {published.mean():+.3f}  "
          f"wins {int((published > 0).sum())}/{published.notna().sum()}")
    print(f"  S4C-S1 nested            : {nested.mean():+.3f}  "
          f"wins {int((nested > 0).sum())}/{nested.notna().sum()}")

    print("\n  per cell (S1 pub | S4C pub | d pub | S1 nested | S4C nested | d nested):")
    for cfg, r in piv.iterrows():
        affinity, threshold, inhibitor, target = cfg[0], cfg[1], cfg[2], cfg[3]
        name = f"{affinity} t{int(float(threshold))} {inhibitor} / {target}"
        print(f"  | {name:44s} | {r[(f'{col}_c', S1)]:.3f} | {r[(f'{col}_c', S4C)]:.3f} "
              f"| {r[(f'{col}_c', S4C)] - r[(f'{col}_c', S1)]:+.3f} "
              f"| {r[(f'{col}_n', S1)]:.3f} | {r[(f'{col}_n', S4C)]:.3f} "
              f"| {r[(f'{col}_n', S4C)] - r[(f'{col}_n', S1)]:+.3f} |")

    bias = common.assign(bias=common[f"{col}_c"] - common[f"{col}_n"])
    print("\n  bias by method (the 'more stages, more bias' ordering):")
    print(bias.groupby("method")["bias"].mean()
          .sort_values(ascending=False).round(3).to_string())
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", choices=[*COLUMN, "both"], default="both")
    ap.add_argument("--results", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results")
    args = ap.parse_args()

    champions, allm = load(args.results)
    for metric in (COLUMN if args.metric == "both" else [args.metric]):
        report(champions, allm, metric)


if __name__ == "__main__":
    main()
