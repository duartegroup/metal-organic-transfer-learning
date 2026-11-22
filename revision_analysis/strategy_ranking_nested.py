#!/usr/bin/env python3
"""Does the strategy-level ranking survive nested evaluation?

Why this exists. The one headline claim that survived the round-2 audit is that the
three-stage strategy S4C (`pretrain_ccsa_then_encoder_task`) is the best training
strategy "across the full configuration search (p < 0.001)". That p-value was computed
under the published protocol, in which the stopping epoch was chosen on the held-out
fold. Guide section 0.2l shows the resulting optimism is *ordered by number of training
stages* -- +0.158 and +0.138 for the two CCSA-then-encoder methods against +0.029 to
+0.061 for the single- and two-stage ones. The bias therefore ranks the strategies in the
same order as the claim, so the claim cannot be defended on published-protocol data.

It can be tested on nested data. `champ_nested_report_allmethods.csv` holds all nine
methods on each of the twelve champion cells (2 endpoints x 3 thresholds x 2 descriptor/
modality settings), three seeds each, and `champ_control_report_allmethods.csv` holds the
same runs under the published protocol. That is a complete paired block design: twelve
blocks, nine methods, both protocols.

This script runs, for each protocol and each metric:
  * a Friedman test across methods over the twelve cells,
  * mean ranks (rank 1 = best within a cell),
  * post-hoc Wilcoxon signed-rank tests of S4C against every other method, Holm-corrected
    over that family.

Two methods (`no_pretrain_ccsa`, `pretrain_ccsa`) are domain adaptation only, with no
classification head, so `mcc_from_probs` is undefined for them. The probability metric --
the paper's convention since 2026-11-12 -- therefore covers seven methods; `val/mcc`
covers all nine and is what section 0.2l reports. Both are run; neither is cherry-picked.

Usage:
    python strategy_ranking_nested.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import stats

S4C = "pretrain_ccsa_then_encoder_task"
CELL = ("affinity", "threshold", "inhibitor", "biological_target", "scaler",
        "scaling_config")
METRICS = {"mcc_from_probs": "mcc_probs_mean", "val/mcc": "mcc_mean"}
PROTOCOLS = {"nested": "champ_nested_report_allmethods.csv",
             "published": "champ_control_report_allmethods.csv"}


def load(path: Path, col: str) -> tuple[list[str], list[tuple], np.ndarray]:
    """Return (methods, cells, matrix[cell, method]) keeping only complete methods."""
    raw, cells, methods = {}, [], []
    for r in csv.DictReader(open(path)):
        cell = tuple(r[k] for k in CELL)
        if cell not in cells:
            cells.append(cell)
        if r["method"] not in methods:
            methods.append(r["method"])
        v = r[col].strip()
        raw[(cell, r["method"])] = float(v) if v else np.nan
    mat = np.array([[raw[(c, m)] for m in methods] for c in cells])
    keep = [i for i in range(len(methods)) if not np.isnan(mat[:, i]).any()]
    return [methods[i] for i in keep], cells, mat[:, keep]


def holm(p: dict) -> dict:
    out, run = {}, 0.0
    for i, (k, v) in enumerate(sorted(p.items(), key=lambda kv: kv[1])):
        run = max(run, min(1.0, (len(p) - i) * v))
        out[k] = run
    return out


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=here / "results")
    ap.add_argument("--out", type=Path,
                    default=here / "results/strategy_ranking_nested.csv")
    args = ap.parse_args()

    rows = []
    for metric, col in METRICS.items():
        print("=" * 94)
        print(f"METRIC: {metric}")
        print("=" * 94)
        store = {}
        for proto, fname in PROTOCOLS.items():
            methods, cells, mat = load(args.results / fname, col)
            # rank within each cell, 1 = best (highest MCC)
            ranks = np.array([stats.rankdata(-row) for row in mat])
            chi, p_fried = stats.friedmanchisquare(*mat.T)
            store[proto] = (methods, mat, ranks, p_fried)

            print(f"\n{proto.upper()} -- {len(cells)} cells x {len(methods)} methods")
            print(f"  Friedman chi2 = {chi:.2f}, p = {p_fried:.2e}")
            order = np.argsort(ranks.mean(axis=0))
            print(f"  {'method':40s}{'mean rank':>11}{'mean MCC':>11}{'wins':>7}")
            best = mat.argmax(axis=1)
            for i in order:
                w = int((best == i).sum())
                print(f"  {methods[i]:40s}{ranks[:, i].mean():>11.2f}"
                      f"{mat[:, i].mean():>11.4f}{w:>7d}")

            # post-hoc: S4C against every other method
            j = methods.index(S4C)
            praw, delta = {}, {}
            for i, m in enumerate(methods):
                if i == j:
                    continue
                d = mat[:, j] - mat[:, i]
                delta[m] = d.mean()
                praw[m] = float(stats.wilcoxon(mat[:, j], mat[:, i]).pvalue)
            padj = holm(praw)
            print(f"\n  S4C vs each, Wilcoxon signed-rank (n={len(cells)}), "
                  f"Holm over {len(praw)}:")
            print(f"  {'comparator':40s}{'dMCC':>9}{'p_raw':>10}{'p_holm':>10}   ")
            for m in sorted(praw, key=lambda k: praw[k]):
                flag = "SIG" if padj[m] <= 0.05 else "ns"
                print(f"  {m:40s}{delta[m]:>+9.4f}{praw[m]:>10.4f}"
                      f"{padj[m]:>10.4f}   {flag}")
                rows.append(dict(metric=metric, protocol=proto, comparator=m,
                                 n_cells=len(cells),
                                 delta_mcc=round(delta[m], 4),
                                 p_raw=round(praw[m], 4),
                                 p_holm=round(padj[m], 4),
                                 significant=padj[m] <= 0.05,
                                 friedman_p=f"{p_fried:.3e}",
                                 s4c_mean_rank=round(ranks[:, j].mean(), 2),
                                 n_methods=len(methods)))

        # how far did S4C's standing move between protocols?
        print("\n  PROTOCOL SHIFT")
        for proto in PROTOCOLS:
            methods, mat, ranks, pf = store[proto]
            j = methods.index(S4C)
            print(f"    {proto:10s} S4C mean rank {ranks[:, j].mean():.2f} of "
                  f"{len(methods)},  Friedman p = {pf:.2e},  "
                  f"S4C best in {int((mat.argmax(axis=1) == j).sum())}/{mat.shape[0]} cells")
        print()

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
