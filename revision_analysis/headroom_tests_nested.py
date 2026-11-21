#!/usr/bin/env python3
"""Paired TL-vs-S1 comparisons and headroom, recomputed under nested evaluation.

This is `headroom_tests.py` run against `runs/table1/` instead of the repository's
shipped `top_models/` outputs. Same design, same tests, same ceiling file -- only the
model outputs change.

Why it had to be redone: the published §1.6 statistics (0.54 -> 0.64 on pKi, the -0.03
paired difference on pIC50, "5 of 15 fold-threshold pairs") were computed from the
per-fold metrics of the two Table 1 configurations under the published, non-nested
protocol. Those same two configurations have now been retrained under nested evaluation
across three seeds (job 795473), so every one of those statistics moves.

Two differences from the published analysis, both forced by the new data:

1. **Three seeds instead of one.** The published run had a single seed, giving 15
   fold-threshold pairs per endpoint. Here each fold-threshold cell has three seeds. The
   headline analysis averages MCC over seeds within a cell, preserving the published
   15-pair design and treating the seed as the nuisance parameter it is. The pooled
   45-pair test over all (fold, threshold, seed) units is printed underneath as a
   robustness check; it is not the number to quote, because its pairs are not independent.
2. **`mcc_from_probs` instead of `val/mcc`.** The decision of 2026-11-12 (REVISION_GUIDE
   §0.3a). `headroom_tests.py` reads `val/mcc`.

Ceiling file defaults to `noise_ceiling_unfiltered_basis.csv` (sigma_pIC50 = 0.86), which
is the one that reproduces the ceilings quoted in §2.2 and in the published
`headroom_vs_ceiling.csv`. The ceilings are properties of the measurements and do NOT
change under nested evaluation -- only the model MCCs beneath them do.

Usage:
    python headroom_tests_nested.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ARMS = {"no_pretrain_encoder_task": "S1", "pretrain_ccsa_then_encoder_task": "TL"}
METRIC = "val/mcc_from_probs"
THRS = (5.0, 6.0, 7.0)
ENDPOINTS = ("pIC50", "pKi")

# Published Table S41 per-cell means, for the side-by-side.
PUBLISHED_CELL = {
    ("pIC50", "S1"): {5.0: 0.460, 6.0: 0.606, 7.0: 0.581},
    ("pIC50", "TL"): {5.0: 0.429, 6.0: 0.622, 7.0: 0.517},
    ("pKi", "S1"): {5.0: 0.475, 6.0: 0.467, 7.0: 0.664},
    ("pKi", "TL"): {5.0: 0.608, 6.0: 0.583, 7.0: 0.715},
}
# Published pooled statistics from tl_vs_s1_tests.csv, for the side-by-side.
PUBLISHED_POOLED = {
    "pIC50": dict(mean_S1=0.549, mean_TL=0.523, diff=-0.026, ci=(-0.089, 0.030),
                  w=0.4236, w_holm=0.4236, tost=0.0167, better=5),
    "pKi": dict(mean_S1=0.536, mean_TL=0.635, diff=0.100, ci=(0.021, 0.198),
                w=0.0209, w_holm=0.0418, tost=0.4990, better=7),
}


def load_folds(root: Path) -> dict:
    """{(endpoint, threshold, fold, seed, arm): mcc}"""
    out = {}
    for f in glob.glob(str(root / "**/*summary.csv"), recursive=True):
        m = re.search(r"/(pic50|pk)_t(\d)_.*_s(\d+)/", f)
        if not m:
            continue
        endpoint = "pIC50" if m.group(1) == "pic50" else "pKi"
        thr, seed = float(m.group(2)), int(m.group(3))
        for r in csv.DictReader(open(f)):
            arm = ARMS.get(r["method"])
            if arm is None or not r.get(METRIC):
                continue
            try:
                fold = int(r["fold"])
            except ValueError:
                continue
            out[(endpoint, thr, fold, seed, arm)] = float(r[METRIC])
    if not out:
        raise SystemExit(f"no usable rows under {root}")
    return out


def load_ceiling(path: Path) -> dict:
    return {(r["endpoint"], float(r["threshold"]), int(r["fold"])): float(r["mcc"])
            for r in csv.DictReader(open(path))}


def bootstrap_ci(d: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def tost(d: np.ndarray, margin: float) -> float:
    """Two one-sided t-tests; returns the larger (governing) p-value."""
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    return float(max(stats.t.sf((d.mean() + margin) / se, df=n - 1),
                     stats.t.cdf((d.mean() - margin) / se, df=n - 1)))


def wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        return float("nan")


def holm(pvals: dict) -> dict:
    out, running = {}, 0.0
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (len(items) - i) * p))
        out[k] = running
    return out


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=here / "results/table1")
    ap.add_argument("--ceiling", type=Path,
                    default=here / "results/noise_ceiling_unfiltered_basis.csv")
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=here / "results")
    args = ap.parse_args()

    folds = load_folds(args.runs)
    ceil = load_ceiling(args.ceiling)
    seeds = sorted({k[3] for k in folds})
    print(f"loaded {len(folds)} (endpoint, threshold, fold, seed, arm) cells; "
          f"seeds {seeds}\n")

    # --- seed-averaged per (endpoint, threshold, fold, arm) ------------------------
    avg = defaultdict(list)
    for (e, t, f, s, a), v in folds.items():
        avg[(e, t, f, a)].append(v)
    avg = {k: float(np.mean(v)) for k, v in avg.items()}

    per_fold_rows = []
    for (e, t, f, a), v in sorted(avg.items()):
        per_fold_rows.append({"endpoint": e, "threshold": t, "fold": f, "model": a,
                              "mcc": round(v, 4),
                              "n_seeds": len([1 for s in seeds
                                              if (e, t, f, s, a) in folds])})
    with open(args.out / "table1_fold_metrics_nested.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_fold_rows[0]))
        w.writeheader()
        w.writerows(per_fold_rows)

    # --- paired tests --------------------------------------------------------------
    def pairs(e, thr=None, per_seed=False):
        if per_seed:
            keys = [(t, f, s) for t in THRS for f in range(1, 6) for s in seeds
                    if (e, t, f, s, "S1") in folds and (e, t, f, s, "TL") in folds]
            if thr is not None:
                keys = [k for k in keys if k[0] == thr]
            return (np.array([folds[(e, t, f, s, "S1")] for t, f, s in keys]),
                    np.array([folds[(e, t, f, s, "TL")] for t, f, s in keys]))
        keys = [(t, f) for t in THRS for f in range(1, 6)
                if (e, t, f, "S1") in avg and (e, t, f, "TL") in avg]
        if thr is not None:
            keys = [k for k in keys if k[0] == thr]
        return (np.array([avg[(e, t, f, "S1")] for t, f in keys]),
                np.array([avg[(e, t, f, "TL")] for t, f in keys]))

    rows, raw_p = [], {}
    for per_seed in (False, True):
        for e in ENDPOINTS:
            groups = [("pooled", None)] + [(f"theta={t:g}", t) for t in THRS]
            for name, thr in groups:
                s1, tl = pairs(e, thr, per_seed)
                if len(s1) < 3:
                    continue
                d = tl - s1
                lo, hi = bootstrap_ci(d, args.n_boot, args.seed)
                wp = wilcoxon(tl, s1)
                rows.append({
                    "basis": "per_seed" if per_seed else "seed_averaged",
                    "endpoint": e, "group": name, "n_pairs": len(d),
                    "mean_S1": round(s1.mean(), 3), "mean_TL": round(tl.mean(), 3),
                    "mean_diff": round(d.mean(), 3),
                    "median_diff": round(float(np.median(d)), 3),
                    "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                    "wilcoxon_p": round(wp, 4) if wp == wp else None,
                    "tost_p": round(tost(d, args.margin), 4),
                    "n_folds_TL_better": int((d > 0).sum()),
                })
                if name == "pooled" and not per_seed:
                    raw_p[e] = wp
    adj = holm(raw_p)
    for r in rows:
        r["wilcoxon_p_holm"] = (round(adj[r["endpoint"]], 4)
                                if r["group"] == "pooled"
                                and r["basis"] == "seed_averaged" else None)

    with open(args.out / "tl_vs_s1_tests_nested.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def show(basis, title):
        print("=" * 100)
        print(title)
        print("=" * 100)
        hdr = ("endpoint", "group", "n", "mean_S1", "mean_TL", "diff", "ci_lo", "ci_hi",
               "wilcox", "holm", "TOST", "TL>S1")
        print(f"{hdr[0]:9}{hdr[1]:10}{hdr[2]:>4}{hdr[3]:>9}{hdr[4]:>9}{hdr[5]:>8}"
              f"{hdr[6]:>8}{hdr[7]:>8}{hdr[8]:>8}{hdr[9]:>8}{hdr[10]:>8}{hdr[11]:>7}")
        for r in rows:
            if r["basis"] != basis:
                continue
            hol = f"{r['wilcoxon_p_holm']:.4f}" if r["wilcoxon_p_holm"] is not None else "—"
            print(f"{r['endpoint']:9}{r['group']:10}{r['n_pairs']:>4}{r['mean_S1']:>9.3f}"
                  f"{r['mean_TL']:>9.3f}{r['mean_diff']:>+8.3f}{r['ci_lo']:>+8.3f}"
                  f"{r['ci_hi']:>+8.3f}{r['wilcoxon_p']:>8.4f}{hol:>8}"
                  f"{r['tost_p']:>8.4f}{r['n_folds_TL_better']:>4}/{r['n_pairs']}")
        print()

    show("seed_averaged", "TL vs S1 — nested, seed-averaged (THIS IS WHAT TO QUOTE)")
    show("per_seed", "TL vs S1 — nested, per-seed pairs (robustness check only; "
                     "pairs are not independent)")

    print("=" * 100)
    print("Published (non-nested) pooled statistics, for the side-by-side")
    print("=" * 100)
    for e in ENDPOINTS:
        p = PUBLISHED_POOLED[e]
        now = next(r for r in rows if r["basis"] == "seed_averaged"
                   and r["endpoint"] == e and r["group"] == "pooled")
        print(f"  {e}")
        print(f"    published : S1 {p['mean_S1']:.3f} -> TL {p['mean_TL']:.3f}  "
              f"diff {p['diff']:+.3f}  95% CI [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]  "
              f"Wilcoxon p={p['w']:.4f} (Holm {p['w_holm']:.4f})  TOST p={p['tost']:.4f}  "
              f"TL better in {p['better']}/15")
        print(f"    nested    : S1 {now['mean_S1']:.3f} -> TL {now['mean_TL']:.3f}  "
              f"diff {now['mean_diff']:+.3f}  "
              f"95% CI [{now['ci_lo']:+.3f}, {now['ci_hi']:+.3f}]  "
              f"Wilcoxon p={now['wilcoxon_p']:.4f} "
              f"(Holm {now['wilcoxon_p_holm']:.4f})  TOST p={now['tost_p']:.4f}  "
              f"TL better in {now['n_folds_TL_better']}/{now['n_pairs']}")
    print()

    # --- headroom ------------------------------------------------------------------
    head_rows = []
    for e in ENDPOINTS:
        for t in THRS:
            for a in ("S1", "TL"):
                gaps, mccs, ceils = [], [], []
                for f in range(1, 6):
                    if (e, t, f, a) not in avg or (e, t, f) not in ceil:
                        continue
                    mccs.append(avg[(e, t, f, a)])
                    ceils.append(ceil[(e, t, f)])
                    gaps.append(ceil[(e, t, f)] - avg[(e, t, f, a)])
                if not gaps:
                    continue
                g = np.array(gaps)
                head_rows.append({
                    "endpoint": e, "threshold": t, "model": a,
                    "mcc": round(float(np.mean(mccs)), 3),
                    "ceiling": round(float(np.mean(ceils)), 3),
                    "gap": round(float(g.mean()), 3),
                    "gap_sd": round(float(g.std(ddof=1)), 3),
                    "folds_at_or_above": int((g <= 0).sum()),
                    "published_mcc": PUBLISHED_CELL[(e, a)][t],
                    "published_gap": round(
                        float(np.mean(ceils)) - PUBLISHED_CELL[(e, a)][t], 3),
                })
    with open(args.out / "headroom_vs_ceiling_nested.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(head_rows[0]))
        w.writeheader()
        w.writerows(head_rows)

    print("=" * 100)
    print(f"Headroom against the ceiling ({args.ceiling.name}) — nested, seed-averaged")
    print("  the ceiling is a property of the measurements and does not move;")
    print("  only the model MCC beneath it changes.")
    print("=" * 100)
    print(f"{'endpoint':9}{'theta':>6}{'arm':>5}{'ceiling':>9}{'MCC':>8}{'gap':>8}"
          f"{'gap sd':>8}{'folds>=':>8}   {'published MCC':>13}{'published gap':>14}")
    for r in head_rows:
        print(f"{r['endpoint']:9}{r['threshold']:>6.0f}{r['model']:>5}"
              f"{r['ceiling']:>9.3f}{r['mcc']:>8.3f}{r['gap']:>8.3f}{r['gap_sd']:>8.3f}"
              f"{r['folds_at_or_above']:>8}   {r['published_mcc']:>13.3f}"
              f"{r['published_gap']:>14.3f}")
    print()

    # --- the sentence §1.6 actually makes -------------------------------------------
    print("=" * 100)
    print("The §1.6 headroom claim, recomputed")
    print("=" * 100)
    for e in ENDPOINTS:
        for t in THRS:
            s1 = next(r for r in head_rows
                      if r["endpoint"] == e and r["threshold"] == t and r["model"] == "S1")
            tl = next(r for r in head_rows
                      if r["endpoint"] == e and r["threshold"] == t and r["model"] == "TL")
            rec = (tl["mcc"] - s1["mcc"]) / s1["gap"] if s1["gap"] > 0 else float("nan")
            print(f"  {e:6} theta={t:.0f}  ceiling {s1['ceiling']:.3f}  "
                  f"S1 {s1['mcc']:.3f}  gap {s1['gap']:.3f}  "
                  f"TL {tl['mcc']:.3f}  TL recovers {rec:6.1%} of the gap")
    print()
    print(f"wrote {args.out / 'tl_vs_s1_tests_nested.csv'}")
    print(f"wrote {args.out / 'headroom_vs_ceiling_nested.csv'}")
    print(f"wrote {args.out / 'table1_fold_metrics_nested.csv'}")


if __name__ == "__main__":
    main()
