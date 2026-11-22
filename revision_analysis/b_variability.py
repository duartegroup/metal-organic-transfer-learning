#!/usr/bin/env python3
"""How much does the non-neural baseline B move between hyperparameter-search draws?

Why this exists. `table_s40_full.py` needs a B column with per-fold values, which the
published work did not deposit. We re-ran the two champion B configurations on aleph
under the published protocol and the published search seed (1996) and got only a partial
reproduction of the published per-threshold means (worst |Delta| = 0.076 on MCC). Since
the seed matched, the drift is library drift: `RandomizedSearchCV` in the installed
scikit-learn does not draw the same 20 candidates as the version Matthijs ran, so a
different configuration wins the inner search.

Before writing any B column into the guide we therefore have to know how big B's own
run-to-run spread is. Two further search seeds (7, 2024) were run; this script measures
the spread and -- the decisive part -- re-runs the whole Table S40 comparison once per
seed to check whether the conclusion ("no comparison is significant") depends on which
draw of B you happen to hold.

Usage:
    python b_variability.py
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
NEURAL_COL = {"Accuracy": "val/accuracy_from_probs", "F1": "val/f1_from_probs",
              "MCC": "val/mcc_from_probs"}
B_COL = {"Accuracy": "accuracy", "F1": "f1", "MCC": "mcc"}
METRICS = ["Accuracy", "F1", "MCC"]
THRS = (5, 6, 7)
SEEDS = (1996, 7, 2024)

# Published per-threshold B means (champion_config.json) -- the values behind Table S40.
PUB_B = {"pIC50": {"MCC": (0.2929, 0.4783, 0.5357), "Accuracy": (0.6934, 0.7713, 0.7504),
                   "F1": (0.7021, 0.7444, 0.7071)},
         "pKi": {"MCC": (0.3881, 0.4987, 0.5408), "Accuracy": (0.8240, 0.7253, 0.8157),
                 "F1": (0.8893, 0.6360, 0.6235)}}
# Published Table S40 TL-vs-B relative differences, for the contrast.
PUB_TLvB = {"pIC50": {"Accuracy": "+8.5% (0.017)", "F1": "+8.0% (0.17, ns)",
                      "MCC": "+20.0% (0.030)"},
            "pKi": {"Accuracy": "+9.7% (0.007)", "F1": "+17.3% (0.002)",
                    "MCC": "+33.5% (0.025)"}}


def load_neural(runs: Path) -> dict:
    acc = defaultdict(list)
    for f in glob.glob(str(runs / "**/*summary.csv"), recursive=True):
        m = re.search(r"/(pic50|pk)_t(\d)_.*_s(\d+)/", f)
        if not m:
            continue
        e = "pIC50" if m.group(1) == "pic50" else "pKi"
        t = int(m.group(2))
        for r in csv.DictReader(open(f)):
            arm = ARMS.get(r["method"])
            if not arm:
                continue
            try:
                fold = int(r["fold"])
            except ValueError:
                continue
            for lab, col in NEURAL_COL.items():
                if r.get(col):
                    acc[(e, t, fold, arm, lab)].append(float(r[col]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def load_b_seeds(path: Path) -> dict:
    """{(endpoint, threshold, fold, seed, metric): value}"""
    out = {}
    for r in csv.DictReader(open(path)):
        for lab, col in B_COL.items():
            out[(r["endpoint"], int(float(r["threshold"])), int(r["fold"]),
                 int(r["seed"]), lab)] = float(r[col])
    return out


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
    ap.add_argument("--runs", type=Path, default=here / "results/table1")
    ap.add_argument("--seeds", type=Path,
                    default=here / "results/baseline_b_per_fold_seeds.csv")
    ap.add_argument("--out", type=Path,
                    default=here / "results/table_s40_b_variability.csv")
    args = ap.parse_args()

    nv = load_neural(args.runs)
    bv = load_b_seeds(args.seeds)

    def bvec(e, s, lab):
        return np.array([bv[(e, t, f, s, lab)] for t in THRS for f in range(1, 6)])

    def nvec(e, arm, lab):
        return np.array([nv[(e, t, f, arm, lab)] for t in THRS for f in range(1, 6)])

    # ---- 1. spread of B across search draws -----------------------------------------
    print("=" * 100)
    print("1. B's OWN SPREAD ACROSS HYPERPARAMETER-SEARCH DRAWS")
    print("   per-threshold means; 'published' is champion_config.json, search seed 1996")
    print("=" * 100)
    for e in ("pIC50", "pKi"):
        print(f"\n{e}")
        print(f"{'metric':10}{'thr':>5}{'s1996':>9}{'s7':>9}{'s2024':>9}"
              f"{'mean':>9}{'sd':>8}{'published':>11}   verdict")
        for lab in METRICS:
            for i, t in enumerate(THRS):
                xs = {s: np.mean([bv[(e, t, f, s, lab)] for f in range(1, 6)])
                      for s in SEEDS}
                vals = np.array(list(xs.values()))
                pub = PUB_B[e][lab][i]
                lo, hi = vals.min(), vals.max()
                # tolerance: published is "consistent with" the draws if it lies within
                # the observed range, or within one sd of their mean.
                ok = (lo - 1e-9) <= pub <= (hi + 1e-9) or \
                     abs(pub - vals.mean()) <= vals.std(ddof=1)
                print(f"{lab:10}{t:>5}" + "".join(f"{xs[s]:>9.4f}" for s in SEEDS)
                      + f"{vals.mean():>9.4f}{vals.std(ddof=1):>8.4f}{pub:>11.4f}"
                      + ("   consistent" if ok else "   *** above/below all draws"))

    # ---- 2. pooled B, the number that actually enters Table S40 ----------------------
    print("\n" + "=" * 100)
    print("2. POOLED B (15 folds), the column Table S40 reports")
    print("=" * 100)
    print(f"{'endpoint':10}{'metric':10}" + "".join(f"{'s'+str(s):>9}" for s in SEEDS)
          + f"{'mean':>9}{'sd':>8}{'published':>11}")
    pooled_pub = {}
    for e in ("pIC50", "pKi"):
        for lab in METRICS:
            vals = np.array([bvec(e, s, lab).mean() for s in SEEDS])
            pub = float(np.mean(PUB_B[e][lab]))
            pooled_pub[(e, lab)] = pub
            print(f"{e:10}{lab:10}" + "".join(f"{v:>9.4f}" for v in vals)
                  + f"{vals.mean():>9.4f}{vals.std(ddof=1):>8.4f}{pub:>11.4f}")

    # ---- 3. does the Table S40 conclusion depend on which B you hold? ----------------
    print("\n" + "=" * 100)
    print("3. TABLE S40 TL-vs-B, RECOMPUTED ONCE PER SEARCH DRAW")
    print("   15 fold-pairs, Wilcoxon, Holm over the two planned TL comparisons")
    print("=" * 100)
    rows = []
    for e in ("pIC50", "pKi"):
        print(f"\n{e}")
        print(f"{'metric':10}" + "".join(f"{'TL vs B s'+str(s):>22}" for s in SEEDS)
              + "     published")
        for lab in METRICS:
            tl, s1 = nvec(e, "TL", lab), nvec(e, "S1", lab)
            cells = []
            for s in SEEDS:
                b = bvec(e, s, lab)
                p_raw = {"TLvB": float(stats.wilcoxon(tl, b).pvalue),
                         "TLvS1": float(stats.wilcoxon(tl, s1).pvalue)}
                a = holm(p_raw)
                pct = (tl.mean() - b.mean()) / b.mean() * 100
                cells.append(f"{pct:+.1f}% ({a['TLvB']:.3f}"
                             f"{', ns' if a['TLvB'] > 0.05 else ' SIG'})")
                rows.append(dict(endpoint=e, metric=lab, search_seed=s,
                                 B=round(b.mean(), 4), S1=round(s1.mean(), 4),
                                 TL=round(tl.mean(), 4),
                                 TL_vs_B_pct=round(pct, 1),
                                 TL_vs_B_p_holm=round(a["TLvB"], 4),
                                 significant=a["TLvB"] <= 0.05,
                                 published_TL_vs_B=PUB_TLvB[e][lab]))
            print(f"{lab:10}" + "".join(f"{c:>22}" for c in cells)
                  + f"     {PUB_TLvB[e][lab]}")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    nsig = sum(r["significant"] for r in rows)
    print(f"\nwrote {args.out}")
    print(f"\nSIGNIFICANT TL-vs-B COMPARISONS: {nsig} of {len(rows)} "
          f"(6 metric-endpoint cells x 3 search draws). Published: 5 of 6.")


if __name__ == "__main__":
    main()
