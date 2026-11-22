#!/usr/bin/env python3
"""The three motif-ablation significance tests at all three thresholds.

The manuscript's conclusion that "the model's predictions are largely driven by the
STF-31 ligand" is cross-threshold by construction: it rests on spectator modifications
NOT being significantly greater than pharmacophore modifications at theta 5 and 7, while
they are at theta 6. Keeping that conclusion therefore needs all three thresholds, so
this reruns the tests everywhere on the regenerated predictions.

It also reports, for each threshold, the size of the deltas being tested and whether the
three seeds agree on their sign -- because a test can fail to reach significance either
because the two groups genuinely differ little, or because the model produces nothing
distinguishable from noise at that threshold. Those are not the same finding, and only
the first supports the published argument.

Method follows motif_ablation.ipynb: one-sided Welch, ttest_ind(..., equal_var=False,
alternative="less"), on per-compound deltas from the manuscript's parents.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_ind

CAGED_PARENT = {"ablation_caged", "spectator", "photoproduct", "reaction"}


def deltas(d: pd.DataFrame, thr: int) -> pd.DataFrame:
    s = d[d.threshold == thr]
    ref = s[s.molecule.isin(["caged_PACT", "stf_31"])].set_index(["molecule", "seed"])["p"]
    rows = []
    for _, r in s.iterrows():
        if r.molecule in ("caged_PACT", "stf_31"):
            continue
        parent = "caged_PACT" if r.series in CAGED_PARENT else "stf_31"
        try:
            base = ref.loc[(parent, r.seed)]
        except KeyError:
            continue
        rows.append({"molecule": r.molecule, "series": r.series, "seed": r.seed,
                     "delta": r.p - base})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results")
    args = ap.parse_args()
    d = pd.read_csv(args.results / "pact_per_seed.csv")

    print("THE CROSS-THRESHOLD ARGUMENT, RECOMPUTED")
    print("=" * 74)
    print("Test 1: are spectator-ligand modifications' reductions significantly GREATER")
    print("        than STF-31-ligand modifications'?  (published: no at 5 and 7, yes at 6)")
    print()
    print(f"{'theta':>5} {'spectator':>11} {'STF-31 mods':>12} {'p':>8}  {'verdict':>17}"
          f"  {'|delta| size':>12} {'sign-stable':>12}")
    for thr in (5, 6, 7):
        dd = deltas(d, thr)
        means = dd.groupby(["series", "molecule"])["delta"].mean().reset_index()
        spec = means[means.series == "spectator"].delta
        phar = means[means.series == "ablation_caged"].delta
        p = float(ttest_ind(spec, phar, equal_var=False, alternative="less").pvalue)
        # how big are the deltas at all, and do seeds agree on their sign?
        g = dd.groupby("molecule")["delta"]
        big = [m for m, v in g if v.abs().mean() > 0.01]
        stable = sum(1 for m, v in g if m in big and ((v > 0).all() or (v < 0).all()))
        size = means.delta.abs().mean()
        verdict = "significant" if p < 0.05 else "NOT significant"
        print(f"{thr:>5} {spec.mean():>11.3f} {phar.mean():>12.3f} {p:>8.4f}"
              f"  {verdict:>17}  {size:>12.3f} {f'{stable}/{len(big)}':>12}")

    print()
    print("Test 3 (fragments vs all small modifications), for reference:")
    for thr in (5, 6, 7):
        dd = deltas(d, thr)
        means = dd.groupby(["series", "molecule"])["delta"].mean().reset_index()
        frag = means[means.series == "fragment"].delta
        small = means[means.series.isin(["ablation_caged", "ablation_free"])].delta
        p = float(ttest_ind(frag, small, equal_var=False, alternative="less").pvalue)
        print(f"  theta{thr}: {frag.mean():+.3f} vs {small.mean():+.3f}  p={p:.4f}"
              f"{'  significant' if p < 0.05 else '  NOT significant'}")


if __name__ == "__main__":
    main()
