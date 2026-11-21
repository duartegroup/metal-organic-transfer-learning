#!/usr/bin/env python3
"""Everything the PACT case study needs at threshold 6.0, and nothing else.

Scope decided 2026-11-12: the section reports theta 6 only (REVISION_GUIDE, PACT
section). This produces the per-compound probabilities with across-seed spread, the
motif-ablation deltas, and the three significance tests recomputed from the regenerated
predictions.

The significance tests follow the manuscript's method -- one-sided Welch
(`ttest_ind(..., equal_var=False, alternative="less")`), as in `motif_ablation.ipynb` --
but NOT its exact groupings: the notebook used hand-typed subsets (13 spectator values,
5 free-ligand values, 4 fragment values) that cannot be reconstructed from the deposited
panel. Groups here are the full series recorded in each molecule's meta.json, which is
the defensible choice and is stated as such.

Deltas are taken against the parent the manuscript uses:
    ablation_caged, spectator, photoproduct, reaction  ->  caged_PACT
    ablation_free, fragment                            ->  stf_31
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import ttest_ind

THR = 6
CAGED_PARENT = {"ablation_caged", "spectator", "photoproduct", "reaction"}


def load(results: Path) -> pd.DataFrame:
    d = pd.read_csv(results / "pact_per_seed.csv")
    return d[d.threshold == THR].copy()


def add_deltas(d: pd.DataFrame) -> pd.DataFrame:
    ref = d[d.molecule.isin(["caged_PACT", "stf_31"])].set_index(["molecule", "seed"])["p"]
    out = []
    for _, r in d.iterrows():
        if r.molecule in ("caged_PACT", "stf_31"):
            continue
        parent = "caged_PACT" if r.series in CAGED_PARENT else "stf_31"
        try:
            base = ref.loc[(parent, r.seed)]
        except KeyError:
            continue
        out.append({**r.to_dict(), "parent": parent, "delta": r.p - base})
    return pd.DataFrame(out)


def welch(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Is group a's reduction greater (more negative) than group b's?"""
    res = ttest_ind(a, b, equal_var=False, alternative="less")
    return float(res.statistic), float(res.pvalue)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results")
    args = ap.parse_args()

    d = load(args.results)
    dd = add_deltas(d)

    print("=" * 70)
    print("PER-COMPOUND PROBABILITIES AT THRESHOLD 6.0 (mean +/- s.d. over 3 seeds)")
    print("=" * 70)
    per = (d.groupby(["series", "molecule"])["p"]
             .agg(mean="mean", sd="std").reset_index()
             .sort_values(["series", "mean"], ascending=[True, False]))
    for series, grp in per.groupby("series"):
        print(f"\n  {series}")
        for _, r in grp.iterrows():
            call = "active" if r["mean"] > 0.5 else "inactive"
            print(f"    {r.molecule:32s} {r['mean']:.3f} +/- {r['sd']:.3f}  {call}")

    print("\n" + "=" * 70)
    print("MOTIF-ABLATION DELTAS AT THRESHOLD 6.0 (mean +/- s.d. over 3 seeds)")
    print("=" * 70)
    md = (dd.groupby(["series", "molecule"])["delta"]
            .agg(mean="mean", sd="std").reset_index().sort_values("mean"))
    for series, grp in md.groupby("series"):
        print(f"\n  {series} (vs {'caged_PACT' if series in CAGED_PARENT else 'stf_31'})")
        for _, r in grp.sort_values("mean").iterrows():
            print(f"    {r.molecule:32s} {r['mean']:+.3f} +/- {r['sd']:.3f}")

    print("\n" + "=" * 70)
    print("SIGNIFICANCE, recomputed (one-sided Welch on per-compound mean deltas)")
    print("=" * 70)
    means = dd.groupby(["series", "molecule"])["delta"].mean().reset_index()
    g = {s: v.delta for s, v in means.groupby("series")}
    small = pd.concat([g.get("ablation_caged", pd.Series(dtype=float)),
                       g.get("ablation_free", pd.Series(dtype=float))])

    tests = [
        ("spectator modifications vs STF-31-ligand modifications",
         g.get("spectator"), g.get("ablation_caged")),
        ("free-ligand modifications vs caged STF-31 modifications",
         g.get("ablation_free"), g.get("ablation_caged")),
        ("fragments vs all small modifications",
         g.get("fragment"), small),
    ]
    for label, a, b in tests:
        if a is None or b is None or len(a) < 2 or len(b) < 2:
            print(f"  {label}: not enough data")
            continue
        t, p = welch(a, b)
        print(f"  {label}")
        print(f"      mean delta {a.mean():+.3f} (n={len(a)}) vs {b.mean():+.3f} "
              f"(n={len(b)});  t={t:.3f}  p={p:.4f}"
              f"{'  SIGNIFICANT' if p < 0.05 else '  not significant'}")

    print("\n  Per-seed robustness of the fragment test:")
    for seed, s in dd[dd.series.isin(["fragment", "ablation_caged", "ablation_free"])].groupby("seed"):
        frag = s[s.series == "fragment"].delta
        sm = s[s.series != "fragment"].delta
        t, p = welch(frag, sm)
        print(f"    seed {seed}: mean {frag.mean():+.3f} vs {sm.mean():+.3f}  p={p:.4f}")


if __name__ == "__main__":
    main()
