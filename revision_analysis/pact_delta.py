"""Are the motif-ablation DIFFERENCES seed-stable, even though absolutes are not?

The PACT section's main analysis is not absolute probabilities but deltas: a modified
species minus its parent, at the same threshold. If a seed shifts every compound
together, the shift cancels in the delta and the ablation analysis survives even where
the per-compound calls do not.

Parents, following the manuscript:
    caged_M*            -> caged_PACT     (modifications within the caged complex)
    free_M*, frag_*     -> stf_31         (modifications/fragments of the free ligand)
    deriv_*, photo_*    -> caged_PACT     (spectator variants and photoproducts)
"""
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
d = pd.read_csv(RES / "pact_per_seed.csv")


def parent(name):
    if name.startswith(("free_M", "frag_")):
        return "stf_31"
    if name in ("caged_PACT", "stf_31"):
        return None
    return "caged_PACT"


d["parent"] = d.molecule.map(parent)
ref = (d[d.molecule.isin(["caged_PACT", "stf_31"])]
       .set_index(["molecule", "threshold", "seed"])["p"])

rows = []
for _, r in d.dropna(subset=["parent"]).iterrows():
    try:
        base = ref.loc[(r.parent, r.threshold, r.seed)]
    except KeyError:
        continue
    rows.append({"molecule": r.molecule, "threshold": r.threshold, "seed": r.seed,
                 "p": r.p, "delta": r.p - base})
dd = pd.DataFrame(rows)

print("Across-seed spread: absolute probability vs delta-from-parent")
print("=" * 64)
print(f"{'theta':>5}  {'mean sd(p)':>11}  {'mean sd(delta)':>15}  {'reduction':>10}")
for thr in (5, 6, 7):
    s = dd[dd.threshold == thr]
    sp = s.groupby("molecule")["p"].std().mean()
    sd_ = s.groupby("molecule")["delta"].std().mean()
    print(f"{thr:>5}  {sp:>11.4f}  {sd_:>15.4f}  {sp / sd_:>9.1f}x")

print()
print("Do the three seeds agree on the SIGN of each delta?")
print("=" * 64)
for thr in (5, 6, 7):
    s = dd[dd.threshold == thr]
    g = s.groupby("molecule")["delta"]
    agree = sum(1 for _, v in g if (v > 0).all() or (v < 0).all())
    n = g.ngroups
    # only count deltas that are non-trivial in size
    big = [m for m, v in g if v.abs().mean() > 0.01]
    agree_big = sum(1 for m, v in g if m in big and ((v > 0).all() or (v < 0).all()))
    print(f"  theta{thr}: {agree}/{n} overall;  "
          f"{agree_big}/{len(big)} among deltas with |mean| > 0.01")

print()
print("Largest deltas at each threshold (mean over seeds, with across-seed sd)")
print("=" * 64)
for thr in (5, 6, 7):
    s = (dd[dd.threshold == thr].groupby("molecule")["delta"]
         .agg(mean="mean", sd="std").sort_values("mean"))
    print(f"\n  theta{thr}:")
    for name, row in s.head(4).iterrows():
        print(f"    {name:34s} {row['mean']:+.3f} +/- {row['sd']:.3f}")
