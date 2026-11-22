#!/usr/bin/env python3
"""Replacement content for SI Tables S77-S80, the motif-ablation panel.

The published tables have one row per compound and three column groups
(pIC50 = 5, 6, 7), each giving Activity and Probability, where the probability is the
mean of five per-fold outputs from a single seed. These are the same quantities from
the regenerated runs, with the across-seed standard deviation added, since the panel
turned out to be seed-sensitive (REVISION_GUIDE, PACT section).

Two deliberate differences from the published tables, both to be stated in the caption:

1. **Grouped by series rather than split into four arbitrary parts.** The published
   split into "part I-IV" is pagination; grouping by series (the parent compounds, the
   caged-complex modifications, the free-ligand modifications, the fragments, the
   photoproducts, the spectator variants) matches how the main text discusses them.
2. **Activity is left blank where the three seeds disagree.** Reporting a class for a
   compound whose call depends on the random seed would assert a precision the data
   does not support; those rows are marked "indet.".

The published tables identify compounds by drawn structure, not by name, so the text
layer of the PDF carries no compound identifiers. A row-by-row published-vs-new
comparison therefore cannot be made from the PDF alone and is not attempted here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

THRS = (5, 6, 7)
SERIES_ORDER = ["reaction", "ablation_caged", "ablation_free", "fragment",
                "photoproduct", "spectator"]
SERIES_TITLE = {
    "reaction": "Parent species",
    "ablation_caged": "Modifications of STF-31 within the caged complex",
    "ablation_free": "Modifications of the free STF-31 ligand",
    "fragment": "Fragments of the uncaged organic ligand",
    "photoproduct": "Photoproducts",
    "spectator": "Spectator-ligand variants",
}


def cell(group: pd.DataFrame) -> tuple[str, str, bool]:
    """Activity label, 'mean ± sd', and whether the call is seed-fragile.

    The class is assigned from the pooled prediction (all three seeds), which is the
    better estimator and keeps every row usable. A dagger marks rows where the
    across-seed spread crosses 0.5, i.e. where the class would change with the seed --
    the same information as a bare "indeterminate" label, without discarding the
    estimate.
    """
    mean, sd = group.p.mean(), group.p.std()
    fragile = (mean - sd) <= 0.5 <= (mean + sd)
    act = "Active" if mean > 0.5 else "Inactive"
    return (act + ("†" if fragile else ""), f"{mean:.2f} ± {sd:.2f}", fragile)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results")
    args = ap.parse_args()

    d = pd.read_csv(args.results / "pact_per_seed.csv")

    print("Replacement for SI Tables S77-S80 — motif-ablation panel")
    print("Predicted probabilities are ensemble values (mean of five per-fold outputs),")
    print("averaged over three random seeds and reported as mean ± s.d. across seeds.")
    print("The active class is assigned above 0.5 from the pooled three-seed mean.")
    print("A dagger marks calls whose across-seed spread crosses 0.5, i.e. where the")
    print("class would change with the random seed.")
    print()

    n_indet = 0
    for series in SERIES_ORDER:
        sub = d[d.series == series]
        if sub.empty:
            continue
        print(f"\n**{SERIES_TITLE[series]}**\n")
        print("| Compound | " + " | ".join(
            f"pIC50 = {t} activity | pIC50 = {t} probability" for t in THRS) + " |")
        print("|---" * (1 + 2 * len(THRS)) + "|")
        order = (sub[sub.threshold == 6].groupby("molecule").p.mean()
                 .sort_values(ascending=False).index)
        for mol in order:
            cells = []
            for t in THRS:
                g = sub[(sub.molecule == mol) & (sub.threshold == t)]
                if g.empty:
                    cells += ["—", "—"]
                    continue
                act, val, fragile = cell(g)
                n_indet += fragile
                cells += [act, val]
            print(f"| {mol} | " + " | ".join(cells) + " |")

    total = d.molecule.nunique() * len(THRS)
    print(f"\n\n{n_indet} of {total} compound-threshold cells are seed-fragile "
          f"({n_indet / total:.0%}): the across-seed spread crosses 0.5, so the class "
          f"would change with the seed.")
    print("That figure belongs in the caption. It is the honest summary of how much "
          "weight the individual classifications carry, and it is why the main text "
          "argues from differences between compounds rather than from the classes.")


if __name__ == "__main__":
    main()
