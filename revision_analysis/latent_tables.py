#!/usr/bin/env python3
"""Replacement content for SI Tables S44 and S45, from the nested latent-space runs.

The published tables report, per fold, MMD and centroid cosine similarity for the four
source/target class pairs. Those numbers came from an analysis that read the same
validation fold used to select the checkpoint (REVISION_GUIDE, latent-space provenance).
These are the same quantities computed on the true held-out fold.

Column mapping, SI name -> key in separability_metrics.json["source_target_combined"]:

    SA vs TA  ->  cross_domain_active_{mmd,cosine}
    SI vs TI  ->  cross_domain_inactive_{mmd,cosine}
    SA vs TI  ->  cross_domain_active_vs_inactive_{mmd,cosine}
    SI vs TA  ->  cross_domain_inactive_vs_active_{mmd,cosine}

Usage:
    python latent_tables.py [--results ../results/latent_nested]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PAIRS = [("SA vs TA", "cross_domain_active"),
         ("SI vs TI", "cross_domain_inactive"),
         ("SA vs TI", "cross_domain_active_vs_inactive"),
         ("SI vs TA", "cross_domain_inactive_vs_active")]

TABLES = [("S44", "pic50", "pIC50"), ("S45", "pk", "pK$_i$")]


def fmt(v):
    return "—" if v is None else f"{v:.4f}"


def render(path: Path, label: str, pretty: str) -> str:
    block = json.loads(path.read_text())["source_target_combined"]
    folds = sorted(block, key=lambda k: int(k.split("_")[1]))
    out = [f"**Table {label}** — {pretty} dataset, threshold 6, nested evaluation "
           f"(held-out fold).", "",
           "| Fold | " + " | ".join(n for n, _ in PAIRS) + " |",
           "|---|---|---|---|---|"]
    for metric, suffix in (("MMD", "mmd"), ("Cosine similarity", "cosine")):
        out.append(f"| *{metric}* | | | | |")
        for f in folds:
            row = block[f]
            vals = [fmt(row.get(f"{key}_{suffix}")) for _, key in PAIRS]
            out.append(f"| {f.split('_')[1]} | " + " | ".join(vals) + " |")
    missing = [f for f in folds
               if block[f].get("cross_domain_active_mmd") is None]
    if missing:
        out += ["", f"> Folds with no value: {', '.join(missing)} — the held-out fold "
                    "was effectively single-class, so the metric is undefined."]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results/latent_nested")
    args = ap.parse_args()

    for label, key, pretty in TABLES:
        p = args.results / "threshold_6" / key / "separability_metrics.json"
        if not p.exists():
            print(f"missing: {p}")
            continue
        print(render(p, label, pretty))
        print()


if __name__ == "__main__":
    main()
