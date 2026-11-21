"""Quantify the assay heterogeneity of the pIC50 and pKi datasets.

Supports the manuscript's claim that pIC50 measurements are the more heterogeneous of the
two endpoints, which Reviewer 1 asked to see substantiated.

Usage:
    python assay_heterogeneity.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2] / "metal-organic-transfer-learning"
METADATA = REPO / "data" / "excel" / "metadata.xlsx"
OUT = Path(__file__).resolve().parents[1] / "results"


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for endpoint, g in df.groupby("activity_type"):
        # spread of values for one protein measured by more than one source
        spreads = []
        for _, pg in g.groupby("uniprot_id"):
            if pg["doi"].nunique() > 1:
                per_source = pg.groupby("doi")["activity_value"].mean()
                if len(per_source) > 1:
                    spreads.append(per_source.std(ddof=1))
        multi = g.groupby("uniprot_id")["doi"].nunique()
        rows.append(
            {
                "endpoint": endpoint,
                "n_measurements": len(g),
                "n_compounds": g["molecule_id"].nunique(),
                "n_targets": g["uniprot_id"].nunique(),
                "n_assay_formats": g["binding_assay"].nunique(),
                "n_publications": g["doi"].nunique(),
                "measurements_per_publication": round(len(g) / g["doi"].nunique(), 1),
                "targets_with_multiple_sources": int((multi > 1).sum()),
                "pct_targets_multi_source": round(100 * (multi > 1).mean(), 1),
                "median_between_source_sd": (
                    round(float(np.median(spreads)), 3) if spreads else None
                ),
                "mean_between_source_sd": (
                    round(float(np.mean(spreads)), 3) if spreads else None
                ),
                "n_targets_in_spread": len(spreads),
                "value_range": f"{g['activity_value'].min():.2f}-{g['activity_value'].max():.2f}",
                "value_sd": round(g["activity_value"].std(ddof=1), 3),
            }
        )
    return pd.DataFrame(rows)


def assay_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    tab = (
        df.groupby(["activity_type", "binding_assay"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["activity_type", "n"], ascending=[True, False])
    )
    return tab


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(METADATA, sheet_name="measurements")

    summary = summarise(df)
    summary.to_csv(OUT / "assay_heterogeneity_summary.csv", index=False)
    print("=== Endpoint comparison ===")
    print(summary.T.to_string())

    breakdown = assay_breakdown(df)
    breakdown.to_csv(OUT / "assay_format_breakdown.csv", index=False)
    print()
    print("=== Assay formats ===")
    print(breakdown.to_string(index=False))


if __name__ == "__main__":
    main()
