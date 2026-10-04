#!/usr/bin/env python
"""Rebuild molecules/index.csv from the per-molecule meta.json files.

The index lists every molecule with its series, parent, charge and composition,
and whether predictions exist for all thresholds. The ORCA and inference job
scripts read it, so rerun this after adding or changing a molecule directory.

    python scripts/build_index.py
"""
import csv
import json
from pathlib import Path

MOLECULES = Path(__file__).resolve().parent.parent / "molecules"
THRESHOLDS = (5, 6, 7)
COLUMNS = ["name", "series", "parent", "motif", "charge", "formula",
           "n_atoms", "n_heavy", "has_predictions", "source", "notes"]


def main() -> None:
    rows = []
    for meta_path in sorted(MOLECULES.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        pred_dir = meta_path.parent / "predictions"
        has_pred = all((pred_dir / f"th{th}.json").exists() for th in THRESHOLDS)
        row = {col: meta.get(col) for col in COLUMNS}
        row["has_predictions"] = int(has_pred)
        rows.append({k: "" if v is None else v for k, v in row.items()})

    with (MOLECULES / "index.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} molecules to {MOLECULES / 'index.csv'}")


if __name__ == "__main__":
    main()
