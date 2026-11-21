#!/usr/bin/env python3
"""Argument lines for retraining the published champion configurations.

The adopted plan (PLAN.md Part 0): retrain the 12 published champions under both
protocols, 3 seeds, and replace the paper's numbers with the nested ones.

A `full_pipeline` invocation produces all nine methods, so the 12 champions collapse
to the distinct (configuration, endpoint, threshold) combinations — two per cell.
`--run_types` is therefore NOT emitted: the full pipeline is what gives the complete
ablation table from the same jobs.

Usage:
    python build_champion_runs.py champion_configs.csv --seeds 42 43 44 \
        --output-root /path/to/runs/champ_nested --out champ_nested.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from build_rerun_matrix import MODALITY_FLAGS, SCALERS, parse_scaling_config

CONFIG_KEYS = [
    "inhibitor",
    "biological_target",
    "scaler",
    "scaling_config",
    "cross_domain_scaling",
    "source_smote",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    configs = pd.read_csv(args.configs)

    # One run per distinct configuration per cell, whichever method it championed.
    unique = configs.drop_duplicates(subset=CONFIG_KEYS + ["affinity", "threshold"])
    print(f"{len(configs)} champion rows -> {len(unique)} distinct (config, cell) runs")

    lines, manifest = [], []
    for _, row in unique.iterrows():
        decoded = parse_scaling_config(str(row["scaling_config"]))
        assert bool(row["cross_domain_scaling"]) == decoded["cross_domain"], row
        assert bool(row["source_smote"]) == decoded["source_smote"], row

        for seed in args.seeds:
            name = (
                f"{row['affinity']}_t{int(row['threshold'])}_"
                f"{row['inhibitor']}_{row['biological_target']}_s{seed}"
            )
            flags = [
                "--affinity_type", str(row["affinity"]),
                "--threshold", f"{float(row['threshold'])}",
                "--inhibitor_descriptor", str(row["inhibitor"]),
                *MODALITY_FLAGS[str(row["biological_target"])],
                "--data_scaler", SCALERS[str(row["scaler"])],
                "--seed", str(seed),
                "--n_folds", "5",
                "--output_dir", f"{args.output_root}/{name}",
            ]
            if decoded["cross_domain"]:
                flags.append("--cross_domain_scaling")
            if decoded["source_smote"]:
                flags.append("--source_smote")
            if decoded["target_smote"]:
                flags.append("--target_smote")

            lines.append(" ".join(flags))
            manifest.append({"index": len(lines) - 1, "name": name,
                             "championed": row["method"],
                             "published_mcc": row["published_mcc"]})

    args.out.write_text("\n".join(lines) + "\n")
    pd.DataFrame(manifest).to_csv(args.out.with_suffix(".manifest.csv"), index=False)
    print(f"{len(lines)} runs -> {args.out}")
    print(f"manifest -> {args.out.with_suffix('.manifest.csv')}")


if __name__ == "__main__":
    main()
