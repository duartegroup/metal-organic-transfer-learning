#!/usr/bin/env python3
"""Turn the recovered champion configurations into one argument line per run.

The sweep launcher was in the purged scratch folder, so the per-run settings are
recovered from the columns of all_transfer_learning_results.csv. This writes a
plain text file with one `main.py` argument string per line, which the sbatch
array indexes by SLURM_ARRAY_TASK_ID.

Usage:
    python build_rerun_matrix.py champion_configs.csv --seeds 42 43 44 --out runs.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MODALITY_FLAGS = {
    "molecule": ["--use_mol"],
    "protein": ["--use_protein"],
    "pocket": ["--use_pocket"],
    "molecule_protein": ["--use_mol", "--use_protein"],
    "molecule_pocket": ["--use_mol", "--use_pocket"],
    "protein_pocket": ["--use_protein", "--use_pocket"],
    "molecule_protein_pocket": ["--use_mol", "--use_protein", "--use_pocket"],
}

SCALERS = {
    "minmax_scaler": "minmax",
    "standard_scaler": "standard",
    "None_scaler": "none",
    "none_scaler": "none",
}


def parse_scaling_config(text: str) -> dict:
    """Decode e.g. 'cross_domain_no_source_smote_target_smote'.

    Order matters: 'no_source_smote' must be tested before 'source_smote', or the
    negated form matches the positive one.
    """
    cross = text.startswith("cross_domain")
    source = "no_source_smote" not in text and "source_smote" in text
    target = "no_target_smote" not in text and "target_smote" in text
    return {"cross_domain": cross, "source_smote": source, "target_smote": target}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--expand-over",
        default=None,
        metavar="METHOD",
        help="treat the input as a bare configuration table and expand it over both "
        "endpoints and all three thresholds for this method, for a full nested "
        "re-selection rather than a champions-only matrix",
    )
    parser.add_argument(
        "--only-method",
        default=None,
        help="emit only runs for this method, so the two kinds can be submitted as "
        "separate arrays with memory suited to each (target-only runs need ~24 GB, "
        "anything touching the source domain needs ~96 GB)",
    )
    parser.add_argument(
        "--output-root",
        default="/u/fd/sjoh5775/matthijs/runs/matrix",
        help="each run gets its own subdirectory under here",
    )
    args = parser.parse_args()

    configs = pd.read_csv(args.configs)

    # A bare configuration table -- the 432 rows of the published sweep, with no
    # method/affinity/threshold columns -- is expanded over the cells. This is how the
    # full nested re-selection is generated, as opposed to the champions-only matrix.
    if args.expand_over:
        method = args.expand_over
        cells = [(a, t) for t in (5.0, 6.0, 7.0) for a in ("pk", "pic50")]
        configs = pd.concat(
            [
                configs.assign(method=method, affinity=a, threshold=t,
                               published_mcc=float("nan"))
                for a, t in cells
            ],
            ignore_index=True,
        )
    lines, manifest = [], []

    if args.only_method:
        configs = configs[configs["method"] == args.only_method]
        if configs.empty:
            raise SystemExit(f"no configurations with method={args.only_method}")

    for _, row in configs.iterrows():
        decoded = parse_scaling_config(str(row["scaling_config"]))

        # The two sources must agree; the CSV carries both and a mismatch would mean
        # the column decoding is wrong rather than merely redundant.
        assert bool(row["cross_domain_scaling"]) == decoded["cross_domain"], row
        assert bool(row["source_smote"]) == decoded["source_smote"], row

        modality = MODALITY_FLAGS[str(row["biological_target"])]
        scaler = SCALERS[str(row["scaler"])]

        for seed in args.seeds:
            name = (
                f"{row['method']}_{row['affinity']}_t{int(row['threshold'])}_s{seed}"
            )
            flags = [
                "--run_types", str(row["method"]),
                "--affinity_type", str(row["affinity"]),
                "--threshold", f"{float(row['threshold'])}",
                "--inhibitor_descriptor", str(row["inhibitor"]),
                *modality,
                "--data_scaler", scaler,
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
            manifest.append(
                {
                    "index": len(lines) - 1,
                    "name": name,
                    "published_mcc": row["published_mcc"],
                }
            )

    args.out.write_text("\n".join(lines) + "\n")
    pd.DataFrame(manifest).to_csv(args.out.with_suffix(".manifest.csv"), index=False)
    print(f"{len(lines)} runs written to {args.out}")
    print(f"manifest: {args.out.with_suffix('.manifest.csv')}")
    print()
    print("first line:")
    print(" ", lines[0])


if __name__ == "__main__":
    main()
