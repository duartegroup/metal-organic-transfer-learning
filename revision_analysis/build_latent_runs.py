#!/usr/bin/env python3
"""Emit embedding_visualization.py invocations for the non-circular latent-space analysis.

One run per (affinity, threshold): the S4C champion configuration for that endpoint,
seed 42, nested checkpoints. Mirrors the published output layout
results/latent_space_analysis/results/threshold_{N}/{affinity}/.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from build_rerun_matrix import SCALERS, parse_scaling_config

S4C = "pretrain_ccsa_then_encoder_task"
PHASE = "pretrain_ccsa_encoder_task"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=Path, default=Path("champion_configs.csv"))
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = pd.read_csv(args.configs)
    s4c = cfg[cfg.method == S4C]
    lines = []
    for (aff, thr), grp in s4c.groupby(["affinity", "threshold"]):
        if len(grp) != 1:
            raise SystemExit(f"{aff} t{thr}: expected one S4C champion, got {len(grp)}")
        r = grp.iloc[0]
        dec = parse_scaling_config(str(r.scaling_config))
        name = (f"{r.affinity}_t{int(r.threshold)}_{r.inhibitor}_"
                f"{r.biological_target}_s{args.seed}")
        model = (f"{args.runs_root}/{name}/{r.inhibitor}/{r.biological_target}/"
                 f"{r.scaler}/{r.affinity}/{r.scaling_config}/"
                 f"threshold_{int(r.threshold)}/seed_{args.seed}")
        flags = [
            "--model_path", model,
            "--output_dir", args.output_dir,
            "--phase", PHASE,
            "--n_folds", "5",
            "--use_simple_structure",
            "--with_predictions",
            "--inhibitor_descriptor", str(r.inhibitor),
            "--biological_descriptor", "ESM-C",
            "--modality", str(r.biological_target),
            "--affinity_type", str(r.affinity),
            "--data_scaler", SCALERS[str(r.scaler)],
            "--threshold", f"{float(r.threshold)}",
            "--seed", str(args.seed),
        ]
        if not dec["cross_domain"]:
            flags.append("--separate_domains")
        if dec["source_smote"]:
            flags.append("--source_smote")
        if dec["target_smote"]:
            flags.append("--target_smote")
        lines.append(" ".join(flags))
        print(f"{aff} t{int(thr)}: {r.inhibitor}/{r.biological_target}/{r.scaler}/{r.scaling_config}")

    args.out.write_text("\n".join(lines) + "\n")
    print(f"\n{len(lines)} runs -> {args.out}")

if __name__ == "__main__":
    main()
