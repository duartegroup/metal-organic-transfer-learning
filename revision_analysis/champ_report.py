#!/usr/bin/env python3
"""Aggregate champion retraining runs and compare against published MCCs.

Keys every run off its RESULTS PATH, not the output-directory name. Two champion
configurations can share an output_dir name (they differ only in a flag that the
name does not encode, e.g. target_smote) while writing to distinct subtrees; keying
off the name silently collapsed them.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd

TAIL = re.compile(r"threshold_(?P<thr>[\d.]+)/seed_(?P<seed>\d+)$")

def parse(summary: Path, root: Path) -> dict | None:
    rel = summary.relative_to(root).parts
    # <name>/<inhibitor>/<modality>/<scaler>/<affinity>/<scaling_config>/threshold_t/seed_s/<file>
    if len(rel) < 9:
        return None
    _, inhib, modality, scaler, affinity, scaling, thr, seed = rel[:8]
    if not thr.startswith("threshold_") or not seed.startswith("seed_"):
        return None
    return dict(inhibitor=inhib, biological_target=modality, scaler=scaler,
                affinity=affinity, scaling_config=scaling,
                threshold=int(float(thr.split("_", 1)[1])),
                seed=int(seed.split("_", 1)[1]))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--configs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="nested")
    args = ap.parse_args()

    rows, skipped = [], []
    for summary in sorted(args.root.rglob("*summary.csv")):
        meta = parse(summary, args.root)
        if meta is None:
            skipped.append(str(summary)); continue
        df = pd.read_csv(summary)
        df = df[df["fold"] != "single_run"]
        g = df.groupby("method").agg(
            mcc=("val/mcc", "mean"), mcc_probs=("val/mcc_from_probs", "mean"),
            auroc=("val/auroc", "mean"), n_folds=("val/mcc", "size"),
        ).reset_index()
        for k, v in meta.items():
            g[k] = v
        rows.append(g)
    for s in skipped:
        print(f"skip (unparsed path) {s}")

    runs = pd.concat(rows, ignore_index=True)
    cfgkeys = ["affinity", "threshold", "inhibitor", "biological_target",
               "scaler", "scaling_config"]
    keys = cfgkeys + ["method"]
    agg = runs.groupby(keys).agg(
        mcc_mean=("mcc", "mean"), mcc_sd=("mcc", "std"),
        mcc_probs_mean=("mcc_probs", "mean"), mcc_probs_sd=("mcc_probs", "std"),
        auroc_mean=("auroc", "mean"), n_seeds=("mcc", "size"),
    ).reset_index()

    cfg = pd.read_csv(args.configs)[cfgkeys + ["method", "published_mcc"]]
    merged = cfg.merge(agg, on=keys, how="left").sort_values(
        ["affinity", "threshold", "biological_target"])
    merged["delta"] = merged["mcc_mean"] - merged["published_mcc"]
    merged["delta_probs"] = merged["mcc_probs_mean"] - merged["published_mcc"]
    merged.to_csv(args.out, index=False)
    runs.to_csv(args.out.with_name(args.out.stem + "_perseed.csv"), index=False)
    agg.to_csv(args.out.with_name(args.out.stem + "_allmethods.csv"), index=False)

    pd.set_option("display.width", 250, "display.max_columns", 50)
    show = keys + ["published_mcc", "mcc_mean", "mcc_sd", "delta",
                   "mcc_probs_mean", "delta_probs", "n_seeds"]
    print(f"\n=== {args.label}: {merged['mcc_mean'].notna().sum()}/{len(merged)} champions ===")
    print(merged[show].to_string(index=False))
    done = merged.dropna(subset=["mcc_mean"])
    if len(done):
        print(f"\nmean delta (published convention): {done['delta'].mean():+.4f}")
        print(f"mean delta (corrected convention): {done['delta_probs'].mean():+.4f}")
        bad = done[done.n_seeds != 3]
        if len(bad):
            print(f"\nWARNING: {len(bad)} champion(s) not on 3 seeds")

if __name__ == "__main__":
    main()
