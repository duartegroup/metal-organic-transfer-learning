#!/usr/bin/env python3
"""How much does the decision-boundary correction move MCC?

Reads every per-fold summary CSV produced so far by the nested sweep and compares
val/mcc (fed raw logits, so thresholded at 0.5 on the logit whenever every logit
lands inside [0, 1]) against val/mcc_from_probs (fed sigmoid probabilities, so
always thresholded at p = 0.5).
"""
import glob
import sys

import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "/u/fd/sjoh5775/matthijs/runs/matrix"
files = glob.glob(f"{root}/**/*summary*.csv", recursive=True)
if not files:
    raise SystemExit(f"no summary CSVs under {root}")

frames = []
for path in files:
    frame = pd.read_csv(path)
    frame["source"] = path
    frames.append(frame)
data = pd.concat(frames, ignore_index=True)

DIAG = "val/logit_batches_in_unit_interval"
MCC = "val/mcc"
MCC_P = "val/mcc_from_probs"

missing = [c for c in (DIAG, MCC, MCC_P) if c not in data.columns]
if missing:
    raise SystemExit(f"columns absent: {missing}; have {list(data.columns)}")

print(f"runs: {len(files)}   fold evaluations: {len(data)}")
affected = data[DIAG] == 1
print(f"affected folds (diagnostic == 1): {int(affected.sum())} "
      f"({100 * affected.mean():.1f}%)")
print()

for label, subset in (
    ("all folds", data),
    ("affected only", data[affected]),
    ("unaffected only", data[~affected]),
):
    if subset.empty:
        continue
    delta = subset[MCC_P] - subset[MCC]
    print(
        f"{label:16s} n={len(subset):4d}  "
        f"mcc {subset[MCC].mean():+.4f} -> from_probs {subset[MCC_P].mean():+.4f}  "
        f"delta {delta.mean():+.4f}  (sd {delta.std():.4f}, "
        f"min {delta.min():+.4f}, max {delta.max():+.4f})"
    )

print()
disagree = (data[MCC] != data[MCC_P]).sum()
print(f"folds where the two differ at all: {disagree} of {len(data)}")
unaffected_disagree = int(((~affected) & (data[MCC] != data[MCC_P])).sum())
print(
    f"  of which the diagnostic said were unaffected: {unaffected_disagree} "
    "(should be 0 -- any other value means the diagnostic is not measuring what "
    "the metric is doing)"
)
