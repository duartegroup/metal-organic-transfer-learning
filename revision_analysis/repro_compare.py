#!/usr/bin/env python3
"""Published vs reproduced, for one configuration, all nine methods.

The control run (job 785086) used the published protocol on the published folds for
the configuration that is the published S4C champion for pKi at threshold 6:
SOAP / molecule_protein / None_scaler / separate_domain_no_source_smote_target_smote.

Comparing its output against the *champion* figures would be wrong, because each
method has its own champion configuration. The right comparison is this
configuration's own row in the published sweep.
"""
import pandas as pd

SWEEP = ("/u/fd/sjoh5775/matthijs/extracted/sweep_results/results/"
         "transfer_learning_results/final_runs_threshold_6/"
         "all_transfer_learning_results.csv")

CONFIG = {
    "inhibitor": "SOAP",
    "biological_target": "molecule_protein",
    "scaler": "None_scaler",
    "affinity_type": "pk",
    "scaling_config": "separate_domain_no_source_smote_target_smote",
}

# Read off the control run's summary table.
REPRODUCED = {
    "pretrain_ccsa": 0.083,
    "pretrain_ccsa_then_task": 0.342,
    "pretrain_ccsa_then_encoder_task": 0.570,
    "pretrain_encoder_task": 0.310,
    "pretrain_task": 0.122,
    "no_pretrain_ccsa": 0.104,
    "no_pretrain_ccsa_then_encoder_task": 0.593,
    "no_pretrain_ccsa_then_task": 0.064,
    "no_pretrain_encoder_task": 0.348,
}

data = pd.read_csv(SWEEP)
mask = pd.Series(True, index=data.index)
for column, value in CONFIG.items():
    mask &= data[column].astype(str) == value
subset = data[mask]

print(f"published rows for this configuration: {len(subset)} "
      f"(expect 9 methods x 5 folds = 45, plus pretrain_src)")
print()

published = subset.groupby("method")["val/mcc"].agg(["mean", "std", "count"])

rows = []
for method, repro in REPRODUCED.items():
    if method not in published.index:
        rows.append((method, None, repro, None))
        continue
    pub = published.loc[method, "mean"]
    rows.append((method, pub, repro, repro - pub))

out = pd.DataFrame(rows, columns=["method", "published", "reproduced", "difference"])
out = out.sort_values("published", ascending=False)
pd.set_option("display.width", 160)
print(out.round(4).to_string(index=False))
print()

valid = out.dropna(subset=["difference"])
print(f"mean absolute difference: {valid.difference.abs().mean():.4f}")
print(f"max absolute difference:  {valid.difference.abs().max():.4f} "
      f"({valid.loc[valid.difference.abs().idxmax(), 'method']})")
print(f"correlation across methods: {valid.published.corr(valid.reproduced):.4f}")
