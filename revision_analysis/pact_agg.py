"""Aggregate the regenerated PACT panel: 39 molecules x 3 thresholds x 3 seeds.

Each threshold uses its own S4C champion configuration, so probabilities are comparable
*within* a threshold (across molecules) but NOT across thresholds.

Published values for the intact caged complex come from SI Table S77 row 1:
theta5 0.78, theta6 0.35, theta7 0.44.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path.home() / "matthijs/runs/pact_out"
MOL = Path.home() / "matthijs/metal-organic-transfer-learning/results/extrapolation_test/molecules"
SEEDS = [42, 43, 44]
PUBLISHED = {5: 0.7823, 6: 0.3496, 7: 0.4405}  # caged_PACT, from the published JSONs

rows = []
for d in sorted(ROOT.iterdir()):
    if not d.is_dir() or d.name == "geom":
        continue
    meta_p = MOL / d.name / "meta.json"
    series = json.loads(meta_p.read_text()).get("series") if meta_p.exists() else None
    for thr in (5, 6, 7):
        for seed in SEEDS:
            f = d / f"t{thr}_s{seed}.json"
            if not f.is_file():
                continue
            p = json.loads(f.read_text())["prediction"]
            rows.append({"molecule": d.name, "series": series, "threshold": thr,
                         "seed": seed, "p": p["ensemble_probability"],
                         "n_models": p["n_models_used"]})

df = pd.DataFrame(rows)
print(f"predictions: {len(df)}  molecules: {df.molecule.nunique()}")
bad = df[df.n_models != 5]
print(f"runs not using all 5 fold models: {len(bad)}")

agg = (df.groupby(["molecule", "series", "threshold"])["p"]
         .agg(mean="mean", sd="std", lo="min", hi="max").reset_index())
agg.to_csv(Path.home() / "matthijs/runs/pact_summary.csv", index=False)
df.to_csv(Path.home() / "matthijs/runs/pact_per_seed.csv", index=False)

print("\n" + "=" * 72)
print("caged_PACT -- the headline compound (Table S77 row 1)")
print("=" * 72)
print(f"{'theta':>5}  {'published':>9}  {'nested mean':>11}  {'sd':>6}  "
      f"{'per-seed':>24}  {'class':>9}  {'was':>9}")
for thr in (5, 6, 7):
    s = df[(df.molecule == "caged_PACT") & (df.threshold == thr)].sort_values("seed")
    if s.empty:
        continue
    m, sd = s.p.mean(), s.p.std()
    per = " ".join(f"{v:.3f}" for v in s.p)
    pub = PUBLISHED[thr]
    print(f"{thr:>5}  {pub:>9.4f}  {m:>11.4f}  {sd:>6.4f}  {per:>24}  "
          f"{'ACTIVE' if m > 0.5 else 'inactive':>9}  "
          f"{'ACTIVE' if pub > 0.5 else 'inactive':>9}")

print("\n" + "=" * 72)
print("Classification changes vs published, caged_PACT only")
print("=" * 72)
for thr in (5, 6, 7):
    s = df[(df.molecule == "caged_PACT") & (df.threshold == thr)]
    if s.empty:
        continue
    now, was = s.p.mean() > 0.5, PUBLISHED[thr] > 0.5
    unanimous = (s.p > 0.5).all() or (s.p <= 0.5).all()
    flag = "FLIPPED" if now != was else "same"
    print(f"  theta{thr}: {flag:8s}  seeds unanimous: {unanimous}")

print("\n" + "=" * 72)
print("Per-series mean probability, by threshold")
print("=" * 72)
piv = agg.pivot_table(index="series", columns="threshold", values="mean").round(3)
print(piv.to_string())

print("\n" + "=" * 72)
print("Seed stability: molecules whose seeds disagree on the class")
print("=" * 72)
unstable = []
for (mol, thr), s in df.groupby(["molecule", "threshold"]):
    if 0 < (s.p > 0.5).sum() < len(s):
        unstable.append((mol, thr, " ".join(f"{v:.3f}" for v in s.sort_values('seed').p)))
if unstable:
    for mol, thr, per in unstable:
        print(f"  {mol:32s} theta{thr}  {per}")
    print(f"  ({len(unstable)} of {df.molecule.nunique() * 3} molecule-threshold cells)")
else:
    print("  none -- all three seeds agree on the class everywhere")
