"""Noise-ceiling reproduction and endpoint-specific recomputation.

Stage 1 reproduces the published Tables S6-S8 using the pooled sigma = 0.69 so that the
reimplementation can be checked against the manuscript before anything is changed.
Stage 2 recomputes the ceiling with endpoint-specific sigma.

The ceiling procedure follows Crusius et al.: experimental error is modelled by adding
Gaussian noise to the measured target values, both the measured and the perturbed values
are classified at the activity threshold, and the agreement between the two label sets is
the best score any predictor could achieve given that measurement error. Only target
values and fold assignments are required -- no descriptors.

Usage:
    python noise_ceiling.py --stage reproduce
    python noise_ceiling.py --stage recompute --sigma-pic50 0.68 --sigma-pki 0.54
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

REPO = Path(__file__).resolve().parents[2] / "metal-organic-transfer-learning"
METADATA = REPO / "data" / "excel" / "metadata.xlsx"
OUT = Path(__file__).resolve().parents[1] / "results"

THRESHOLDS = (5.0, 6.0, 7.0)
POOLED_SIGMA = 0.69

# Published Tables S6-S8, MCC only, indexed [threshold][endpoint][fold 1-5].
PUBLISHED_MCC = {
    5.0: {
        "pIC50": [0.645, 0.359, 0.690, 0.765, 0.448],
        "pKi": [0.572, 0.522, 0.324, 0.654, 0.587],
    },
    6.0: {
        "pIC50": [0.677, 0.623, 0.799, 0.898, 0.557],
        "pKi": [0.817, 0.704, 0.617, 0.700, 0.551],
    },
    7.0: {
        "pIC50": [0.712, 0.717, 0.854, 0.570, 0.588],
        "pKi": [0.943, 0.614, 0.788, 0.702, 0.633],
    },
}


def load_measurements() -> pd.DataFrame:
    df = pd.read_excel(METADATA, sheet_name="measurements")
    # metadata folds are 0-indexed; the SI tables number them 1-5.
    df["fold_si"] = df["fold"].astype(int) + 1
    return df


def ceiling(values: np.ndarray, threshold: float, sigma: float, n_boot: int,
            rng: np.random.Generator) -> dict[str, float]:
    """Mean attainable score over n_boot noise realisations."""
    true_labels = (values >= threshold).astype(int)
    mcc = np.empty(n_boot)
    acc = np.empty(n_boot)
    f1 = np.empty(n_boot)
    for i in range(n_boot):
        noisy = values + rng.normal(0.0, sigma, size=values.shape)
        noisy_labels = (noisy >= threshold).astype(int)
        mcc[i] = matthews_corrcoef(true_labels, noisy_labels)
        acc[i] = accuracy_score(true_labels, noisy_labels)
        f1[i] = f1_score(true_labels, noisy_labels, zero_division=0)
    return {
        "mcc": float(mcc.mean()),
        "mcc_sd": float(mcc.std(ddof=1)),
        "accuracy": float(acc.mean()),
        "f1": float(f1.mean()),
    }


def run(sigmas: dict[str, float], n_boot: int, seed: int) -> pd.DataFrame:
    df = load_measurements()
    rng = np.random.default_rng(seed)
    rows = []
    for endpoint, sigma in sigmas.items():
        sub = df[df["activity_type"] == endpoint]
        for threshold in THRESHOLDS:
            for fold in sorted(sub["fold_si"].unique()):
                values = sub.loc[sub["fold_si"] == fold, "activity_value"].to_numpy()
                stats = ceiling(values, threshold, sigma, n_boot, rng)
                rows.append(
                    {
                        "endpoint": endpoint,
                        "threshold": threshold,
                        "fold": int(fold),
                        "n": len(values),
                        "sigma": sigma,
                        **stats,
                    }
                )
    return pd.DataFrame(rows)


def compare_to_published(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in res.iterrows():
        key = "pIC50" if r["endpoint"] == "pIC50" else "pKi"
        published = PUBLISHED_MCC[r["threshold"]][key][int(r["fold"]) - 1]
        rows.append(
            {
                "endpoint": r["endpoint"],
                "threshold": r["threshold"],
                "fold": int(r["fold"]),
                "published": published,
                "reproduced": round(r["mcc"], 3),
                "diff": round(r["mcc"] - published, 3),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["reproduce", "recompute"], required=True)
    p.add_argument("--sigma-pic50", type=float, default=0.68)
    p.add_argument("--sigma-pki", type=float, default=0.54)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.stage == "reproduce":
        sigmas = {"pIC50": POOLED_SIGMA, "pKi": POOLED_SIGMA}
    else:
        sigmas = {"pIC50": args.sigma_pic50, "pKi": args.sigma_pki}

    res = run(sigmas, args.n_boot, args.seed)
    res.to_csv(OUT / f"noise_ceiling_{args.stage}.csv", index=False)

    if args.stage == "reproduce":
        cmp = compare_to_published(res)
        cmp.to_csv(OUT / "noise_ceiling_reproduction_check.csv", index=False)
        print(cmp.to_string(index=False))
        print()
        print(f"max |diff| = {cmp['diff'].abs().max():.3f}")
        print(f"mean |diff| = {cmp['diff'].abs().mean():.3f}")
    else:
        summary = (
            res.groupby(["endpoint", "threshold"])
            .agg(mcc=("mcc", "mean"), sigma=("sigma", "first"))
            .reset_index()
        )
        print(summary.to_string(index=False))
        summary.to_json(OUT / "noise_ceiling_recomputed_summary.json", orient="records", indent=2)


if __name__ == "__main__":
    main()
