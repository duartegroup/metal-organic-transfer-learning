"""Paired TL-vs-S1 comparisons and headroom against the noise ceiling.

Uses the champion per-fold metrics shipped in the repository
(results/transfer_learning_results/top_models/**/ *_raw_results.json) together with the
ceiling values produced by noise_ceiling.py.

Strategy keys:
    no_pretrain_encoder_task        = S1  (from scratch on metal-organic only)
    pretrain_ccsa_then_encoder_task = S4C (SSL -> CCSA -> end-to-end fine-tuning)

Usage:
    python headroom_tests.py --ceiling results/noise_ceiling_recompute.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2] / "metal-organic-transfer-learning"
TOP = REPO / "results" / "transfer_learning_results" / "top_models"
OUT = Path(__file__).resolve().parents[1] / "results"

S1 = "no_pretrain_encoder_task"
TL = "pretrain_ccsa_then_encoder_task"
ENDPOINT = {"pic50": "pIC50", "pk": "pKi"}


def load_champion_folds() -> pd.DataFrame:
    rows = []
    for path in sorted(TOP.glob("*/threshold_*/*_raw_results.json")):
        endpoint = ENDPOINT[path.parts[-3]]
        threshold = float(path.parts[-2].split("_")[1])
        data = json.loads(path.read_text())
        for strategy, label in ((S1, "S1"), (TL, "TL")):
            if strategy not in data:
                continue
            for fold_key in sorted(data[strategy]):
                mcc = data[strategy][fold_key].get("val/mcc")
                if mcc is None:
                    continue
                rows.append(
                    {
                        "endpoint": endpoint,
                        "threshold": threshold,
                        "fold": int(fold_key),
                        "model": label,
                        "mcc": float(mcc),
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_ci(diffs: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def tost(diffs: np.ndarray, margin: float) -> float:
    """Two one-sided t-tests; returns the larger (governing) p-value."""
    n = len(diffs)
    mean, sd = diffs.mean(), diffs.std(ddof=1)
    se = sd / np.sqrt(n)
    t_lower = (mean + margin) / se
    t_upper = (mean - margin) / se
    p_lower = stats.t.sf(t_lower, df=n - 1)
    p_upper = stats.t.cdf(t_upper, df=n - 1)
    return float(max(p_lower, p_upper))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def paired_tests(df: pd.DataFrame, margin: float, n_boot: int, seed: int) -> pd.DataFrame:
    wide = df.pivot_table(
        index=["endpoint", "threshold", "fold"], columns="model", values="mcc"
    ).reset_index()
    rows = []
    raw_p: dict[tuple, float] = {}

    for endpoint in sorted(wide["endpoint"].unique()):
        sub_all = wide[wide["endpoint"] == endpoint]
        groups = [("pooled", sub_all)] + [
            (f"theta={t:g}", sub_all[sub_all["threshold"] == t])
            for t in sorted(sub_all["threshold"].unique())
        ]
        for name, sub in groups:
            diffs = (sub["TL"] - sub["S1"]).to_numpy()
            if len(diffs) < 3:
                continue
            try:
                w_p = float(stats.wilcoxon(sub["TL"], sub["S1"]).pvalue)
            except ValueError:
                w_p = float("nan")
            lo, hi = bootstrap_ci(diffs, n_boot, seed)
            rows.append(
                {
                    "endpoint": endpoint,
                    "group": name,
                    "n_pairs": len(diffs),
                    "mean_S1": round(sub["S1"].mean(), 3),
                    "mean_TL": round(sub["TL"].mean(), 3),
                    "mean_diff": round(diffs.mean(), 3),
                    "median_diff": round(float(np.median(diffs)), 3),
                    "ci_lo": round(lo, 3),
                    "ci_hi": round(hi, 3),
                    "wilcoxon_p": round(w_p, 4) if w_p == w_p else None,
                    "tost_p": round(tost(diffs, margin), 4),
                    "n_folds_TL_better": int((diffs > 0).sum()),
                }
            )
            if name == "pooled":
                raw_p[(endpoint, name)] = w_p

    res = pd.DataFrame(rows)
    adj = holm({f"{k[0]}": v for k, v in raw_p.items()})
    res["wilcoxon_p_holm"] = res.apply(
        lambda r: round(adj[r["endpoint"]], 4) if r["group"] == "pooled" else None, axis=1
    )
    return res


def headroom(df: pd.DataFrame, ceiling_csv: Path) -> pd.DataFrame:
    ceil = pd.read_csv(ceiling_csv)[["endpoint", "threshold", "fold", "mcc"]]
    ceil = ceil.rename(columns={"mcc": "ceiling"})
    merged = df.merge(ceil, on=["endpoint", "threshold", "fold"], how="left")
    merged["gap"] = merged["ceiling"] - merged["mcc"]
    summary = (
        merged.groupby(["endpoint", "threshold", "model"])
        .agg(
            mcc=("mcc", "mean"),
            ceiling=("ceiling", "mean"),
            gap=("gap", "mean"),
            gap_sd=("gap", "std"),
            folds_at_or_above=("gap", lambda s: int((s <= 0).sum())),
        )
        .reset_index()
        .round(3)
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ceiling", type=Path, default=OUT / "noise_ceiling_recompute.csv")
    p.add_argument("--margin", type=float, default=0.10,
                   help="equivalence margin in MCC units for TOST")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = load_champion_folds()
    df.to_csv(OUT / "champion_fold_metrics.csv", index=False)

    tests = paired_tests(df, args.margin, args.n_boot, args.seed)
    tests.to_csv(OUT / "tl_vs_s1_tests.csv", index=False)
    print("=== TL vs S1 (champion configurations) ===")
    print(tests.to_string(index=False))

    head = headroom(df, args.ceiling)
    head.to_csv(OUT / "headroom_vs_ceiling.csv", index=False)
    print()
    print(f"=== Headroom against ceiling ({args.ceiling.name}) ===")
    print(head.to_string(index=False))


if __name__ == "__main__":
    main()
