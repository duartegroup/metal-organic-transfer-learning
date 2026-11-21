#!/usr/bin/env python3
"""Regenerate Figure 5 -- the fold-by-fold MCC comparison at the 1 uM threshold -- from
the nested runs.

Why this is not `main.py --champion`. That entry point re-ranks the champions from the
full configuration search held in the repository's `results/`, which is the ORIGINAL
protocol's data; running it would faithfully reproduce the published figure, not the
revised one. The nested re-runs exist only for the twelve champion settings, so the
figure has to be drawn from those directly. The panel design here is a transcription of
`plot.py::_plot_champion_panel` -- same arm colours, bar widths, Mean column, dashed
separator, rotated value labels, population s.d. on the error bar (np.nanstd, ddof=0, as
in the original) -- so the revised figure sits beside the unrevised SI figures without a
visible change of style.

The baseline arm. The published Figure 5 drew B as five per-fold bars, but those per-fold
values were never deposited and are not recoverable (guide section 2.5d; the repository
holds no conventional-baseline results at all), so the published column cannot appear
here at all. Since the rest of the figure was produced on different hardware with current
libraries, the baseline is drawn from the re-execution under the published search seed
(1996), which puts every arm in the same environment -- the decision recorded in guide
section 2.6h. Tables S40 and S41 switch with it, and both report the published values
alongside so the choice between them is visible.
`--baseline-mode` switches this: `rerun` (default, per-fold bars from the
re-execution at search seed 1996), `line`, or `none`.

Data
  S1, TL per fold : results/table1_fold_metrics_nested.csv (seed-averaged, 3 seeds)
  Max per fold    : adopted ceilings, sigma_pKi = 0.69 / sigma_pIC50 = 0.862
                    (guide section 3.4; NOT noise_ceiling_recompute.csv, which carries
                    the +26% variant sigma = 0.869)
  B               : champion_config.json per-threshold means (published)

Usage:
    python figure5_nested.py                  # theta = 6, MCC -> PDF + PNG
    python figure5_nested.py --threshold 7
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

FOLDS = (1, 2, 3, 4, 5)
# Adopted per-fold pIC50 ceilings, sigma_pIC50 = 0.862 (guide section 3.4). These are
# NOT in any results CSV: noise_ceiling_recompute.csv holds the +26% sensitivity variant
# at sigma = 0.869, which gives a theta=6 mean of 0.643 rather than the adopted 0.646.
# Three decimals is all the guide carries, and all the figure needs -- bar labels are
# typeset at two.
CEILING_PIC50 = {
    5: (0.572, 0.308, 0.629, 0.730, 0.388),
    6: (0.618, 0.561, 0.751, 0.821, 0.479),
    7: (0.649, 0.674, 0.810, 0.524, 0.488),
}
# pKi ceilings are unchanged by the sigma revision, so they come from the results file at
# full precision (sigma_pKi = 0.69 is the adopted value).
CEILING_PKI_SRC = "results/noise_ceiling_recompute.csv"
# Published conventional-baseline per-threshold mean MCC (champion_config.json).
B_PUBLISHED = {("pIC50", 5): 0.2929, ("pIC50", 6): 0.4783, ("pIC50", 7): 0.5357,
               ("pKi", 5): 0.3881, ("pKi", 6): 0.4987, ("pKi", 7): 0.5408}
COL = {"B": "#5B7FA5", "S1": "#B08A3E", "TL": "#6BA368",
       "max_face": "#E8C8C3", "max_edge": "#C0796E"}


def load_ceilings(root: Path, thr: int) -> dict:
    """{endpoint: (5 per-fold ceiling MCCs)} for this threshold."""
    pki = {}
    for r in csv.DictReader(open(root / CEILING_PKI_SRC)):
        if r["endpoint"] == "pKi" and int(float(r["threshold"])) == thr:
            pki[int(r["fold"])] = float(r["mcc"])
    assert sorted(pki) == list(FOLDS), f"pKi ceiling folds incomplete: {sorted(pki)}"
    return {"pIC50": CEILING_PIC50[thr], "pKi": tuple(pki[f] for f in FOLDS)}


def load_folds(path: Path, thr: int) -> dict:
    out = {}
    for r in csv.DictReader(open(path)):
        if int(float(r["threshold"])) != thr:
            continue
        out[(r["endpoint"], r["model"], int(r["fold"]))] = float(r["mcc"])
    return out


def load_b_rerun(path: Path, thr: int, seed: int = 1996) -> dict:
    out = {}
    for r in csv.DictReader(open(path)):
        if int(float(r["threshold"])) != thr or int(r["seed"]) != seed:
            continue
        out[(r["endpoint"], int(r["fold"]))] = float(r["mcc"])
    return out


def panel(ax, arms, fold_labels, title, panel_label, b_line=None, metric="MCC"):
    """Transcription of plot.py::_plot_champion_panel."""
    n = len(fold_labels)
    x = np.arange(n + 1)
    width = 0.75 / len(arms)
    offsets = (np.arange(len(arms)) - (len(arms) - 1) / 2) * width
    mean_x = x[-1]
    err_kw = dict(fmt="none", color="black", lw=1.2, capsize=3, capthick=1.2, zorder=5)
    label_specs, observed = [], []

    for (label, vals, face, edge, hatch, show_err), off in zip(arms, offsets):
        vals = np.asarray(vals, dtype=float)
        mean, std = float(np.nanmean(vals)), float(np.nanstd(vals))  # ddof=0, as original
        heights = np.append(vals, mean)
        bars = ax.bar(x + off, heights, width, color=face, edgecolor=edge,
                      linewidth=1.0 if hatch else 0.5, hatch=hatch, label=label, zorder=3)
        if show_err:
            ax.errorbar(mean_x + off, mean, yerr=std, **err_kw)
        label_specs.append((bars, [0.0] * n + [std if show_err else 0.0]))
        low = float(np.nanmin(heights))
        observed.append(min(low, mean - std) if show_err else low)

    if b_line is not None:
        ax.axhline(b_line, color=COL["B"], lw=1.6, ls=(0, (6, 3)), zorder=4,
                   label="B (published mean)")
        observed.append(b_line)

    ax.axvline(x=n - 0.5, color="grey", lw=0.8, ls="--", zorder=1)
    fs = 12 if len(arms) == 3 else 10
    for bars, stds in label_specs:
        for bar, std in zip(bars, stds):
            h = bar.get_height()
            y, va = (h + std + 0.012, "bottom") if h >= 0 else (h - std - 0.012, "top")
            ax.text(bar.get_x() + bar.get_width() / 2, y, f"{h:.2f}",
                    ha="center", va=va, fontsize=fs, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(list(fold_labels) + ["Mean"], fontsize=14)
    ax.set_ylabel(metric, fontsize=17)
    lowest = min([0.0] + [v for v in observed if np.isfinite(v)])
    y_min = 0.0 if lowest >= 0 else float(np.floor((lowest - 0.10) * 10) / 10)
    ax.set_ylim(y_min, 1.1)
    if y_min < 0:
        ax.axhline(0, color="#888", lw=0.8, zorder=2)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(axis="y", which="major", color="#ddd", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=18)
    ax.text(-0.08, 1.04, panel_label, transform=ax.transAxes,
            fontsize=20, fontweight="bold", va="bottom", ha="left")


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=int, default=6, choices=(5, 6, 7))
    ap.add_argument("--baseline-mode", choices=("rerun", "line", "none"), default="rerun")
    ap.add_argument("--folds", type=Path,
                    default=here / "results/table1_fold_metrics_nested.csv")
    ap.add_argument("--b-rerun", type=Path,
                    default=here / "results/baseline_b_per_fold_seeds.csv")
    ap.add_argument("--outdir", type=Path, default=here / "results/figures")
    args = ap.parse_args()

    thr = args.threshold
    pf = load_folds(args.folds, thr)
    ceiling = load_ceilings(here, thr)
    b_rr = load_b_rerun(args.b_rerun, thr) if args.baseline_mode == "rerun" else {}
    args.outdir.mkdir(parents=True, exist_ok=True)

    endpoints = [("pIC50", "pIC$_{50}$", "A"), ("pKi", "pK$_{i}$", "B")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    print(f"Figure 5 -- MCC, threshold {thr}, baseline mode '{args.baseline_mode}'")
    for ax, (ep, disp, lab) in zip(axes, endpoints):
        s1 = [pf[(ep, "S1", f)] for f in FOLDS]
        tl = [pf[(ep, "TL", f)] for f in FOLDS]
        mx = list(ceiling[ep])
        arms = []
        if args.baseline_mode == "rerun":
            arms.append(("B", [b_rr[(ep, f)] for f in FOLDS], COL["B"], "white", None, True))
        arms += [("S1", s1, COL["S1"], "white", None, True),
                 ("TL", tl, COL["TL"], "white", None, True),
                 ("Max (upper bound)", mx, COL["max_face"], COL["max_edge"], "//", False)]
        panel(ax, arms, [str(f) for f in FOLDS], disp, lab,
              b_line=B_PUBLISHED[(ep, thr)] if args.baseline_mode == "line" else None)
        b_drawn = (np.mean([b_rr[(ep, f)] for f in FOLDS])
                   if args.baseline_mode == "rerun" else
                   B_PUBLISHED[(ep, thr)] if args.baseline_mode == "line" else float("nan"))
        print(f"  {ep:6s} S1 {np.mean(s1):.3f}  TL {np.mean(tl):.3f}  "
              f"Max {np.mean(mx):.3f}  B {b_drawn:.3f} ({args.baseline_mode})"
              f"   [B published {B_PUBLISHED[(ep, thr)]:.3f}]")

    axes[1].set_ylabel("")
    for ax in axes:
        ax.set_xlabel("Cross-validation fold", fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=16,
               frameon=True, fancybox=False, edgecolor="#ccc", bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    stem = args.outdir / f"figure5_mcc_threshold_{thr}_nested"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=600, bbox_inches="tight")
        print(f"wrote {stem}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
