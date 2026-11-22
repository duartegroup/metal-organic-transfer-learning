#!/usr/bin/env python3
"""Regenerate the fold-by-fold champion figures: main-text Figure 5 and SI S23-S31.

What this family is. One chart per metric per threshold, panel A pIC50 and panel B pKi,
showing every arm's value in each cross-validation fold plus a Mean column:

                theta = 5    theta = 6    theta = 7
    MCC            S23          S26          S29
    Accuracy       S24          S27          S30
    F1             S25          S28          S31

Figure 5 of the main text is the theta = 6 MCC member, so it and S26 must agree cell for
cell; this script emits both from one call to the same code path.

Why not `main.py --champion`. That re-ranks the champions from the full configuration
search held in the repository, which is ORIGINAL-protocol data, so it would reproduce the
published figures. The nested re-runs exist only at the twelve champion settings. The
panel code here transcribes `plot.py::_plot_champion_panel` -- arm colours, bar widths,
Mean column, dashed separator, rotated value labels, population s.d. (ddof = 0) on the
error bar -- so the replacements match the style of the SI figures that are not changing.

Four arms, not the published three. The published S23-S31 drew B, TL and Max; S1 was
added to this chart type later and appears in Figure 5 only. Carrying it through all nine
puts the TL-vs-S1 comparison -- the one the paper's surviving claim rests on -- in front
of the reader at every threshold. Decided 2026-11-16; guide section 3.5.

Data, all per fold:
  S1, TL  results/table1/**/*summary.csv, `*_from_probs`, averaged over the three
          training seeds within each fold.
  B       results/baseline_b_per_fold_seeds.csv at the published search seed (1996);
          see guide section 2.6h for why the published baseline cannot be used here.
  Max     pIC50 from results/noise_ceiling_pic50_adopted.csv (sigma = 0.862, the
          adopted value, transcribed from guide section 3.4) -- NOT
          noise_ceiling_recompute.csv, which holds the +26% sensitivity variant at
          sigma = 0.869 and gives a theta=6 mean of 0.643 rather than 0.646.
          pKi from results/noise_ceiling_pki_published.csv -- the values printed in the
          current SI, which the revision does not change.

Usage:
    python champion_fold_figures.py              # all nine, plus Figure 5
    python champion_fold_figures.py --only S26
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ARMS = {"no_pretrain_encoder_task": "S1", "pretrain_ccsa_then_encoder_task": "TL"}
NEURAL_COL = {"Accuracy": "val/accuracy_from_probs", "F1": "val/f1_from_probs",
              "MCC": "val/mcc_from_probs"}
B_COL = {"Accuracy": "accuracy", "F1": "f1", "MCC": "mcc"}
METRICS = ("MCC", "Accuracy", "F1")
THRS = (5, 6, 7)
FOLDS = (1, 2, 3, 4, 5)
B_SEED = 1996
# metric -> {threshold: SI figure number}
SI = {"MCC": {5: "S23", 6: "S26", 7: "S29"},
      "Accuracy": {5: "S24", 6: "S27", 7: "S30"},
      "F1": {5: "S25", 6: "S28", 7: "S31"}}
COL = {"B": "#5B7FA5", "S1": "#B08A3E", "TL": "#6BA368",
       "max_face": "#E8C8C3", "max_edge": "#C0796E"}


def load_neural(runs: Path) -> dict:
    """{(endpoint, thr, fold, arm, metric): seed-averaged value}"""
    acc = defaultdict(list)
    for f in glob.glob(str(runs / "**/*summary.csv"), recursive=True):
        m = re.search(r"/(pic50|pk)_t(\d)_.*_s(\d+)/", f)
        if not m:
            continue
        e = "pIC50" if m.group(1) == "pic50" else "pKi"
        t = int(m.group(2))
        for r in csv.DictReader(open(f)):
            arm = ARMS.get(r["method"])
            if not arm:
                continue
            try:
                fold = int(r["fold"])
            except ValueError:
                continue
            for lab, c in NEURAL_COL.items():
                if r.get(c):
                    acc[(e, t, fold, arm, lab)].append(float(r[c]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def load_baseline(path: Path) -> dict:
    out = {}
    for r in csv.DictReader(open(path)):
        if int(r["seed"]) != B_SEED:
            continue
        for lab, c in B_COL.items():
            out[(r["endpoint"], int(float(r["threshold"])), int(r["fold"]), "B", lab)] = \
                float(r[c])
    return out


def load_ceilings(root: Path, pic50_csv: Path) -> dict:
    """{(endpoint, thr, metric): (5 per-fold values)}"""
    out = {}
    pic = defaultdict(dict)
    for r in csv.DictReader(open(pic50_csv)):
        for lab, c in (("MCC", "mcc"), ("Accuracy", "accuracy"), ("F1", "f1")):
            pic[(int(float(r["threshold"])), lab)][int(r["fold"])] = float(r[c])
    for (t, lab), d in pic.items():
        assert sorted(d) == list(FOLDS), f"pIC50 ceiling folds incomplete at t{t} {lab}"
        out[("pIC50", t, lab)] = tuple(d[f] for f in FOLDS)
    # pKi: the PUBLISHED ceilings. sigma_pKi is unchanged by the revision, so these
    # tables do not move. noise_ceiling_recompute.csv re-runs the same estimator and
    # lands within 0.004 of them -- Monte Carlo noise, not a correction -- but one cell
    # (t7 f3: 0.788 published, 0.784 recomputed) differs at the two decimals the bar
    # labels use, which would have put the figure out of step with Table S8.
    pki = defaultdict(dict)
    for r in csv.DictReader(open(root / "results/noise_ceiling_pki_published.csv")):
        for lab, c in (("MCC", "mcc"), ("Accuracy", "accuracy"), ("F1", "f1")):
            pki[(int(float(r["threshold"])), lab)][int(r["fold"])] = float(r[c])
    for (t, lab), d in pki.items():
        assert sorted(d) == list(FOLDS), f"pKi ceiling folds incomplete at t{t} {lab}"
        out[("pKi", t, lab)] = tuple(d[f] for f in FOLDS)
    return out


def panel(ax, arms, fold_labels, title, panel_label, metric):
    """Transcription of plot.py::_plot_champion_panel."""
    n = len(fold_labels)
    x = np.arange(n + 1)
    width = 0.75 / len(arms)
    offsets = (np.arange(len(arms)) - (len(arms) - 1) / 2) * width
    err_kw = dict(fmt="none", color="black", lw=1.2, capsize=3, capthick=1.2, zorder=5)
    label_specs, observed = [], []
    for (label, vals, face, edge, hatch, show_err), off in zip(arms, offsets):
        vals = np.asarray(vals, dtype=float)
        mean, std = float(np.nanmean(vals)), float(np.nanstd(vals))  # ddof=0, as original
        heights = np.append(vals, mean)
        bars = ax.bar(x + off, heights, width, color=face, edgecolor=edge,
                      linewidth=1.0 if hatch else 0.5, hatch=hatch, label=label, zorder=3)
        if show_err:
            ax.errorbar(x[-1] + off, mean, yerr=std, **err_kw)
        label_specs.append((bars, [0.0] * n + [std if show_err else 0.0]))
        low = float(np.nanmin(heights))
        observed.append(min(low, mean - std) if show_err else low)
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


def build(vals, ceil, metric, thr, outdir, stem, summary):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, (ep, disp, lab) in zip(axes, [("pIC50", "pIC$_{50}$", "A"),
                                          ("pKi", "pK$_{i}$", "B")]):
        series = {a: [vals[(ep, thr, f, a, metric)] for f in FOLDS]
                  for a in ("B", "S1", "TL")}
        mx = list(ceil[(ep, thr, metric)])
        arms = [("B", series["B"], COL["B"], "white", None, True),
                ("S1", series["S1"], COL["S1"], "white", None, True),
                ("TL", series["TL"], COL["TL"], "white", None, True),
                ("Max (upper bound)", mx, COL["max_face"], COL["max_edge"], "//", False)]
        panel(ax, arms, [str(f) for f in FOLDS], disp, lab, metric)
        # eight decimals: the figures label bars at two, and re-rounding a
        # four-decimal record drifts on the half-way cases (pKi theta=7 S1 MCC is
        # 0.47497, which must print as 0.47, not 0.48).
        summary.append(dict(figure=stem, metric=metric, threshold=thr, endpoint=ep,
                            **{a: round(float(np.mean(v)), 8) for a, v in series.items()},
                            Max=round(float(np.mean(mx)), 8)))
    axes[1].set_ylabel("")
    for ax in axes:
        ax.set_xlabel("Cross-validation fold", fontsize=15)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=16, frameon=True,
               fancybox=False, edgecolor="#ccc", bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{stem}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=here / "results/table1")
    ap.add_argument("--baseline", type=Path,
                    default=here / "results/baseline_b_per_fold_seeds.csv")
    ap.add_argument("--pic50-ceilings", type=Path,
                    default=here / "results/noise_ceiling_pic50_adopted.csv")
    ap.add_argument("--outdir", type=Path, default=here / "results/figures")
    ap.add_argument("--only", help="one SI number, e.g. S26")
    args = ap.parse_args()

    vals = load_neural(args.runs)
    vals.update(load_baseline(args.baseline))
    ceil = load_ceilings(here, args.pic50_ceilings)
    missing = [(e, t, f, a, m) for e in ("pIC50", "pKi") for t in THRS for f in FOLDS
               for a in ("B", "S1", "TL") for m in METRICS if (e, t, f, a, m) not in vals]
    assert not missing, f"{len(missing)} cells missing, e.g. {missing[:3]}"
    print(f"loaded {len(vals)} per-fold values, {len(ceil)} ceiling series\n")
    args.outdir.mkdir(parents=True, exist_ok=True)

    summary = []
    for metric in METRICS:
        for thr in THRS:
            fig_id = SI[metric][thr]
            if args.only and args.only.upper() != fig_id:
                continue
            build(vals, ceil, metric, thr, args.outdir,
                  f"figure_{fig_id}_{metric.lower()}_threshold_{thr}", summary)
            print(f"  {fig_id}  {metric:9s} theta={thr}")
    # Figure 5 is the theta = 6 MCC member, emitted again under its own name
    if not args.only or args.only.upper() in ("S26", "FIG5"):
        build(vals, ceil, "MCC", 6, args.outdir,
              "figure5_mcc_threshold_6_nested", summary)
        print("  Fig5  MCC       theta=6  (same data as S26)")

    out = here / "results/champion_figure_means.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
