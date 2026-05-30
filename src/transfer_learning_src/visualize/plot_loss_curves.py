"""Generate training loss-curve and pre-training metric figures for the champion models.

Reads champion_config.json (produced by `python main.py --champion`) to identify the
current top-ranked model for each affinity type, derives the Slurm .out filename from
that config, parses the WandB run IDs directly from the .out file, then fetches run
histories and saves three publication-ready figures:

  pretraining_curves.png   – SSL pre-training KNN-AUROC  (pIC50, pKi)
  loss_curves_pic50.png    – per-fold CCSA + fine-tune val/loss  (pIC50)
  loss_curves_pk.png       – per-fold CCSA + fine-tune val/loss  (pKi)

The plot functions are kept identical to the original Jupyter notebook so that the
output figures are pixel-exact reproductions.

Usage
-----
  python plot_loss_curves.py \\
      [--champion-config PATH]  \\
      [--archive-dir      PATH] \\
      [--threshold        N]    \\
      [--output-dir       PATH] \\
      [--wandb-api-key    KEY]  \\
      [--font-dir         PATH]
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s – %(message)s")
logger = logging.getLogger(__name__)

# WandB project path used in all run references
_WANDB_PROJECT = "matthijshakkennes-leiden-university/tl-final-runs"

# Module-level api handle – populated in main() after wandb.login()
api = None

# ---------------------------------------------------------------------------
# Modality → Slurm .out filename abbreviation
# ---------------------------------------------------------------------------
_MODALITY_TO_ABBREV = {
    "molecule":                "mol_only",
    "protein":                 "prot_only",
    "pocket":                  "pocket_only",
    "molecule_protein":        "mol_prot",
    "molecule_pocket":         "mol_pocket",
    "molecule_protein_pocket": "mol_prot_pocket",
}


# ---------------------------------------------------------------------------
# .out filename construction
# ---------------------------------------------------------------------------

def _build_out_filename(tl_cfg: dict, affinity_type: str) -> str:
    inhibitor = tl_cfg["inhibitor"]
    scaler    = tl_cfg["scaler"]
    modality  = tl_cfg["modality"]
    src_smote = tl_cfg["source_smote"]
    tgt_smote = tl_cfg["target_smote"]
    cross     = tl_cfg["cross_domain"]

    abbrev    = _MODALITY_TO_ABBREV[modality]
    smote_str = "smote_true"       if src_smote else "smote_false"
    tgt_str   = "targetsmote_true" if tgt_smote else "targetsmote_false"
    cross_str = "cross_true"       if cross     else "cross_false"

    return f"{inhibitor}_{scaler}_{affinity_type}_{abbrev}_{smote_str}_{tgt_str}_{cross_str}.out"


# ---------------------------------------------------------------------------
# WandB run-ID parser
# ---------------------------------------------------------------------------

def _parse_run_ids(out_path: str, n_folds: int = 5) -> dict:
    """Return first-occurrence WandB run IDs for pretrain_src and each fold."""
    with open(out_path) as f:
        lines = f.readlines()

    def _next_id(start: int) -> Optional[str]:
        for j in range(start + 1, min(start + 6, len(lines))):
            m = re.search(r"View run at https://[^/]+/.+/runs/([a-z0-9]+)", lines[j])
            if m:
                return m.group(1)
        return None

    pretrain_src = None
    fold_ccsa: Dict[int, str] = {}
    fold_enc:  Dict[int, str] = {}

    for i, line in enumerate(lines):
        m = re.search(r"Syncing run (\S+)", line)
        if not m:
            continue
        name   = m.group(1)
        run_id = _next_id(i)
        if run_id is None:
            continue

        if name.startswith("pretrain_src") and pretrain_src is None:
            pretrain_src = run_id
            continue

        m_ccsa = re.match(r"ccsa_.+_fold_(\d+)$", name)
        if m_ccsa:
            fold = int(m_ccsa.group(1))
            if fold not in fold_ccsa:
                fold_ccsa[fold] = run_id
            continue

        m_enc = re.match(r"encoder_task_.+_fold_(\d+)_consistent_scaling$", name)
        if m_enc:
            fold = int(m_enc.group(1))
            if fold not in fold_enc:
                fold_enc[fold] = run_id
            continue

    if pretrain_src is None:
        logger.warning(f"pretrain_src run ID not found in {out_path}")

    folds = []
    for fold in range(1, n_folds + 1):
        ccsa_id = fold_ccsa.get(fold)
        enc_id  = fold_enc.get(fold)
        if ccsa_id is None or enc_id is None:
            logger.warning(f"Missing run IDs for fold {fold} in {out_path}")
        # Build full WandB path strings matching the notebook's convention:
        #   ccsa          → project/run_id         (no "runs/" prefix)
        #   encoder_task  → project/runs/run_id
        folds.append((
            f"{_WANDB_PROJECT}/{ccsa_id}",
            f"{_WANDB_PROJECT}/runs/{enc_id}",
        ))
    return {"pretrain_src": pretrain_src, "folds": folds}


# ---------------------------------------------------------------------------
# Font setup  (identical rcParams to the notebook; path is configurable)
# ---------------------------------------------------------------------------

def _setup_fonts(font_dir: str) -> None:
    from matplotlib import font_manager as fm
    import matplotlib.pyplot as plt

    for fname in [
        "Palatino_Linotype.ttf",
        "Palatino_Linotype_Bold.ttf",
        "Palatino_Linotype_Italic.ttf",
        "Palatino_Linotype_Bold_Italic.ttf",
    ]:
        full = os.path.join(font_dir, fname)
        if os.path.exists(full):
            fm.fontManager.addfont(full)

    plt.rcParams["mathtext.fontset"]      = "custom"
    plt.rcParams["mathtext.rm"]           = "Palatino Linotype"
    plt.rcParams["mathtext.it"]           = "Palatino Linotype:italic"
    plt.rcParams["mathtext.bf"]           = "Palatino Linotype:bold"
    plt.rcParams["font.family"]           = "Palatino Linotype"
    plt.rcParams["text.usetex"]           = False
    plt.rcParams["text.latex.preamble"]   = r"\usepackage{mathpazo}"

    available = {f.name for f in fm.fontManager.ttflist}
    if "Palatino Linotype" in available:
        logger.info("Palatino Linotype font loaded.")
    else:
        logger.warning(
            f"Palatino Linotype not found in {font_dir!r}. "
            "Figures will use the Matplotlib default font."
        )


# ---------------------------------------------------------------------------
# Plot functions – copied verbatim from the notebook, plt.show() → plt.savefig()
# ---------------------------------------------------------------------------

def plot_val_loss_grid(fold_paths, output_path: str) -> None:
    """Create a 3x2 grid of val/loss plots with the legend only on the sixth subplot."""
    import matplotlib.pyplot as plt
    import pandas as pd

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), dpi=600)
    axes = axes.flatten()

    label_size = 18
    plot_labels = ["A", "B", "C", "D", "E"]  # Labels for 6 subplots

    # First, compute global min and max for y-axis
    global_min = float('inf')
    global_max = float('-inf')
    histories = []

    for ccsa_path, encoder_path in fold_paths:
        ccsa_run = api.run(ccsa_path)
        ccsa_hist = ccsa_run.history(keys=["val/loss", "epoch"])

        encoder_run = api.run(encoder_path)
        encoder_hist = encoder_run.history(keys=["val/loss", "epoch"])
        encoder_hist["epoch_offset"] = encoder_hist["epoch"] + ccsa_hist["epoch"].max() + 1

        histories.append((ccsa_hist, encoder_hist))
        combined_loss = pd.concat([ccsa_hist["val/loss"], encoder_hist["val/loss"]])
        global_min = min(global_min, combined_loss.min())
        global_max = max(global_max, combined_loss.max())

    # Now plot all subplots with same y-axis
    for i, ((ccsa_hist, encoder_hist), (ccsa_path, encoder_path)) in enumerate(zip(histories, fold_paths)):
        ax = axes[i]

        # Offset encoder epochs
        last_ccsa_epoch = ccsa_hist["epoch"].max()
        encoder_hist["epoch_offset"] = encoder_hist["epoch"] + last_ccsa_epoch + 1

        # Merge for global min per plot
        ccsa_df = ccsa_hist[["epoch", "val/loss"]].copy()
        ccsa_df["epoch_plot"] = ccsa_df["epoch"]
        enc_df = encoder_hist[["epoch", "val/loss"]].copy()
        enc_df["epoch_plot"] = encoder_hist["epoch_offset"]
        combined = pd.concat([ccsa_df, enc_df], ignore_index=True)
        min_idx = combined["val/loss"].idxmin()
        best_epoch = combined.loc[min_idx, "epoch_plot"]
        best_loss = combined.loc[min_idx, "val/loss"]

        # Plot curves
        ax.plot(ccsa_hist["epoch"], ccsa_hist["val/loss"], label="CCSA", color="tab:blue")
        ax.plot(encoder_hist["epoch_offset"], encoder_hist["val/loss"], label="Finetune Encoder Task", color="tab:orange")

        # Mark transitions & best loss
        ax.axvline(x=last_ccsa_epoch, color="red", linestyle="--", alpha=0.7, label="End of CCSA")
        ax.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.8, label="Best Loss")
        ax.scatter(best_epoch, best_loss, color="green", zorder=5)

        # Styling
        ax.tick_params(axis='both', which='major', labelsize=label_size)
        # set the y-axis to 2 decimal places
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.2f}'))

        if i >= 4:  # Bottom row
            ax.set_xlabel("Epoch", fontsize=label_size)
        if i % 2 == 0:  # Left column
            ax.set_ylabel("Validation loss", fontsize=label_size)

        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)

        # Subplot label
        ax.text(-0.085, 1.07, f'{plot_labels[i]})', transform=ax.transAxes,
                fontsize=label_size + 2, va='top', ha='right')

    # Correct legend placement and column settings
    axes[3].set_xlabel("Epoch", fontsize=label_size)

    handles, labels = axes[4].get_legend_handles_labels()
    axes[5].spines['top'].set_visible(False)
    axes[5].spines['right'].set_visible(False)
    axes[5].spines['left'].set_visible(False)
    axes[5].spines['bottom'].set_visible(False)
    # do not show the ticks and the label of the ticks
    # Hide ticks and tick labels
    axes[5].tick_params(axis='both', which='both', length=0)
    axes[5].set_xticks([])
    axes[5].set_yticks([])
    axes[5].set_xticklabels([])
    axes[5].set_yticklabels([])
    axes[5].legend(handles, labels, fontsize=label_size, loc='center', ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {output_path}")


def plot_pretraining_metric(fold_paths, output_path: str, metric="val/knn_auroc", plot_labels=None) -> None:
    """
    Create a 1x2 grid of metric plots (e.g., val/loss, val/accuracy) with a shared legend
    and subplot labels.

    Args:
        fold_paths (list of tuples): Each tuple is (ccsa_run_path, encoder_run_path).
        metric (str): The metric to plot from WandB history.
        plot_labels (list of str, optional): Subplot labels, default ['A', 'B'].
    """
    import matplotlib.pyplot as plt

    if plot_labels is None:
        plot_labels = ["A", "B"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=600)
    axes = axes.flatten()

    label_size = 18

    for i, ccsa_path in enumerate(fold_paths):
        ax = axes[i]

        # Load histories
        ccsa_run = api.run(ccsa_path)
        ccsa_hist = ccsa_run.history(keys=[metric, "epoch"])

        # Offset encoder epochs
        last_ccsa_epoch = ccsa_hist["epoch"].max()

        # Merge for finding global best
        ccsa_df = ccsa_hist[["epoch", metric]].copy()
        ccsa_df["epoch_plot"] = ccsa_df["epoch"]

        # Determine best (min or max depending on metric)
        if "loss" in metric.lower() or "error" in metric.lower():
            best_idx = ccsa_df[metric].idxmin()
        else:
            best_idx = ccsa_df[metric].idxmax()
        best_epoch = ccsa_df.loc[best_idx, "epoch_plot"]
        best_value = ccsa_df.loc[best_idx, metric]

        # Plot curves
        ax.plot(ccsa_hist["epoch"], ccsa_hist[metric], label="SSL", color="tab:blue")

        # Mark transitions & best
        ax.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.8)
        ax.scatter(best_epoch, best_value, color="green", zorder=5, label=f"Best AUROC")

        # Styling
        ax.tick_params(axis='both', which='major', labelsize=label_size)
        # format the y-axis to 2 decimal places
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.1f}'))
        ax.set_xlabel("Epoch", fontsize=label_size)
        ax.set_ylabel("Validation AUROC", fontsize=label_size)
        ax.grid(alpha=0.3)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)

        # Subplot label
        ax.text(-0.075, 1.07, f'{plot_labels[i]})', transform=ax.transAxes,
                fontsize=label_size + 2, va='top', ha='right')

    # Shared legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=label_size)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global api

    repo = Path(__file__).resolve().parents[4]

    parser = argparse.ArgumentParser(
        description="Generate loss-curve figures from champion_config.json."
    )
    parser.add_argument(
        "--champion-config",
        default=str(repo / "results" / "statistical_tests" / "champion" / "champion_config.json"),
        help="Path to champion_config.json written by main.py --champion.",
    )
    parser.add_argument(
        "--archive-dir",
        default="/archive/mhakkennes/projects/final_ml_runs/transfer_learning_results",
        help="Root of the SURF archive containing per-threshold Slurm .out files.",
    )
    parser.add_argument(
        "--threshold", type=int, default=6,
        help="Activity threshold whose .out file is used (default 6).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo / "results" / "loss_curves"),
        help="Directory where PNG figures are saved.",
    )
    parser.add_argument(
        "--wandb-api-key",
        default=None,
        help="WandB API key (falls back to WANDB_API_KEY env var if not set).",
    )
    parser.add_argument(
        "--font-dir",
        default=None,
        help="Directory containing Palatino Linotype .ttf files (optional).",
    )
    args = parser.parse_args()

    # ── champion config ──────────────────────────────────────────────────────
    if not os.path.exists(args.champion_config):
        logger.error(
            f"champion_config.json not found at {args.champion_config}\n"
            "Run:  python main.py --champion   (from src/statistical_tests)"
        )
        sys.exit(1)

    with open(args.champion_config) as f:
        champion = json.load(f)

    # ── fonts (same rcParams as the notebook) ────────────────────────────────
    _setup_fonts(args.font_dir)

    # ── WandB login ──────────────────────────────────────────────────────────
    import wandb
    api_key = args.wandb_api_key or os.environ.get("WANDB_API_KEY")
    wandb.login(key=api_key)
    api = wandb.Api()

    # ── output dir ───────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── parse run IDs and build fold_paths for each affinity type ─────────────
    pretrain_paths: List[str]           = []
    fold_paths_by_affinity: Dict        = {}
    affinity_order                      = list(champion.keys())   # preserves insertion order

    for affinity_type in affinity_order:
        tl_cfg   = champion[affinity_type]["tl"]
        out_file = _build_out_filename(tl_cfg, affinity_type)
        out_path = os.path.join(args.archive_dir, f"threshold_{args.threshold}", out_file)

        if not os.path.exists(out_path):
            logger.error(f".out file not found: {out_path}")
            sys.exit(1)

        logger.info(f"Parsing {affinity_type} run IDs from: {out_path}")
        run_ids = _parse_run_ids(out_path)

        # Pretraining paths list (one per affinity, order matches affinity_order)
        pretrain_id = run_ids["pretrain_src"]
        pretrain_paths.append(f"{_WANDB_PROJECT}/runs/{pretrain_id}")

        fold_paths_by_affinity[affinity_type] = run_ids["folds"]

    # ── generate figures ─────────────────────────────────────────────────────
    logger.info("Generating pretraining curves …")
    plot_pretraining_metric(
        pretrain_paths,
        output_path=os.path.join(args.output_dir, "pretraining_curves.png"),
    )

    for affinity_type in affinity_order:
        logger.info(f"Generating fold loss curves for {affinity_type} …")
        plot_val_loss_grid(
            fold_paths_by_affinity[affinity_type],
            output_path=os.path.join(args.output_dir, f"loss_curves_{affinity_type}.png"),
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
