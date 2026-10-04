"""
Verify that the inference pipeline reproduces the training pipeline exactly.

For every threshold / fold of the target ("caged") dataset this script:

1. Recomputes the fold's min-max statistics from the target training split and
   compares them against the ``normalization_stats.pkl`` written during training.
2. Rebuilds the fold's validation set through the training data pipeline
   (``split_data_by_group`` -> ``prepare_datasets`` -> ``BindingAffinitiesDataset``),
   runs the stored checkpoint over it and compares the resulting validation
   metrics against the summary CSV produced during training.
3. Pushes the same raw HDF5 embeddings through the *inference* code path
   (``prepare_input_with_fold_stats``) and compares the per-compound
   probabilities against the training path.

Any non-zero difference in step 3 means inference and training disagree.

It also records, per fold, the ``ModelCheckpoint`` state embedded in the
``.ckpt`` file. The ``best_model_score`` stored there is the monitored
``val/loss``; when it equals the summary CSV's ``val/loss`` the checkpoint on
disk is provably the one training selected, independent of any forward pass.

A note on ``val/mcc``: a difference between the reproduced MCC and the summary
CSV is *expected* on folds where the model predicted a single class, and does
not indicate a bad checkpoint. Training logged MCC with
``torchmetrics.BinaryMatthewsCorrCoef``, whose ``_matthews_corrcoef_reduce``
falls back to an ``eps``-based branch when the confusion matrix is degenerate
and returns a fabricated non-zero value where MCC is mathematically undefined.
For an all-positive prediction over ``n`` samples with ``P`` positives it
reduces to ``(2P - n) / sqrt(n * P * (n - P))`` -- e.g. 0.214478 at threshold 5
fold 1 (n=62, P=51) and 0.631748 at fold 2 (n=29, P=27). ``sklearn`` returns the
correct 0.0 for the same confusion matrix. Such folds are flagged below with
``degenerate_prediction`` and compared against ``mcc_torchmetrics`` instead.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryMatthewsCorrCoef

REPO_ROOT = Path(__file__).resolve().parents[3]
INFERENCE_DIR = REPO_ROOT / "src" / "transfer_learning_src" / "inference"
sys.path.insert(0, str(INFERENCE_DIR))

from inference import (  # noqa: E402
    build_params,
    load_model_checkpoint,
    prepare_input_with_fold_stats,
)
from data.load_data import load_data_hdf5  # noqa: E402
from data.split_data import split_data_by_group  # noqa: E402
from data.utils_data import compute_scaling_stats, prepare_datasets  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

MODALITY = "molecule_protein"
INHIBITOR_DESCRIPTOR = "SOAP"
BIOLOGICAL_DESCRIPTOR = "ESM-C"
AFFINITY_TYPE = "pic50"
DATA_SCALER = "minmax"
N_FOLDS = 5
DATASET = "caged"
TRAIN_CONFIG = "separate_domain_source_smote_no_target_smote"
SUMMARY_CSV = "mol_protein_SOAP_ESM-C_summary.csv"

# Map checkpoint directory names to the "method" column of the summary CSV.
PHASE_TO_METHOD = {
    "pretrain_ccsa": "pretrain_ccsa",
    "pretrain_ccsa_encoder_task": "pretrain_ccsa_then_encoder_task",
    "pretrain_ccsa_task": "pretrain_ccsa_then_task",
    "pretrain_encoder_task": "pretrain_encoder_task",
    "pretrain_task": "pretrain_task",
    "no_pretrain_ccsa": "no_pretrain_ccsa",
    "no_pretrain_ccsa_encoder_task": "no_pretrain_ccsa_then_encoder_task",
    "no_pretrain_ccsa_task": "no_pretrain_ccsa_then_task",
    "no_pretrain_encoder_task": "no_pretrain_encoder_task",
}


def make_params(threshold: float) -> Dict[str, Any]:
    """Build the model parameter dict used for both training and inference."""
    return build_params(
        threshold=threshold,
        inhibitor_descriptor=INHIBITOR_DESCRIPTOR,
        biological_descriptor=BIOLOGICAL_DESCRIPTOR,
        affinity_type=AFFINITY_TYPE,
        modality=MODALITY,
        data_scaler=DATA_SCALER,
        cross_domain_scaling=False,
        source_smote=True,
        target_smote=False,
        n_folds=N_FOLDS,
    )


def compare_stats(computed: Dict[str, Any], saved: Dict[str, Any]) -> Dict[str, float]:
    """Return the maximum absolute deviation between two min-max stat dicts."""
    return {
        "min_maxabs_diff": float(
            np.max(np.abs(np.asarray(computed["combined_min"], dtype=np.float64)
                          - np.asarray(saved["combined_min"], dtype=np.float64)))
        ),
        "max_maxabs_diff": float(
            np.max(np.abs(np.asarray(computed["combined_max"], dtype=np.float64)
                          - np.asarray(saved["combined_max"], dtype=np.float64)))
        ),
    }


def checkpoint_provenance(
    checkpoint_dir: Path, summary_loss: Any
) -> Dict[str, Any]:
    """
    Read the ``ModelCheckpoint`` state stored inside the fold's checkpoint.

    Lightning writes the checkpoint callback's own state into every ``.ckpt``.
    ``best_model_score`` is the value of the monitored metric (``val/loss``)
    that caused this file to be kept, and ``best_model_path`` is the file it
    wrote. When both agree with the summary CSV and with the file present on
    disk, the weights being loaded are provably the ones training selected --
    no forward pass required.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory holding the fold's ``.ckpt`` file(s).
    summary_loss : Any
        The ``val/loss`` reported for this fold in the training summary CSV,
        or None when the fold has no summary row.

    Returns
    -------
    Dict[str, Any]
        Provenance fields, including ``matches_summary_loss``.
    """
    ckpt_files = sorted(checkpoint_dir.glob("*.ckpt"))
    # load_model_checkpoint takes checkpoint_files[0]; record the count so a
    # directory with more than one file cannot pass unnoticed.
    checkpoint = torch.load(ckpt_files[0], map_location="cpu", weights_only=False)

    callbacks = checkpoint.get("callbacks", {})
    state = next(
        (v for k, v in callbacks.items() if "ModelCheckpoint" in str(k)), {}
    )
    best_score = state.get("best_model_score")
    best_score = float(best_score) if best_score is not None else None
    best_path = state.get("best_model_path")

    return {
        "n_ckpt_files": len(ckpt_files),
        "loaded_file": ckpt_files[0].name,
        "best_model_path": Path(best_path).name if best_path else None,
        "best_model_score": best_score,
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "summary_loss": (
            float(summary_loss) if summary_loss is not None else None
        ),
        "matches_summary_loss": (
            best_score is not None
            and summary_loss is not None
            and abs(best_score - float(summary_loss)) < 1e-6
        ),
        "path_matches_file": (
            best_path is not None and Path(best_path).name == ckpt_files[0].name
        ),
    }


def evaluate_training_path(model, dataset, device) -> Dict[str, np.ndarray]:
    """Run the model over a dataset built by the training pipeline."""
    mol = dataset.mol_vecs.to(device)
    protein = dataset.protein_vecs.to(device)
    pocket = torch.empty(0).to(device)
    labels = dataset.affinities.to(device)

    with torch.no_grad():
        logits = model.forward((mol, protein, pocket, labels))
        probs = torch.sigmoid(logits)

    return {
        "keys": np.asarray(dataset.keys),
        "affinities": dataset.affinities.cpu().numpy(),
        "logits": logits.cpu().numpy(),
        "probs": probs.cpu().numpy(),
    }


def evaluate_inference_path(model, tgt_val, keys, stats, device) -> np.ndarray:
    """Run the model over the same compounds using the inference code path."""
    probs = []
    with torch.no_grad():
        for key in keys:
            mol_raw = np.asarray(tgt_val["inhibitor_feats"][key])
            prot_raw = np.asarray(tgt_val["protein_feats"][key])
            batch = prepare_input_with_fold_stats(
                mol_raw, prot_raw, None, stats, MODALITY, device
            )
            logits = model.forward(batch)
            probs.append(float(torch.sigmoid(logits).cpu().numpy()[0]))
    return np.asarray(probs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=str(REPO_ROOT / "results/transfer_learning_results/top_models"),
    )
    parser.add_argument("--phases", nargs="+", default=["pretrain_ccsa_encoder_task"])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[5.0, 6.0, 7.0])
    parser.add_argument(
        "--output", default=str(Path(__file__).resolve().parent.parent / "results" / "equivalence_report.json")
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print(f"Loading {DATASET}/{AFFINITY_TYPE} HDF5 dataset...", flush=True)
    data = load_data_hdf5(DATASET, INHIBITOR_DESCRIPTOR, AFFINITY_TYPE)

    report = []

    for threshold in args.thresholds:
        params = make_params(threshold)
        summary_path = (
            Path(args.model_path)
            / AFFINITY_TYPE
            / f"threshold_{threshold:1.0f}"
            / SUMMARY_CSV
        )
        summary = pd.read_csv(summary_path)

        for phase in args.phases:
            method = PHASE_TO_METHOD[phase]
            for fold in range(1, N_FOLDS + 1):
                entry: Dict[str, Any] = {
                    "threshold": threshold,
                    "phase": phase,
                    "fold": fold,
                }

                row = summary[
                    (summary["method"] == method) & (summary["fold"].astype(str) == str(fold))
                ]
                entry["training_metrics"] = (
                    {
                        "loss": float(row["val/loss"].iloc[0]),
                        "accuracy": float(row["val/accuracy"].iloc[0]),
                        "f1": float(row["val/f1"].iloc[0]),
                        "auroc": float(row["val/auroc"].iloc[0]),
                        "mcc": float(row["val/mcc"].iloc[0]),
                    }
                    if len(row)
                    else None
                )

                tgt_train, tgt_val = split_data_by_group(
                    data["inhibitor_feats"],
                    data["protein_feats"],
                    data["pocket_feats"],
                    data["binding_affs"],
                    dataset=DATASET,
                    affinity_type=AFFINITY_TYPE,
                    fold=fold,
                    n_folds=N_FOLDS,
                )

                stats_path = (
                    Path(args.model_path)
                    / AFFINITY_TYPE
                    / f"threshold_{threshold:1.0f}"
                    / f"fold_{fold}"
                    / phase
                    / "normalization_stats.pkl"
                )
                with open(stats_path, "rb") as handle:
                    saved_stats = pickle.load(handle)

                # 1. Are the saved stats the ones the training split produces?
                recomputed = compute_scaling_stats(tgt_train, MODALITY, DATA_SCALER)
                entry["stats_check"] = compare_stats(recomputed, saved_stats)

                # 2. Rebuild the validation set exactly as training did.
                (_, _, _, tgt_val_ds), _ = prepare_datasets(
                    None,
                    None,
                    tgt_train,
                    tgt_val,
                    MODALITY,
                    scaling_method=DATA_SCALER,
                    scaling_stats=None,
                    source_smote=False,
                    target_smote=False,
                    cross_domain_scaling=False,
                    with_keys=True,
                )

                model = load_model_checkpoint(
                    args.model_path,
                    fold,
                    phase,
                    params,
                    descriptor_type=INHIBITOR_DESCRIPTOR,
                    modality=MODALITY,
                    scaler_type=DATA_SCALER,
                    affinity_type=AFFINITY_TYPE,
                    train_config=TRAIN_CONFIG,
                ).to(device)
                model.eval()

                # 2b. Is the file we just loaded the checkpoint training kept?
                entry["checkpoint_provenance"] = checkpoint_provenance(
                    stats_path.parent / "checkpoints",
                    entry["training_metrics"]["loss"]
                    if entry["training_metrics"]
                    else None,
                )

                train_path = evaluate_training_path(model, tgt_val_ds, device)
                labels = (train_path["affinities"] > threshold).astype(int)
                preds = (train_path["probs"] > 0.5).astype(int)

                # val/loss is the metric ModelCheckpoint monitored, so
                # reproducing it end-to-end is the strongest check that these
                # are the right weights. The model uses pos_weight=None.
                logits = torch.from_numpy(train_path["logits"])
                loss = F.binary_cross_entropy_with_logits(
                    logits, torch.from_numpy(labels).float().reshape(logits.shape)
                )

                # torchmetrics fabricates a non-zero MCC on a degenerate
                # confusion matrix; report both so the CSV can be compared
                # against the metric that actually produced it.
                tm_mcc = BinaryMatthewsCorrCoef()(
                    torch.from_numpy(preds).float(), torch.from_numpy(labels)
                )
                cm = confusion_matrix(labels, preds, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()

                entry["n_val"] = int(len(labels))
                entry["confusion_matrix"] = {
                    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
                }
                entry["degenerate_prediction"] = bool(tp + fp == 0 or tn + fn == 0)
                entry["reproduced_metrics"] = {
                    "loss": float(loss),
                    "accuracy": float(accuracy_score(labels, preds)),
                    "f1": float(f1_score(labels, preds, zero_division=0)),
                    "auroc": (
                        float(roc_auc_score(labels, train_path["probs"]))
                        if len(np.unique(labels)) > 1
                        else None
                    ),
                    "mcc": float(matthews_corrcoef(labels, preds)),
                    "mcc_torchmetrics": float(tm_mcc),
                }

                # 3. Same compounds through the inference code path, using the
                #    stats inference now recomputes (must match training), and
                #    for contrast the stale saved pkl stats.
                inference_probs = evaluate_inference_path(
                    model, tgt_val, train_path["keys"], recomputed, device
                )
                diff = np.abs(inference_probs - train_path["probs"])
                entry["inference_vs_training"] = {
                    "max_abs_prob_diff": float(diff.max()),
                    "mean_abs_prob_diff": float(diff.mean()),
                }

                saved_probs = evaluate_inference_path(
                    model, tgt_val, train_path["keys"], saved_stats, device
                )
                saved_diff = np.abs(saved_probs - train_path["probs"])
                entry["saved_pkl_stats_vs_training"] = {
                    "max_abs_prob_diff": float(saved_diff.max()),
                    "mean_abs_prob_diff": float(saved_diff.mean()),
                }

                prov = entry["checkpoint_provenance"]
                repro = entry["reproduced_metrics"]
                training = entry["training_metrics"]

                if entry["degenerate_prediction"]:
                    # MCC is undefined here; compare against the torchmetrics
                    # placeholder the CSV actually holds, not against sklearn.
                    mcc_note = (
                        f"mcc=undefined (single-class prediction) "
                        f"torchmetrics={repro['mcc_torchmetrics']:.4f} "
                        f"(training {training['mcc']:.4f})"
                    )
                else:
                    mcc_note = (
                        f"mcc={repro['mcc']:.4f} (training {training['mcc']:.4f})"
                    )

                print(
                    f"th{threshold:.0f} {phase} fold{fold}: "
                    f"ckpt={prov['loaded_file']} "
                    f"best_score={'OK' if prov['matches_summary_loss'] else 'MISMATCH'} "
                    f"loss={repro['loss']:.6f} (training {training['loss']:.6f}) "
                    f"stats_diff={entry['stats_check']['max_maxabs_diff']:.3e} "
                    f"{mcc_note} "
                    f"max|inf-train| = {diff.max():.3e} "
                    f"(with stale pkl stats: {saved_diff.max():.3e})",
                    flush=True,
                )
                report.append(entry)

    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
