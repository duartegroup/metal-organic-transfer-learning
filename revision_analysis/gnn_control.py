"""GNN control experiment requested by Reviewer 1 (comment 2).

Trains a 3D message-passing network (SchNet-style continuous-filter convolutions) from
scratch on the metal-organic dataset alone, replacing only the molecular representation
used by training strategy S1 while holding everything else fixed: the same five published
cross-validation folds, the same three activity thresholds, the same metrics, and the same
from-scratch training regime.

Operating on the GFN2-xTB geometries rather than on perceived bonds avoids RDKit's valence
model, which rejects most of these complexes (only 56 of 130 SDFs and 24 of 130 curated
SMILES survive default sanitization), and matches how the SOAP/ACSF/MACE descriptors were
computed -- so the comparison isolates architecture rather than confounding it with a
change of representation.

Usage:
    python gnn_control.py --endpoint pKi
    python gnn_control.py --endpoint pIC50 --thresholds 6
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

REPO = Path(__file__).resolve().parents[2] / "metal-organic-transfer-learning"
METADATA = REPO / "data" / "excel" / "metadata.xlsx"
SDF_DIR = REPO / "data" / "caged_structures" / "sdf"
# Per-residue ESM-C embeddings, keyed by entry_name. The pipeline represents a protein
# by the CLS token -- the first row -- see dataset.py::_compute_fixed_vectors, which the
# protein branch below reproduces exactly so the control differs from S1 in the MOLECULAR
# representation only.
H5 = {"pIC50": REPO.parent / "caged" / "caged_pic50_dataset.h5",
      "pKi": REPO.parent / "caged" / "caged_pk_dataset.h5"}
OUT = Path(__file__).resolve().parents[1] / "results"

# Elements present in the dataset (SI Section S1.2).
SPECIES = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "I", "Ru"]
Z_INDEX = {s: i for i, s in enumerate(SPECIES)}


# ---------------------------------------------------------------- data loading


def read_sdf(path: Path) -> tuple[list[str], np.ndarray] | None:
    """Minimal V2000 SDF reader: element symbols and 3D coordinates.

    Deliberately avoids RDKit, whose valence model rejects most Ru complexes here.
    """
    lines = path.read_text().splitlines()
    if len(lines) < 4:
        return None
    try:
        n_atoms = int(lines[3][0:3])
    except (ValueError, IndexError):
        return None
    symbols, coords = [], []
    for line in lines[4 : 4 + n_atoms]:
        try:
            x, y, z = float(line[0:10]), float(line[10:20]), float(line[20:30])
            sym = line[31:34].strip()
        except (ValueError, IndexError):
            return None
        if sym not in Z_INDEX:
            return None
        symbols.append(sym)
        coords.append((x, y, z))
    return symbols, np.asarray(coords, dtype=np.float32)


def load_protein_cls(endpoint: str, entry_names) -> dict:
    """{entry_name: CLS vector}. Same reduction the descriptor pipeline uses."""
    import h5py
    out = {}
    with h5py.File(H5[endpoint], "r") as f:
        for name in entry_names:
            if name not in f or "protein_embeddings" not in f[name]:
                continue
            emb = np.asarray(f[name]["protein_embeddings"])  # (1, n_res, 1152)
            out[name] = np.ascontiguousarray(emb.squeeze(0)[0], dtype=np.float32)
    return out


def load_dataset(endpoint: str, cutoff: float) -> pd.DataFrame:
    """Load measurements and precompute each molecule's radius graph.

    Geometries are fixed, so the neighbour list never changes during training; building it
    once per molecule avoids an O(N^2) distance matrix over the whole batch.
    """
    meas = pd.read_excel(METADATA, sheet_name="measurements")
    meas = meas[meas["activity_type"] == endpoint].copy()
    meas["fold_si"] = meas["fold"].astype(int) + 1

    geometries: dict[str, dict] = {}
    for mol_id in meas["molecule_id"].unique():
        n = str(mol_id).split("_")[-1]
        parsed = read_sdf(SDF_DIR / f"compound_{n}.sdf")
        if parsed is None:
            continue
        symbols, coords = parsed
        d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        mask = (d < cutoff) & (d > 1e-8)
        src, dst = np.nonzero(mask)
        geometries[mol_id] = {
            "z": np.asarray([Z_INDEX[s] for s in symbols], dtype=np.int64),
            "edges": np.stack([src, dst]).astype(np.int64),
            "dist": d[src, dst].astype(np.float32),
            "n_atoms": len(symbols),
        }

    meas = meas[meas["molecule_id"].isin(geometries)].copy()
    meas.attrs["geometries"] = geometries
    return meas


def collate(rows: pd.DataFrame, geometries: dict, threshold: float, device: str,
            protein: dict | None = None, prot_stats: tuple | None = None):
    z_list, edge_list, dist_list, batch_idx = [], [], [], []
    offset = 0
    for i, mol_id in enumerate(rows["molecule_id"]):
        g = geometries[mol_id]
        z_list.append(torch.from_numpy(g["z"]))
        edge_list.append(torch.from_numpy(g["edges"]) + offset)
        dist_list.append(torch.from_numpy(g["dist"]))
        batch_idx.append(torch.full((g["n_atoms"],), i, dtype=torch.long))
        offset += g["n_atoms"]
    y = torch.tensor(
        (rows["activity_value"].to_numpy() >= threshold).astype(np.float32)
    )
    out_prot = None
    if protein is not None:
        P = np.stack([protein[n] for n in rows["entry_name"]])
        if prot_stats is not None:                      # train-fold statistics only
            mu, sd = prot_stats
            P = (P - mu) / sd
        out_prot = torch.from_numpy(P.astype(np.float32)).to(device)
    return {
        "prot": out_prot,
        "z": torch.cat(z_list).to(device),
        "edges": torch.cat(edge_list, dim=1).to(device),
        "dist": torch.cat(dist_list).to(device),
        "batch": torch.cat(batch_idx).to(device),
        "y": y.to(device),
        "n_graphs": len(rows),
    }


# ---------------------------------------------------------------------- model


class Interaction(nn.Module):
    """Continuous-filter convolution (SchNet-style)."""

    def __init__(self, hidden: int, n_rbf: int):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.pre = nn.Linear(hidden, hidden)
        self.post = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, h, edge_index, rbf, cutoff_weight):
        src, dst = edge_index
        w = self.filter_net(rbf) * cutoff_weight.unsqueeze(-1)
        msg = self.pre(h)[src] * w
        agg = torch.zeros_like(h).index_add_(0, dst, msg)
        return h + self.post(agg)


class SchNetLite(nn.Module):
    def __init__(self, hidden: int = 64, n_interactions: int = 3, n_rbf: int = 32,
                 cutoff: float = 5.0, protein_dim: int = 0):
        super().__init__()
        self.cutoff = cutoff
        self.embedding = nn.Embedding(len(SPECIES), hidden)
        self.register_buffer("centres", torch.linspace(0.0, cutoff, n_rbf))
        self.width = (cutoff / n_rbf) * 2.0
        self.interactions = nn.ModuleList(
            Interaction(hidden, n_rbf) for _ in range(n_interactions)
        )
        # Protein branch. Absent (protein_dim = 0) the module list, the head width and
        # therefore every parameter shape are exactly as before, so compound-only runs
        # reproduce bit for bit.
        self.protein_net = (
            nn.Sequential(nn.Linear(protein_dim, hidden), nn.SiLU(),
                          nn.Linear(hidden, hidden))
            if protein_dim else None
        )
        head_in = hidden * (2 if protein_dim else 1)
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(self, batch_data):
        z, edges, d = batch_data["z"], batch_data["edges"], batch_data["dist"]
        batch, n_graphs = batch_data["batch"], batch_data["n_graphs"]

        rbf = torch.exp(-((d.unsqueeze(-1) - self.centres) ** 2) / self.width**2)
        cutoff_weight = 0.5 * (torch.cos(d * np.pi / self.cutoff) + 1.0)

        h = self.embedding(z)
        for block in self.interactions:
            h = block(h, (edges[0], edges[1]), rbf, cutoff_weight)

        pooled = torch.zeros(n_graphs, h.shape[1], device=h.device).index_add_(0, batch, h)
        if self.protein_net is not None:
            pooled = torch.cat([pooled, self.protein_net(batch_data["prot"])], dim=-1)
        return self.head(pooled).squeeze(-1)


# -------------------------------------------------------------------- training


def run_fold(train_rows, val_rows, test_rows, geometries, threshold, args, device, seed,
             protein=None):
    torch.manual_seed(seed)
    prot_dim = len(next(iter(protein.values()))) if protein else 0
    model = SchNetLite(args.hidden, args.interactions, cutoff=args.cutoff,
                       protein_dim=prot_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    # Standardise the protein features on the TRAINING fold only. The raw ESM-C values
    # would otherwise dominate the learned molecular representation, and statistics taken
    # over all folds would leak the test fold into training.
    prot_stats = None
    if protein:
        P = np.stack([protein[n] for n in train_rows["entry_name"]])
        mu, sd = P.mean(0), P.std(0)
        prot_stats = (mu, np.where(sd < 1e-8, 1.0, sd))

    train_batch = collate(train_rows, geometries, threshold, device, protein, prot_stats)
    val_batch = collate(val_rows, geometries, threshold, device, protein, prot_stats)
    test_batch = collate(test_rows, geometries, threshold, device, protein, prot_stats)

    y_train = train_batch["y"]
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor(
        [n_neg / n_pos if n_pos > 0 else 1.0], device=device
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    y_val_t = val_batch["y"]
    y_val = y_val_t.cpu().numpy()

    # Checkpoint selection. `loss` is the default and the only rule used for anything
    # reported; `mcc` reproduces the original rule and exists solely to re-derive the
    # discarded 2026-11-08 numbers (guide 2.6x-e).
    #
    # Why the original rule had to go. Validation MCC on a ~40-row fold is a step
    # function: one flipped prediction moves it by ~0.05, so a last-decimal change in a
    # logit -- the kind thread count alone produces, since reduction order in index_add_
    # and the linear layers is not fixed -- selects a different epoch and therefore a
    # materially different model. Run at one configuration in four environments the
    # control spanned 0.2264 to 0.3402 MCC, a range wider than the 0.098 effect it is
    # meant to measure. Cross-entropy is continuous in the logits, so a perturbation of
    # that size moves the argmin by nothing at all.
    #
    # This also brings the control INTO line with the pipeline it is compared against:
    # Table S13 gives `val/loss` as the monitored metric for both the CCSA and
    # fine-tuning stages. The control was the odd one out.
    best_score, best_state, stale = float("inf"), None, 0
    best_val_mcc, best_val_loss, best_epoch = float("nan"), float("nan"), -1
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        logits = model(train_batch)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()
        sched.step()

        if epoch % args.val_every:
            continue
        model.eval()
        with torch.no_grad():
            val_logits = model(val_batch)
            val_loss = float(loss_fn(val_logits, y_val_t))
            val_pred = (val_logits > 0).float().cpu().numpy()
        val_mcc = matthews_corrcoef(y_val, val_pred)
        # Both rules are written as minimisation so the comparison below is one branch.
        score = val_loss if args.select_metric == "loss" else -val_mcc
        if score < best_score:
            best_score, stale = score, 0
            best_val_mcc, best_val_loss, best_epoch = val_mcc, val_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += args.val_every
            if stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = (model(test_batch) > 0).float().cpu().numpy()
    true = test_batch["y"].cpu().numpy()
    return {
        "mcc": matthews_corrcoef(true, pred),
        "accuracy": accuracy_score(true, pred),
        "f1": f1_score(true, pred, zero_division=0),
        "n_test": len(true),
        # Inner-validation scores at the CHOSEN checkpoint. Hyperparameters MUST be
        # selected on these, never on the test MCC above, or the control is tuned on
        # the same data it is evaluated on.
        "val_mcc": best_val_mcc,
        "val_loss": best_val_loss,
        # Which epoch won. If this moves between environments at a fixed configuration,
        # the selection rule is unstable -- that is the whole 2.6x-e failure in one
        # column, so it is recorded rather than inferred.
        "best_epoch": best_epoch,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", choices=["pIC50", "pKi"], default="pKi")
    p.add_argument("--thresholds", type=float, nargs="+", default=[5.0, 6.0, 7.0])
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--interactions", type=int, default=3)
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--patience", type=int, default=120)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--threads", type=int, default=0,
                   help="torch CPU threads; 0 leaves the default (set to 1-2 when "
                        "running several configurations in parallel)")
    p.add_argument("--val-every", type=int, default=1,
                   help="evaluate the validation fold every N epochs (was 5; every "
                        "epoch narrows the gap between successive candidates, which "
                        "makes the chosen checkpoint less sensitive to tiny "
                        "perturbations)")
    p.add_argument("--select-metric", choices=["loss", "mcc"], default="loss",
                   help="checkpoint selection criterion. 'loss' (default) is continuous "
                        "in the logits and matches Table S13's val/loss for the CCSA and "
                        "fine-tuning stages. 'mcc' reproduces the discarded pre-2026-11-18 "
                        "rule and must not be used for anything reported.")
    p.add_argument("--use-protein", action="store_true",
                   help="add the protein CLS embedding, matching a molecule_protein "
                        "baseline; omit for a molecule-only baseline")
    p.add_argument("--select-on", choices=["inner", "test"], default="inner",
                   help="'inner' (default) holds a separate fold out for early stopping; "
                        "'test' reproduces the published pipeline's protocol of selecting "
                        "on the evaluation fold, for a like-for-like comparison")
    p.add_argument("--tag", type=str, default="",
                   help="suffix for output filenames, to keep parallel runs separate")
    args = p.parse_args()

    device = "cpu"
    if args.threads:
        torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)

    data = load_dataset(args.endpoint, args.cutoff)

    protein = None

    if args.use_protein:

        protein = load_protein_cls(args.endpoint, data["entry_name"].unique())

        missing = set(data["entry_name"]) - set(protein)

        if missing:

            raise SystemExit(f"no protein embedding for {len(missing)} rows, "

                             f"e.g. {sorted(missing)[:3]}")

        print(f"protein branch ON: {len(protein)} entries, "

              f"dim {len(next(iter(protein.values())))}")
    geometries = data.attrs["geometries"]
    print(f"{args.endpoint}: {len(data)} measurements, "
          f"{data['molecule_id'].nunique()} compounds with geometries")
    print(f"selection: {args.select_metric} every {args.val_every} epoch(s) | "
          f"threads {torch.get_num_threads()} | torch {torch.__version__} | "
          f"host {platform.node()}")
    if args.select_metric == "mcc":
        print("  !! validation MCC selection is the discarded rule (guide 2.6x-e); "
              "nothing produced this way may be reported")

    rows = []
    for threshold in args.thresholds:
        for test_fold in sorted(data["fold_si"].unique()):
            # inner split: hold out one of the remaining folds for early stopping
            test_rows = data[data["fold_si"] == test_fold]
            if args.select_on == "test":
                # Deliberately reproduces the published pipeline's protocol, in which
                # the held-out fold is also the early-stopping set (FACTCHECK.md §1.1).
                # Needed so the control can be compared against S1 and S4C on equal
                # terms: a nested control against a non-nested baseline understates the
                # control by the selection bias, which is ~0.10 MCC.
                val_rows = test_rows
                train_rows = data[data["fold_si"] != test_fold]
            else:
                val_fold = (test_fold % 5) + 1
                val_rows = data[data["fold_si"] == val_fold]
                train_rows = data[~data["fold_si"].isin([test_fold, val_fold])]

            per_seed = [
                run_fold(train_rows, val_rows, test_rows, geometries,
                         threshold, args, device, seed, protein)
                for seed in args.seeds
            ]
            rec = {
                "endpoint": args.endpoint,
                "threshold": threshold,
                "fold": int(test_fold),
                "n_test": per_seed[0]["n_test"],
                "mcc": float(np.mean([r["mcc"] for r in per_seed])),
                "mcc_sd_seeds": float(np.std([r["mcc"] for r in per_seed], ddof=1)),
                "val_mcc": float(np.mean([r["val_mcc"] for r in per_seed])),
                "val_loss": float(np.mean([r["val_loss"] for r in per_seed])),
                "accuracy": float(np.mean([r["accuracy"] for r in per_seed])),
                "f1": float(np.mean([r["f1"] for r in per_seed])),
                # Provenance, so a results file can never again be compared against one
                # produced under a different rule or a different machine without it
                # being obvious (guide 2.6x-e).
                "select_metric": args.select_metric,
                "val_every": args.val_every,
                "best_epochs": "|".join(str(r["best_epoch"]) for r in per_seed),
                "threads": torch.get_num_threads(),
                "torch": torch.__version__,
                "host": platform.node(),
            }
            rows.append(rec)
            print(f"  theta={threshold:g} fold {test_fold}: "
                  f"MCC {rec['mcc']:.3f} (+/-{rec['mcc_sd_seeds']:.3f})")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"gnn_control_{args.endpoint}{args.tag}.csv", index=False)

    summary = res.groupby("threshold")[["mcc", "accuracy", "f1"]].agg(["mean", "std"])
    print()
    print(summary.round(3).to_string())
    summary.to_json(OUT / f"gnn_control_{args.endpoint}{args.tag}_summary.json", indent=2)


if __name__ == "__main__":
    main()
