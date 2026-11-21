"""Size the test-fold selection bias in the S1 arm.

FACTCHECK.md §1.1: the published pipeline uses the held-out test fold as the Lightning
validation loader, so early stopping and checkpoint selection both see the fold the
metric is finally reported on. This script measures how much that inflates the number.

Design. Rather than try to reproduce the published absolute MCCs -- which would make the
comparison hostage to every small implementation difference -- it reimplements the S1
architecture once and runs it under BOTH protocols on the same folds, same seeds, same
everything:

  protocol "published"  early stopping and checkpoint on the TEST fold (what the repo does)
  protocol "nested"     early stopping and checkpoint on an INNER fold held out of train,
                        test fold untouched until the final evaluation

The difference between the two is the bias, and it is measured within one implementation,
so it does not depend on matching Leiden's numbers.

Architecture follows the verified reading of src/transfer_learning_src/model/multitask.py:
encoder 2-layer FFN (hidden 2*embed, out 128) with PReLU and dropout; feature extractor
BatchNorm -> Linear -> PReLU -> Dropout -> Linear -> PReLU -> Dropout -> L2-normalise;
head input -> h/2 -> h/4 -> 1 with PReLU and dropout 0.4. AdamW lr 1e-4, weight decay
1e-5, BCE, batch 64, max 500 epochs, patience 35, grad clip 1.0, min-max scaling fit on
the training split only.

SMOTE is deliberately NOT applied: the champion S1 arm runs with target_smote=False, and
the hard-coded-threshold bug (FACTCHECK §1.2) is a separate question.

Usage:
    python s1_selection_bias.py --endpoint pKi --thresholds 5 6 7 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

OUT = Path("results")
H5 = {"pIC50": "../caged/caged_pic50_dataset.h5", "pKi": "../caged/caged_pk_dataset.h5"}
META = Path("../metal-organic-transfer-learning/data/excel/metadata.xlsx")


# ------------------------------------------------------------------ data

def load(endpoint: str):
    """SOAP molecular descriptors, mean-pooled over atoms (the pKi champion modality)."""
    rows, feats, ys = [], [], []
    with h5py.File(H5[endpoint], "r") as f:
        for key in sorted(f.keys()):
            g = f[key]
            if "mol_soap_embedding" not in g:
                continue
            rows.append(key)
            feats.append(np.mean(g["mol_soap_embedding"][...], axis=0))
            ys.append(float(g["target_value"][()]))
    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)

    meta = pd.read_excel(META, sheet_name="measurements")
    meta = meta[meta.activity_type == endpoint]
    fold_by_entry = dict(zip(meta.entry_name, meta.fold))
    folds = np.array([fold_by_entry.get(r, -1) for r in rows])
    keep = folds >= 0
    return X[keep], y[keep], folds[keep] + 1  # 1-indexed, as in the SI


def minmax(train: np.ndarray, *others):
    lo, hi = train.min(0), train.max(0)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    return [np.clip((a - lo) / rng, 0, 1) for a in (train, *others)]


# ----------------------------------------------------------------- model

class S1Net(nn.Module):
    def __init__(self, in_dim: int, embed: int = 128, dropout: float = 0.4):
        super().__init__()
        hidden = embed * 2
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.PReLU(),
            nn.Linear(hidden, embed), nn.PReLU(), nn.Dropout(dropout),
        )
        self.features = nn.Sequential(
            nn.BatchNorm1d(embed),
            nn.Linear(embed, embed), nn.PReLU(), nn.Dropout(dropout),
            nn.Linear(embed, embed), nn.PReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(embed, embed // 2), nn.PReLU(),
            nn.Linear(embed // 2, embed // 4), nn.PReLU(), nn.Dropout(dropout),
            nn.Linear(embed // 4, 1),
        )

    def forward(self, x):
        z = self.features(self.encoder(x))
        z = nn.functional.normalize(z, dim=1)
        return self.head(z).squeeze(-1)


def metrics(true, logits):
    pred = (logits > 0).astype(int)
    return {
        "mcc": matthews_corrcoef(true, pred),
        "accuracy": accuracy_score(true, pred),
        "f1": f1_score(true, pred, zero_division=0),
    }


def train_one(Xtr, ytr, Xsel, ysel, Xte, yte, args, seed):
    """Train once; `Xsel` is whatever the protocol uses for early stopping."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = S1Net(Xtr.shape[1], dropout=args.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()

    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    Xsel_t = torch.tensor(Xsel); Xte_t = torch.tensor(Xte)

    best, best_state, stale = np.inf, None, 0
    n = len(Xtr_t)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            sel_loss = lossf(model(Xsel_t), torch.tensor(ysel)).item()
        if sel_loss < best:
            best, stale = sel_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return metrics(yte, model(Xte_t).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="pKi", choices=["pKi", "pIC50"])
    ap.add_argument("--thresholds", type=float, nargs="+", default=[5.0, 6.0, 7.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=35)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for output files, so parallel "
                                              "per-threshold runs do not clobber each other")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    X, y, folds = load(args.endpoint)
    print(f"{args.endpoint}: {X.shape[0]} measurements, {X.shape[1]} SOAP dims")

    rows = []
    for th in args.thresholds:
        labels = (y > th).astype(np.float32)
        for test_fold in sorted(set(folds)):
            inner = (test_fold % 5) + 1
            te = folds == test_fold
            inn = folds == inner
            tr_pub = ~te                      # published: train on everything but test
            tr_nest = ~te & ~inn              # nested: inner fold held out of training

            for seed in args.seeds:
                Xa, Xb, Xc = minmax(X[tr_pub], X[te], X[te])
                pub = train_one(Xa, labels[tr_pub], Xb, labels[te],
                                Xc, labels[te], args, seed)

                Xa, Xb, Xc = minmax(X[tr_nest], X[inn], X[te])
                nest = train_one(Xa, labels[tr_nest], Xb, labels[inn],
                                 Xc, labels[te], args, seed)

                rows.append({"threshold": th, "fold": int(test_fold), "seed": seed,
                             **{f"published_{k}": v for k, v in pub.items()},
                             **{f"nested_{k}": v for k, v in nest.items()}})
            last = rows[-1]
            print(f"  th={th:g} fold {test_fold}: published MCC "
                  f"{np.mean([r['published_mcc'] for r in rows if r['threshold']==th and r['fold']==test_fold]):.3f}  "
                  f"nested MCC "
                  f"{np.mean([r['nested_mcc'] for r in rows if r['threshold']==th and r['fold']==test_fold]):.3f}")

    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / f"s1_selection_bias_{args.endpoint}{args.tag}.csv", index=False)

    print("\n=== selection bias (published protocol minus nested) ===")
    summary = {}
    for m in ("mcc", "accuracy", "f1"):
        d = df[f"published_{m}"] - df[f"nested_{m}"]
        summary[m] = {"published": float(df[f"published_{m}"].mean()),
                      "nested": float(df[f"nested_{m}"].mean()),
                      "bias": float(d.mean()),
                      "bias_sd": float(d.std(ddof=1))}
        print(f"  {m:9s} published {summary[m]['published']:.4f}  "
              f"nested {summary[m]['nested']:.4f}  bias {summary[m]['bias']:+.4f} "
              f"(sd {summary[m]['bias_sd']:.4f})")
    (OUT / f"s1_selection_bias_{args.endpoint}{args.tag}.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
