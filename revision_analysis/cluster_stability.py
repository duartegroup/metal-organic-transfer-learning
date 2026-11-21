"""Inferential statistics for the Figure 3 / Figure 4 projections (Reviewer 1, comment 3).

The reviewer's objection is that the clustering in these figures is descriptive and is
read as though it were inferential: no variance explained is reported, no stability
analysis is given, and the protein-family interpretation is open to post-hoc
rationalisation. This script supplies the four things needed to answer that, for an
arbitrary feature matrix:

  1. PCA explained variance -- how much of the structure the 2-D picture actually shows.
  2. Cluster stability -- Hennig-style bootstrap Jaccard per cluster, plus ARI between
     the original partition and partitions recovered from resampled data.
  3. Cluster validity -- silhouette, and the gap statistic against a uniform null for
     choosing k honestly rather than by eye.
  4. Family association -- whether clusters align with protein family better than chance,
     by permutation. This is the one that directly addresses "post hoc rationalisation":
     without it, any visible grouping can be narrated after the fact.

Usage:
    python cluster_stability.py --features X.npy [--labels fam.csv] [--k 4] --tag proteins
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

OUT = Path("results")


# --------------------------------------------------------------------------- PCA

def pca_variance(X: np.ndarray, n_report: int = 10) -> dict:
    """Explained variance, which the manuscript currently does not report."""
    p = PCA().fit(X)
    evr = p.explained_variance_ratio_
    n_90 = int(np.searchsorted(np.cumsum(evr), 0.90) + 1)
    return {
        "pc1": float(evr[0]),
        "pc2": float(evr[1]),
        "pc1_pc2": float(evr[:2].sum()),
        "scree": [float(v) for v in evr[:n_report]],
        "cumulative": [float(v) for v in np.cumsum(evr)[:n_report]],
        "n_components_for_90pct": n_90,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }


# ------------------------------------------------------------------- stability

def bootstrap_stability(X, k, n_boot=500, seed=0):
    """Hennig's bootstrap: how reproducible is each cluster under resampling?

    For each bootstrap resample we re-cluster and, for every original cluster, record
    the maximum Jaccard overlap with any bootstrap cluster. Hennig's rule of thumb:
    mean Jaccard < 0.6 means the cluster is not trustworthy; > 0.75 means it is stable.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    base = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)

    jacc = {c: [] for c in range(k)}
    aris = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        uniq = np.unique(idx)
        boot_lab = KMeans(n_clusters=k, n_init=10,
                          random_state=int(rng.integers(1 << 31))).fit_predict(X[idx])
        # map back to original points (first occurrence of each sampled index)
        lab_by_point = {}
        for pos, orig in enumerate(idx):
            lab_by_point.setdefault(orig, boot_lab[pos])
        shared = np.array(sorted(lab_by_point))
        aris.append(adjusted_rand_score(base[shared],
                                        [lab_by_point[i] for i in shared]))
        for c in range(k):
            orig_set = set(np.where(base == c)[0]) & set(uniq)
            if not orig_set:
                jacc[c].append(0.0)
                continue
            best = 0.0
            for bc in np.unique(boot_lab):
                boot_set = {i for i in lab_by_point if lab_by_point[i] == bc}
                inter = len(orig_set & boot_set)
                union = len(orig_set | boot_set)
                if union:
                    best = max(best, inter / union)
            jacc[c].append(best)

    return {
        "k": k,
        "mean_jaccard_per_cluster": {int(c): float(np.mean(v)) for c, v in jacc.items()},
        "min_cluster_jaccard": float(min(np.mean(v) for v in jacc.values())),
        "mean_ari_vs_bootstrap": float(np.mean(aris)),
        "ari_ci95": [float(np.percentile(aris, 2.5)), float(np.percentile(aris, 97.5))],
        "n_boot": n_boot,
        "cluster_sizes": {int(c): int((base == c).sum()) for c in range(k)},
    }


# ----------------------------------------------------------------- gap statistic

def elbow_k(X, max_k=10, seed=42):
    """Replicate `choose_optimal_k_elbow` from the figure script exactly.

    The published figures pick k as the (k, inertia) point furthest from the chord
    joining the first and last points of the inertia curve, on the UNSCALED features.
    Reproducing it here is what lets the stability numbers describe the published
    partition rather than one of our own choosing.
    """
    ks = list(range(2, max_k + 1))
    inertia = [KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X).inertia_
               for k in ks]
    pts = np.column_stack([ks, inertia]).astype(float)
    line = pts[-1] - pts[0]
    line /= np.sqrt((line ** 2).sum())
    vec = pts - pts[0]
    proj = np.outer(vec @ line, line)
    dist = np.sqrt(((vec - proj) ** 2).sum(axis=1))
    return int(ks[int(np.argmax(dist))]), [float(v) for v in inertia]


def gap_statistic(X, k_max=8, n_ref=50, seed=0):
    """Tibshirani's gap statistic against a uniform null in the PCA-aligned bounding box."""
    rng = np.random.default_rng(seed)

    def dispersion(data, k):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(data)
        return np.log(km.inertia_ + 1e-12)

    lo, hi = X.min(axis=0), X.max(axis=0)
    rows = []
    for k in range(1, k_max + 1):
        obs = dispersion(X, k)
        refs = [dispersion(rng.uniform(lo, hi, size=X.shape), k) for _ in range(n_ref)]
        gap = float(np.mean(refs) - obs)
        sk = float(np.std(refs) * np.sqrt(1 + 1 / n_ref))
        rows.append({"k": k, "gap": gap, "sk": sk})

    # smallest k with gap(k) >= gap(k+1) - s(k+1)
    best = None
    for i in range(len(rows) - 1):
        if rows[i]["gap"] >= rows[i + 1]["gap"] - rows[i + 1]["sk"]:
            best = rows[i]["k"]
            break
    return {"table": rows, "k_selected": best}


# ------------------------------------------------------- family association test

def family_association(X, families, k, n_perm=5000, seed=0):
    """Do clusters track protein family better than chance?

    Answers the 'post hoc rationalisation' objection directly: a narrated grouping that
    does not beat permuted family labels is not evidence of anything.
    """
    rng = np.random.default_rng(seed)
    fam = pd.Series(families).astype("category").cat.codes.to_numpy()
    keep = fam >= 0
    Xk, famk = X[keep], fam[keep]
    lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xk)

    obs_ari = adjusted_rand_score(famk, lab)
    obs_ami = adjusted_mutual_info_score(famk, lab)
    null_ari, null_ami = [], []
    for _ in range(n_perm):
        perm = rng.permutation(famk)
        null_ari.append(adjusted_rand_score(perm, lab))
        null_ami.append(adjusted_mutual_info_score(perm, lab))
    null_ari, null_ami = np.array(null_ari), np.array(null_ami)
    return {
        "n_used": int(keep.sum()),
        "n_families": int(len(np.unique(famk))),
        "ari_observed": float(obs_ari),
        "ari_null_mean": float(null_ari.mean()),
        "ari_p": float((np.sum(null_ari >= obs_ari) + 1) / (n_perm + 1)),
        "ami_observed": float(obs_ami),
        "ami_null_mean": float(null_ami.mean()),
        "ami_p": float((np.sum(null_ami >= obs_ami) + 1) / (n_perm + 1)),
        "n_perm": n_perm,
    }


# -------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help=".npy feature matrix, rows = items")
    ap.add_argument("--labels", help="optional CSV with a 'family' column, same row order")
    ap.add_argument("--tag", required=True, help="output name, e.g. proteins / compounds")
    ap.add_argument("--k", type=int, default=0, help="0 = choose by gap statistic")
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--dedup", action="store_true",
                    help="collapse exactly duplicated rows before analysis; see note")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X_raw = np.load(args.features)
    if X_raw.ndim != 2:
        raise SystemExit(f"expected a 2-D matrix, got shape {X_raw.shape}")

    fam_series = pd.read_csv(args.labels)["family"] if args.labels else None

    # Rows of the published figures are compound-protein pairs, so a protein's CLS
    # vector repeats once per compound tested against it -- exactly, bit for bit.
    # Duplicated points always agree with themselves under resampling, which inflates
    # silhouette and bootstrap stability. --dedup reports the honest version.
    n_before = X_raw.shape[0]
    if args.dedup:
        _, keep = np.unique(np.round(X_raw, 6), axis=0, return_index=True)
        keep = np.sort(keep)
        X_raw = X_raw[keep]
        if fam_series is not None:
            fam_series = fam_series.iloc[keep].reset_index(drop=True)

    X = X_raw if args.no_scale else StandardScaler().fit_transform(X_raw)

    OUT.mkdir(parents=True, exist_ok=True)
    res = {"tag": args.tag, "scaled": not args.no_scale, "dedup": args.dedup,
           "n_rows_input": int(n_before), "n_rows_analysed": int(X.shape[0])}

    res["pca"] = pca_variance(X)
    print(f"[{args.tag}] n={X.shape[0]} d={X.shape[1]}")
    print(f"  PC1 {res['pca']['pc1']:.1%}, PC2 {res['pca']['pc2']:.1%}, "
          f"together {res['pca']['pc1_pc2']:.1%}; "
          f"{res['pca']['n_components_for_90pct']} PCs needed for 90%")

    res["gap"] = gap_statistic(X, k_max=args.k_max, seed=args.seed)
    k = args.k or res["gap"]["k_selected"] or 2
    res["k_used"] = k
    print(f"  gap statistic selects k={res['gap']['k_selected']}; using k={k}")

    lab = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit_predict(X)
    res["silhouette"] = float(silhouette_score(X, lab))
    print(f"  silhouette {res['silhouette']:.3f}")

    res["stability"] = bootstrap_stability(X, k, n_boot=args.n_boot, seed=args.seed)
    print(f"  bootstrap ARI {res['stability']['mean_ari_vs_bootstrap']:.3f} "
          f"{res['stability']['ari_ci95']}, "
          f"weakest cluster Jaccard {res['stability']['min_cluster_jaccard']:.3f}")

    if fam_series is not None:
        res["family"] = family_association(X, fam_series, k,
                                           n_perm=args.n_perm, seed=args.seed)
        print(f"  family association: ARI {res['family']['ari_observed']:.3f} "
              f"(null {res['family']['ari_null_mean']:.3f}, p={res['family']['ari_p']:.4f})")

    path = OUT / f"cluster_stability_{args.tag}.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
