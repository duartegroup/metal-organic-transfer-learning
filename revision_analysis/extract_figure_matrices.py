"""Rebuild the exact feature matrices behind manuscript Figures 3, 4 and S6.

This mirrors `aggregate_descriptors_by_identifier` in
`data/data_comparison/dataset_split/clustering_affinity_distribution.py`
line for line, so the stability statistics describe the published figures rather than a
plausible reconstruction of them. Three details matter and are easy to get wrong:

  * Rows are unique **compound-protein** identifiers, not compounds and not proteins.
    The same protein's CLS vector therefore repeats across every compound tested
    against it. "134 compounds / 62 proteins" describes the study, not the scatter.
  * Pooling differs per descriptor: MACE is mean-pooled over atoms, the protein vector
    is the CLS token `[0, 0]`, and the pocket vector is mean-pooled over P2Rank residues.
  * Replicate markers (`compound_12_A_MLCK-7`) are stripped and the replicates averaged.

Usage:
    python extract_figure_matrices.py --h5-dir ../caged
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

OUT = Path("results/figure_matrices")

AFFINITY_FILES = {"pic50": "caged_pic50_dataset.h5", "pki": "caged_pk_dataset.h5"}


def base_identifier(name: str) -> str:
    """Strip an A_/B_ replicate marker, exactly as the figure script does."""
    parts = name.split("_", 2)
    if len(parts) < 2:
        return name
    ident = f"{parts[0]}_{parts[1]}"
    if len(parts) > 2:
        protein_part = parts[2]
        if (len(protein_part) > 1 and protein_part[1] == "_"
                and "A" <= protein_part[0] <= "Z"):
            protein_part = protein_part[2:]
        ident = f"{ident}_{protein_part}"
    return ident


def protein_name(identifier: str) -> str:
    """`compound_10_MLCK-182` -> `MLCK`, as count_proteins_in_clusters does."""
    parts = identifier.split("_", 2)
    if len(parts) < 3:
        return identifier
    p = parts[2]
    return p.rsplit("-", 1)[0] if "-" in p else p


def extract(h5_path: Path):
    groups: dict[str, list[str]] = {}
    with h5py.File(h5_path, "r") as f:
        names = [k for k in f.keys() if "mol_mace_embedding" in f[k]]
        for n in names:
            groups.setdefault(base_identifier(n), []).append(n)

        idents = sorted(groups)
        mace, prot, pock, aff = [], [], [], []
        for ident in idents:
            members = groups[ident]
            mace.append(np.mean([np.mean(f[n]["mol_mace_embedding"][...], axis=0)
                                 for n in members], axis=0))
            prot.append(np.mean([f[n]["protein_embeddings"][...][0, 0]
                                 for n in members], axis=0))
            pock.append(np.mean([np.mean(f[n]["pocket_embeddings"][...], axis=0)
                                 for n in members], axis=0))
            aff.append(float(np.mean([f[n]["target_value"][()] for n in members])))

    return (idents, np.asarray(mace), np.asarray(prot),
            np.asarray(pock), np.asarray(aff), len(names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5-dir", default="../caged")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    report = {}

    for aff_type, fname in AFFINITY_FILES.items():
        path = Path(args.h5_dir) / fname
        if not path.exists():
            raise SystemExit(f"missing {path}")
        idents, mace, prot, pock, aff, n_raw = extract(path)

        proteins = [protein_name(i) for i in idents]
        compounds = ["_".join(i.split("_", 2)[:2]) for i in idents]

        for tag, mat in (("mace", mace), ("protein", prot), ("pocket", pock)):
            np.save(OUT / f"{tag}_{aff_type}.npy", mat)
        pd.DataFrame({"identifier": idents, "compound": compounds,
                      "protein": proteins, "affinity": aff}).to_csv(
            OUT / f"rows_{aff_type}.csv", index=False)

        report[aff_type] = {
            "raw_hdf5_groups": n_raw,
            "unique_identifiers_plotted": len(idents),
            "unique_compounds": len(set(compounds)),
            "unique_proteins": len(set(proteins)),
            "mace_shape": list(mace.shape),
            "protein_shape": list(prot.shape),
            "pocket_shape": list(pock.shape),
        }
        print(f"[{aff_type}] {n_raw} groups -> {len(idents)} plotted rows "
              f"({len(set(compounds))} compounds, {len(set(proteins))} proteins); "
              f"mace {mace.shape}, protein {prot.shape}, pocket {pock.shape}")

    (OUT / "extraction_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
