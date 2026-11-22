#!/usr/bin/env python3
"""Shrink an organic/caged HDF5 dataset without losing anything the paper uses.

By default this makes exactly ONE change: it truncates `protein_embeddings`.
BindingAffinitiesDataset._compute_fixed_vectors takes `prot[0]` -- the ESM CLS
token -- and discards every other residue row, and the paper defines no other
protein descriptor ("Protein descriptor: Global" throughout Table 1). The full
per-residue array is ~53% of each file.

ALL THREE molecular descriptors are kept. SOAP in particular is the descriptor
used by the S1 and TL champions in Table 1, and ACSF appears in SI Tables S22,
S23, S36 and S37, so neither can be dropped if the paper is to be reproducible.
`--drop-descriptors` exists for completeness and should normally stay unused.

Everything works by exclusion, so any key not named in DROP survives untouched.

NOTE the two-row floor. The dataset calls `.squeeze()` before indexing, so an
array stored with a single row collapses from (1, 1152) to (1152,) and `prot[0]`
then returns a *scalar* instead of the embedding -- silently wrong rather than a
crash. Keeping two rows preserves the shape; the second row is never read.

Verified on both caged files: 1256 MB -> 597 MB and 1128 MB -> 577 MB, with the
model inputs rebuilt through BindingAffinitiesDataset bit-for-bit identical for
SOAP, ACSF and MACE, along with the pocket vectors and affinities.

Usage:
    python strip_h5.py in.h5 out.h5                  # what you want (~2.1x)
    python strip_h5.py in.h5 out.h5 --keep-protein   # copy, no reduction
"""
from __future__ import annotations

import argparse

import h5py

DROP = {
    "mol_soap_embedding",
    "mol_acsf_embedding",
    "mol_mbtr_embedding",
    "protein_attentions",
}

PROTEIN_KEY = "protein_embeddings"
PROTEIN_ROWS = 2  # see the note above: one row would collapse under .squeeze()


def strip(
    src_path: str,
    dst_path: str,
    truncate_protein: bool,
    drop_descriptors: bool = False,
) -> None:
    kept = dropped = truncated = 0

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for attr, value in src.attrs.items():
            dst.attrs[attr] = value

        for entry_name in src:
            entry = src[entry_name]

            # Top-level datasets (e.g. `entry_names`) are copied as they are.
            if not isinstance(entry, h5py.Group):
                src.copy(entry, dst, name=entry_name)
                continue

            group = dst.create_group(entry_name)
            for attr, value in entry.attrs.items():
                group.attrs[attr] = value

            for key in entry:
                if drop_descriptors and key in DROP:
                    dropped += 1
                    continue

                node = entry[key]
                if (
                    truncate_protein
                    and key == PROTEIN_KEY
                    and isinstance(node, h5py.Dataset)
                    and node.ndim >= 2
                    and node.shape[-2] > PROTEIN_ROWS
                ):
                    group.create_dataset(
                        key, data=node[..., :PROTEIN_ROWS, :], dtype=node.dtype
                    )
                    truncated += 1
                    continue

                src.copy(node, group, name=key)
                kept += 1

    print(
        f"{src_path} -> {dst_path}: kept {kept}, dropped {dropped}, "
        f"truncated {truncated}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument(
        "--keep-protein",
        action="store_true",
        help="keep full per-residue protein embeddings (no reduction at all)",
    )
    parser.add_argument(
        "--drop-descriptors",
        action="store_true",
        help="ALSO drop SOAP/ACSF/MBTR. Only safe if you will never re-run the "
             "published champions or the SI descriptor tables, which use SOAP.",
    )
    args = parser.parse_args()
    strip(
        args.src,
        args.dst,
        truncate_protein=not args.keep_protein,
        drop_descriptors=args.drop_descriptors,
    )


if __name__ == "__main__":
    main()
