# Architecture control

A 3D message-passing network (SchNet-style continuous-filter convolutions) trained from
scratch on the metal-organic dataset alone. It replaces only the molecular representation
used by training strategy S1 and holds everything else fixed: the same five
cross-validation folds, the same three activity thresholds, the same metrics and the same
from-scratch training regime. It answers whether the descriptor-based representation is
what limits the from-scratch baseline.

It operates on the GFN2-xTB geometries rather than on perceived bonds, which avoids
RDKit's valence model — that rejects most of these complexes — and matches how the SOAP,
ACSF and MACE descriptors were computed, so the comparison isolates the architecture.

## Running it

```bash
python gnn_control/gnn_control.py --endpoint pKi
python gnn_control/gnn_control.py --endpoint pIC50 --thresholds 6
```

Reads `data/excel/metadata.xlsx` and `data/caged_structures/sdf/`, both in this
repository. The pIC50 variant also needs the per-residue ESM-C embeddings, which live in
the HDF5 datasets:

```bash
python data/hf5_dataset/caged/create_caged_h5.py     # writes data/hf5_dataset/caged/*.h5
```

Those files are not in the repository (`*.h5` is gitignored). Pass `--h5-dir` if you keep
them elsewhere.

Results are written to `results/gnn_control/`. `--help` lists the architecture and
training options; the configuration used in the paper is given in the Supporting
Information.
