# Caged compound structures

Optimized 3D structures of the 134 Ru complexes in the caged dataset in two formats:

- `sdf/compound_N.sdf`: MDL molfile (V2000) with bond table
- `pdb/compound_N.pdb`: HETATM records with CONECT bond records

Coordinates are the geometries used to compute the molecular descriptors
(GFN2-xTB, optimized with ORCA: `! XTB2 OPT`, `! DDCOSMO(water)`, `! TightOpt`).
The atom order matches the atom order stored in the HDF5 datasets.

