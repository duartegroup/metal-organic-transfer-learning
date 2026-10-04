# Caged compound structures

Optimized 3D structures of the 130 Ru complexes in the caged dataset
(`data/excel/caged/{pic50,pk}/caged_{pic50,pk}.csv`, joined on `compound_n`),
in two formats:

- `sdf/compound_N.sdf`: MDL molfile (V2000) with bond table
- `pdb/compound_N.pdb`: HETATM records with CONECT bond records

Coordinates are the geometries used to compute the molecular descriptors
(GFN2-xTB, optimized with ORCA: `! XTB2 OPT`, `! DDCOSMO(water)`, `! TightOpt`).
The atom order matches the atom order stored in the HDF5 datasets.

Bonds follow the drawn input structures. Metal-ligand bonds are written as
single bonds; every Ru-heteroatom (N, O, S, P, Se, halogen) contact within
1.15x the sum of covalent radii is bonded. Ru-arene and Ru-cyclopentadienyl
coordination is represented by the bonds of the drawn structure.
Formal charges and bond orders around the metal are therefore not
chemically exact, and the files should be read without sanitization
(e.g. `Chem.MolFromMolFile(path, sanitize=False, removeHs=False)` in RDKit).

Two optimized structures differ from their drawn input:

- compound 34: NH2 on C35 in place of H, and a formyl H on the Ru-bound C32
- compound 59: no H on the Ru-bound cyclopentadienyl carbon C27
