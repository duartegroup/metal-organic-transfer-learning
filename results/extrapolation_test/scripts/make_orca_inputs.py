#!/usr/bin/env python
"""Write ORCA inputs that reproduce the training set's geometry protocol exactly.

The training-set geometries were optimized with ORCA using

    ! XTB2 OPT
    ! DDCOSMO(water)
    ! TightOpt

A gas-phase optimization is not equivalent: in a test run every monodentate
neutral donor came off the ruthenium -- H2O in uncaged_PACT_H2O and
photo_pyridines dissociated outright (Ru-O 1.99 -> 3.04 A and 2.15 -> 3.08 A),
the other two aqua complexes stretched to ~2.42 A, and the Ru-NH3 set
deriv_10..15 stretched to 2.6-3.4 A. Nothing that is chelated moved. Without a
dielectric to screen the +2 metal there is nothing holding a neutral donor on.

Inputs start from input.xyz.

    python scripts/make_orca_inputs.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MOLECULES = ROOT / "molecules"

# Same keywords as the ORCA setup that produced the training geometries.
METHOD = "XTB2"
SOLVENT = "water"
MULTIPLICITY = 1

# The one deliberate departure from the training inputs, which used 48. This is
# a resource setting, not a method setting: it changes how the work is divided
# between cores, not what is computed. XTB2 scales poorly past a handful of
# cores anyway -- a 118-atom complex took 3m15s on 48 -- so the panel runs as a
# 39-way job array at 16 cores each instead, which finishes in minutes rather
# than hours and costs fewer core-hours. All 39 use the same value, so nothing
# within the panel is compared across different settings.
NPROCS = 16


def write_input(name: str, charge: int) -> Path:
    mol_dir = MOLECULES / name
    atoms = (mol_dir / "input.xyz").read_text().splitlines()
    n_atoms = int(atoms[0].split()[0])
    body = atoms[2 : 2 + n_atoms]

    orca_dir = mol_dir / "orca"
    orca_dir.mkdir(exist_ok=True)
    inp = orca_dir / f"{name}.inp"

    with inp.open("w") as f:
        f.write(f"! {METHOD} OPT\n")
        f.write(f"! DDCOSMO({SOLVENT})\n")
        f.write("! TightOpt\n\n")
        f.write("%geom\n")
        f.write("  MaxIter 1500\n")
        f.write("end\n")
        f.write("%output\n")
        f.write("  PrintLevel Huge\n")
        f.write("  Print[ P_Mayer ] 1  # default = on\n")
        f.write("end\n\n")
        f.write("%scf\n")
        f.write("   Convergence Extreme\n")
        f.write("end\n\n")
        f.write(f"%pal nprocs {NPROCS} end\n\n")
        f.write(f"* xyz {int(charge)} {int(MULTIPLICITY)}\n")
        for line in body:
            sym, x, y, z = line.split()[:4]
            f.write(f"{sym} {float(x):.6f} {float(y):.6f} {float(z):.6f}\n")
        f.write("*\n")
    return inp


def main() -> None:
    with (MOLECULES / "index.csv").open() as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        name, charge = row["name"], int(row["charge"])
        inp = write_input(name, charge)
        n = int((MOLECULES / name / "input.xyz").read_text().split("\n", 1)[0])
        print(f"{name:32s} charge {charge:+d}  {n:3d} atoms  -> {inp.relative_to(ROOT)}")

    print(f"\n{len(rows)} ORCA inputs written")


if __name__ == "__main__":
    main()
