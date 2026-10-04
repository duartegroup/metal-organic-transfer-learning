#!/usr/bin/env python
"""Check the ORCA/DDCOSMO geometries, then promote them to optimized.xyz.

Reports, for every molecule, the ruthenium coordination and fragment count of
the drawn input, the current optimized.xyz (if any) and the new ORCA geometry,
so a dissociated donor or split complex is caught before it is used. Promotion is
refused for any molecule whose composition drifted, or whose ORCA run did not
converge unless it is named in ACCEPTED_UNCONVERGED with a reason.

The geometry being replaced is overwritten, not archived.

    python scripts/promote_orca_geometries.py            # report only
    python scripts/promote_orca_geometries.py --promote  # report, then write
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from ase.data import covalent_radii
from ase.io import read

ROOT = Path(__file__).resolve().parent.parent
MOLECULES = ROOT / "molecules"

# A donor this far from the metal is no longer bonded to it. Ru-N and Ru-O in
# these complexes sit at 1.9-2.2 A; the gas-phase run pushed several past 3 A.
BOND_MAX = 2.6

# Molecules promoted without ORCA's convergence banner. Each is a deliberate,
# named exception, not a loosened threshold -- the input files still carry
# TightOpt, so the protocol still matches the training set for every molecule.
# They are printed as "accepted unconverged" in the table and listed again
# after it, so nothing here is silent.
ACCEPTED_UNCONVERGED = {
    "caged_PACT": (
        "limit cycle on a soft torsion: energy flat to 0.009 kcal/mol over 415 "
        "cycles, RMS gradient below tolerance (2.0e-5 vs 3.0e-5), only the two "
        "step-size criteria failing with MAX step pinned at 0.0154 A. A restart "
        "from a fresh Hessian (stages/stage1 -> stage2) reproduced the stall "
        "cycle for cycle, so the geometry is converged in energy and force and "
        "the residue is torsional wander"
    ),
}


def n_fragments(atoms, scale: float = 1.3) -> int:
    pos = atoms.get_positions()
    radii = np.array([covalent_radii[z] for z in atoms.get_atomic_numbers()])
    dist = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    cutoff = scale * (radii[:, None] + radii[None, :])
    np.fill_diagonal(cutoff, 0.0)
    adjacency = dist < cutoff

    seen: set[int] = set()
    count = 0
    for start in range(len(atoms)):
        if start in seen:
            continue
        count += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(np.flatnonzero(adjacency[node]).tolist())
    return count


def ru_donors(atoms) -> list[tuple[str, float]]:
    """Ru-N and Ru-O distances, nearest first. Empty when there is no metal."""
    symbols = atoms.get_chemical_symbols()
    if "Ru" not in symbols:
        return []
    ru = symbols.index("Ru")
    dist = np.linalg.norm(atoms.get_positions() - atoms.get_positions()[ru], axis=1)
    donors = [
        (symbols[j], dist[j])
        for j in np.argsort(dist)
        if symbols[j] in ("N", "O") and dist[j] < 4.0
    ]
    return donors[:7]


def describe(atoms) -> str:
    donors = ru_donors(atoms)
    if not donors:
        return f"{n_fragments(atoms)} frag"
    bound = sum(1 for _, d in donors if d <= BOND_MAX)
    worst = max(d for _, d in donors)
    return f"{n_fragments(atoms)} frag  {bound} bound  max {worst:.2f}"


def converged(out_file: Path) -> bool:
    if not out_file.exists():
        return False
    return "THE OPTIMIZATION HAS CONVERGED" in out_file.read_text(errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", action="store_true", help="write optimized.xyz")
    args = parser.parse_args()

    with (MOLECULES / "index.csv").open() as f:
        rows = list(csv.DictReader(f))

    header = f"{'compound':32s} {'q':>2s} | {'input':22s} | {'in use now':22s} | {'DDCOSMO(water)':22s}"
    print(header)
    print("-" * len(header))

    ready, blocked = [], []
    for row in rows:
        name = row["name"]
        mol_dir = MOLECULES / name
        orca_xyz = mol_dir / "orca" / f"{name}.xyz"
        orca_out = mol_dir / "orca" / f"{name}.out"

        inp = read(mol_dir / "input.xyz")
        # Whatever optimized.xyz currently holds -- on a first run the structure
        # from an earlier protocol, on a re-run the ORCA geometry itself. Shown
        # so the promotion can be seen to change something, or nothing.
        cur = read(mol_dir / "optimized.xyz") if (mol_dir / "optimized.xyz").exists() else None
        cur_col = describe(cur) if cur is not None else "-- none --"

        if not orca_xyz.exists():
            print(f"{name:32s} {row['charge']:>2s} | {describe(inp):22s} | {cur_col:22s} | -- not run --")
            blocked.append((name, "no ORCA geometry"))
            continue

        new = read(orca_xyz)
        reason = None
        note = None
        if sorted(new.get_chemical_symbols()) != sorted(inp.get_chemical_symbols()):
            reason = f"composition changed ({new.get_chemical_formula()})"
        elif not converged(orca_out):
            if name in ACCEPTED_UNCONVERGED:
                note = "accepted unconverged"
            else:
                reason = "did not converge"

        print(f"{name:32s} {row['charge']:>2s} | {describe(inp):22s} | {cur_col:22s} | {describe(new):22s}"
              + (f"  <== {reason or note}" if (reason or note) else ""))

        (blocked if reason else ready).append((name, reason or note))

    print(f"\n{len(ready)} ready to promote, {len(blocked)} blocked")
    for name, reason in blocked:
        print(f"  blocked: {name} -- {reason}")
    for name, _ in ready:
        if name in ACCEPTED_UNCONVERGED:
            print(f"  accepted unconverged: {name} -- {ACCEPTED_UNCONVERGED[name]}")

    if not args.promote:
        print("\nReport only. Re-run with --promote to write optimized.xyz.")
        return
    if blocked:
        print("\nRefusing to promote while any molecule is blocked.")
        return

    stale = 0
    for name, _ in ready:
        mol_dir = MOLECULES / name
        shutil.copy2(mol_dir / "orca" / f"{name}.xyz", mol_dir / "optimized.xyz")
        if (mol_dir / "predictions").is_dir():
            stale += 1

    print(f"\nPromoted {len(ready)} geometries into optimized.xyz")
    print(f"{stale} prediction sets were computed from the replaced geometry and are now")
    print("stale -- re-run scripts/run_inference.sh --force before reading any of them")


if __name__ == "__main__":
    main()
