"""
Generate the motif-ablation panel by graph surgery on the two parent structures.

Free ligands are cut from ``molecules/stf_31/input.xyz``; caged complexes from
``molecules/caged_PACT/input.xyz``, whose Ru(terpyridine)(biquinoline) fragment is
never touched -- only the STF-31 part is edited, so the spectator ligands cannot
reintroduce the size confound the original derivative panel suffered from.

Functional groups are located by connectivity pattern, not by hard-coded index,
so the same rules find them in both parents:

    sulfonyl S       S bonded to {C, N, O, O}
    sulfonamide N    N bonded to {C, H, S}
    amide carbonyl C C bonded to {C, N, O} with the O terminal
    amide N          N bonded to {C, C, H}
    tert-butyl C     C bonded to four C
    pyridine N       free:  N bonded to exactly {C, C}
                     caged: the Ru-bound N inside the STF-31 fragment

Geometries are built to be chemically sane, not converged -- the inference
pipeline runs GFN2-xTB at fmax 0.05 before featurizing. New bond lengths use
C-H 1.09, N-C 1.47 A and tetrahedral angles.

Usage
-----
    python generate_structures.py
"""

import itertools
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.io import read, write

HERE = Path(__file__).resolve().parent
MOLECULES = HERE.parent / "molecules"

CH = 1.09          # C-H
NC = 1.47          # N-C(methyl)
TET = np.deg2rad(109.47)


# ---------------------------------------------------------------- connectivity
def adjacency(atoms, scale: float = 1.25) -> np.ndarray:
    numbers = atoms.get_atomic_numbers()
    pos = atoms.get_positions()
    radii = np.array([covalent_radii[z] for z in numbers])
    dist = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    cut = (radii[:, None] + radii[None, :]) * scale
    np.fill_diagonal(cut, 0.0)
    return dist < cut


def neighbours(adj, i):
    return list(np.where(adj[i])[0])


def unit(v):
    return v / np.linalg.norm(v)


def locate(atoms):
    """Find every motif this panel edits. Returns a dict of atom indices."""
    adj = adjacency(atoms)
    sym = atoms.get_chemical_symbols()
    kinds = lambda i: sorted(sym[j] for j in neighbours(adj, i))
    found = {}

    for i, s in enumerate(sym):
        if s == "S" and kinds(i) == ["C", "N", "O", "O"]:
            found["sulfonyl_S"] = i
            found["sulfonyl_O"] = [j for j in neighbours(adj, i) if sym[j] == "O"]
        elif s == "N" and kinds(i) == ["C", "H", "S"]:
            found["sulfonamide_N"] = i
        elif s == "N" and kinds(i) == ["C", "C", "H"]:
            found["amide_N"] = i
        elif s == "C" and kinds(i) == ["C", "N", "O"]:
            oxy = [j for j in neighbours(adj, i)
                   if sym[j] == "O" and adj[j].sum() == 1]
            if oxy:
                found["amide_C"] = i
                found["amide_O"] = oxy[0]
        elif s == "C" and kinds(i) == ["C", "C", "C", "C"]:
            found["tbu_C"] = i

    # The STF-31 pyridine N: bare {C,C} when free; when caged it also binds Ru,
    # so pick the Ru-bound N that sits in the fragment carrying the sulfonyl S.
    ru = [i for i, s in enumerate(sym) if s == "Ru"]
    if not ru:
        found["pyridine_N"] = next(
            i for i, s in enumerate(sym) if s == "N" and kinds(i) == ["C", "C"]
        )
    else:
        cut = adj.copy()
        for i in ru:
            cut[i, :] = False
            cut[:, i] = False
        seen, stack, frag = set(), [found["sulfonyl_S"]], set()
        while stack:
            j = stack.pop()
            if j in seen:
                continue
            seen.add(j)
            frag.add(j)
            stack += [k for k in np.where(cut[j])[0] if k not in seen]
        found["pyridine_N"] = next(
            i for i in frag
            if sym[i] == "N" and sorted(sym[j] for j in neighbours(adj, i)) == ["C", "C", "Ru"]
        )
        found["Ru"] = ru[0]

    # The tert-butyl's aryl attachment point, needed when the group is removed.
    if "tbu_C" in found:
        q = found["tbu_C"]
        found["tbu_aryl_C"] = next(
            j for j in neighbours(adj, q)
            if sum(1 for k in neighbours(adj, j) if sym[k] == "H") == 0
        )
    return found


# ------------------------------------------------------------------- geometry
def sp3_completions(centre, kept, n_new, dist):
    """Directions completing a tetrahedral centre that has ``kept`` neighbours."""
    us = [unit(p - centre) for p in kept]
    if len(us) == 3 and n_new == 1:
        return [centre + unit(-sum(us)) * dist]
    if len(us) == 2 and n_new == 2:
        bis = unit(-(us[0] + us[1]))
        perp = unit(np.cross(us[0], us[1]))
        half = TET / 2
        return [centre + (bis * np.cos(half) + perp * np.sin(half)) * dist,
                centre + (bis * np.cos(half) - perp * np.sin(half)) * dist]
    if len(us) == 2 and n_new == 1:
        return [centre + unit(-(us[0] + us[1])) * dist]
    raise ValueError(f"unsupported completion: {len(us)} kept, {n_new} new")


def methyl(anchor, direction):
    """Positions of a methyl carbon and its three hydrogens."""
    c = anchor + unit(direction) * NC
    back = unit(anchor - c)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, back)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = unit(np.cross(back, ref))
    e2 = np.cross(back, e1)
    hs = []
    for k in range(3):
        a = 2 * np.pi * k / 3
        d = back * np.cos(TET) + (e1 * np.cos(a) + e2 * np.sin(a)) * np.sin(TET)
        hs.append(c + unit(d) * CH)
    return c, hs


def rebuild(atoms, drop, add, retype=None):
    """New Atoms with ``drop`` removed, ``add`` [(symbol, pos)] appended."""
    keep = [i for i in range(len(atoms)) if i not in set(drop)]
    sym = list(atoms.get_chemical_symbols())
    for i, s in (retype or {}).items():
        sym[i] = s
    pos = atoms.get_positions()
    out = Atoms(symbols=[sym[i] for i in keep], positions=[pos[i] for i in keep])
    for s, p in add:
        out.append(s)
        out.positions[-1] = p
    return out


def group_atoms(atoms, root, blocked):
    """All atoms reachable from ``root`` without passing through ``blocked``."""
    adj = adjacency(atoms)
    seen, stack = {blocked}, [root]
    got = set()
    while stack:
        j = stack.pop()
        if j in seen:
            continue
        seen.add(j)
        got.add(j)
        stack += [k for k in np.where(adj[j])[0] if k not in seen]
    return got


# --------------------------------------------------------------------- edits
def edit_M1(atoms, g):                      # pyridin-3-yl -> phenyl
    adj = adjacency(atoms)
    n = g["pyridine_N"]
    pos = atoms.get_positions()
    ring = [j for j in neighbours(adj, n) if atoms.get_chemical_symbols()[j] == "C"]
    h = sp3_completions(pos[n], [pos[j] for j in ring], 1, CH)
    return rebuild(atoms, [], [("H", h[0])], retype={n: "C"})


def edit_M2(atoms, g):                      # C(=O)NH -> CH2NH
    adj = adjacency(atoms)
    c, o = g["amide_C"], g["amide_O"]
    pos = atoms.get_positions()
    kept = [pos[j] for j in neighbours(adj, c) if j != o]
    return rebuild(atoms, [o], [("H", p) for p in sp3_completions(pos[c], kept, 2, CH)])


def edit_M3(atoms, g):                      # amide N-H -> N-CH3
    return _n_methylate(atoms, g["amide_N"])


def edit_M5(atoms, g):                      # sulfonamide N-H -> N-CH3
    return _n_methylate(atoms, g["sulfonamide_N"])


def _n_methylate(atoms, n):
    adj = adjacency(atoms)
    sym = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    h = next(j for j in neighbours(adj, n) if sym[j] == "H")
    c, hs = methyl(pos[n], pos[h] - pos[n])
    return rebuild(atoms, [h], [("C", c)] + [("H", p) for p in hs])


def edit_M4(atoms, g):                      # SO2NH -> CH2NH
    adj = adjacency(atoms)
    s, oxy = g["sulfonyl_S"], g["sulfonyl_O"]
    pos = atoms.get_positions()
    kept = [pos[j] for j in neighbours(adj, s) if j not in oxy]
    return rebuild(atoms, oxy,
                   [("H", p) for p in sp3_completions(pos[s], kept, 2, CH)],
                   retype={s: "C"})


def edit_M6(atoms, g):                      # tert-butyl -> H
    q, aryl = g["tbu_C"], g["tbu_aryl_C"]
    pos = atoms.get_positions()
    drop = group_atoms(atoms, q, aryl)
    h = pos[aryl] + unit(pos[q] - pos[aryl]) * CH
    return rebuild(atoms, drop, [("H", h)])


EDITS = {
    "M1_pyridineN_to_CH":       (edit_M1, "pyridin-3-yl -> phenyl (isosteric)"),
    "M2_amideCO_to_CH2":        (edit_M2, "amide carbonyl removed"),
    "M3_amideNH_to_NMe":        (edit_M3, "amide N-H methylated"),
    "M4_sulfonyl_to_CH2":       (edit_M4, "sulfonyl removed"),
    "M5_sulfonamideNH_to_NMe":  (edit_M5, "sulfonamide N-H methylated"),
    "M6_tBu_to_H":              (edit_M6, "tert-butyl removed"),
}
# M1 deletes the only atom that binds Ru, so it has no caged form.
CAGED_SKIP = {"M1_pyridineN_to_CH"}


def sanity(atoms, name):
    """Reject anything with an impossible contact before it reaches xTB."""
    pos = atoms.get_positions()
    d = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    lo = d.min()
    if lo < 0.75:
        raise SystemExit(f"{name}: atoms {lo:.2f} A apart - geometry is broken")
    return lo


def main() -> None:
    parents = {"free": read(MOLECULES / "stf_31" / "input.xyz"),
               "caged": read(MOLECULES / "caged_PACT" / "input.xyz")}

    print(f"{'file':42s} {'atoms':>6s} {'formula':<22s} {'min d':>6s}  edit")
    print("-" * 104)
    for form, parent in parents.items():
        groups = locate(parent)
        for tag, (fn, note) in EDITS.items():
            if form == "caged" and tag in CAGED_SKIP:
                continue
            out = fn(parent.copy(), groups)
            name = f"{form}_{tag}"
            lo = sanity(out, name)
            (MOLECULES / name).mkdir(exist_ok=True)
            write(MOLECULES / name / "input.xyz", out, format="xyz")
            print(f"{name:42s} {len(out):6d} {out.get_chemical_formula():<22s} "
                  f"{lo:6.2f}  {note}")

    print("\ncharges: free = 0, caged = +2")


if __name__ == "__main__":
    main()
