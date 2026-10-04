"""
Build the fragment and photoproduct series of the motif-ablation panel.

Two families are added to ``molecules/``:

Fragments — pieces of STF-31 and the spectator ligands, each presented as a
real neutral molecule. Cuts are placed on C-C or C-N single bonds only and both
ends are capped with hydrogen, so no functional group is split: the sulfonamide,
the amide and the pyridine each end up whole in exactly one fragment. The
warhead and tail halves reconstruct STF-31 exactly (plus the two caps).

Free terpyridine and biquinoline are lifted straight out of ``caged_PACT`` and
need no capping, since they were only ever bonded to Ru. They are the panel's
control: neither has any reason to inhibit NAMPT, so if the model scores them
active the fragment readout is not reporting chemistry.

Photoproducts — what each caged complex becomes after releasing its drug.
Built from ``deriv_4`` / ``deriv_5`` / ``deriv_8`` by deleting the STF-31
fragment and placing an aqua ligand on the vacated coordination site, so each
pairs with a caged complex already in the panel and the release contrast is
measured at four points rather than one.

Structures inherit their parent's optimized geometry and are not converged --
the inference pipeline runs GFN2-xTB before featurizing.

Usage
-----
    python build_molecules.py [--dry-run]
"""

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.io import read, write

BASE = Path(__file__).resolve().parent.parent
MOL = BASE / "molecules"

CH = 1.09       # C-H cap
RU_O = 2.15     # Ru-OH2
OH = 0.97       # O-H
HOH = np.deg2rad(104.5)


def adjacency(atoms, scale: float = 1.25) -> np.ndarray:
    z = atoms.get_atomic_numbers()
    p = atoms.get_positions()
    r = np.array([covalent_radii[n] for n in z])
    d = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
    cut = (r[:, None] + r[None, :]) * scale
    np.fill_diagonal(cut, 0.0)
    return d < cut


def nbrs(adj, i):
    return list(np.where(adj[i])[0])


def unit(v):
    return v / np.linalg.norm(v)


def reachable(adj, start, blocked):
    seen, stack, got = {blocked}, [start], set()
    while stack:
        j = stack.pop()
        if j in seen:
            continue
        seen.add(j)
        got.add(j)
        stack += [k for k in np.where(adj[j])[0] if k not in seen]
    return got


def subset(atoms, idx, extra=()):
    pos, sym = atoms.get_positions(), atoms.get_chemical_symbols()
    idx = sorted(idx)
    out = Atoms(symbols=[sym[i] for i in idx], positions=[pos[i] for i in idx])
    for s, p in extra:
        out.append(s)
        out.positions[-1] = p
    return out


def cut_bond(atoms, a, b, keep):
    """Break the a-b bond, keep the side containing ``keep``, cap it with H."""
    adj = adjacency(atoms)
    other = b if keep == a else a
    pos = atoms.get_positions()
    cap = pos[keep] + unit(pos[other] - pos[keep]) * CH
    return subset(atoms, reachable(adj, keep, other), [("H", cap)])


def locate_drug(atoms):
    """Anchor atoms of STF-31, by connectivity rather than index."""
    adj = adjacency(atoms)
    sym = atoms.get_chemical_symbols()
    kinds = lambda i: sorted(sym[j] for j in nbrs(adj, i))
    g = {}
    for i, s in enumerate(sym):
        if s == "S" and kinds(i) == ["C", "N", "O", "O"]:
            g["S"] = i
        elif s == "N" and kinds(i) == ["C", "H", "S"]:
            g["N_sulfonamide"] = i
        elif s == "N" and kinds(i) == ["C", "C", "H"]:
            g["N_amide"] = i
        elif s == "C" and kinds(i) == ["C", "N", "O"] and any(
                sym[j] == "O" and adj[j].sum() == 1 for j in nbrs(adj, i)):
            g["C_amide"] = i
    g["C_ch2"] = next(i for i in nbrs(adj, g["N_sulfonamide"])
                      if sym[i] == "C"
                      and sum(1 for j in nbrs(adj, i) if sym[j] == "H") == 2)
    g["C_aryl_ch2"] = next(j for j in nbrs(adj, g["C_ch2"])
                           if sym[j] == "C"
                           and not any(sym[k] == "H" for k in nbrs(adj, j)))
    g["C_aryl_S"] = next(j for j in nbrs(adj, g["S"]) if sym[j] == "C")
    return g


def split_ru(atoms):
    """Fragments of a Ru complex, keyed by role."""
    adj = adjacency(atoms)
    sym = atoms.get_chemical_symbols()
    ru = sym.index("Ru")
    cut = adj.copy()
    cut[ru, :] = False
    cut[:, ru] = False
    seen, frags = {ru}, []
    for i in range(len(sym)):
        if i in seen:
            continue
        comp, stack = set(), [i]
        while stack:
            j = stack.pop()
            if j in seen:
                continue
            seen.add(j)
            comp.add(j)
            stack += [k for k in np.where(cut[j])[0] if k not in seen]
        frags.append(comp)
    drug = next((f for f in frags if any(sym[i] == "S" for i in f)), None)
    return ru, drug, [f for f in frags if f is not drug]


def aqua(atoms, ru, donor):
    """An H2O placed where ``donor`` used to coordinate."""
    pos = atoms.get_positions()
    axis = unit(pos[donor] - pos[ru])
    o = pos[ru] + axis * RU_O
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, axis)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e = unit(np.cross(axis, ref))
    half = HOH / 2
    return [("O", o),
            ("H", o + (axis * np.cos(half) + e * np.sin(half)) * OH),
            ("H", o + (axis * np.cos(half) - e * np.sin(half)) * OH)]


def build_fragments():
    stf = read(MOL / "stf_31" / "optimized.xyz")
    caged = read(MOL / "caged_PACT" / "optimized.xyz")
    g = locate_drug(stf)
    out = {}

    # Split STF-31 at the benzylic CH2-aryl bond.
    out["frag_warhead_half"] = (
        cut_bond(stf, g["C_ch2"], g["C_aryl_ch2"], g["C_aryl_ch2"]), 0,
        "stf_31", "amide + pyridine + central ring")
    out["frag_tail_half"] = (
        cut_bond(stf, g["C_ch2"], g["C_aryl_ch2"], g["C_ch2"]), 0,
        "stf_31", "sulfonamide + tert-butylphenyl")
    # Terminal groups.
    out["frag_aminopyridine"] = (
        cut_bond(stf, g["N_amide"], g["C_amide"], g["N_amide"]), 0,
        "stf_31", "terminal: nicotinamide mimic")
    out["frag_tertbutylbenzene"] = (
        cut_bond(stf, g["S"], g["C_aryl_S"], g["C_aryl_S"]), 0,
        "stf_31", "terminal: lipophilic anchor")

    # Spectator ligands, straight out of the caged complex.
    _, _, spectators = split_ru(caged)
    sym = caged.get_chemical_symbols()
    for f in spectators:
        n_n = sum(1 for i in f if sym[i] == "N")
        name = "frag_terpyridine" if n_n == 3 else "frag_biquinoline"
        out[name] = (subset(caged, f), 0, "caged_PACT",
                     "CONTROL: spectator ligand, no expected NAMPT activity")
    return out


def build_photoproducts():
    out = {}
    sources = {
        "photo_phenanthroline": ("deriv_4", "Ru(terpy)(phenanthroline)(H2O)"),
        "photo_bipyridine": ("deriv_5", "Ru(terpy)(bipyridine)(H2O)"),
        "photo_pyridines": ("deriv_8", "Ru(pyridine)5(H2O)"),
    }
    for name, (src, note) in sources.items():
        atoms = read(MOL / src / "optimized.xyz")
        adj = adjacency(atoms)
        ru, drug, _ = split_ru(atoms)
        donor = next(i for i in drug if adj[i, ru])
        keep = [i for i in range(len(atoms)) if i not in drug]
        frag = subset(atoms, keep, aqua(atoms, ru, donor))
        out[name] = (frag, 2, src, f"photoproduct of {src}: {note}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    built = {}
    for name, spec in build_fragments().items():
        built[name] = ("fragment",) + spec
    for name, spec in build_photoproducts().items():
        built[name] = ("photoproduct",) + spec

    print(f"{'molecule':26s} {'atoms':>5s} {'formula':<18s} {'q':>2s} {'min d':>6s}  role")
    print("-" * 108)
    for name, (series, atoms, charge, parent, note) in built.items():
        p = atoms.get_positions()
        d = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        lo = float(d.min())
        if lo < 0.75:
            raise SystemExit(f"{name}: closest contact {lo:.2f} A - broken geometry")
        print(f"{name:26s} {len(atoms):5d} {atoms.get_chemical_formula():<18s} "
              f"{charge:2d} {lo:6.2f}  {note}")

        if not args.dry_run:
            d_out = MOL / name
            d_out.mkdir(parents=True, exist_ok=True)
            write(d_out / "input.xyz", atoms, format="xyz")
            sym = atoms.get_chemical_symbols()
            (d_out / "meta.json").write_text(json.dumps({
                "name": name, "series": series, "parent": parent, "motif": note,
                "charge": charge, "formula": atoms.get_chemical_formula(),
                "n_atoms": len(sym),
                "n_heavy": sum(1 for s in sym if s != "H"),
                "source": f"built by scripts/build_molecules.py from {parent}",
            }, indent=2) + "\n")

    stf = len(read(MOL / "stf_31" / "optimized.xyz"))
    w = len(built["frag_warhead_half"][1])
    t = len(built["frag_tail_half"][1])
    print(f"\nreconstruction: warhead {w} + tail {t} = {w + t} = stf_31 {stf} + 2 caps  "
          f"{'OK' if w + t == stf + 2 else 'MISMATCH'}")

    for name in ("photo_phenanthroline", "photo_bipyridine", "photo_pyridines"):
        a = built[name][1]
        adj = adjacency(a)
        ru = a.get_chemical_symbols().index("Ru")
        print(f"{name}: Ru is {int(adj[ru].sum())}-coordinate")


if __name__ == "__main__":
    main()
