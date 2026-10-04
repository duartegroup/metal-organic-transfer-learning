"""Draw 2D structures for every molecule in ../molecules, in the house ChemDraw
style of results/extrapolation_test/complexes_1-30.png:

* the metal is a shaded blue sphere labelled "Ru";
* the octahedron is drawn with four in-plane bonds plus one bold wedge (toward
  the viewer) and one hashed wedge (away from the viewer);
* the overall charge sits on a corner bracket at the top right;
* serif labels, bold compound name underneath.

The 3D arrangement is taken from the optimized geometry, not invented: the six
donors are matched to the six drawing positions by a proper rotation of the
real octahedron, so which donors are trans to each other and the handedness
(which ligand comes toward the viewer) are those of the computed structure.

For each molecule this writes, into ../structures/:
    <name>.cdxml  -- editable in ChemDraw (wedges, sphere, bracket included)
    <name>.mol    -- MDL molfile with the same 2D coordinates and wedges
    <name>.svg / <name>.png -- rendered depiction
plus overview.svg / overview.png, a grid of all structures.

Bonds are perceived from optimized.xyz (falling back to input.xyz). RDKit has
no valence model for Ru, so ligands are perceived as neutral fragments and the
metal is reattached to every donor atom (N/O/S/P) within DONOR_CUTOFF of it.
"""

import itertools
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor, rdDetermineBonds, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
MOLECULES = HERE.parent / "molecules"
OUT = HERE.parent / "structures"

DONOR_CUTOFF = 2.5  # Angstrom, Ru-donor bond
DONORS = {"N", "O", "S", "P"}
CDXML_BOND = 14.4  # ChemDraw default bond length in points
MIN_GAP = 1.3  # closest allowed approach of two ligands, in 2D units (bond = 1.5)
MAX_PUSH = 2.5  # furthest a ligand is slid outward to clear a clash
BOND_CLEAR = 0.6  # how close a metal-donor bond may pass to another ligand's atom
FONT = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSerif.ttf"
SERIF = "'DejaVu Serif', 'Times New Roman', Times, serif"

# Drawing positions around the metal: (name, 2D angle in degrees, bond style),
# listed as trans pairs (axis +, axis -).
#   "X": the layout of the reference complexes -- two donors up, two down, a
#        wedge to the left and a hash to the right (cf. compound 20).
#   "T": for a meridional tridentate (terpy-like) ligand, which is drawn flat
#        in the paper plane: left / top / right, the fourth in-plane donor at
#        the bottom, and the wedge/hash pair below on either side.
TEMPLATES = {
    "X": [(("TL", 120, "plain"), ("BR", 300, "plain")),
          (("TR", 60, "plain"), ("BL", 240, "plain")),
          (("L", 180, "wedge"), ("R", 0, "hash"))],
    "T": [(("Top", 90, "plain"), ("Bottom", 270, "plain")),
          (("Right", 0, "plain"), ("Left", 180, "plain")),
          (("W", 215, "wedge"), ("H", 325, "hash"))],
}


# --------------------------------------------------------------------------
# Structure perception
# --------------------------------------------------------------------------
def read_xyz(path: Path) -> list[tuple[str, float, float, float]]:
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    atoms = []
    for line in lines[2 : 2 + n]:
        s, x, y, z = line.split()[:4]
        atoms.append((s, float(x), float(y), float(z)))
    return atoms


def _perceive(atoms, charge: int) -> Chem.Mol:
    body = "\n".join(f"{s} {x} {y} {z}" for s, x, y, z in atoms)
    fd, tmp = tempfile.mkstemp(suffix=".xyz")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(f"{len(atoms)}\n\n{body}\n")
        mol = Chem.MolFromXYZFile(tmp)
    finally:
        os.unlink(tmp)
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    return mol


def build_mol(atoms, charge: int) -> Chem.Mol:
    """Molecule with its 3D conformer kept (needed for the stereo assignment)."""
    metal_idx = [i for i, a in enumerate(atoms) if a[0] == "Ru"]
    if not metal_idx:
        return Chem.RemoveHs(_perceive(atoms, charge))

    ligand_atoms = [a for a in atoms if a[0] != "Ru"]
    # every ligand in this panel is neutral, so the complex charge sits on the metal
    lig = _perceive(ligand_atoms, 0)
    rw = Chem.RWMol(lig)
    conf = rw.GetConformer()
    for i in metal_idx:
        s, x, y, z = atoms[i]
        ru = Chem.Atom("Ru")
        ru.SetFormalCharge(charge // len(metal_idx))
        ru.SetNoImplicit(True)
        ri = rw.AddAtom(ru)
        conf.SetAtomPosition(ri, (x, y, z))
        for j, (sj, xj, yj, zj) in enumerate(ligand_atoms):
            if sj not in DONORS:
                continue
            d = ((x - xj) ** 2 + (y - yj) ** 2 + (z - zj) ** 2) ** 0.5
            if d < DONOR_CUTOFF:
                rw.AddBond(j, ri, Chem.BondType.DATIVE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return Chem.RemoveHs(mol)


def _ligands(mol: Chem.Mol, ru: int) -> list[list[int]]:
    """Connected components of the molecule with the metal removed."""
    seen, comps = {ru}, []
    for start in range(mol.GetNumAtoms()):
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            i = stack.pop()
            comp.append(i)
            for nb in mol.GetAtomWithIdx(i).GetNeighbors():
                if nb.GetIdx() not in seen:
                    seen.add(nb.GetIdx())
                    stack.append(nb.GetIdx())
        comps.append(sorted(comp))
    return comps


# --------------------------------------------------------------------------
# Layout of a metal complex
# --------------------------------------------------------------------------
def _ligand_shape(mol: Chem.Mol, ru: int, lig: list[int]) -> np.ndarray:
    """2D coords of one ligand drawn as a chelate around the metal at the
    origin (so chelating donors come out syn). Rows follow ``lig``."""
    sub = Chem.RWMol()
    idx = {}
    for i in lig + [ru]:
        a = mol.GetAtomWithIdx(i)
        na = Chem.Atom(a.GetAtomicNum())
        na.SetNoImplicit(True)
        na.SetNumExplicitHs(a.GetTotalNumHs())
        na.SetIsAromatic(a.GetIsAromatic())
        idx[i] = sub.AddAtom(na)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in idx and j in idx:
            dative = b.GetBondType() == Chem.BondType.DATIVE
            sub.AddBond(idx[i], idx[j], Chem.BondType.SINGLE if dative else b.GetBondType())
            if b.GetIsAromatic():
                sub.GetBondBetweenAtoms(idx[i], idx[j]).SetIsAromatic(True)
    sub = sub.GetMol()
    sub.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(sub)
    rdDepictor.SetPreferCoordGen(False)  # regular chelate rings around the metal
    rdDepictor.Compute2DCoords(sub)
    pos = sub.GetConformer().GetPositions()[:, :2]
    # normalise to the 1.5 bond length the drawing positions assume
    bl = np.median([np.linalg.norm(pos[b.GetBeginAtomIdx()] - pos[b.GetEndAtomIdx()])
                    for b in sub.GetBonds()])
    pos = pos * (1.5 / bl)
    return pos[[idx[i] for i in lig]] - pos[idx[ru]]


def _rot(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def _unit2(deg: float) -> np.ndarray:
    return np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])


def _slot_vec3(angle: float, style: str) -> np.ndarray:
    """Rough 3D direction a drawing position stands for (z toward viewer)."""
    z = {"plain": 0.0, "wedge": 0.8, "hash": -0.8}[style]
    return np.array([*_unit2(angle), z])


def _trans_pairs(vecs: dict[int, np.ndarray]) -> list[tuple[int, int]]:
    """Group the donors into trans pairs. A vacant site (5-coordinate metal)
    gets a virtual donor, id -1, opposite its unpaired neighbour."""
    ids = list(vecs)
    pairs, used = [], set()
    for i in ids:
        if i in used:
            continue
        j = min((k for k in ids if k != i and k not in used),
                key=lambda k: vecs[i] @ vecs[k], default=None)
        if j is not None and vecs[i] @ vecs[j] < -0.7:
            pairs.append((i, j))
            used |= {i, j}
    for i in ids:
        if i not in used:
            vecs[-1] = -vecs[i]
            pairs.append((i, -1))
    if len(pairs) != 3:
        raise ValueError("not an octahedral (or square-pyramidal) metal centre")
    return pairs


def _assignments(vecs, pairs, template):
    """All donor -> drawing-position maps that are proper rotations of the real
    octahedron, i.e. that keep trans pairs trans and preserve handedness."""
    ref = [_slot_vec3(*template[k][0][1:]) for k in range(3)]
    ref_sign = np.sign(np.linalg.det(np.array(ref)))
    for perm in itertools.permutations(range(3)):
        for flips in itertools.product((0, 1), repeat=3):
            plus = [pairs[perm[k]][flips[k]] for k in range(3)]
            if np.sign(np.linalg.det(np.array([vecs[d] for d in plus]))) != ref_sign:
                continue
            slot = {}
            for k in range(3):
                a, b = pairs[perm[k]][flips[k]], pairs[perm[k]][1 - flips[k]]
                slot[a], slot[b] = template[k]
            yield slot


def _place_chelate(shape: np.ndarray, donor_rows: list[int], dirs: np.ndarray):
    """Rigidly place a multidentate ligand (rotation, translation, optional
    mirror; no distortion of its rings) so its donors sit on the given bond
    directions. The metal-donor distance R is free -- as in the hand-drawn
    complexes, chelate bonds may be drawn longer than ring bonds -- with a
    small cost for every bit of extra length. Returns (cost, coords)."""
    best = (np.inf, None)
    for mirror in (False, True):
        sh = shape * ([1, -1] if mirror else [1, 1])
        p = sh[donor_rows]
        pc = p - p.mean(axis=0)
        for R in np.arange(1.5, 4.01, 0.1):
            q = R * dirs
            qc = q - q.mean(axis=0)
            theta = np.arctan2((pc[:, 0] * qc[:, 1] - pc[:, 1] * qc[:, 0]).sum(), (pc * qc).sum())
            rot = _rot(theta)
            xy = sh @ rot.T + (q.mean(axis=0) - p.mean(axis=0) @ rot.T)
            cost = ((xy[donor_rows] - q) ** 2).sum() + 0.25 * (R - 1.5)
            # the ligand body must sit beyond its donors, never over the metal
            body = np.delete(xy, donor_rows, axis=0)
            dmin = np.linalg.norm(xy[donor_rows], axis=1).min()
            cost += 5.0 * max(0.0, dmin + 0.3 - np.linalg.norm(body, axis=1).min())
            if cost < best[0]:
                best = (cost, xy)
    return best


def _place_mono(shape: np.ndarray, donor_row: int, angle: float, mirror: bool) -> np.ndarray:
    """Monodentate ligand: donor on the bond at one bond length, body pointing
    straight outward."""
    sh = shape * ([1, -1] if mirror else [1, 1])
    c = sh.mean(axis=0) - sh[donor_row]
    theta = np.radians(angle) - np.arctan2(c[1], c[0])
    xy = (sh - sh[donor_row]) @ _rot(theta).T
    return xy + 1.5 * _unit2(angle)


def _clash(a: np.ndarray, b: np.ndarray, ra: np.ndarray, rb: np.ndarray) -> float:
    """How far two ligands intrude on each other (0 = clear). Labelled atoms
    (heteroatoms, NH3, OH2 ...) get a larger radius than bare carbons."""
    d = np.linalg.norm(a[:, None] - b[None], axis=-1)
    return float(np.clip(ra[:, None] + rb[None] - d, 0, None).max(initial=0.0))


def _resolve(placed, radii, out_dir, sizes):
    """Slide the smaller ligand of each clashing pair outward along its
    bond(s), at most MAX_PUSH each. Returns (placed, remaining clash, push)."""
    placed = list(placed)
    pushed = [0.0] * len(placed)
    for _ in range(int(MAX_PUSH / 0.1) * len(placed)):
        moved = False
        for a, b in itertools.combinations(range(len(placed)), 2):
            if _clash(placed[a], placed[b], radii[a], radii[b]) == 0.0:
                continue
            for k in sorted((a, b), key=lambda q: (sizes[q], pushed[q])):
                if pushed[k] < MAX_PUSH:
                    placed[k] = placed[k] + 0.1 * out_dir[k]
                    pushed[k] += 0.1
                    moved = True
                    break
        if not moved:
            break
    left = max((_clash(placed[a], placed[b], radii[a], radii[b])
                for a, b in itertools.combinations(range(len(placed)), 2)), default=0.0)
    return placed, left, sum(pushed)


def _bond_crossing(placed, donor_rows) -> float:
    """How close any metal-donor bond (a segment from the origin) passes to an
    atom of another ligand (0 = clear by BOND_CLEAR)."""
    worst = 0.0
    for k, rows_k in enumerate(donor_rows):
        for r in rows_k:
            end = placed[k][r]
            L2 = end @ end
            for j, pts in enumerate(placed):
                if j == k:
                    continue
                t = np.clip(pts @ end / L2, 0.0, 1.0)
                dist = np.linalg.norm(pts - t[:, None] * end, axis=1).min()
                worst = max(worst, BOND_CLEAR - dist)
    return worst


def layout_complex(mol: Chem.Mol, ru: int):
    """Return (2D coords, {donor: bond style}) for a mononuclear complex."""
    pos3 = mol.GetConformer().GetPositions()
    ligs = _ligands(mol, ru)
    donors = sorted(a.GetIdx() for a in mol.GetAtomWithIdx(ru).GetNeighbors())
    vecs = {d: (pos3[d] - pos3[ru]) / np.linalg.norm(pos3[d] - pos3[ru]) for d in donors}
    pairs = _trans_pairs(vecs)

    lig_donors = [[d for d in donors if d in set(l)] for l in ligs]
    shapes = [_ligand_shape(mol, ru, l) for l in ligs]
    rows = [{a: r for r, a in enumerate(l)} for l in ligs]
    sizes = [len(l) for l in ligs]
    radii = [np.array([MIN_GAP / 2 + (0.35 if mol.GetAtomWithIdx(i).GetAtomicNum() != 6 else 0.0)
                       for i in l]) for l in ligs]
    donor_rows = [[rows[k][d] for d in lds] for k, lds in enumerate(lig_donors)]
    tail = max(range(len(ligs)), key=lambda k: sizes[k])  # hangs downward
    has_mer = any(len(ld) == 3 for ld in lig_donors)
    # mirror images only matter for monodentate ligands that are not symmetric
    # about their bond; trying them for every one would explode the search
    flippable = [k for k, lds in enumerate(lig_donors) if len(lds) == 1 and sizes[k] > 6]

    best = None
    for tname, template in TEMPLATES.items():
        for slot in _assignments(vecs, pairs, template):
            base, chel = 0.0, {}
            for k, lds in enumerate(lig_donors):
                if len(lds) > 1:
                    dirs = np.array([_unit2(slot[d][1]) for d in lds])
                    cost, xy = _place_chelate(shapes[k], [rows[k][d] for d in lds], dirs)
                    chel[k] = xy
                    base += cost
                    if len(lds) == 3 and any(slot[d][2] != "plain" for d in lds):
                        base += 5.0  # keep a terpy-like ligand flat in the paper
            base += 2.0 * (1 + np.sin(np.radians(slot[lig_donors[tail][0]][1])))
            if tname == "T" and not has_mer:
                base += 0.5  # the reference layout, unless a mer ligand needs T
            out_dir = []
            for lds in lig_donors:
                v = sum(_unit2(slot[d][1]) for d in lds)
                out_dir.append(v / (np.linalg.norm(v) or 1.0))
            for flips in itertools.product((False, True), repeat=len(flippable)):
                fl = dict(zip(flippable, flips))
                placed = [chel[k] if k in chel else
                          _place_mono(shapes[k], rows[k][lds[0]], slot[lds[0]][1], fl.get(k, False))
                          for k, lds in enumerate(lig_donors)]
                placed, left, push = _resolve(placed, radii, out_dir, sizes)
                cross = _bond_crossing(placed, donor_rows)
                score = base + 20.0 * left + 20.0 * cross + 0.5 * push
                if best is None or score < best[0]:
                    best = (score, slot, placed)
    _, slot, placed = best

    coords = np.zeros((mol.GetNumAtoms(), 2))
    for l, xy in zip(ligs, placed):
        coords[l] = xy
    return coords, {d: slot[d][2] for d in donors}


def depict(mol: Chem.Mol) -> Chem.Mol:
    """Kekulé 2D depiction. Metal-donor bonds become single bonds starting at
    the metal, with wedge / hash directions from the 3D geometry."""
    ru = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Ru"]
    if not ru:
        m = Chem.Mol(mol)
        m.RemoveAllConformers()
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(m)
        rdDepictor.StraightenDepiction(m)
        Chem.Kekulize(m, clearAromaticFlags=True)
        return m
    ru = ru[0]
    coords, style = layout_complex(mol, ru)

    m = Chem.RWMol(mol)
    Chem.Kekulize(m, clearAromaticFlags=True)
    for d in style:
        m.RemoveBond(d, ru)
        m.AddBond(ru, d, Chem.BondType.SINGLE)  # begin atom = metal: wedge narrow end
        m.GetBondBetweenAtoms(ru, d).SetBondDir(
            {"wedge": Chem.BondDir.BEGINWEDGE, "hash": Chem.BondDir.BEGINDASH}.get(
                style[d], Chem.BondDir.NONE))
    m = m.GetMol()
    m.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(m)
    m.RemoveAllConformers()
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, (x, y) in enumerate(coords):
        conf.SetAtomPosition(i, (float(x), float(y), 0.0))
    m.AddConformer(conf, assignId=True)
    return m


# --------------------------------------------------------------------------
# Output: SVG / PNG
# --------------------------------------------------------------------------
GRADIENT = (
    "<defs><radialGradient id='ruball' cx='38%' cy='32%' r='70%'>"
    "<stop offset='0%' stop-color='#FFFFFF'/>"
    "<stop offset='22%' stop-color='#B8E2FF'/>"
    "<stop offset='65%' stop-color='#2F96E8'/>"
    "<stop offset='100%' stop-color='#0D4E96'/>"
    "</radialGradient></defs>"
)


def _charge_label(charge: int) -> str:
    if not charge:
        return ""
    return (f"{abs(charge)}" if abs(charge) > 1 else "") + ("+" if charge > 0 else "-")


def render_svg(m: Chem.Mol, name: str, charge: int) -> str:
    drawer = rdMolDraw2D.MolDraw2DSVG(-1, -1)  # canvas sized to the molecule
    opts = drawer.drawOptions()
    opts.useBWAtomPalette()
    opts.fixedBondLength = 32
    opts.bondLineWidth = 1.6
    opts.multipleBondOffset = 0.18
    opts.padding = 0.08
    opts.addStereoAnnotation = False
    opts.prepareMolsBeforeDrawing = False
    opts.clearBackground = False
    if FONT.exists():
        opts.fontFile = str(FONT)
    ru = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "Ru"]
    for i in ru:
        opts.atomLabels[i] = " "  # replaced by the sphere below
    drawer.DrawMolecule(m)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    w = float(re.search(r"width='([\d.]+)px'", svg).group(1))
    h = float(re.search(r"height='([\d.]+)px'", svg).group(1))
    body = svg[svg.index(">", svg.index("<svg")) + 1 : svg.rindex("</svg>")]

    pts = np.array([[drawer.GetDrawCoords(i).x, drawer.GetDrawCoords(i).y]
                    for i in range(m.GetNumAtoms())])
    extras = []
    for i in ru:
        x, y = pts[i]
        r = 0.42 * opts.fixedBondLength
        extras.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r:.1f}' fill='url(#ruball)' "
            f"stroke='#0D4E96' stroke-width='0.6'/>"
            f"<text x='{x:.1f}' y='{y + 0.36 * r:.1f}' text-anchor='middle' "
            f"font-family=\"{SERIF}\" font-size='{0.95 * r:.1f}' font-weight='bold' "
            f"fill='#000'>Ru</text>"
        )
    pad_top = 0.0
    if charge:
        # corner bracket at the top right of the structure, charge as superscript
        bx, by = pts[:, 0].max() + 18, pts[:, 1].min() - 14
        pad_top = max(0.0, 22 - by)
        extras.append(
            f"<path d='M {bx - 14:.1f} {by:.1f} L {bx:.1f} {by:.1f} L {bx:.1f} {by + 16:.1f}' "
            f"fill='none' stroke='#000' stroke-width='1.4'/>"
            f"<text x='{bx + 3:.1f}' y='{by + 4:.1f}' font-family=\"{SERIF}\" "
            f"font-size='14' fill='#000'>{_charge_label(charge)}</text>"
        )
        w = max(w, bx + 34)

    shift = max(0.0, (11 * len(name) + 20 - w) / 2)  # room for the bold name
    w += 2 * shift
    legend_y = h + 28
    total_h = legend_y + 14 + pad_top
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' "
        f"version='1.1' width='{w:.0f}' height='{total_h:.0f}' "
        f"viewBox='0 0 {w:.0f} {total_h:.0f}'>{GRADIENT}"
        f"<rect width='100%' height='100%' fill='#FFFFFF'/>"
        f"<g transform='translate(0 {pad_top:.1f})'><g transform='translate({shift:.1f} 0)'>{body}{''.join(extras)}</g>"
        f"<text x='{w / 2:.1f}' y='{legend_y:.1f}' text-anchor='middle' "
        f"font-family=\"{SERIF}\" font-size='17' font-weight='bold'>{escape(name)}</text>"
        f"</g></svg>\n"
    )


def svg_to_png(svg_path: Path, png_path: Path, zoom: float = 3.0) -> None:
    subprocess.run(["rsvg-convert", "-z", str(zoom), "-b", "white", "-o", str(png_path),
                    str(svg_path)], check=True)


def overview_svg(svgs: list[str], per_row: int = 5) -> str:
    sizes = [(float(re.search(r"width='([\d.]+)'", s).group(1)),
              float(re.search(r"height='([\d.]+)'", s).group(1))) for s in svgs]
    cell_w = max(w for w, _ in sizes) + 20
    rows = [sizes[i : i + per_row] for i in range(0, len(sizes), per_row)]
    row_h = [max(h for _, h in r) + 20 for r in rows]
    parts, y = [], 0.0
    for r_idx, row in enumerate(rows):
        for c_idx, (w, h) in enumerate(row):
            k = r_idx * per_row + c_idx
            s = svgs[k].replace("ruball", f"ruball{k}")
            x = c_idx * cell_w + (cell_w - w) / 2
            parts.append(s.replace("<svg ", f"<svg x='{x:.0f}' y='{y + row_h[r_idx] - h - 10:.0f}' ", 1))
        y += row_h[r_idx]
    W = cell_w * per_row
    return (f"<svg xmlns='http://www.w3.org/2000/svg' version='1.1' width='{W:.0f}' "
            f"height='{y:.0f}' viewBox='0 0 {W:.0f} {y:.0f}'>"
            f"<rect width='100%' height='100%' fill='#FFFFFF'/>{''.join(parts)}</svg>\n")


# --------------------------------------------------------------------------
# Output: CDXML
# --------------------------------------------------------------------------
def to_cdxml(m: Chem.Mol, title: str, charge: int) -> str:
    """CDXML with Kekulé bonds, wedge/hash metal bonds, a shaded blue sphere on
    the metal and the charge bracket."""
    conf = m.GetConformer()
    scale = CDXML_BOND / 1.5
    xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                   for i in range(m.GetNumAtoms())])
    x0, y1 = xy[:, 0].min(), xy[:, 1].max()
    pos = {i: (60 + (x - x0) * scale, 60 + (y1 - y) * scale) for i, (x, y) in enumerate(xy)}

    nodes, bonds, spheres, extras = [], [], [], []
    for a in m.GetAtoms():
        i = a.GetIdx()
        x, y = pos[i]
        attrs = f'id="{1000 + i}" p="{x:.2f} {y:.2f}"'
        if a.GetAtomicNum() == 6:
            nodes.append(f"<n {attrs}/>")
            continue
        hs = a.GetTotalNumHs()
        attrs += f' Element="{a.GetAtomicNum()}" NumHydrogens="{hs}"'
        label = a.GetSymbol() + (("H" + (str(hs) if hs > 1 else "")) if hs else "")
        if a.GetSymbol() != "Ru" and a.GetFormalCharge():
            attrs += f' Charge="{a.GetFormalCharge()}"'
            label += _charge_label(a.GetFormalCharge())
        face = ' face="1"' if a.GetSymbol() == "Ru" else ""
        nodes.append(
            f'<n {attrs}><t p="{x - 3.5:.2f} {y + 3.5:.2f}" LabelJustification="Left">'
            f'<s font="3" size="10" color="0"{face}>{escape(label)}</s></t></n>'
        )
        if a.GetSymbol() == "Ru":
            r = 0.42 * CDXML_BOND
            spheres.append(
                f'<graphic id="{9000 + i}" BoundingBox="{x + r:.2f} {y:.2f} {x:.2f} {y:.2f}" '
                f'GraphicType="Oval" OvalType="Circle Shaded" color="4" '
                f'Center3D="{x:.2f} {y:.2f} 0" MajorAxisEnd3D="{x + r:.2f} {y:.2f} 0" '
                f'MinorAxisEnd3D="{x:.2f} {y + r:.2f} 0"/>'
            )
    order = {Chem.BondType.SINGLE: "1", Chem.BondType.DOUBLE: "2", Chem.BondType.TRIPLE: "3"}
    display = {Chem.BondDir.BEGINWEDGE: ' Display="WedgeBegin"',
               Chem.BondDir.BEGINDASH: ' Display="WedgedHashBegin"'}
    for b in m.GetBonds():
        bonds.append(
            f'<b id="{5000 + b.GetIdx()}" B="{1000 + b.GetBeginAtomIdx()}" '
            f'E="{1000 + b.GetEndAtomIdx()}" Order="{order.get(b.GetBondType(), "1")}"'
            f'{display.get(b.GetBondDir(), "")}/>'
        )
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if charge:
        bx, by = max(xs) + 14, min(ys) - 10
        extras.append(f'<graphic id="9900" BoundingBox="{bx - 10:.2f} {by:.2f} {bx:.2f} {by:.2f}" GraphicType="Line"/>')
        extras.append(f'<graphic id="9901" BoundingBox="{bx:.2f} {by:.2f} {bx:.2f} {by + 12:.2f}" GraphicType="Line"/>')
        extras.append(f'<t id="9902" p="{bx + 2:.2f} {by + 3:.2f}"><s font="3" size="10" color="0">'
                      f'{_charge_label(charge)}</s></t>')
    extras.append(f'<t id="9903" p="{(min(xs) + max(xs)) / 2 - 3 * len(title):.2f} {max(ys) + 28:.2f}">'
                  f'<s font="3" size="12" face="1" color="0">{escape(title)}</s></t>')
    return (
        '<?xml version="1.0" encoding="UTF-8" ?>\n'
        '<!DOCTYPE CDXML SYSTEM "http://www.cambridgesoft.com/xml/cdxml.dtd" >\n'
        f'<CDXML BondLength="{CDXML_BOND}" LabelFont="3" LabelSize="10" CaptionFont="3" CaptionSize="10">'
        '<colortable><color r="1" g="1" b="1"/><color r="0" g="0" b="0"/>'
        '<color r="1" g="0" b="0"/><color r="0.18" g="0.59" b="0.91"/></colortable>'
        '<fonttable><font id="3" charset="iso-8859-1" name="Times New Roman"/></fonttable>'
        # sphere first so the Ru label and bonds are drawn on top of it
        f'<page id="1">{"".join(spheres)}'
        f'<fragment id="2">{"".join(nodes)}{"".join(bonds)}</fragment>'
        f'{"".join(extras)}</page></CDXML>\n'
    )


# --------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(exist_ok=True)
    svgs = []
    for meta_path in sorted(MOLECULES.glob("*/meta.json")):
        d = meta_path.parent
        meta = json.loads(meta_path.read_text())
        name, charge = meta["name"], meta["charge"]
        xyz = d / "optimized.xyz"
        if not xyz.exists():
            xyz = d / "input.xyz"
        mol = build_mol(read_xyz(xyz), charge)
        m = depict(mol)

        formula = rdMolDescriptors.CalcMolFormula(mol)
        dirs = [b.GetBondDir() for b in m.GetBonds()
                if "Ru" in (b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol())]
        n_w = dirs.count(Chem.BondDir.BEGINWEDGE)
        n_h = dirs.count(Chem.BondDir.BEGINDASH)
        print(f"{name:32s} {formula:22s} Ru bonds: {len(dirs)} (wedge {n_w}, hash {n_h})")

        Chem.MolToMolFile(m, str(OUT / f"{name}.mol"), kekulize=False)
        (OUT / f"{name}.cdxml").write_text(to_cdxml(m, name, charge))
        svg = render_svg(m, name, charge)
        (OUT / f"{name}.svg").write_text(svg)
        svg_to_png(OUT / f"{name}.svg", OUT / f"{name}.png")
        svgs.append(svg)

    (OUT / "overview.svg").write_text(overview_svg(svgs))
    svg_to_png(OUT / "overview.svg", OUT / "overview.png", zoom=1.0)
    print(f"Wrote {len(svgs)} structures to {OUT}")


if __name__ == "__main__":
    main()
