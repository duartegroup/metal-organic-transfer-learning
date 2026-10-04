#!/usr/bin/env python
"""Build the shareable handout: one row per compound x threshold.

Emits results/panel_handout.csv and results/panel_handout.html, the latter a
self-contained file (no external requests) that can be mailed as an attachment.

    python scripts/export_handout.py
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
MOLECULES = ROOT / "molecules"
RESULTS = ROOT / "results"
THRESHOLDS = (5, 6, 7)

# --------------------------------------------------------------------------
# Expected activity.
#
# There is NO measured activity for these compounds: 38 of 39 were drawn or
# generated for this study and NAMPT is not in the training set. The only
# entries below are ones with a stated, checkable basis. Everything else is
# deliberately left blank rather than guessed.
#
#   literature - reported in the literature
#   control    - designed negative control of this study
#   chemistry  - contains no part of the pharmacophore; near-certain
#   hypothesis - the study's own premise, under test, NOT ground truth
# --------------------------------------------------------------------------
ALL = {5: False, 6: False, 7: False}
EXPECTED: dict[str, tuple[dict[int, bool | None], str, str]] = {
    "stf_31": (
        {5: True, 6: True, 7: True},
        "literature",
        "Established NAMPT inhibitor: STF-31, first reported as a GLUT1 inhibitor, was "
        "subsequently identified as acting through NAMPT (Adams et al., 2014). Reported "
        "potency is nanomolar, which clears all three thresholds. Confirm the citation "
        "and value against your own source before relying on it.",
    ),
    "caged_PACT": (
        dict(ALL),
        "hypothesis",
        "The prodrug design premise, not a measurement. Ru coordinates the pyridine N "
        "that NAMPT inhibitors rely on, so the caged complex is intended to be inactive "
        "until photorelease. This is the hypothesis the study tests, so treat agreement "
        "here as weaker evidence than the controls.",
    ),
    "uncaged_PACT": (
        dict(ALL),
        "chemistry",
        "Ru photoproduct containing no STF-31 whatsoever.",
    ),
    "uncaged_PACT_H2O": (
        dict(ALL),
        "chemistry",
        "Ru aqua photoproduct containing no STF-31 whatsoever.",
    ),
    "photo_bipyridine": (
        dict(ALL),
        "chemistry",
        "Ru photoproduct of deriv_5; contains no STF-31.",
    ),
    "photo_phenanthroline": (
        dict(ALL),
        "chemistry",
        "Ru photoproduct of deriv_4; contains no STF-31.",
    ),
    "photo_pyridines": (
        dict(ALL),
        "chemistry",
        "Ru photoproduct of deriv_8; contains no STF-31.",
    ),
    "frag_terpyridine": (
        dict(ALL),
        "control",
        "Ancillary ligand alone, carrying no part of the STF-31 pharmacophore. Included "
        "as a designed negative control: if this scores like the drug, the model is not "
        "reading chemistry at that threshold.",
    ),
    "frag_biquinoline": (
        dict(ALL),
        "control",
        "Ancillary ligand alone, carrying no part of the STF-31 pharmacophore. Included "
        "as a designed negative control: if this scores like the drug, the model is not "
        "reading chemistry at that threshold.",
    ),
    "frag_tertbutylbenzene": (
        dict(ALL),
        "chemistry",
        "Isolated lipophilic fragment. No pharmacophore and no plausible route to NAMPT "
        "inhibition at any of these thresholds.",
    ),
    "frag_aminopyridine": (
        dict(ALL),
        "chemistry",
        "Isolated 3-aminopyridine. Nicotinamide-like, but far too small to inhibit at "
        "any of these thresholds.",
    ),
    "frag_tail_half": (
        {5: None, 6: None, 7: False},
        "chemistry",
        "Half of the inhibitor (sulfonamide + tert-butylphenyl). A fragment of this size "
        "is not expected to retain the parent's nanomolar potency, so an expectation is "
        "set at pIC50>7 only; at the weaker thresholds its true activity is genuinely "
        "unknown.",
    ),
    "frag_warhead_half": (
        {5: None, 6: None, 7: False},
        "chemistry",
        "Half of the inhibitor (amide + pyridine + central ring). A fragment of this size "
        "is not expected to retain the parent's nanomolar potency, so an expectation is "
        "set at pIC50>7 only; at the weaker thresholds its true activity is genuinely "
        "unknown.",
    ),
}
NO_BASIS = (
    "Novel structure generated for this study. No measured or literature activity "
    "exists, and none is asserted here."
)

# One line per threshold, shown beside the computed tally, saying what the tally
# means -- because at a threshold that has collapsed to one class the tally reads
# backwards. Every figure in it is computed from the panel: a hand-typed number
# here survived one change of geometry protocol while being wrong, so none of
# them are hand-typed any more.
def reading_line(t: int, compounds: list[dict]) -> str:
    n = len(compounds)
    p = {c["name"]: c["predictions"][t] for c in compounds}
    active = sum(1 for v in p.values() if v["predicted"])
    scored = [v for v in p.values() if v["expected"] is not None]
    agree = sum(1 for v in scored if v["predicted"] == v["expected"])

    stf = p["stf_31"]["p"]
    terpy, biq = p["frag_terpyridine"]["p"], p["frag_biquinoline"]["p"]
    tbu = p["frag_tertbutylbenzene"]["p"]

    if active == n:
        return (
            f"Calls all {n} compounds active — free terpyridine {terpy:.2f}, "
            f"tert-butylbenzene {tbu:.2f}. It matches {agree} of {len(scored)} stated "
            "expectations, and does so by accident."
        )
    if active == 0:
        return (
            f"{agree} of {len(scored)} correct, and it means nothing: the model predicts "
            f"inactive for all {n} compounds, STF-31 included ({stf:.2f}). That tally is an "
            "artifact of a majority-class predictor, not evidence the threshold works."
        )
    return (
        f"{agree} of {len(scored)} stated expectations met, including STF-31 active at "
        f"{stf:.2f} while the spectator-ligand controls sit at {terpy:.2f} and {biq:.2f}. "
        f"{active} of {n} compounds called active — the only threshold reading chemistry."
    )

SERIES_LABEL = {
    "reaction": "reaction species",
    "spectator": "spectator derivative",
    "ablation_free": "ablation (free ligand)",
    "ablation_caged": "ablation (caged complex)",
    "fragment": "fragment",
    "photoproduct": "photoproduct",
}


# --------------------------------------------------------------------------
# Structures
# --------------------------------------------------------------------------
def read_xyz(path: Path) -> list[tuple[str, str, str, str]]:
    """Extended-XYZ tolerant reader: keeps only symbol + coordinates."""
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    return [tuple(line.split()[:4]) for line in lines[2 : 2 + n]]


def _smiles_from_atoms(atoms, charge: int) -> str:
    body = "\n".join(" ".join(a) for a in atoms)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".xyz", delete=False) as fh:
            fh.write(f"{len(atoms)}\n\n{body}\n")
            tmp = fh.name
        mol = Chem.MolFromXYZFile(tmp)
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    finally:
        if tmp:
            os.unlink(tmp)


def smiles_for(atoms, charge: int, formula: str) -> tuple[str, str]:
    """Return (smiles, caveat). RDKit has no valence model for Ru, so metal
    complexes are perceived ligand-by-ligand with the metal reattached as an ion."""
    if "Ru" not in formula:
        return _smiles_from_atoms(atoms, charge), ""
    ligands = [a for a in atoms if a[0] != "Ru"]
    n_ru = len(atoms) - len(ligands)
    # every ligand in this panel is neutral, so the complex charge sits on the metal
    smi = _smiles_from_atoms(ligands, 0)
    metal = ".".join([f"[Ru+{charge}]"] * n_ru) if charge else ".".join(["[Ru]"] * n_ru)
    return f"{metal}.{smi}", (
        "Ru-ligand dative bonds are not representable in SMILES: written as the metal "
        "ion plus disconnected neutral ligands. Use the XYZ for anything geometric."
    )


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def build() -> list[dict]:
    compounds = []
    for meta_path in sorted(MOLECULES.glob("*/meta.json")):
        d = meta_path.parent
        meta = json.loads(meta_path.read_text())
        xyz_path = d / "optimized.xyz"
        # The protocol the training set used: ORCA XTB2 / DDCOSMO(water) /
        # TightOpt. Naming the solvent matters -- optimizing these complexes in
        # the gas phase lets every monodentate neutral donor leave the
        # ruthenium, so "GFN2-xTB" alone does not identify a usable geometry.
        # A structure with no orca/ directory was not produced that way and is
        # labelled as unknown rather than silently claiming the protocol.
        geometry = "GFN2-xTB, ORCA/DDCOSMO(water)"
        if not (d / "orca").is_dir():
            geometry = "GFN2-xTB, protocol unrecorded"
        if not xyz_path.exists():
            xyz_path, geometry = d / "input.xyz", "as built (not optimized)"
        atoms = read_xyz(xyz_path)
        smi, smi_note = smiles_for(atoms, meta["charge"], meta["formula"])

        exp_map, basis_type, basis = EXPECTED.get(
            meta["name"], ({t: None for t in THRESHOLDS}, "none", NO_BASIS)
        )

        preds = {}
        for t in THRESHOLDS:
            p = json.loads((d / "predictions" / f"th{t}.json").read_text())["prediction"]
            folds = p.get("fold_probabilities") or []
            mean = sum(folds) / len(folds) if folds else None
            sd = (
                (sum((f - mean) ** 2 for f in folds) / (len(folds) - 1)) ** 0.5
                if len(folds) > 1
                else None
            )
            preds[t] = {
                "p": p["ensemble_probability"],
                "predicted": bool(p["binary_prediction"]),
                "folds": folds,
                "fold_sd": sd,
                "n_models": p.get("n_models_used"),
                "expected": exp_map.get(t),
            }

        compounds.append(
            {
                "name": meta["name"],
                "series": meta["series"],
                "series_label": SERIES_LABEL.get(meta["series"], meta["series"]),
                "parent": meta["parent"],
                "motif": meta["motif"],
                "formula": meta["formula"],
                "charge": meta["charge"],
                "n_atoms": meta["n_atoms"],
                "n_heavy": meta["n_heavy"],
                "smiles": smi,
                "smiles_note": smi_note,
                "geometry": geometry,
                "xyz": "\n".join(" ".join(a) for a in atoms),
                "basis_type": basis_type,
                "basis": basis,
                "predictions": preds,
            }
        )
    return compounds


CSV_COLUMNS = [
    "threshold",
    "compound",
    "series",
    "parent",
    "motif_changed",
    "formula",
    "charge",
    "n_atoms",
    "n_heavy",
    "smiles",
    "smiles_caveat",
    "geometry",
    "activity_expected",
    "activity_expected_basis_type",
    "activity_expected_basis",
    "activity_predicted",
    "value_predicted",
    "agrees_with_expectation",
    "fold_sd",
    "fold_min",
    "fold_max",
    "fold_1",
    "fold_2",
    "fold_3",
    "fold_4",
    "fold_5",
]


def csv_rows(compounds) -> list[list]:
    rows = []
    for t in THRESHOLDS:
        for c in sorted(compounds, key=lambda c: (c["series"], c["name"])):
            p = c["predictions"][t]
            exp = p["expected"]
            folds = p["folds"]
            rows.append(
                [
                    t,
                    c["name"],
                    c["series"],
                    c["parent"] or "",
                    c["motif"] or "",
                    c["formula"],
                    c["charge"],
                    c["n_atoms"],
                    c["n_heavy"],
                    c["smiles"],
                    c["smiles_note"],
                    c["geometry"],
                    "" if exp is None else str(exp).upper(),
                    c["basis_type"],
                    c["basis"],
                    str(p["predicted"]).upper(),
                    f"{p['p']:.6f}",
                    "" if exp is None else str(exp == p["predicted"]).upper(),
                    f"{p['fold_sd']:.6f}" if p["fold_sd"] is not None else "",
                    f"{min(folds):.6f}" if folds else "",
                    f"{max(folds):.6f}" if folds else "",
                    *[f"{f:.6f}" for f in folds] + [""] * (5 - len(folds)),
                ]
            )
    return rows


def write_csv(compounds, path: Path) -> None:
    import csv

    # LF, not the csv module's default CRLF, so this file is byte-identical to the
    # one the HTML page's own "Download CSV" button produces
    with path.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(CSV_COLUMNS)
        w.writerows(csv_rows(compounds))


# The same six compounds scripts/analyse.py fits its size axis through, so the
# page and the analysis cannot quote different slopes. They vary only the
# ancillary ligands around an intact STF-31 and Ru, which is what makes the fit
# pure composition; the later derivatives change donor type as well and would
# not measure the same thing.
SPECTATOR_FIT = ["caged_PACT", "deriv_1", "deriv_2", "deriv_3", "deriv_4", "deriv_5"]


def size_fit(compounds, t: int):
    """Least-squares fit of prediction against heavy-atom count over the
    spectator-only members -- their edits cannot matter chemically, so the
    slope is composition alone."""
    from scipy.stats import linregress

    pts = [
        (c["n_heavy"], c["predictions"][t]["p"])
        for c in compounds
        if c["name"] in SPECTATOR_FIT
    ]
    fit = linregress([x for x, _ in pts], [y for _, y in pts])
    return fit.slope, fit.rvalue, fit.pvalue, len(pts)


def caveat_numbers(compounds) -> dict[str, str]:
    """The figures quoted in the fixed prose, recomputed from the panel."""
    by_name = {c["name"]: c for c in compounds}

    ranges = sorted(
        max(c["predictions"][7]["folds"]) - min(c["predictions"][7]["folds"])
        for c in compounds
    )
    median_range = ranges[len(ranges) // 2]
    folds = by_name["caged_PACT"]["predictions"][7]["folds"]

    # The drug, its two halves, and the compounds that share none of its
    # pharmacophore: if threshold 6 could discriminate, these would not overlap.
    six = [
        by_name[n]["predictions"][6]["p"]
        for n in (
            "stf_31",
            "frag_tail_half",
            "frag_warhead_half",
            "frag_tertbutylbenzene",
            "frag_terpyridine",
            "frag_biquinoline",
        )
    ]

    sentences = []
    for t in THRESHOLDS:
        slope, r, pv, _ = size_fit(compounds, t)
        verdict = "significant" if pv < 0.05 else "not significant"
        sentences.append(
            f"at pIC50&nbsp;&gt;&nbsp;{t} the slope is {slope:+.4f} per heavy atom "
            f"(r&nbsp;=&nbsp;{r:+.2f}, p&nbsp;=&nbsp;{pv:.2f}, {verdict})"
        )

    return {
        "__FOLD_SPREAD_7__": f"{median_range:.2f}",
        "__CAGED_FOLDS_7__": ", ".join(f"{f:.3f}" for f in folds[:-1])
        + f" and {folds[-1]:.3f}",
        "__TH6_SPAN__": f"{max(six) - min(six):.4f}",
        "__SIZE_FIT__": "; ".join(sentences) + ".",
    }


def write_html(compounds, path: Path) -> None:
    payload = json.dumps(
        {
            "columns": CSV_COLUMNS,
            "compounds": compounds,
            "thresholds": list(THRESHOLDS),
            "reading": {str(t): reading_line(t, compounds) for t in THRESHOLDS},
        },
        separators=(",", ":"),
    ).replace("</", "<\\/")
    html = (Path(__file__).parent / "handout_template.html").read_text()
    html = html.replace("/*__DATA__*/null", payload)
    for token, value in caveat_numbers(compounds).items():
        assert token in html, f"template lost its {token} placeholder"
        html = html.replace(token, value)
    path.write_text(html)


if __name__ == "__main__":
    compounds = build()
    RESULTS.mkdir(exist_ok=True)
    write_csv(compounds, RESULTS / "panel_handout.csv")
    write_html(compounds, RESULTS / "panel_handout.html")
    n_exp = sum(
        1 for c in compounds for t in THRESHOLDS if c["predictions"][t]["expected"] is not None
    )
    print(f"{len(compounds)} compounds x {len(THRESHOLDS)} thresholds = "
          f"{len(compounds) * len(THRESHOLDS)} rows")
    print(f"{n_exp} rows carry a stated expected activity; "
          f"{len(compounds) * len(THRESHOLDS) - n_exp} deliberately blank")
    print(f"wrote {RESULTS / 'panel_handout.csv'}")
    print(f"wrote {RESULTS / 'panel_handout.html'}")
