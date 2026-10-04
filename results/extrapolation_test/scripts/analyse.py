"""
Motif attribution for the PACT panel.

Tests whether the model's predicted activity at a given pIC50 threshold can be
attributed to particular structural motifs.

The primary experiment is removal. If a motif is deleted from its parent and the
prediction drops, that motif was carrying the prediction, provided the drop
exceeds what deleting a comparable amount of molecule would have caused.
deriv_1..deriv_5 alter only spectator ligands, never the pharmacophore, yet their
predictions still move with molecular size. This script therefore reports

    motif effect = observed dP  -  (spectator slope x d_heavy)

with the spectator slope measured per threshold from those five compounds. The
raw dP is printed beside the residual so the size of the correction stays visible.

``free_M1_pyridineN_to_CH`` is the anchor: it swaps the pyridine N for CH,
changing zero heavy atoms, so its correction is zero and its dP is its motif
effect. It is the one edit in the panel that molecular size cannot explain.

Standalone fragment scores are reported as corroboration only. An isolated
fragment is a different molecule -- capping hydrogens, different conformation, no
intramolecular context -- so its score is not the contribution that fragment made
inside the parent, and it is never used as primary evidence.

Every difference is computed as a paired per-fold shift (fold k of one
molecule against fold k of the other), which cancels each model's bias, and is
judged against the spread between folds, which is large.

Usage
-----
    python analyse.py [--outdir ../results]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
MOL = BASE / "molecules"
THRESHOLDS = (5, 6, 7)

# Spectator-only members: they change the ancillary ligands and never touch the
# pharmacophore, so a fit through them measures composition alone.
SPECTATOR_FIT = ["caged_PACT", "deriv_1", "deriv_2", "deriv_3", "deriv_4", "deriv_5"]

# Each ablated motif and the fragment that presents the same motif on its own.
MOTIF_FRAGMENT = {
    "M1_pyridineN_to_CH": "frag_aminopyridine",
    "M2_amideCO_to_CH2": "frag_warhead_half",
    "M3_amideNH_to_NMe": "frag_warhead_half",
    "M4_sulfonyl_to_CH2": "frag_tail_half",
    "M5_sulfonamideNH_to_NMe": "frag_tail_half",
    "M6_tBu_to_H": "frag_tertbutylbenzene",
}

# Caged complex -> the photoproduct it becomes on releasing the drug.
RELEASE_PAIRS = {
    "caged_PACT": "uncaged_PACT_H2O",
    "deriv_4": "photo_phenanthroline",
    "deriv_5": "photo_bipyridine",
    "deriv_8": "photo_pyridines",
}


def load() -> pd.DataFrame:
    rows = []
    for d in sorted(MOL.iterdir()):
        if not d.is_dir() or not (d / "meta.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        for th in THRESHOLDS:
            p = d / "predictions" / f"th{th}.json"
            if not p.exists():
                continue
            pred = json.loads(p.read_text())["prediction"]
            rows.append({
                "name": meta["name"], "series": meta["series"],
                "parent": meta.get("parent"), "motif": meta.get("motif"),
                "n_heavy": meta["n_heavy"], "threshold": th,
                "p": pred["ensemble_probability"],
                "folds": np.asarray(pred.get("fold_probabilities", []), float),
            })
    return pd.DataFrame(rows)


def paired(df, a, b, th):
    """Per-fold shift of molecule ``a`` relative to ``b``. Returns dict or None."""
    ra = df[(df.name == a) & (df.threshold == th)]
    rb = df[(df.name == b) & (df.threshold == th)]
    if ra.empty or rb.empty:
        return None
    fa, fb = ra.folds.iloc[0], rb.folds.iloc[0]
    if len(fa) == 0 or len(fa) != len(fb):
        return {"delta": float(ra.p.iloc[0] - rb.p.iloc[0]), "sd": np.nan, "p": np.nan}
    d = fa - fb
    t, pv = stats.ttest_rel(fa, fb)
    return {"delta": float(d.mean()), "sd": float(d.std(ddof=1)), "p": float(pv)}


def noise_floor(df):
    out = []
    for th in THRESHOLDS:
        sub = df[(df.threshold == th) & df.folds.apply(len).gt(0)]
        if sub.empty:
            continue
        out.append({"threshold": th, "n": len(sub),
                    "median_fold_sd": sub.folds.apply(np.std).median(),
                    "median_fold_range": sub.folds.apply(lambda f: f.max() - f.min()).median()})
    return pd.DataFrame(out)


def spectator_slope(df):
    out = []
    for th in THRESHOLDS:
        sub = df[(df.threshold == th) & df.name.isin(SPECTATOR_FIT)]
        if len(sub) < 3:
            continue
        fit = stats.linregress(sub.n_heavy, sub.p)
        out.append({"threshold": th, "slope": fit.slope, "r": fit.rvalue,
                    "p": fit.pvalue, "n": len(sub)})
    return pd.DataFrame(out)


def ablation_table(df, slopes):
    rows = []
    heavy = df.drop_duplicates("name").set_index("name").n_heavy
    for th in THRESHOLDS:
        s = slopes[slopes.threshold == th]
        if s.empty:
            continue
        slope = s.slope.iloc[0]
        for _, r in df[(df.threshold == th)
                       & df.series.isin(["ablation_free", "ablation_caged"])].iterrows():
            pair = paired(df, r["name"], r.parent, th)
            if pair is None:
                continue
            dh = int(heavy[r["name"]] - heavy[r.parent])
            expected = slope * dh
            tag = r["name"].split("_", 1)[1] if r["name"].startswith("free_") \
                else r["name"].split("_", 1)[1]
            frag = MOTIF_FRAGMENT.get(tag)
            fr = df[(df.name == frag) & (df.threshold == th)] if frag else None
            rows.append({
                "threshold": th, "molecule": r["name"], "form": r.series.split("_")[1],
                "motif": r.motif, "parent": r.parent, "d_heavy": dh,
                "delta": pair["delta"], "delta_sd": pair["sd"], "p_paired": pair["p"],
                "size_expected": expected,
                "motif_effect": pair["delta"] - expected,
                "fragment": frag,
                "fragment_p": float(fr.p.iloc[0]) if fr is not None and not fr.empty else np.nan,
                "above_noise": abs(pair["delta"] - expected) > (pair["sd"] if pair["sd"] == pair["sd"] else np.inf),
            })
    return pd.DataFrame(rows)


def controls(df):
    rows = []
    for th in THRESHOLDS:
        get = lambda n: df[(df.name == n) & (df.threshold == th)]
        for name, why in [("frag_terpyridine", "spectator ligand alone - expect inactive"),
                          ("frag_biquinoline", "spectator ligand alone - expect inactive")]:
            r = get(name)
            if not r.empty:
                rows.append({"threshold": th, "control": name, "value": float(r.p.iloc[0]),
                             "note": why})
        w = paired(df, "uncaged_PACT_H2O", "uncaged_PACT", th)
        if w:
            rows.append({"threshold": th, "control": "aqua adduct (3 atoms)",
                         "value": w["delta"],
                         "note": "smallest real chemical change in the panel"})
        dup = paired(df, "deriv_9", "deriv_8", th)
        if dup:
            rows.append({"threshold": th, "control": "deriv_9 vs deriv_8",
                         "value": dup["delta"], "note": "identical structures - must be 0"})
    return pd.DataFrame(rows)


def release(df):
    rows = []
    for th in THRESHOLDS:
        for caged, photo in RELEASE_PAIRS.items():
            pr = paired(df, photo, caged, th)
            drug = df[(df.name == "stf_31") & (df.threshold == th)]
            if pr is None or drug.empty:
                continue
            rows.append({"threshold": th, "caged": caged, "photoproduct": photo,
                         "photo_minus_caged": pr["delta"], "sd": pr["sd"],
                         "released_drug_p": float(drug.p.iloc[0])})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(BASE / "results"))
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    df = load()
    if df.empty:
        raise SystemExit("no predictions found under molecules/")
    print("=" * 80)
    print("MOTIF ATTRIBUTION")
    print("=" * 80)
    print(f"{df.name.nunique()} molecules, {len(df)} predictions\n")
    print(df.drop_duplicates("name").groupby("series").size().to_string())

    print("\n" + "-" * 80)
    print("1. NOISE FLOOR - disagreement between the five fold models")
    print("-" * 80)
    nf = noise_floor(df)
    print(nf.round(4).to_string(index=False))

    print("\n" + "-" * 80)
    print("2. COMPOSITION BASELINE - fitted on spectator-only compounds")
    print("   (these never touch the pharmacophore, so the slope is pure size)")
    print("-" * 80)
    sl = spectator_slope(df)
    print(sl.round(4).to_string(index=False))

    print("\n" + "-" * 80)
    print("3. REMOVAL, CORRECTED FOR COMPOSITION")
    print("   motif_effect = delta - (slope x d_heavy);  M1 has d_heavy = 0")
    print("-" * 80)
    ab = ablation_table(df, sl)
    for th in THRESHOLDS:
        sub = ab[ab.threshold == th]
        if sub.empty:
            continue
        print(f"\n  threshold {th}:")
        print(sub[["molecule", "form", "d_heavy", "delta", "delta_sd",
                   "size_expected", "motif_effect", "fragment_p", "above_noise"]]
              .round(4).to_string(index=False))

    print("\n" + "-" * 80)
    print("4. CONTROLS")
    print("-" * 80)
    ct = controls(df)
    if not ct.empty:
        print(ct.round(4).to_string(index=False))

    print("\n" + "-" * 80)
    print("5. RELEASE CONTRAST - caged complex vs its own photoproduct")
    print("-" * 80)
    rl = release(df)
    if not rl.empty:
        print(rl.round(4).to_string(index=False))

    tidy = df.drop(columns="folds")
    for frame, name in [(tidy, "panel_predictions.csv"), (nf, "noise_floor.csv"),
                        (sl, "composition_baseline.csv"), (ab, "motif_attribution.csv"),
                        (ct, "controls.csv"), (rl, "release_contrast.csv")]:
        if not frame.empty:
            frame.round(6).to_csv(out / name, index=False)
    print(f"\nWrote analysis CSVs to {out}")


if __name__ == "__main__":
    main()
