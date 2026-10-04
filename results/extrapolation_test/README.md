# PACT extrapolation study — motif ablation

Predicting NAMPT activity for a Ru-caged photoactivated prodrug of STF-31, its
photoproducts, and systematically modified versions of all three, to ask whether the
model's output can be attributed to identifiable structural motifs.

The compound is `caged_PACT` = [Ru(terpyridine)(biquinoline)(STF-31)]²⁺, in which the Ru
binds **through STF-31's pyridine nitrogen** — the nicotinamide mimic NAMPT inhibitors rely
on. Caging masks exactly the pharmacophore, and the four reaction species balance exactly:

```
caged_PACT + H₂O  →  uncaged_PACT_H₂O + stf_31        (120 atoms both sides)
caged_PACT        =  uncaged_PACT ⊕ stf_31            (atom for atom)
```

## Layout

```
molecules/<name>/          one directory per molecule
    input.xyz              structure as drawn or generated
    optimized.xyz          ORCA XTB2 / DDCOSMO(water) / TightOpt
    orca/                  the ORCA input, output and trajectory it came from
    meta.json              series, parent, motif, charge, formula, provenance
    predictions/th{5,6,7}.json
molecules/index.csv        index of all molecules (scripts/build_index.py); drives the job scripts
shared/                    NAMPT.pdb, P2Rank pocket predictions
scripts/                   build, run, analyse (see below)
results/                   analysis outputs
logs/slurm/                job logs
```

Adding a molecule means adding a directory with `input.xyz` and `meta.json`, then running
`python scripts/build_index.py` — no script edits.

## Series

| Series | What it is |
|--------|-----------|
| `reaction` | the four balanced species above |
| `spectator` | `deriv_1`…`deriv_15`, varying only the ancillary ligands. STF-31 and Ru are intact in all of them, so these measure **composition alone** and provide the size baseline every motif claim is corrected against. `deriv_9` duplicates `deriv_8`; `deriv_10`–`15` had Ru–NH₂ repaired to Ru–NH₃. |
| `ablation_free` | STF-31 with one motif removed: pyridine N, amide C=O, amide N–H, sulfonyl, sulfonamide N–H, *tert*-butyl |
| `ablation_caged` | the same edits inside the caged complex, with the Ru fragment untouched. M1 has no caged form — it deletes the atom that binds Ru |
| `fragment` | motifs presented alone, plus free terpyridine and biquinoline as **controls** |
| `photoproduct` | what each caged complex becomes after release, so the contrast is measured at four points |

## Reproducing

```bash
python scripts/build_molecules.py          # fragments + photoproducts
python scripts/generate_ablations.py       # the ablation series
python scripts/build_index.py              # refresh molecules/index.csv
python scripts/make_orca_inputs.py         # ORCA XTB2/DDCOSMO(water)/TightOpt inputs
sbatch --chdir="$PWD" scripts/run_orca_opt.sh    # 39-way array, one molecule per task
python scripts/promote_orca_geometries.py        # report; --promote writes optimized.xyz
sbatch --chdir="$PWD" scripts/run_inference.sh   # everything without predictions
python scripts/analyse.py
python scripts/export_handout.py         # shareable table: results/panel_handout.{csv,html}
```

Inference reuses `optimized.xyz` when its composition matches `input.xyz`, so it featurizes the
promoted geometries rather than re-optimizing them; `--force` re-runs predictions only.

`export_handout.py` writes one row per compound and threshold — structure, expected
activity where one can be stated, predicted activity, value and per-fold spread. The HTML
is self-contained (no network requests) and can be sent as an attachment; its
*Download CSV* button reproduces the CSV byte for byte.

`scripts/run_inference.sh --only <name>` runs a single molecule; `--force` re-runs one that
already has results. Every run uses phase `pretrain_ccsa_encoder_task` so the whole panel
stays comparable.

## Reading the results

Two things bound every conclusion, and the analysis states both before any finding:

**The ensemble disagrees with itself.** Predictions average five fold models whose median
range is 0.57 at pIC50>7. `caged_PACT` returns 0.112, 0.119, 0.369, 0.741, 0.862 — an ensemble
mean of 0.441 over models that do not agree the compound is active at all. Comparisons are
therefore made **paired per fold** (fold *k* against fold *k*), and a shift smaller than the
fold spread means nothing.

**Removing atoms can move the prediction on its own,** because the descriptor is a per-atom
SOAP mean-pooled over atoms. The spectator series measures that axis directly, since it never
touches the pharmacophore: on the current structures the slope is +0.0123 per heavy atom at
pIC50>6 (r = 0.88, p = 0.02), but it is **not significant at pIC50>7** (+0.0064, r = 0.35,
p = 0.50) or at pIC50>5. The size axis is therefore refitted from the geometries in use rather
than assumed, and a motif's effect is the *residual* after subtracting it — a correction that
is currently small at the threshold that matters.

`free_M1_pyridineN_to_CH` is the anchor experiment: swapping the pyridine N for CH changes
**zero** heavy atoms, so no correction applies and molecular size cannot explain its result.

Standalone fragment scores corroborate but are not primary evidence — an isolated fragment
is a different molecule, with capping hydrogens, a different conformation and no
intramolecular context. Ablated compounds are also re-optimized, so each change includes
conformational relaxation, not only the missing motif.

## Geometries

All 39 structures are optimized with the protocol that produced the training set — ORCA 6.0.1,
`! XTB2 OPT`, `! DDCOSMO(water)`, `! TightOpt` (`scripts/make_orca_inputs.py`,
`scripts/run_orca_opt.sh`). The earlier gas-phase ASE/xtb run is superseded: without a
dielectric screening the +2 metal, every **monodentate neutral donor dissociated** — water left
outright in `uncaged_PACT_H2O` and `photo_pyridines` (Ru–O/N past 3.0 Å), `deriv_10`…`15`
stretched Ru–NH₃ to 2.6–3.4 Å, and several complexes split into two fragments. Nothing chelated
moved, which is why it was easy to miss. Every complex now comes back as one fragment with a
six-coordinate Ru and no donor beyond 2.28 Å.

The gas-phase structures and the predictions made from them have been deleted rather than
archived: they were computed under a protocol that does not match the training set, so keeping
them invites a comparison that is not the one this study is making. Every number in `results/`
comes from the geometries described above.

`caged_PACT` is the one exception — it does not converge under TightOpt, limit-cycling on a soft
torsion (MAX step pinned at 0.0154 Å) while the energy is flat to 0.009 kcal/mol and the RMS
gradient is below tolerance; a fresh-Hessian restart reproduced the stall cycle for cycle. Its
geometry is used, listed in `ACCEPTED_UNCONVERGED` in `scripts/promote_orca_geometries.py` with
that reasoning.

## Known limits

- The target training set is 94 Ru complexes, 29–96 atoms, across 38 targets — **NAMPT is not
  among them**, and `caged_PACT` at 117 atoms is larger than any of them. Free organic
  fragments are further outside the domain still.
- `val/mcc` in the training summary CSVs is unreliable on folds where a model predicted a
  single class; see `scripts/verify_equivalence.py`, which also confirms these checkpoints are
  the ones training selected.
