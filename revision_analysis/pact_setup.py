"""Build model_path trees for the PACT re-run, one per (threshold, seed).

inference.py expects

    <model_path>/<affinity>/threshold_<t>/fold_<n>/<phase>/checkpoints/

but a champion run writes

    <run>/<desc>/<modality>/<scaler>/<affinity>/<scaling>/threshold_<t>/seed_<s>/fold_<n>/...

so there are two extra levels. Rather than patch the inference code, this symlinks the
fold directories into the layout it wants. Nothing is copied and nothing is modified.

Per-threshold S4C champion configurations (REVISION_GUIDE, PACT section):
    theta 5  ACSF / molecule
    theta 6  SOAP / molecule_protein
    theta 7  SOAP / molecule_pocket
"""
from pathlib import Path
import sys

HOME = Path.home() / "matthijs"
CHAMP = HOME / "runs/champ_nested"
OUT = HOME / "runs/pact_models"
PHASE = "pretrain_ccsa_encoder_task"
SEEDS = [42, 43, 44]
# threshold -> run-directory stem (descriptor and modality of that threshold's S4C champion)
CELLS = {5: "pic50_t5_ACSF_molecule", 6: "pic50_t6_SOAP_molecule_protein",
         7: "pic50_t7_SOAP_molecule_pocket"}

problems = []
for thr, stem in CELLS.items():
    for seed in SEEDS:
        run = CHAMP / f"{stem}_s{seed}"
        if not run.is_dir():
            problems.append(f"missing run directory: {run}")
            continue
        # the seed_<s> directory is unique within the run
        seeds_dirs = list(run.glob(f"*/*/*/pic50/*/threshold_{thr}/seed_{seed}"))
        if len(seeds_dirs) != 1:
            problems.append(f"{run.name}: expected 1 seed dir, found {len(seeds_dirs)}")
            continue
        src = seeds_dirs[0]
        dest = OUT / f"t{thr}_s{seed}" / "pic50" / f"threshold_{thr}"
        dest.mkdir(parents=True, exist_ok=True)
        linked = 0
        for fold in range(1, 6):
            f = src / f"fold_{fold}"
            if not (f / PHASE / "checkpoints").is_dir():
                problems.append(f"{run.name} fold_{fold}: no {PHASE}/checkpoints")
                continue
            link = dest / f"fold_{fold}"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(f)
            linked += 1
        print(f"t{thr} s{seed}: {linked}/5 folds -> {dest}")

print()
if problems:
    print("PROBLEMS:")
    for p in problems:
        print(" ", p)
    sys.exit(1)
print("all model_path trees built cleanly")
