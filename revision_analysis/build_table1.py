"""Argument lines for the two configurations the paper actually reports.

Manuscript Table 1 specifies ONE configuration per endpoint, selected by cross-threshold
ranking and evaluated at all three thresholds. That is what Table S41, Figure 5 and the
abstract report. It is NOT any of the twelve per-cell champions retrained earlier
(REVISION_GUIDE, the warning block before Part 1).

Table 1, read directly from the manuscript:

                          pIC50                      pKi
    inhibitor descriptor  SOAP                       SOAP
    protein descriptor    Global                     -        (molecule only)
    scaler                Min-max                    Min-max
    cross-domain scaling  No                         Yes
    SMOTE (source)        Yes                        Yes
    SMOTE (target)        No                         No

S1 and TL share the same configuration within each endpoint, so one full_pipeline run
per (endpoint, threshold, seed) yields both arms — 2 x 3 x 3 = 18 runs.
"""
from pathlib import Path

SEEDS = [42, 43, 44]
THRESHOLDS = [5.0, 6.0, 7.0]
ROOT = "/u/fd/sjoh5775/matthijs/runs/table1"

CONFIGS = {
    "pic50": dict(
        descriptor="SOAP",
        modality_flags=["--use_mol", "--use_protein"],
        scaler="minmax",
        cross_domain=False,          # Table 1: cross-domain scaling = No
        source_smote=True,
        target_smote=False,
        tag="SOAP_molecule_protein",
    ),
    "pk": dict(
        descriptor="SOAP",
        modality_flags=["--use_mol"],
        scaler="minmax",
        cross_domain=True,           # Table 1: cross-domain scaling = Yes
        source_smote=True,
        target_smote=False,
        tag="SOAP_molecule",
    ),
}

lines = []
for affinity, c in CONFIGS.items():
    for thr in THRESHOLDS:
        for seed in SEEDS:
            name = f"{affinity}_t{int(thr)}_{c['tag']}_s{seed}"
            flags = [
                "--affinity_type", affinity,
                "--threshold", f"{thr}",
                "--inhibitor_descriptor", c["descriptor"],
                *c["modality_flags"],
                "--data_scaler", c["scaler"],
                "--seed", str(seed),
                "--n_folds", "5",
                "--output_dir", f"{ROOT}/{name}",
            ]
            if c["cross_domain"]:
                flags.append("--cross_domain_scaling")
            if c["source_smote"]:
                flags.append("--source_smote")
            if c["target_smote"]:
                flags.append("--target_smote")
            lines.append(" ".join(flags))

out = Path("/u/fd/sjoh5775/matthijs/jobs/table1.txt")
print(f"{len(lines)} runs")
for l in lines[:2]:
    print("  ", l)
Path("table1.txt").write_text("\n".join(lines) + "\n")
print("written to table1.txt")
