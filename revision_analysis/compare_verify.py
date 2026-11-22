"""Three-way neutrality test for the source-load skip.

A = runs/control          original run, published behaviour, 24G
B = runs/ctrl_verify_force re-run, FORCE_SOURCE_LOAD=1 (published behaviour)
C = runs/ctrl_verify_patch re-run, skip active

B vs A isolates run-to-run nondeterminism. C vs A is the test. If B==A and C==A
bit-identically, the skip is neutral and the pipeline is deterministic.
"""
import glob
from pathlib import Path
import pandas as pd

ROOT = Path.home()/"matthijs/runs"
A, B, C = ROOT/"control", ROOT/"ctrl_verify_force", ROOT/"ctrl_verify_patch"
COLS = ["val/loss","val/accuracy","val/f1","val/auroc","val/mcc",
        "val/logit_batches_in_unit_interval","val/accuracy_from_probs",
        "val/f1_from_probs","val/mcc_from_probs"]

rels = sorted(str(Path(f).relative_to(C)) for f in glob.glob(str(C/"**/*summary.csv"), recursive=True))
print(f"configurations re-run under the skip: {len(rels)}\n")

def rd(root, rel):
    p = root/rel
    if not p.exists():
        return None
    d = pd.read_csv(p).sort_values(["method","fold"]).reset_index(drop=True)
    return d[[c for c in COLS if c in d.columns]]

def cmp(x, y):
    if x is None or y is None:
        return "MISSING"
    if x.shape != y.shape:
        return f"SHAPE {x.shape} vs {y.shape}"
    if x.equals(y):
        return "identical"
    diff = (x - y).abs().max().max()
    return f"DIFFERS max|d|={diff:.3e}"

nb = nc = 0
for rel in rels:
    a, b, c = rd(A, rel), rd(B, rel), rd(C, rel)
    rb, rc = cmp(a, b), cmp(a, c)
    nb += rb == "identical"; nc += rc == "identical"
    short = "/".join(Path(rel).parts[:6])
    print(f"  {short}\n      B(force) vs A(original): {rb}\n      C(skip)  vs A(original): {rc}")

print(f"\nB==A (nondeterminism check): {nb}/{len(rels)}")
print(f"C==A (THE TEST)            : {nc}/{len(rels)}")
print()
if nb == nc == len(rels):
    print("VERDICT: deterministic, and the skip changes nothing. Safe to use for the SOAP re-run.")
elif nb < len(rels):
    print("VERDICT: pipeline is NOT deterministic -- B differs from A too. Compare the")
    print("         magnitudes above: the skip is only suspect if C-vs-A exceeds B-vs-A.")
else:
    print("VERDICT: the skip CHANGES RESULTS. Do not use it. Fall back to the 128G re-run.")
