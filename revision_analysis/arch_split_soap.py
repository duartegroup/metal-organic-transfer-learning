"""Architecture-matched vs mismatched selection bias, now including SOAP.

SOAP needs separate treatment: its control runs (job 794021) were pinned to the A40
nodes, while its nested runs were assigned freely and ~21% landed on the P100. So for
SOAP the mismatch is structured (control always A40) rather than random, and if the
P100 scored systematically differently that would bias SOAP's estimate in a way the
earlier random-mismatch check could not detect.

Mapping is done by the summary-CSV path each job logs, so every job is handled the same
way regardless of how its argument file was numbered.
"""
import re
import subprocess
from pathlib import Path

import pandas as pd

HOME = Path.home() / "matthijs"
ARCH = {"comp0800": "P100", "comp0817": "A40", "comp0818": "A40", "comp0819": "A40"}
# job -> (log directory, log filename prefix)
JOBS = {785889: ("logs", "s1"), 785890: ("logs", "s1"), 785891: ("logs", "s1"),
        786305: ("logs_control", "c"), 786306: ("logs_control", "c"),
        786307: ("logs_control", "c"), 794021: ("logs_control", "s")}


def nodes(job):
    out = subprocess.run(["sacct", "-X", "-j", str(job),
                          "--format=JobID%20,NodeList%12,State", "-n"],
                         capture_output=True, text=True).stdout
    d = {}
    for ln in out.splitlines():
        p = ln.split()
        if len(p) >= 3 and "_" in p[0] and p[2] == "COMPLETED":
            try:
                d[int(p[0].split("_")[1])] = p[1]
            except ValueError:
                pass
    return d


SUMMARY = re.compile(r"Summary CSV:\s*(\S+)")


def harvest(job, logdir, prefix):
    """-> {relative summary path: node} for one job."""
    out = {}
    for idx, node in nodes(job).items():
        f = HOME / "runs" / logdir / f"{prefix}_{job}_{idx}.err"
        if not f.exists():
            continue
        m = SUMMARY.search(f.read_text(errors="ignore"))
        if not m:
            continue
        p = m.group(1)
        for root in ("/runs/matrix/", "/runs/control/"):
            if root in p:
                out[p.split(root, 1)[1]] = node
    return out


nested, control = {}, {}
for job, (ld, pre) in JOBS.items():
    (nested if job in (785889, 785890, 785891) else control).update(harvest(job, ld, pre))
print(f"nested runs located : {len(nested)}")
print(f"control runs located: {len(control)}")

rows = []
for rel, nnode in nested.items():
    cnode = control.get(rel)
    if cnode is None:
        continue
    a, b = HOME / "runs/matrix" / rel, HOME / "runs/control" / rel
    if not (a.exists() and b.exists()):
        continue
    da, db = pd.read_csv(a), pd.read_csv(b)
    rows.append({
        "descriptor": Path(rel).parts[1],
        "matched": ARCH.get(nnode) == ARCH.get(cnode),
        "n_arch": ARCH.get(nnode), "c_arch": ARCH.get(cnode),
        "bias": db["val/mcc"].mean() - da["val/mcc"].mean(),
        "bias_p": db["val/mcc_from_probs"].mean() - da["val/mcc_from_probs"].mean(),
    })
m = pd.DataFrame(rows)
print(f"pairs compared      : {len(m)}\n")

print("architecture pairing by descriptor:")
print(pd.crosstab(m.descriptor, m.matched).to_string(), "\n")

for col, lab in [("bias", "val/mcc"), ("bias_p", "mcc_from_probs")]:
    print(f"=== {lab} ===")
    g = m.groupby("matched")[col].agg(["count", "mean", "std"]).round(4)
    g.index = ["MISMATCHED" if not i else "MATCHED" for i in g.index]
    print(g.to_string())
    print("  SOAP only:")
    s = m[m.descriptor == "SOAP"]
    gs = s.groupby("matched")[col].agg(["count", "mean", "std"]).round(4)
    gs.index = ["MISMATCHED" if not i else "MATCHED" for i in gs.index]
    print("   ", gs.to_string().replace("\n", "\n    "))
    print(f"  all pairs: n={len(m)} mean={m[col].mean():+.4f} sd={m[col].std():.4f}\n")
