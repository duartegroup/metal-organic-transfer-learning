"""Is run-to-run disagreement explained by GPU architecture alone?

Four comparisons over the same eight configurations:
  patch  vs patch2   - identical settings, repeated
  force  vs force2   - identical settings, repeated
  patch  vs force    - differs only by the source-load skip
  patch2 vs force2   - differs only by the source-load skip

If architecture explains everything, results are identical exactly when the two runs
shared a GPU type, regardless of which comparison they came from.
"""
import glob
import subprocess
from pathlib import Path

import pandas as pd

R = Path.home() / "matthijs/runs"
ARCH = {"comp0800": "P100", "comp0817": "A40", "comp0818": "A40", "comp0819": "A40"}
JOBS = {"ctrl_verify_patch": 793987, "ctrl_verify_force": 793988,
        "ctrl_verify_patch2": 794008, "ctrl_verify_force2": 794009}
COLS = ["val/loss", "val/accuracy", "val/f1", "val/auroc", "val/mcc"]


def nodes(job):
    out = subprocess.run(["sacct", "-X", "-j", str(job), "--format=JobID%20,NodeList%12", "-n"],
                         capture_output=True, text=True).stdout
    d = {}
    for ln in out.splitlines():
        p = ln.split()
        if len(p) >= 2 and "_" in p[0]:
            try:
                d[int(p[0].split("_")[1])] = p[1]
            except ValueError:
                pass
    return d


N = {k: nodes(j) for k, j in JOBS.items()}
base = R / "ctrl_verify_patch"
rels = sorted(str(Path(f).relative_to(base))
              for f in glob.glob(str(base / "**/*summary.csv"), recursive=True))

order = [l.strip() for l in (Path.home() / "matthijs/jobs/verify.txt").read_text().splitlines() if l.strip()]
idx_of = {}
for i, line in enumerate(order):
    name = [t for t in line.split() if "/runs/control/" in t][0].rstrip("/").split("/")[-1]
    for rel in rels:
        if Path(rel).parts[0] == name and rel not in idx_of.values():
            idx_of[i] = rel
            break


def rd(root, rel):
    p = R / root / rel
    if not p.exists():
        return None
    d = pd.read_csv(p).sort_values(["method", "fold"]).reset_index(drop=True)
    return d[[c for c in COLS if c in d.columns]]


pairs = [("ctrl_verify_patch", "ctrl_verify_patch2"),
         ("ctrl_verify_force", "ctrl_verify_force2"),
         ("ctrl_verify_patch", "ctrl_verify_force"),
         ("ctrl_verify_patch2", "ctrl_verify_force2")]

agree_same = agree_diff = tot_same = tot_diff = 0
print("comparison".ljust(26), "task", "archA ", "archB ", "sameArch", "equal")
for x, y in pairs:
    for i, rel in sorted(idx_of.items()):
        a, b = rd(x, rel), rd(y, rel)
        if a is None or b is None:
            continue
        nx, ny = N[x].get(i, "?"), N[y].get(i, "?")
        ax, bx = ARCH.get(nx, "?"), ARCH.get(ny, "?")
        if "?" in (ax, bx):
            continue
        same, eq = ax == bx, a.equals(b)
        if same:
            tot_same += 1
            agree_same += eq
        else:
            tot_diff += 1
            agree_diff += eq
        label = x.replace("ctrl_verify_", "") + " vs " + y.replace("ctrl_verify_", "")
        print(label.ljust(26), str(i).ljust(4), ax.ljust(6), bx.ljust(6),
              str(same).ljust(8), str(eq).ljust(6), nx.ljust(9), ny.ljust(9),
              "SAME-NODE" if nx == ny else "")

print()
print("SAME architecture      : identical %d/%d" % (agree_same, tot_same))
print("DIFFERENT architecture : identical %d/%d" % (agree_diff, tot_diff))

# Same-node is the stricter grouping: comp0817/0818/0819 are all A40 but are
# distinct machines, so same-architecture is not the same claim as same-node.
sn_eq = sn_tot = 0
for x, y in pairs:
    for i, rel in sorted(idx_of.items()):
        a, b = rd(x, rel), rd(y, rel)
        if a is None or b is None:
            continue
        nx, ny = N[x].get(i, "?"), N[y].get(i, "?")
        if nx == "?" or ny == "?" or nx != ny:
            continue
        sn_tot += 1
        sn_eq += a.equals(b)
print("SAME NODE              : identical %d/%d" % (sn_eq, sn_tot))

# Does the skip ever explain a difference that architecture does not?
print()
print("Disagreement rate, identical settings vs across the skip:")
for lab, ps in [("identical settings", pairs[:2]), ("across the skip", pairs[2:])]:
    eq = tot = 0
    for x, y in ps:
        for i, rel in sorted(idx_of.items()):
            a, b = rd(x, rel), rd(y, rel)
            if a is None or b is None:
                continue
            tot += 1
            eq += a.equals(b)
    print("  %-20s identical %d/%d" % (lab, eq, tot))
