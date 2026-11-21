"""Split the 1728 matched sweep pairs by whether the two arms shared a GPU architecture."""
import re, subprocess
from pathlib import Path
import pandas as pd

HOME = Path.home()/"matthijs"
ARCH = {"comp0800":"P100","comp0817":"A40","comp0818":"A40","comp0819":"A40","comp0812":"RTX2080"}
NESTED = {785889:0, 785890:1000, 785891:2000}
CONTROL = {786305:0, 786306:1000, 786307:2000}

def nodes(job):
    out = subprocess.run(["sacct","-X","-j",str(job),"--format=JobID%20,NodeList%12,State","-n"],
                         capture_output=True, text=True).stdout
    d = {}
    for ln in out.splitlines():
        p = ln.split()
        if len(p) < 3 or "_" not in p[0]: continue
        try: idx = int(p[0].split("_")[1])
        except ValueError: continue
        if p[2] == "COMPLETED": d[idx] = p[1]
    return d

def line_to_node(jobs):
    m = {}
    for job, off in jobs.items():
        for idx, nd in nodes(job).items():
            m[off + idx + 1] = nd
    return m

ln_n, ln_c = line_to_node(NESTED), line_to_node(CONTROL)

def relpath(job, idx, logdir, pat):
    f = HOME/"runs"/logdir/(pat % (job, idx))
    if not f.exists(): return None
    for line in f.read_text(errors="ignore").splitlines():
        if "Summary CSV:" in line:
            return line.split("Summary CSV:")[1].strip()
    return None

rows = []
for job, off in NESTED.items():
    for idx, nd in nodes(job).items():
        L = off + idx + 1
        if L not in ln_c: continue
        p = relpath(job, idx, "logs", "s1_%d_%d.err")
        if not p: continue
        rel = p.split("/runs/matrix/",1)[-1]
        rows.append({"line":L, "rel":rel, "n_node":nd, "c_node":ln_c[L]})
df = pd.DataFrame(rows).drop_duplicates("line")
df["matched"] = df.n_node.map(ARCH) == df.c_node.map(ARCH)
print(f"pairs located: {len(df)}")
print(df.matched.value_counts().to_string())

def mcc(root, rel):
    p = HOME/"runs"/root/rel
    if not p.exists(): return None, None
    d = pd.read_csv(p)
    return d["val/mcc"].mean(), d["val/mcc_from_probs"].mean()

recs = []
for _, r in df.iterrows():
    a = mcc("matrix", r.rel); b = mcc("control", r.rel)
    if a[0] is None or b[0] is None: continue
    recs.append({"matched":r.matched, "bias":b[0]-a[0], "bias_p":b[1]-a[1]})
m = pd.DataFrame(recs)
print(f"\npairs with both results: {len(m)}")
for col,lab in [("bias","val/mcc"),("bias_p","mcc_from_probs")]:
    print(f"\n=== {lab} ===")
    g = m.groupby("matched")[col].agg(["count","mean","std"]).round(4)
    g.index = ["architecture MISMATCHED","architecture MATCHED"]
    print(g.to_string())
    print(f"  all pairs: n={len(m)} mean={m[col].mean():+.4f} sd={m[col].std():.4f}")
