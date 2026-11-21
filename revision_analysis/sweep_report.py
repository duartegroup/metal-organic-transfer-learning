"""Selection-bias across the full sweep: nested vs published protocol, matched pairs."""
import glob, re, sys
import pandas as pd

TAIL = re.compile(r"threshold_([\d.]+)/seed_(\d+)$")

def load(root, label):
    rows = []
    for f in glob.glob(root + "/**/*summary.csv", recursive=True):
        rel = f[len(root):].strip("/").split("/")
        if len(rel) < 9:
            continue
        _, descr, modality, scaler, affinity, scaling, thr, seed = rel[:8]
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if "val/mcc" not in d.columns:
            continue
        d = d.assign(descriptor=descr, modality=modality, scaler=scaler,
                     affinity=affinity, scaling=scaling,
                     threshold=thr.replace("threshold_", ""),
                     seed=seed.replace("seed_", ""), arm=label)
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    print(f"{label}: {len(rows)} files, {len(out)} fold-rows", file=sys.stderr)
    return out

KEYS = ["descriptor","modality","scaler","affinity","scaling","threshold","seed","method"]
n = load("/u/fd/sjoh5775/matthijs/runs/matrix", "nested")
c = load("/u/fd/sjoh5775/matthijs/runs/control", "control")
agg = lambda d: d.groupby(KEYS)[["val/mcc","val/mcc_from_probs"]].mean().reset_index()
m = agg(n).merge(agg(c), on=KEYS, suffixes=("_n","_c"))
m.to_csv("/u/fd/sjoh5775/matthijs/runs/sweep_matched.csv", index=False)
print(f"\nMATCHED PAIRS: {len(m)}")
print("by descriptor:"); print(m.descriptor.value_counts().to_string())
for col in ["val/mcc","val/mcc_from_probs"]:
    b = (m[col+"_c"] - m[col+"_n"]).dropna()
    print(f"\n=== {col} ===  n={len(b)}")
    print(f"  mean selection bias : {b.mean():+.4f}")
    print(f"  median              : {b.median():+.4f}")
    print(f"  s.d.                : {b.std():.4f}")
    print(f"  fraction positive   : {(b>0).mean():.3f}")
    sub = m.assign(bias=m[col+"_c"]-m[col+"_n"]).dropna(subset=["bias"])
    print("  by descriptor:"); print(sub.groupby("descriptor")["bias"].agg(["count","mean"]).round(4).to_string())
    print("  by method:");     print(sub.groupby("method")["bias"].agg(["count","mean"]).round(4).sort_values("mean", ascending=False).to_string())
