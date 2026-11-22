"""Is the assay-condition effect even identifiable in our data?

pIC50 is a conditional quantity: for competitive inhibition,
    pIC50 = pKi - log10(1 + [S]/Km)
so the same true affinity maps to different pIC50 under different probe
concentrations. The question for this manuscript is not whether pIC50 values
scatter more (they do not, measurably, in our 19 multi-source targets) but
whether the assay-condition term is separable from the target term at all.
If every target is measured under a single format, the two are confounded and
the effect is unidentifiable from our data -- which is itself the finding.
"""
import pandas as pd, numpy as np
from pathlib import Path

REPO = Path("../metal-organic-transfer-learning")
OUT = Path("results"); OUT.mkdir(exist_ok=True)

m = pd.read_excel(REPO / "data/excel/metadata.xlsx", sheet_name="measurements")
m.columns = [c.strip() for c in m.columns]

def norm(s):
    # NaN must stay NaN: 8 pKi measurements carry no assay label, and coercing
    # them to a string makes them look like a distinct assay format, which
    # inflates both the format count and the multi-format target count.
    if pd.isna(s): return pd.NA
    s = str(s).strip().lower()
    s = s.replace("luminsecent", "luminescent")
    s = s.replace(" assay", "")
    return s

m["fmt"] = m["binding_assay"].map(norm)

rows = []
for ep, sub in m.groupby("activity_type"):
    tgt = "protein"
    ct = pd.crosstab(sub[tgt], sub["fmt"])
    n_tgt = len(ct)
    multi = (ct > 0).sum(axis=1)
    rows.append({
        "endpoint": ep,
        "measurements": len(sub),
        "targets": n_tgt,
        "formats": sub["fmt"].nunique(),
        "targets_with_multiple_formats": int((multi > 1).sum()),
        "pct_targets_multi_format": round(100 * (multi > 1).mean(), 1),
    })
    print(f"\n=== {ep} ===")
    print(f"{len(sub)} measurements, {n_tgt} targets, {sub['fmt'].nunique()} formats (normalised)")
    print(f"targets measured under >1 format: {(multi>1).sum()} / {n_tgt}")
    print("\nformat counts:"); print(sub["fmt"].value_counts().to_string())

    # within-target, between-format spread: the only direct estimate available
    diffs = []
    for t, g in sub.groupby(tgt):
        if g["fmt"].nunique() < 2: continue
        means = g.groupby("fmt")["activity_value"].mean()
        if means is not None and len(means) > 1:
            diffs.append(means.max() - means.min())
    if diffs:
        print(f"\nwithin-target between-format range: n={len(diffs)}, "
              f"median={np.median(diffs):.2f}, max={max(diffs):.2f}")
    else:
        print("\nwithin-target between-format range: NOT ESTIMABLE "
              "(no target measured under more than one format)")

pd.DataFrame(rows).to_csv(OUT / "assay_identifiability.csv", index=False)
print("\n" + pd.DataFrame(rows).to_string(index=False))
