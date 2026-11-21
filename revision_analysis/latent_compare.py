#!/usr/bin/env python3
"""Compare published (circular) against nested (non-circular) latent-space metrics.

The published analysis read embeddings from tgt_val_loader - the loader that also drove
checkpoint selection - so its class separation is partly a restatement of the selection
criterion. The nested run reads tgt_test_loader. Same metric, same folds; only the
provenance of the points differs.
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

CELLS = [(t, a) for t in (5, 6, 7) for a in ("pic50", "pk")]

# the quantities the paper's claims rest on
KEYS = [
    ("target_only", "JSD", "target active-vs-inactive JSD"),
    ("target_only", "Bhattacharyya", "target active-vs-inactive Bhattacharyya"),
    ("combined", "cross_domain_active_cosine", "same-class (active) cosine"),
    ("combined", "cross_domain_inactive_cosine", "same-class (inactive) cosine"),
    ("combined", "cross_domain_active_vs_inactive_cosine", "different-class cosine"),
    ("combined", "jsd_target_active_vs_inactive", "target JSD (combined fit)"),
]

def load(path: Path) -> dict | None:
    f = path / "separability_metrics.json"
    return json.loads(f.read_text()) if f.is_file() else None

def series(j: dict, where: str, key: str) -> list[float]:
    out = []
    block = j.get("target_only" if where == "target_only" else "source_target_combined", {})
    for fold, v in sorted(block.items()):
        if where == "target_only":
            x = v.get("active_vs_inactive", {}).get(key)
        else:
            x = v.get(key)
        if isinstance(x, (int, float)):
            out.append(float(x))
    return out

def fmt(xs: list[float]) -> str:
    if not xs:
        return "     --     "
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{m:+.3f} ± {s:.3f}"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--published", type=Path, required=True)
    ap.add_argument("--nested", type=Path, required=True)
    args = ap.parse_args()

    for thr, aff in CELLS:
        pub = load(args.published / f"threshold_{thr}" / aff)
        nst = load(args.nested / f"threshold_{thr}" / aff)
        label = f"{'pKi' if aff=='pk' else 'pIC50'} θ{thr}"
        if pub is None and nst is None:
            print(f"\n=== {label}: neither present ==="); continue
        print(f"\n=== {label} ===")
        print(f"  {'quantity':42s} {'published':>16s} {'nested':>16s}   {'n folds':>8s}")
        for where, key, desc in KEYS:
            p = series(pub, where, key) if pub else []
            n = series(nst, where, key) if nst else []
            print(f"  {desc:42s} {fmt(p):>16s} {fmt(n):>16s}   {len(p)}/{len(n):>3}")

if __name__ == "__main__":
    main()
