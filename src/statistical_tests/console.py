# console.py
"""
Central output module — replaces Python logging throughout the pipeline.

All terminal output goes through either:
  - printer.*      for diagnostic messages (info / warning / error / debug)
  - print_*        for structured result tables and section headers

Colour is used in exactly three places:
  1. Significance column: green *** / ** / *  or  yellow ~  or plain ns
  2. Delta values:        green if positive,  red if negative
  3. Champion test panel: green border = significant,  yellow = not significant

A one-line legend is printed once at start-up via printer.legend().
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

_console = Console()
# Exposed for use by other modules that need direct console access
console = _console


# ---------------------------------------------------------------------------
# Printer — drop-in replacement for logging
# ---------------------------------------------------------------------------

class Printer:
    """
    Structured terminal output without Python logging.

    Usage
    -----
    from console import printer

    printer.info("Loading data...")
    printer.warning("Missing folds for threshold 7")
    printer.error("File not found: results.csv")
    printer.debug("Pivot shape: 48×2")   # only shown when verbose=True
    """

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def set_verbose(self, verbose: bool) -> None:
        self.verbose = verbose

    def info(self, msg: str) -> None:
        _console.print(f"  {msg}")

    def warning(self, msg: str) -> None:
        _console.print(f"  [yellow]Warning: {msg}[/yellow]")

    def error(self, msg: str) -> None:
        _console.print(f"  [red]✗  {msg}[/red]")

    def debug(self, msg: str) -> None:
        if self.verbose:
            _console.print(f"  [dim]·  {msg}[/dim]")

    def legend(self) -> None:
        """Print the colour legend once at start-up."""
        _console.print(
            "  [dim]Colour key: "
            "[green]green ***/**/* = significant (p<0.05)[/green]  ·  "
            "[yellow]yellow ~ = trend (p<0.10)[/yellow]  ·  "
            "ns = not significant  ·  "
            "[green]+Δ[/green]/[red]−Δ[/red] = direction of change[/dim]"
        )
        _console.print()


# Single shared instance — import this everywhere
printer = Printer()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sig(p: float) -> str:
    """Return a significance marker string (colour only in this column)."""
    if p != p:
        return "n/a"
    if p < 0.001:
        return "[green]***[/green]"
    if p < 0.01:
        return "[green]** [/green]"
    if p < 0.05:
        return "[green]*  [/green]"
    if p < 0.10:
        return "[yellow]~  [/yellow]"
    return "ns"


def _p(p: float) -> str:
    return "n/a" if p != p else f"{p:.4f}"


def _delta(d: float, fmt: str = "+.4f") -> str:
    color = "green" if d >= 0 else "red"
    return f"[{color}]{d:{fmt}}[/{color}]"


def _delta_pct(d: float) -> str:
    color = "green" if d >= 0 else "red"
    return f"[{color}]{d:+.2f}%[/{color}]"


# ---------------------------------------------------------------------------
# Level 1 — Phase banners
# ---------------------------------------------------------------------------

def print_phase_banner(phase: str, subtitle: str = "") -> None:
    """Bold rule line marking a major pipeline phase."""
    _console.print()
    _console.print(Rule(f"[bold]{phase}[/bold]", style="bold blue"))
    if subtitle:
        _console.print(f"  [dim]{subtitle}[/dim]")
    _console.print()


# ---------------------------------------------------------------------------
# Level 2 — Analysis context block
# ---------------------------------------------------------------------------

def print_analysis_context(
    name: str,
    factor: str,
    groups: List[str],
    n_pairs: int,
    learning_type: str = "SL",
) -> None:
    """Framed context block printed before each factor analysis."""
    n_groups = len(groups)
    group_str = ", ".join(str(g) for g in groups)

    if n_groups == 2:
        test_line = "Wilcoxon signed-rank  (two-sided, paired by experiment ID)"
        h0_line   = f"No difference in performance between: {groups[0]}  vs  {groups[1]}"
    else:
        test_line = "Friedman  →  post-hoc pairwise Wilcoxon + Holm–Bonferroni correction"
        h0_line   = f"No difference in performance across {n_groups} groups"

    lines = "\n".join([
        f"[bold]{learning_type} · Effect of {name}[/bold]",
        f"  Factor   : {factor}  ({n_groups} groups: {group_str})",
        f"  Test     : {test_line}",
        f"  H₀       : {h0_line}",
        f"  Pairs (n): {n_pairs}",
    ])
    _console.print(Panel(lines, expand=False, padding=(0, 1)))


# ---------------------------------------------------------------------------
# Level 3a — Two-group result (Wilcoxon)
# ---------------------------------------------------------------------------

def print_two_group_result(
    metric: str,
    group_stats: Dict[str, Dict[str, float]],
    wilcoxon_result: Dict[str, Any],
    better_group: Optional[str] = None,
) -> None:
    groups = list(group_stats.keys())

    t = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold",
        title=f"[bold]{metric}[/bold]",
        title_style="",
        pad_edge=True,
    )
    t.add_column("Group",        no_wrap=True, min_width=26)
    t.add_column("μ ± σ",        justify="right", min_width=16)
    t.add_column("W stat",       justify="right")
    t.add_column("p-value",      justify="right")
    t.add_column("Significance", justify="left")

    stat = wilcoxon_result.get("statistic", float("nan"))
    p    = wilcoxon_result.get("p_value",   float("nan"))
    err  = wilcoxon_result.get("error")

    for i, grp in enumerate(groups):
        s      = group_stats[grp]
        mu_str = f"{s['mean']:.4f} ± {s['std']:.4f}"
        is_last = i == len(groups) - 1

        stat_str = f"{stat:.2f}" if (is_last and not err and stat == stat) else ""
        p_str    = _p(p)          if (is_last and not err) else ""
        sig_str  = _sig(p)        if (is_last and not err) else ""

        grp_label = f"[bold]{grp}[/bold]" if grp == better_group else grp
        t.add_row(grp_label, mu_str, stat_str, p_str, sig_str)

    _console.print(t)

    if err:
        _console.print(f"  [red]Test failed: {err}[/red]\n")
    elif better_group and p == p:
        tag = "significant" if p < 0.05 else ("trend" if p < 0.10 else "not significant")
        _console.print(
            f"  → [bold]{better_group}[/bold] is better  (p = {_p(p)}, {tag})\n"
        )
    else:
        _console.print(f"  → No clear winner  (p = {_p(p)})\n")


# ---------------------------------------------------------------------------
# Level 3b — Multi-group result (Friedman + pairwise Wilcoxon)
# ---------------------------------------------------------------------------

def print_multi_group_result(
    metric: str,
    group_stats: Dict[str, Dict[str, float]],
    friedman_result: Dict[str, Any],
    pairwise_results: Optional[Dict[str, Any]],
    best_groups: List[str],
) -> None:
    f_p    = friedman_result.get("p_value",   float("nan"))
    f_stat = friedman_result.get("statistic", float("nan"))
    is_sig = friedman_result.get("significant", False)

    t = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold",
        title=f"[bold]{metric}[/bold]  — Friedman test",
        title_style="",
    )
    t.add_column("Group",   no_wrap=True, min_width=26)
    t.add_column("μ ± σ",  justify="right", min_width=16)
    t.add_column("Best?",  justify="center")

    for grp, s in sorted(group_stats.items()):
        mark      = "✓" if grp in best_groups else ""
        grp_label = f"[bold]{grp}[/bold]" if grp in best_groups else grp
        t.add_row(grp_label, f"{s['mean']:.4f} ± {s['std']:.4f}", mark)

    _console.print(t)
    _console.print(
        f"  Friedman χ² = {f_stat:.3f}  │  p = {_p(f_p)}  │  {_sig(f_p)}"
    )

    if not is_sig:
        _console.print(
            "  → No significant overall difference — all groups treated as equivalent.\n"
        )
        return

    if not pairwise_results:
        _console.print("  → Significant overall difference (no pairwise data).\n")
        return

    _console.print("  [dim]Post-hoc pairwise Wilcoxon (Holm–Bonferroni):[/dim]")

    pw = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    pw.add_column("Comparison",    no_wrap=True, min_width=38)
    pw.add_column("p (raw)",       justify="right")
    pw.add_column("p (corrected)", justify="right")
    pw.add_column("Sig.",          justify="left")
    pw.add_column("Better",        justify="left")

    for key, res in sorted(pairwise_results.items()):
        p_raw  = res.get("p_value",           float("nan"))
        p_corr = res.get("p_value_corrected", float("nan"))
        better = res.get("better", "—")
        pw.add_row(
            key.replace("_vs_", " vs "),
            _p(p_raw),
            _p(p_corr),
            _sig(p_corr),
            f"[bold]{better}[/bold]" if res.get("significant_corrected") else better,
        )

    _console.print(pw)
    best_str = ", ".join(f"[bold]{g}[/bold]" for g in best_groups)
    _console.print(f"  → Best group(s): {best_str}\n")


# ---------------------------------------------------------------------------
# SL vs TL comparison
# ---------------------------------------------------------------------------

def print_comparison_table(
    comparison_results: Dict[str, Any],
    affinity_type: str,
    context: str = "",
    sl_model: str = "",
    tl_model: str = "",
) -> None:
    """SL-vs-TL comparison: context block then metric table."""
    if not comparison_results:
        return

    n_pairs = 0
    for res in comparison_results.values():
        t = res.get("wilcoxon_test") or {}
        if t.get("n_pairs"):
            n_pairs = t["n_pairs"]
            break

    ctx_label = context or "threshold analysis"
    lines = "\n".join([
        f"[bold]SL vs TL · {affinity_type.upper()} · {ctx_label}[/bold]",
        f"  Test     : Wilcoxon signed-rank  (two-sided, paired by fold)",
        f"  H₀       : No difference between best SL and best TL model",
        f"  Pairs (n): {n_pairs}",
        f"  SL model : {sl_model}" if sl_model else "",
        f"  TL model : {tl_model}" if tl_model else "",
    ])
    _console.print(Panel(lines.strip(), expand=False, padding=(0, 1)))

    t = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    t.add_column("Metric",       style="bold", no_wrap=True, min_width=14)
    t.add_column("SL  μ ± σ",   justify="right", min_width=18)
    t.add_column("TL  μ ± σ",   justify="right", min_width=18)
    t.add_column("Δ %",          justify="right", min_width=8)
    t.add_column("Winner",       justify="center", min_width=22)
    t.add_column("W stat",       justify="right")
    t.add_column("p-value",      justify="right")
    t.add_column("Significance", justify="left")

    winners: List[str] = []
    for metric, res in sorted(comparison_results.items()):
        sl   = res["supervised_learning"]
        tl   = res["transfer_learning"]
        test = res.get("wilcoxon_test") or {}
        p    = test.get("p_value",   float("nan"))
        stat = test.get("statistic", float("nan"))

        winner = res.get("better_approach", "—")
        winners.append(winner)

        t.add_row(
            metric,
            f"{sl['mean']:.4f} ± {sl['std']:.4f}",
            f"{tl['mean']:.4f} ± {tl['std']:.4f}",
            _delta_pct(res.get("improvement", 0.0)),
            f"[bold]{winner}[/bold]",
            f"{stat:.2f}" if stat == stat else "n/a",
            _p(p),
            _sig(p),
        )

    _console.print(t)

    if winners:
        overall = max(set(winners), key=winners.count)
        _console.print(f"  → Overall winner (by metric count): [bold]{overall}[/bold]\n")


# ---------------------------------------------------------------------------
# Champion showdown
# ---------------------------------------------------------------------------

def print_champion_showdown(
    paired: pd.DataFrame,
    sl_champion_id: str,
    tl_champion_id: str,
    sl_avg_rank: float,
    tl_avg_rank: float,
    statistic: float,
    p_value: float,
    metric: str = "val/mcc",
    bl_champion_id: Optional[str] = None,
    bl_label: str = "S1",
    bl_statistic: Optional[float] = None,
    bl_p_value: Optional[float] = None,
) -> None:
    """Per-threshold breakdown + Wilcoxon result for global champion comparison.

    When ``bl_champion_id`` is given, the transfer learning baseline strategy is
    shown as a third arm: the champion's own configuration trained without
    pretraining or domain adaptation, so the gap between them is attributable to
    the training strategy alone.
    """

    has_bl = bl_champion_id is not None and "bl" in paired.columns

    # Context block
    context = [
        "[bold]Champion Showdown · Global Cross-Threshold Analysis[/bold]",
        "  Champions selected by lowest average rank across all thresholds.",
        f"  Test     : Wilcoxon signed-rank  (two-sided, paired by (threshold, fold))",
        f"  H₀       : No difference between SL and TL champion on {metric}",
        f"  Pairs (n): {len(paired)}",
        f"  SL champion : {sl_champion_id}  (avg rank {sl_avg_rank:.2f})",
        f"  TL champion : {tl_champion_id}  (avg rank {tl_avg_rank:.2f})",
    ]
    if has_bl:
        context.append(f"  {bl_label} baseline  : {bl_champion_id}")
    lines = "\n".join(context)
    _console.print(Panel(lines, expand=False, padding=(0, 1)))

    # Per-threshold breakdown
    bt = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold",
        title="Per-Threshold Breakdown",
        title_style="bold",
    )
    bt.add_column("Threshold", justify="center")
    bt.add_column("SL  μ ± σ",  justify="right", min_width=16)
    if has_bl:
        bt.add_column(f"{bl_label}  μ ± σ", justify="right", min_width=16)
    bt.add_column("TL  μ ± σ",  justify="right", min_width=16)
    bt.add_column("N pairs",    justify="center")
    bt.add_column("Δ (TL−SL)", justify="right")
    if has_bl:
        bt.add_column(f"Δ (TL−{bl_label})", justify="right")

    for thr in sorted(paired["threshold"].unique()):
        sub = paired[paired["threshold"] == thr]
        row = [str(thr), f"{sub['sl'].mean():.4f} ± {sub['sl'].std():.4f}"]
        if has_bl:
            row.append(f"{sub['bl'].mean():.4f} ± {sub['bl'].std():.4f}")
        row.append(f"{sub['tl'].mean():.4f} ± {sub['tl'].std():.4f}")
        row.append(str(len(sub)))
        row.append(_delta(sub["tl"].mean() - sub["sl"].mean()))
        if has_bl:
            row.append(_delta(sub["tl"].mean() - sub["bl"].mean()))
        bt.add_row(*row)

    _console.print(bt)

    pooled = (
        f"\n  Pooled SL μ = {paired['sl'].mean():.4f} ± {paired['sl'].std():.4f}  "
    )
    if has_bl:
        pooled += (
            f"│  Pooled {bl_label} μ = {paired['bl'].mean():.4f} "
            f"± {paired['bl'].std():.4f}  "
        )
    pooled += (
        f"│  Pooled TL μ = {paired['tl'].mean():.4f} ± {paired['tl'].std():.4f}  "
        f"│  Δ TL−SL = {_delta(paired['tl'].mean() - paired['sl'].mean())}"
    )
    if has_bl:
        pooled += (
            f"  │  Δ TL−{bl_label} = "
            f"{_delta(paired['tl'].mean() - paired['bl'].mean())}"
        )
    _console.print(pooled)

    # Result panel — only here do we use a coloured border
    sig    = p_value < 0.05
    winner = (
        "Transfer Learning" if paired["tl"].mean() > paired["sl"].mean()
        else "Supervised Learning"
    )
    verdict = (
        f"[bold]SIGNIFICANT[/bold]  →  Winner: [bold]{winner}[/bold]"
        if sig
        else "[bold]NOT SIGNIFICANT[/bold]  →  Both approaches perform equivalently"
    )
    body = f"TL vs SL   Wilcoxon W = {statistic:.2f}  │  p = {p_value:.4f}"
    if has_bl and bl_p_value is not None:
        bl_sig = bl_p_value < 0.05
        bl_verdict = "significant" if bl_sig else "not significant"
        body += (
            f"\nTL vs {bl_label}   Wilcoxon W = {bl_statistic:.2f}  "
            f"│  p = {bl_p_value:.4f}  ({bl_verdict})"
        )
        body += (
            f"\n\n[dim]p-values above are uncorrected. The Holm-corrected values "
            f"for the two pre-specified comparisons are reported in the pooled "
            f"champion summary.[/dim]"
        )
    body += f"\n\n{verdict}"
    _console.print(
        Panel(
            body,
            title="[bold]Test Result[/bold]",
            style="green" if sig else "yellow",
            expand=False,
            padding=(0, 2),
        )
    )
    _console.print()


# ---------------------------------------------------------------------------
# Ranking tables
# ---------------------------------------------------------------------------

def print_ranking_table(top_df: pd.DataFrame, label: str) -> None:
    _console.print(Rule(f"[bold]Top {len(top_df)} models — {label}[/bold]"))

    t = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    t.add_column("#",        justify="right", style="dim", width=3)
    t.add_column("Model",    no_wrap=False, min_width=30)
    t.add_column("Avg Rank", justify="right")
    t.add_column("Score",    justify="right")
    t.add_column("Method",   justify="left")

    for i, (_, row) in enumerate(top_df.iterrows(), start=1):
        t.add_row(
            str(i),
            str(row["Model"]),
            f"{row['Rank']:.2f}",
            f"{row['Score']:.4f}",
            str(row["Method"]),
            style="bold" if i == 1 else "",
        )

    _console.print(t)
    _console.print()


# TEMPORARY — investigative only; remove when done
def print_per_threshold_scores(top_models: List[str], df_avg: "pd.DataFrame", label: str) -> None:
    """Print per-threshold ranking_metric and within-threshold rank for the top-N models."""
    import pandas as pd  # already imported at module level, but guard for type checker

    thresholds = sorted(df_avg["threshold"].unique())
    subset = df_avg[df_avg["exp_id"].isin(top_models)].copy()

    _console.print(Rule(f"[bold]Per-threshold scores — {label}[/bold]"))
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("Model", no_wrap=False, min_width=30)
    for thr in thresholds:
        t.add_column(f"T{thr} score", justify="right")
        t.add_column(f"T{thr} rank",  justify="right", style="dim")

    for model in top_models:
        row_vals: List[str] = [model]
        m_df = subset[subset["exp_id"] == model]
        for thr in thresholds:
            thr_row = m_df[m_df["threshold"] == thr]
            if thr_row.empty:
                row_vals += ["-", "-"]
            else:
                score = thr_row.iloc[0]["ranking_metric"]
                rank  = thr_row.iloc[0]["rank"]
                row_vals += [f"{score:.4f}", f"{int(rank)}"]
        t.add_row(*row_vals)

    _console.print(t)
    _console.print()
# TEMPORARY — investigative only; remove when done
def print_per_fold_scores(top_models: List[str], df_fold: "pd.DataFrame", label: str) -> None:
    """Print per-fold ranks for each threshold for the top-N models."""
    thresholds = sorted(df_fold["threshold"].unique())
    folds = sorted(df_fold["fold"].unique())
    subset = df_fold[df_fold["exp_id"].isin(top_models)].copy()

    # Pre-compute the max rank digit width per threshold so every cell aligns
    rank_width = {
        thr: len(str(int(subset[subset["threshold"] == thr]["rank"].max())))
        if not subset[subset["threshold"] == thr].empty else 1
        for thr in thresholds
    }
    fold_header = "/".join(str(f) for f in folds)

    _console.print(Rule(f"[bold]Per-fold ranks — {label}[/bold]"))
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("Model", no_wrap=False, min_width=30)
    for thr in thresholds:
        t.add_column(f"T{thr} fold ranks ({fold_header})", justify="left", no_wrap=True)
        t.add_column(f"T{thr} avg", justify="right", style="dim")

    for model in top_models:
        row_vals: List[str] = [model]
        m_df = subset[subset["exp_id"] == model]
        for thr in thresholds:
            thr_df = m_df[m_df["threshold"] == thr].sort_values("fold")
            if thr_df.empty:
                row_vals += ["-", "-"]
            else:
                w = rank_width[thr]
                fold_ranks = "  ".join(f"{int(r):>{w}}" for r in thr_df["rank"])
                avg_rank = thr_df["rank"].mean()
                row_vals += [fold_ranks, f"{avg_rank:.1f}"]
        t.add_row(*row_vals)

    _console.print(t)
    _console.print()
# END TEMPORARY


def print_best_per_threshold(best_df: pd.DataFrame, label: str) -> None:
    _console.print(Rule(f"[bold]Best model per threshold — {label}[/bold]"))

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("Threshold", justify="center")
    t.add_column("Model",     no_wrap=False)
    t.add_column("Score",     justify="right")

    for _, row in best_df.iterrows():
        t.add_row(str(row["threshold"]), str(row["exp_id"]), f"{row['ranking_metric']:.4f}")

    _console.print(t)
    _console.print()
