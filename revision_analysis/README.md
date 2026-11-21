# Revision analysis

The scripts behind the numbers reported in the revised manuscript and Supporting
Information. They are analysis code, not part of the `transfer_learning_src` package: each
runs standalone against the run artefacts produced by `main.py`, and each carries a module
docstring stating what it computes and which artefacts it reads.

They are deposited so that every reported figure can be traced to the code that produced
it. The run artefacts themselves (~176 GB of per-fold summaries, checkpoints and sweep
output) are not in this repository.

## What produced what

| Reported in | Script |
|---|---|
| Table 1 configurations, nested | `build_table1.py`, `table1_report.py`, `champ_report.py` |
| Table S40 (B / S1 / TL / GNN, pooled) | `table_s40_full.py` |
| Table S41 (per threshold, with fold spreads) | `table_s41_fold_spreads.py` |
| Figure 5 (per-fold champion comparison) | `figure5_nested.py`, `champion_fold_figures.py` |
| Selection-bias table, per training strategy | `selection_bias_sweep.py`, `s1_selection_bias.py` |
| Nine-strategy ranking at the twelve settings | `strategy_ranking_nested.py` |
| GNN control | `gnn_control.py`, `gnn_vs_nested.py` |
| Noise ceilings and headroom | `noise_ceiling.py`, `headroom_tests_nested.py`, `headroom_tests.py` |
| Cluster stability (Hennig bootstrap) | `cluster_stability.py` |
| Latent-space metrics | `latent_compare.py`, `latent_tables.py`, `build_latent_runs.py` |
| PACT motif ablation, Tables S77–S80 | `pact_tables.py`, `pact_agg.py`, `pact_all_thresholds.py`, `pact_delta.py`, `pact_theta6.py`, `pact_setup.py` |
| Assay heterogeneity and identifiability | `assay_heterogeneity.py`, `assay_identifiability.py` |
| Conventional-baseline re-execution | `b_variability.py`, `repro_compare.py` |
| Decision-boundary diagnostic | `metric_delta.py`, `compare_verify.py` |
| Descriptor / architecture correlations | `arch_correlation.py`, `arch_split.py`, `arch_split_soap.py`, `decomposition.py` |
| Run construction and bookkeeping | `build_champion_runs.py`, `build_rerun_matrix.py`, `fold_indices_from_metadata.py`, `extract_figure_matrices.py`, `strip_h5.py`, `sweep_report.py`, `table_s40.py` |

## The GNN control

`gnn_control.py` holds the whole control: a SchNet-style continuous-filter convolutional
network over the three-dimensional geometries, its training loop, and the evaluation. It
uses the same folds, thresholds, metrics and selection rule as the descriptor-based models.
The Slurm array scripts that drove the tuning and production runs are environment-specific
and are not deposited; the configuration they select is recorded in the Supporting
Information.
