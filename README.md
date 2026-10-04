# Transfer learning for predicting protein binding affinities of metal-organic inhibitors

This repository contains the full code for the paper *(citation coming soon)*. It provides a framework for predicting protein–metal-organic inhibitor activity using transfer learning, comparing supervised learning (SL) baselines against transfer learning (TL) approaches across multiple molecular descriptor types, protein embeddings, and domain adaptation strategies.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Reproducing Paper Figures](#reproducing-paper-figures)
3. [Full Pipeline (for continuing the work)](#full-pipeline)
   - [Data Preparation](#data-preparation)
   - [Data Analysis & Splitting](#data-analysis--splitting)
   - [Running Experiments](#running-experiments)
   - [Statistical Analysis](#statistical-analysis)
   - [Latent Space Analysis](#latent-space-analysis)
4. [Directory Structure](#directory-structure)

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate metal-transfer-learning
```

> **Note:** A GPU is only required if you need to recompute molecular descriptors (MACE, SOAP, ACSF) or protein embeddings (ESM-C). All analysis and plotting scripts run on CPU.

---

## Reproducing Paper Figures

This section maps each figure in the paper to the exact script and output file that produces it. All results are already computed and live in `results/`. You only need to re-run these scripts if you want to regenerate the figures from existing results.

| Figure | Description | Script | Key arguments | Output location |
|--------|-------------|--------|---------------|-----------------|
| Fig 3 | PCA scatter plot of dataset clusters | `data/data_comparison/dataset_split/clustering_affinity_distribution.py` | `--dim_reduction pca --output_dir ./mace_analysis` | `data/data_comparison/dataset_split/mace_analysis/` |
| Fig 4 | Champion bar chart (B vs S1 vs TL, per threshold) | `src/statistical_tests/main.py` | `--champion` | `results/statistical_tests/champion/fig_champion_comparison_threshold_{N}_{mcc,accuracy,f1}.png` |
| Fig 5 | Latent space visualisation | `src/transfer_learning_src/visualize/embedding_visualization.py` | `--model_path <path> --output_dir <path> --phase pretrain_ccsa_encoder_task` | `results/latent_space_analysis/threshold_{N}/{affinity_type}/` |

### Figure 3 — PCA clustering plot

Run from `data/data_comparison/dataset_split/`:

```bash
python clustering_affinity_distribution.py \
    --dataset caged \
    --dim_reduction pca \
    --output_dir ./mace_analysis
```

The PCA scatter plot is saved to `mace_analysis/`.

### Figure 4 — Champion comparison bar chart

Run from `src/statistical_tests/`:

```bash
python main.py --champion
```

**How the champion is chosen.** Within each threshold every configuration is scored by `mean MCC − SD` across the five cross-validation folds and ranked; those per-threshold ranks are averaged over θ ∈ {5, 6, 7}, and the configuration with the lowest average rank is the champion. Ranking within a threshold before averaging keeps a threshold where MCC happens to run high from dominating the selection, and averaging ranks across all three rewards configurations that hold up at every activity cut-off rather than winning once. Both champions in Table 1 come out of this procedure — there are no hardcoded model IDs.

One PNG per metric (MCC, accuracy, F1) per threshold is saved to `results/statistical_tests/champion/`. **All three thresholds must be loaded** — passing `--thresholds 6` alone will abort with a warning, because cross-threshold ranking is meaningless with a single threshold.

Each panel carries up to four bars:

| Bar | What it is |
|-----|------------|
| `B` | supervised learning champion |
| `S1` | **transfer learning baseline** — the TL champion's own configuration (same descriptor, modality, scaler, SMOTE and scaling settings) trained with strategy S1: plain fine-tuning, no pretraining, no domain adaptation |
| `TL` | transfer learning champion |
| `Max` | noise ceiling from `results/noise_estimation/` — not the best observed score |

`S1` is the *matched sibling* rather than the best-ranked S1 model on purpose: it differs from the TL champion in exactly one factor, so the `TL − S1` gap is attributable to pretraining and domain adaptation alone. Without it, a `TL > B` result cannot distinguish "transfer learning helps" from "this descriptor and modality happen to be good". The arm is resolved automatically; if the champion is already an S1 model or the sibling was never run, it is skipped and the chart falls back to three bars.

### Figure 5 — Latent space visualisation

Run from `src/transfer_learning_src/visualize/`:

```bash
python embedding_visualization.py \
    --model_path <path/to/trained/model> \
    --output_dir ../../results/latent_space_analysis \
    --phase pretrain_ccsa_encoder_task \
    --n_folds 5
```

Plots are saved under `results/latent_space_analysis/threshold_{N}/{affinity_type}/`.

---

## Full Pipeline

Use this section if you need to re-run experiments or continue development. The correct order is:

```
1. Data preparation (HDF5 creation)
2. Fold splitting
3. Noise estimation
4. Run experiments (SL + TL)
5. Extract results
6. Statistical analysis + figure generation
7. Latent space analysis
```

### Data Preparation

Pre-built HDF5 datasets are already at `data/hf5_dataset/`. Download links: *(coming soon)*

To rebuild from raw data (requires GPU for MACE/ESM-C):

```bash
# Caged (metal-containing) compounds
python data/hf5_dataset/caged/create_caged_h5.py

# Organic (metal-free) compounds
python data/hf5_dataset/organic/create_organic_h5.py

# Inspect a dataset
python data/hf5_dataset/inspect_h5.py --dataset_path data/hf5_dataset/caged/caged_pic50_dataset.h5
```

Raw input files (PDB structures, SDF ligands, Excel metadata) go in:
- Protein/ligand structures: see paths inside the creation scripts
- Binding affinity metadata: `data/excel/`
- Optimized caged compound structures (SDF and PDB): `data/caged_structures/`

### Data Analysis & Splitting

**Generate spectral clustering folds** (run from `data/fold_split/`):

```bash
python spectral_clustering_split.py
```

Outputs fold index PKL files and distribution CSVs to `data/fold_split/spectral_clustering_analysis/`.

**Visualise fold splits:**

```bash
python spectral_plot_split.py
```

**MCS similarity analysis** (run from `data/data_comparison/fold_dist_mcs_analysis/`):

```bash
python fold_dist_mcs_analysis.py
```

Outputs heatmaps to `mcs_analysis_pic50/`, `mcs_analysis_pk/`, `mcs_images_pic50/`, `mcs_images_pk/`.

**Dataset clustering + PCA** (Figure 3, run from `data/data_comparison/dataset_split/`):

```bash
python clustering_affinity_distribution.py --dataset caged --dim_reduction pca --output_dir ./mace_analysis
```

### Noise Estimation

Must be run before statistical analysis — the results are used as the theoretical performance ceiling in plots and tables.

```bash
python data/noise_estimation/noise_estimation.py
python data/noise_estimation/extract_noise_results.py
```

Output: `results/noise_estimation/threshold_{5.0,6.0,7.0}/{pic50,pk}/`

### Running Experiments

#### Supervised Learning

```bash
python src/supervised_learning_src/supervised_learning.py
```

Results land in `results/supervised_learning_results/threshold_{5,6,7}/`.

#### Transfer Learning

Run from `src/transfer_learning_src/`:

```bash
# Default run (all folds, threshold 6)
python main.py

# Specific threshold with all folds
python main.py --threshold 6.0 --all_folds

# With domain adaptation (CCSA)
python main.py --use_ccsa --ccsa_lambda 0.1

# Specific descriptor and modality
python main.py --descriptor_type MACE --modality protein_mace
```

After experiments finish, extract and aggregate results:

```bash
python src/transfer_learning_src/extract/extract_results.py \
    --base-dir results/transfer_learning_results \
    --output-dir results/transfer_learning_results
```

This produces `all_transfer_learning_results.csv` inside each `final_runs_threshold_{N}/` directory, which the statistical analysis pipeline reads.

#### Inference on Novel Compounds

```bash
python src/transfer_learning_src/inference/inference.py
```

Results are saved to `results/extrapolation_test/`.

### Statistical Analysis

All statistical tests, LaTeX tables, and figures are generated from `src/statistical_tests/`. **Run from that directory** (paths are relative). Each output type has its own flag and they can be combined freely:

```bash
cd src/statistical_tests

# Scatter plots of all individual SL and TL results
python main.py --scatter

# Champion comparison bar chart (requires all 3 thresholds)
python main.py --champion

# LaTeX parameter comparison tables
python main.py --tables

# All three outputs at once
python main.py --scatter --champion --tables

# Restrict to a single affinity type
python main.py --scatter --tables --affinity-types pic50

# Restrict thresholds (note: --champion requires at least 2)
python main.py --scatter --tables --thresholds 5 6

# Verbose output (detailed per-fold breakdowns in console)
python main.py --tables -v
```

**What gets produced:**

| Flag | Output | Location |
|------|--------|----------|
| `--scatter` | Mean-vs-std scatter plots (SL and TL, per threshold) | `results/statistical_tests/scatter/threshold_{N}/` |
| `--champion` | Champion comparison bar chart, one per metric (PNG) | `results/statistical_tests/champion/fig_champion_comparison_threshold_{N}_{mcc,accuracy,f1}.png` |
| `--champion` | Champion + S1 baseline configurations and global means (JSON) | `results/statistical_tests/champion/champion_config.json` |
| `--tables` | LaTeX factor-analysis tables (PCA, Scaler, Descriptor, …) | `results/statistical_tests/tables/threshold_{N}/{supervised,transfer}/` |
| `--tables` | Same tables using all thresholds combined (cross-threshold) | `results/statistical_tests/tables/global/{supervised,transfer}/` |

Champion models are selected **automatically** by averaging per-threshold `mean_mcc − std` ranks across all three thresholds — there are no hardcoded model IDs. The console prints a full top-10 leaderboard, per-threshold best models, and a pooled Wilcoxon comparison whenever `--champion` is run.

**Reading the champion statistics.** A summary table is printed per threshold, then one pooled across all three. Two comparisons are pre-specified — `TL vs B` and `TL vs S1` — and are Holm-corrected within each metric; both raw and corrected p-values are printed. The `S1 − B` gap is shown as a descriptive delta with no p-value, because it is determined by the other two and testing it as well would dress two degrees of freedom up as three.

Only the **pooled** test (n = 15 fold pairs) can reach significance: **at n = 5 the smallest two-sided Wilcoxon p is 0.0625**, already above 0.05, so no per-threshold test can clear α = 0.05 no matter how large the effect. The per-threshold tables are therefore read as effect sizes (Δ and Δ%) and are left uncorrected.

### Input and output paths

All reads are rooted at `--results-base` and all writes at `--output-base`, so the pipeline can analyse a results tree without writing anywhere near it:

```bash
python main.py --champion \
    --results-base /path/to/results \
    --output-base  /scratch/my_analysis
```

Defaults are `./../../results` and `<results-base>/statistical_tests`; `--noise-results-dir` defaults to `<results-base>/noise_estimation`. Every artefact is written to a temporary sibling and renamed into place (`io_utils.atomic_write`), so a run that dies part-way — out of disk, over quota, cancelled — leaves the previous version intact rather than a truncated file.

### Latent Space Analysis

```bash
python src/transfer_learning_src/visualize/embedding_visualization.py \
    --model_path <path/to/trained/model> \
    --output_dir results/latent_space_analysis \
    --phase pretrain_ccsa_encoder_task \
    --n_folds 5

# With detailed misclassification analysis
python src/transfer_learning_src/visualize/embedding_visualization.py \
    --model_path <path/to/trained/model> \
    --output_dir results/latent_space_analysis \
    --with_predictions
```

LaTeX misclassification tables:

```bash
python src/transfer_learning_src/visualize/misclassification_to_latex.py
```

---

## Directory Structure

```
.
├── data/
│   ├── excel/                          # Binding affinity metadata (Excel)
│   │   ├── caged/{pic50,pk}/
│   │   └── organic/{pic50,pk,total_dataset}/
│   ├── caged_structures/{sdf,pdb}/     # Optimized caged compound structures
│   ├── hf5_dataset/                    # HDF5 molecular datasets
│   │   ├── caged/                      # Caged compound datasets + creation script
│   │   ├── organic/                    # Organic compound datasets + creation script
│   │   ├── inspect_h5.py
│   │   └── mace_model/MACE-MP-0b3.model
│   ├── fold_split/                     # Fold splitting scripts and outputs
│   │   ├── spectral_clustering_split.py
│   │   ├── spectral_plot_split.py
│   │   └── spectral_clustering_analysis/
│   ├── data_comparison/
│   │   ├── dataset_split/              # PCA/clustering analysis (→ Figure 3)
│   │   │   ├── clustering_affinity_distribution.py
│   │   │   └── mace_analysis/          # PCA plots, elbow curves
│   │   └── fold_dist_mcs_analysis/     # MCS similarity across folds
│   │       ├── fold_dist_mcs_analysis.py
│   │       └── mcs_analysis_{pic50,pk}/, mcs_images_{pic50,pk}/
│   └── noise_estimation/
│       ├── noise_estimation.py
│       └── extract_noise_results.py
│
├── src/
│   ├── data_src/                       # Shared HDF5 utilities
│   ├── supervised_learning_src/
│   │   └── supervised_learning.py
│   ├── transfer_learning_src/
│   │   ├── main.py                     # TL experiment entry point
│   │   ├── data/                       # Dataset, loading, splitting
│   │   ├── model/multitask.py          # Neural network architecture
│   │   ├── train/                      # Training pipeline and config
│   │   ├── utils/                      # Checkpointing, config loading
│   │   ├── extract/extract_results.py  # Aggregate TL results to CSV
│   │   ├── inference/inference.py      # Inference on novel compounds
│   │   └── visualize/
│   │       ├── embedding_visualization.py   # Latent space plots (→ Figure 5)
│   │       ├── plot_loss_curves.py          # Training/validation loss curves
│   │       └── misclassification_to_latex.py
│   └── statistical_tests/              # Statistical analysis pipeline
│       ├── main.py                     # Entry point — --scatter / --champion / --tables
│       ├── analysis.py                 # MainAnalysisController; dynamic champion ranking
│       ├── reporting.py                # LaTeX table output
│       ├── plot.py                     # Figure generation (→ Figure 4)
│       ├── io_utils.py                 # Atomic write / savefig helpers
│       ├── statistical_tests.py        # Wilcoxon, Friedman, ModelRanker
│       ├── data_loader.py
│       ├── data_quality.py
│       ├── constants.py
│       ├── console.py
│       └── max_value_extractor.py      # Noise ceiling from noise_estimation/
│
├── results/
│   ├── supervised_learning_results/
│   │   └── threshold_{5,6,7}/          # SL outputs per threshold
│   ├── transfer_learning_results/
│   │   └── final_runs_threshold_{5,6,7}/
│   │       └── all_transfer_learning_results.csv   # Input for statistical tests
│   ├── noise_estimation/
│   │   └── threshold_{5.0,6.0,7.0}/{pic50,pk}/    # Used by statistical tests
│   ├── statistical_tests/
│   │   ├── scatter/
│   │   │   └── threshold_{5,6,7}/      # Mean-vs-std scatter plots (--scatter)
│   │   ├── champion/                   # Figure 4 PNGs (--champion)
│   │   │   ├── fig_champion_comparison_threshold_{N}_{mcc,accuracy,f1}.png
│   │   │   └── champion_config.json    # Champion + S1 baseline configs
│   │   └── tables/                     # LaTeX factor tables (--tables)
│   │       ├── threshold_{5,6,7}/
│   │       │   ├── supervised/
│   │       │   └── transfer/
│   │       └── global/                 # Cross-threshold (all thresholds combined)
│   │           ├── supervised/
│   │           └── transfer/
│   ├── latent_space_analysis/
│   │   └── threshold_{5,6,7}/{pic50,pk}/           # Figure 5 plots
│   └── extrapolation_test/                          # Inference outputs
│
├── environment.yml
└── README.md
```
