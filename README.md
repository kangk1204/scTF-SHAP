# IRF8-High CD8+ T Cell Subtype Discovery in the Melanoma Tumor Microenvironment

Single-cell RNA-seq analysis pipeline that identifies a novel **IRF8-high CD8+ T cell subtype** in the melanoma tumor microenvironment using random forest classification and SHAP-based transcription factor (TF) ranking, validated across three independent cohorts.

## Overview

This repository contains all analysis code for the manuscript. The pipeline:

1. Trains a random forest classifier to distinguish cell types, then uses SHAP values to rank identity-determining TFs
2. Discovers CD8+ T cell subtypes by clustering on the top SHAP-ranked TFs
3. Identifies an IRF8-high subtype with unique exhaustion and XCL1/XCL2 chemokine expression
4. Validates findings across three independent melanoma scRNA-seq cohorts

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/kangk1204/scTF-SHAP.git
cd scTF-SHAP
```

### 2. Set up Python environment

Python **3.9 or higher** is required. We recommend using a virtual environment:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows

# Install all dependencies
pip install -r requirements.txt
```

### 3. Download raw data from GEO

Download the following files and place them in a `data/` folder at the repository root:

```
<repo-root>/
├── data/                          ← Create this folder
│   ├── GSE115978_tpm.csv.gz
│   ├── GSE115978_cell.annotations.csv.gz
│   ├── GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz
│   ├── GSE120575_patient_ID_single_cells.txt.gz
│   └── GSE72056_melanoma_single_cell_revised_v2.txt.gz
├── 01_tf_ml_analysis.py
├── ...
```

| File | GEO Accession | Download from |
|------|---------------|---------------|
| `GSE115978_tpm.csv.gz` | [GSE115978](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115978) | Supplementary file |
| `GSE115978_cell.annotations.csv.gz` | [GSE115978](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115978) | Supplementary file |
| `GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz` | [GSE120575](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120575) | Supplementary file |
| `GSE120575_patient_ID_single_cells.txt.gz` | [GSE120575](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120575) | Supplementary file |
| `GSE72056_melanoma_single_cell_revised_v2.txt.gz` | [GSE72056](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE72056) | Supplementary file |

### 4. Run the pipeline

Run scripts **in numeric order**. Each script prints progress to the terminal.

```bash
python 01_tf_ml_analysis.py
python 02_subtype_discovery.py
python 03_validation_GSE120575.py
python 04_supplementary_analyses.py
python 05_additional_analyses.py
python 06_all_celltype_subclustering.py
python 07_doublet_detection.py
python 08_diffusion_pseudotime_irf8.py
python 10_gse72056_validation.py      # Run BEFORE 09
python 11_irf8_target_enrichment.py
python 09_final_figures.py             # Run LAST (generates all figures)
```

> **Important:** Script `10` must run before `09`, because `09_final_figures.py` uses the Tirosh validation results saved by `10_gse72056_validation.py`.

Or run the entire pipeline at once:

```bash
for script in 01 02 03 04 05 06 07 08 10 11 09; do
    echo "=== Running ${script}*.py ==="
    python ${script}*.py
done
```

### 5. Check outputs

After the pipeline completes, two output folders will be created:

```
<repo-root>/
├── analysis/    ← Intermediate results (CSVs, h5ad files, statistics)
├── figures/     ← Publication-ready figures (PDF + TIFF)
```

**Generated figures:**
| File | Description |
|------|-------------|
| `Figure1.*` | RF-SHAP method and TF ranking |
| `Figure2.*` | CD8+ T cell subtype discovery and pseudotime |
| `Figure3.*` | IRF8-high characterization (DEGs, pathways) |
| `Figure4.*` | XCL1/XCL2 functional axis |
| `FigureS1.*` | TF expression and lineage markers |
| `FigureS2.*` | Pseudotime details |
| `FigureS3.*` | Validation cohort (GSE120575) details |
| `FigureS4.*` | Tirosh cohort (GSE72056) validation |
| `FigureS5.*` | Treatment changes and malignant subtypes |

## Pipeline Details

| # | Script | What it does | Key outputs |
|---|--------|-------------|-------------|
| 01 | `01_tf_ml_analysis.py` | Trains RF classifier on all cell types; computes SHAP TF importance | `tf_importance_*.csv` |
| 02 | `02_subtype_discovery.py` | Clusters CD8+ T cells and malignant cells using top 15 SHAP TFs | `adata_cd8_subtypes.h5ad`, `adata_mal_subtypes.h5ad` |
| 03 | `03_validation_GSE120575.py` | Validates subtypes in Sade-Feldman et al. (GSE120575) | Validation statistics |
| 04 | `04_supplementary_analyses.py` | Contamination checks, patient-level pseudoreplication analysis | Quality control results |
| 05 | `05_additional_analyses.py` | DEG analysis, pathway enrichment, response association | `irf8high_deg_*.csv`, `irf8high_pathway_*.csv` |
| 06 | `06_all_celltype_subclustering.py` | TF-based subclustering for all cell types (CD4, B, macrophage, etc.) | `subclustering_*_summary.csv` |
| 07 | `07_doublet_detection.py` | Scrublet doublet detection; tests IRF8-high enrichment | Doublet statistics |
| 08 | `08_diffusion_pseudotime_irf8.py` | Diffusion pseudotime ordering of CD8+ T cells | `irf8_pseudotime_analysis.csv` |
| 10 | `10_gse72056_validation.py` | Third-cohort validation in Tirosh et al. (GSE72056) | `adata_cd8_tirosh.h5ad`, `tirosh_irf8_degs.csv` |
| 11 | `11_irf8_target_enrichment.py` | Fisher's exact test for IRF8 target gene enrichment | `irf8_target_enrichment_results.csv` |
| 09 | `09_final_figures.py` | Generates all publication figures (Figs 1-4, S1-S5) | `figures/*.pdf`, `figures/*.tiff` |

## Data Sources

| Dataset | Reference | Role | Cells |
|---------|-----------|------|-------|
| GSE115978 | Jerby-Arnon et al., *Cell* 2018 | Discovery cohort | 7,186 |
| GSE120575 | Sade-Feldman et al., *Cell* 2018 | Validation cohort 1 | 16,291 |
| GSE72056 | Tirosh et al., *Science* 2016 | Validation cohort 2 | 4,645 |

## System Requirements

- **Python**: 3.9+
- **RAM**: 16 GB recommended (scRNA-seq data loading)
- **Disk**: ~2 GB for raw data, ~1 GB for outputs
- **OS**: Tested on macOS and Linux

## License

See manuscript for citation details.
