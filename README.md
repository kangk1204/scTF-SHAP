# IRF8-High CD8+ T Cell Subtype Discovery (scTF-SHAP)

Single-cell RNA-seq analysis pipeline for discovering and validating an **IRF8-high CD8+ T cell subtype** in melanoma.

This repository includes:
- end-to-end analysis scripts (`01` to `11`)
- reproducible intermediate outputs (`analysis/`)
- publication-ready figures (`figures/`)

## What This Pipeline Does

1. Trains a Random Forest model on TF expression.
2. Uses SHAP to rank identity-driving TFs.
3. Discovers CD8+ and malignant subtypes using top SHAP TFs.
4. Validates findings in two independent cohorts (GSE120575, GSE72056).
5. Performs pseudotime, DEG, and enrichment analyses.
6. Generates all manuscript figures (Figure 1-4, Figure S1-S5).

## System Requirements

- Python 3.9+
- RAM: 16 GB recommended
- Disk: about 2 GB (raw data) + about 1 GB (outputs)
- OS: macOS/Linux tested

## Repository Structure

```text
scTF-SHAP/
├── 01_tf_ml_analysis.py
├── 02_subtype_discovery.py
├── 03_validation_GSE120575.py
├── 04_supplementary_analyses.py
├── 05_additional_analyses.py
├── 06_all_celltype_subclustering.py
├── 07_doublet_detection.py
├── 08_diffusion_pseudotime_irf8.py
├── 09_final_figures.py
├── 10_gse72056_validation.py
├── 11_irf8_target_enrichment.py
├── requirements.txt
├── data/        # input files from GEO (you create this)
├── analysis/    # intermediate outputs (auto-generated)
└── figures/     # final figures (auto-generated)
```

## Step-by-Step (Beginner Friendly)

### 1. Clone repository

```bash
git clone https://github.com/kangk1204/scTF-SHAP.git
cd scTF-SHAP
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare data folder and download GEO files

Create `data/` at repository root and place the exact files below:

| File name | GEO accession |
|---|---|
| `GSE115978_tpm.csv.gz` | GSE115978 |
| `GSE115978_cell.annotations.csv.gz` | GSE115978 |
| `GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz` | GSE120575 |
| `GSE120575_patient_ID_single_cells.txt.gz` | GSE120575 |
| `GSE72056_melanoma_single_cell_revised_v2.txt.gz` | GSE72056 |

GEO links:
- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115978
- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE120575
- https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE72056

### 5. Run scripts in this exact order

```bash
python 01_tf_ml_analysis.py
python 02_subtype_discovery.py
python 03_validation_GSE120575.py
python 04_supplementary_analyses.py
python 05_additional_analyses.py
python 06_all_celltype_subclustering.py
python 07_doublet_detection.py
python 08_diffusion_pseudotime_irf8.py
python 10_gse72056_validation.py
python 11_irf8_target_enrichment.py
python 09_final_figures.py
```

Important:
- Run `10_gse72056_validation.py` **before** `09_final_figures.py`.
- Run `09_final_figures.py` **last**.

### 6. Check outputs

After completion:
- `analysis/` contains CSV/H5AD intermediate results.
- `figures/` contains:
  - `Figure1.*` to `Figure4.*`
  - `FigureS1.*` to `FigureS5.*`
  - each in `.pdf` and `.tiff`

## Key Outputs by Script

| Script | Main outputs |
|---|---|
| `01_tf_ml_analysis.py` | `tf_importance_*.csv`, `tf_treatment_changes_all.csv`, `cd8_tf_treatment_changes.csv` |
| `02_subtype_discovery.py` | `adata_cd8_subtypes.h5ad`, `adata_mal_subtypes.h5ad`, subtype summary CSVs |
| `03_validation_GSE120575.py` | validation subtype/treatment statistics |
| `04_supplementary_analyses.py` | contamination and pseudoreplication checks |
| `05_additional_analyses.py` | `irf8high_deg_*.csv`, `irf8high_pathway_*.csv`, response association outputs |
| `06_all_celltype_subclustering.py` | `subclustering_*_summary.csv` and cluster DEGs |
| `07_doublet_detection.py` | scrublet-based doublet statistics |
| `08_diffusion_pseudotime_irf8.py` | `dpt_subtype_pseudotime_stats.csv`, `dpt_irf8_vs_subtypes_tests.csv`, related DPT outputs |
| `10_gse72056_validation.py` | `adata_cd8_tirosh.h5ad`, `tirosh_irf8_degs.csv`, `tirosh_fig_data.json` |
| `11_irf8_target_enrichment.py` | `irf8_target_enrichment_results.csv`, `irf8_target_correlations.csv` |
| `09_final_figures.py` | all main/supplementary figures in `figures/` |

## Troubleshooting

### Error: file not found in `data/`
- Check file names match exactly (including `.gz` and capitalization).
- Confirm files are in `scTF-SHAP/data/`.

### Error during figure generation (`09_final_figures.py`)
- Usually caused by missing upstream outputs.
- Re-run scripts in the required order (especially `10`, `11`, then `09`).

### Memory issues
- Close other large processes.
- Use a machine with >=16 GB RAM.

## Data Sources

- Discovery: Jerby-Arnon et al., Cell 2018 (GSE115978)
- Validation 1: Sade-Feldman et al., Cell 2018 (GSE120575)
- Validation 2: Tirosh et al., Science 2016 (GSE72056)

## Citation

If you use this pipeline, please cite the associated manuscript and source datasets above.
