"""
07_doublet_detection.py
Scrublet-based doublet detection for CD8+ T cells in both cohorts.
Tests whether IRF8-high cells are enriched for predicted doublets.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
import scrublet as scr
from scipy.stats import mannwhitneyu, fisher_exact
from scipy.sparse import issparse
import os, gzip

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Top 15 SHAP TFs (same as paper)
top15_tfs = ['IRF8', 'IKZF3', 'TCF4', 'SOX10', 'ETV5', 'MITF', 'PAX5',
             'BCL11A', 'MAFB', 'TOX', 'ID2', 'SPI1', 'PRDM1', 'TCF7', 'WWTR1']

def run_tf_subclustering(adata, top_tfs, resolution=0.25):
    """Run TF-based subclustering identical to the paper pipeline."""
    available = [tf for tf in top_tfs if tf in adata.var_names]
    adata_tf = adata[:, available].copy()
    sc.pp.scale(adata_tf, max_value=10)
    n_comps = min(10, len(available) - 1)
    sc.tl.pca(adata_tf, n_comps=n_comps)
    sc.pp.neighbors(adata_tf, n_neighbors=15, n_pcs=n_comps)
    sc.tl.leiden(adata_tf, resolution=resolution, key_added='tf_subtype')
    return adata_tf.obs['tf_subtype']

def identify_irf8_high(adata, clusters):
    """Identify the IRF8-high cluster by highest mean IRF8 expression."""
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    irf8_idx = list(adata.var_names).index('IRF8')
    irf8_expr = X[:, irf8_idx]
    means = {}
    for cl in clusters.unique():
        means[cl] = irf8_expr[clusters.values == cl].mean()
    best = max(means, key=means.get)
    return best, means

def summarize_doublet_results(name, irf8_scores, other_scores, irf8_doublets, other_doublets):
    """Print and return summary statistics."""
    print(f"\n    --- {name} ---")
    print(f"    IRF8-high: n={len(irf8_scores)}, score={irf8_scores.mean():.4f} (median {irf8_scores.median():.4f}), "
          f"doublets={irf8_doublets.sum()}/{len(irf8_doublets)} ({irf8_doublets.sum()/len(irf8_doublets)*100:.1f}%)")
    print(f"    Others:    n={len(other_scores)}, score={other_scores.mean():.4f} (median {other_scores.median():.4f}), "
          f"doublets={other_doublets.sum()}/{len(other_doublets)} ({other_doublets.sum()/len(other_doublets)*100:.1f}%)")

    stat, p_w = mannwhitneyu(irf8_scores, other_scores, alternative='two-sided')
    print(f"    Wilcoxon p = {p_w:.4e}")

    a, b = int(irf8_doublets.sum()), int(len(irf8_doublets) - irf8_doublets.sum())
    c, d = int(other_doublets.sum()), int(len(other_doublets) - other_doublets.sum())
    if a + c > 0:
        odds, p_f = fisher_exact([[a, b], [c, d]])
        print(f"    Fisher p = {p_f:.4e} (OR = {odds:.3f})")
    else:
        p_f = 1.0
        print(f"    Fisher: no doublets detected")

    return {'irf8_mean': irf8_scores.mean(), 'other_mean': other_scores.mean(),
            'p_wilcox': p_w, 'p_fisher': p_f,
            'irf8_rate': irf8_doublets.sum()/len(irf8_doublets),
            'other_rate': other_doublets.sum()/len(other_doublets)}

print("=" * 70)
print("DOUBLET DETECTION ANALYSIS")
print("=" * 70)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. DISCOVERY COHORT (GSE115978)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[1] Discovery cohort (GSE115978)")
print("    Loading TPM data...")

with gzip.open(os.path.join(DATA_DIR, 'GSE115978_tpm.csv.gz'), 'rt') as f:
    df_tpm = pd.read_csv(f, index_col=0)
print(f"    TPM matrix: {df_tpm.shape}")

# Genes are rows, cells are columns → transpose
df_tpm = df_tpm.T
print(f"    Transposed: {df_tpm.shape} (cells × genes)")

# Load annotations
with gzip.open(os.path.join(DATA_DIR, 'GSE115978_cell.annotations.csv.gz'), 'rt') as f:
    df_anno = pd.read_csv(f, index_col=0)
print(f"    Annotations: {df_anno.shape}")

# Merge and filter
common = df_tpm.index.intersection(df_anno.index)
df_anno = df_anno.loc[common]
df_tpm = df_tpm.loc[common]

# Remove unclassified
mask_valid = ~df_anno['cell.types'].str.contains(r'\?', na=False)
df_anno = df_anno[mask_valid]
df_tpm = df_tpm.loc[df_anno.index]
print(f"    After filtering: {df_tpm.shape[0]} cells")

# Create AnnData (log1p TPM)
adata_disc = sc.AnnData(
    X=np.log1p(df_tpm.values.astype(np.float32)),
    obs=df_anno,
    var=pd.DataFrame(index=df_tpm.columns)
)

# Run Scrublet on full dataset
print("    Running Scrublet...")
X_counts = np.expm1(adata_disc.X)  # reverse log1p for Scrublet
scrub = scr.Scrublet(X_counts, expected_doublet_rate=0.05)
scores, preds = scrub.scrub_doublets(min_counts=2, min_cells=3,
                                      min_gene_variability_pctl=85,
                                      n_prin_comps=30, verbose=False)
# If auto-threshold fails, use manual threshold of 0.25
if preds is None:
    preds = scores > 0.25
    print("    (Auto-threshold failed, using 0.25)")
adata_disc.obs['doublet_score'] = scores
adata_disc.obs['predicted_doublet'] = preds
print(f"    Doublets: {preds.sum()}/{len(preds)} ({preds.sum()/len(preds)*100:.1f}%)")

# Extract CD8+ T cells and subcluster
cd8_mask = adata_disc.obs['cell.types'] == 'T.CD8'
adata_cd8_disc = adata_disc[cd8_mask].copy()
print(f"    CD8+ T cells: {adata_cd8_disc.shape[0]}")

clusters_disc = run_tf_subclustering(adata_cd8_disc, top15_tfs, resolution=0.25)
irf8_cl, cl_means = identify_irf8_high(adata_cd8_disc, clusters_disc)
print(f"    IRF8-high cluster: {irf8_cl} (mean IRF8 per cluster: {cl_means})")

irf8_mask_d = clusters_disc == irf8_cl
disc_results = summarize_doublet_results(
    "Discovery Cohort",
    adata_cd8_disc.obs.loc[irf8_mask_d, 'doublet_score'],
    adata_cd8_disc.obs.loc[~irf8_mask_d, 'doublet_score'],
    adata_cd8_disc.obs.loc[irf8_mask_d, 'predicted_doublet'],
    adata_cd8_disc.obs.loc[~irf8_mask_d, 'predicted_doublet'],
)

# Per-cluster breakdown
print("\n    Per-cluster breakdown (Discovery):")
for cl in sorted(clusters_disc.unique()):
    m = clusters_disc == cl
    sc_cl = adata_cd8_disc.obs.loc[m, 'doublet_score']
    db_cl = adata_cd8_disc.obs.loc[m, 'predicted_doublet']
    tag = " *** IRF8-high ***" if cl == irf8_cl else ""
    print(f"      Cluster {cl}: n={m.sum()}, IRF8={cl_means[cl]:.2f}, "
          f"score={sc_cl.mean():.4f}, doublets={db_cl.sum()}/{m.sum()} ({db_cl.sum()/m.sum()*100:.1f}%){tag}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. VALIDATION COHORT (GSE120575)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n\n[2] Validation cohort (GSE120575)")
print("    Loading TPM data...")

tpm_file = os.path.join(DATA_DIR, 'GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz')
with gzip.open(tpm_file, 'rt') as f:
    cell_ids = f.readline().strip().split('\t')
    patient_treat_info = f.readline().strip().split('\t')
df_val = pd.read_csv(tpm_file, sep='\t', skiprows=2, index_col=0, header=None, low_memory=False)
if df_val.shape[1] >= len(cell_ids):
    df_val = df_val.iloc[:, :len(cell_ids)]
df_val.columns = cell_ids[:df_val.shape[1]]
df_val = df_val.T  # cells x genes
print(f"    TPM matrix: {df_val.shape}")

# Build annotation from header line 2
df_val_anno = pd.DataFrame(index=cell_ids[:df_val.shape[0]])
treat_info = patient_treat_info[:df_val.shape[0]]
df_val_anno['patient_treatment'] = treat_info
print(f"    Annotations: {df_val_anno.shape}")

common_v = df_val.index.intersection(df_val_anno.index)
df_val = df_val.loc[common_v]
df_val_anno = df_val_anno.loc[common_v]

adata_val = sc.AnnData(
    X=np.log1p(df_val.values.astype(np.float32)),
    obs=df_val_anno,
    var=pd.DataFrame(index=df_val.columns)
)
print(f"    AnnData: {adata_val.shape}")

# Run Scrublet on full validation dataset
print("    Running Scrublet...")
X_val_counts = np.expm1(adata_val.X)
scrub_v = scr.Scrublet(X_val_counts, expected_doublet_rate=0.05)
scores_v, preds_v = scrub_v.scrub_doublets(min_counts=2, min_cells=3,
                                             min_gene_variability_pctl=85,
                                             n_prin_comps=30, verbose=False)
if preds_v is None:
    preds_v = scores_v > 0.25
    print("    (Auto-threshold failed, using 0.25)")
adata_val.obs['doublet_score'] = scores_v
adata_val.obs['predicted_doublet'] = preds_v
print(f"    Doublets: {preds_v.sum()}/{len(preds_v)} ({preds_v.sum()/len(preds_v)*100:.1f}%)")

# Identify CD8+ T cells
X_v = adata_val.X
if issparse(X_v):
    X_v = X_v.toarray()

cd8a_i = list(adata_val.var_names).index('CD8A') if 'CD8A' in adata_val.var_names else None
cd3d_i = list(adata_val.var_names).index('CD3D') if 'CD3D' in adata_val.var_names else None
cd4_i = list(adata_val.var_names).index('CD4') if 'CD4' in adata_val.var_names else None

cd8_mask_v = (X_v[:, cd8a_i] > 1.0) & (X_v[:, cd3d_i] > 0) & (X_v[:, cd8a_i] > X_v[:, cd4_i])
adata_val_cd8 = adata_val[cd8_mask_v].copy()
print(f"    CD8+ T cells: {adata_val_cd8.shape[0]}")

# Subcluster
clusters_val = run_tf_subclustering(adata_val_cd8, top15_tfs, resolution=0.25)
irf8_cl_v, cl_means_v = identify_irf8_high(adata_val_cd8, clusters_val)
print(f"    IRF8-high cluster: {irf8_cl_v} (mean IRF8: {cl_means_v})")

irf8_mask_v = clusters_val == irf8_cl_v
val_results = summarize_doublet_results(
    "Validation Cohort",
    adata_val_cd8.obs.loc[irf8_mask_v, 'doublet_score'],
    adata_val_cd8.obs.loc[~irf8_mask_v, 'doublet_score'],
    adata_val_cd8.obs.loc[irf8_mask_v, 'predicted_doublet'],
    adata_val_cd8.obs.loc[~irf8_mask_v, 'predicted_doublet'],
)

print("\n    Per-cluster breakdown (Validation):")
for cl in sorted(clusters_val.unique()):
    m = clusters_val == cl
    sc_cl = adata_val_cd8.obs.loc[m, 'doublet_score']
    db_cl = adata_val_cd8.obs.loc[m, 'predicted_doublet']
    tag = " *** IRF8-high ***" if cl == irf8_cl_v else ""
    print(f"      Cluster {cl}: n={m.sum()}, IRF8={cl_means_v[cl]:.2f}, "
          f"score={sc_cl.mean():.4f}, doublets={db_cl.sum()}/{m.sum()} ({db_cl.sum()/m.sum()*100:.1f}%){tag}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nDiscovery: IRF8-high doublet score = {disc_results['irf8_mean']:.4f} vs others = {disc_results['other_mean']:.4f}, Wilcoxon p = {disc_results['p_wilcox']:.4e}")
print(f"Validation: IRF8-high doublet score = {val_results['irf8_mean']:.4f} vs others = {val_results['other_mean']:.4f}, Wilcoxon p = {val_results['p_wilcox']:.4e}")
print(f"\nConclusion: IRF8-high cells {'DO NOT show' if disc_results['p_wilcox'] > 0.05 and val_results['p_wilcox'] > 0.05 else 'show'} elevated doublet scores compared to other CD8+ T cell subtypes.")
print("=" * 70)
