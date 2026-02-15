"""
10_gse72056_validation.py
Third-cohort validation using GSE72056 (Tirosh et al., Science 2016).
Loads melanoma scRNA-seq data, identifies CD8+ T cells, applies SHAP-TF
subclustering, and performs DEG analysis with cross-cohort overlap.
Saves processed data (h5ad + JSON) for figure generation by 09_final_figures.py.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import mannwhitneyu
import os
import gzip
import json

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)
DATA_DIR = os.path.join(BASE_DIR, 'data')

###############################################################################
#  STEP 1: Load GSE72056 (Tirosh et al.)
###############################################################################
print("=" * 70)
print("LOADING GSE72056 (Tirosh et al., Science 2016)")
print("=" * 70)

tirosh_file = f'{DATA_DIR}/GSE72056_melanoma_single_cell_revised_v2.txt.gz'

# Read header lines for metadata
print("Reading header rows...")
with gzip.open(tirosh_file, 'rt') as f:
    header_line = f.readline().strip().split('\t')
    row1 = f.readline().strip().split('\t')  # tumor
    row2 = f.readline().strip().split('\t')  # malignant status (1=no, 2=yes, 0=unresolved)
    row3 = f.readline().strip().split('\t')  # cell type (1=T, 2=B, 3=Macro, 4=Endo, 5=CAF, 6=NK)

cell_ids = header_line[1:]
tumor_ids = row1[1:]
malignant_status = row2[1:]
cell_type_codes = row3[1:]

print(f"  Cells in header: {len(cell_ids)}")

# Read expression matrix (skip 3 metadata rows)
print("Reading expression matrix...")
expr = pd.read_csv(tirosh_file, sep='\t', skiprows=3, index_col=0, low_memory=False)
expr.columns = cell_ids[:expr.shape[1]]
print(f"  Expression matrix: {expr.shape[0]} genes x {expr.shape[1]} cells")

# Build metadata
meta_df = pd.DataFrame({
    'cell_id': cell_ids[:expr.shape[1]],
    'tumor': tumor_ids[:expr.shape[1]],
    'malignant': malignant_status[:expr.shape[1]],
    'cell_type_code': cell_type_codes[:expr.shape[1]],
}, index=cell_ids[:expr.shape[1]])

# Cell type mapping: 1=T, 2=B, 3=Macro, 4=Endo, 5=CAF, 6=NK
ct_map = {'1': 'T cell', '2': 'B cell', '3': 'Macrophage',
          '4': 'Endothelial', '5': 'CAF', '6': 'NK'}
meta_df['cell_type'] = meta_df['cell_type_code'].map(ct_map)

# Build AnnData
# Data is in log2(TPM/10 + 1) format; convert to log1p(TPM) for consistency
# TPM = 10 * (2^x - 1)
print("Converting from log2(TPM/10+1) to log1p(TPM)...")
expr_T = expr.T.astype(np.float32)
tpm_vals = 10.0 * (np.power(2.0, expr_T.values) - 1.0)
tpm_vals = np.clip(tpm_vals, 0, None)  # ensure non-negative
log1p_vals = np.log1p(tpm_vals)

adata_tirosh = sc.AnnData(X=log1p_vals.astype(np.float32))
adata_tirosh.obs_names = expr_T.index.tolist()
adata_tirosh.var_names = expr.index.tolist()

# Deduplicate gene names
adata_tirosh.var_names_make_unique()
print(f"  Unique genes after dedup: {adata_tirosh.n_vars}")
for col in meta_df.columns:
    adata_tirosh.obs[col] = meta_df.loc[adata_tirosh.obs_names, col].values

print(f"  Total cells: {adata_tirosh.n_obs}")
print(f"  Cell types: {adata_tirosh.obs['cell_type'].value_counts().to_dict()}")

###############################################################################
#  STEP 2: Identify CD8+ T cells
###############################################################################
print("\nIdentifying CD8+ T cells...")

cd8a = adata_tirosh[:, 'CD8A'].X.flatten() if 'CD8A' in adata_tirosh.var_names else np.zeros(adata_tirosh.n_obs)
cd4_e = adata_tirosh[:, 'CD4'].X.flatten() if 'CD4' in adata_tirosh.var_names else np.zeros(adata_tirosh.n_obs)
cd3d = adata_tirosh[:, 'CD3D'].X.flatten() if 'CD3D' in adata_tirosh.var_names else np.zeros(adata_tirosh.n_obs)

# Use same criteria as other cohorts: CD8A > 1.0, CD3D > 0, CD8A > CD4
# Only from non-malignant T cells (cell_type_code == '1' and malignant != '2')
t_cell_mask = (adata_tirosh.obs['cell_type_code'] == '1') & (adata_tirosh.obs['malignant'] != '2') & (adata_tirosh.obs['malignant'] != '0')
cd8_marker_mask = (cd8a > 1.0) & (cd3d > 0) & (cd8a > cd4_e)
cd8_mask = t_cell_mask & cd8_marker_mask

adata_cd8_tir = adata_tirosh[cd8_mask].copy()
print(f"  CD8+ T cells selected: {adata_cd8_tir.n_obs}")
print(f"  From {adata_cd8_tir.obs['tumor'].nunique()} tumors")

# Verify lineage
cd163_expr = adata_cd8_tir[:, 'CD163'].X.flatten() if 'CD163' in adata_cd8_tir.var_names else np.zeros(adata_cd8_tir.n_obs)
csf1r_expr = adata_cd8_tir[:, 'CSF1R'].X.flatten() if 'CSF1R' in adata_cd8_tir.var_names else np.zeros(adata_cd8_tir.n_obs)
cd8a_pct = (adata_cd8_tir[:, 'CD8A'].X.flatten() > 0).mean() * 100
cd3d_pct = (adata_cd8_tir[:, 'CD3D'].X.flatten() > 0).mean() * 100
cd163_pct = (cd163_expr > 0).mean() * 100
csf1r_pct = (csf1r_expr > 0).mean() * 100
print(f"  Lineage: CD8A={cd8a_pct:.1f}%, CD3D={cd3d_pct:.1f}%, CD163={cd163_pct:.1f}%, CSF1R={csf1r_pct:.1f}%")

###############################################################################
#  STEP 3: SHAP-TF subclustering (same pipeline as other cohorts)
###############################################################################
print("\nPerforming SHAP-TF subclustering...")

# Load discovery TF importance ranking
tf_cd8_discovery = pd.read_csv(f'{OUT_DIR}/tf_importance_T_CD8.csv')
top_cd8_tfs = tf_cd8_discovery.head(15)['TF'].tolist()
available_tfs = [tf for tf in top_cd8_tfs if tf in adata_cd8_tir.var_names]
print(f"  Top 15 TFs available: {len(available_tfs)}/{len(top_cd8_tfs)}")
print(f"  TFs: {available_tfs}")

# Subclustering
adata_cd8_tir_tf = adata_cd8_tir[:, available_tfs].copy()
sc.pp.scale(adata_cd8_tir_tf, max_value=10)
sc.tl.pca(adata_cd8_tir_tf, n_comps=min(10, len(available_tfs) - 1), random_state=42)
sc.pp.neighbors(adata_cd8_tir_tf, n_pcs=min(10, len(available_tfs) - 1),
                n_neighbors=15, random_state=42)
sc.tl.leiden(adata_cd8_tir_tf, resolution=0.25, key_added='tf_subtype', random_state=42)
sc.tl.umap(adata_cd8_tir_tf, random_state=42)

adata_cd8_tir.obs['tf_subtype'] = adata_cd8_tir_tf.obs['tf_subtype'].values
adata_cd8_tir.obsm['X_umap_tf'] = adata_cd8_tir_tf.obsm['X_umap']

# Assign subtype labels
print("\nSubtype assignment:")
tir_subtype_stats = {}
for st in sorted(adata_cd8_tir.obs['tf_subtype'].unique()):
    mask = adata_cd8_tir.obs['tf_subtype'] == st
    tir_subtype_stats[st] = {
        'n': int(mask.sum()),
        'irf8': float(adata_cd8_tir[mask, 'IRF8'].X.mean()) if 'IRF8' in adata_cd8_tir.var_names else 0,
        'tox': float(adata_cd8_tir[mask, 'TOX'].X.mean()) if 'TOX' in adata_cd8_tir.var_names else 0,
        'tcf7': float(adata_cd8_tir[mask, 'TCF7'].X.mean()) if 'TCF7' in adata_cd8_tir.var_names else 0,
        'prdm1': float(adata_cd8_tir[mask, 'PRDM1'].X.mean()) if 'PRDM1' in adata_cd8_tir.var_names else 0,
        'id2': float(adata_cd8_tir[mask, 'ID2'].X.mean()) if 'ID2' in adata_cd8_tir.var_names else 0,
    }

# For labeling: use relative IRF8 expression.
# Find cluster with highest IRF8 — that is the IRF8-high cluster.
# Other clusters get labeled by their dominant TF.
irf8_vals = {st: stats['irf8'] for st, stats in tir_subtype_stats.items()}
max_irf8_cluster = max(irf8_vals, key=irf8_vals.get)

# Only label clusters with mean IRF8 clearly above background as IRF8-high
# In the discovery cohort, IRF8-high has mean IRF8 = 1.73 while others are < 0.2
# In this dataset, cluster 3 has IRF8 = 5.79 (clearly IRF8-high) while cluster 2 has IRF8 = 1.30
# Use threshold of 2.0 to identify genuinely IRF8-high clusters
irf8_high_threshold = 2.0

tir_labels = {}
for st, stats in sorted(tir_subtype_stats.items()):
    if stats['irf8'] >= irf8_high_threshold:
        tir_labels[st] = 'IRF8-high'
    elif stats['tcf7'] > 1.5 and stats['tox'] < stats['tcf7']:
        tir_labels[st] = 'Memory (TCF7+)'
    elif stats['tox'] > 3.0:
        tir_labels[st] = 'Exhausted (TOX-hi)'
    elif stats.get('id2', 0) > 1.7 and stats.get('prdm1', 0) > 0.8:
        tir_labels[st] = 'Cytotoxic (ID2+PRDM1+)'
    elif stats.get('id2', 0) > 1.5:
        tir_labels[st] = 'Innate-like (ID2+)'
    elif stats.get('prdm1', 0) > 0.8:
        tir_labels[st] = 'Effector (PRDM1+)'
    else:
        tir_labels[st] = 'Effector'
    print(f"  Tirosh CD8 subtype {st}: n={stats['n']}, IRF8={stats['irf8']:.2f}, "
          f"TOX={stats['tox']:.2f}, TCF7={stats['tcf7']:.2f} -> {tir_labels[st]}")

adata_cd8_tir.obs['subtype_label'] = adata_cd8_tir.obs['tf_subtype'].map(tir_labels)

# Count IRF8-high
irf8_high_mask = adata_cd8_tir.obs['subtype_label'] == 'IRF8-high'
n_irf8 = irf8_high_mask.sum()
pct_irf8 = n_irf8 / adata_cd8_tir.n_obs * 100
print(f"\n  IRF8-high: n={n_irf8} ({pct_irf8:.1f}%)")

# Per-tumor IRF8-high proportions
print("\n  Per-tumor IRF8-high proportions:")
tumor_stats = []
for tumor in sorted(adata_cd8_tir.obs['tumor'].unique()):
    tmask = adata_cd8_tir.obs['tumor'] == tumor
    n_total = tmask.sum()
    n_irf8_t = (tmask & irf8_high_mask).sum()
    prop = n_irf8_t / n_total if n_total > 0 else 0
    tumor_stats.append({'tumor': tumor, 'n_total': n_total, 'n_irf8': n_irf8_t, 'prop': prop})
    print(f"    {tumor}: {n_irf8_t}/{n_total} ({prop:.1%})")

# Verify markers in IRF8-high cells
print("\n  IRF8-high lineage check:")
irf8_cells = adata_cd8_tir[irf8_high_mask]
for g in ['CD8A', 'CD3D', 'CD163', 'CSF1R', 'CD14']:
    if g in irf8_cells.var_names:
        pct = (irf8_cells[:, g].X.flatten() > 0).mean() * 100
        mean_e = irf8_cells[:, g].X.flatten().mean()
        print(f"    {g}: {pct:.1f}% expressing, mean={mean_e:.2f}")

# Exhaustion/effector markers
print("\n  IRF8-high vs Other markers:")
irf8_idx = adata_cd8_tir.obs['subtype_label'] == 'IRF8-high'
other_idx = adata_cd8_tir.obs['subtype_label'] != 'IRF8-high'
for g in ['PDCD1', 'HAVCR2', 'TIGIT', 'LAG3', 'TOX', 'PRF1', 'IFNG', 'XCL1', 'XCL2']:
    if g in adata_cd8_tir.var_names:
        v1 = adata_cd8_tir[irf8_idx, g].X.flatten()
        v2 = adata_cd8_tir[other_idx, g].X.flatten()
        pct1 = (v1 > 0).mean() * 100
        pct2 = (v2 > 0).mean() * 100
        _, pval = mannwhitneyu(v1, v2, alternative='two-sided')
        print(f"    {g}: IRF8-high {pct1:.1f}% vs Other {pct2:.1f}% (p={pval:.2e})")

###############################################################################
#  STEP 4: DEG analysis (IRF8-high vs Other)
###############################################################################
print("\nPerforming DEG analysis...")
adata_cd8_tir.obs['is_irf8_high'] = (adata_cd8_tir.obs['subtype_label'] == 'IRF8-high').astype(str)

# First, pre-filter genes: at least 10% in either group
irf8_mask_de = adata_cd8_tir.obs['is_irf8_high'] == 'True'
other_mask_de = adata_cd8_tir.obs['is_irf8_high'] == 'False'
pct_irf8_all = np.array((adata_cd8_tir[irf8_mask_de].X > 0).mean(axis=0)).flatten()
pct_other_all = np.array((adata_cd8_tir[other_mask_de].X > 0).mean(axis=0)).flatten()
gene_mask = (pct_irf8_all >= 0.10) | (pct_other_all >= 0.10)
genes_to_test = np.array(adata_cd8_tir.var_names)[gene_mask]
print(f"  Genes passing 10% filter: {len(genes_to_test)}")

# Subset to filtered genes for DEG analysis
adata_cd8_de = adata_cd8_tir[:, genes_to_test].copy()
adata_cd8_de.obs['is_irf8_high'] = adata_cd8_tir.obs['is_irf8_high'].values

sc.tl.rank_genes_groups(adata_cd8_de, groupby='is_irf8_high', groups=['True'],
                        reference='False', method='wilcoxon')

# Extract results
result = adata_cd8_de.uns['rank_genes_groups']
deg_tirosh = pd.DataFrame({
    'gene': result['names']['True'],
    'log2fc': result['logfoldchanges']['True'],
    'pval': result['pvals']['True'],
    'padj': result['pvals_adj']['True'],
})
# Debug: check raw results before filtering
print(f"  Raw DEG results: {len(deg_tirosh)} genes")
print(f"  Genes with padj < 0.05 (before pct filter): {(deg_tirosh['padj'] < 0.05).sum()}")
print(f"  Genes with pval < 0.05 (before pct filter): {(deg_tirosh['pval'] < 0.05).sum()}")
if len(deg_tirosh) > 0:
    print(f"  Min padj: {deg_tirosh['padj'].min():.2e}")
    print(f"  Top 5 by p-value:")
    print(deg_tirosh.nsmallest(5, 'pval')[['gene', 'log2fc', 'pval', 'padj']])

deg_sig_tir = deg_tirosh[deg_tirosh['padj'] < 0.05].copy()
n_up_tir = (deg_sig_tir['log2fc'] > 0).sum()
n_down_tir = (deg_sig_tir['log2fc'] < 0).sum()
print(f"  Tirosh DEGs (FDR<0.05): {len(deg_sig_tir)} ({n_up_tir} up, {n_down_tir} down)")

# Save DEG results
deg_sig_tir.to_csv(f'{OUT_DIR}/tirosh_irf8_degs.csv', index=False)

###############################################################################
#  STEP 5: Cross-cohort DEG overlap
###############################################################################
print("\nComputing 3-cohort DEG overlap...")

# Load pre-computed discovery and validation DEGs (from earlier pipeline)
deg_disc = pd.read_csv(f'{OUT_DIR}/irf8high_deg_discovery.csv')
deg_val = pd.read_csv(f'{OUT_DIR}/irf8high_deg_validation.csv')
# Harmonize column names if needed
if 'pvalue' in deg_disc.columns and 'padj' not in deg_disc.columns:
    deg_disc = deg_disc.rename(columns={'pvalue': 'pval'})
if 'pvalue' in deg_val.columns and 'padj' not in deg_val.columns:
    deg_val = deg_val.rename(columns={'pvalue': 'pval'})

# Compute overlaps
disc_genes = set(deg_disc[deg_disc['padj'] < 0.05]['gene'])
val_genes = set(deg_val[deg_val['padj'] < 0.05]['gene'])
tir_genes = set(deg_sig_tir['gene'])

overlap_dv = disc_genes & val_genes
overlap_dt = disc_genes & tir_genes
overlap_vt = val_genes & tir_genes
overlap_all = disc_genes & val_genes & tir_genes

print(f"\n  Discovery DEGs: {len(disc_genes)}")
print(f"  Validation DEGs: {len(val_genes)}")
print(f"  Tirosh DEGs: {len(tir_genes)}")
print(f"  Discovery ∩ Validation: {len(overlap_dv)}")
print(f"  Discovery ∩ Tirosh: {len(overlap_dt)}")
print(f"  Validation ∩ Tirosh: {len(overlap_vt)}")
print(f"  All three: {len(overlap_all)}")

# Direction concordance for 3-cohort overlap
disc_dir = dict(zip(deg_disc['gene'], np.sign(deg_disc['log2fc'])))
val_dir = dict(zip(deg_val['gene'], np.sign(deg_val['log2fc'])))
tir_dir = dict(zip(deg_sig_tir['gene'], np.sign(deg_sig_tir['log2fc'])))

concordant_3 = 0
discordant_3 = 0
for g in overlap_all:
    if g in disc_dir and g in val_dir and g in tir_dir:
        if disc_dir[g] == val_dir[g] == tir_dir[g]:
            concordant_3 += 1
        else:
            discordant_3 += 1

total_3 = concordant_3 + discordant_3
conc_pct = concordant_3 / total_3 * 100 if total_3 > 0 else 0
print(f"\n  3-cohort overlap: {total_3} genes, {concordant_3} concordant ({conc_pct:.1f}%)")

# Also compute Discovery-Tirosh overlap direction concordance
concordant_dt = 0
for g in overlap_dt:
    if g in disc_dir and g in tir_dir:
        if disc_dir[g] == tir_dir[g]:
            concordant_dt += 1
conc_dt_pct = concordant_dt / len(overlap_dt) * 100 if len(overlap_dt) > 0 else 0
print(f"  Discovery-Tirosh: {len(overlap_dt)} genes, {concordant_dt} concordant ({conc_dt_pct:.1f}%)")

# How many of the 3-cohort genes are up vs down
up_3 = sum(1 for g in overlap_all if disc_dir.get(g, 0) > 0)
down_3 = sum(1 for g in overlap_all if disc_dir.get(g, 0) < 0)
print(f"  3-cohort: {up_3} up, {down_3} down")

###############################################################################
#  STEP 5b: Save processed data for reuse by 09_final_figures.py
###############################################################################
print("\nSaving Tirosh processed data for figure generation...")
adata_cd8_tir.write(f'{OUT_DIR}/adata_cd8_tirosh.h5ad')
# Save DEG gene sets and concordance data
tirosh_fig_data = {
    'disc_genes': sorted(list(disc_genes)),
    'val_genes': sorted(list(val_genes)),
    'tir_genes': sorted(list(tir_genes)),
    'concordant_3': concordant_3,
    'total_3': total_3,
    'conc_pct': conc_pct,
    'available_tfs': available_tfs,
}
with open(f'{OUT_DIR}/tirosh_fig_data.json', 'w') as f:
    json.dump(tirosh_fig_data, f)
print("  Saved adata_cd8_tirosh.h5ad and tirosh_fig_data.json")

###############################################################################
#  SUMMARY STATISTICS
###############################################################################
print("\n" + "=" * 70)
print("GSE72056 VALIDATION SUMMARY")
print("=" * 70)
print(f"  Total CD8+ T cells: {adata_cd8_tir.n_obs}")
print(f"  Tumors: {adata_cd8_tir.obs['tumor'].nunique()}")
print(f"  IRF8-high: n={n_irf8} ({pct_irf8:.1f}%)")
print(f"  Mean IRF8 in IRF8-high: {float(adata_cd8_tir[irf8_high_mask, 'IRF8'].X.mean()):.2f}")
print(f"  Tirosh DEGs: {len(deg_sig_tir)} ({n_up_tir} up, {n_down_tir} down)")
print(f"  3-cohort shared DEGs: {len(overlap_all)} ({conc_pct:.1f}% direction concordance)")
print(f"  Discovery-Tirosh shared: {len(overlap_dt)} ({conc_dt_pct:.1f}% concordance)")
print("=== DONE ===")
