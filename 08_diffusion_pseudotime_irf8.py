"""
08_diffusion_pseudotime_irf8.py
Diffusion pseudotime analysis of CD8+ T cells to determine where IRF8-high
cells sit on the exhaustion trajectory.

Pipeline:
1. Load raw GSE115978 data
2. Filter to CD8+ T cells
3. Subcluster using top 15 SHAP TFs (resolution=0.25) to identify IRF8-high cells
4. Compute diffusion map on full transcriptome (top 2000 HVGs)
5. Set root to Memory/TCF7+ cells (earliest differentiation state)
6. Analyze IRF8-high positioning on pseudotime axis
7. Correlation analyses: IRF8 vs pseudotime, TOX, PDCD1, PRF1
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import scanpy as sc
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)
DATA_DIR = os.path.join(BASE_DIR, 'data')

print("=" * 70)
print("DIFFUSION PSEUDOTIME ANALYSIS: IRF8-high CD8+ T cells")
print("=" * 70)

###############################################################################
# STEP 1: Load raw data from CSV
###############################################################################
print("\n[1] Loading raw GSE115978 data...")
annot = pd.read_csv(f'{DATA_DIR}/GSE115978_cell.annotations.csv.gz')
annot = annot.set_index('cells')
tpm = pd.read_csv(f'{DATA_DIR}/GSE115978_tpm.csv.gz', index_col=0)
tpm_T = tpm.T  # genes x cells -> cells x genes

adata_all = sc.AnnData(X=tpm_T.values.astype(np.float32))
adata_all.obs_names = tpm_T.index.tolist()
adata_all.var_names = tpm_T.columns.tolist()

# Add annotations
common_cells = adata_all.obs_names.intersection(annot.index)
adata_all = adata_all[common_cells, :].copy()
for col in annot.columns:
    adata_all.obs[col] = annot.loc[adata_all.obs_names, col].values
adata_all = adata_all[adata_all.obs['cell.types'] != '?'].copy()

# Log-transform TPM
sc.pp.log1p(adata_all)
print(f"    Total cells after QC: {adata_all.n_obs}")
print(f"    Total genes: {adata_all.n_vars}")

###############################################################################
# STEP 2: Filter to CD8+ T cells
###############################################################################
print("\n[2] Filtering to CD8+ T cells...")
adata_cd8 = adata_all[adata_all.obs['cell.types'] == 'T.CD8'].copy()
print(f"    CD8+ T cells: {adata_cd8.n_obs}")
print(f"    Treatment groups: {adata_cd8.obs['treatment.group'].value_counts().to_dict()}")

# Store raw log1p expression for gene correlations later
adata_cd8.raw = adata_cd8.copy()

###############################################################################
# STEP 3: TF-based subclustering to identify IRF8-high cells
###############################################################################
print("\n[3] TF-based subclustering (top 15 SHAP TFs, resolution=0.25)...")

top_cd8_tfs = ['IKZF3', 'ID2', 'TOX', 'TCF4', 'MEF2C', 'BCL11A', 'PRDM1',
               'IRF8', 'ETV1', 'WWTR1', 'EOMES', 'BCL11B', 'RUNX3', 'LEF1', 'TCF7']
available_tfs = [tf for tf in top_cd8_tfs if tf in adata_cd8.var_names]
print(f"    Available TFs: {len(available_tfs)}/{len(top_cd8_tfs)}: {available_tfs}")

# Subset to TF features, scale, PCA, cluster
adata_cd8_tf = adata_cd8[:, available_tfs].copy()
sc.pp.scale(adata_cd8_tf, max_value=10)
n_comps = min(10, len(available_tfs) - 1)
sc.tl.pca(adata_cd8_tf, n_comps=n_comps, random_state=RANDOM_STATE)
sc.pp.neighbors(adata_cd8_tf, n_pcs=n_comps, n_neighbors=15, random_state=RANDOM_STATE)
sc.tl.leiden(adata_cd8_tf, resolution=0.25, random_state=RANDOM_STATE, key_added='tf_cluster')
sc.tl.umap(adata_cd8_tf, random_state=RANDOM_STATE)

# Transfer cluster labels back to main object
adata_cd8.obs['tf_cluster'] = adata_cd8_tf.obs['tf_cluster'].values
adata_cd8.obsm['X_umap_tf'] = adata_cd8_tf.obsm['X_umap']

n_clusters = adata_cd8.obs['tf_cluster'].nunique()
print(f"    Discovered {n_clusters} clusters at resolution=0.25")

###############################################################################
# STEP 3b: Identify and label clusters based on marker expression
###############################################################################
print("\n[3b] Characterizing clusters...")

# Signature gene sets
sig_genes = {
    'exhaustion': ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'TOX', 'ENTPD1'],
    'effector': ['GZMA', 'GZMB', 'GZMK', 'PRF1', 'IFNG', 'NKG7', 'GNLY'],
    'memory': ['TCF7', 'LEF1', 'CCR7', 'SELL', 'IL7R'],
    'proliferation': ['MKI67', 'TOP2A', 'PCNA'],
}

for sig_name, genes in sig_genes.items():
    avail = [g for g in genes if g in adata_cd8.var_names]
    if avail:
        sc.tl.score_genes(adata_cd8, avail, score_name=f'{sig_name}_score',
                          random_state=RANDOM_STATE)

# Compute mean IRF8 per cluster and identify IRF8-high cluster
irf8_idx = list(adata_cd8.var_names).index('IRF8')
adata_cd8.obs['IRF8_expr'] = np.asarray(adata_cd8.X[:, irf8_idx]).flatten()

cluster_stats = []
for cl in sorted(adata_cd8.obs['tf_cluster'].unique(), key=int):
    mask = adata_cd8.obs['tf_cluster'] == cl
    n = mask.sum()
    irf8_mean = adata_cd8.obs.loc[mask, 'IRF8_expr'].mean()
    exhaust = adata_cd8.obs.loc[mask, 'exhaustion_score'].mean()
    effector = adata_cd8.obs.loc[mask, 'effector_score'].mean()
    memory = adata_cd8.obs.loc[mask, 'memory_score'].mean()
    naive_pct = (adata_cd8.obs.loc[mask, 'treatment.group'] == 'treatment.naive').mean() * 100

    # Key TF means
    tox_mean = np.asarray(adata_cd8.X[mask.values, list(adata_cd8.var_names).index('TOX')]).mean()
    tcf7_mean = np.asarray(adata_cd8.X[mask.values, list(adata_cd8.var_names).index('TCF7')]).mean()

    cluster_stats.append({
        'cluster': cl, 'n_cells': n, 'IRF8_mean': irf8_mean,
        'exhaustion': exhaust, 'effector': effector, 'memory': memory,
        'naive_pct': naive_pct, 'TOX_mean': tox_mean, 'TCF7_mean': tcf7_mean
    })

df_clusters = pd.DataFrame(cluster_stats)

print(f"\n    {'Cluster':<10} {'N':<8} {'IRF8':<8} {'Exhaust':<10} {'Effector':<10} {'Memory':<10} {'TCF7':<8} {'TOX':<8} {'%Naive':<8}")
print("    " + "-" * 80)
for _, row in df_clusters.iterrows():
    print(f"    {row['cluster']:<10} {row['n_cells']:<8} {row['IRF8_mean']:<8.3f} "
          f"{row['exhaustion']:<10.3f} {row['effector']:<10.3f} {row['memory']:<10.3f} "
          f"{row['TCF7_mean']:<8.3f} {row['TOX_mean']:<8.3f} {row['naive_pct']:<8.1f}")

# Identify IRF8-high cluster (highest mean IRF8)
irf8_high_cluster = df_clusters.loc[df_clusters['IRF8_mean'].idxmax(), 'cluster']
print(f"\n    >>> IRF8-high cluster: {irf8_high_cluster} (mean IRF8 = {df_clusters.loc[df_clusters['IRF8_mean'].idxmax(), 'IRF8_mean']:.3f})")

# Identify Memory/TCF7+ cluster (highest TCF7 expression)
memory_cluster = df_clusters.loc[df_clusters['TCF7_mean'].idxmax(), 'cluster']
print(f"    >>> Memory/TCF7+ cluster: {memory_cluster} (mean TCF7 = {df_clusters.loc[df_clusters['TCF7_mean'].idxmax(), 'TCF7_mean']:.3f})")

# Assign biological labels
def assign_label(row):
    if row['cluster'] == irf8_high_cluster:
        return 'IRF8-high'
    elif row['cluster'] == memory_cluster:
        return 'Memory (TCF7+)'
    elif row['exhaustion'] > 0.1 and row['TOX_mean'] > 0.8:
        return 'Exhausted (TOX+)'
    elif row['effector'] > 0.5:
        return 'Effector'
    elif row['memory'] > 0:
        return 'Memory-like'
    else:
        return f'Cluster-{row["cluster"]}'

df_clusters['label'] = df_clusters.apply(assign_label, axis=1)
cluster_to_label = dict(zip(df_clusters['cluster'], df_clusters['label']))
adata_cd8.obs['subtype_label'] = adata_cd8.obs['tf_cluster'].map(cluster_to_label)

print("\n    Final subtype assignments:")
print(adata_cd8.obs['subtype_label'].value_counts().to_string())

###############################################################################
# STEP 4: Compute diffusion map using top 2000 HVGs (full transcriptome)
###############################################################################
print("\n[4] Computing diffusion map (top 2000 HVGs)...")

# Work on a copy for HVG selection and diffusion map
adata_dm = adata_cd8.copy()

# HVG selection: scanpy flavor for non-count data (seurat_v3 needs counts)
# Since data is log1p(TPM), use 'seurat' flavor
sc.pp.highly_variable_genes(adata_dm, n_top_genes=2000, flavor='seurat')
n_hvg = adata_dm.var['highly_variable'].sum()
print(f"    Selected {n_hvg} highly variable genes")

# Scale the data
sc.pp.scale(adata_dm, max_value=10)

# PCA on HVGs
sc.tl.pca(adata_dm, n_comps=50, use_highly_variable=True, random_state=RANDOM_STATE)
print(f"    PCA computed (50 components)")

# Neighbors for diffusion map
sc.pp.neighbors(adata_dm, n_pcs=30, n_neighbors=20, random_state=RANDOM_STATE)
print(f"    Neighbor graph computed (n_pcs=30, n_neighbors=20)")

# Diffusion map
sc.tl.diffmap(adata_dm, n_comps=15)
print(f"    Diffusion map computed (15 components)")

# Transfer diffusion map coordinates back
adata_cd8.obsm['X_diffmap'] = adata_dm.obsm['X_diffmap']
adata_cd8.uns['diffmap_evals'] = adata_dm.uns['diffmap_evals']
# Also need iroot and neighbors for dpt
adata_cd8.obsp = adata_dm.obsp.copy()
adata_cd8.uns['neighbors'] = adata_dm.uns['neighbors'].copy()

###############################################################################
# STEP 5: Set root to Memory/TCF7+ cells (earliest differentiation state)
###############################################################################
print("\n[5] Setting root cell to Memory/TCF7+ cluster...")

# Find the cell in the Memory/TCF7+ cluster with highest TCF7 expression
memory_mask = adata_cd8.obs['subtype_label'] == 'Memory (TCF7+)'
if 'TCF7' not in adata_cd8.var_names:
    raise RuntimeError("TCF7 not found in expression matrix; cannot define pseudotime root.")
tcf7_idx = list(adata_cd8.var_names).index('TCF7')
tcf7_expr = np.asarray(adata_cd8.X[:, tcf7_idx]).flatten()
used_memory_label_root = memory_mask.sum() > 0

if used_memory_label_root:
    # Among memory-labelled cells, choose the highest TCF7 cell
    memory_indices = np.where(memory_mask.values)[0]
    memory_tcf7 = tcf7_expr[memory_indices]
    root_local = np.argmax(memory_tcf7)
    root_cell_idx = memory_indices[root_local]
else:
    # Fallback for robustness: if no Memory label exists, use global max TCF7 cell
    print("    WARNING: no 'Memory (TCF7+)' label found; using global max-TCF7 cell as root.")
    root_cell_idx = int(np.argmax(tcf7_expr))

root_cell_name = adata_cd8.obs_names[root_cell_idx]
print(f"    Root cell: {root_cell_name} (TCF7 = {tcf7_expr[root_cell_idx]:.3f})")

adata_cd8.uns['iroot'] = root_cell_idx

###############################################################################
# STEP 5b: Compute diffusion pseudotime
###############################################################################
print("\n[5b] Computing diffusion pseudotime...")
sc.tl.dpt(adata_cd8, n_dcs=10)
print(f"    DPT computed successfully")
print(f"    Pseudotime range: [{adata_cd8.obs['dpt_pseudotime'].min():.4f}, {adata_cd8.obs['dpt_pseudotime'].max():.4f}]")

# Handle inf values (disconnected components)
inf_mask = np.isinf(adata_cd8.obs['dpt_pseudotime'])
n_inf = inf_mask.sum()
if n_inf > 0:
    print(f"    WARNING: {n_inf} cells have infinite pseudotime (disconnected components)")
    # Replace inf with max finite value
    max_finite = adata_cd8.obs.loc[~inf_mask, 'dpt_pseudotime'].max()
    adata_cd8.obs.loc[inf_mask, 'dpt_pseudotime'] = max_finite
    print(f"    Replaced with max finite value: {max_finite:.4f}")

###############################################################################
# STEP 6: Analyze where IRF8-high falls on the pseudotime axis
###############################################################################
print("\n" + "=" * 70)
print("STEP 6: PSEUDOTIME ANALYSIS BY SUBTYPE")
print("=" * 70)

pt = adata_cd8.obs['dpt_pseudotime']

print(f"\n{'Subtype':<25} {'N':<8} {'Mean PT':<10} {'Median PT':<10} {'SD PT':<10} {'Min':<8} {'Max':<8}")
print("-" * 80)

subtype_pt_stats = []
for label in sorted(adata_cd8.obs['subtype_label'].unique()):
    mask = adata_cd8.obs['subtype_label'] == label
    pt_vals = pt[mask]
    row = {
        'subtype': label,
        'n_cells': mask.sum(),
        'mean_pt': pt_vals.mean(),
        'median_pt': pt_vals.median(),
        'sd_pt': pt_vals.std(),
        'min_pt': pt_vals.min(),
        'max_pt': pt_vals.max(),
    }
    subtype_pt_stats.append(row)
    print(f"{label:<25} {row['n_cells']:<8} {row['mean_pt']:<10.4f} {row['median_pt']:<10.4f} "
          f"{row['sd_pt']:<10.4f} {row['min_pt']:<8.4f} {row['max_pt']:<8.4f}")

df_pt_stats = pd.DataFrame(subtype_pt_stats).sort_values('mean_pt')

# Rank subtypes by pseudotime
print("\n--- Subtypes ranked by mean pseudotime (early -> late) ---")
for i, (_, row) in enumerate(df_pt_stats.iterrows()):
    marker = " <<<" if row['subtype'] == 'IRF8-high' else ""
    print(f"    {i+1}. {row['subtype']:<25} (mean PT = {row['mean_pt']:.4f}){marker}")

# Statistical test: IRF8-high vs each other subtype
print("\n--- IRF8-high vs other subtypes (Mann-Whitney U test) ---")
irf8_pt = pt[adata_cd8.obs['subtype_label'] == 'IRF8-high']
print(f"    IRF8-high pseudotime: mean={irf8_pt.mean():.4f}, median={irf8_pt.median():.4f}, n={len(irf8_pt)}")

comparison_results = []
for label in sorted(adata_cd8.obs['subtype_label'].unique()):
    if label == 'IRF8-high':
        continue
    other_pt = pt[adata_cd8.obs['subtype_label'] == label]
    stat, pval = stats.mannwhitneyu(irf8_pt, other_pt, alternative='two-sided')
    effect_size = stat / (len(irf8_pt) * len(other_pt))  # rank-biserial
    comparison_results.append({
        'comparison': f'IRF8-high vs {label}',
        'IRF8_high_mean_pt': irf8_pt.mean(),
        'other_mean_pt': other_pt.mean(),
        'U_statistic': stat,
        'p_value': pval,
        'rank_biserial': 2 * effect_size - 1,
        'direction': 'IRF8-high later' if irf8_pt.mean() > other_pt.mean() else 'IRF8-high earlier'
    })
    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
    print(f"    vs {label:<25}: U={stat:.0f}, p={pval:.2e} {sig}, "
          f"diff={irf8_pt.mean() - other_pt.mean():+.4f} ({comparison_results[-1]['direction']})")

df_comparisons = pd.DataFrame(comparison_results)

# Apply Benjamini-Hochberg correction
if len(df_comparisons) > 0:
    _, padj, _, _ = multipletests(df_comparisons["p_value"], method="fdr_bh")
    df_comparisons["padj"] = padj
    print("\n--- Benjamini-Hochberg adjusted p-values ---")
    for _, row in df_comparisons.iterrows():
        sig = "***" if row["padj"] < 0.001 else ("**" if row["padj"] < 0.01 else ("*" if row["padj"] < 0.05 else "ns"))
        print(f"    {row['comparison']:<35}: padj={row['padj']:.2e} {sig}")

# Kruskal-Wallis test across all subtypes
print("\n--- Kruskal-Wallis test (all subtypes) ---")
groups = [pt[adata_cd8.obs['subtype_label'] == label].values
          for label in sorted(adata_cd8.obs['subtype_label'].unique())]
kw_stat, kw_pval = stats.kruskal(*groups)
print(f"    H = {kw_stat:.2f}, p = {kw_pval:.2e}")

###############################################################################
# STEP 7: Correlation analyses
###############################################################################
print("\n" + "=" * 70)
print("STEP 7: CORRELATION ANALYSES")
print("=" * 70)

# Get gene expression values (from raw log1p data)
genes_of_interest = ['IRF8', 'TOX', 'PDCD1', 'PRF1', 'TCF7', 'HAVCR2', 'LAG3',
                     'TIGIT', 'GZMB', 'EOMES', 'IKZF3', 'ID2', 'ENTPD1']
gene_expr = {}
for g in genes_of_interest:
    if g in adata_cd8.var_names:
        idx = list(adata_cd8.var_names).index(g)
        gene_expr[g] = np.asarray(adata_cd8.raw.X[:, idx]).flatten()

# 7a: IRF8 expression vs pseudotime
print("\n--- 7a: IRF8 expression vs pseudotime ---")
valid_mask = ~np.isinf(pt.values) & ~np.isnan(pt.values)
r_irf8_pt, p_irf8_pt = stats.spearmanr(gene_expr['IRF8'][valid_mask], pt.values[valid_mask])
r_irf8_pt_pearson, p_irf8_pt_pearson = stats.pearsonr(gene_expr['IRF8'][valid_mask], pt.values[valid_mask])
print(f"    Spearman: rho = {r_irf8_pt:.4f}, p = {p_irf8_pt:.2e}")
print(f"    Pearson:  r   = {r_irf8_pt_pearson:.4f}, p = {p_irf8_pt_pearson:.2e}")

# 7b: IRF8 vs key genes (all CD8+ T cells)
print("\n--- 7b: IRF8 vs key genes (Spearman, all CD8+ T cells) ---")
irf8_corr_results = []
for g in genes_of_interest:
    if g == 'IRF8' or g not in gene_expr:
        continue
    r, p = stats.spearmanr(gene_expr['IRF8'], gene_expr[g])
    irf8_corr_results.append({'gene': g, 'spearman_rho': r, 'p_value': p})
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"    IRF8 vs {g:<10}: rho = {r:+.4f}, p = {p:.2e} {sig}")

df_irf8_corr = pd.DataFrame(irf8_corr_results).sort_values('spearman_rho', ascending=False)

# 7c: Pseudotime vs key exhaustion/effector genes
print("\n--- 7c: Pseudotime vs key genes (Spearman) ---")
pt_corr_results = []
for g in genes_of_interest:
    if g not in gene_expr:
        continue
    r, p = stats.spearmanr(pt.values[valid_mask], gene_expr[g][valid_mask])
    pt_corr_results.append({'gene': g, 'spearman_rho': r, 'p_value': p})
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"    PT vs {g:<10}: rho = {r:+.4f}, p = {p:.2e} {sig}")

df_pt_corr = pd.DataFrame(pt_corr_results).sort_values('spearman_rho', ascending=False)

# 7d: IRF8 correlations within IRF8-high cells only
print("\n--- 7d: IRF8 vs genes WITHIN IRF8-high cells only ---")
irf8_mask = (adata_cd8.obs['subtype_label'] == 'IRF8-high').values
irf8_within_corr = []
for g in ['TOX', 'PDCD1', 'PRF1', 'GZMB', 'HAVCR2', 'LAG3', 'TCF7', 'EOMES']:
    if g not in gene_expr:
        continue
    r, p = stats.spearmanr(gene_expr['IRF8'][irf8_mask], gene_expr[g][irf8_mask])
    irf8_within_corr.append({'gene': g, 'spearman_rho': r, 'p_value': p})
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"    IRF8 vs {g:<10} (within IRF8-high): rho = {r:+.4f}, p = {p:.2e} {sig}")

###############################################################################
# STEP 7e: Treatment effect on pseudotime
###############################################################################
print("\n--- 7e: Pseudotime by treatment group ---")
for label in sorted(adata_cd8.obs['subtype_label'].unique()):
    mask_sub = adata_cd8.obs['subtype_label'] == label
    for tx in ['treatment.naive', 'post.treatment']:
        mask_tx = adata_cd8.obs['treatment.group'] == tx
        combined = mask_sub & mask_tx
        if combined.sum() > 0:
            mean_pt = pt[combined].mean()
            n = combined.sum()
            # Abbreviated
            tx_short = 'Pre' if 'naive' in tx else 'Post'
            print(f"    {label:<25} {tx_short:<6}: mean PT = {mean_pt:.4f} (n={n})")

# Test treatment effect within IRF8-high
print("\n--- Treatment effect within IRF8-high cells ---")
irf8_naive = pt[(adata_cd8.obs['subtype_label'] == 'IRF8-high') & 
                (adata_cd8.obs['treatment.group'] == 'treatment.naive')]
irf8_post = pt[(adata_cd8.obs['subtype_label'] == 'IRF8-high') & 
               (adata_cd8.obs['treatment.group'] == 'post.treatment')]
if len(irf8_naive) > 0 and len(irf8_post) > 0:
    stat, pval = stats.mannwhitneyu(irf8_naive, irf8_post, alternative='two-sided')
    print(f"    Pre-treatment:  mean PT = {irf8_naive.mean():.4f} (n={len(irf8_naive)})")
    print(f"    Post-treatment: mean PT = {irf8_post.mean():.4f} (n={len(irf8_post)})")
    print(f"    Mann-Whitney U = {stat:.0f}, p = {pval:.2e}")

###############################################################################
# STEP 8: Save results
###############################################################################
print("\n" + "=" * 70)
print("STEP 8: SAVING RESULTS")
print("=" * 70)

# 8a: Per-cell pseudotime data
df_cell = pd.DataFrame({
    'cell': adata_cd8.obs_names,
    'subtype_label': adata_cd8.obs['subtype_label'].values,
    'tf_cluster': adata_cd8.obs['tf_cluster'].values,
    'treatment_group': adata_cd8.obs['treatment.group'].values,
    'sample': adata_cd8.obs['samples'].values,
    'dpt_pseudotime': adata_cd8.obs['dpt_pseudotime'].values,
    'diffmap_DC1': adata_cd8.obsm['X_diffmap'][:, 1],
    'diffmap_DC2': adata_cd8.obsm['X_diffmap'][:, 2],
    'IRF8_expr': gene_expr['IRF8'],
    'TOX_expr': gene_expr['TOX'],
    'PDCD1_expr': gene_expr.get('PDCD1', np.nan),
    'PRF1_expr': gene_expr.get('PRF1', np.nan),
    'TCF7_expr': gene_expr.get('TCF7', np.nan),
    'GZMB_expr': gene_expr.get('GZMB', np.nan),
    'exhaustion_score': adata_cd8.obs['exhaustion_score'].values,
    'effector_score': adata_cd8.obs['effector_score'].values,
    'memory_score': adata_cd8.obs['memory_score'].values,
})
df_cell.to_csv(f'{OUT_DIR}/dpt_cd8_cell_data.csv', index=False)
print(f"    Saved: dpt_cd8_cell_data.csv ({df_cell.shape[0]} cells)")

# 8b: Subtype pseudotime statistics
df_pt_stats.to_csv(f'{OUT_DIR}/dpt_subtype_pseudotime_stats.csv', index=False)
print(f"    Saved: dpt_subtype_pseudotime_stats.csv")

# 8c: Pairwise comparisons
df_comparisons.to_csv(f'{OUT_DIR}/dpt_irf8_vs_subtypes_tests.csv', index=False)
print(f"    Saved: dpt_irf8_vs_subtypes_tests.csv")

# 8d: IRF8 correlation results
df_irf8_corr.to_csv(f'{OUT_DIR}/dpt_irf8_gene_correlations.csv', index=False)
print(f"    Saved: dpt_irf8_gene_correlations.csv")

# 8e: Pseudotime-gene correlations
df_pt_corr.to_csv(f'{OUT_DIR}/dpt_pseudotime_gene_correlations.csv', index=False)
print(f"    Saved: dpt_pseudotime_gene_correlations.csv")

# 8f: Cluster characterization
df_clusters['label'] = df_clusters.apply(assign_label, axis=1)
df_clusters.to_csv(f'{OUT_DIR}/dpt_cluster_characterization.csv', index=False)
print(f"    Saved: dpt_cluster_characterization.csv")

###############################################################################
# COMPREHENSIVE SUMMARY
###############################################################################
print("\n" + "=" * 70)
print("COMPREHENSIVE SUMMARY")
print("=" * 70)

# Where does IRF8-high sit?
irf8_rank = list(df_pt_stats['subtype'].values).index('IRF8-high') + 1
total_subtypes = len(df_pt_stats)
irf8_mean_pt = df_pt_stats.loc[df_pt_stats['subtype'] == 'IRF8-high', 'mean_pt'].values[0]
memory_rows = df_pt_stats.loc[df_pt_stats['subtype'] == 'Memory (TCF7+)']
memory_mean_pt = memory_rows['mean_pt'].values[0] if len(memory_rows) > 0 else np.nan

print(f"\n1. PSEUDOTIME POSITIONING:")
print(f"   - IRF8-high cells rank {irf8_rank}/{total_subtypes} on pseudotime axis (1=earliest)")
print(f"   - IRF8-high mean pseudotime: {irf8_mean_pt:.4f}")
if np.isfinite(memory_mean_pt):
    print(f"   - Memory (TCF7+) mean pseudotime: {memory_mean_pt:.4f} (root)")
else:
    print("   - Memory (TCF7+) subtype absent; root was set by max TCF7 fallback.")

if np.isfinite(memory_mean_pt) and irf8_mean_pt > memory_mean_pt:
    pt_position = "LATER than Memory/TCF7+ (more differentiated)"
elif np.isfinite(memory_mean_pt):
    pt_position = "EARLIER or SIMILAR to Memory/TCF7+ (less differentiated)"
else:
    pt_position = "interpreted relative to max-TCF7 root (Memory label missing)"
print(f"   - IRF8-high is {pt_position}")

# Check if IRF8 is at intermediate/late/terminal position
pt_range = df_pt_stats['mean_pt'].max() - df_pt_stats['mean_pt'].min()
irf8_relative = (irf8_mean_pt - df_pt_stats['mean_pt'].min()) / pt_range if pt_range > 0 else 0
if irf8_relative < 0.33:
    position_desc = "EARLY (progenitor-like)"
elif irf8_relative < 0.66:
    position_desc = "INTERMEDIATE (transitional)"
else:
    position_desc = "LATE (terminally differentiated)"
print(f"   - Relative position: {irf8_relative:.2f} ({position_desc})")

print(f"\n2. IRF8-PSEUDOTIME CORRELATION:")
print(f"   - Spearman rho = {r_irf8_pt:.4f}, p = {p_irf8_pt:.2e}")
if r_irf8_pt > 0:
    print(f"   - IRF8 expression INCREASES along pseudotime (toward exhaustion)")
else:
    print(f"   - IRF8 expression DECREASES along pseudotime")

print(f"\n3. IRF8 CO-EXPRESSION PATTERNS:")
for _, row in df_irf8_corr.iterrows():
    direction = "positive" if row['spearman_rho'] > 0 else "negative"
    print(f"   - IRF8 vs {row['gene']:<10}: rho={row['spearman_rho']:+.4f} ({direction})")

print(f"\n4. BIOLOGICAL INTERPRETATION:")
# Check correlations with key markers
tox_corr = df_irf8_corr.loc[df_irf8_corr['gene'] == 'TOX', 'spearman_rho'].values[0] if 'TOX' in df_irf8_corr['gene'].values else 0
pdcd1_corr = df_irf8_corr.loc[df_irf8_corr['gene'] == 'PDCD1', 'spearman_rho'].values[0] if 'PDCD1' in df_irf8_corr['gene'].values else 0
prf1_corr = df_irf8_corr.loc[df_irf8_corr['gene'] == 'PRF1', 'spearman_rho'].values[0] if 'PRF1' in df_irf8_corr['gene'].values else 0

if tox_corr > 0.1 and pdcd1_corr > 0.1:
    print(f"   - IRF8 co-expressed with TOX (rho={tox_corr:+.3f}) and PDCD1 (rho={pdcd1_corr:+.3f})")
    print(f"   - Suggests IRF8-high cells are associated with exhaustion program")
elif tox_corr < -0.1:
    print(f"   - IRF8 NEGATIVELY correlated with TOX (rho={tox_corr:+.3f})")
    print(f"   - Suggests IRF8-high cells are distinct from canonical exhaustion")

if prf1_corr > 0.1:
    print(f"   - IRF8 positively correlated with PRF1 (rho={prf1_corr:+.3f}) - cytotoxic potential")
elif prf1_corr < -0.1:
    print(f"   - IRF8 negatively correlated with PRF1 (rho={prf1_corr:+.3f}) - reduced cytotoxicity")

print(f"\n5. N CELLS PER SUBTYPE:")
for _, row in df_pt_stats.iterrows():
    print(f"   - {row['subtype']:<25}: n={row['n_cells']}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
