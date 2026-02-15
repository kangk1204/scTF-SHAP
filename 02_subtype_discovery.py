"""
02_subtype_discovery.py
SHAP-derived TF-based subclustering to discover novel cell subtypes
Focus: CD8+ T cells (most clinically relevant) and Malignant cells
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import scanpy as sc
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)
DATA_DIR = os.path.join(BASE_DIR, 'data')

print("=" * 60)
print("SHAP-derived TF-based subclustering analysis")
print("=" * 60)

# ── Load data ────────────────────────────────────────────────
print("\n[1] Loading data...")
annot = pd.read_csv(f'{DATA_DIR}/GSE115978_cell.annotations.csv.gz')
annot = annot.set_index('cells')
tpm = pd.read_csv(f'{DATA_DIR}/GSE115978_tpm.csv.gz', index_col=0)
tpm_T = tpm.T
adata_all = sc.AnnData(X=tpm_T.values.astype(np.float32))
adata_all.obs_names = tpm_T.index.tolist()
adata_all.var_names = tpm_T.columns.tolist()
common_cells = adata_all.obs_names.intersection(annot.index)
adata_all = adata_all[common_cells, :].copy()
for col in annot.columns:
    adata_all.obs[col] = annot.loc[adata_all.obs_names, col].values
adata_all = adata_all[adata_all.obs['cell.types'] != '?'].copy()
sc.pp.log1p(adata_all)

# ── Load SHAP TF results ────────────────────────────────────
tf_cd8 = pd.read_csv(f'{OUT_DIR}/tf_importance_T_CD8.csv')
tf_mal = pd.read_csv(f'{OUT_DIR}/tf_importance_Mal.csv')

###############################################################################
# PART 1: CD8+ T cell subclustering using top SHAP TFs
###############################################################################
print("\n[2] CD8+ T cell subclustering using top SHAP TFs...")

adata_cd8 = adata_all[adata_all.obs['cell.types'] == 'T.CD8'].copy()
print(f"    CD8+ T cells: {adata_cd8.n_obs}")

# Top 15 SHAP TFs for CD8
top_cd8_tfs = tf_cd8.head(15)['TF'].tolist()
available_tfs = [tf for tf in top_cd8_tfs if tf in adata_cd8.var_names]
print(f"    Using TFs: {available_tfs}")

# Subset to TF features only, then PCA + clustering
adata_cd8_tf = adata_cd8[:, available_tfs].copy()
sc.pp.scale(adata_cd8_tf, max_value=10)
sc.tl.pca(adata_cd8_tf, n_comps=min(10, len(available_tfs) - 1))
sc.pp.neighbors(adata_cd8_tf, n_pcs=min(10, len(available_tfs) - 1), n_neighbors=15)
sc.tl.leiden(adata_cd8_tf, resolution=0.5, key_added='tf_subtype')
sc.tl.umap(adata_cd8_tf)

# Transfer results back
adata_cd8.obs['tf_subtype'] = adata_cd8_tf.obs['tf_subtype'].values
adata_cd8.obsm['X_umap_tf'] = adata_cd8_tf.obsm['X_umap']

n_subtypes = adata_cd8.obs['tf_subtype'].nunique()
print(f"    Discovered {n_subtypes} CD8+ T cell subtypes")

# Characterize each subtype
print("\n[3] Characterizing CD8+ T cell subtypes...")

# Key signature genes for characterization
exhaustion_genes = ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'TOX', 'ENTPD1']
effector_genes = ['GZMA', 'GZMB', 'GZMK', 'PRF1', 'IFNG', 'NKG7', 'GNLY']
memory_genes = ['TCF7', 'LEF1', 'CCR7', 'SELL', 'IL7R']
proliferation_genes = ['MKI67', 'TOP2A', 'PCNA']

for sig_name, sig_genes in [('exhaustion', exhaustion_genes), ('effector', effector_genes),
                             ('memory', memory_genes), ('proliferation', proliferation_genes)]:
    available_sig = [g for g in sig_genes if g in adata_cd8.var_names]
    if available_sig:
        sc.tl.score_genes(adata_cd8, available_sig, score_name=f'{sig_name}_score')

# Print subtype characterization
print(f"\n    {'Subtype':<10} {'N cells':<10} {'%Naive':<10} {'%Post-tx':<10} {'Exhaust':<10} {'Effector':<10} {'Memory':<10}")
print("    " + "-" * 70)

subtype_stats = []
for st in sorted(adata_cd8.obs['tf_subtype'].unique()):
    mask = adata_cd8.obs['tf_subtype'] == st
    n = mask.sum()
    naive_frac = (adata_cd8.obs.loc[mask, 'treatment.group'] == 'treatment.naive').mean() * 100
    post_frac = 100 - naive_frac

    exhaust = adata_cd8.obs.loc[mask, 'exhaustion_score'].mean() if 'exhaustion_score' in adata_cd8.obs else 0
    effector = adata_cd8.obs.loc[mask, 'effector_score'].mean() if 'effector_score' in adata_cd8.obs else 0
    memory = adata_cd8.obs.loc[mask, 'memory_score'].mean() if 'memory_score' in adata_cd8.obs else 0

    print(f"    {st:<10} {n:<10} {naive_frac:<10.1f} {post_frac:<10.1f} {exhaust:<10.3f} {effector:<10.3f} {memory:<10.3f}")

    # Mean TF expression per subtype
    tf_means = {}
    for tf in available_tfs:
        tf_means[tf] = adata_cd8[mask, tf].X.mean()

    subtype_stats.append({
        'subtype': st, 'n_cells': n,
        'naive_pct': naive_frac, 'post_pct': post_frac,
        'exhaustion': exhaust, 'effector': effector, 'memory': memory,
        **tf_means
    })

subtype_df = pd.DataFrame(subtype_stats)
subtype_df.to_csv(f'{OUT_DIR}/cd8_tf_subtypes.csv', index=False)

# Identify the most treatment-enriched subtypes
print("\n[4] Treatment-enriched subtypes:")
from scipy.stats import chi2_contingency, fisher_exact

overall_naive = (adata_cd8.obs['treatment.group'] == 'treatment.naive').mean()
for st in sorted(adata_cd8.obs['tf_subtype'].unique()):
    mask = adata_cd8.obs['tf_subtype'] == st
    n = mask.sum()
    naive_n = (adata_cd8.obs.loc[mask, 'treatment.group'] == 'treatment.naive').sum()
    post_n = n - naive_n
    rest_naive = (adata_cd8.obs.loc[~mask, 'treatment.group'] == 'treatment.naive').sum()
    rest_post = (~mask).sum() - rest_naive

    # Fisher's exact test
    table = [[naive_n, post_n], [rest_naive, rest_post]]
    odds, pval = fisher_exact(table)
    enrichment = 'Naive-enriched' if odds > 1 else 'Post-tx-enriched'
    sig = '*' if pval < 0.05 else ''
    print(f"    Subtype {st}: {naive_n} naive, {post_n} post-tx | OR={odds:.2f}, p={pval:.3e} {enrichment} {sig}")

# TF expression heatmap data for subtypes
print("\n[5] Top TFs per subtype (mean expression):")
for st in sorted(adata_cd8.obs['tf_subtype'].unique()):
    mask = adata_cd8.obs['tf_subtype'] == st
    print(f"\n    Subtype {st} (n={mask.sum()}):")
    tf_expr = {}
    for tf in available_tfs:
        tf_expr[tf] = adata_cd8[mask, tf].X.mean()
    sorted_tfs = sorted(tf_expr.items(), key=lambda x: x[1], reverse=True)
    for tf, val in sorted_tfs[:5]:
        print(f"      {tf}: {val:.3f}")

# Assign biological labels
print("\n[6] Assigning biological labels to CD8+ subtypes...")
labels = {}
for st in sorted(adata_cd8.obs['tf_subtype'].unique()):
    mask = adata_cd8.obs['tf_subtype'] == st
    exhaust = adata_cd8.obs.loc[mask, 'exhaustion_score'].mean() if 'exhaustion_score' in adata_cd8.obs else 0
    effector = adata_cd8.obs.loc[mask, 'effector_score'].mean() if 'effector_score' in adata_cd8.obs else 0
    memory = adata_cd8.obs.loc[mask, 'memory_score'].mean() if 'memory_score' in adata_cd8.obs else 0
    tox_mean = adata_cd8[mask, 'TOX'].X.mean() if 'TOX' in adata_cd8.var_names else 0
    ikzf3_mean = adata_cd8[mask, 'IKZF3'].X.mean() if 'IKZF3' in adata_cd8.var_names else 0

    # Heuristic labeling
    scores = {'exhaustion': exhaust, 'effector': effector, 'memory': memory}
    dominant = max(scores, key=scores.get)

    if dominant == 'exhaustion' and tox_mean > 1.0:
        labels[st] = f'CD8-Exhausted (TOX-hi)'
    elif dominant == 'exhaustion':
        labels[st] = f'CD8-Exhausted'
    elif dominant == 'effector' and ikzf3_mean > 1.5:
        labels[st] = f'CD8-Effector (IKZF3-hi)'
    elif dominant == 'effector':
        labels[st] = f'CD8-Effector'
    elif dominant == 'memory':
        labels[st] = f'CD8-Memory'
    else:
        labels[st] = f'CD8-Transitional'

    print(f"    Subtype {st} → {labels[st]}")

adata_cd8.obs['tf_subtype_label'] = adata_cd8.obs['tf_subtype'].map(labels)

###############################################################################
# PART 2: Malignant cell subclustering using top SHAP TFs
###############################################################################
print("\n[7] Malignant cell subclustering using top SHAP TFs...")

adata_mal = adata_all[adata_all.obs['cell.types'] == 'Mal'].copy()
print(f"    Malignant cells: {adata_mal.n_obs}")

top_mal_tfs = tf_mal.head(15)['TF'].tolist()
available_mal_tfs = [tf for tf in top_mal_tfs if tf in adata_mal.var_names]
print(f"    Using TFs: {available_mal_tfs}")

adata_mal_tf = adata_mal[:, available_mal_tfs].copy()
sc.pp.scale(adata_mal_tf, max_value=10)
sc.tl.pca(adata_mal_tf, n_comps=min(10, len(available_mal_tfs) - 1))
sc.pp.neighbors(adata_mal_tf, n_pcs=min(10, len(available_mal_tfs) - 1), n_neighbors=15)
sc.tl.leiden(adata_mal_tf, resolution=0.5, key_added='tf_subtype')
sc.tl.umap(adata_mal_tf)

adata_mal.obs['tf_subtype'] = adata_mal_tf.obs['tf_subtype'].values
adata_mal.obsm['X_umap_tf'] = adata_mal_tf.obsm['X_umap']

n_subtypes_mal = adata_mal.obs['tf_subtype'].nunique()
print(f"    Discovered {n_subtypes_mal} malignant subtypes")

# Characterize malignant subtypes
print(f"\n    {'Subtype':<10} {'N cells':<10} {'%Naive':<10} {'%Post-tx':<10} {'SOX10':<10} {'MITF':<10} {'ETV5':<10}")
print("    " + "-" * 70)

mal_stats = []
for st in sorted(adata_mal.obs['tf_subtype'].unique()):
    mask = adata_mal.obs['tf_subtype'] == st
    n = mask.sum()
    naive_frac = (adata_mal.obs.loc[mask, 'treatment.group'] == 'treatment.naive').mean() * 100
    sox10 = adata_mal[mask, 'SOX10'].X.mean() if 'SOX10' in adata_mal.var_names else 0
    mitf = adata_mal[mask, 'MITF'].X.mean() if 'MITF' in adata_mal.var_names else 0
    etv5 = adata_mal[mask, 'ETV5'].X.mean() if 'ETV5' in adata_mal.var_names else 0

    print(f"    {st:<10} {n:<10} {naive_frac:<10.1f} {100-naive_frac:<10.1f} {sox10:<10.3f} {mitf:<10.3f} {etv5:<10.3f}")

    tf_means = {}
    for tf in available_mal_tfs:
        tf_means[tf] = adata_mal[mask, tf].X.mean()
    mal_stats.append({
        'subtype': st, 'n_cells': n,
        'naive_pct': naive_frac, 'post_pct': 100 - naive_frac,
        **tf_means
    })

    # Fisher test
    naive_n = (adata_mal.obs.loc[mask, 'treatment.group'] == 'treatment.naive').sum()
    post_n = n - naive_n
    rest_naive = (adata_mal.obs.loc[~mask, 'treatment.group'] == 'treatment.naive').sum()
    rest_post = (~mask).sum() - rest_naive
    table = [[naive_n, post_n], [rest_naive, rest_post]]
    odds, pval = fisher_exact(table)
    enrichment = 'Naive' if odds > 1 else 'Post-tx'
    sig = ' *' if pval < 0.05 else ''
    print(f"             Fisher: OR={odds:.2f}, p={pval:.3e} → {enrichment}{sig}")

mal_df = pd.DataFrame(mal_stats)
mal_df.to_csv(f'{OUT_DIR}/malignant_tf_subtypes.csv', index=False)

# Label malignant subtypes
print("\n[8] Labeling malignant subtypes...")
mal_labels = {}
for st in sorted(adata_mal.obs['tf_subtype'].unique()):
    mask = adata_mal.obs['tf_subtype'] == st
    sox10 = adata_mal[mask, 'SOX10'].X.mean() if 'SOX10' in adata_mal.var_names else 0
    mitf = adata_mal[mask, 'MITF'].X.mean() if 'MITF' in adata_mal.var_names else 0

    if sox10 > 1.5 and mitf > 1.5:
        mal_labels[st] = 'Melanocytic (SOX10-hi/MITF-hi)'
    elif sox10 > 1.0:
        mal_labels[st] = 'Melanocytic (SOX10-hi)'
    elif mitf > 1.0:
        mal_labels[st] = 'Melanocytic (MITF-hi)'
    else:
        mal_labels[st] = 'Dedifferentiated (SOX10-lo/MITF-lo)'

    print(f"    Subtype {st} → {mal_labels[st]}")

adata_mal.obs['tf_subtype_label'] = adata_mal.obs['tf_subtype'].map(mal_labels)

###############################################################################
# PART 3: Novel TF cross-lineage analysis
###############################################################################
print("\n[9] Cross-lineage TF analysis (Novel findings)...")

# Unexpected TF-cell type associations from SHAP
novel_associations = [
    ('IKZF3', 'Mal', 'Immune TF in malignant cells'),
    ('IRF8', 'T.CD8', 'Myeloid TF in CD8+ T cells'),
    ('BCL11A', 'T.CD8', 'B cell TF in CD8+ T cells'),
    ('ZEB2', 'Macrophage', 'EMT TF in macrophages'),
    ('WWTR1', 'T.CD8', 'Hippo pathway TF in CD8+ T cells'),
    ('ETV5', 'Mal', 'ETS family in malignant cells'),
    ('MAF', 'NK', 'bZIP TF in NK cells'),
]

from scipy.stats import mannwhitneyu

print(f"\n    {'TF':<10} {'Cell type':<12} {'All mean':<12} {'Celltype mean':<15} {'Enrichment':<12} {'Description'}")
print("    " + "-" * 90)
for tf, ct, desc in novel_associations:
    if tf in adata_all.var_names:
        ct_mask = adata_all.obs['cell.types'] == ct
        ct_expr = adata_all[ct_mask, tf].X.flatten().mean()
        all_expr = adata_all[:, tf].X.flatten().mean()
        ratio = ct_expr / all_expr if all_expr > 0 else 0
        print(f"    {tf:<10} {ct:<12} {all_expr:<12.3f} {ct_expr:<15.3f} {ratio:<12.2f}x  {desc}")

###############################################################################
# Save processed data
###############################################################################
print("\n[10] Saving results...")
adata_cd8.write(f'{OUT_DIR}/adata_cd8_subtypes.h5ad')
adata_mal.write(f'{OUT_DIR}/adata_mal_subtypes.h5ad')

print("\n=== SUBTYPE DISCOVERY COMPLETE ===")
