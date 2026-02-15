"""
03_validation_GSE120575.py
Independent validation of IRF8-high CD8+ T cell subtype using GSE120575
(Sade-Feldman et al., Cell 2018)
- 16,291 immune cells from 48 melanoma biopsies
- Smart-seq2, same protocol as GSE115978
- Pre/Post checkpoint immunotherapy
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, fisher_exact
from statsmodels.stats.multitest import multipletests
import os
import gzip

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
DATA_DIR = os.path.join(BASE_DIR, 'data')

plt.rcParams.update({
    'font.size': 8, 'font.family': 'Arial',
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.5,
})

###############################################################################
# STEP 1: Load GSE120575 data
###############################################################################
print("=" * 60)
print("STEP 1: Loading GSE120575 (Sade-Feldman et al., Cell 2018)")
print("=" * 60)

tpm_file = f'{DATA_DIR}/GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz'

# Read header rows first to get cell IDs and treatment info
# Line 1: cell IDs (no row label), Line 2: Pre/Post patient info, Line 3+: gene_name + values
print("Reading header rows...")
with gzip.open(tpm_file, 'rt') as f:
    line1 = f.readline().strip().split('\t')  # cell IDs (16291 elements)
    line2 = f.readline().strip().split('\t')  # patient/treatment info (16291 elements)

cell_ids = line1  # all elements are cell IDs (no leading empty field)
patient_treat_info = line2

print(f"  Cell IDs: {len(cell_ids)}")
print(f"  Patient/treatment info: {len(patient_treat_info)}")

# Read expression data (line 3+: gene_name \t value1 \t value2 ...)
# Data rows have gene_name as first field + 16291 values = 16292 columns total
# Header line 1 has 16291 cell IDs (no gene name field)
print("Reading TPM expression matrix (this may take a while)...")
tpm = pd.read_csv(tpm_file, sep='\t', skiprows=2, index_col=0, header=None, low_memory=False)
# Now tpm has 16291 columns (index_col=0 consumed gene name)
print(f"  Raw expression matrix: {tpm.shape[0]} genes x {tpm.shape[1]} columns")
# Ensure column count matches
if tpm.shape[1] == len(cell_ids):
    tpm.columns = cell_ids
elif tpm.shape[1] == len(cell_ids) + 1:
    # Extra column at the end (possibly empty)
    tpm = tpm.iloc[:, :len(cell_ids)]
    tpm.columns = cell_ids
else:
    print(f"  WARNING: column mismatch: {tpm.shape[1]} vs {len(cell_ids)} cell IDs")
    # Trim to minimum
    n_cols = min(tpm.shape[1], len(cell_ids))
    tpm = tpm.iloc[:, :n_cols]
    cell_ids = cell_ids[:n_cols]
    patient_treat_info = patient_treat_info[:n_cols]
    tpm.columns = cell_ids
print(f"  Expression matrix: {tpm.shape[0]} genes x {tpm.shape[1]} cells")

# Read metadata for response/therapy info
print("Reading metadata...")
meta = pd.read_csv(f'{DATA_DIR}/GSE120575_patient_ID_single_cells.txt.gz',
                   sep='\t', skiprows=19, low_memory=False, encoding='latin-1')

# Build annotation DataFrame using header line 2 for treatment info
# and metadata file for response/therapy
meta_dict = {}
for _, row in meta.iterrows():
    title = str(row['title'])
    meta_dict[title] = {
        'response': str(row.iloc[5]) if pd.notna(row.iloc[5]) else 'Unknown',
        'therapy': str(row.iloc[6]) if pd.notna(row.iloc[6]) else 'Unknown',
    }

annot_data = []
for i, cid in enumerate(cell_ids):
    pt_info = patient_treat_info[i] if i < len(patient_treat_info) else 'Unknown'
    resp = meta_dict.get(cid, {}).get('response', 'Unknown')
    ther = meta_dict.get(cid, {}).get('therapy', 'Unknown')
    treatment = 'Pre' if str(pt_info).startswith('Pre') else ('Post' if str(pt_info).startswith('Post') else 'Unknown')
    annot_data.append({
        'cell_id': cid,
        'patient_treatment': pt_info,
        'response': resp,
        'therapy': ther,
        'treatment': treatment,
    })

annot_df = pd.DataFrame(annot_data, index=cell_ids)

print(f"\nTreatment distribution:")
print(annot_df['treatment'].value_counts())
print(f"\nTherapy distribution:")
print(annot_df['therapy'].value_counts())

###############################################################################
# STEP 2: Identify CD8+ T cells in GSE120575
###############################################################################
print("\n" + "=" * 60)
print("STEP 2: Identifying CD8+ T cells by marker expression")
print("=" * 60)

# Build AnnData
tpm_T = tpm.T
common_cells = tpm_T.index.intersection(annot_df.index)
print(f"Common cells between TPM and annotation: {len(common_cells)}")

adata = sc.AnnData(X=tpm_T.loc[common_cells].values.astype(np.float32))
adata.obs_names = common_cells.tolist()
adata.var_names = tpm.index.tolist()
for col in annot_df.columns:
    adata.obs[col] = annot_df.loc[common_cells, col].values

# Log transform
sc.pp.log1p(adata)

# Identify CD8+ T cells by expression: CD8A > 0 AND CD3D/CD3E > 0 AND CD8A > CD4
# This dataset is CD45+ sorted immune cells
cd8a_expr = adata[:, 'CD8A'].X.flatten() if 'CD8A' in adata.var_names else np.zeros(adata.n_obs)
cd4_expr = adata[:, 'CD4'].X.flatten() if 'CD4' in adata.var_names else np.zeros(adata.n_obs)
cd3d_expr = adata[:, 'CD3D'].X.flatten() if 'CD3D' in adata.var_names else np.zeros(adata.n_obs)

print(f"CD8A >0: {(cd8a_expr > 0).sum()} cells")
print(f"CD3D >0: {(cd3d_expr > 0).sum()} cells")
print(f"CD4 >0: {(cd4_expr > 0).sum()} cells")

# CD8+ T cell selection: CD8A > 1.0 (log1p TPM) and CD3D > 0 and CD8A > CD4
cd8_mask = (cd8a_expr > 1.0) & (cd3d_expr > 0) & (cd8a_expr > cd4_expr)
adata_cd8 = adata[cd8_mask].copy()

print(f"\nCD8+ T cells selected: {adata_cd8.n_obs}")
print(f"  Pre-treatment: {(adata_cd8.obs['treatment'] == 'Pre').sum()}")
print(f"  Post-treatment: {(adata_cd8.obs['treatment'] == 'Post').sum()}")

###############################################################################
# STEP 3: Apply TF-based subclustering using same TFs as discovery cohort
###############################################################################
print("\n" + "=" * 60)
print("STEP 3: TF-based subclustering of CD8+ T cells")
print("=" * 60)

# Use the same top 15 SHAP-ranked TFs from discovery cohort
tf_cd8_discovery = pd.read_csv(f'{OUT_DIR}/tf_importance_T_CD8.csv')
top_cd8_tfs = tf_cd8_discovery.head(15)['TF'].tolist()
available_tfs = [tf for tf in top_cd8_tfs if tf in adata_cd8.var_names]
print(f"Top 15 SHAP TFs from discovery: {top_cd8_tfs}")
print(f"Available in validation dataset: {available_tfs}")

# TF-based subclustering
adata_cd8_tf = adata_cd8[:, available_tfs].copy()
sc.pp.scale(adata_cd8_tf, max_value=10)
sc.tl.pca(adata_cd8_tf, n_comps=min(10, len(available_tfs) - 1), random_state=42)
sc.pp.neighbors(adata_cd8_tf, n_pcs=min(10, len(available_tfs) - 1), n_neighbors=15, random_state=42)
sc.tl.leiden(adata_cd8_tf, resolution=0.25, key_added='tf_subtype', random_state=42)
sc.tl.umap(adata_cd8_tf, random_state=42)

adata_cd8.obs['tf_subtype'] = adata_cd8_tf.obs['tf_subtype'].values
adata_cd8.obsm['X_umap_tf'] = adata_cd8_tf.obsm['X_umap']

n_subtypes = adata_cd8.obs['tf_subtype'].nunique()
print(f"\nDiscovered {n_subtypes} CD8+ T cell subtypes in validation cohort")

###############################################################################
# STEP 4: Identify IRF8-high subtype and characterize
###############################################################################
print("\n" + "=" * 60)
print("STEP 4: Characterizing CD8+ T cell subtypes")
print("=" * 60)

# Compute TF expression per subtype
subtype_stats = {}
for st in sorted(adata_cd8.obs['tf_subtype'].unique()):
    mask = adata_cd8.obs['tf_subtype'] == st
    n = mask.sum()
    stats = {'n': n}

    for tf in available_tfs:
        stats[tf] = adata_cd8[mask, tf].X.mean()

    # Treatment composition
    pre_n = (adata_cd8.obs.loc[mask, 'treatment'] == 'Pre').sum()
    post_n = (adata_cd8.obs.loc[mask, 'treatment'] == 'Post').sum()
    stats['pre_n'] = pre_n
    stats['post_n'] = post_n
    stats['post_pct'] = post_n / n * 100 if n > 0 else 0

    subtype_stats[st] = stats

# Print characterization
print(f"\n{'Subtype':<8} {'N':>5} {'IRF8':>7} {'TOX':>7} {'ID2':>7} {'PRDM1':>7} {'TCF7':>7} {'Post%':>7}")
print("-" * 60)
for st, stats in sorted(subtype_stats.items()):
    irf8 = stats.get('IRF8', 0)
    tox = stats.get('TOX', 0)
    id2 = stats.get('ID2', 0)
    prdm1 = stats.get('PRDM1', 0)
    tcf7 = stats.get('TCF7', 0)
    print(f"{st:<8} {stats['n']:>5} {irf8:>7.2f} {tox:>7.2f} {id2:>7.2f} {prdm1:>7.2f} {tcf7:>7.2f} {stats['post_pct']:>6.1f}%")

# Identify IRF8-high subtype
irf8_high_cluster = None
irf8_max = 0
for st, stats in subtype_stats.items():
    irf8_val = stats.get('IRF8', 0)
    if irf8_val > irf8_max:
        irf8_max = irf8_val
        irf8_high_cluster = st

print(f"\nIRF8-high cluster: {irf8_high_cluster} (IRF8 mean = {irf8_max:.3f})")

# Check if IRF8-high subtype exists (IRF8 > 0.3 threshold like discovery)
if irf8_max > 0.3:
    print("*** IRF8-high CD8+ T cell subtype VALIDATED ***")
    irf8_stats = subtype_stats[irf8_high_cluster]
    print(f"  N cells: {irf8_stats['n']}")
    print(f"  Post-treatment: {irf8_stats['post_pct']:.1f}%")
    print(f"  IRF8: {irf8_stats.get('IRF8', 0):.3f}")
    print(f"  TOX: {irf8_stats.get('TOX', 0):.3f}")

    # Fisher's exact test for treatment enrichment
    mask_irf8 = adata_cd8.obs['tf_subtype'] == irf8_high_cluster
    irf8_pre = (adata_cd8.obs.loc[mask_irf8, 'treatment'] == 'Pre').sum()
    irf8_post = (adata_cd8.obs.loc[mask_irf8, 'treatment'] == 'Post').sum()
    other_pre = (adata_cd8.obs.loc[~mask_irf8, 'treatment'] == 'Pre').sum()
    other_post = (adata_cd8.obs.loc[~mask_irf8, 'treatment'] == 'Post').sum()

    table = [[irf8_pre, irf8_post], [other_pre, other_post]]
    odds, pval = fisher_exact(table)
    enrichment = 'Post-treatment enriched' if odds < 1 else 'Pre-treatment enriched'
    print(f"  Fisher's exact: OR={odds:.3f}, p={pval:.4e} → {enrichment}")
else:
    print("IRF8-high subtype NOT clearly identified (max IRF8 < 0.3)")

###############################################################################
# STEP 5: IRF8 expression comparison Pre vs Post in CD8+ T cells
###############################################################################
print("\n" + "=" * 60)
print("STEP 5: IRF8 expression Pre vs Post treatment in CD8+ T cells")
print("=" * 60)

if 'IRF8' in adata_cd8.var_names:
    pre_mask = adata_cd8.obs['treatment'] == 'Pre'
    post_mask = adata_cd8.obs['treatment'] == 'Post'

    irf8_pre = adata_cd8[pre_mask, 'IRF8'].X.flatten()
    irf8_post = adata_cd8[post_mask, 'IRF8'].X.flatten()

    stat, pval = mannwhitneyu(irf8_pre, irf8_post, alternative='two-sided')
    # log2FC = (mean_log1p_post - mean_log1p_pre) / ln(2)
    log2fc = (irf8_post.mean() - irf8_pre.mean()) / np.log(2)

    print(f"  IRF8 Pre-treatment: mean={irf8_pre.mean():.4f} (n={len(irf8_pre)})")
    print(f"  IRF8 Post-treatment: mean={irf8_post.mean():.4f} (n={len(irf8_post)})")
    print(f"  Log2 FC: {log2fc:.4f}")
    print(f"  Wilcoxon p-value: {pval:.4e}")

    # Also check TOX
    if 'TOX' in adata_cd8.var_names:
        tox_pre = adata_cd8[pre_mask, 'TOX'].X.flatten()
        tox_post = adata_cd8[post_mask, 'TOX'].X.flatten()
        stat_tox, pval_tox = mannwhitneyu(tox_pre, tox_post, alternative='two-sided')
        # Correct log2FC: difference of log1p means divided by ln(2)
        log2fc_tox = (tox_post.mean() - tox_pre.mean()) / np.log(2)
        print(f"\n  TOX Pre-treatment: mean={tox_pre.mean():.4f}")
        print(f"  TOX Post-treatment: mean={tox_post.mean():.4f}")
        print(f"  Log2 FC: {log2fc_tox:.4f}")
        print(f"  Wilcoxon p-value: {pval_tox:.4e}")

###############################################################################
# STEP 6: Assign labels and generate validation figure
###############################################################################
print("\n" + "=" * 60)
print("STEP 6: Generating validation figure")
print("=" * 60)

# Assign labels to subtypes using same logic as discovery
labels = {}
for st, stats in sorted(subtype_stats.items()):
    irf8 = stats.get('IRF8', 0)
    tox = stats.get('TOX', 0)
    tcf7 = stats.get('TCF7', 0)
    prdm1 = stats.get('PRDM1', 0)
    id2 = stats.get('ID2', 0)

    if irf8 > 0.3:
        labels[st] = 'IRF8-high'
    elif tcf7 > 0.5:
        labels[st] = 'Memory (TCF7+)'
    elif tox > 1.1:
        labels[st] = 'Exhausted (TOX-hi)'
    elif id2 > 1.7 and prdm1 > 0.8:
        labels[st] = 'Cytotoxic (ID2+PRDM1+)'
    elif id2 > 1.7:
        labels[st] = 'Innate-like (ID2+)'
    elif prdm1 > 0.8:
        labels[st] = 'Effector (PRDM1+)'
    else:
        labels[st] = 'Effector'

    print(f"  Subtype {st}: {labels[st]} (n={stats['n']}, Post={stats['post_pct']:.1f}%)")

adata_cd8.obs['subtype_label'] = adata_cd8.obs['tf_subtype'].map(labels)

# Merge clusters with the same label for cleaner visualization
merged_labels = sorted(set(labels.values()))
print(f"\nMerging {n_subtypes} Leiden clusters into {len(merged_labels)} merged subtypes:")

# Compute merged stats
merged_stats = {}
for ml in merged_labels:
    mask = adata_cd8.obs['subtype_label'] == ml
    n = mask.sum()
    pre_n = (adata_cd8.obs.loc[mask, 'treatment'] == 'Pre').sum()
    post_n = (adata_cd8.obs.loc[mask, 'treatment'] == 'Post').sum()
    merged_stats[ml] = {
        'n': n,
        'pre_n': pre_n,
        'post_n': post_n,
        'post_pct': post_n / n * 100 if n > 0 else 0,
    }
    # Mean TF expression
    for tf in available_tfs:
        merged_stats[ml][tf] = adata_cd8[mask, tf].X.mean()
    print(f"  {ml}: n={n} (Post={merged_stats[ml]['post_pct']:.1f}%)")

# Generate Supplementary Figure: Validation (with merged labels)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

# Define consistent color palette for merged subtypes (match discovery cohort style)
merged_color_map = {
    'IRF8-high': '#E31A1C',
    'Exhausted (TOX-hi)': '#FF7F00',
    'Memory (TCF7+)': '#33A02C',
    'Cytotoxic (ID2+PRDM1+)': '#1F78B4',
    'Innate-like (ID2+)': '#A6CEE3',
    'Effector (PRDM1+)': '#6A3D9A',
    'Effector': '#B2DF8A',
}

# Panel A: UMAP colored by merged subtype label
ax = axes[0]
# Order for legend: IRF8-high first (our key finding)
label_order = ['IRF8-high', 'Exhausted (TOX-hi)', 'Memory (TCF7+)',
               'Cytotoxic (ID2+PRDM1+)', 'Innate-like (ID2+)',
               'Effector (PRDM1+)', 'Effector']
label_order = [l for l in label_order if l in set(adata_cd8.obs['subtype_label'])]

for lbl in label_order:
    mask = (adata_cd8.obs['subtype_label'] == lbl).values
    n = mask.sum()
    color = merged_color_map.get(lbl, '#999999')
    ax.scatter(
        adata_cd8.obsm['X_umap_tf'][mask, 0], adata_cd8.obsm['X_umap_tf'][mask, 1],
        s=2, alpha=0.6, c=color, label=f'{lbl} ({n})',
        rasterized=True
    )
ax.set_title('Validation: CD8+ T cell subtypes\n(GSE120575, Sade-Feldman et al.)', fontsize=8)
ax.set_xlabel('UMAP1', fontsize=7)
ax.set_ylabel('UMAP2', fontsize=7)
ax.set_xticks([])
ax.set_yticks([])
ax.legend(fontsize=5.5, markerscale=3, frameon=False, loc='best', handletextpad=0.2)
ax.text(-0.05, 1.05, 'A', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')

# Panel B: IRF8 expression violin Pre vs Post
ax = axes[1]
if 'IRF8' in adata_cd8.var_names:
    plot_data = pd.DataFrame({
        'IRF8': adata_cd8[:, 'IRF8'].X.flatten(),
        'Treatment': ['Pre' if t == 'Pre' else 'Post' for t in adata_cd8.obs['treatment']]
    })
    treat_pal = {'Pre': '#4575B4', 'Post': '#D73027'}
    sns.violinplot(data=plot_data, x='Treatment', y='IRF8', palette=treat_pal,
                   ax=ax, linewidth=0.5, inner=None, cut=0)
    # Add jitter
    for treat, color in [('Pre', '#4575B4'), ('Post', '#D73027')]:
        sub = plot_data[plot_data['Treatment'] == treat]
        if len(sub) > 200:
            sub = sub.sample(200, random_state=42)
        x_idx = 0 if treat == 'Pre' else 1
        jitter = np.random.normal(0, 0.05, len(sub))
        ax.scatter(x_idx + jitter, sub['IRF8'].values, s=0.5, alpha=0.3, c=color, rasterized=True)

    # Significance
    stat, pval = mannwhitneyu(
        plot_data[plot_data['Treatment'] == 'Pre']['IRF8'],
        plot_data[plot_data['Treatment'] == 'Post']['IRF8'],
        alternative='two-sided'
    )
    sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
    y_max = plot_data['IRF8'].max() * 1.05
    ax.text(0.5, y_max, sig, ha='center', fontsize=8, fontweight='bold')
    ax.set_title(f'IRF8 expression in CD8+ T cells\n(p = {pval:.2e})', fontsize=8)
    ax.set_ylabel('IRF8 expression (log1p TPM)', fontsize=7)
ax.text(-0.12, 1.05, 'B', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')

# Panel C: Treatment composition bar chart (merged subtypes)
ax = axes[2]
comp_data = []
for lbl in label_order:
    ms = merged_stats[lbl]
    comp_data.append({
        'Subtype': f'{lbl}\n(n={ms["n"]})',
        'Pre': ms['pre_n'] / ms['n'] * 100,
        'Post': ms['post_n'] / ms['n'] * 100,
        'color': merged_color_map.get(lbl, '#999999'),
    })
comp_df = pd.DataFrame(comp_data)
# Sort by Post% descending
comp_df = comp_df.sort_values('Post', ascending=True).reset_index(drop=True)

y_pos = range(len(comp_df))
ax.barh(y_pos, comp_df['Pre'].values, color='#4575B4', label='Pre-tx', edgecolor='white', linewidth=0.3)
ax.barh(y_pos, comp_df['Post'].values, left=comp_df['Pre'].values,
        color='#D73027', label='Post-tx', edgecolor='white', linewidth=0.3)
ax.set_yticks(y_pos)
ax.set_yticklabels(comp_df['Subtype'].values, fontsize=6.5)
ax.set_xlabel('Percentage (%)', fontsize=7)
ax.set_title('Validation: Treatment composition', fontsize=8)
ax.legend(fontsize=6, frameon=False, loc='lower right')
ax.axvline(x=50, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)
ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')

plt.tight_layout()
fig.savefig(f'{FIG_DIR}/FigureS1_validation.tiff', dpi=300, format='tiff')
fig.savefig(f'{FIG_DIR}/FigureS1_validation.pdf', dpi=300, format='pdf')
plt.close()
print("  Supplementary Figure S1 saved.")

###############################################################################
# Summary
###############################################################################
print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
print(f"Discovery cohort: GSE115978 (Jerby-Arnon et al., 2018)")
print(f"  (See 09_final_figures.py for current discovery CD8+ subtype counts)")
print(f"\nValidation cohort: GSE120575 (Sade-Feldman et al., 2018)")
print(f"  CD8+ T cells: {adata_cd8.n_obs}")
if 'IRF8-high' in merged_stats:
    ms = merged_stats['IRF8-high']
    print(f"  IRF8-high subtype (merged): n={ms['n']} ({ms['post_pct']:.1f}% post-treatment)")
    print(f"  *** IRF8-high CD8+ T cell subtype SUCCESSFULLY VALIDATED ***")

    # Fisher's exact test on merged IRF8-high
    irf8_mask_merged = adata_cd8.obs['subtype_label'] == 'IRF8-high'
    m_irf8_pre = (adata_cd8.obs.loc[irf8_mask_merged, 'treatment'] == 'Pre').sum()
    m_irf8_post = (adata_cd8.obs.loc[irf8_mask_merged, 'treatment'] == 'Post').sum()
    m_other_pre = (adata_cd8.obs.loc[~irf8_mask_merged, 'treatment'] == 'Pre').sum()
    m_other_post = (adata_cd8.obs.loc[~irf8_mask_merged, 'treatment'] == 'Post').sum()
    table_m = [[m_irf8_pre, m_irf8_post], [m_other_pre, m_other_post]]
    odds_m, pval_m = fisher_exact(table_m)
    enrich_m = 'Post-treatment enriched' if odds_m < 1 else 'Pre-treatment enriched'
    print(f"  Fisher's exact (merged): OR={odds_m:.3f}, p={pval_m:.4e} → {enrich_m}")
else:
    print(f"  IRF8-high subtype: not identified")

print("\n=== VALIDATION COMPLETE ===")
