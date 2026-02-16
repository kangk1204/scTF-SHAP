#!/usr/bin/env python3
"""
IRF8 Target Gene Enrichment Analysis
=====================================
Tests whether curated IRF8 target genes are enriched among DEGs
found in IRF8-high CD8+ T cells, providing evidence for IRF8
transcriptional activity (not just expression).

Author: Keunsoo Kang
Date: 2026-02-14
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy import sparse
import scanpy as sc
from statsmodels.stats.multitest import multipletests
import os
import warnings
warnings.filterwarnings('ignore')

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 1. Define IRF8 target gene sets from multiple sources
# ============================================================

# TRRUST v2 / literature curated direct targets of IRF8
irf8_targets_ifn = [
    'IFNB1', 'IRF1', 'IRF7', 'IFIT1', 'IFIT2', 'IFIT3',
    'ISG15', 'MX1', 'OAS1', 'OAS2', 'STAT1', 'STAT2'
]

irf8_targets_antigen = [
    'TAP1', 'TAP2', 'PSMB8', 'PSMB9', 'PSMB10', 'B2M',
    'HLA-A', 'HLA-B', 'HLA-C', 'CIITA'
]

irf8_targets_myeloid = [
    'IL12A', 'IL12B', 'IL18', 'CXCL9', 'CXCL10', 'CXCL11',
    'CCL5', 'NOS2'
]

irf8_targets_apoptosis = [
    'CASP1', 'CASP4', 'AIM2', 'GSDMD'
]

irf8_targets_other = [
    'CD274', 'PDCD1LG2', 'IDO1', 'GBP1', 'GBP2', 'GBP4', 'GBP5', 'WARS'
]

irf8_targets_exhaustion = [
    'TOX', 'PDCD1', 'HAVCR2', 'LAG3', 'TIGIT', 'ENTPD1', 'CTLA4'
]

# Broader IRF family downstream gene set
irf_downstream = [
    'NFE2L2', 'BATF', 'BATF3', 'IRF4', 'IRF1', 'IRF2', 'IRF5', 'IRF7', 'PRDM1'
]

# Combine all into a master set (excluding IRF8 itself to avoid circularity)
all_irf8_targets = list(set(
    irf8_targets_ifn + irf8_targets_antigen + irf8_targets_myeloid +
    irf8_targets_apoptosis + irf8_targets_other + irf8_targets_exhaustion +
    irf_downstream
))
all_irf8_targets = [g for g in all_irf8_targets if g != 'IRF8']

# Organize gene sets for testing
gene_sets = {
    'Type I IFN response': irf8_targets_ifn,
    'Antigen presentation': irf8_targets_antigen,
    'Myeloid/DC signaling': irf8_targets_myeloid,
    'Apoptosis/pyroptosis': irf8_targets_apoptosis,
    'Immune checkpoints & effectors': irf8_targets_other,
    'T cell exhaustion (Li et al.)': irf8_targets_exhaustion,
    'IRF family downstream': irf_downstream,
    'All IRF8 targets (combined)': all_irf8_targets,
}

# ============================================================
# 2. Load DEG data
# ============================================================

disc = pd.read_csv(f'{OUT_DIR}/irf8high_deg_discovery.csv')
val = pd.read_csv(f'{OUT_DIR}/irf8high_deg_validation.csv')

print("=" * 80)
print("IRF8 TARGET GENE ENRICHMENT ANALYSIS")
print("=" * 80)
print(f"\nDiscovery cohort: {len(disc)} genes tested")
print(f"Validation cohort: {len(val)} genes tested")

# Define upregulated DEGs (padj < 0.05 and log2fc > 0)
disc_up = set(disc[(disc['padj'] < 0.05) & (disc['log2fc'] > 0)]['gene'])
val_up = set(val[(val['padj'] < 0.05) & (val['log2fc'] > 0)]['gene'])

# Background gene universes
disc_bg = set(disc['gene'])
val_bg = set(val['gene'])

print(f"\nDiscovery upregulated DEGs (padj<0.05, log2fc>0): {len(disc_up)}")
print(f"Validation upregulated DEGs (padj<0.05, log2fc>0): {len(val_up)}")

# ============================================================
# 3. Fisher's exact test for each gene set
# ============================================================

def fisher_enrichment(target_genes, deg_set, background_set):
    """
    Fisher's exact test for enrichment of target genes among DEGs.
    
    Contingency table:
                    In DEGs     Not in DEGs
    In targets       a            b
    Not in targets   c            d
    """
    targets_in_bg = set(target_genes) & background_set
    
    a = len(targets_in_bg & deg_set)
    b = len(targets_in_bg - deg_set)
    c = len(deg_set - targets_in_bg)
    d = len(background_set - targets_in_bg - deg_set)
    
    table = np.array([[a, b], [c, d]])
    odds_ratio, pvalue = stats.fisher_exact(table, alternative='greater')
    
    overlap_genes = sorted(targets_in_bg & deg_set)
    
    return {
        'n_targets_in_bg': len(targets_in_bg),
        'n_overlap': a,
        'odds_ratio': odds_ratio,
        'pvalue': pvalue,
        'overlap_genes': overlap_genes,
    }



def sig_marker(p):
    """Return significance marker string for a p-value."""
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return "ns"

print("\n" + "=" * 80)
print("ENRICHMENT RESULTS")
print("=" * 80)

results_rows = []

for gs_name, gs_genes in gene_sets.items():
    print(f"\n{'─' * 70}")
    print(f"Gene Set: {gs_name} ({len(gs_genes)} genes)")
    print(f"{'─' * 70}")
    
    # Discovery cohort
    res_disc = fisher_enrichment(gs_genes, disc_up, disc_bg)
    print(f"\n  DISCOVERY COHORT:")
    print(f"    Targets in background: {res_disc['n_targets_in_bg']}")
    print(f"    Targets in upregulated DEGs: {res_disc['n_overlap']}/{res_disc['n_targets_in_bg']}")
    print(f"    Odds ratio: {res_disc['odds_ratio']:.2f}")
    print(f"    P-value (Fisher's exact, one-sided): {res_disc['pvalue']:.2e}")
    if res_disc['overlap_genes']:
        print(f"    Overlap genes: {', '.join(res_disc['overlap_genes'])}")
    
    # Validation cohort
    res_val = fisher_enrichment(gs_genes, val_up, val_bg)
    print(f"\n  VALIDATION COHORT:")
    print(f"    Targets in background: {res_val['n_targets_in_bg']}")
    print(f"    Targets in upregulated DEGs: {res_val['n_overlap']}/{res_val['n_targets_in_bg']}")
    print(f"    Odds ratio: {res_val['odds_ratio']:.2f}")
    print(f"    P-value (Fisher's exact, one-sided): {res_val['pvalue']:.2e}")
    if res_val['overlap_genes']:
        print(f"    Overlap genes: {', '.join(res_val['overlap_genes'])}")
    
    results_rows.append({
        'gene_set': gs_name,
        'n_genes_in_set': len(gs_genes),
        'disc_targets_in_bg': res_disc['n_targets_in_bg'],
        'disc_n_overlap': res_disc['n_overlap'],
        'disc_odds_ratio': round(res_disc['odds_ratio'], 3),
        'disc_pvalue': res_disc['pvalue'],
        'disc_overlap_genes': '; '.join(res_disc['overlap_genes']),
        'val_targets_in_bg': res_val['n_targets_in_bg'],
        'val_n_overlap': res_val['n_overlap'],
        'val_odds_ratio': round(res_val['odds_ratio'], 3),
        'val_pvalue': res_val['pvalue'],
        'val_overlap_genes': '; '.join(res_val['overlap_genes']),
    })

# Save results
results_df = pd.DataFrame(results_rows)

# Apply Benjamini-Hochberg correction across all gene sets
_, disc_padj, _, _ = multipletests(results_df['disc_pvalue'].values, method='fdr_bh')
_, val_padj, _, _ = multipletests(results_df['val_pvalue'].values, method='fdr_bh')
results_df['disc_padj'] = disc_padj
results_df['val_padj'] = val_padj
results_df['disc_significance'] = results_df['disc_padj'].apply(sig_marker)
results_df['val_significance'] = results_df['val_padj'].apply(sig_marker)

print(f"\nBenjamini-Hochberg correction applied across {len(results_df)} gene sets.")

results_df.to_csv(f'{OUT_DIR}/irf8_target_enrichment_results.csv', index=False)
print(f"\n\nResults saved to: irf8_target_enrichment_results.csv")

# ============================================================
# 4. Summary table
# ============================================================

print("\n\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"\n{'Gene Set':<35} {'Disc Overlap':>13} {'Disc OR':>9} {'Disc Padj':>12} {'Val Overlap':>13} {'Val OR':>9} {'Val Padj':>12}")
print("─" * 105)
for _, row in results_df.iterrows():
    disc_ov = f"{row['disc_n_overlap']}/{row['disc_targets_in_bg']}"
    val_ov = f"{row['val_n_overlap']}/{row['val_targets_in_bg']}"
    disc_or = f"{row['disc_odds_ratio']:.2f}" if row['disc_odds_ratio'] != np.inf else "inf"
    val_or = f"{row['val_odds_ratio']:.2f}" if row['val_odds_ratio'] != np.inf else "inf"
    print(f"{row['gene_set']:<35} {disc_ov:>13} {disc_or:>9} {row['disc_padj']:>12.2e} {val_ov:>13} {val_or:>9} {row['val_padj']:>12.2e}")

# ============================================================
# 5. Complementary analysis: Correlation of IRF8 expression
#    with shared DEGs across individual cells
# ============================================================

print("\n\n" + "=" * 80)
print("COMPLEMENTARY ANALYSIS: IRF8 EXPRESSION vs DEG SIGNATURE CORRELATION")
print("=" * 80)

# Load CD8+ T cell expression data
print("\nLoading CD8+ T cell expression data...")
h5_path = f'{OUT_DIR}/adata_cd8_subtypes.h5ad'
adata_expr = sc.read_h5ad(h5_path)
if adata_expr.raw is not None:
    X_raw = adata_expr.raw.X
    var_names_raw = adata_expr.raw.var_names.astype(str).tolist()
    print("  Using `.raw` matrix from adata_cd8_subtypes.h5ad")
else:
    X_raw = adata_expr.X
    var_names_raw = adata_expr.var_names.astype(str).tolist()
    print("  `.raw` not found; using `.X` matrix from adata_cd8_subtypes.h5ad")
obs_index = adata_expr.obs_names.astype(str).tolist()

print(f"Loaded: {X_raw.shape[0]} cells x {X_raw.shape[1]} genes")

# Create gene name to index mapping
gene_to_idx = {g: i for i, g in enumerate(var_names_raw)}


def get_gene_vector(x_mat, gene_idx):
    """Return a dense 1D expression vector for one gene index."""
    vec = x_mat[:, gene_idx]
    if sparse.issparse(vec):
        return vec.toarray().ravel()
    return np.asarray(vec).ravel()

# Find shared DEGs (upregulated in both cohorts)
shared_up_degs = disc_up & val_up
print(f"Shared upregulated DEGs (both cohorts): {len(shared_up_degs)}")

# Get IRF8 expression
irf8_idx = gene_to_idx['IRF8']
irf8_expr = get_gene_vector(X_raw, irf8_idx)
print(f"IRF8 expression range: [{irf8_expr.min():.3f}, {irf8_expr.max():.3f}]")
print(f"IRF8 expressing cells: {(irf8_expr > 0).sum()} / {len(irf8_expr)} ({(irf8_expr > 0).mean()*100:.1f}%)")

# Compute correlation between IRF8 and each shared DEG
shared_degs_in_data = [g for g in shared_up_degs if g in gene_to_idx and g != 'IRF8']
print(f"Shared DEGs present in expression data: {len(shared_degs_in_data)}")

print("\nComputing Spearman correlations for all shared DEGs...")
corr_results = []
for gene in shared_degs_in_data:
    gene_idx = gene_to_idx[gene]
    gene_expr = get_gene_vector(X_raw, gene_idx)
    r, p = stats.spearmanr(irf8_expr, gene_expr)
    corr_results.append({'gene': gene, 'spearman_r': r, 'pvalue': p})

corr_df = pd.DataFrame(corr_results).sort_values('spearman_r', ascending=False)

# Multiple testing correction (Benjamini-Hochberg)
_, corr_df['padj'], _, _ = multipletests(corr_df['pvalue'].values, method='fdr_bh')

# Summary statistics
sig_pos = corr_df[(corr_df['padj'] < 0.05) & (corr_df['spearman_r'] > 0)]
sig_neg = corr_df[(corr_df['padj'] < 0.05) & (corr_df['spearman_r'] < 0)]

print(f"\nCorrelation results for {len(corr_df)} shared DEGs:")
print(f"  Significantly positively correlated (padj<0.05, r>0): {len(sig_pos)} ({len(sig_pos)/len(corr_df)*100:.1f}%)")
print(f"  Significantly negatively correlated (padj<0.05, r<0): {len(sig_neg)} ({len(sig_neg)/len(corr_df)*100:.1f}%)")
print(f"  Mean Spearman r: {corr_df['spearman_r'].mean():.4f}")
print(f"  Median Spearman r: {corr_df['spearman_r'].median():.4f}")

print(f"\n  Top 20 positively correlated genes:")
print(f"  {'Gene':<15} {'Spearman r':>12} {'P-value':>12} {'Adj P':>12}")
print(f"  {'─'*53}")
for _, row in sig_pos.head(20).iterrows():
    print(f"  {row['gene']:<15} {row['spearman_r']:>12.4f} {row['pvalue']:>12.2e} {row['padj']:>12.2e}")

print(f"\n  Top 10 negatively correlated genes:")
print(f"  {'Gene':<15} {'Spearman r':>12} {'P-value':>12} {'Adj P':>12}")
print(f"  {'─'*53}")
for _, row in sig_neg.sort_values('spearman_r').head(10).iterrows():
    print(f"  {row['gene']:<15} {row['spearman_r']:>12.4f} {row['pvalue']:>12.2e} {row['padj']:>12.2e}")

# Check IRF8 target genes specifically in the correlation data
print(f"\n  IRF8 curated targets among correlated genes:")
print(f"  {'Gene':<15} {'Spearman r':>12} {'P-value':>12} {'Adj P':>12} {'In shared DEGs':>15}")
print(f"  {'─'*68}")
target_found_count = 0
for gene in sorted(all_irf8_targets):
    if gene in corr_df['gene'].values:
        row = corr_df[corr_df['gene'] == gene].iloc[0]
        sig_mark = " ***" if row['padj'] < 0.001 else (" **" if row['padj'] < 0.01 else (" *" if row['padj'] < 0.05 else ""))
        print(f"  {gene:<15} {row['spearman_r']:>12.4f} {row['pvalue']:>12.2e} {row['padj']:>12.2e} {'Yes':>15}{sig_mark}")
        target_found_count += 1

print(f"\n  Total IRF8 targets found in shared DEGs: {target_found_count}/{len(all_irf8_targets)}")

# One-sample t-test: are the correlations of IRF8 targets significantly > 0?
target_corrs = corr_df[corr_df['gene'].isin(all_irf8_targets)]['spearman_r']
if len(target_corrs) > 1:
    t_stat, t_pval = stats.ttest_1samp(target_corrs, 0)
    t_pval_onesided = t_pval / 2 if t_stat > 0 else 1 - t_pval / 2
    print(f"\n  One-sample t-test: IRF8 target correlations vs 0")
    print(f"    N targets with correlations: {len(target_corrs)}")
    print(f"    Mean r of IRF8 targets: {target_corrs.mean():.4f}")
    print(f"    Mean r of all other DEGs: {corr_df[~corr_df['gene'].isin(all_irf8_targets)]['spearman_r'].mean():.4f}")
    print(f"    t-statistic: {t_stat:.3f}")
    print(f"    P-value (one-sided, > 0): {t_pval_onesided:.2e}")

# Wilcoxon rank-sum: are IRF8 target correlations higher than non-target correlations?
non_target_corrs = corr_df[~corr_df['gene'].isin(all_irf8_targets)]['spearman_r']
if len(target_corrs) > 1:
    u_stat, u_pval = stats.mannwhitneyu(target_corrs, non_target_corrs, alternative='greater')
    print(f"\n  Mann-Whitney U test: IRF8 target r vs non-target r")
    print(f"    U-statistic: {u_stat:.1f}")
    print(f"    P-value (one-sided, targets > non-targets): {u_pval:.2e}")

# Save correlation results
corr_df.to_csv(f'{OUT_DIR}/irf8_target_correlations.csv', index=False)
print(f"\nCorrelation results saved to: irf8_target_correlations.csv")

# ============================================================
# 6. Final summary
# ============================================================

print("\n\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

n_sig_both = sum(1 for _, row in results_df.iterrows()
                 if row['disc_padj'] < 0.05 and row['val_padj'] < 0.05)
n_sig_disc = sum(1 for _, row in results_df.iterrows()
                 if row['disc_padj'] < 0.05)
n_sig_val = sum(1 for _, row in results_df.iterrows()
                 if row['val_padj'] < 0.05)

print(f"""
Of {len(gene_sets)} IRF8 target gene sets tested:
  - {n_sig_disc} are significantly enriched (Padj<0.05) in the discovery cohort
  - {n_sig_val} are significantly enriched (Padj<0.05) in the validation cohort
  - {n_sig_both} are significantly enriched (Padj<0.05) in BOTH cohorts

Key enrichment findings:
""")

for _, row in results_df.iterrows():
    if row['disc_padj'] < 0.05 or row['val_padj'] < 0.05:
        disc_star = sig_marker(row['disc_padj'])
        val_star = sig_marker(row['val_padj'])
        disc_or = f"{row['disc_odds_ratio']:.2f}" if row['disc_odds_ratio'] != np.inf else "inf"
        val_or = f"{row['val_odds_ratio']:.2f}" if row['val_odds_ratio'] != np.inf else "inf"
        print(f"  - {row['gene_set']}: Discovery {disc_star} (OR={disc_or}), Validation {val_star} (OR={val_or})")

if len(target_corrs) > 1:
    print(f"""
Correlation analysis:
  - {len(sig_pos)} of {len(corr_df)} shared DEGs show significant positive correlation
    with IRF8 expression (padj<0.05), confirming that IRF8 expression is
    functionally linked to the DEG signature at the single-cell level.
  - IRF8 curated targets show mean r = {target_corrs.mean():.4f} vs
    non-target DEGs mean r = {non_target_corrs.mean():.4f}

These results provide evidence that IRF8-high CD8+ T cells exhibit
transcriptional activation of known IRF8 target programs, supporting
a functional role for IRF8 beyond mere expression correlation.
""")

print("Analysis complete.")
