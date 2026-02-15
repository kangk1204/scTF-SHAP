"""
06_all_celltype_subclustering.py
TF-based subclustering for ALL cell types in the melanoma TME.
Previously done: CD8+ T cells, Malignant cells.
New: CD4+ T, B cell, Macrophage, T cell (undifferentiated), NK, CAF, Endothelial.

For each cell type:
1. Extract cells from the full dataset
2. Use the top 15 SHAP-ranked TFs for that cell type
3. PCA → neighbors → Leiden clustering
4. Characterize subtypes by TF expression + treatment composition
5. Test for treatment-associated subtypes (Fisher's exact)
6. Run DEG analysis for interesting subtypes
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ── Load full dataset ─────────────────────────────────────────────────────────
print("=" * 70)
print("LOADING FULL DATASET (GSE115978)")
print("=" * 70)

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

# Log-transform
sc.pp.log1p(adata_all)
print(f"Total cells: {adata_all.n_obs}")

# ── Cell type mapping ─────────────────────────────────────────────────────────
celltype_map = {
    'T.CD4': ('T_CD4', 'CD4+ T cells'),
    'B.cell': ('B_cell', 'B cells'),
    'Macrophage': ('Macrophage', 'Macrophages'),
    'T.cell': ('T_cell', 'T cells (undifferentiated)'),
    'NK': ('NK', 'NK cells'),
    'CAF': ('CAF', 'Cancer-associated fibroblasts'),
    'Endo.': ('Endo_', 'Endothelial cells'),
}

# ── Known markers for subtype annotation ──────────────────────────────────────
known_markers = {
    'T.CD4': {
        'Treg': ['FOXP3', 'IL2RA', 'CTLA4', 'IKZF2'],
        'Th1': ['TBX21', 'IFNG', 'STAT4', 'CXCR3'],
        'Th17': ['RORC', 'IL17A', 'CCR6'],
        'Tfh': ['BCL6', 'CXCR5', 'ICOS', 'PDCD1'],
        'Naive/Memory': ['TCF7', 'LEF1', 'CCR7', 'SELL', 'IL7R'],
        'Exhaustion': ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'TOX'],
        'Cytotoxic': ['GZMA', 'GZMB', 'PRF1', 'NKG7'],
    },
    'B.cell': {
        'Naive': ['MS4A1', 'CD19', 'IGHD', 'FCER2', 'TCL1A'],
        'Memory': ['CD27', 'IGHG1', 'IGHG2'],
        'Plasma': ['SDC1', 'XBP1', 'MZB1', 'JCHAIN', 'PRDM1'],
        'GC': ['BCL6', 'AICDA', 'MKI67'],
        'Regulatory': ['IL10', 'CD24', 'CD38'],
    },
    'Macrophage': {
        'M1_proinflam': ['TNF', 'IL1B', 'IL6', 'NOS2', 'CD80', 'CD86', 'IRF5'],
        'M2_antiinflam': ['CD163', 'MRC1', 'CD206', 'ARG1', 'IL10', 'TGFB1', 'MAF'],
        'TAM': ['SPP1', 'TREM2', 'APOE', 'C1QA', 'C1QB', 'C1QC'],
        'IFN_stimulated': ['ISG15', 'MX1', 'IFIT1', 'IFI44L', 'STAT1'],
        'Phagocytic': ['CD14', 'FCGR1A', 'FCGR3A', 'CSF1R'],
        'Antigen_presenting': ['HLA-DRA', 'HLA-DRB1', 'CD74', 'CIITA'],
    },
    'T.cell': {
        'gdT': ['TRDC', 'TRGC1', 'TRGC2'],
        'NKT': ['KLRB1', 'KLRD1', 'NKG7', 'GNLY'],
        'MAIT': ['SLC4A10', 'KLRB1'],
        'Naive': ['TCF7', 'LEF1', 'CCR7', 'SELL'],
        'Activated': ['CD69', 'CD25', 'TNFRSF4', 'TNFRSF9'],
    },
    'NK': {
        'CD56bright': ['NCAM1', 'GZMK', 'XCL1', 'XCL2'],
        'CD56dim': ['FCGR3A', 'GZMB', 'PRF1', 'GNLY'],
        'Cytotoxic': ['GZMA', 'GZMB', 'PRF1', 'NKG7', 'GNLY'],
        'IFN_responsive': ['ISG15', 'MX1', 'IFIT1', 'STAT1'],
    },
    'CAF': {
        'Myofibroblast': ['ACTA2', 'TAGLN', 'MYL9', 'TPM2'],
        'Inflammatory': ['IL6', 'CXCL12', 'CCL2', 'IL1B'],
        'Matrix': ['COL1A1', 'COL1A2', 'COL3A1', 'FN1'],
        'Antigen_presenting': ['HLA-DRA', 'HLA-DRB1', 'CD74'],
    },
    'Endo.': {
        'Tip': ['CXCR4', 'APLN', 'ESM1', 'DLL4'],
        'Stalk': ['SELP', 'VWF', 'ACKR1'],
        'Lymphatic': ['PROX1', 'LYVE1', 'FLT4', 'PDPN'],
        'IFN_stimulated': ['ISG15', 'MX1', 'STAT1'],
        'Proliferating': ['MKI67', 'TOP2A', 'PCNA'],
    },
}


def run_subclustering(adata_all, celltype_key, tf_file_key, celltype_name,
                      resolution=0.3, n_top_tfs=15, min_cells=50):
    """Run TF-based subclustering for a given cell type."""
    print(f"\n{'='*70}")
    print(f"SUBCLUSTERING: {celltype_name} ({celltype_key})")
    print(f"{'='*70}")

    # Extract cells
    mask = adata_all.obs['cell.types'] == celltype_key
    n_cells = mask.sum()
    print(f"  Total cells: {n_cells}")

    if n_cells < min_cells:
        print(f"  SKIPPING: too few cells (< {min_cells})")
        return None

    adata_sub = adata_all[mask].copy()

    # Load SHAP TFs
    tf_file = f'{OUT_DIR}/tf_importance_{tf_file_key}.csv'
    tf_df = pd.read_csv(tf_file)
    top_tfs = tf_df.head(n_top_tfs)['TF'].tolist()
    available_tfs = [tf for tf in top_tfs if tf in adata_sub.var_names]
    print(f"  SHAP TFs: {available_tfs}")

    if len(available_tfs) < 5:
        print(f"  SKIPPING: too few TFs available (< 5)")
        return None

    # TF-based subclustering
    adata_tf = adata_sub[:, available_tfs].copy()
    sc.pp.scale(adata_tf, max_value=10)
    n_pcs = min(10, len(available_tfs) - 1)
    sc.tl.pca(adata_tf, n_comps=n_pcs, random_state=42)
    sc.pp.neighbors(adata_tf, n_pcs=n_pcs, n_neighbors=15, random_state=42)

    # Adjust resolution based on cell count
    if n_cells < 150:
        resolution = 0.15
    elif n_cells < 300:
        resolution = 0.2

    sc.tl.leiden(adata_tf, resolution=resolution, key_added='tf_subtype', random_state=42)
    sc.tl.umap(adata_tf, random_state=42)

    adata_sub.obs['tf_subtype'] = adata_tf.obs['tf_subtype'].values
    adata_sub.obsm['X_umap_tf'] = adata_tf.obsm['X_umap']

    n_subtypes = adata_sub.obs['tf_subtype'].nunique()
    print(f"  Leiden clusters: {n_subtypes} (resolution={resolution})")

    # ── Characterize subtypes ─────────────────────────────────────────────────
    results = []
    for cluster in sorted(adata_sub.obs['tf_subtype'].unique()):
        cl_mask = adata_sub.obs['tf_subtype'] == cluster
        n_cl = cl_mask.sum()

        # Treatment composition
        treat_vals = adata_sub.obs.loc[cl_mask, 'treatment.group']
        n_naive = (treat_vals == 'treatment.naive').sum()
        n_post = n_cl - n_naive
        post_pct = n_post / n_cl * 100

        # TF expression
        tf_expr = {}
        for tf in available_tfs:
            if tf in adata_sub.var_names:
                tf_expr[tf] = float(adata_sub[cl_mask, tf].X.mean())

        # Known marker expression
        marker_scores = {}
        if celltype_key in known_markers:
            for sig_name, sig_genes in known_markers[celltype_key].items():
                avail_genes = [g for g in sig_genes if g in adata_sub.var_names]
                if avail_genes:
                    marker_scores[sig_name] = float(adata_sub[cl_mask][:, avail_genes].X.mean())
                else:
                    marker_scores[sig_name] = 0.0

        # Fisher's exact test for treatment enrichment
        # Contingency: [this_cluster_post, this_cluster_naive], [other_post, other_naive]
        other_mask = ~cl_mask
        other_naive = (adata_sub.obs.loc[other_mask, 'treatment.group'] == 'treatment.naive').sum()
        other_post = other_mask.sum() - other_naive
        table = [[n_post, n_naive], [other_post, other_naive]]
        or_val, fisher_p = fisher_exact(table)

        row = {
            'cluster': cluster,
            'n': n_cl,
            'n_naive': n_naive,
            'n_post': n_post,
            'post_pct': post_pct,
            'fisher_OR': or_val,
            'fisher_p': fisher_p,
        }
        row.update({f'TF_{k}': v for k, v in tf_expr.items()})
        row.update({f'Marker_{k}': v for k, v in marker_scores.items()})
        results.append(row)

    results_df = pd.DataFrame(results)

    # Print summary
    print(f"\n  Subtype summary:")
    for _, row in results_df.iterrows():
        sig = '*' if row['fisher_p'] < 0.05 else ''
        direction = 'POST-enriched' if row['post_pct'] > 60 else ('NAIVE-enriched' if row['post_pct'] < 40 else 'balanced')
        tf_strs = []
        for tf in available_tfs[:6]:
            col = f'TF_{tf}'
            if col in row:
                tf_strs.append(f"{tf}={row[col]:.2f}")
        marker_strs = []
        for col in row.index:
            if col.startswith('Marker_') and row[col] > 0.3:
                marker_strs.append(f"{col.replace('Marker_', '')}={row[col]:.2f}")

        print(f"    Cluster {row['cluster']}: n={row['n']}, post={row['post_pct']:.0f}% "
              f"(OR={row['fisher_OR']:.2f}, p={row['fisher_p']:.2e}{sig}) [{direction}]")
        print(f"      TFs: {', '.join(tf_strs)}")
        if marker_strs:
            print(f"      High markers: {', '.join(marker_strs[:6])}")

    # ── DEG analysis for treatment-enriched subtypes ──────────────────────────
    deg_results = {}
    for _, row in results_df.iterrows():
        cluster = row['cluster']
        # Only run DEG if strongly enriched and enough cells
        if (row['post_pct'] > 65 or row['post_pct'] < 35) and row['n'] >= 20 and row['fisher_p'] < 0.1:
            cl_mask = adata_sub.obs['tf_subtype'] == cluster
            other_mask = ~cl_mask

            # DEG: subtype vs rest
            genes_tested = []
            for gene_idx in range(adata_sub.n_vars):
                gene = adata_sub.var_names[gene_idx]
                expr_in = adata_sub[cl_mask, gene].X.flatten()
                expr_out = adata_sub[other_mask, gene].X.flatten()

                pct_in = (expr_in > 0).mean()
                pct_out = (expr_out > 0).mean()

                if pct_in < 0.1 and pct_out < 0.1:
                    continue

                try:
                    stat, pval = mannwhitneyu(expr_in, expr_out, alternative='two-sided')
                except:
                    continue

                mean_in = expr_in.mean()
                mean_out = expr_out.mean()
                log2fc = np.log2((mean_in + 0.01) / (mean_out + 0.01))

                genes_tested.append({
                    'gene': gene,
                    'mean_subtype': mean_in,
                    'mean_other': mean_out,
                    'log2fc': log2fc,
                    'pct_subtype': pct_in,
                    'pct_other': pct_out,
                    'pvalue': pval,
                })

            if genes_tested:
                deg_df = pd.DataFrame(genes_tested)
                _, padj, _, _ = multipletests(deg_df['pvalue'].values, method='fdr_bh')
                deg_df['padj'] = padj
                deg_df = deg_df.sort_values('padj')

                n_sig = (deg_df['padj'] < 0.05).sum()
                n_up = ((deg_df['padj'] < 0.05) & (deg_df['log2fc'] > 0)).sum()
                n_down = ((deg_df['padj'] < 0.05) & (deg_df['log2fc'] < 0)).sum()

                direction_label = 'POST' if row['post_pct'] > 50 else 'NAIVE'
                print(f"\n    DEG for cluster {cluster} ({direction_label}-enriched): "
                      f"{n_sig} DEGs ({n_up} up, {n_down} down)")

                # Top genes
                top_up = deg_df[(deg_df['padj'] < 0.05) & (deg_df['log2fc'] > 0)].nlargest(10, 'log2fc')
                top_down = deg_df[(deg_df['padj'] < 0.05) & (deg_df['log2fc'] < 0)].nsmallest(10, 'log2fc')

                if len(top_up) > 0:
                    print(f"      Top up: {', '.join(top_up['gene'].values[:8])}")
                if len(top_down) > 0:
                    print(f"      Top down: {', '.join(top_down['gene'].values[:8])}")

                deg_results[cluster] = deg_df

    # Save results
    save_prefix = f'{OUT_DIR}/subclustering_{tf_file_key}'
    results_df.to_csv(f'{save_prefix}_summary.csv', index=False)
    print(f"\n  Saved: {save_prefix}_summary.csv")

    for cluster, deg_df in deg_results.items():
        deg_df.to_csv(f'{save_prefix}_cluster{cluster}_deg.csv', index=False)
        print(f"  Saved: {save_prefix}_cluster{cluster}_deg.csv")

    return {
        'adata_sub': adata_sub,
        'results_df': results_df,
        'deg_results': deg_results,
        'available_tfs': available_tfs,
    }


# ── Run all cell types ────────────────────────────────────────────────────────
all_results = {}

for celltype_key, (tf_file_key, celltype_name) in celltype_map.items():
    result = run_subclustering(
        adata_all, celltype_key, tf_file_key, celltype_name,
        resolution=0.3, n_top_tfs=15
    )
    if result is not None:
        all_results[celltype_key] = result


# ── Summary of all findings ───────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("COMPREHENSIVE SUMMARY: ALL CELL TYPE SUBCLUSTERING")
print("=" * 70)

for celltype_key, result in all_results.items():
    tf_file_key = celltype_map[celltype_key][0]
    celltype_name = celltype_map[celltype_key][1]
    results_df = result['results_df']

    print(f"\n{celltype_name} ({celltype_key}):")
    print(f"  Total clusters: {len(results_df)}")

    # Treatment-enriched subtypes
    sig_post = results_df[(results_df['post_pct'] > 60) & (results_df['fisher_p'] < 0.05)]
    sig_naive = results_df[(results_df['post_pct'] < 40) & (results_df['fisher_p'] < 0.05)]

    if len(sig_post) > 0:
        for _, row in sig_post.iterrows():
            print(f"  ** POST-enriched: cluster {row['cluster']} "
                  f"(n={row['n']}, {row['post_pct']:.0f}%, p={row['fisher_p']:.2e})")
            # Top TFs
            tf_cols = [c for c in row.index if c.startswith('TF_')]
            tf_vals = [(c.replace('TF_', ''), row[c]) for c in tf_cols if row[c] > 0.3]
            tf_vals.sort(key=lambda x: -x[1])
            if tf_vals:
                print(f"    High TFs: {', '.join(f'{t}={v:.2f}' for t, v in tf_vals[:5])}")
            # Top markers
            mk_cols = [c for c in row.index if c.startswith('Marker_')]
            mk_vals = [(c.replace('Marker_', ''), row[c]) for c in mk_cols if row[c] > 0.3]
            mk_vals.sort(key=lambda x: -x[1])
            if mk_vals:
                print(f"    High markers: {', '.join(f'{t}={v:.2f}' for t, v in mk_vals[:5])}")

    if len(sig_naive) > 0:
        for _, row in sig_naive.iterrows():
            print(f"  ** NAIVE-enriched: cluster {row['cluster']} "
                  f"(n={row['n']}, {row['post_pct']:.0f}%, p={row['fisher_p']:.2e})")
            tf_cols = [c for c in row.index if c.startswith('TF_')]
            tf_vals = [(c.replace('TF_', ''), row[c]) for c in tf_cols if row[c] > 0.3]
            tf_vals.sort(key=lambda x: -x[1])
            if tf_vals:
                print(f"    High TFs: {', '.join(f'{t}={v:.2f}' for t, v in tf_vals[:5])}")
            mk_cols = [c for c in row.index if c.startswith('Marker_')]
            mk_vals = [(c.replace('Marker_', ''), row[c]) for c in mk_cols if row[c] > 0.3]
            mk_vals.sort(key=lambda x: -x[1])
            if mk_vals:
                print(f"    High markers: {', '.join(f'{t}={v:.2f}' for t, v in mk_vals[:5])}")

    if len(sig_post) == 0 and len(sig_naive) == 0:
        print("  No significantly treatment-enriched subtypes (Fisher's exact p > 0.05)")

    # DEG summary
    for cluster, deg_df in result['deg_results'].items():
        n_sig = (deg_df['padj'] < 0.05).sum()
        if n_sig > 0:
            print(f"  DEGs for cluster {cluster}: {n_sig} genes")


print("\n\n=== ALL ANALYSES COMPLETE ===")
