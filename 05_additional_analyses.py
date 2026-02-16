"""
05_additional_analyses.py
Additional analyses to strengthen the IRF8-high CD8+ T cell finding:
1. DEG analysis: IRF8-high vs other CD8+ T cells (discovery + validation)
2. Gene set enrichment analysis (GSEA) on IRF8-high DEGs
3. Response association analysis (GSE120575 responder/non-responder)
4. TCR clonality data check (GSE120575)
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency
from statsmodels.stats.multitest import multipletests
import os
import gzip

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)
DATA_DIR = os.path.join(BASE_DIR, 'data')

###############################################################################
# HELPER: Load discovery cohort and re-cluster CD8+ T cells
###############################################################################
def load_discovery_cd8():
    """Load discovery cohort and return adata_cd8 with subtype labels."""
    print("Loading discovery cohort (GSE115978)...")
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

    # Get CD8+ T cells
    adata_cd8_sub = sc.read_h5ad(f'{OUT_DIR}/adata_cd8_subtypes.h5ad')

    # Re-cluster with res=0.25 (same as 07 script)
    tf_cd8 = pd.read_csv(f'{OUT_DIR}/tf_importance_T_CD8.csv')
    top_cd8_tfs = tf_cd8.head(15)['TF'].tolist()
    available_cd8_tfs = [tf for tf in top_cd8_tfs if tf in adata_cd8_sub.var_names]

    adata_cd8_tf = adata_cd8_sub[:, available_cd8_tfs].copy()
    sc.pp.scale(adata_cd8_tf, max_value=10)
    sc.tl.pca(adata_cd8_tf, n_comps=10, random_state=42)
    sc.pp.neighbors(adata_cd8_tf, n_pcs=10, n_neighbors=15, random_state=42)
    sc.tl.leiden(adata_cd8_tf, resolution=0.25, key_added='tf_subtype_v2', random_state=42)

    adata_cd8_sub.obs['tf_subtype_v2'] = adata_cd8_tf.obs['tf_subtype_v2'].values

    # Assign labels
    cd8_sub_stats = {}
    for st in sorted(adata_cd8_sub.obs['tf_subtype_v2'].unique()):
        mask = adata_cd8_sub.obs['tf_subtype_v2'] == st
        cd8_sub_stats[st] = {
            'n': mask.sum(),
            'irf8': adata_cd8_sub[mask, 'IRF8'].X.mean() if 'IRF8' in adata_cd8_sub.var_names else 0,
            'tox': adata_cd8_sub[mask, 'TOX'].X.mean() if 'TOX' in adata_cd8_sub.var_names else 0,
            'tcf7': adata_cd8_sub[mask, 'TCF7'].X.mean() if 'TCF7' in adata_cd8_sub.var_names else 0,
            'prdm1': adata_cd8_sub[mask, 'PRDM1'].X.mean() if 'PRDM1' in adata_cd8_sub.var_names else 0,
            'id2': adata_cd8_sub[mask, 'ID2'].X.mean() if 'ID2' in adata_cd8_sub.var_names else 0,
        }
        exhaustion_genes = ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'TOX', 'ENTPD1']
        effector_genes = ['GZMA', 'GZMB', 'GZMK', 'PRF1', 'IFNG', 'NKG7', 'GNLY']
        memory_genes = ['TCF7', 'LEF1', 'CCR7', 'SELL', 'IL7R']
        for sig_name, sig_genes in [('exhaust', exhaustion_genes), ('effector', effector_genes), ('memory', memory_genes)]:
            avail = [g for g in sig_genes if g in adata_cd8_sub.var_names]
            if avail:
                cd8_sub_stats[st][sig_name] = adata_cd8_sub[mask][:, avail].X.mean()

    cd8_labels = {}
    for st, stats in sorted(cd8_sub_stats.items()):
        if stats['irf8'] > 0.3:
            cd8_labels[st] = 'IRF8-high'
        elif stats.get('memory', 0) > 0.3 and stats['tcf7'] > 0.5:
            cd8_labels[st] = 'Memory (TCF7+)'
        elif stats.get('exhaust', 0) > 0.15 and stats['tox'] > 1.1:
            cd8_labels[st] = 'Exhausted (TOX-hi)'
        elif stats.get('effector', 0) > 0.6 and stats.get('id2', 0) > 1.8:
            cd8_labels[st] = 'Cytotoxic (ID2+PRDM1+)'
        elif stats.get('id2', 0) > 1.7 and stats.get('prdm1', 0) < 0.5:
            cd8_labels[st] = 'Innate-like (ID2+)'
        elif stats.get('effector', 0) > 0.5 and stats.get('prdm1', 0) > 0.8:
            cd8_labels[st] = 'Effector (PRDM1+)'
        elif stats.get('effector', 0) > 0.3 and stats.get('effector', 0) <= 0.5:
            cd8_labels[st] = 'Transitional'
        else:
            cd8_labels[st] = 'Effector'

    adata_cd8_sub.obs['subtype_label'] = adata_cd8_sub.obs['tf_subtype_v2'].map(cd8_labels)
    print(f"  CD8+ T cells: {adata_cd8_sub.n_obs}")
    print(f"  IRF8-high: {(adata_cd8_sub.obs['subtype_label'] == 'IRF8-high').sum()}")
    return adata_cd8_sub


def load_validation_cd8():
    """Load validation cohort and return adata_cd8 with subtype labels and response info."""
    print("\nLoading validation cohort (GSE120575)...")
    tpm_file = f'{DATA_DIR}/GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz'

    with gzip.open(tpm_file, 'rt') as f:
        line1 = f.readline().strip().split('\t')
        line2 = f.readline().strip().split('\t')

    cell_ids_val = line1
    patient_treat_info = line2

    tpm_val = pd.read_csv(tpm_file, sep='\t', skiprows=2, index_col=0, header=None, low_memory=False)
    if tpm_val.shape[1] == len(cell_ids_val):
        tpm_val.columns = cell_ids_val
    elif tpm_val.shape[1] == len(cell_ids_val) + 1:
        tpm_val = tpm_val.iloc[:, :len(cell_ids_val)]
        tpm_val.columns = cell_ids_val
    else:
        n_cols = min(tpm_val.shape[1], len(cell_ids_val))
        tpm_val = tpm_val.iloc[:, :n_cols]
        cell_ids_val = cell_ids_val[:n_cols]
        patient_treat_info = patient_treat_info[:n_cols]
        tpm_val.columns = cell_ids_val

    # Read metadata for response/therapy info
    meta = pd.read_csv(f'{DATA_DIR}/GSE120575_patient_ID_single_cells.txt.gz',
                       sep='\t', skiprows=19, low_memory=False, encoding='latin-1')

    meta_dict = {}
    for _, row in meta.iterrows():
        title = str(row['title'])
        meta_dict[title] = {
            'response': str(row.iloc[5]) if pd.notna(row.iloc[5]) else 'Unknown',
            'therapy': str(row.iloc[6]) if pd.notna(row.iloc[6]) else 'Unknown',
        }

    annot_data = []
    for i, cid in enumerate(cell_ids_val):
        pt_info = patient_treat_info[i] if i < len(patient_treat_info) else 'Unknown'
        resp = meta_dict.get(cid, {}).get('response', 'Unknown')
        ther = meta_dict.get(cid, {}).get('therapy', 'Unknown')
        treatment = 'Pre' if str(pt_info).startswith('Pre') else ('Post' if str(pt_info).startswith('Post') else 'Unknown')
        parts = str(pt_info).split('_', 1)
        patient_id = parts[1] if len(parts) > 1 else pt_info
        annot_data.append({
            'cell_id': cid,
            'patient_treatment': pt_info,
            'treatment': treatment,
            'patient_id': patient_id,
            'response': resp,
            'therapy': ther,
        })

    annot_val = pd.DataFrame(annot_data, index=cell_ids_val)

    tpm_val_T = tpm_val.T
    adata_val = sc.AnnData(X=tpm_val_T.values.astype(np.float32))
    adata_val.obs_names = tpm_val_T.index.tolist()
    adata_val.var_names = tpm_val.index.tolist()
    for col in annot_val.columns:
        adata_val.obs[col] = annot_val.loc[adata_val.obs_names, col].values

    sc.pp.log1p(adata_val)

    # Select CD8+ T cells
    cd8a_expr = adata_val[:, 'CD8A'].X.flatten() if 'CD8A' in adata_val.var_names else np.zeros(adata_val.n_obs)
    cd4_expr = adata_val[:, 'CD4'].X.flatten() if 'CD4' in adata_val.var_names else np.zeros(adata_val.n_obs)
    cd3d_expr = adata_val[:, 'CD3D'].X.flatten() if 'CD3D' in adata_val.var_names else np.zeros(adata_val.n_obs)

    cd8_mask_val = (cd8a_expr > 1.0) & (cd3d_expr > 0) & (cd8a_expr > cd4_expr)
    adata_cd8_val = adata_val[cd8_mask_val].copy()

    # TF-based subclustering
    tf_cd8_discovery = pd.read_csv(f'{OUT_DIR}/tf_importance_T_CD8.csv')
    top_cd8_tfs = tf_cd8_discovery.head(15)['TF'].tolist()
    available_tfs_val = [tf for tf in top_cd8_tfs if tf in adata_cd8_val.var_names]

    adata_cd8_val_tf = adata_cd8_val[:, available_tfs_val].copy()
    sc.pp.scale(adata_cd8_val_tf, max_value=10)
    sc.tl.pca(adata_cd8_val_tf, n_comps=min(10, len(available_tfs_val) - 1), random_state=42)
    sc.pp.neighbors(adata_cd8_val_tf, n_pcs=min(10, len(available_tfs_val) - 1), n_neighbors=15, random_state=42)
    sc.tl.leiden(adata_cd8_val_tf, resolution=0.25, key_added='tf_subtype', random_state=42)

    adata_cd8_val.obs['tf_subtype'] = adata_cd8_val_tf.obs['tf_subtype'].values

    # Assign labels
    val_subtype_stats = {}
    for st in sorted(adata_cd8_val.obs['tf_subtype'].unique()):
        mask = adata_cd8_val.obs['tf_subtype'] == st
        val_subtype_stats[st] = {
            'n': mask.sum(),
            'irf8': adata_cd8_val[mask, 'IRF8'].X.mean() if 'IRF8' in adata_cd8_val.var_names else 0,
            'tox': adata_cd8_val[mask, 'TOX'].X.mean() if 'TOX' in adata_cd8_val.var_names else 0,
            'tcf7': adata_cd8_val[mask, 'TCF7'].X.mean() if 'TCF7' in adata_cd8_val.var_names else 0,
            'prdm1': adata_cd8_val[mask, 'PRDM1'].X.mean() if 'PRDM1' in adata_cd8_val.var_names else 0,
            'id2': adata_cd8_val[mask, 'ID2'].X.mean() if 'ID2' in adata_cd8_val.var_names else 0,
        }

    val_labels = {}
    for st, stats in sorted(val_subtype_stats.items()):
        if stats['irf8'] > 0.3:
            val_labels[st] = 'IRF8-high'
        elif stats['tcf7'] > 0.5:
            val_labels[st] = 'Memory (TCF7+)'
        elif stats['tox'] > 1.1:
            val_labels[st] = 'Exhausted (TOX-hi)'
        elif stats.get('id2', 0) > 1.7 and stats.get('prdm1', 0) > 0.8:
            val_labels[st] = 'Cytotoxic (ID2+PRDM1+)'
        elif stats.get('id2', 0) > 1.7:
            val_labels[st] = 'Innate-like (ID2+)'
        elif stats.get('prdm1', 0) > 0.8:
            val_labels[st] = 'Effector (PRDM1+)'
        else:
            val_labels[st] = 'Effector'

    adata_cd8_val.obs['subtype_label'] = adata_cd8_val.obs['tf_subtype'].map(val_labels)
    print(f"  CD8+ T cells: {adata_cd8_val.n_obs}")
    print(f"  IRF8-high: {(adata_cd8_val.obs['subtype_label'] == 'IRF8-high').sum()}")
    return adata_cd8_val


###############################################################################
# ANALYSIS 1: DEG analysis - IRF8-high vs other CD8+ T cells
###############################################################################
def run_deg_analysis(adata_cd8, cohort_name):
    """Wilcoxon rank-sum DEG analysis: IRF8-high vs other CD8+ T cells."""
    print(f"\n{'='*70}")
    print(f"ANALYSIS 1: DEG Analysis - IRF8-high vs Other CD8+ T cells ({cohort_name})")
    print(f"{'='*70}")

    irf8_mask = adata_cd8.obs['subtype_label'] == 'IRF8-high'
    n_irf8 = irf8_mask.sum()
    n_other = (~irf8_mask).sum()
    print(f"  IRF8-high: n={n_irf8}")
    print(f"  Other CD8+ T: n={n_other}")

    # Filter genes: expressed in at least 10% of either group
    n_genes = adata_cd8.shape[1]
    results = []

    irf8_data = adata_cd8[irf8_mask].X
    other_data = adata_cd8[~irf8_mask].X

    # Handle sparse matrix
    if hasattr(irf8_data, 'toarray'):
        irf8_data = irf8_data.toarray()
    if hasattr(other_data, 'toarray'):
        other_data = other_data.toarray()

    print(f"  Testing {n_genes} genes...")
    tested = 0
    for i in range(n_genes):
        gene = adata_cd8.var_names[i]
        irf8_expr = irf8_data[:, i]
        other_expr = other_data[:, i]

        # Filter: expressed in >=10% of either group
        pct_irf8 = (irf8_expr > 0).mean()
        pct_other = (other_expr > 0).mean()

        if pct_irf8 < 0.10 and pct_other < 0.10:
            continue

        tested += 1
        try:
            stat, pval = mannwhitneyu(irf8_expr, other_expr, alternative='two-sided')
        except ValueError:
            continue

        mean_irf8 = irf8_expr.mean()
        mean_other = other_expr.mean()
        # log2FC = (mean_log1p_irf8 - mean_log1p_other) / ln(2)
        # Equivalent to log2(geometric_mean_irf8 / geometric_mean_other) for TPM+1
        log2fc = (mean_irf8 - mean_other) / np.log(2)

        results.append({
            'gene': gene,
            'mean_irf8_high': mean_irf8,
            'mean_other': mean_other,
            'log2fc': log2fc,
            'pct_irf8_high': pct_irf8 * 100,
            'pct_other': pct_other * 100,
            'pvalue': pval,
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("  No genes passed filtering")
        return df

    # BH correction
    _, padj, _, _ = multipletests(df['pvalue'].values, method='fdr_bh')
    df['padj'] = padj

    # Sort by padj
    df = df.sort_values('pvalue').reset_index(drop=True)

    n_sig = (df['padj'] < 0.05).sum()
    n_up = ((df['padj'] < 0.05) & (df['log2fc'] > 0)).sum()
    n_down = ((df['padj'] < 0.05) & (df['log2fc'] < 0)).sum()

    print(f"\n  Genes tested: {tested}")
    print(f"  Significant DEGs (FDR < 0.05): {n_sig}")
    print(f"    Upregulated in IRF8-high: {n_up}")
    print(f"    Downregulated in IRF8-high: {n_down}")

    # Top 20 up
    sig_up = df[(df['padj'] < 0.05) & (df['log2fc'] > 0)].head(20)
    print(f"\n  Top 20 upregulated genes in IRF8-high:")
    print(f"  {'Gene':<12} {'log2FC':>8} {'pct_IRF8':>9} {'pct_other':>10} {'FDR':>12}")
    print(f"  {'-'*55}")
    for _, row in sig_up.iterrows():
        print(f"  {row['gene']:<12} {row['log2fc']:>8.3f} {row['pct_irf8_high']:>8.1f}% {row['pct_other']:>9.1f}% {row['padj']:>12.2e}")

    # Top 20 down
    sig_down = df[(df['padj'] < 0.05) & (df['log2fc'] < 0)].sort_values('log2fc').head(20)
    print(f"\n  Top 20 downregulated genes in IRF8-high:")
    print(f"  {'Gene':<12} {'log2FC':>8} {'pct_IRF8':>9} {'pct_other':>10} {'FDR':>12}")
    print(f"  {'-'*55}")
    for _, row in sig_down.iterrows():
        print(f"  {row['gene']:<12} {row['log2fc']:>8.3f} {row['pct_irf8_high']:>8.1f}% {row['pct_other']:>9.1f}% {row['padj']:>12.2e}")

    # Check key functional genes
    print(f"\n  Key functional genes:")
    key_genes = {
        'Exhaustion': ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'TOX', 'ENTPD1'],
        'Effector': ['GZMA', 'GZMB', 'GZMK', 'PRF1', 'IFNG', 'NKG7', 'GNLY', 'FASLG'],
        'Cytokine': ['IL2', 'IFNG', 'TNF', 'IL10', 'IL21', 'CSF2'],
        'Costim': ['CD28', 'ICOS', 'TNFRSF9', 'CD27', 'TNFRSF4'],
        'Memory': ['TCF7', 'LEF1', 'CCR7', 'SELL', 'IL7R'],
        'Proliferation': ['MKI67', 'TOP2A', 'PCNA'],
        'TF': ['IRF8', 'IRF1', 'IRF4', 'IRF7', 'TOX', 'EOMES', 'TBX21', 'PRDM1', 'BATF', 'ID2', 'TCF7'],
        'Antigen_presentation': ['HLA-A', 'HLA-B', 'HLA-C', 'B2M', 'TAP1', 'TAP2'],
        'Interferon': ['STAT1', 'STAT2', 'IRF1', 'IRF7', 'IRF9', 'ISG15', 'MX1', 'OAS1', 'IFIT1', 'IFIT3'],
    }

    for cat, genes in key_genes.items():
        print(f"\n  [{cat}]")
        for gene in genes:
            if gene in df['gene'].values:
                row = df[df['gene'] == gene].iloc[0]
                sig_mark = '*' if row['padj'] < 0.05 else ''
                dir_mark = '↑' if row['log2fc'] > 0 else '↓'
                print(f"    {gene:<12} log2FC={row['log2fc']:>7.3f} FDR={row['padj']:>10.2e} {dir_mark}{sig_mark}")

    return df


###############################################################################
# ANALYSIS 2: Gene Set Enrichment / Pathway Analysis
###############################################################################
def run_pathway_analysis(deg_df, cohort_name):
    """Simple GO-term-like pathway analysis using curated gene sets."""
    print(f"\n{'='*70}")
    print(f"ANALYSIS 2: Pathway Analysis on IRF8-high DEGs ({cohort_name})")
    print(f"{'='*70}")

    sig = deg_df[deg_df['padj'] < 0.05].copy()
    up_genes = set(sig[sig['log2fc'] > 0]['gene'].values)
    down_genes = set(sig[sig['log2fc'] < 0]['gene'].values)
    all_tested = set(deg_df['gene'].values)

    print(f"  Significant DEGs: {len(sig)} (up={len(up_genes)}, down={len(down_genes)})")
    print(f"  Total tested genes: {len(all_tested)}")

    # Curated gene sets for enrichment
    gene_sets = {
        # Immune pathways
        'Type_I_IFN_signaling': ['STAT1', 'STAT2', 'IRF1', 'IRF7', 'IRF9', 'ISG15', 'ISG20',
                                  'MX1', 'MX2', 'OAS1', 'OAS2', 'OAS3', 'OASL', 'IFIT1', 'IFIT2',
                                  'IFIT3', 'IFIT5', 'IFI44', 'IFI44L', 'IFI6', 'IFI27', 'IFI35',
                                  'IFITM1', 'IFITM2', 'IFITM3', 'BST2', 'XAF1', 'RSAD2',
                                  'USP18', 'HERC5', 'DDX58', 'IFIH1', 'EIF2AK2', 'GBP1', 'GBP2',
                                  'GBP4', 'GBP5', 'PARP9', 'PARP14', 'DTX3L', 'TRIM22', 'TRIM25'],
        'Type_II_IFN_signaling': ['STAT1', 'IRF1', 'IRF8', 'IRF9', 'GBP1', 'GBP2', 'GBP4', 'GBP5',
                                   'TAP1', 'TAP2', 'PSMB8', 'PSMB9', 'PSMB10', 'HLA-A', 'HLA-B',
                                   'HLA-C', 'HLA-E', 'HLA-F', 'B2M', 'CIITA', 'NLRC5',
                                   'CXCL9', 'CXCL10', 'CXCL11', 'IDO1', 'WARS'],
        'T_cell_exhaustion': ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'TOX', 'TOX2',
                              'ENTPD1', 'CD244', 'CD160', 'BTLA', 'EOMES', 'NFATC1',
                              'PRDM1', 'BATF', 'IRF4', 'NR4A1', 'NR4A2', 'NR4A3'],
        'T_cell_effector': ['GZMA', 'GZMB', 'GZMH', 'GZMK', 'GZMM', 'PRF1', 'IFNG', 'TNF',
                            'NKG7', 'GNLY', 'FASLG', 'CST7', 'KLRK1', 'KLRD1', 'FCGR3A'],
        'T_cell_costimulation': ['CD28', 'ICOS', 'TNFRSF9', 'CD27', 'TNFRSF4', 'TNFRSF18',
                                  'CD226', 'SLAMF6', 'SLAMF7', 'CD2'],
        'Antigen_processing_presentation': ['HLA-A', 'HLA-B', 'HLA-C', 'HLA-E', 'HLA-F',
                                            'B2M', 'TAP1', 'TAP2', 'TAPBP', 'CALR', 'CANX',
                                            'PDIA3', 'PSME1', 'PSME2', 'PSMB8', 'PSMB9', 'PSMB10',
                                            'ERAP1', 'ERAP2'],
        'TCR_signaling': ['CD3D', 'CD3E', 'CD3G', 'CD247', 'LCK', 'FYN', 'ZAP70',
                          'LAT', 'SLP76', 'PLCG1', 'ITK', 'CARD11', 'BCL10', 'MALT1',
                          'NFKBIA', 'NFKB1', 'NFKB2', 'REL', 'RELA'],
        'Cell_cycle_proliferation': ['MKI67', 'TOP2A', 'PCNA', 'MCM2', 'MCM3', 'MCM4',
                                     'MCM5', 'MCM6', 'MCM7', 'CDK1', 'CDK2', 'CDK4',
                                     'CCNA2', 'CCNB1', 'CCNB2', 'CCND1', 'CCNE1',
                                     'AURKA', 'AURKB', 'BUB1', 'BUB1B'],
        'Apoptosis': ['BCL2', 'BCL2L1', 'MCL1', 'BAX', 'BAK1', 'BID', 'BAD', 'BIM',
                      'CASP3', 'CASP8', 'CASP9', 'CYCS', 'APAF1', 'XIAP',
                      'FAS', 'FASLG', 'TNFRSF10A', 'TNFRSF10B'],
        'Cytokine_signaling': ['IL2', 'IL2RA', 'IL2RB', 'IL2RG', 'IL7R', 'IL15', 'IL15RA',
                               'IL21', 'IL21R', 'IL10', 'IL10RA', 'IFNG', 'IFNGR1', 'IFNGR2',
                               'TNF', 'TNFRSF1A', 'TNFRSF1B', 'CSF2', 'CXCR3', 'CXCR6',
                               'CCR5', 'CCR7', 'CX3CR1', 'CXCL13'],
        'Ribosome': ['RPS2', 'RPS3', 'RPS3A', 'RPS4X', 'RPS5', 'RPS6', 'RPS7', 'RPS8',
                     'RPS9', 'RPS10', 'RPS11', 'RPS12', 'RPS13', 'RPS14', 'RPS15',
                     'RPS15A', 'RPS16', 'RPS17', 'RPS18', 'RPS19', 'RPS20', 'RPS21',
                     'RPS23', 'RPS24', 'RPS25', 'RPS26', 'RPS27', 'RPS27A', 'RPS28',
                     'RPS29', 'RPL3', 'RPL4', 'RPL5', 'RPL6', 'RPL7', 'RPL7A', 'RPL8',
                     'RPL9', 'RPL10', 'RPL10A', 'RPL11', 'RPL12', 'RPL13', 'RPL13A',
                     'RPL14', 'RPL15', 'RPL17', 'RPL18', 'RPL18A', 'RPL19', 'RPL21',
                     'RPL22', 'RPL23', 'RPL23A', 'RPL24', 'RPL26', 'RPL27', 'RPL27A',
                     'RPL28', 'RPL29', 'RPL30', 'RPL31', 'RPL32', 'RPL34', 'RPL35',
                     'RPL35A', 'RPL36', 'RPL37', 'RPL37A', 'RPL38', 'RPL39', 'RPL41'],
        'Oxidative_phosphorylation': ['NDUFA1', 'NDUFA2', 'NDUFA3', 'NDUFA4', 'NDUFA5',
                                      'NDUFB1', 'NDUFB2', 'NDUFB3', 'NDUFB4', 'NDUFB5',
                                      'NDUFS1', 'NDUFS2', 'NDUFS3', 'NDUFS4', 'NDUFS5',
                                      'SDHA', 'SDHB', 'SDHC', 'SDHD',
                                      'UQCRC1', 'UQCRC2', 'UQCRH', 'UQCRFS1',
                                      'COX4I1', 'COX5A', 'COX5B', 'COX6A1', 'COX6B1',
                                      'COX6C', 'COX7A2', 'COX7B', 'COX7C', 'COX8A',
                                      'ATP5F1A', 'ATP5F1B', 'ATP5F1C', 'ATP5F1D', 'ATP5F1E',
                                      'ATP5MC1', 'ATP5MC2', 'ATP5MC3', 'ATP5PB', 'ATP5PD',
                                      'ATP5PF', 'ATP5PO', 'ATP5ME', 'ATP5MF', 'ATP5MG'],
        'MHC_class_II': ['HLA-DRA', 'HLA-DRB1', 'HLA-DPA1', 'HLA-DPB1', 'HLA-DQA1',
                          'HLA-DQB1', 'HLA-DMA', 'HLA-DMB', 'CD74', 'CIITA'],
    }

    # Fisher's exact test for enrichment
    print(f"\n  Gene set enrichment (Fisher's exact test):")
    print(f"  {'Gene Set':<35} {'Dir':>4} {'Overlap':>8} {'Set_in_bg':>10} {'OR':>7} {'p-value':>12} {'FDR':>12} {'Genes'}")
    print(f"  {'-'*130}")

    enrichment_results = []
    for gs_name, gs_genes in gene_sets.items():
        gs_in_tested = set(gs_genes) & all_tested
        if len(gs_in_tested) < 3:
            continue

        for direction, deg_set, dir_label in [(up_genes, up_genes, 'UP'), (down_genes, down_genes, 'DOWN')]:
            overlap = gs_in_tested & deg_set
            n_overlap = len(overlap)
            n_gs_no_deg = len(gs_in_tested) - n_overlap
            n_deg_no_gs = len(deg_set) - n_overlap
            n_neither = len(all_tested) - len(gs_in_tested) - len(deg_set) + n_overlap

            if n_overlap < 2:
                continue

            table = [[n_overlap, n_gs_no_deg], [n_deg_no_gs, n_neither]]
            try:
                odds, pval = fisher_exact(table, alternative='greater')
            except ValueError:
                continue

            enrichment_results.append({
                'gene_set': gs_name,
                'direction': dir_label,
                'overlap': n_overlap,
                'set_size_in_bg': len(gs_in_tested),
                'odds_ratio': odds,
                'pvalue': pval,
                'overlap_genes': ','.join(sorted(overlap)),
            })

    if enrichment_results:
        enr_df = pd.DataFrame(enrichment_results)
        _, padj, _, _ = multipletests(enr_df['pvalue'].values, method='fdr_bh')
        enr_df['padj'] = padj
        enr_df = enr_df.sort_values('pvalue').reset_index(drop=True)

        for _, row in enr_df.iterrows():
            sig_str = '***' if row['padj'] < 0.001 else ('**' if row['padj'] < 0.01 else ('*' if row['padj'] < 0.05 else ''))
            genes_str = row['overlap_genes']
            if len(genes_str) > 50:
                genes_str = genes_str[:50] + '...'
            print(f"  {row['gene_set']:<35} {row['direction']:>4} {row['overlap']:>5}/{row['set_size_in_bg']:<4} {row['odds_ratio']:>7.2f} {row['pvalue']:>12.2e} {row['padj']:>12.2e}{sig_str:>3} {genes_str}")

        # Save
        outfile = f'{OUT_DIR}/irf8high_pathway_enrichment_{cohort_name.lower()}.csv'
        enr_df.to_csv(outfile, index=False)
        print(f"\n  Saved to {outfile}")
        return enr_df
    else:
        print("  No significant enrichments found.")
        return pd.DataFrame()


###############################################################################
# ANALYSIS 3: Response Association (GSE120575)
###############################################################################
def run_response_analysis(adata_cd8_val):
    """Check if IRF8-high proportion differs between responders and non-responders."""
    print(f"\n{'='*70}")
    print(f"ANALYSIS 3: Response Association (GSE120575)")
    print(f"{'='*70}")

    # Check response data
    print(f"\n  Response distribution in all CD8+ T cells:")
    print(adata_cd8_val.obs['response'].value_counts())

    # Clean response categories
    resp = adata_cd8_val.obs['response'].astype(str)
    # Standardize
    resp_clean = resp.copy()
    resp_clean = resp_clean.str.strip()

    print(f"\n  Unique response values: {resp_clean.unique()}")

    # Map to Responder / Non-responder
    resp_map = {}
    for v in resp_clean.unique():
        vl = v.lower()
        if 'respond' in vl and 'non' not in vl:
            resp_map[v] = 'Responder'
        elif 'non' in vl or 'no response' in vl or 'progressive' in vl:
            resp_map[v] = 'Non-responder'
        elif 'partial' in vl or 'complete' in vl or 'stable' in vl:
            # RECIST criteria
            if 'complete' in vl or 'partial' in vl:
                resp_map[v] = 'Responder'
            else:
                resp_map[v] = 'Non-responder'  # Stable disease → non-responder
        else:
            resp_map[v] = v  # Keep as-is

    print(f"\n  Response mapping: {resp_map}")
    adata_cd8_val.obs['response_binary'] = resp_clean.map(resp_map)

    print(f"\n  Binary response distribution:")
    print(adata_cd8_val.obs['response_binary'].value_counts())

    # Cell-level: IRF8-high proportion by response
    for resp_val in ['Responder', 'Non-responder']:
        mask = adata_cd8_val.obs['response_binary'] == resp_val
        if mask.sum() == 0:
            continue
        n_total = mask.sum()
        n_irf8 = ((adata_cd8_val.obs['subtype_label'] == 'IRF8-high') & mask).sum()
        print(f"\n  {resp_val}: {n_irf8}/{n_total} IRF8-high ({n_irf8/n_total*100:.1f}%)")

    # Cell-level Fisher's exact: IRF8-high vs response
    resp_mask = adata_cd8_val.obs['response_binary'] == 'Responder'
    nonresp_mask = adata_cd8_val.obs['response_binary'] == 'Non-responder'
    irf8_mask = adata_cd8_val.obs['subtype_label'] == 'IRF8-high'

    if resp_mask.sum() > 0 and nonresp_mask.sum() > 0:
        a = (irf8_mask & resp_mask).sum()  # IRF8-high & responder
        b = (irf8_mask & nonresp_mask).sum()  # IRF8-high & non-responder
        c = (~irf8_mask & resp_mask).sum()  # Other & responder
        d = (~irf8_mask & nonresp_mask).sum()  # Other & non-responder

        table = [[a, b], [c, d]]
        odds, pval = fisher_exact(table)
        print(f"\n  Cell-level Fisher's exact (IRF8-high × Response):")
        print(f"    IRF8-high Responder: {a}, IRF8-high Non-resp: {b}")
        print(f"    Other Responder: {c}, Other Non-resp: {d}")
        print(f"    OR = {odds:.3f}, p = {pval:.4e}")
        if odds > 1:
            print(f"    → IRF8-high cells enriched in RESPONDERS")
        else:
            print(f"    → IRF8-high cells enriched in NON-RESPONDERS")

    # Patient-level: proportion of IRF8-high by response
    print(f"\n  --- Patient-level response analysis ---")
    patient_ids = adata_cd8_val.obs['patient_id'].unique()

    patient_resp_data = []
    for pt in patient_ids:
        pt_mask = adata_cd8_val.obs['patient_id'] == pt
        n_total = pt_mask.sum()
        if n_total < 5:
            continue
        n_irf8 = (adata_cd8_val.obs.loc[pt_mask, 'subtype_label'] == 'IRF8-high').sum()
        prop_irf8 = n_irf8 / n_total

        # Get response (majority vote)
        resp_vals = adata_cd8_val.obs.loc[pt_mask, 'response_binary'].value_counts()
        if len(resp_vals) == 0:
            continue
        resp = resp_vals.index[0]

        # Get treatment
        treat_vals = adata_cd8_val.obs.loc[pt_mask, 'treatment'].value_counts()
        treat = treat_vals.index[0]

        patient_resp_data.append({
            'patient': pt,
            'response': resp,
            'treatment': treat,
            'n_cd8': n_total,
            'n_irf8': n_irf8,
            'prop_irf8': prop_irf8,
        })

    pt_resp_df = pd.DataFrame(patient_resp_data)
    print(f"\n  Patients with ≥5 CD8+ T cells and response data:")
    print(pt_resp_df.groupby('response').agg({
        'patient': 'count',
        'prop_irf8': ['mean', 'median', 'std']
    }))

    # Filter to responder/non-responder only
    resp_pts = pt_resp_df[pt_resp_df['response'] == 'Responder']
    nonresp_pts = pt_resp_df[pt_resp_df['response'] == 'Non-responder']

    if len(resp_pts) >= 3 and len(nonresp_pts) >= 3:
        resp_props = resp_pts['prop_irf8'].values
        nonresp_props = nonresp_pts['prop_irf8'].values

        print(f"\n  Responders (n={len(resp_pts)}): IRF8-high prop mean={resp_props.mean():.4f}, median={np.median(resp_props):.4f}")
        print(f"  Non-responders (n={len(nonresp_pts)}): IRF8-high prop mean={nonresp_props.mean():.4f}, median={np.median(nonresp_props):.4f}")

        stat, pval = mannwhitneyu(resp_props, nonresp_props, alternative='two-sided')
        print(f"  Wilcoxon rank-sum (patient-level): p = {pval:.4e}")

        # Per-patient details
        print(f"\n  Per-patient details:")
        print(f"  {'Patient':<30} {'Response':<15} {'Treatment':<8} {'N_CD8':>6} {'N_IRF8':>6} {'Prop':>7}")
        print(f"  {'-'*80}")
        for _, row in pt_resp_df.sort_values(['response', 'prop_irf8']).iterrows():
            if row['response'] in ['Responder', 'Non-responder']:
                print(f"  {str(row['patient']):<30} {row['response']:<15} {row['treatment']:<8} {row['n_cd8']:>6} {row['n_irf8']:>6} {row['prop_irf8']:>7.3f}")
    else:
        print(f"\n  Not enough patients in each group for statistical testing")
        print(f"    Responder patients: {len(resp_pts)}")
        print(f"    Non-responder patients: {len(nonresp_pts)}")

    # Also check: all subtypes by response (for context)
    print(f"\n  --- All CD8+ T cell subtypes by response ---")
    subtypes = sorted(adata_cd8_val.obs['subtype_label'].unique())
    for st in subtypes:
        st_mask = adata_cd8_val.obs['subtype_label'] == st
        n_st = st_mask.sum()
        n_resp = (st_mask & resp_mask).sum()
        n_nonresp = (st_mask & nonresp_mask).sum()
        pct_resp = n_resp / (n_resp + n_nonresp) * 100 if (n_resp + n_nonresp) > 0 else 0
        print(f"  {st:<25} n={n_st:>5}, Resp={n_resp:>5} ({pct_resp:.1f}%), Non-resp={n_nonresp:>5}")

    # Save patient response data
    outfile = f'{OUT_DIR}/irf8high_response_patient_data.csv'
    pt_resp_df.to_csv(outfile, index=False)
    print(f"\n  Saved to {outfile}")
    return pt_resp_df


###############################################################################
# ANALYSIS 4: TCR Clonality Check
###############################################################################
def check_tcr_data(adata_cd8_val):
    """Check if TCR data is available in the GSE120575 dataset."""
    print(f"\n{'='*70}")
    print(f"ANALYSIS 4: TCR Clonality Data Check (GSE120575)")
    print(f"{'='*70}")

    # Check for TCR-related genes in the expression data
    tcr_genes = ['TRAC', 'TRBC1', 'TRBC2', 'TRDC', 'TRGC1', 'TRGC2',
                 'TRAV', 'TRBV', 'TRDV', 'TRGV']

    print(f"\n  Checking for TCR genes in expression matrix:")
    tcr_found = []
    for gene in adata_cd8_val.var_names:
        if gene.startswith(('TRA', 'TRB', 'TRD', 'TRG')) and not gene.startswith('TRAF'):
            tcr_found.append(gene)

    if tcr_found:
        print(f"  Found {len(tcr_found)} TCR-related genes:")
        for g in sorted(tcr_found)[:30]:
            expr = adata_cd8_val[:, g].X.flatten()
            pct = (expr > 0).mean() * 100
            print(f"    {g}: {pct:.1f}% expressing")

        # Check if V-gene diversity can be used as proxy for clonality
        # Higher diversity of V-genes → more diverse TCR repertoire
        v_genes = [g for g in tcr_found if any(g.startswith(p) for p in ['TRAV', 'TRBV'])]
        if v_genes:
            print(f"\n  TCR V-gene analysis (proxy for clonality):")
            print(f"  Found {len(v_genes)} V-genes")

            # For each cell, count number of V-genes expressed → proxy for poly-clonal signal
            # In Smart-seq2, a clonally expanded population would show more uniform V-gene usage
            irf8_mask = adata_cd8_val.obs['subtype_label'] == 'IRF8-high'

            # Compare V-gene expression patterns
            for prefix, label in [('TRAV', 'TCRα V'), ('TRBV', 'TCRβ V')]:
                vg = [g for g in v_genes if g.startswith(prefix)]
                if len(vg) < 3:
                    continue

                irf8_data = adata_cd8_val[irf8_mask, vg].X
                other_data = adata_cd8_val[~irf8_mask, vg].X

                if hasattr(irf8_data, 'toarray'):
                    irf8_data = irf8_data.toarray()
                if hasattr(other_data, 'toarray'):
                    other_data = other_data.toarray()

                # Number of V-genes expressed per cell
                irf8_n_vgenes = (irf8_data > 0).sum(axis=1)
                other_n_vgenes = (other_data > 0).sum(axis=1)

                stat, pval = mannwhitneyu(irf8_n_vgenes, other_n_vgenes, alternative='two-sided')
                print(f"\n  {label} genes expressed per cell:")
                print(f"    IRF8-high: mean={irf8_n_vgenes.mean():.2f}, median={np.median(irf8_n_vgenes):.0f}")
                print(f"    Other: mean={other_n_vgenes.mean():.2f}, median={np.median(other_n_vgenes):.0f}")
                print(f"    Wilcoxon p = {pval:.4e}")
        else:
            print("  No TCR V-genes found for clonality analysis")
    else:
        print("  No TCR-related genes found in expression matrix")
        print("  TCR clonality analysis cannot be performed with this dataset")

    # Check for TCR-related columns in metadata
    print(f"\n  Checking metadata columns for TCR info:")
    for col in adata_cd8_val.obs.columns:
        if 'tcr' in col.lower() or 'clone' in col.lower() or 'cdr3' in col.lower():
            print(f"    Found: {col}")

    # Check supplementary files
    print(f"\n  Checking for additional TCR data files:")
    import glob
    tcr_files = glob.glob(f'{DATA_DIR}/*tcr*') + glob.glob(f'{DATA_DIR}/*TCR*') + glob.glob(f'{DATA_DIR}/*clone*')
    if tcr_files:
        for f in tcr_files:
            print(f"    {f}")
    else:
        print(f"    No TCR-specific data files found in {DATA_DIR}")
        print(f"    Note: Sade-Feldman et al. reported TCR data but it may require separate GEO download")


###############################################################################
# MAIN EXECUTION
###############################################################################
if __name__ == '__main__':
    print("=" * 70)
    print("ADDITIONAL ANALYSES TO STRENGTHEN IRF8-HIGH FINDING")
    print("=" * 70)

    # Load data
    adata_cd8_disc = load_discovery_cd8()
    adata_cd8_val = load_validation_cd8()

    # Analysis 1: DEG (Discovery)
    deg_disc = run_deg_analysis(adata_cd8_disc, 'Discovery')
    deg_disc.to_csv(f'{OUT_DIR}/irf8high_deg_discovery.csv', index=False)
    print(f"  Saved to {OUT_DIR}/irf8high_deg_discovery.csv")

    # Analysis 1: DEG (Validation)
    deg_val = run_deg_analysis(adata_cd8_val, 'Validation')
    deg_val.to_csv(f'{OUT_DIR}/irf8high_deg_validation.csv', index=False)
    print(f"  Saved to {OUT_DIR}/irf8high_deg_validation.csv")

    # Overlap of DEGs between cohorts
    print(f"\n{'='*70}")
    print(f"DEG OVERLAP ANALYSIS")
    print(f"{'='*70}")
    sig_disc = set(deg_disc[deg_disc['padj'] < 0.05]['gene'])
    sig_val = set(deg_val[deg_val['padj'] < 0.05]['gene'])
    overlap = sig_disc & sig_val

    up_disc = set(deg_disc[(deg_disc['padj'] < 0.05) & (deg_disc['log2fc'] > 0)]['gene'])
    up_val = set(deg_val[(deg_val['padj'] < 0.05) & (deg_val['log2fc'] > 0)]['gene'])
    down_disc = set(deg_disc[(deg_disc['padj'] < 0.05) & (deg_disc['log2fc'] < 0)]['gene'])
    down_val = set(deg_val[(deg_val['padj'] < 0.05) & (deg_val['log2fc'] < 0)]['gene'])

    up_overlap = up_disc & up_val
    down_overlap = down_disc & down_val

    print(f"  Discovery DEGs: {len(sig_disc)} ({len(up_disc)} up, {len(down_disc)} down)")
    print(f"  Validation DEGs: {len(sig_val)} ({len(up_val)} up, {len(down_val)} down)")
    print(f"  Overlap (any direction): {len(overlap)}")
    print(f"  Consistent UP overlap: {len(up_overlap)}")
    print(f"  Consistent DOWN overlap: {len(down_overlap)}")

    # Check direction concordance for overlapping genes
    if len(overlap) > 0:
        concordant = 0
        discordant = 0
        for gene in overlap:
            fc_disc = deg_disc[deg_disc['gene'] == gene]['log2fc'].values[0]
            fc_val = deg_val[deg_val['gene'] == gene]['log2fc'].values[0]
            if (fc_disc > 0 and fc_val > 0) or (fc_disc < 0 and fc_val < 0):
                concordant += 1
            else:
                discordant += 1
        print(f"  Direction concordance: {concordant}/{len(overlap)} ({concordant/len(overlap)*100:.1f}%)")
        print(f"  Discordant: {discordant}/{len(overlap)}")

    # Top concordant genes
    if len(up_overlap) > 0:
        print(f"\n  Top concordantly upregulated genes (in both cohorts):")
        for gene in sorted(up_overlap)[:20]:
            r_d = deg_disc[deg_disc['gene'] == gene].iloc[0]
            r_v = deg_val[deg_val['gene'] == gene].iloc[0]
            print(f"    {gene:<12} Disc: log2FC={r_d['log2fc']:.3f} FDR={r_d['padj']:.2e} | Val: log2FC={r_v['log2fc']:.3f} FDR={r_v['padj']:.2e}")

    if len(down_overlap) > 0:
        print(f"\n  Top concordantly downregulated genes (in both cohorts):")
        for gene in sorted(down_overlap)[:20]:
            r_d = deg_disc[deg_disc['gene'] == gene].iloc[0]
            r_v = deg_val[deg_val['gene'] == gene].iloc[0]
            print(f"    {gene:<12} Disc: log2FC={r_d['log2fc']:.3f} FDR={r_d['padj']:.2e} | Val: log2FC={r_v['log2fc']:.3f} FDR={r_v['padj']:.2e}")

    # Analysis 2: Pathway enrichment (Discovery)
    enr_disc = run_pathway_analysis(deg_disc, 'Discovery')

    # Analysis 2: Pathway enrichment (Validation)
    enr_val = run_pathway_analysis(deg_val, 'Validation')

    # Analysis 3: Response association
    resp_df = run_response_analysis(adata_cd8_val)

    # Analysis 4: TCR clonality check
    check_tcr_data(adata_cd8_val)

    ###########################################################################
    # FINAL SUMMARY
    ###########################################################################
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY OF ADDITIONAL ANALYSES")
    print(f"{'='*70}")

    print(f"\n1. DEG ANALYSIS:")
    n_sig_d = (deg_disc['padj'] < 0.05).sum()
    n_sig_v = (deg_val['padj'] < 0.05).sum()
    print(f"   Discovery: {n_sig_d} DEGs (FDR<0.05)")
    print(f"   Validation: {n_sig_v} DEGs (FDR<0.05)")
    if len(overlap) > 0:
        print(f"   Overlap: {len(overlap)} genes ({concordant}/{len(overlap)} concordant direction)")

    print(f"\n2. PATHWAY ENRICHMENT:")
    if len(enr_disc) > 0:
        sig_enr = enr_disc[enr_disc['padj'] < 0.05]
        print(f"   Discovery: {len(sig_enr)} significant pathways (FDR<0.05)")
        for _, row in sig_enr.iterrows():
            print(f"     {row['gene_set']} ({row['direction']}): OR={row['odds_ratio']:.2f}, FDR={row['padj']:.2e}")
    if len(enr_val) > 0:
        sig_enr_v = enr_val[enr_val['padj'] < 0.05]
        print(f"   Validation: {len(sig_enr_v)} significant pathways (FDR<0.05)")
        for _, row in sig_enr_v.iterrows():
            print(f"     {row['gene_set']} ({row['direction']}): OR={row['odds_ratio']:.2f}, FDR={row['padj']:.2e}")

    print(f"\n3. RESPONSE ASSOCIATION: See above for details")
    print(f"\n4. TCR CLONALITY: See above for data availability")

    print(f"\n{'='*70}")
    print(f"ALL ADDITIONAL ANALYSES COMPLETE")
    print(f"{'='*70}")
