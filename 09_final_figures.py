"""
09_final_figures.py
Final publication-quality figures for BBRC Short Communication.
Replaces 12_generate_reorganized_figures.py with revised layout emphasizing
IRF8-high as an independent terminal differentiation state, incorporating
pseudotime analysis results.

Figure layout:
  Main Figure 1 (1x4): RF-SHAP Method + TF Ranking
  Main Figure 2 (1x4): CD8+ Subtype Discovery + Pseudotime + IRF8 co-expression
  Main Figure 3 (1x3): IRF8-high Characterization (DEGs + Pathways)
  Main Figure 4 (1x3): XCL1/XCL2 Functional Axis (cDC1 recruitment)
  Supplementary Figure S1 (1x3): CD8+ Lineage Markers + TF Expression
  Supplementary Figure S2 (2x2): Pseudotime Details
  Supplementary Figure S3 (2x3): Validation Cohort Details
  Supplementary Figure S4 (2x2): Tirosh Validation
  Supplementary Figure S5 (2x3): Treatment Changes + Malignant Subtypes
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu, fisher_exact, kruskal, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score
import os
from statsmodels.stats.multitest import multipletests
import gzip

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(BASE_DIR, 'figures')
DATA_DIR = os.path.join(BASE_DIR, 'data')

os.makedirs(FIG_DIR, exist_ok=True)

# ── Global rcParams ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 8,
    'font.family': 'Arial',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
})

# ── Color palettes ───────────────────────────────────────────────────────────
celltype_colors = {
    'Mal': '#E64B35', 'T.CD8': '#4DBBD5', 'T.CD4': '#00A087',
    'B.cell': '#3C5488', 'T.cell': '#F39B7F', 'Macrophage': '#8491B4',
    'CAF': '#91D1C2', 'Endo.': '#DC9FB4', 'NK': '#7E6148',
}
celltype_labels = {
    'Mal': 'Malignant', 'T.CD8': 'CD8+ T', 'T.CD4': 'CD4+ T',
    'B.cell': 'B cell', 'T.cell': 'T cell', 'Macrophage': 'Macrophage',
    'CAF': 'CAF', 'Endo.': 'Endothelial', 'NK': 'NK',
}
treat_palette = {'Pre-tx': '#4575B4', 'Post-tx': '#D73027'}

# CD8 subtype colors (consistent across discovery and validation)
merged_color_map = {
    'IRF8-high': '#E31A1C',
    'Exhausted (TOX-hi)': '#FF7F00',
    'Memory (TCF7+)': '#33A02C',
    'Cytotoxic (ID2+PRDM1+)': '#1F78B4',
    'Innate-like (ID2+)': '#A6CEE3',
    'Effector (PRDM1+)': '#6A3D9A',
    'Effector': '#B2DF8A',
    'Transitional': '#FDBF6F',
    # For pseudotime figures (simplified subtype names)
    'Memory-like': '#FDBF6F',
    'Exhausted (TOX+)': '#FF7F00',
}

# TF families list for RF (same as 07 script)
tf_families = {
    'FOX': ['FOXP1','FOXP3','FOXO1','FOXO3','FOXO4','FOXA1','FOXA2','FOXM1','FOXN1','FOXJ1','FOXK1','FOXK2','FOXL1','FOXL2','FOXC1','FOXC2','FOXD1','FOXD3','FOXE1','FOXF1','FOXF2','FOXG1','FOXH1','FOXI1','FOXP2','FOXP4','FOXQ1','FOXR1','FOXS1'],
    'TBX': ['TBX21','EOMES','TBX1','TBX3','TBX5','TBX6','TBX15','TBX18','TBX19','TBX20','BRACHYURY'],
    'bHLH': ['MYC','MYCN','MYCL','MAX','TCF3','TCF4','TCF12','HIF1A','ARNT','AHR','CLOCK','ID1','ID2','ID3','ID4','TWIST1','TWIST2','HEY1','HEY2','HES1','HES5','SNAI1','SNAI2','ASCL1','ASCL2','NEUROD1','NEUROG1','OLIG1','OLIG2','BHLHE40','BHLHE41'],
    'ZNF': ['GATA1','GATA2','GATA3','GATA4','GATA5','GATA6','KLF2','KLF4','KLF5','KLF6','KLF7','KLF9','KLF10','KLF13','EGR1','EGR2','EGR3','EGR4','ZEB1','ZEB2','PRDM1','PRDM2','PRDM5','IKZF1','IKZF2','IKZF3','IKZF4','IKZF5','ZNF683','ZNF281','ZNF148','ZNF652','WT1','GLI1','GLI2','GLI3','CTCF','YY1','SP1','SP3','SP4','SP100','REST','SNAI1','SNAI2'],
    'NR': ['NR4A1','NR4A2','NR4A3','NR3C1','RORA','RORC','PPARG','PPARA','PPARD','RXRA','RARA','RARB','RARG','VDR','ESR1','ESR2','AR','PGR','HNF4A','NR1H3','NR1H4','NR2F1','NR2F2','NR2F6','NR5A1','NR5A2','THRB','THRA'],
    'STAT': ['STAT1','STAT2','STAT3','STAT4','STAT5A','STAT5B','STAT6'],
    'IRF': ['IRF1','IRF2','IRF3','IRF4','IRF5','IRF7','IRF8','IRF9'],
    'NFKB': ['NFKB1','NFKB2','RELA','RELB','REL'],
    'AP1': ['FOS','FOSB','FOSL1','FOSL2','JUN','JUNB','JUND','BATF','BATF2','BATF3','ATF1','ATF2','ATF3','ATF4','ATF5','ATF6','ATF7'],
    'RUNX': ['RUNX1','RUNX2','RUNX3','CBFB'],
    'SOX': ['SOX2','SOX4','SOX5','SOX6','SOX7','SOX8','SOX9','SOX10','SOX11','SOX13','SOX17','SOX18','SOX21','SRY'],
    'ETS': ['ETS1','ETS2','ELF1','ELF4','ELF5','ELK1','ELK3','ELK4','SPI1','SPIB','SPIC','FLI1','ERG','ETV1','ETV4','ETV5','ETV6','GABPA','GABPB1','FEV','EHF','ESE1'],
    'NFAT': ['NFATC1','NFATC2','NFATC3','NFATC4','NFAT5'],
    'HOX': ['PAX5','PAX6','PAX3','PAX7','PAX8','PBX1','PBX3','MEIS1','MEIS2','HOXA9','HOXA10','HOXB4','HOXC8','HOXD10','LEF1','TCF7','TCF7L1','TCF7L2','CDX1','CDX2','DLX1','DLX5','MSX1','MSX2','LHX1','LHX2'],
    'OTHER': ['TP53','TP63','TP73','MYB','MYBL1','MYBL2','E2F1','E2F2','E2F3','E2F4','E2F5','RB1','CEBPA','CEBPB','CEBPD','CEBPE','CEBPG','BACH1','BACH2','MAFB','MAFK','MAFF','MAFG','MAF','NFE2','NFE2L2','NFE2L3','SMAD1','SMAD2','SMAD3','SMAD4','SMAD5','SMAD7','NOTCH1','NOTCH2','NOTCH3','BCL6','BCL11A','BCL11B','TOX','TOX2','TOX3','YBX1','YBX3','HMGA1','HMGA2','HMGB1','HMGB2','MXD1','MXD3','MXD4','MGA','MNT','ZFP36','ZFP36L1','ZFP36L2','TFEB','TFE3','MITF','USF1','USF2','XBP1','CREB1','CREB3','CREB5','EBF1','MEF2A','MEF2B','MEF2C','MEF2D','TEAD1','TEAD2','TEAD3','TEAD4','WWTR1','YAP1','HIPK2','SREBF1','SREBF2','MLX','MLXIP','HSF1','HSF2'],
}

# ── Helper: CD8 subtype labeling ─────────────────────────────────────────────
def assign_cd8_labels(adata_cd8_sub, subtype_col='tf_subtype_v2'):
    """Compute stats and assign biologically meaningful labels to CD8 subtypes.

    NOTE on cutoff thresholds (below): These thresholds are applied to
    log1p(TPM) mean expression values within Leiden clusters.  They are
    empirically chosen for the GSE115978 dataset (Smart-seq2, TPM).
    Different normalisation or library-prep methods will require
    re-calibration.  When cache_version matches, cached labels are used
    instead of re-running this function.
    """
    # Score gene sets
    exhaustion_genes = ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'TOX', 'ENTPD1']
    effector_genes = ['GZMA', 'GZMB', 'GZMK', 'PRF1', 'IFNG', 'NKG7', 'GNLY']
    memory_genes = ['TCF7', 'LEF1', 'CCR7', 'SELL', 'IL7R']
    for sig_name, sig_genes in [('exhaustion', exhaustion_genes),
                                 ('effector', effector_genes),
                                 ('memory', memory_genes)]:
        avail = [g for g in sig_genes if g in adata_cd8_sub.var_names]
        if avail:
            sc.tl.score_genes(adata_cd8_sub, avail, score_name=f'{sig_name}_score')

    cd8_sub_stats = {}
    for st in sorted(adata_cd8_sub.obs[subtype_col].unique()):
        mask = adata_cd8_sub.obs[subtype_col] == st
        cd8_sub_stats[st] = {
            'n': mask.sum(),
            'exhaust': adata_cd8_sub.obs.loc[mask, 'exhaustion_score'].mean() if 'exhaustion_score' in adata_cd8_sub.obs else 0,
            'effector': adata_cd8_sub.obs.loc[mask, 'effector_score'].mean() if 'effector_score' in adata_cd8_sub.obs else 0,
            'memory': adata_cd8_sub.obs.loc[mask, 'memory_score'].mean() if 'memory_score' in adata_cd8_sub.obs else 0,
            'tox': adata_cd8_sub[mask, 'TOX'].X.mean() if 'TOX' in adata_cd8_sub.var_names else 0,
            'irf8': adata_cd8_sub[mask, 'IRF8'].X.mean() if 'IRF8' in adata_cd8_sub.var_names else 0,
            'tcf7': adata_cd8_sub[mask, 'TCF7'].X.mean() if 'TCF7' in adata_cd8_sub.var_names else 0,
            'post_pct': (adata_cd8_sub.obs.loc[mask, 'treatment.group'] != 'treatment.naive').mean() * 100
                        if 'treatment.group' in adata_cd8_sub.obs else 0,
        }
        for tf in ['PRDM1', 'ID2', 'IKZF3', 'EOMES', 'LEF1', 'BCL11B']:
            if tf in adata_cd8_sub.var_names:
                cd8_sub_stats[st][tf.lower()] = adata_cd8_sub[mask, tf].X.mean()

    cd8_labels = {}
    for st, stats in sorted(cd8_sub_stats.items()):
        if stats['irf8'] > 0.3:
            label = 'IRF8-high'
        elif stats.get('memory', 0) > 0.3 and stats['tcf7'] > 0.5:
            label = 'Memory (TCF7+)'
        elif stats.get('exhaust', 0) > 0.15 and stats['tox'] > 1.1:
            label = 'Exhausted (TOX-hi)'
        elif stats.get('effector', 0) > 0.6 and stats.get('id2', 0) > 1.8:
            label = 'Cytotoxic (ID2+PRDM1+)'
        elif stats.get('id2', 0) > 1.7 and stats.get('prdm1', 0) < 0.5:
            label = 'Innate-like (ID2+)'
        elif stats.get('effector', 0) > 0.5 and stats.get('prdm1', 0) > 0.8:
            label = 'Effector (PRDM1+)'
        elif stats.get('effector', 0) > 0.3 and stats.get('effector', 0) <= 0.5:
            label = 'Transitional'
        else:
            label = 'Effector'
        cd8_labels[st] = label
        print(f"  CD8 subtype {st}: n={stats['n']}, post%={stats['post_pct']:.0f}, "
              f"IRF8={stats['irf8']:.2f}, TOX={stats['tox']:.2f} -> {label}")

    adata_cd8_sub.obs['subtype_label'] = adata_cd8_sub.obs[subtype_col].map(cd8_labels)
    return cd8_labels, cd8_sub_stats


def assign_mal_labels(adata_mal_sub, subtype_col='tf_subtype_v2'):
    """Compute stats and assign labels to malignant subtypes."""
    mal_sub_stats = {}
    for st in sorted(adata_mal_sub.obs[subtype_col].unique()):
        mask = adata_mal_sub.obs[subtype_col] == st
        mal_sub_stats[st] = {
            'n': mask.sum(),
            'sox10': adata_mal_sub[mask, 'SOX10'].X.mean() if 'SOX10' in adata_mal_sub.var_names else 0,
            'mitf': adata_mal_sub[mask, 'MITF'].X.mean() if 'MITF' in adata_mal_sub.var_names else 0,
            'etv5': adata_mal_sub[mask, 'ETV5'].X.mean() if 'ETV5' in adata_mal_sub.var_names else 0,
            'pax3': adata_mal_sub[mask, 'PAX3'].X.mean() if 'PAX3' in adata_mal_sub.var_names else 0,
            'sox4': adata_mal_sub[mask, 'SOX4'].X.mean() if 'SOX4' in adata_mal_sub.var_names else 0,
            'post_pct': (adata_mal_sub.obs.loc[mask, 'treatment.group'] != 'treatment.naive').mean() * 100,
        }

    mal_labels = {}
    for st, stats in sorted(mal_sub_stats.items()):
        sox10 = stats['sox10']
        mitf = stats['mitf']
        etv5 = stats['etv5']
        post_pct = stats['post_pct']

        if sox10 < 0.3 and mitf < 0.5:
            label = 'Dedifferentiated'
        elif sox10 > 1.0 and mitf > 1.3 and etv5 > 1.0:
            label = 'Melanocytic (high)'
        elif mitf > 1.3 and etv5 < 0.5:
            label = 'MITF-dom. (ETV5-lo)'
        elif sox10 > 0.9 and mitf > 1.2:
            label = 'Melanocytic (moderate)'
        elif sox10 > 0.8 and post_pct > 55:
            label = 'Transitional (post-tx)'
        elif sox10 > 0.7 and stats['n'] > 200:
            label = 'Intermediate'
        else:
            label = 'Transitional'

        mal_labels[st] = label
        print(f"  Mal subtype {st}: n={stats['n']}, post%={post_pct:.0f}, "
              f"SOX10={sox10:.2f}, MITF={mitf:.2f} -> {label}")

    adata_mal_sub.obs['subtype_label'] = adata_mal_sub.obs[subtype_col].map(mal_labels)
    return mal_labels, mal_sub_stats


# ── Helper: Volcano plot ─────────────────────────────────────────────────────
def plot_volcano(ax, deg_df, title, panel_letter, key_genes):
    """Draw a volcano plot on the given axes."""
    deg_df = deg_df.copy()
    # Cap -log10 FDR at 50 to prevent extreme outliers from compressing the view
    deg_df['neg_log10_padj_raw'] = -np.log10(deg_df['padj'].clip(lower=1e-300))
    Y_CAP = 50
    deg_df['neg_log10_padj'] = deg_df['neg_log10_padj_raw'].clip(upper=Y_CAP)
    has_capped = (deg_df['neg_log10_padj_raw'] > Y_CAP).any()

    colors = []
    for _, row in deg_df.iterrows():
        if row['padj'] < 0.05 and row['log2fc'] > 0:
            colors.append('#D73027')
        elif row['padj'] < 0.05 and row['log2fc'] < 0:
            colors.append('#4575B4')
        else:
            colors.append('#CCCCCC')

    ax.scatter(deg_df['log2fc'], deg_df['neg_log10_padj'],
               c=colors, s=3, alpha=0.4, edgecolors='none', rasterized=True)

    # Mark capped points with upward triangles to indicate values beyond y-axis
    capped_mask = deg_df['neg_log10_padj_raw'] > Y_CAP
    if capped_mask.any():
        capped_rows = deg_df[capped_mask]
        cap_colors = ['#D73027' if r['log2fc'] > 0 else '#4575B4' for _, r in capped_rows.iterrows()]
        ax.scatter(capped_rows['log2fc'], capped_rows['neg_log10_padj'],
                   c=cap_colors, s=20, marker='^', edgecolors='black', linewidth=0.3, zorder=5)
        # Auto-label capped points not already in key_genes
        for _, r in capped_rows.iterrows():
            if r['gene'] not in key_genes:
                color = '#D73027' if r['log2fc'] > 0 else '#4575B4'
                ax.annotate(r['gene'], (r['log2fc'], r['neg_log10_padj']),
                            fontsize=5.5, color=color, fontstyle='italic', fontweight='bold',
                            xytext=(0, 6), textcoords='offset points', ha='center')

    # Label key genes using adjustText (ggrepel-equivalent label repulsion)
    try:
        from adjustText import adjust_text
        texts = []
        for gene in key_genes:
            if gene in deg_df['gene'].values:
                row = deg_df[deg_df['gene'] == gene].iloc[0]
                if row['padj'] < 0.05:
                    color = '#D73027' if row['log2fc'] > 0 else '#4575B4'
                    texts.append(ax.text(row['log2fc'], row['neg_log10_padj'],
                                         gene, fontsize=5.5, color=color,
                                         fontstyle='italic', fontweight='bold'))
        if texts:
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle='-', color='grey', lw=0.5, alpha=0.6),
                        force_points=(0.8, 0.8), force_text=(1.0, 1.0),
                        expand_points=(2.0, 2.0), expand_text=(1.5, 1.5),
                        lim=5000)
    except ImportError:
        for gene in key_genes:
            if gene in deg_df['gene'].values:
                row = deg_df[deg_df['gene'] == gene].iloc[0]
                if row['padj'] < 0.05:
                    color = '#D73027' if row['log2fc'] > 0 else '#4575B4'
                    ax.annotate(gene, (row['log2fc'], row['neg_log10_padj']),
                                fontsize=5.5, color=color, fontstyle='italic')

    ax.axhline(-np.log10(0.05), color='grey', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='grey', linestyle='--', linewidth=0.5)
    ax.set_xlabel('log$_2$ fold change (IRF8-high / Other)', fontsize=7)
    ax.set_ylabel('$-$log$_{10}$ FDR', fontsize=7)

    # Set y-axis with headroom for labels
    ax.set_ylim(-2, Y_CAP * 1.18)

    n_up = ((deg_df['padj'] < 0.05) & (deg_df['log2fc'] > 0)).sum()
    n_down = ((deg_df['padj'] < 0.05) & (deg_df['log2fc'] < 0)).sum()
    ax.set_title(f'{title}\n{n_up} up, {n_down} down', fontsize=8)
    ax.text(-0.12, 1.05, panel_letter, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

    legend_els = [Patch(facecolor='#D73027', label=f'Up ({n_up})'),
                  Patch(facecolor='#4575B4', label=f'Down ({n_down})'),
                  Patch(facecolor='#CCCCCC', label='NS')]
    ax.legend(handles=legend_els, fontsize=5.5, frameon=False, loc='upper left')


###############################################################################
#  PART 1: LOAD DISCOVERY COHORT
###############################################################################
print("=" * 70)
print("LOADING DISCOVERY COHORT (GSE115978)")
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

sc.pp.log1p(adata_all)
sc.pp.highly_variable_genes(adata_all, n_top_genes=2000, flavor='seurat_v3')
sc.tl.pca(adata_all, n_comps=30, use_highly_variable=True, random_state=42)
sc.pp.neighbors(adata_all, n_pcs=20, random_state=42)
sc.tl.umap(adata_all, random_state=42)

print(f"  All cells: {adata_all.n_obs}")

# ── Load pre-computed analysis CSVs ──────────────────────────────────────────
tf_global = pd.read_csv(f'{OUT_DIR}/tf_importance_global.csv')
tf_treat_all = pd.read_csv(f'{OUT_DIR}/tf_treatment_changes_all.csv')
cd8_treat = pd.read_csv(f'{OUT_DIR}/cd8_tf_treatment_changes.csv')

celltype_tfs = {}
for cls in ['B_cell', 'CAF', 'Endo_', 'Macrophage', 'Mal', 'NK', 'T_CD4', 'T_CD8', 'T_cell']:
    fpath = f'{OUT_DIR}/tf_importance_{cls}.csv'
    if os.path.exists(fpath):
        celltype_tfs[cls] = pd.read_csv(fpath)

# ── Load and cluster CD8 subtypes (cached for reproducibility) ────────────────
adata_cd8_sub = sc.read_h5ad(f'{OUT_DIR}/adata_cd8_subtypes.h5ad')
tf_cd8 = pd.read_csv(f'{OUT_DIR}/tf_importance_T_CD8.csv')
top_cd8_tfs = tf_cd8.head(15)['TF'].tolist()
available_cd8_tfs = [tf for tf in top_cd8_tfs if tf in adata_cd8_sub.var_names]

# Cache version string: encodes pipeline + labeling rule so cache is invalidated on logic changes
CD8_LABEL_RULE_VERSION = "irf8_gt_0.3_multicluster"
CD8_CACHE_VERSION_BASE = f"tfs={'|'.join(sorted(available_cd8_tfs))};res=0.25;seed=42;n_pcs=10;n_neighbors=15"
CD8_CACHE_VERSION = f"{CD8_CACHE_VERSION_BASE};label_rule={CD8_LABEL_RULE_VERSION}"
_cached_version = adata_cd8_sub.uns.get('cache_version', '') if 'cache_version' in adata_cd8_sub.uns else ''
_cached_version_ok = _cached_version in {CD8_CACHE_VERSION, CD8_CACHE_VERSION_BASE}

if 'tf_subtype_v2' in adata_cd8_sub.obs.columns and 'subtype_label' in adata_cd8_sub.obs.columns \
        and 'X_umap_tf' in adata_cd8_sub.obsm and _cached_version_ok:
    print("\nUsing cached CD8 clustering from h5ad...")
    cd8_labels_v2 = dict(zip(adata_cd8_sub.obs['tf_subtype_v2'], adata_cd8_sub.obs['subtype_label']))
    # Re-compute stats for printing
    cd8_sub_stats = {}
    for st in sorted(adata_cd8_sub.obs['tf_subtype_v2'].unique()):
        mask = adata_cd8_sub.obs['tf_subtype_v2'] == st
        cd8_sub_stats[st] = {
            'n': mask.sum(),
            'irf8': adata_cd8_sub[mask, 'IRF8'].X.mean() if 'IRF8' in adata_cd8_sub.var_names else 0,
            'tox': adata_cd8_sub[mask, 'TOX'].X.mean() if 'TOX' in adata_cd8_sub.var_names else 0,
        }
        label = cd8_labels_v2[st]
        post_pct = (adata_cd8_sub.obs.loc[mask, 'treatment.group'] != 'treatment.naive').mean() * 100
        print(f"  CD8 subtype {st}: n={cd8_sub_stats[st]['n']}, post%={post_pct:.0f}, "
              f"IRF8={cd8_sub_stats[st]['irf8']:.2f}, TOX={cd8_sub_stats[st]['tox']:.2f} -> {label}")
    # Promote legacy cache key to explicit rule-aware cache key
    if _cached_version != CD8_CACHE_VERSION:
        adata_cd8_sub.uns['cache_version'] = CD8_CACHE_VERSION
        adata_cd8_sub.write(f'{OUT_DIR}/adata_cd8_subtypes.h5ad')
else:
    print("\nRe-clustering CD8+ T cells (res=0.25, seed=42)...")
    adata_cd8_tf = adata_cd8_sub[:, available_cd8_tfs].copy()
    sc.pp.scale(adata_cd8_tf, max_value=10)
    sc.tl.pca(adata_cd8_tf, n_comps=10, random_state=42)
    sc.pp.neighbors(adata_cd8_tf, n_pcs=10, n_neighbors=15, random_state=42)
    sc.tl.leiden(adata_cd8_tf, resolution=0.25, key_added='tf_subtype_v2', random_state=42)
    sc.tl.umap(adata_cd8_tf, random_state=42)

    adata_cd8_sub.obs['tf_subtype_v2'] = adata_cd8_tf.obs['tf_subtype_v2'].values
    adata_cd8_sub.obsm['X_umap_tf'] = adata_cd8_tf.obsm['X_umap']

    cd8_labels_v2, cd8_sub_stats = assign_cd8_labels(adata_cd8_sub)
    # Save to h5ad for reproducibility (with version tag)
    adata_cd8_sub.uns['cache_version'] = CD8_CACHE_VERSION
    adata_cd8_sub.write(f'{OUT_DIR}/adata_cd8_subtypes.h5ad')
    print("  CD8 clustering saved to adata_cd8_subtypes.h5ad")

# ── Load and cluster Malignant subtypes (cached for reproducibility) ──────────
adata_mal_sub = sc.read_h5ad(f'{OUT_DIR}/adata_mal_subtypes.h5ad')
tf_mal = pd.read_csv(f'{OUT_DIR}/tf_importance_Mal.csv')
top_mal_tfs = tf_mal.head(15)['TF'].tolist()
available_mal_tfs = [tf for tf in top_mal_tfs if tf in adata_mal_sub.var_names]

MAL_CACHE_VERSION = f"tfs={'|'.join(sorted(available_mal_tfs))};res=0.2;seed=42;n_pcs=10;n_neighbors=15"
_mal_cached_version = adata_mal_sub.uns.get('cache_version', '') if 'cache_version' in adata_mal_sub.uns else ''

if 'tf_subtype_v2' in adata_mal_sub.obs.columns and 'subtype_label' in adata_mal_sub.obs.columns \
        and 'X_umap_tf' in adata_mal_sub.obsm and _mal_cached_version == MAL_CACHE_VERSION:
    print("\nUsing cached Malignant clustering from h5ad...")
    mal_labels_v2 = dict(zip(adata_mal_sub.obs['tf_subtype_v2'], adata_mal_sub.obs['subtype_label']))
    mal_sub_stats = {}
    for st in sorted(adata_mal_sub.obs['tf_subtype_v2'].unique()):
        mask = adata_mal_sub.obs['tf_subtype_v2'] == st
        post_pct = (adata_mal_sub.obs.loc[mask, 'treatment.group'] != 'treatment.naive').mean() * 100
        sox10 = adata_mal_sub[mask, 'SOX10'].X.mean() if 'SOX10' in adata_mal_sub.var_names else 0
        mitf = adata_mal_sub[mask, 'MITF'].X.mean() if 'MITF' in adata_mal_sub.var_names else 0
        mal_sub_stats[st] = {'n': mask.sum(), 'sox10': sox10, 'mitf': mitf}
        label = mal_labels_v2[st]
        print(f"  Mal subtype {st}: n={mask.sum()}, post%={post_pct:.0f}, "
              f"SOX10={sox10:.2f}, MITF={mitf:.2f} -> {label}")
else:
    print("\nRe-clustering malignant cells (res=0.2, seed=42)...")
    adata_mal_tf = adata_mal_sub[:, available_mal_tfs].copy()
    sc.pp.scale(adata_mal_tf, max_value=10)
    sc.tl.pca(adata_mal_tf, n_comps=10, random_state=42)
    sc.pp.neighbors(adata_mal_tf, n_pcs=10, n_neighbors=15, random_state=42)
    sc.tl.leiden(adata_mal_tf, resolution=0.2, key_added='tf_subtype_v2', random_state=42)
    sc.tl.umap(adata_mal_tf, random_state=42)

    adata_mal_sub.obs['tf_subtype_v2'] = adata_mal_tf.obs['tf_subtype_v2'].values
    adata_mal_sub.obsm['X_umap_tf'] = adata_mal_tf.obsm['X_umap']

    mal_labels_v2, mal_sub_stats = assign_mal_labels(adata_mal_sub)
    # Save to h5ad for reproducibility (with version tag)
    adata_mal_sub.uns['cache_version'] = MAL_CACHE_VERSION
    adata_mal_sub.write(f'{OUT_DIR}/adata_mal_subtypes.h5ad')
    print("  Malignant clustering saved to adata_mal_subtypes.h5ad")

# ── RF 5-fold CV (cached to avoid stochastic variation between runs) ──────────
RF_CACHE = f'{OUT_DIR}/rf_cv_f1_scores.csv'
if os.path.exists(RF_CACHE):
    print("\nLoading cached RF CV results...")
    _rf_cache = pd.read_csv(RF_CACHE)
    cv_report = {}
    for _, row in _rf_cache.iterrows():
        cv_report[row['class']] = {'f1-score': row['f1_score']}
    cv_weighted_f1 = _rf_cache[_rf_cache['class'] == '_weighted']['f1_score'].values[0]
    le = LabelEncoder()
    y_rf = le.fit_transform(adata_all.obs['cell.types'].values)
    print(f"  CV weighted F1: {cv_weighted_f1:.3f} (cached)")
else:
    print("\nRunning RF 5-fold CV for per-class F1 scores...")
    all_tfs = list(set(tf for genes in tf_families.values() for tf in genes))
    tfs_in_data = [tf for tf in all_tfs if tf in adata_all.var_names]
    tf_expr = pd.DataFrame(adata_all[:, tfs_in_data].X, index=adata_all.obs_names, columns=tfs_in_data)
    tf_pct = (tf_expr > 0).mean()
    tfs_expressed = tf_pct[tf_pct >= 0.03].index.tolist()
    tf_expr_filtered = tf_expr[tfs_expressed]

    X_rf = tf_expr_filtered.values
    le = LabelEncoder()
    y_rf = le.fit_transform(adata_all.obs['cell.types'].values)

    rf = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_leaf=3,
                                random_state=42, n_jobs=1, class_weight='balanced')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_cv_pred = cross_val_predict(rf, X_rf, y_rf, cv=cv)

    cv_report = classification_report(y_rf, y_cv_pred, target_names=le.classes_, output_dict=True)
    cv_weighted_f1 = f1_score(y_rf, y_cv_pred, average='weighted')
    print(f"  CV weighted F1: {cv_weighted_f1:.3f}")

    # Save to cache CSV
    rows = [{'class': cls, 'f1_score': cv_report[cls]['f1-score']} for cls in le.classes_]
    rows.append({'class': '_weighted', 'f1_score': cv_weighted_f1})
    pd.DataFrame(rows).to_csv(RF_CACHE, index=False)
    print(f"  Saved RF CV cache to {RF_CACHE}")

# ── Load pre-computed DEG / pathway CSVs ─────────────────────────────────────
deg_disc = pd.read_csv(f'{OUT_DIR}/irf8high_deg_discovery.csv')
deg_val = pd.read_csv(f'{OUT_DIR}/irf8high_deg_validation.csv')
enr_disc = pd.read_csv(f'{OUT_DIR}/irf8high_pathway_enrichment_discovery.csv')
enr_val = pd.read_csv(f'{OUT_DIR}/irf8high_pathway_enrichment_validation.csv')

# Key genes for volcano labels (curated to minimize crowding while labeling key biology)
key_genes_volcano = [
    'IRF8', 'PDCD1', 'TIGIT', 'HAVCR2', 'TOX', 'PRF1', 'IFNG',
    'LAG3', 'IL7R', 'MKI67', 'GZMB',
]

# ── Load pseudotime and correlation CSVs ─────────────────────────────────────
# NOTE: dpt_subtype_pseudotime_stats.csv = 5-subtype (legacy, from 08_diffusion_pseudotime_irf8.py)
#       dpt_subtype_pseudotime_stats_6st.csv = 6-subtype (current, generated below at Fig 2C)
# The 5-subtype file is loaded here only for backward-compat; Figure 2C uses 6-subtype stats.
print("\nLoading pseudotime and correlation data...")
dpt_cell_data = pd.read_csv(f'{OUT_DIR}/dpt_cd8_cell_data.csv')
dpt_subtype_stats = pd.read_csv(f'{OUT_DIR}/dpt_subtype_pseudotime_stats.csv')
dpt_irf8_corr = pd.read_csv(f'{OUT_DIR}/dpt_irf8_gene_correlations.csv')
dpt_pt_gene_corr = pd.read_csv(f'{OUT_DIR}/dpt_pseudotime_gene_correlations.csv')
irf8_target_enr = pd.read_csv(f'{OUT_DIR}/irf8_target_enrichment_results.csv')

print(f"  Pseudotime cell data: {len(dpt_cell_data)} cells")
print(f"  IRF8 gene correlations: {len(dpt_irf8_corr)} genes")
print(f"  Pseudotime gene correlations: {len(dpt_pt_gene_corr)} genes")
print(f"  IRF8 target enrichment: {len(irf8_target_enr)} gene sets")


###############################################################################
#  FIGURE 1: RF-SHAP Method + TF Ranking  (2x2)
###############################################################################
print("\n" + "=" * 70)
print("GENERATING FIGURE 1: RF-SHAP Method + TF Ranking")
print("=" * 70)

fig = plt.figure(figsize=(16, 3.5))
gs = gridspec.GridSpec(1, 4, wspace=0.45, width_ratios=[1, 1, 1, 1.2])

# ── Panel A: UMAP of all cells ──────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
plot_order = ['Mal', 'T.CD8', 'T.CD4', 'B.cell', 'T.cell', 'Macrophage', 'CAF', 'Endo.', 'NK']
for ct in plot_order:
    mask = adata_all.obs['cell.types'] == ct
    if mask.sum() > 0:
        ax_a.scatter(
            adata_all.obsm['X_umap'][mask, 0], adata_all.obsm['X_umap'][mask, 1],
            s=1, alpha=0.5, c=celltype_colors[ct],
            label=celltype_labels[ct], rasterized=True
        )
ax_a.set_xlabel('UMAP1')
ax_a.set_ylabel('UMAP2')
ax_a.set_title(f'Melanoma TME\n({adata_all.n_obs:,} cells, 32 patients)', fontsize=8)
ax_a.legend(markerscale=5, frameon=False, fontsize=6, loc='best', handletextpad=0.2)
ax_a.set_xticks([])
ax_a.set_yticks([])
ax_a.text(-0.05, 1.05, 'A', transform=ax_a.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel B: Per-class F1 bar chart ──────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])

cv_f1_per_class = {cls: cv_report[cls]['f1-score'] for cls in le.classes_}
celltypes_sorted = sorted(cv_f1_per_class.keys(), key=lambda x: cv_f1_per_class[x], reverse=True)
colors_bar = [celltype_colors.get(ct, '#999999') for ct in celltypes_sorted]
labels_fig = [celltype_labels.get(ct, ct) for ct in celltypes_sorted]
values = [cv_f1_per_class[ct] for ct in celltypes_sorted]

bars = ax_b.barh(range(len(celltypes_sorted)), values, color=colors_bar,
                  edgecolor='white', linewidth=0.3)
ax_b.set_yticks(range(len(celltypes_sorted)))
ax_b.set_yticklabels(labels_fig, fontsize=7)
ax_b.set_xlabel('F1 score (5-fold CV)')
ax_b.set_title(f'RF classification performance\n(weighted F1 = {cv_weighted_f1:.3f})', fontsize=8)
ax_b.set_xlim(0, 1.08)
ax_b.axvline(x=0.8, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)
for i, v in enumerate(values):
    ax_b.text(v + 0.015, i, f'{v:.3f}', va='center', fontsize=6)
ax_b.invert_yaxis()
ax_b.text(-0.12, 1.05, 'B', transform=ax_b.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel C: Top 20 globally important TFs ───────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
top20 = tf_global.head(20).sort_values('mean_abs_shap', ascending=True)
bar_colors = []
for tf in top20['TF'].values:
    if tf in ['ETV5', 'WWTR1', 'ZEB2', 'MEF2C']:
        bar_colors.append('#E64B35')   # novel / cross-lineage
    else:
        bar_colors.append('#4DBBD5')   # known

ax_c.barh(range(len(top20)), top20['mean_abs_shap'], color=bar_colors,
           edgecolor='white', linewidth=0.3)
ax_c.set_yticks(range(len(top20)))
ax_c.set_yticklabels(top20['TF'].values, fontsize=7, fontstyle='italic')
ax_c.set_xlabel('Mean |SHAP| value')
ax_c.set_title('Top 20 identity-determining TFs', fontsize=8)
legend_elements = [Patch(facecolor='#4DBBD5', label='Known'),
                   Patch(facecolor='#E64B35', label='Cross-lineage')]
ax_c.legend(handles=legend_elements, fontsize=6, frameon=False, loc='lower right')
ax_c.text(-0.18, 1.05, 'C', transform=ax_c.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel D: Cell-type-specific SHAP heatmap ─────────────────────────────────
ax_d = fig.add_subplot(gs[0, 3])
key_celltypes = ['Mal', 'T_CD8', 'T_CD4', 'B_cell', 'Macrophage', 'NK', 'Endo_', 'CAF']
key_ct_labels = ['Malignant', 'CD8+ T', 'CD4+ T', 'B cell', 'Macrophage', 'NK', 'Endothelial', 'CAF']
top_tfs_list = tf_global.head(20)['TF'].tolist()

heatmap_data = pd.DataFrame(index=top_tfs_list, columns=key_ct_labels)
for ct, ct_label in zip(key_celltypes, key_ct_labels):
    if ct in celltype_tfs:
        df = celltype_tfs[ct].set_index('TF')
        for tf in top_tfs_list:
            if tf in df.index:
                heatmap_data.loc[tf, ct_label] = df.loc[tf, 'mean_shap']
            else:
                heatmap_data.loc[tf, ct_label] = 0

heatmap_data = heatmap_data.astype(float)
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
# Cluster rows (TFs) and columns (cell types)
row_linkage = linkage(pdist(heatmap_data.values, metric='euclidean'), method='ward')
col_linkage = linkage(pdist(heatmap_data.values.T, metric='euclidean'), method='ward')
row_order = dendrogram(row_linkage, no_plot=True)['leaves']
col_order = dendrogram(col_linkage, no_plot=True)['leaves']
heatmap_data = heatmap_data.iloc[row_order, col_order]

sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, ax=ax_d,
            vmin=-0.02, vmax=0.02,
            yticklabels=True, xticklabels=True,
            cbar_kws={'shrink': 0.6, 'label': 'Mean SHAP'},
            linewidths=0.5, linecolor='white')
ax_d.set_title('Cell-type-specific TF importance', fontsize=8)
ax_d.tick_params(axis='y', labelsize=7)
ax_d.tick_params(axis='x', labelsize=7, rotation=45)
ax_d.set_yticklabels(ax_d.get_yticklabels(), fontstyle='italic')
ax_d.text(-0.12, 1.05, 'D', transform=ax_d.transAxes, fontsize=12, fontweight='bold', va='top')

plt.tight_layout()
fig.savefig(f'{FIG_DIR}/Figure1.tiff', dpi=300, format='tiff')
fig.savefig(f'{FIG_DIR}/Figure1.pdf', dpi=300, format='pdf')
plt.close()
print("  Figure 1 saved.")


###############################################################################
#  FIGURE 2: CD8+ Subtype Discovery + Pseudotime  (2x2)
###############################################################################
print("\n" + "=" * 70)
print("GENERATING FIGURE 2: CD8+ Subtype Discovery + Pseudotime")
print("=" * 70)

fig = plt.figure(figsize=(16, 3.5))
gs2 = gridspec.GridSpec(1, 4, wspace=0.50, width_ratios=[1, 1.2, 1, 1])

# ── Panel A: CD8 UMAP by subtype ────────────────────────────────────────────
ax_a = fig.add_subplot(gs2[0, 0])
subtypes_cd8 = sorted(adata_cd8_sub.obs['tf_subtype_v2'].unique())
# Use consistent merged_color_map by label
label_order_cd8 = ['IRF8-high', 'Exhausted (TOX-hi)', 'Memory (TCF7+)',
                   'Cytotoxic (ID2+PRDM1+)', 'Innate-like (ID2+)',
                   'Effector (PRDM1+)', 'Effector', 'Transitional']
label_order_cd8 = [l for l in label_order_cd8 if l in set(adata_cd8_sub.obs['subtype_label'])]

for lbl in label_order_cd8:
    mask = (adata_cd8_sub.obs['subtype_label'] == lbl).values
    n = mask.sum()
    color = merged_color_map.get(lbl, '#999999')
    ax_a.scatter(
        adata_cd8_sub.obsm['X_umap_tf'][mask, 0], adata_cd8_sub.obsm['X_umap_tf'][mask, 1],
        s=2, alpha=0.6, c=color, label=f'{lbl} ({n})',
        rasterized=True
    )
ax_a.set_title('CD8+ T cell TF-based subtypes', fontsize=8)
ax_a.set_xlabel('UMAP1', fontsize=7)
ax_a.set_ylabel('UMAP2', fontsize=7)
ax_a.set_xticks([])
ax_a.set_yticks([])
ax_a.legend(fontsize=4.5, markerscale=3, frameon=False,
            bbox_to_anchor=(0.0, -0.08), loc='upper left', ncol=2,
            handletextpad=0.1, columnspacing=0.3)
ax_a.text(-0.05, 1.05, 'A', transform=ax_a.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel B: CD8 subtype TF heatmap (z-scored) ──────────────────────────────
ax_b = fig.add_subplot(gs2[0, 1])


score_data = []
for lbl in label_order_cd8:
    mask = (adata_cd8_sub.obs['subtype_label'] == lbl).values
    n = mask.sum()
    post_pct = (adata_cd8_sub.obs.loc[mask, 'treatment.group'] != 'treatment.naive').mean() * 100
    row = {
        'Subtype': lbl,
        'TOX': adata_cd8_sub[mask, 'TOX'].X.mean() if 'TOX' in adata_cd8_sub.var_names else 0,
        'IRF8': adata_cd8_sub[mask, 'IRF8'].X.mean() if 'IRF8' in adata_cd8_sub.var_names else 0,
        'TCF7': adata_cd8_sub[mask, 'TCF7'].X.mean() if 'TCF7' in adata_cd8_sub.var_names else 0,
        'IKZF3': adata_cd8_sub[mask, 'IKZF3'].X.mean() if 'IKZF3' in adata_cd8_sub.var_names else 0,
        'PRDM1': adata_cd8_sub[mask, 'PRDM1'].X.mean() if 'PRDM1' in adata_cd8_sub.var_names else 0,
        'ID2': adata_cd8_sub[mask, 'ID2'].X.mean() if 'ID2' in adata_cd8_sub.var_names else 0,
        'Post-tx %': post_pct,
    }
    score_data.append(row)

score_df = pd.DataFrame(score_data).set_index('Subtype')
tf_cols = ['TOX', 'IRF8', 'TCF7', 'IKZF3', 'PRDM1', 'ID2', 'Post-tx %']

score_z = score_df[tf_cols].copy()
for col in tf_cols:
    vals = score_z[col].values.astype(float)
    if vals.std() > 0:
        score_z[col] = (vals - vals.mean()) / vals.std()

score_z = score_z.astype(float)

# Cluster TF rows (exclude Post-tx %) using Ward's method
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
tf_only_cols = [c for c in tf_cols if c != 'Post-tx %']
tf_z_for_clust = score_z[tf_only_cols].T  # rows=TFs, cols=subtypes
row_linkage_b = linkage(pdist(tf_z_for_clust.values, metric='euclidean'), method='ward')
row_order_b = dendrogram(row_linkage_b, no_plot=True)['leaves']
tf_ordered = [tf_only_cols[i] for i in row_order_b]
# Final column order: clustered TFs + Post-tx % at end
tf_cols_ordered = tf_ordered + ['Post-tx %']
score_z_ordered = score_z[tf_cols_ordered]

# Annotation: z-scores for TF rows, raw % for Post-tx %
annot_vals = score_z_ordered.astype(float).T.copy()
annot_vals.loc['Post-tx %'] = score_df['Post-tx %'].values  # overwrite with raw %
annot_vals = annot_vals.round(1)
sns.heatmap(score_z_ordered.T, cmap='RdBu_r', center=0, ax=ax_b,
            yticklabels=True, xticklabels=True,
            cbar_kws={'shrink': 0.5, 'label': 'Z-score'},
            linewidths=0.5, linecolor='white', annot=annot_vals,
            fmt='.1f', annot_kws={'fontsize': 5.5})
ax_b.set_title('CD8+ subtype characterization', fontsize=8)
ax_b.tick_params(axis='y', labelsize=7)
ax_b.tick_params(axis='x', labelsize=6, rotation=45)
# Make TF names italic, but not Post-tx %
ylabels = ax_b.get_yticklabels()
for yl in ylabels:
    if yl.get_text() != 'Post-tx %':
        yl.set_fontstyle('italic')
ax_b.set_yticklabels(ylabels)
ax_b.text(-0.12, 1.05, 'B', transform=ax_b.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel C: Pseudotime boxplot by 6-subtype ─────────────────────────────────
ax_c = fig.add_subplot(gs2[0, 2])

# Map 6-subtype labels to pseudotime cells
dpt_cell_data_6st = dpt_cell_data.copy()
cell_to_label6 = dict(zip(adata_cd8_sub.obs.index, adata_cd8_sub.obs['subtype_label']))
dpt_cell_data_6st['subtype_label_6'] = dpt_cell_data_6st['cell'].map(cell_to_label6)

dpt_plot = dpt_cell_data_6st.dropna(subset=['subtype_label_6']).copy()
dpt_plot['subtype_label_6'] = pd.Categorical(dpt_plot['subtype_label_6'],
                                              categories=label_order_cd8, ordered=True)

box_palette_c = [merged_color_map.get(s, '#999999') for s in label_order_cd8]
sns.boxplot(data=dpt_plot, x='subtype_label_6', y='dpt_pseudotime',
            palette=box_palette_c, ax=ax_c, linewidth=0.5, fliersize=1,
            width=0.6, showfliers=False)
sns.stripplot(data=dpt_plot, x='subtype_label_6', y='dpt_pseudotime',
              palette=box_palette_c, ax=ax_c, size=0.8, alpha=0.3,
              jitter=True, rasterized=True)

# Kruskal-Wallis test
groups_c = [dpt_plot[dpt_plot['subtype_label_6'] == s]['dpt_pseudotime'].values
            for s in label_order_cd8 if s in dpt_plot['subtype_label_6'].values]
if len(groups_c) >= 2:
    kw_stat, kw_p = kruskal(*groups_c)
    ax_c.text(0.5, 0.97, f'Kruskal-Wallis p = {kw_p:.2e}',
              transform=ax_c.transAxes, ha='center', va='top',
              fontsize=6.5, fontstyle='italic')

# Save 6-subtype pseudotime statistics CSV
pt_stats_rows = []
for s in label_order_cd8:
    vals = dpt_plot[dpt_plot['subtype_label_6'] == s]['dpt_pseudotime']
    if len(vals) > 0:
        pt_stats_rows.append({
            'subtype': s, 'n_cells': len(vals),
            'mean_pt': vals.mean(), 'median_pt': vals.median(),
            'sd_pt': vals.std(), 'min_pt': vals.min(), 'max_pt': vals.max()
        })
pd.DataFrame(pt_stats_rows).to_csv(f'{OUT_DIR}/dpt_subtype_pseudotime_stats_6st.csv', index=False)

# IRF8-high vs others Mann-Whitney pairwise tests
irf8_vals = dpt_plot[dpt_plot['subtype_label_6'] == 'IRF8-high']['dpt_pseudotime'].values
pt_test_rows = []
for s in label_order_cd8:
    if s == 'IRF8-high':
        continue
    other_vals = dpt_plot[dpt_plot['subtype_label_6'] == s]['dpt_pseudotime'].values
    if len(other_vals) > 0 and len(irf8_vals) > 0:
        u_stat, u_p = mannwhitneyu(irf8_vals, other_vals, alternative='two-sided')
        rb = 1 - 2 * u_stat / (len(irf8_vals) * len(other_vals))  # rank-biserial
        direction = 'IRF8-high later' if irf8_vals.mean() > other_vals.mean() else 'IRF8-high earlier'
        pt_test_rows.append({
            'comparison': f'IRF8-high vs {s}',
            'IRF8_high_mean_pt': irf8_vals.mean(), 'other_mean_pt': other_vals.mean(),
            'U_statistic': u_stat, 'p_value': u_p, 'rank_biserial': rb, 'direction': direction
        })
# Apply Benjamini-Hochberg correction to pairwise tests
df_pt_tests = pd.DataFrame(pt_test_rows)
if len(df_pt_tests) > 0:
    _, padj, _, _ = multipletests(df_pt_tests["p_value"], method="fdr_bh")
    df_pt_tests["padj"] = padj
df_pt_tests.to_csv(f'{OUT_DIR}/dpt_irf8_vs_subtypes_tests_6st.csv', index=False)
print(f"  6-subtype pseudotime stats saved (KW p={kw_p:.2e})")

ax_c.set_xlabel('')
ax_c.set_ylabel('Pseudotime (DPT)', fontsize=7)
ax_c.set_title('Pseudotime distribution by subtype', fontsize=8)
pt_short_labels = [s.replace(' (', '\n(').replace('Exhausted\n(TOX-hi)', 'Exhausted\n(TOX-hi)')
                   for s in label_order_cd8]
ax_c.set_xticklabels(pt_short_labels, fontsize=5, rotation=45, ha='right')
ax_c.text(-0.12, 1.05, 'C', transform=ax_c.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel D: IRF8 co-expression bar plot (moved from Figure 3D) ──────────────
ax_d = fig.add_subplot(gs2[0, 3])

irf8_corr_sorted = dpt_irf8_corr.sort_values('spearman_rho', ascending=True)

bar_colors_corr_2d = []
for _, row in irf8_corr_sorted.iterrows():
    if row['p_value'] < 0.001:
        bar_colors_corr_2d.append('#D73027' if row['spearman_rho'] > 0 else '#4575B4')
    elif row['p_value'] < 0.05:
        bar_colors_corr_2d.append('#FF7F7F' if row['spearman_rho'] > 0 else '#89CFF0')
    else:
        bar_colors_corr_2d.append('#999999')

ax_d.barh(range(len(irf8_corr_sorted)), irf8_corr_sorted['spearman_rho'].values,
          color=bar_colors_corr_2d, edgecolor='white', linewidth=0.3, height=0.7)
ax_d.set_yticks(range(len(irf8_corr_sorted)))
ax_d.set_yticklabels(irf8_corr_sorted['gene'].values, fontsize=7, fontstyle='italic')
ax_d.set_xlabel('Spearman ρ with IRF8', fontsize=7)
ax_d.set_title('IRF8 co-expression with\nexhaustion/effector genes', fontsize=8)
ax_d.axvline(0, color='grey', linestyle='--', linewidth=0.5)

# Build legend: only show categories present in data
_has_pos_001 = any(r['spearman_rho'] > 0 and r['p_value'] < 0.001 for _, r in irf8_corr_sorted.iterrows())
_has_pos_05 = any(r['spearman_rho'] > 0 and 0.001 <= r['p_value'] < 0.05 for _, r in irf8_corr_sorted.iterrows())
_has_neg_001 = any(r['spearman_rho'] < 0 and r['p_value'] < 0.001 for _, r in irf8_corr_sorted.iterrows())
_has_neg_05 = any(r['spearman_rho'] < 0 and 0.001 <= r['p_value'] < 0.05 for _, r in irf8_corr_sorted.iterrows())
_has_ns = any(r['p_value'] >= 0.05 for _, r in irf8_corr_sorted.iterrows())
legend_elements_corr_2d = []
if _has_pos_001: legend_elements_corr_2d.append(Patch(facecolor='#D73027', label='Positive (p < 0.001)'))
if _has_pos_05:  legend_elements_corr_2d.append(Patch(facecolor='#FF7F7F', label='Positive (p < 0.05)'))
if _has_neg_001: legend_elements_corr_2d.append(Patch(facecolor='#4575B4', label='Negative (p < 0.001)'))
if _has_neg_05:  legend_elements_corr_2d.append(Patch(facecolor='#89CFF0', label='Negative (p < 0.05)'))
if _has_ns:      legend_elements_corr_2d.append(Patch(facecolor='#999999', label='NS'))
ax_d.legend(handles=legend_elements_corr_2d, fontsize=5.5, frameon=False, loc='lower right')
ax_d.text(-0.12, 1.05, 'D', transform=ax_d.transAxes, fontsize=12, fontweight='bold', va='top')

plt.tight_layout()
fig.savefig(f'{FIG_DIR}/Figure2.tiff', dpi=300, format='tiff')
fig.savefig(f'{FIG_DIR}/Figure2.pdf', dpi=300, format='pdf')
plt.close()
print("  Figure 2 saved.")


###############################################################################
#  PART 2: LOAD VALIDATION COHORT
###############################################################################
print("\n" + "=" * 70)
print("LOADING VALIDATION COHORT (GSE120575)")
print("=" * 70)

tpm_file = f'{DATA_DIR}/GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz'

print("Reading header rows...")
with gzip.open(tpm_file, 'rt') as f:
    line1 = f.readline().strip().split('\t')
    line2 = f.readline().strip().split('\t')

cell_ids_val = line1
patient_treat_info = line2

print(f"  Cell IDs: {len(cell_ids_val)}")

print("Reading TPM expression matrix...")
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
print(f"  Expression matrix: {tpm_val.shape[0]} genes x {tpm_val.shape[1]} cells")

# Read metadata
print("Reading metadata...")
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
    treatment = 'Pre' if str(pt_info).startswith('Pre') else (
        'Post' if str(pt_info).startswith('Post') else 'Unknown')
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

# Build AnnData
tpm_val_T = tpm_val.T
common_cells_val = tpm_val_T.index.intersection(annot_val.index)
adata_val = sc.AnnData(X=tpm_val_T.loc[common_cells_val].values.astype(np.float32))
adata_val.obs_names = common_cells_val.tolist()
adata_val.var_names = tpm_val.index.tolist()
for col in annot_val.columns:
    adata_val.obs[col] = annot_val.loc[common_cells_val, col].values

sc.pp.log1p(adata_val)

# Select CD8+ T cells
cd8a_expr = adata_val[:, 'CD8A'].X.flatten() if 'CD8A' in adata_val.var_names else np.zeros(adata_val.n_obs)
cd4_expr = adata_val[:, 'CD4'].X.flatten() if 'CD4' in adata_val.var_names else np.zeros(adata_val.n_obs)
cd3d_expr = adata_val[:, 'CD3D'].X.flatten() if 'CD3D' in adata_val.var_names else np.zeros(adata_val.n_obs)

cd8_mask_val = (cd8a_expr > 1.0) & (cd3d_expr > 0) & (cd8a_expr > cd4_expr)
adata_cd8_val = adata_val[cd8_mask_val].copy()
print(f"  CD8+ T cells: {adata_cd8_val.n_obs}")

# TF-based subclustering (cached for reproducibility)
VAL_CACHE = f'{OUT_DIR}/adata_cd8_val_cached.h5ad'
VAL_LABEL_RULE_VERSION = "irf8_gt_0.3_multicluster"
tf_cd8_discovery = pd.read_csv(f'{OUT_DIR}/tf_importance_T_CD8.csv')
top_cd8_tfs_val = tf_cd8_discovery.head(15)['TF'].tolist()
available_tfs_val = [tf for tf in top_cd8_tfs_val if tf in adata_cd8_val.var_names]

if os.path.exists(VAL_CACHE):
    print("  Using cached validation CD8 clustering...")
    _val_cache = sc.read_h5ad(VAL_CACHE)
    _cached_val_label_rule = _val_cache.uns.get('label_rule_version', VAL_LABEL_RULE_VERSION)
    # Align cached labels to current cells
    common_cached = adata_cd8_val.obs_names.intersection(_val_cache.obs_names)
    if len(common_cached) == adata_cd8_val.n_obs and _cached_val_label_rule == VAL_LABEL_RULE_VERSION:
        adata_cd8_val.obs['tf_subtype'] = _val_cache.obs.loc[adata_cd8_val.obs_names, 'tf_subtype'].values
        adata_cd8_val.obs['subtype_label'] = _val_cache.obs.loc[adata_cd8_val.obs_names, 'subtype_label'].values
        adata_cd8_val.obs['response_binary'] = _val_cache.obs.loc[adata_cd8_val.obs_names, 'response_binary'].values
        adata_cd8_val.obsm['X_umap_tf'] = _val_cache[adata_cd8_val.obs_names].obsm['X_umap_tf']
        val_labels = dict(zip(_val_cache.obs['tf_subtype'], _val_cache.obs['subtype_label']))
        for st in sorted(adata_cd8_val.obs['tf_subtype'].unique()):
            mask = adata_cd8_val.obs['tf_subtype'] == st
            irf8_mean = adata_cd8_val[mask, 'IRF8'].X.mean() if 'IRF8' in adata_cd8_val.var_names else 0
            print(f"  Val CD8 subtype {st}: n={mask.sum()}, IRF8={irf8_mean:.2f} -> {val_labels[st]}")
    else:
        print(f"  Cache mismatch (cells/rule). Re-clustering validation CD8...")
        os.remove(VAL_CACHE)
        # Fall through to re-clustering below

if 'tf_subtype' not in adata_cd8_val.obs.columns:
    adata_cd8_val_tf = adata_cd8_val[:, available_tfs_val].copy()
    sc.pp.scale(adata_cd8_val_tf, max_value=10)
    sc.tl.pca(adata_cd8_val_tf, n_comps=min(10, len(available_tfs_val) - 1), random_state=42)
    sc.pp.neighbors(adata_cd8_val_tf, n_pcs=min(10, len(available_tfs_val) - 1),
                    n_neighbors=15, random_state=42)
    sc.tl.leiden(adata_cd8_val_tf, resolution=0.25, key_added='tf_subtype', random_state=42)
    sc.tl.umap(adata_cd8_val_tf, random_state=42)

    adata_cd8_val.obs['tf_subtype'] = adata_cd8_val_tf.obs['tf_subtype'].values
    adata_cd8_val.obsm['X_umap_tf'] = adata_cd8_val_tf.obsm['X_umap']

    # Assign validation labels
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
            'post_pct': (adata_cd8_val.obs.loc[mask, 'treatment'] == 'Post').mean() * 100,
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
        print(f"  Val CD8 subtype {st}: n={stats['n']}, IRF8={stats['irf8']:.2f} -> {val_labels[st]}")

    adata_cd8_val.obs['subtype_label'] = adata_cd8_val.obs['tf_subtype'].map(val_labels)

    # Map response to binary
    resp = adata_cd8_val.obs['response'].astype(str).str.strip()
    resp_map = {}
    for v in resp.unique():
        vl = v.lower()
        if 'respond' in vl and 'non' not in vl:
            resp_map[v] = 'Responder'
        elif 'non' in vl or 'no response' in vl or 'progressive' in vl:
            resp_map[v] = 'Non-responder'
        elif 'complete' in vl or 'partial' in vl:
            resp_map[v] = 'Responder'
        elif 'stable' in vl:
            resp_map[v] = 'Non-responder'
        else:
            resp_map[v] = v
    adata_cd8_val.obs['response_binary'] = resp.map(resp_map)

    # Save cache for reproducibility
    adata_cd8_val.uns['label_rule_version'] = VAL_LABEL_RULE_VERSION
    adata_cd8_val.write(VAL_CACHE)
    print("  Validation CD8 clustering saved to adata_cd8_val_cached.h5ad")

print(f"\n  Response distribution:")
print(adata_cd8_val.obs['response_binary'].value_counts())


###############################################################################
#  FIGURE 3: IRF8-high Characterization  (2x2)
###############################################################################
print("\n" + "=" * 70)
print("GENERATING FIGURE 3: IRF8-high Characterization")
print("=" * 70)

fig = plt.figure(figsize=(12, 3.5))
gs3 = gridspec.GridSpec(1, 3, wspace=0.55, width_ratios=[1, 1, 1.5])

# ── Panel A: Volcano - Discovery IRF8-high DEGs ─────────────────────────────
ax_a = fig.add_subplot(gs3[0, 0])
plot_volcano(ax_a, deg_disc, 'Discovery (GSE115978)', 'A', key_genes_volcano)

# ── Panel B: Volcano - Validation IRF8-high DEGs ────────────────────────────
ax_b = fig.add_subplot(gs3[0, 1])

plot_volcano(ax_b, deg_val, 'Validation (GSE120575)', 'B', key_genes_volcano)

# ── Panel C: Pathway enrichment comparison ───────────────────────────────────
ax_c = fig.add_subplot(gs3[0, 2])

pathways_of_interest = [
    'T_cell_exhaustion',
    'Cell_cycle_proliferation',
    'T_cell_costimulation',
    'Ribosome',
    'Oxidative_phosphorylation',
    'Apoptosis',
    'Antigen_processing_presentation',
    'Type_II_IFN_signaling',
]

pathway_data = []
for pw in pathways_of_interest:
    for enr_df_src, label in [(enr_disc, 'Discovery'), (enr_val, 'Validation')]:
        hits = enr_df_src[enr_df_src['gene_set'] == pw]
        if len(hits) > 0:
            best = hits.sort_values('pvalue').iloc[0]
            neg_log10 = -np.log10(max(best['padj'], 1e-80))
            direction = best['direction']
        else:
            neg_log10 = 0
            direction = 'UP'
        pathway_data.append({
            'pathway': pw.replace('_', ' '),
            'cohort': label,
            'neg_log10_fdr': neg_log10,
            'direction': direction,
        })

pw_df = pd.DataFrame(pathway_data)
pathways_labels = [pw.replace('_', ' ') for pw in pathways_of_interest]
y_pos = np.arange(len(pathways_labels))
bar_width = 0.35

disc_vals = []
val_vals = []
disc_dirs = []
val_dirs = []
for pw_label in pathways_labels:
    d_row = pw_df[(pw_df['pathway'] == pw_label) & (pw_df['cohort'] == 'Discovery')]
    v_row = pw_df[(pw_df['pathway'] == pw_label) & (pw_df['cohort'] == 'Validation')]
    disc_vals.append(d_row['neg_log10_fdr'].values[0] if len(d_row) > 0 else 0)
    val_vals.append(v_row['neg_log10_fdr'].values[0] if len(v_row) > 0 else 0)
    disc_dirs.append(d_row['direction'].values[0] if len(d_row) > 0 else 'UP')
    val_dirs.append(v_row['direction'].values[0] if len(v_row) > 0 else 'UP')

disc_colors = ['#D73027' if d == 'UP' else '#4575B4' for d in disc_dirs]
val_colors = ['#FF7F7F' if d == 'UP' else '#89CFF0' for d in val_dirs]

# Cap extreme values to improve readability (Ribosome can be >70)
PATHWAY_CAP = 20
disc_vals_capped = [min(v, PATHWAY_CAP) for v in disc_vals]
val_vals_capped = [min(v, PATHWAY_CAP) for v in val_vals]

ax_c.barh(y_pos + bar_width / 2, disc_vals_capped, bar_width, color=disc_colors,
          edgecolor='white', linewidth=0.3)
ax_c.barh(y_pos - bar_width / 2, val_vals_capped, bar_width, color=val_colors,
          edgecolor='white', linewidth=0.3)

# Mark capped bars with break indicator
for i, (dv, dvr) in enumerate(zip(disc_vals_capped, disc_vals)):
    if dvr > PATHWAY_CAP:
        ax_c.text(PATHWAY_CAP + 0.2, y_pos[i] + bar_width / 2, f'({dvr:.0f})',
                  fontsize=5, va='center', ha='left', color=disc_colors[i])
for i, (vv, vvr) in enumerate(zip(val_vals_capped, val_vals)):
    if vvr > PATHWAY_CAP:
        ax_c.text(PATHWAY_CAP + 0.2, y_pos[i] - bar_width / 2, f'({vvr:.0f})',
                  fontsize=5, va='center', ha='left', color=val_colors[i])

ax_c.axvline(-np.log10(0.05), color='grey', linestyle='--', linewidth=0.5)
ax_c.set_yticks(y_pos)
pathways_short = [p.replace('Antigen processing presentation', 'Antigen proc./pres.')
                       .replace('Oxidative phosphorylation', 'Oxid. phosphorylation')
                   for p in pathways_labels]
ax_c.set_yticklabels(pathways_short, fontsize=6)
ax_c.set_xlabel('$-$log$_{10}$ FDR', fontsize=7)
ax_c.set_title('Pathway enrichment in IRF8-high DEGs', fontsize=8)
ax_c.set_xlim(-0.5, PATHWAY_CAP * 1.35)
ax_c.invert_yaxis()

legend_elements = [
    Patch(facecolor='#D73027', label='Disc. UP'),
    Patch(facecolor='#4575B4', label='Disc. DOWN'),
    Patch(facecolor='#FF7F7F', label='Val. UP'),
    Patch(facecolor='#89CFF0', label='Val. DOWN'),
]
ax_c.legend(handles=legend_elements, fontsize=5.5, frameon=False, loc='lower right')
ax_c.text(-0.20, 1.05, 'C', transform=ax_c.transAxes, fontsize=12, fontweight='bold', va='top')

plt.tight_layout()
fig.savefig(f'{FIG_DIR}/Figure3.tiff', dpi=300, format='tiff')
fig.savefig(f'{FIG_DIR}/Figure3.pdf', dpi=300, format='pdf')
plt.close()
print("  Figure 3 saved.")


###############################################################################
#  FIGURE 4: XCL1/XCL2 Functional Axis  (1x3)
###############################################################################
print("\n" + "=" * 70)
print("GENERATING FIGURE 4: XCL1/XCL2 Functional Axis")
print("=" * 70)

fig = plt.figure(figsize=(12, 3.8))
gs4 = gridspec.GridSpec(1, 3, wspace=0.55, width_ratios=[1, 1.1, 0.9])

# Shared data for XCL panels
subtype_order_fig4 = ['Memory (TCF7+)', 'Transitional', 'Innate-like (ID2+)',
                      'Effector (PRDM1+)', 'Exhausted (TOX-hi)', 'IRF8-high']
subtype_order_fig4 = [s for s in subtype_order_fig4 if s in adata_cd8_sub.obs['subtype_label'].unique()]
subtype_short_fig4 = {
    'Memory (TCF7+)': 'Memory\n(TCF7+)', 'Transitional': 'Trans.',
    'Innate-like (ID2+)': 'Innate-like\n(ID2+)', 'Effector (PRDM1+)': 'Effector\n(PRDM1+)',
    'Exhausted (TOX-hi)': 'Exhausted\n(TOX-hi)', 'IRF8-high': 'IRF8-\nhigh'
}
n_st_fig4 = len(subtype_order_fig4)

# ── Panel A: XCL1/XCL2 boxplot ───────────────────────────────────────────────
ax_a = fig.add_subplot(gs4[0, 0])
plot_data_xcl = []
for st in subtype_order_fig4:
    mask = adata_cd8_sub.obs['subtype_label'] == st
    cells = adata_cd8_sub.obs_names[mask]
    cells_in = [c for c in cells if c in adata_all.obs_names]
    for c in cells_in:
        idx = list(adata_all.obs_names).index(c)
        for gene in ['XCL1', 'XCL2']:
            if gene in adata_all.var_names:
                gene_idx = list(adata_all.var_names).index(gene)
                val = adata_all.X[idx, gene_idx]
                plot_data_xcl.append({'cell': c, 'subtype': subtype_short_fig4[st], 'gene': gene, 'expr': val})
plot_df_xcl = pd.DataFrame(plot_data_xcl)
subtype_short_order_fig4 = [subtype_short_fig4[s] for s in subtype_order_fig4]

for i, gene in enumerate(['XCL1', 'XCL2']):
    gene_df = plot_df_xcl[plot_df_xcl['gene'] == gene]
    positions = [j + (i - 0.5) * 0.35 for j in range(n_st_fig4)]
    bp_data = [gene_df[gene_df['subtype'] == s]['expr'].values for s in subtype_short_order_fig4]
    bp = ax_a.boxplot(bp_data, positions=positions, widths=0.30, patch_artist=True,
                      showfliers=False, medianprops=dict(color='black', linewidth=0.8),
                      whiskerprops=dict(linewidth=0.5), capprops=dict(linewidth=0.5),
                      boxprops=dict(linewidth=0.5))
    gene_color = '#D73027' if i == 0 else '#4575B4'
    for j, patch in enumerate(bp['boxes']):
        alpha_v = 0.9 if subtype_short_order_fig4[j] == 'IRF8-\nhigh' else 0.4
        patch.set_facecolor(gene_color)
        patch.set_alpha(alpha_v)
ax_a.set_xticks(range(n_st_fig4))
ax_a.set_xticklabels(subtype_short_order_fig4, fontsize=5.5, ha='center')
ax_a.set_ylabel('Expression (log1p TPM)', fontsize=7)
ax_a.set_title('XCL1/XCL2 expression by\nCD8+ T cell subtype', fontsize=8)
ax_a.tick_params(axis='both', labelsize=6.5)
legend_xcl = [Patch(facecolor='#D73027', alpha=0.7, label='XCL1'),
              Patch(facecolor='#4575B4', alpha=0.7, label='XCL2')]
ax_a.legend(handles=legend_xcl, fontsize=6, frameon=False, loc='upper left')
ax_a.text(-0.15, 1.05, 'A', transform=ax_a.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel B: IRF8 vs XCL1 scatter ────────────────────────────────────────────
ax_b = fig.add_subplot(gs4[0, 1])
for st in subtype_order_fig4:
    mask = adata_cd8_sub.obs['subtype_label'] == st
    cells = adata_cd8_sub.obs_names[mask]
    cells_in = [c for c in cells if c in adata_all.obs_names]
    if not cells_in: continue
    idxs = [list(adata_all.obs_names).index(c) for c in cells_in]
    irf8_idx_f4 = list(adata_all.var_names).index('IRF8') if 'IRF8' in adata_all.var_names else None
    xcl1_idx_f4 = list(adata_all.var_names).index('XCL1') if 'XCL1' in adata_all.var_names else None
    if irf8_idx_f4 is None or xcl1_idx_f4 is None: continue
    irf8_v = adata_all.X[idxs, irf8_idx_f4]
    xcl1_v = adata_all.X[idxs, xcl1_idx_f4]
    color = merged_color_map.get(st, '#999999')
    alpha_v = 0.7 if st == 'IRF8-high' else 0.15
    size_v = 5 if st == 'IRF8-high' else 1.5
    zorder_v = 5 if st == 'IRF8-high' else 1
    ax_b.scatter(irf8_v, xcl1_v, s=size_v, alpha=alpha_v, c=color,
                 label=st, edgecolors='none', rasterized=True, zorder=zorder_v)
cd8_cells_f4 = list(adata_cd8_sub.obs_names)
cd8_in_f4 = [c for c in cd8_cells_f4 if c in adata_all.obs_names]
cd8_idxs_f4 = [list(adata_all.obs_names).index(c) for c in cd8_in_f4]
all_irf8_f4 = adata_all.X[cd8_idxs_f4, list(adata_all.var_names).index('IRF8')]
all_xcl1_f4 = adata_all.X[cd8_idxs_f4, list(adata_all.var_names).index('XCL1')]
rho_f4, pval_f4 = spearmanr(all_irf8_f4, all_xcl1_f4)
ax_b.text(0.03, 0.97, f'Spearman \u03c1 = {rho_f4:.3f}\np = {pval_f4:.2e}',
          transform=ax_b.transAxes, fontsize=6.5, va='top',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='grey', alpha=0.8))
ax_b.set_xlabel('IRF8 expression (log1p TPM)', fontsize=7)
ax_b.set_ylabel('XCL1 expression (log1p TPM)', fontsize=7)
ax_b.set_title('IRF8 vs XCL1 co-expression\nacross CD8+ T cells', fontsize=8)
ax_b.tick_params(axis='both', labelsize=6.5)
handles_b = [plt.scatter([], [], s=8, c=merged_color_map.get(st, '#999999'), label=st) for st in subtype_order_fig4]
ax_b.legend(handles=handles_b, fontsize=5, frameon=False, loc='upper right',
            markerscale=1.5, handletextpad=0.3)
ax_b.text(-0.12, 1.05, 'B', transform=ax_b.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel C: Surface molecules IRF8-high vs TOX-hi ──────────────────────────
ax_c = fig.add_subplot(gs4[0, 2])
diff_genes_f4 = ['XCL2', 'TNFRSF9', 'XCL1', 'IFNG', 'IL21R', 'BTLA', 'VCAM1', 'TNFRSF18']
diff_labels_f4 = ['XCL2', 'TNFRSF9\n(4-1BB)', 'XCL1', 'IFNG', 'IL21R', 'BTLA', 'VCAM1', 'TNFRSF18\n(GITR)']
diff_vals_f4 = []
diff_pvals_f4 = []
irf8_mask_f4 = adata_cd8_sub.obs['subtype_label'] == 'IRF8-high'
tox_mask_f4 = adata_cd8_sub.obs['subtype_label'] == 'Exhausted (TOX-hi)'
irf8_cells_f4c = [c for c in adata_cd8_sub.obs_names[irf8_mask_f4] if c in adata_all.obs_names]
tox_cells_f4c = [c for c in adata_cd8_sub.obs_names[tox_mask_f4] if c in adata_all.obs_names]
irf8_idxs_f4c = [list(adata_all.obs_names).index(c) for c in irf8_cells_f4c]
tox_idxs_f4c = [list(adata_all.obs_names).index(c) for c in tox_cells_f4c]
print(f"  Fig4C: IRF8-high n={len(irf8_idxs_f4c)}, Exhausted TOX-hi n={len(tox_idxs_f4c)}")
for gene in diff_genes_f4:
    gene_idx = list(adata_all.var_names).index(gene) if gene in adata_all.var_names else None
    if gene_idx is None:
        diff_vals_f4.append(0); diff_pvals_f4.append(1); continue
    iv = adata_all.X[irf8_idxs_f4c, gene_idx]
    tv = adata_all.X[tox_idxs_f4c, gene_idx]
    diff_vals_f4.append(float(iv.mean() - tv.mean()))
    _, pv = mannwhitneyu(iv, tv, alternative='two-sided')
    diff_pvals_f4.append(pv)
bar_colors_f4 = ['#D73027' if pv < 0.001 else '#FC8D59' if pv < 0.01 else '#FEE08B' if pv < 0.05 else '#999999' for pv in diff_pvals_f4]
ax_c.barh(range(len(diff_genes_f4)), diff_vals_f4, color=bar_colors_f4, edgecolor='white', linewidth=0.3, height=0.6)
ax_c.set_yticks(range(len(diff_genes_f4)))
ax_c.set_yticklabels(diff_labels_f4, fontsize=6.5, fontstyle='italic')
ax_c.set_xlabel('Expression difference\n(IRF8-high \u2212 Exhausted TOX-hi)', fontsize=7)
ax_c.set_title('Surface molecules elevated\nin IRF8-high vs Exhausted (TOX-hi)', fontsize=8)
ax_c.axvline(0, color='grey', linestyle='--', linewidth=0.5)
ax_c.tick_params(axis='both', labelsize=6.5)
legend_f4 = [Patch(facecolor='#D73027', label='p < 0.001'),
             Patch(facecolor='#FC8D59', label='p < 0.01'),
             Patch(facecolor='#FEE08B', label='p < 0.05')]
ax_c.legend(handles=legend_f4, fontsize=5, frameon=False, loc='upper right')
ax_c.text(-0.18, 1.05, 'C', transform=ax_c.transAxes, fontsize=12, fontweight='bold', va='top')

plt.tight_layout()
fig.savefig(f'{FIG_DIR}/Figure4.tiff', dpi=300, format='tiff')
fig.savefig(f'{FIG_DIR}/Figure4.pdf', dpi=300, format='pdf')
plt.close()
print("  Figure 4 saved (XCL1/XCL2 functional axis).")


###############################################################################
#  Prepare patient-level data (shared by S3F and S5C)
###############################################################################
print("\nPreparing patient-level data...")
resp_patient_data = pd.read_csv(f'{OUT_DIR}/irf8high_response_patient_data.csv')
disc_patient_data = []
pt_col = 'samples' if 'samples' in adata_cd8_sub.obs.columns else ('patient.id' if 'patient.id' in adata_cd8_sub.obs.columns else None)
disc_patients = adata_cd8_sub.obs[pt_col].unique() if pt_col else []
for pt in disc_patients:
    pt_mask = adata_cd8_sub.obs[pt_col] == pt
    n_total = pt_mask.sum()
    if n_total < 5:
        continue
    n_irf8 = (adata_cd8_sub.obs.loc[pt_mask, 'subtype_label'] == 'IRF8-high').sum()
    prop_irf8 = n_irf8 / n_total
    treat_vals = adata_cd8_sub.obs.loc[pt_mask, 'treatment.group'].value_counts()
    treat = 'Post' if treat_vals.index[0] != 'treatment.naive' else 'Pre'
    disc_patient_data.append({
        'patient': pt,
        'treatment': treat,
        'prop_irf8': prop_irf8,
        'cohort': 'Discovery',
    })
disc_pt_df = pd.DataFrame(disc_patient_data)
val_pt_df = resp_patient_data.copy()
val_pt_df['cohort'] = 'Validation'
print("  Patient-level data prepared.")


###############################################################################
#  SUPPLEMENTARY FIGURE S5 (2x3): Treatment Changes + Malignant Subtypes
#    Row 1: A=TF treatment bar, B=CD8 violin, C=Patient treatment
#    Row 2: D=Malignant UMAP, E=Malignant TF heatmap, F=Mal composition
###############################################################################
print("\n" + "=" * 70)
print("GENERATING SUPPLEMENTARY FIGURE S5: Treatment Changes + Malignant Subtypes")
print("=" * 70)

fig = plt.figure(figsize=(16, 9))
gs_s5 = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.50,
                           width_ratios=[1, 0.9, 0.9])

# ── Panel A: All-cell TF treatment changes (moved from old Fig4A) ────────────
ax_s5a = fig.add_subplot(gs_s5[0, 0])
tf_treat_sig = tf_treat_all[tf_treat_all['padj'] < 0.05].copy()
tf_up = tf_treat_sig[tf_treat_sig['log2_fc'] > 0].nlargest(6, 'log2_fc')
tf_down = tf_treat_sig[tf_treat_sig['log2_fc'] < 0].nsmallest(6, 'log2_fc')
tf_treat_plot = pd.concat([tf_up, tf_down]).sort_values('log2_fc')

colors_bar = ['#D73027' if fc > 0 else '#4575B4' for fc in tf_treat_plot['log2_fc']]
ax_s5a.barh(range(len(tf_treat_plot)), tf_treat_plot['log2_fc'], color=colors_bar,
            edgecolor='white', linewidth=0.3, height=0.7)
ax_s5a.set_yticks(range(len(tf_treat_plot)))
ax_s5a.set_yticklabels(tf_treat_plot['TF'].values, fontsize=7, fontstyle='italic')
ax_s5a.set_xlabel('log$_2$ fold change\n(Post-treatment / Pre-tx)', fontsize=7)
ax_s5a.set_title('Treatment-associated TF changes\n(All cells, FDR < 0.05)', fontsize=8)
ax_s5a.axvline(x=0, color='black', linewidth=0.5)
ax_s5a.text(-0.18, 1.05, 'A', transform=ax_s5a.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel B: CD8 violin + strip for key TFs (moved from old Fig4B) ──────────
ax_s5b = fig.add_subplot(gs_s5[0, 1])
cd8_key_tfs = ['TOX', 'PRDM1', 'IRF8', 'IRF1']
adata_cd8_disc = adata_all[adata_all.obs['cell.types'] == 'T.CD8'].copy()

plot_data_s5b = []
for tf in cd8_key_tfs:
    if tf in adata_cd8_disc.var_names:
        expr = adata_cd8_disc[:, tf].X.flatten()
        for i, val in enumerate(expr):
            treat = 'Pre-tx' if adata_cd8_disc.obs['treatment.group'].values[i] == 'treatment.naive' else 'Post-tx'
            plot_data_s5b.append({'TF': tf, 'Expression': float(val), 'Treatment': treat})

plot_df_s5b = pd.DataFrame(plot_data_s5b)

sns.violinplot(data=plot_df_s5b, x='TF', y='Expression', hue='Treatment',
               palette=treat_palette, ax=ax_s5b, linewidth=0.5, width=0.8,
               inner=None, cut=0, split=True, density_norm='width')

# Add jitter
np.random.seed(42)
for tf_idx, tf in enumerate(cd8_key_tfs):
    for treat, color in [('Pre-tx', '#4575B4'), ('Post-tx', '#D73027')]:
        sub = plot_df_s5b[(plot_df_s5b['TF'] == tf) & (plot_df_s5b['Treatment'] == treat)]
        if len(sub) > 100:
            sub = sub.sample(100, random_state=42)
        offset = -0.15 if treat == 'Pre-tx' else 0.15
        jitter = np.random.normal(0, 0.03, len(sub))
        ax_s5b.scatter(tf_idx + offset + jitter, sub['Expression'].values,
                       s=0.3, alpha=0.3, c=color, rasterized=True)

# Significance
for i, tf in enumerate(cd8_key_tfs):
    row = cd8_treat[cd8_treat['TF'] == tf]
    if len(row) > 0:
        padj = row['padj'].values[0]
        if padj < 0.001:
            sig = '***'
        elif padj < 0.01:
            sig = '**'
        elif padj < 0.05:
            sig = '*'
        else:
            sig = 'ns'
        ax_s5b.text(i, 2.65, sig, ha='center', fontsize=7, fontweight='bold')

ax_s5b.set_xlabel('')
ax_s5b.set_ylabel('Expression (log1p TPM)', fontsize=7)
ax_s5b.set_ylim(0, 3.0)
ax_s5b.set_xlim(-0.6, len(cd8_key_tfs) - 0.4)
ax_s5b.set_title('CD8+ T cell TF changes after ICB', fontsize=8)
ax_s5b.set_xticklabels(cd8_key_tfs, fontstyle='italic', fontsize=7)
handles_s5b, labels_s5b = ax_s5b.get_legend_handles_labels()
ax_s5b.legend(handles_s5b[:2], labels_s5b[:2], fontsize=6, frameon=False, loc='upper right')
ax_s5b.text(-0.12, 1.05, 'B', transform=ax_s5b.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel C: Patient-level IRF8-high proportion by TREATMENT ─────────────────
ax_s5c = fig.add_subplot(gs_s5[0, 2])

disc_pre = disc_pt_df[disc_pt_df['treatment'] == 'Pre']
disc_post = disc_pt_df[disc_pt_df['treatment'] == 'Post']

np.random.seed(42)
if len(disc_pre) > 0:
    jitter = np.random.normal(0, 0.08, len(disc_pre))
    ax_s5c.scatter(0 - 0.15 + jitter, disc_pre['prop_irf8'].values,
               s=15, alpha=0.7, c='#4575B4', edgecolors='white', linewidth=0.3,
               label='Pre-tx', zorder=3)
if len(disc_post) > 0:
    jitter = np.random.normal(0, 0.08, len(disc_post))
    ax_s5c.scatter(0 + 0.15 + jitter, disc_post['prop_irf8'].values,
               s=15, alpha=0.7, c='#D73027', edgecolors='white', linewidth=0.3,
               label='Post-tx', zorder=3)

if len(disc_pre) >= 3 and len(disc_post) >= 3:
    stat_d, pval_d = mannwhitneyu(disc_pre['prop_irf8'].values,
                                   disc_post['prop_irf8'].values,
                                   alternative='two-sided')
    sig_d = f'p={pval_d:.2e}'
else:
    sig_d = ''

val_pre = val_pt_df[val_pt_df['treatment'] == 'Pre']
val_post = val_pt_df[val_pt_df['treatment'] == 'Post']

if len(val_pre) > 0:
    jitter = np.random.normal(0, 0.08, len(val_pre))
    ax_s5c.scatter(1 - 0.15 + jitter, val_pre['prop_irf8'].values,
               s=15, alpha=0.7, c='#4575B4', edgecolors='white', linewidth=0.3,
               zorder=3)
if len(val_post) > 0:
    jitter = np.random.normal(0, 0.08, len(val_post))
    ax_s5c.scatter(1 + 0.15 + jitter, val_post['prop_irf8'].values,
               s=15, alpha=0.7, c='#D73027', edgecolors='white', linewidth=0.3,
               zorder=3)

if len(val_pre) >= 3 and len(val_post) >= 3:
    stat_v, pval_v = mannwhitneyu(val_pre['prop_irf8'].values,
                                   val_post['prop_irf8'].values,
                                   alternative='two-sided')
    sig_v = f'p={pval_v:.2e}'
else:
    sig_v = ''

for x, pre_data, post_data in [(0, disc_pre, disc_post), (1, val_pre, val_post)]:
    if len(pre_data) > 0:
        ax_s5c.plot([x - 0.25, x - 0.05], [pre_data['prop_irf8'].mean()] * 2,
                color='#4575B4', linewidth=1.5, zorder=4)
    if len(post_data) > 0:
        ax_s5c.plot([x + 0.05, x + 0.25], [post_data['prop_irf8'].mean()] * 2,
                color='#D73027', linewidth=1.5, zorder=4)

y_max_plot = max(
    disc_pt_df['prop_irf8'].max() if len(disc_pt_df) > 0 else 0,
    val_pt_df['prop_irf8'].max() if len(val_pt_df) > 0 else 0,
) * 1.1

if sig_d:
    ax_s5c.text(0, y_max_plot, sig_d, ha='center', fontsize=6, fontweight='bold')
if sig_v:
    ax_s5c.text(1, y_max_plot, sig_v, ha='center', fontsize=6, fontweight='bold')

ax_s5c.set_xticks([0, 1])
ax_s5c.set_xticklabels(['Discovery\n(GSE115978)', 'Validation\n(GSE120575)'], fontsize=7)
ax_s5c.set_ylabel('IRF8-high proportion per patient', fontsize=7)
ax_s5c.set_title('Patient-level IRF8-high proportion\nby treatment', fontsize=8)
ax_s5c.set_ylim(-0.02, y_max_plot * 1.15)

handles_s5c, labels_s5c = ax_s5c.get_legend_handles_labels()
by_label_s5c = dict(zip(labels_s5c, handles_s5c))
ax_s5c.legend(by_label_s5c.values(), by_label_s5c.keys(), fontsize=6, frameon=False, loc='upper right')
ax_s5c.text(-0.12, 1.05, 'C', transform=ax_s5c.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel D: Malignant UMAP (moved from old S5A) ────────────────────────────
ax_s5d = fig.add_subplot(gs_s5[1, 0])
subtypes_mal = sorted(adata_mal_sub.obs['tf_subtype_v2'].unique())
subtype_colors_mal = plt.cm.Set1(np.linspace(0, 1, adata_mal_sub.obs['tf_subtype_v2'].nunique()))
for i, st in enumerate(subtypes_mal):
    mask = adata_mal_sub.obs['tf_subtype_v2'] == st
    label_str = mal_labels_v2.get(st, st)
    n = mask.sum()
    ax_s5d.scatter(
        adata_mal_sub.obsm['X_umap_tf'][mask, 0], adata_mal_sub.obsm['X_umap_tf'][mask, 1],
        s=2, alpha=0.6, c=[subtype_colors_mal[i]], label=f'{label_str} ({n})',
        rasterized=True
    )
ax_s5d.set_title('Malignant cell TF-based subtypes', fontsize=8)
ax_s5d.set_xlabel('UMAP1', fontsize=7)
ax_s5d.set_ylabel('UMAP2', fontsize=7)
ax_s5d.set_xticks([])
ax_s5d.set_yticks([])
ax_s5d.legend(fontsize=5, markerscale=3, frameon=False,
              bbox_to_anchor=(0.0, -0.08), loc='upper left', ncol=2,
              handletextpad=0.1, columnspacing=0.3)
ax_s5d.text(-0.05, 1.05, 'D', transform=ax_s5d.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel E: Malignant subtype TF characterization heatmap (moved from old S5B) ──
ax_s5e = fig.add_subplot(gs_s5[1, 1])

mal_char_data = []
used_labels = []
for st in subtypes_mal:
    lbl = mal_labels_v2.get(st, st)
    if lbl not in used_labels:
        used_labels.append(lbl)

for lbl in used_labels:
    mask = (adata_mal_sub.obs['subtype_label'] == lbl).values
    n = mask.sum()
    post_pct = (adata_mal_sub.obs.loc[mask, 'treatment.group'] != 'treatment.naive').mean() * 100
    row = {'Subtype': lbl}
    for tf in ['SOX10', 'MITF', 'ETV5', 'PAX3', 'SOX4']:
        if tf in adata_mal_sub.var_names:
            row[tf] = float(adata_mal_sub[mask, tf].X.mean())
        else:
            row[tf] = 0.0
    row['Post-tx %'] = post_pct
    mal_char_data.append(row)

mal_char_df = pd.DataFrame(mal_char_data).set_index('Subtype')
mal_tf_cols = ['SOX10', 'MITF', 'ETV5', 'PAX3', 'SOX4', 'Post-tx %']

mal_char_z = mal_char_df[mal_tf_cols].copy()
for col in mal_tf_cols:
    vals = mal_char_z[col].values.astype(float)
    if vals.std() > 0:
        mal_char_z[col] = (vals - vals.mean()) / vals.std()

mal_char_z = mal_char_z.astype(float)
mal_annot_vals = mal_char_df[mal_tf_cols].astype(float).T.round(1)
sns.heatmap(mal_char_z.T, cmap='RdBu_r', center=0, ax=ax_s5e,
            yticklabels=True, xticklabels=True,
            cbar_kws={'shrink': 0.6, 'label': 'Z-score'},
            linewidths=0.5, linecolor='white', annot=mal_annot_vals,
            fmt='.1f', annot_kws={'fontsize': 6})
ax_s5e.set_title('Malignant subtype TF characterization', fontsize=8)
ax_s5e.tick_params(axis='y', labelsize=7)
ax_s5e.tick_params(axis='x', labelsize=6, rotation=45)
ylabels_s5e = ax_s5e.get_yticklabels()
for yl in ylabels_s5e:
    if yl.get_text() != 'Post-tx %':
        yl.set_fontstyle('italic')
ax_s5e.set_yticklabels(ylabels_s5e)
ax_s5e.text(-0.12, 1.05, 'E', transform=ax_s5e.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel F: Malignant subtype treatment composition (moved from old Fig4C) ──
ax_s5f = fig.add_subplot(gs_s5[1, 2])

mal_comp = []
for st in subtypes_mal:
    mask = adata_mal_sub.obs['tf_subtype_v2'] == st
    label_str = mal_labels_v2.get(st, st)
    n = mask.sum()
    naive_pct = (adata_mal_sub.obs.loc[mask, 'treatment.group'] == 'treatment.naive').mean() * 100
    mal_comp.append({
        'Subtype': label_str,
        'Pre-tx': naive_pct,
        'Post-tx': 100 - naive_pct,
    })

mal_comp_df = pd.DataFrame(mal_comp).sort_values('Pre-tx', ascending=True)

y_pos_mal = range(len(mal_comp_df))
ax_s5f.barh(y_pos_mal, mal_comp_df['Pre-tx'].values, color='#4575B4', label='Pre-tx',
            edgecolor='white', linewidth=0.3)
ax_s5f.barh(y_pos_mal, mal_comp_df['Post-tx'].values, left=mal_comp_df['Pre-tx'].values,
            color='#D73027', label='Post-tx', edgecolor='white', linewidth=0.3)
ax_s5f.set_yticks(y_pos_mal)
ax_s5f.set_yticklabels(mal_comp_df['Subtype'].values, fontsize=6)
ax_s5f.set_xlabel('Percentage (%)', fontsize=7)
ax_s5f.set_title('Malignant subtype\ntreatment composition', fontsize=8)
ax_s5f.legend(fontsize=6, frameon=False, loc='lower right')
ax_s5f.axvline(x=50, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)
ax_s5f.text(-0.12, 1.05, 'F', transform=ax_s5f.transAxes, fontsize=12, fontweight='bold', va='top')

plt.tight_layout()
fig.savefig(f'{FIG_DIR}/FigureS5.tiff', dpi=300, format='tiff')
fig.savefig(f'{FIG_DIR}/FigureS5.pdf', dpi=300, format='pdf')
plt.close()
print("  Supplementary Figure S5 saved (Treatment changes + Malignant subtypes, 2x3).")


###############################################################################
#  SUPPLEMENTARY FIGURE S1: CD8+ Lineage Markers + TF Expression by Cell Type
#    A: Top 20 TF mean expression across all TME cell types (heatmap)
#    B: % cells expressing lineage markers per CD8+ subtype (bar)
#    C: Mean lineage marker expression per CD8+ subtype (heatmap)
###############################################################################
print("\n" + "=" * 70)
print("GENERATING SUPPLEMENTARY FIGURE S1: Lineage Markers + TF Expression")
print("=" * 70)

# ── Panel A+B: Lineage markers per CD8+ subtype ─────────────────────────────
lineage_markers = ['CD8A', 'CD3D', 'CD163', 'CSF1R', 'CD14']
marker_labels = ['CD8A\n(CD8+ T)', 'CD3D\n(T cell)', 'CD163\n(Macro.)', 'CSF1R\n(Macro.)', 'CD14\n(Mono.)']

subtype_order_s1 = ['IRF8-high', 'Exhausted (TOX-hi)', 'Effector (PRDM1+)',
                    'Innate-like (ID2+)', 'Transitional', 'Memory (TCF7+)']
subtype_order_s1 = [s for s in subtype_order_s1 if s in adata_cd8_sub.obs['subtype_label'].unique()]

pct_data_s1 = []
mean_data_s1 = []
for st in subtype_order_s1:
    mask = adata_cd8_sub.obs['subtype_label'] == st
    n_cells = mask.sum()
    for marker in lineage_markers:
        if marker in adata_cd8_sub.var_names:
            expr = adata_cd8_sub[mask, marker].X.flatten()
            pct_expr = (expr > 0).mean() * 100
            mean_expr = float(expr.mean())
        else:
            pct_expr = 0.0
            mean_expr = 0.0
        pct_data_s1.append({'Subtype': st, 'Marker': marker, 'Pct': pct_expr, 'n': n_cells})
        mean_data_s1.append({'Subtype': st, 'Marker': marker, 'Mean': mean_expr})

pct_df_s1 = pd.DataFrame(pct_data_s1)
mean_df_s1 = pd.DataFrame(mean_data_s1)
pct_pivot_s1 = pct_df_s1.pivot(index='Subtype', columns='Marker', values='Pct').loc[subtype_order_s1, lineage_markers]
mean_pivot_s1 = mean_df_s1.pivot(index='Subtype', columns='Marker', values='Mean').loc[subtype_order_s1, lineage_markers]

fig = plt.figure(figsize=(16, 5))
gs_s1 = gridspec.GridSpec(1, 3, wspace=0.40, width_ratios=[1.2, 1.3, 0.9])

# Panel B: Grouped bar (% expressing lineage markers)
ax_a_s1 = fig.add_subplot(gs_s1[0, 1])
n_st_s1 = len(subtype_order_s1)
n_mk_s1 = len(lineage_markers)
x_s1 = np.arange(n_mk_s1)
bar_w = 0.8 / n_st_s1

for i, st in enumerate(subtype_order_s1):
    st_vals = pct_df_s1[pct_df_s1['Subtype'] == st]
    vals = [st_vals[st_vals['Marker'] == m]['Pct'].values[0] for m in lineage_markers]
    n_c = st_vals['n'].values[0]
    color = merged_color_map.get(st, '#999999')
    ax_a_s1.bar(x_s1 + i * bar_w - (n_st_s1 - 1) * bar_w / 2,
                vals, bar_w * 0.9, label=f'{st} (n={n_c})',
                color=color, alpha=0.85, edgecolor='white', linewidth=0.3)

ax_a_s1.set_ylabel('% cells expressing', fontsize=8)
ax_a_s1.set_title('Lineage marker expression\nby CD8+ T cell subtype', fontsize=8)
ax_a_s1.set_xticks(x_s1)
ax_a_s1.set_xticklabels(marker_labels, fontsize=7)
ax_a_s1.set_ylim(0, 115)
ax_a_s1.axhline(y=100, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax_a_s1.legend(fontsize=5.5, frameon=False, loc='upper right', ncol=2, handletextpad=0.2)
ax_a_s1.tick_params(axis='both', labelsize=7)
ax_a_s1.spines['top'].set_visible(False)
ax_a_s1.spines['right'].set_visible(False)
ax_a_s1.text(-0.08, 1.05, 'B', transform=ax_a_s1.transAxes, fontsize=12, fontweight='bold', va='top')

# Panel C: Mean expression heatmap (lineage markers)
ax_b_s1 = fig.add_subplot(gs_s1[0, 2])
short_names_s1 = {
    'IRF8-high': 'IRF8-high', 'Exhausted (TOX-hi)': 'Exh. TOX-hi',
    'Effector (PRDM1+)': 'Eff. PRDM1+', 'Innate-like (ID2+)': 'Innate ID2+',
    'Transitional': 'Transitional', 'Memory (TCF7+)': 'Mem. TCF7+',
}
annot_str_s1 = np.empty_like(mean_pivot_s1.values, dtype=object)
for i, st in enumerate(subtype_order_s1):
    for j, marker in enumerate(lineage_markers):
        m_val = mean_pivot_s1.loc[st, marker]
        p_val = pct_pivot_s1.loc[st, marker]
        annot_str_s1[i, j] = f'{m_val:.1f}\n({p_val:.0f}%)'

sns.heatmap(mean_pivot_s1, ax=ax_b_s1, cmap='YlOrRd', linewidths=0.5,
            annot=annot_str_s1, fmt='', annot_kws={'size': 6},
            cbar_kws={'shrink': 0.7, 'label': 'Mean log1p(TPM)'})
ax_b_s1.set_title('Mean expression\n(% expressing)', fontsize=8)
ax_b_s1.set_xticklabels(lineage_markers, rotation=45, ha='right', fontsize=7, fontstyle='italic')
ylabels_s1 = [short_names_s1.get(st, st) for st in subtype_order_s1]
ax_b_s1.set_yticklabels(ylabels_s1, fontsize=6.5, rotation=0)
for tick_label, st in zip(ax_b_s1.get_yticklabels(), subtype_order_s1):
    tick_label.set_color(merged_color_map.get(st, '#333333'))
ax_b_s1.text(-0.15, 1.05, 'C', transform=ax_b_s1.transAxes, fontsize=12, fontweight='bold', va='top')

# Panel A: Top 20 TF mean expression by cell type (whole TME)
ax_c_s1 = fig.add_subplot(gs_s1[0, 0])

key_celltypes_s1 = ['Mal', 'T_CD8', 'T_CD4', 'B_cell', 'Macrophage', 'NK', 'Endo_', 'CAF']
key_ct_labels_s1 = ['Malignant', 'CD8+ T', 'CD4+ T', 'B cell', 'Macrophage', 'NK', 'Endothelial', 'CAF']
top_tfs_s1 = tf_global.head(20)['TF'].tolist()

# Compute mean expression from adata_all (log1p TPM already applied)
# Map cell type annotation names to underscore keys
ct_name_map = {'Mal': 'Mal', 'T.CD8': 'T_CD8', 'T.CD4': 'T_CD4', 'B.cell': 'B_cell',
               'Macrophage': 'Macrophage', 'NK': 'NK', 'Endo.': 'Endo_', 'CAF': 'CAF'}
ct_name_map_rev = {v: k for k, v in ct_name_map.items()}

expr_heatmap = pd.DataFrame(index=top_tfs_s1, columns=key_ct_labels_s1, dtype=float)
for ct_key, ct_label in zip(key_celltypes_s1, key_ct_labels_s1):
    ct_annot_name = ct_name_map_rev.get(ct_key, ct_key)
    mask = adata_all.obs['cell.types'] == ct_annot_name
    if mask.sum() > 0:
        for tf in top_tfs_s1:
            if tf in adata_all.var_names:
                expr_heatmap.loc[tf, ct_label] = float(adata_all[mask, tf].X.mean())
            else:
                expr_heatmap.loc[tf, ct_label] = 0.0

expr_heatmap = expr_heatmap.fillna(0.0).astype(float)

# Use same row/col clustering order as Figure 1D SHAP heatmap for direct comparison
heatmap_data_1d = pd.DataFrame(index=top_tfs_s1, columns=key_ct_labels_s1)
for ct, ct_label in zip(key_celltypes_s1, key_ct_labels_s1):
    if ct in celltype_tfs:
        df = celltype_tfs[ct].set_index('TF')
        for tf in top_tfs_s1:
            heatmap_data_1d.loc[tf, ct_label] = df.loc[tf, 'mean_shap'] if tf in df.index else 0
heatmap_data_1d = heatmap_data_1d.fillna(0).astype(float)

row_link_s1 = linkage(pdist(heatmap_data_1d.values, metric='euclidean'), method='ward')
col_link_s1 = linkage(pdist(heatmap_data_1d.values.T, metric='euclidean'), method='ward')
row_ord_s1 = dendrogram(row_link_s1, no_plot=True)['leaves']
col_ord_s1 = dendrogram(col_link_s1, no_plot=True)['leaves']
expr_heatmap = expr_heatmap.iloc[row_ord_s1, col_ord_s1]

sns.heatmap(expr_heatmap, cmap='YlOrRd', ax=ax_c_s1,
            vmin=0, yticklabels=True, xticklabels=True,
            cbar_kws={'shrink': 0.6, 'label': 'Mean log1p(TPM)'},
            linewidths=0.5, linecolor='white',
            annot=True, fmt='.1f', annot_kws={'fontsize': 5})
ax_c_s1.set_title('TF expression across cell types\n(cf. Fig. 1D: SHAP importance)', fontsize=8)
ax_c_s1.tick_params(axis='y', labelsize=7)
ax_c_s1.tick_params(axis='x', labelsize=7, rotation=45)
ax_c_s1.set_yticklabels(ax_c_s1.get_yticklabels(), fontstyle='italic')
ax_c_s1.text(-0.12, 1.05, 'A', transform=ax_c_s1.transAxes, fontsize=12, fontweight='bold', va='top')

plt.tight_layout()
fig.savefig(f'{FIG_DIR}/FigureS1.tiff', dpi=300, format='tiff')
fig.savefig(f'{FIG_DIR}/FigureS1.pdf', dpi=300, format='pdf')
plt.close()
print("  Supplementary Figure S1 saved.")


# (Old S4 standalone removed — patient data prep moved earlier, panels to S3F + S5C)


###############################################################################
#  SUPPLEMENTARY FIGURE S3: Validation Cohort Details  (2x3, 5 panels)
#    Row 1: A=UMAP, B=Treatment composition, C=IRF8 violin (pre/post)
#    Row 2: D=Lineage markers, E=Response by subtype
###############################################################################
print("\n" + "=" * 70)
print("GENERATING SUPPLEMENTARY FIGURE S3: Validation Cohort Details")
print("=" * 70)

fig = plt.figure(figsize=(15, 9))
gs_s3 = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.4,
                          width_ratios=[1, 1, 1])

label_order_val = ['IRF8-high', 'Exhausted (TOX-hi)', 'Memory (TCF7+)',
                   'Cytotoxic (ID2+PRDM1+)', 'Innate-like (ID2+)',
                   'Effector (PRDM1+)', 'Effector']
label_order_val = [l for l in label_order_val if l in set(adata_cd8_val.obs['subtype_label'])]

# ── Panel A: Validation UMAP by subtype ──────────────────────────────────────
ax_a = fig.add_subplot(gs_s3[0, 0])

for lbl in label_order_val:
    mask = (adata_cd8_val.obs['subtype_label'] == lbl).values
    n = mask.sum()
    color = merged_color_map.get(lbl, '#999999')
    ax_a.scatter(
        adata_cd8_val.obsm['X_umap_tf'][mask, 0], adata_cd8_val.obsm['X_umap_tf'][mask, 1],
        s=2, alpha=0.6, c=color, label=f'{lbl} ({n})',
        rasterized=True
    )
ax_a.set_title('Validation: CD8+ T cell subtypes\n(GSE120575, Sade-Feldman et al.)', fontsize=8)
ax_a.set_xlabel('UMAP1', fontsize=7)
ax_a.set_ylabel('UMAP2', fontsize=7)
ax_a.set_xticks([])
ax_a.set_yticks([])
ax_a.legend(fontsize=4.5, markerscale=3, frameon=False,
            bbox_to_anchor=(0.0, -0.08), loc='upper left', ncol=2,
            handletextpad=0.1, columnspacing=0.3)
ax_a.text(-0.05, 1.05, 'A', transform=ax_a.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel B: Treatment composition validation ────────────────────────────────
ax_b = fig.add_subplot(gs_s3[0, 1])

val_merged_stats = {}
for lbl in label_order_val:
    mask_lbl = adata_cd8_val.obs['subtype_label'] == lbl
    n = mask_lbl.sum()
    pre_n = (adata_cd8_val.obs.loc[mask_lbl, 'treatment'] == 'Pre').sum()
    post_n = (adata_cd8_val.obs.loc[mask_lbl, 'treatment'] == 'Post').sum()
    val_merged_stats[lbl] = {
        'n': n, 'pre_n': pre_n, 'post_n': post_n,
        'post_pct': post_n / n * 100 if n > 0 else 0,
    }

comp_data = []
for lbl in label_order_val:
    ms = val_merged_stats[lbl]
    comp_data.append({
        'Subtype': f'{lbl}\n(n={ms["n"]})',
        'Pre': ms['pre_n'] / ms['n'] * 100 if ms['n'] > 0 else 0,
        'Post': ms['post_n'] / ms['n'] * 100 if ms['n'] > 0 else 0,
    })
comp_df = pd.DataFrame(comp_data).sort_values('Post', ascending=True).reset_index(drop=True)

y_pos_c = range(len(comp_df))
ax_b.barh(y_pos_c, comp_df['Pre'].values, color='#4575B4', label='Pre-tx',
          edgecolor='white', linewidth=0.3)
ax_b.barh(y_pos_c, comp_df['Post'].values, left=comp_df['Pre'].values,
          color='#D73027', label='Post-tx', edgecolor='white', linewidth=0.3)
ax_b.set_yticks(y_pos_c)
ax_b.set_yticklabels(comp_df['Subtype'].values, fontsize=6.5)
ax_b.set_xlabel('Percentage (%)', fontsize=7)
ax_b.set_title('Validation: Treatment composition', fontsize=8)
ax_b.legend(fontsize=6, frameon=False, loc='lower right')
ax_b.axvline(x=50, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)
ax_b.text(-0.15, 1.05, 'B', transform=ax_b.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel C: IRF8 expression violin Pre vs Post (validation) ─────────────────
ax_c = fig.add_subplot(gs_s3[0, 2])

if 'IRF8' in adata_cd8_val.var_names:
    plot_data_irf8 = pd.DataFrame({
        'IRF8': adata_cd8_val[:, 'IRF8'].X.flatten(),
        'Treatment': ['Pre-tx' if t == 'Pre' else 'Post-tx' for t in adata_cd8_val.obs['treatment']],
    })
    treat_pal_val = {'Pre-tx': '#4575B4', 'Post-tx': '#D73027'}
    sns.violinplot(data=plot_data_irf8, x='Treatment', y='IRF8', palette=treat_pal_val,
                   ax=ax_c, linewidth=0.5, inner=None, cut=0)
    np.random.seed(42)
    for treat, color in [('Pre-tx', '#4575B4'), ('Post-tx', '#D73027')]:
        sub = plot_data_irf8[plot_data_irf8['Treatment'] == treat]
        if len(sub) > 200:
            sub = sub.sample(200, random_state=42)
        x_idx = 0 if treat == 'Pre-tx' else 1
        jitter = np.random.normal(0, 0.05, len(sub))
        ax_c.scatter(x_idx + jitter, sub['IRF8'].values, s=0.5, alpha=0.3, c=color, rasterized=True)

    stat, pval = mannwhitneyu(
        plot_data_irf8[plot_data_irf8['Treatment'] == 'Pre-tx']['IRF8'],
        plot_data_irf8[plot_data_irf8['Treatment'] == 'Post-tx']['IRF8'],
        alternative='two-sided'
    )
    sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
    y_max = plot_data_irf8['IRF8'].max()
    bracket_y = y_max * 1.08
    ax_c.plot([0, 0, 1, 1], [bracket_y * 0.98, bracket_y, bracket_y, bracket_y * 0.98],
              color='black', linewidth=0.8)
    ax_c.text(0.5, bracket_y * 1.02, sig, ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax_c.set_ylim(-0.3, bracket_y * 1.15)
    ax_c.set_title(f'IRF8 expression in CD8+ T cells\n(Validation cohort)', fontsize=8)
    ax_c.text(0.5, 0.97, f'Mann–Whitney p = {pval:.2e}', transform=ax_c.transAxes,
              ha='center', va='top', fontsize=6.5, fontstyle='italic')
    ax_c.set_ylabel('IRF8 expression (log1p TPM)', fontsize=7)

ax_c.text(-0.12, 1.05, 'C', transform=ax_c.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel D: Validation lineage markers (IRF8-high only) ─────────────────────
ax_d = fig.add_subplot(gs_s3[1, 0])

lineage_markers = ['CD8A', 'CD3D', 'CD163', 'CSF1R', 'CD14']
marker_labels = ['CD8A\n(CD8+ T)', 'CD3D\n(T cell)', 'CD163\n(Macrophage)',
                 'CSF1R\n(Macrophage)', 'CD14\n(Monocyte)']
available_markers = [m for m in lineage_markers if m in adata_cd8_val.var_names]

# Compute % expressing for each subtype in validation
val_subtypes_for_lineage = [l for l in label_order_val if l in adata_cd8_val.obs['subtype_label'].unique()]
lineage_data = {}
for st in val_subtypes_for_lineage:
    st_mask = adata_cd8_val.obs['subtype_label'] == st
    n_st = st_mask.sum()
    marker_pcts = []
    for mk in available_markers:
        expr = adata_cd8_val[st_mask, mk].X.flatten()
        pct = (expr > 0).sum() / n_st * 100
        marker_pcts.append(pct)
    lineage_data[st] = marker_pcts

# Grouped bar plot
x = np.arange(len(available_markers))
n_subtypes = len(val_subtypes_for_lineage)
bar_w = 0.8 / n_subtypes

for i, st in enumerate(val_subtypes_for_lineage):
    offset = (i - n_subtypes / 2 + 0.5) * bar_w
    color = merged_color_map.get(st, '#999999')
    ax_d.bar(x + offset, lineage_data[st], bar_w * 0.9,
             color=color, label=f'{st}', edgecolor='white', linewidth=0.3)

ax_d.set_xticks(x)
mk_labels_avail = [marker_labels[lineage_markers.index(m)] for m in available_markers]
ax_d.set_xticklabels(mk_labels_avail, fontsize=6, rotation=0, ha='center')
ax_d.set_ylabel('% cells expressing', fontsize=7)
ax_d.set_title('Validation: Lineage marker expression\nby CD8+ T cell subtype', fontsize=8)
ax_d.axhline(y=100, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)
ax_d.set_ylim(0, 115)
ax_d.legend(fontsize=4.5, frameon=False, ncol=2,
            bbox_to_anchor=(1.0, -0.08), loc='upper right',
            handletextpad=0.1, columnspacing=0.3)
# Print IRF8-high specific values
irf8_idx = val_subtypes_for_lineage.index('IRF8-high') if 'IRF8-high' in val_subtypes_for_lineage else None
if irf8_idx is not None:
    for j, mk in enumerate(available_markers):
        print(f"  Validation IRF8-high {mk}: {lineage_data['IRF8-high'][j]:.1f}%")

ax_d.text(-0.12, 1.05, 'D', transform=ax_d.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel E: Response association (computed from data) ────────────────────────
ax_e = fig.add_subplot(gs_s3[1, 1])

subtype_resp = {}
_resp_rows = []
for st in adata_cd8_val.obs['subtype_label'].unique():
    mask_st = adata_cd8_val.obs['subtype_label'] == st
    resp_counts = adata_cd8_val.obs.loc[mask_st, 'response_binary'].value_counts()
    n_resp = int(resp_counts.get('Responder', 0))
    n_nonresp = int(resp_counts.get('Non-responder', 0))
    st_display = st if st == 'IRF8-high' else st.replace(' (', '\n(')
    subtype_resp[st_display] = {'Resp': n_resp, 'Non-resp': n_nonresp}
    _resp_rows.append({'subtype': st, 'responder': n_resp, 'non_responder': n_nonresp,
                       'total': n_resp + n_nonresp,
                       'resp_pct': n_resp / (n_resp + n_nonresp) * 100 if (n_resp + n_nonresp) > 0 else 0})

# Fisher's exact test: IRF8-high vs Other
irf8_mask_resp = adata_cd8_val.obs['subtype_label'] == 'IRF8-high'
other_mask_resp = ~irf8_mask_resp
irf8_resp = int((adata_cd8_val.obs.loc[irf8_mask_resp, 'response_binary'] == 'Responder').sum())
irf8_nonresp = int((adata_cd8_val.obs.loc[irf8_mask_resp, 'response_binary'] == 'Non-responder').sum())
other_resp = int((adata_cd8_val.obs.loc[other_mask_resp, 'response_binary'] == 'Responder').sum())
other_nonresp = int((adata_cd8_val.obs.loc[other_mask_resp, 'response_binary'] == 'Non-responder').sum())
_, fisher_p = fisher_exact([[irf8_resp, irf8_nonresp], [other_resp, other_nonresp]], alternative='two-sided')
print(f"  S3E Fisher's exact p (IRF8-high vs Other): {fisher_p:.2e}")

resp_df = pd.DataFrame(_resp_rows).sort_values('subtype')
resp_df['fisher_p_irf8_vs_other'] = fisher_p
resp_df.to_csv(f'{OUT_DIR}/val_cd8_subtype_response_celllevel.csv', index=False)
print(f"  Cell-level response data saved to val_cd8_subtype_response_celllevel.csv")

subtypes_order = [s for s in ['Memory\n(TCF7+)', 'Effector\n(PRDM1+)', 'Cytotoxic\n(ID2+PRDM1+)',
                               'IRF8-high', 'Exhausted\n(TOX-hi)']
                  if s in subtype_resp]

resp_pcts = []
nonresp_pcts = []
for st in subtypes_order:
    total = subtype_resp[st]['Resp'] + subtype_resp[st]['Non-resp']
    resp_pcts.append(subtype_resp[st]['Resp'] / total * 100 if total > 0 else 0)
    nonresp_pcts.append(subtype_resp[st]['Non-resp'] / total * 100 if total > 0 else 0)

y_pos_d = np.arange(len(subtypes_order))
ax_e.barh(y_pos_d, resp_pcts, color='#2CA02C', label='Responder',
          edgecolor='white', linewidth=0.3)
ax_e.barh(y_pos_d, nonresp_pcts, left=resp_pcts, color='#D62728',
          label='Non-responder', edgecolor='white', linewidth=0.3)

ax_e.set_yticks(y_pos_d)
ax_e.set_yticklabels(subtypes_order, fontsize=6.5)
ax_e.set_xlabel('Percentage (%)', fontsize=7)
ax_e.set_title('CD8+ T cell subtypes by response\n(GSE120575, cell-level)', fontsize=8)
ax_e.axvline(50, color='grey', linestyle='--', linewidth=0.3, alpha=0.5)
ax_e.legend(fontsize=6, frameon=False, loc='lower right')

for idx, st in enumerate(subtypes_order):
    if st == 'IRF8-high':
        bars = [c for c in ax_e.get_children() if hasattr(c, 'get_xy')]
        for b in bars:
            xy = b.get_xy()
            if abs(xy[1] - (idx - 0.4)) < 0.1:
                b.set_edgecolor('black')
                b.set_linewidth(1.0)
        ax_e.text(50, idx + 0.42, f'p = {fisher_p:.2e}',
                  fontsize=6, color='#D62728', fontweight='bold',
                  va='bottom', ha='center')

ax_e.text(-0.15, 1.05, 'E', transform=ax_e.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel F: Patient-level IRF8-high proportion by RESPONSE (moved from S4A) ──
ax_f = fig.add_subplot(gs_s3[1, 2])

val_resp = val_pt_df[val_pt_df['response'] == 'Responder']
val_nonresp = val_pt_df[val_pt_df['response'] == 'Non-responder']

np.random.seed(42)
if len(val_resp) > 0:
    jitter = np.random.normal(0, 0.08, len(val_resp))
    ax_f.scatter(0 + jitter, val_resp['prop_irf8'].values,
               s=15, alpha=0.7, c='#2CA02C', edgecolors='white', linewidth=0.3,
               label='Responder', zorder=3)
if len(val_nonresp) > 0:
    jitter = np.random.normal(0, 0.08, len(val_nonresp))
    ax_f.scatter(1 + jitter, val_nonresp['prop_irf8'].values,
               s=15, alpha=0.7, c='#D62728', edgecolors='white', linewidth=0.3,
               label='Non-responder', zorder=3)

if len(val_resp) >= 3 and len(val_nonresp) >= 3:
    stat_r, pval_r = mannwhitneyu(val_resp['prop_irf8'].values,
                                    val_nonresp['prop_irf8'].values,
                                    alternative='two-sided')
    print(f"  S3F Patient-level response: Resp n={len(val_resp)}, Non-resp n={len(val_nonresp)}, p={pval_r:.3f}")
else:
    pval_r = None

if len(val_resp) > 0:
    ax_f.plot([-0.2, 0.2], [val_resp['prop_irf8'].mean()] * 2,
            color='#2CA02C', linewidth=1.5, zorder=4)
if len(val_nonresp) > 0:
    ax_f.plot([0.8, 1.2], [val_nonresp['prop_irf8'].mean()] * 2,
            color='#D62728', linewidth=1.5, zorder=4)

y_max_r = max(
    val_resp['prop_irf8'].max() if len(val_resp) > 0 else 0,
    val_nonresp['prop_irf8'].max() if len(val_nonresp) > 0 else 0,
) * 1.1

if pval_r is not None:
    sig_r = 'ns' if pval_r >= 0.05 else ('*' if pval_r < 0.05 else '**')
    bracket_y_r = y_max_r * 1.02
    ax_f.plot([0, 0, 1, 1], [bracket_y_r * 0.98, bracket_y_r, bracket_y_r, bracket_y_r * 0.98],
            color='black', linewidth=0.8)
    ax_f.text(0.5, bracket_y_r * 1.02, sig_r, ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax_f.text(0.5, 0.97, f'Mann–Whitney p = {pval_r:.2e}', transform=ax_f.transAxes,
            ha='center', va='top', fontsize=6.5, fontstyle='italic')

ax_f.set_xticks([0, 1])
ax_f.set_xticklabels([f'Responder\n(n={len(val_resp)})', f'Non-responder\n(n={len(val_nonresp)})'], fontsize=7)
ax_f.set_ylabel('IRF8-high proportion per patient', fontsize=7)
ax_f.set_title('Patient-level IRF8-high proportion\nby response (Validation)', fontsize=8)
ax_f.set_ylim(-0.02, y_max_r * 1.25)
ax_f.legend(fontsize=6, frameon=False, loc='upper right')

ax_f.text(-0.12, 1.05, 'F', transform=ax_f.transAxes, fontsize=12, fontweight='bold', va='top')

plt.tight_layout()
fig.savefig(f'{FIG_DIR}/FigureS3.tiff', dpi=300, format='tiff')
fig.savefig(f'{FIG_DIR}/FigureS3.pdf', dpi=300, format='pdf')
plt.close()
print("  Supplementary Figure S3 saved (now includes panel F: patient-level response).")


###############################################################################
#  SUPPLEMENTARY FIGURE S2: Pseudotime Details  (1x3)
###############################################################################
print("\n" + "=" * 70)
print("GENERATING SUPPLEMENTARY FIGURE S2: Pseudotime Details")
print("=" * 70)

fig = plt.figure(figsize=(20, 4.5))
gs_s4 = gridspec.GridSpec(1, 4, wspace=0.55, width_ratios=[1, 1, 0.6, 1.4])

# Panel A (diffusion map by subtype) removed — now shown in Fig. 2C with labels

# ── Panel A: IRF8 expression vs pseudotime scatter with regression ───────────
ax_b = fig.add_subplot(gs_s4[0, 0])

pt_vals = dpt_cell_data['dpt_pseudotime'].values
irf8_vals = dpt_cell_data['IRF8_expr'].values

# Plot cells with IRF8=0 in grey (majority), IRF8>0 in red (informative)
mask_zero = irf8_vals == 0
mask_pos = irf8_vals > 0
ax_b.scatter(pt_vals[mask_zero], irf8_vals[mask_zero], s=0.5, alpha=0.1,
             c='#CCCCCC', edgecolors='none', rasterized=True)
ax_b.scatter(pt_vals[mask_pos], irf8_vals[mask_pos], s=3, alpha=0.5,
             c='#E31A1C', edgecolors='none', rasterized=True)

# Add regression line
mask_finite = np.isfinite(pt_vals) & np.isfinite(irf8_vals)
if mask_finite.sum() > 10:
    rho, p_rho = spearmanr(pt_vals[mask_finite], irf8_vals[mask_finite])
    # Linear regression for trend line
    z = np.polyfit(pt_vals[mask_finite], irf8_vals[mask_finite], 1)
    p_line = np.poly1d(z)
    x_range = np.linspace(pt_vals[mask_finite].min(), pt_vals[mask_finite].max(), 100)
    ax_b.plot(x_range, p_line(x_range), color='black', linewidth=1.5, linestyle='--')
    ax_b.text(0.05, 0.95, f'Spearman ρ = {rho:.3f}\np = {p_rho:.2e}',
              transform=ax_b.transAxes, fontsize=6.5, fontstyle='italic',
              va='top', ha='left',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='grey'))

ax_b.set_xlabel('Pseudotime (DPT)', fontsize=7)
ax_b.set_ylabel('IRF8 expression (log1p TPM)', fontsize=7)
ax_b.set_title('IRF8 expression vs pseudotime', fontsize=8)
ax_b.text(-0.12, 1.05, 'A', transform=ax_b.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel B: Pseudotime vs key gene correlations bar ─────────────────────────
ax_c = fig.add_subplot(gs_s4[0, 1])

pt_gene_sorted = dpt_pt_gene_corr.sort_values('spearman_rho', ascending=True)

bar_colors_pt = []
for _, row in pt_gene_sorted.iterrows():
    if row['p_value'] < 0.001:
        if row['spearman_rho'] > 0:
            bar_colors_pt.append('#D73027')
        else:
            bar_colors_pt.append('#4575B4')
    elif row['p_value'] < 0.05:
        if row['spearman_rho'] > 0:
            bar_colors_pt.append('#FF7F7F')
        else:
            bar_colors_pt.append('#89CFF0')
    else:
        bar_colors_pt.append('#999999')

ax_c.barh(range(len(pt_gene_sorted)), pt_gene_sorted['spearman_rho'].values,
          color=bar_colors_pt, edgecolor='white', linewidth=0.3, height=0.7)
ax_c.set_yticks(range(len(pt_gene_sorted)))
ax_c.set_yticklabels(pt_gene_sorted['gene'].values, fontsize=7, fontstyle='italic')
ax_c.set_xlabel('Spearman ρ with pseudotime', fontsize=7)
ax_c.set_title('Gene expression vs pseudotime\ncorrelation', fontsize=8)
ax_c.axvline(0, color='grey', linestyle='--', linewidth=0.5)

legend_elements_pt = [
    Patch(facecolor='#D73027', label='Positive (p < 0.001)'),
    Patch(facecolor='#4575B4', label='Negative (p < 0.001)'),
    Patch(facecolor='#89CFF0', label='Negative (p < 0.05)'),
    Patch(facecolor='#999999', label='NS'),
]
ax_c.legend(handles=legend_elements_pt, fontsize=5, frameon=False, loc='lower right')
ax_c.text(-0.15, 1.05, 'B', transform=ax_c.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel C: Pseudotime Pre vs Post within IRF8-high cells ───────────────────
ax_d = fig.add_subplot(gs_s4[0, 2])

# Filter to IRF8-high cells using 6-subtype labels (consistent with Fig. 2C)
# Legacy dpt_cell_data['subtype_label'] uses 5-subtype system from script 14;
# remap via adata_cd8_sub to match the 6-subtype system used in Fig. 2C.
dpt_cell_data_6st_s2c = dpt_cell_data.copy()
cell_to_label6_s2c = dict(zip(adata_cd8_sub.obs.index, adata_cd8_sub.obs['subtype_label']))
dpt_cell_data_6st_s2c['subtype_label_6'] = dpt_cell_data_6st_s2c['cell'].map(cell_to_label6_s2c)
irf8_cells = dpt_cell_data_6st_s2c[dpt_cell_data_6st_s2c['subtype_label_6'] == 'IRF8-high'].copy()

# Map treatment groups
irf8_cells['treatment_simple'] = irf8_cells['treatment_group'].map({
    'treatment.naive': 'Pre-tx',
    'post.treatment': 'Post-tx',
})
# Drop any unmapped
irf8_cells = irf8_cells.dropna(subset=['treatment_simple'])

treat_pal_pt = {'Pre-tx': '#4575B4', 'Post-tx': '#D73027'}

if len(irf8_cells) > 0:
    sns.boxplot(data=irf8_cells, x='treatment_simple', y='dpt_pseudotime',
                palette=treat_pal_pt, ax=ax_d, linewidth=0.5, width=0.5,
                order=['Pre-tx', 'Post-tx'], showfliers=False)
    sns.stripplot(data=irf8_cells, x='treatment_simple', y='dpt_pseudotime',
                  palette=treat_pal_pt, ax=ax_d, size=2, alpha=0.4,
                  jitter=True, order=['Pre-tx', 'Post-tx'], rasterized=True)

    # Mann-Whitney test
    pre_pt = irf8_cells[irf8_cells['treatment_simple'] == 'Pre-tx']['dpt_pseudotime'].values
    post_pt = irf8_cells[irf8_cells['treatment_simple'] == 'Post-tx']['dpt_pseudotime'].values
    if len(pre_pt) >= 3 and len(post_pt) >= 3:
        stat_mw, pval_mw = mannwhitneyu(pre_pt, post_pt, alternative='two-sided')
        sig_mw = '***' if pval_mw < 0.001 else ('**' if pval_mw < 0.01 else ('*' if pval_mw < 0.05 else 'ns'))
        y_max_d = max(pre_pt.max(), post_pt.max())
        bracket_y = y_max_d * 1.08
        ax_d.plot([0, 0, 1, 1], [bracket_y * 0.98, bracket_y, bracket_y, bracket_y * 0.98],
                  color='black', linewidth=0.8)
        ax_d.text(0.5, bracket_y * 1.02, sig_mw, ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax_d.set_ylim(ax_d.get_ylim()[0], bracket_y * 1.18)
        ax_d.text(0.5, 0.97, f'Mann–Whitney p = {pval_mw:.2e}',
                  transform=ax_d.transAxes, ha='center', va='top',
                  fontsize=6.5, fontstyle='italic')

    # Add sample sizes
    n_pre = len(pre_pt) if len(pre_pt) > 0 else 0
    n_post = len(post_pt) if len(post_pt) > 0 else 0
    ax_d.set_xticklabels([f'Pre-tx\n(n={n_pre})', f'Post-tx\n(n={n_post})'], fontsize=7)

ax_d.set_xlabel('')
ax_d.set_ylabel('Pseudotime (DPT)', fontsize=7)
ax_d.set_title('IRF8-high pseudotime\nby treatment', fontsize=8)
ax_d.text(-0.12, 1.05, 'C', transform=ax_d.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel D: IRF8 target enrichment dot plot (moved from S3D) ────────────────
ax_e = fig.add_subplot(gs_s4[0, 3])

# Exclude "All IRF8 targets (combined)"
irf8_enr_plot = irf8_target_enr[irf8_target_enr['gene_set'] != 'All IRF8 targets (combined)'].copy()

# Compute -log10 p-values for discovery and validation
irf8_enr_plot['disc_neg_log10_p'] = -np.log10(irf8_enr_plot['disc_pvalue'].clip(lower=1e-20))
irf8_enr_plot['val_neg_log10_p'] = -np.log10(irf8_enr_plot['val_pvalue'].clip(lower=1e-20))

# Sort by discovery p-value for consistent ordering
irf8_enr_plot = irf8_enr_plot.sort_values('disc_pvalue', ascending=False).reset_index(drop=True)

gene_set_labels = irf8_enr_plot['gene_set'].str.replace('_', ' ').str.replace(' (Li et al.)', '\n(Li et al.)', regex=False).values
y_pos_enr = np.arange(len(gene_set_labels))
dot_offset = 0.15

# Discovery dots (blue)
ax_e.scatter(irf8_enr_plot['disc_neg_log10_p'].values, y_pos_enr + dot_offset,
             s=40, c='#4575B4', edgecolors='white', linewidth=0.3,
             label='Discovery', zorder=3)
# Validation dots (red)
ax_e.scatter(irf8_enr_plot['val_neg_log10_p'].values, y_pos_enr - dot_offset,
             s=40, c='#D73027', edgecolors='white', linewidth=0.3,
             label='Validation', zorder=3)

# p=0.05 reference line
ax_e.axvline(-np.log10(0.05), color='grey', linestyle='--', linewidth=0.5, label='p = 0.05')

ax_e.set_yticks(y_pos_enr)
ax_e.set_yticklabels(gene_set_labels, fontsize=6)
ax_e.set_xlabel('$-$log$_{10}$ p-value', fontsize=7)
ax_e.set_title('IRF8 target gene enrichment\nin DEGs', fontsize=8)
ax_e.legend(fontsize=5.5, frameon=False, loc='lower right')
ax_e.text(-0.15, 1.05, 'D', transform=ax_e.transAxes, fontsize=12, fontweight='bold', va='top')

plt.tight_layout()
fig.savefig(f'{FIG_DIR}/FigureS2.tiff', dpi=300, format='tiff')
fig.savefig(f'{FIG_DIR}/FigureS2.pdf', dpi=300, format='pdf')
plt.close()
print("  Supplementary Figure S2 saved.")


###############################################################################
#  SUPPLEMENTARY FIGURE S4 (2x2): Tirosh validation
#    Row 1: A=Tirosh UMAP, B=TF heatmap
#    Row 2: C=Lineage/marker heatmap (spans 2), D=Venn diagram
###############################################################################
print("\n" + "=" * 70)
print("GENERATING SUPPLEMENTARY FIGURE S4: Tirosh Validation")
print("=" * 70)

# Load Tirosh processed data (saved by 10_gse72056_validation.py)
import json
adata_cd8_tir = sc.read_h5ad(f'{OUT_DIR}/adata_cd8_tirosh.h5ad')
with open(f'{OUT_DIR}/tirosh_fig_data.json', 'r') as f:
    tir_fig = json.load(f)
disc_genes_s4 = set(tir_fig['disc_genes'])
val_genes_s4 = set(tir_fig['val_genes'])
tir_genes_s4 = set(tir_fig['tir_genes'])
concordant_3_s4 = tir_fig['concordant_3']
total_3_s4 = tir_fig['total_3']
conc_pct_s4 = tir_fig['conc_pct']
available_tfs_tir = tir_fig['available_tfs']

fig = plt.figure(figsize=(15, 9))
gs_s4m = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35,
                           width_ratios=[1, 1.4, 0.9],
                           height_ratios=[1, 1])

# ── Panel A: Tirosh UMAP ──────────────────────────────────────────────────
ax_a = fig.add_subplot(gs_s4m[0, 0])
subtypes_present_tir = sorted(adata_cd8_tir.obs['subtype_label'].unique())
for st in subtypes_present_tir:
    mask = adata_cd8_tir.obs['subtype_label'] == st
    color = merged_color_map.get(st, '#999999')
    umap_coords = adata_cd8_tir.obsm['X_umap_tf'][mask.values]
    n_st = mask.sum()
    ax_a.scatter(umap_coords[:, 0], umap_coords[:, 1],
                 c=color, s=8, alpha=0.7, label=f'{st} (n={n_st})', edgecolors='none', rasterized=True)
ax_a.set_xlabel('UMAP1', fontsize=7); ax_a.set_ylabel('UMAP2', fontsize=7)
ax_a.set_title(f'GSE72056: CD8+ T cell subtypes\n(Tirosh et al., n={adata_cd8_tir.n_obs})', fontsize=8)
ax_a.tick_params(axis='both', labelsize=6)
ax_a.legend(fontsize=5.5, frameon=False, loc='best', markerscale=1.5)
ax_a.text(-0.12, 1.05, 'A', transform=ax_a.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel B: Tirosh TF heatmap ────────────────────────────────────────────
ax_b = fig.add_subplot(gs_s4m[0, 1])
heatmap_tfs_tir = list(available_tfs_tir)
if 'TCF7' not in heatmap_tfs_tir and 'TCF7' in adata_cd8_tir.var_names:
    heatmap_tfs_tir.append('TCF7')
tf_means_tir = {}
for st in subtypes_present_tir:
    mask = adata_cd8_tir.obs['subtype_label'] == st
    tf_means_tir[st] = {tf: float(adata_cd8_tir[mask, tf].X.mean()) for tf in heatmap_tfs_tir if tf in adata_cd8_tir.var_names}
tf_df_tir = pd.DataFrame(tf_means_tir).T
tf_z_tir = (tf_df_tir - tf_df_tir.mean()) / (tf_df_tir.std() + 1e-8)
subtype_order_tir = [s for s in ['IRF8-high', 'Exhausted (TOX-hi)', 'Effector (PRDM1+)',
    'Innate-like (ID2+)', 'Cytotoxic (ID2+PRDM1+)', 'Transitional', 'Effector', 'Memory (TCF7+)']
    if s in tf_z_tir.index]
tf_z_tir = tf_z_tir.loc[subtype_order_tir]
annot_raw_tir = pd.DataFrame(tf_means_tir).T.loc[subtype_order_tir]
annot_str_tir = annot_raw_tir.round(2).astype(str)
sns.heatmap(tf_z_tir, ax=ax_b, cmap='RdBu_r', center=0, linewidths=0.3,
            annot=annot_str_tir.values, fmt='', annot_kws={'size': 5.5},
            cbar_kws={'shrink': 0.6, 'label': 'Z-score'})
ax_b.set_title('Mean TF expression by subtype\n(z-scored, raw annotated)', fontsize=8)
ax_b.set_xticklabels(ax_b.get_xticklabels(), rotation=45, ha='right', fontsize=6, fontstyle='italic')
short_names_tir = {
    'IRF8-high': 'IRF8-hi', 'Exhausted (TOX-hi)': 'Exh. TOX-hi',
    'Effector (PRDM1+)': 'Eff. PRDM1+', 'Innate-like (ID2+)': 'Innate ID2+',
    'Cytotoxic (ID2+PRDM1+)': 'Cyto. ID2+', 'Transitional': 'Trans.',
    'Effector': 'Effector', 'Memory (TCF7+)': 'Mem. TCF7+',
}
ylabels_tir = [f'{short_names_tir.get(st, st)} ({(adata_cd8_tir.obs["subtype_label"]==st).sum()})' for st in subtype_order_tir]
ax_b.set_yticklabels(ylabels_tir, fontsize=6, rotation=0)
ytick_colors_tir = [merged_color_map.get(st, '#333333') for st in subtype_order_tir]
for tick_label, color in zip(ax_b.get_yticklabels(), ytick_colors_tir):
    tick_label.set_color(color)
ax_b.text(-0.18, 1.05, 'B', transform=ax_b.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel C: Tirosh lineage + exhaustion heatmap ──────────────────────────
ax_c = fig.add_subplot(gs_s4m[1, 0:2])
lm_markers = ['CD8A', 'CD3D', 'CD163', 'CSF1R', 'CD14',
              'PDCD1', 'HAVCR2', 'TIGIT', 'LAG3', 'TOX', 'PRF1', 'XCL1', 'XCL2']
lm_available_tir = [m for m in lm_markers if m in adata_cd8_tir.var_names]
lm_pct_data_tir = {}
for st in subtype_order_tir:
    mask = adata_cd8_tir.obs['subtype_label'] == st
    n_st = mask.sum()
    pcts = [(adata_cd8_tir[mask, mk].X.flatten() > 0).sum() / n_st * 100 for mk in lm_available_tir]
    sname = short_names_tir.get(st, st)
    lm_pct_data_tir[f'{sname} ({n_st})'] = pcts
lm_df_tir = pd.DataFrame(lm_pct_data_tir, index=lm_available_tir).T
annot_pct_tir = lm_df_tir.round(1).astype(str) + '%'
sns.heatmap(lm_df_tir, ax=ax_c, cmap='YlOrRd', vmin=0, vmax=100,
            linewidths=0.5, annot=annot_pct_tir.values, fmt='',
            annot_kws={'size': 6}, cbar_kws={'shrink': 0.6, 'label': '% expressing'})
lineage_count_tir = sum(1 for m in lm_available_tir if m in ['CD8A', 'CD3D', 'CD163', 'CSF1R', 'CD14'])
ax_c.axvline(x=lineage_count_tir, color='black', linewidth=1.5)
ax_c.set_title('GSE72056: Lineage and functional marker expression by subtype', fontsize=8)
ax_c.set_xticklabels(ax_c.get_xticklabels(), rotation=45, ha='right', fontsize=7, fontstyle='italic')
for tick_label, color in zip(ax_c.get_yticklabels(), ytick_colors_tir):
    tick_label.set_color(color)
    tick_label.set_fontsize(6.5)
ax_c.text(-0.06, 1.08, 'C', transform=ax_c.transAxes, fontsize=12, fontweight='bold', va='top')

# ── Panel D: 3-cohort Venn diagram ────────────────────────────────────────
ax_d = fig.add_subplot(gs_s4m[1, 2])
try:
    from matplotlib_venn import venn3
    v = venn3([disc_genes_s4, val_genes_s4, tir_genes_s4],
              set_labels=('Discovery\n(GSE115978)', 'Validation\n(GSE120575)', 'Tirosh\n(GSE72056)'),
              ax=ax_d, set_colors=('#4575B4', '#D73027', '#33A02C'), alpha=0.5)
    for label in v.set_labels:
        if label: label.set_fontsize(6.5)
    for label in v.subset_labels:
        if label: label.set_fontsize(7)
    ax_d.set_title(f'3-cohort DEG overlap\n(FDR < 0.05; {concordant_3_s4}/{total_3_s4} concordant, {conc_pct_s4:.1f}%)', fontsize=8)
except Exception as e:
    print(f"  Venn diagram failed ({e}), using text fallback")
    ax_d.text(0.5, 0.5, f'3-cohort overlap:\n{total_3_s4} genes\n{concordant_3_s4} concordant ({conc_pct_s4:.1f}%)',
              ha='center', va='center', fontsize=8, transform=ax_d.transAxes)
    ax_d.set_title('3-cohort DEG overlap', fontsize=8)
ax_d.text(-0.12, 1.08, 'D', transform=ax_d.transAxes, fontsize=12, fontweight='bold', va='top')

plt.tight_layout()
fig.savefig(f'{FIG_DIR}/FigureS4.tiff', dpi=300, format='tiff')
fig.savefig(f'{FIG_DIR}/FigureS4.pdf', dpi=300, format='pdf')
plt.close()
print("  Supplementary Figure S4 saved (Tirosh validation, 2x2).")


###############################################################################
#  SUMMARY
###############################################################################
print("\n" + "=" * 70)
print("ALL FINAL FIGURES GENERATED")
print("=" * 70)

for f in sorted(os.listdir(FIG_DIR)):
    if f.startswith('Figure') and (f.endswith('.tiff') or f.endswith('.pdf')):
        fpath = os.path.join(FIG_DIR, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {f}: {size_mb:.1f} MB")

print("\nFigure layout summary:")
print("  Figure 1: RF-SHAP Method + TF Ranking (A: UMAP, B: F1, C: SHAP bar, D: SHAP heatmap)")
print("  Figure 2: CD8+ Discovery + Pseudotime (A: CD8 UMAP, B: TF heatmap, C: Pseudotime boxplot, D: IRF8 co-expr)")
print("  Figure 3: IRF8-high Characterization (A: Volcano disc, B: Volcano val, C: Pathway)")
print("  Figure 4: XCL1/XCL2 Functional Axis (A: XCL boxplot, B: IRF8-XCL1 scatter, C: Surface molecules)")
print("  Figure S1: TF Expression + Lineage Markers")
print("  Figure S2: Pseudotime Details (A: IRF8 vs PT, B: Gene-PT corr, C: Pre/Post PT, D: Target enrichment)")
print("  Figure S3: Validation GSE120575 (A-E) + Patient Response (F)")
print("  Figure S4: Tirosh GSE72056 (A-D)")
print("  Figure S5: Treatment Changes + Malignant (A: TF bar, B: CD8 violin, C: Patient tx, D: Mal UMAP, E: Mal heatmap, F: Mal composition)")
print("\n=== DONE ===")
