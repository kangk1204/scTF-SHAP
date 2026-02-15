"""
Machine Learning-based Identification of Key Transcription Factors
Determining Cell Identity in Melanoma Tumor Microenvironment

Strategy:
- Use ALL cell types (malignant + immune + stromal) from GSE115978
- RF classifier using TF expression → SHAP to rank identity-determining TFs per cell type
- FOXP1 as positive control (should rank top for T cells)
- Identify novel TFs for each cell type including malignant cells
- Validate with an independent dataset
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import shap
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
DATA_DIR = os.path.join(BASE_DIR, 'data')

plt.rcParams.update({
    'font.size': 7,
    'font.family': 'Arial',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.5,
})

###############################################################################
# STEP 1: Comprehensive human TF gene list
###############################################################################
print("=" * 60)
print("STEP 1: Defining human transcription factor gene list")
print("=" * 60)

tf_families = {
    'FOX': ['FOXP1', 'FOXP3', 'FOXO1', 'FOXO3', 'FOXO4', 'FOXA1', 'FOXA2',
            'FOXM1', 'FOXN1', 'FOXJ1', 'FOXK1', 'FOXK2', 'FOXL1', 'FOXL2',
            'FOXC1', 'FOXC2', 'FOXD1', 'FOXD3', 'FOXE1', 'FOXF1', 'FOXF2',
            'FOXG1', 'FOXH1', 'FOXI1', 'FOXP2', 'FOXP4', 'FOXQ1', 'FOXR1',
            'FOXS1'],
    'TBX': ['TBX21', 'EOMES', 'TBX1', 'TBX3', 'TBX5', 'TBX6', 'TBX15',
            'TBX18', 'TBX19', 'TBX20', 'BRACHYURY'],
    'bHLH': ['MYC', 'MYCN', 'MYCL', 'MAX', 'TCF3', 'TCF4', 'TCF12',
             'HIF1A', 'ARNT', 'AHR', 'CLOCK',
             'ID1', 'ID2', 'ID3', 'ID4', 'TWIST1', 'TWIST2',
             'HEY1', 'HEY2', 'HES1', 'HES5', 'SNAI1', 'SNAI2',
             'ASCL1', 'ASCL2', 'NEUROD1', 'NEUROG1',
             'OLIG1', 'OLIG2', 'BHLHE40', 'BHLHE41'],
    'ZNF': ['GATA1', 'GATA2', 'GATA3', 'GATA4', 'GATA5', 'GATA6',
            'KLF2', 'KLF4', 'KLF5', 'KLF6', 'KLF7', 'KLF9', 'KLF10', 'KLF13',
            'EGR1', 'EGR2', 'EGR3', 'EGR4',
            'ZEB1', 'ZEB2', 'PRDM1', 'PRDM2', 'PRDM5',
            'IKZF1', 'IKZF2', 'IKZF3', 'IKZF4', 'IKZF5',
            'ZNF683', 'ZNF281', 'ZNF148', 'ZNF652',
            'WT1', 'GLI1', 'GLI2', 'GLI3',
            'CTCF', 'YY1', 'SP1', 'SP3', 'SP4', 'SP100',
            'REST', 'SNAI1', 'SNAI2'],
    'NR': ['NR4A1', 'NR4A2', 'NR4A3', 'NR3C1', 'RORA', 'RORC', 'PPARG',
           'PPARA', 'PPARD', 'RXRA', 'RARA', 'RARB', 'RARG', 'VDR',
           'ESR1', 'ESR2', 'AR', 'PGR',
           'HNF4A', 'NR1H3', 'NR1H4', 'NR2F1', 'NR2F2', 'NR2F6',
           'NR5A1', 'NR5A2', 'THRB', 'THRA'],
    'STAT': ['STAT1', 'STAT2', 'STAT3', 'STAT4', 'STAT5A', 'STAT5B', 'STAT6'],
    'IRF': ['IRF1', 'IRF2', 'IRF3', 'IRF4', 'IRF5', 'IRF7', 'IRF8', 'IRF9'],
    'NFKB': ['NFKB1', 'NFKB2', 'RELA', 'RELB', 'REL'],
    'AP1': ['FOS', 'FOSB', 'FOSL1', 'FOSL2', 'JUN', 'JUNB', 'JUND',
            'BATF', 'BATF2', 'BATF3', 'ATF1', 'ATF2', 'ATF3', 'ATF4',
            'ATF5', 'ATF6', 'ATF7'],
    'RUNX': ['RUNX1', 'RUNX2', 'RUNX3', 'CBFB'],
    'SOX': ['SOX2', 'SOX4', 'SOX5', 'SOX6', 'SOX7', 'SOX8', 'SOX9',
            'SOX10', 'SOX11', 'SOX13', 'SOX17', 'SOX18', 'SOX21',
            'SRY'],
    'ETS': ['ETS1', 'ETS2', 'ELF1', 'ELF4', 'ELF5', 'ELK1', 'ELK3', 'ELK4',
            'SPI1', 'SPIB', 'SPIC', 'FLI1', 'ERG', 'ETV1', 'ETV4', 'ETV5', 'ETV6',
            'GABPA', 'GABPB1', 'FEV', 'EHF', 'ESE1'],
    'NFAT': ['NFATC1', 'NFATC2', 'NFATC3', 'NFATC4', 'NFAT5'],
    'HOX': ['PAX5', 'PAX6', 'PAX3', 'PAX7', 'PAX8',
            'PBX1', 'PBX3', 'MEIS1', 'MEIS2',
            'HOXA9', 'HOXA10', 'HOXB4', 'HOXC8', 'HOXD10',
            'LEF1', 'TCF7', 'TCF7L1', 'TCF7L2',
            'CDX1', 'CDX2', 'DLX1', 'DLX5', 'MSX1', 'MSX2',
            'LHX1', 'LHX2'],
    'OTHER': ['TP53', 'TP63', 'TP73', 'MYB', 'MYBL1', 'MYBL2',
              'E2F1', 'E2F2', 'E2F3', 'E2F4', 'E2F5',
              'RB1', 'CEBPA', 'CEBPB', 'CEBPD', 'CEBPE', 'CEBPG',
              'BACH1', 'BACH2', 'MAFB', 'MAFK', 'MAFF', 'MAFG', 'MAF',
              'NFE2', 'NFE2L2', 'NFE2L3',
              'SMAD1', 'SMAD2', 'SMAD3', 'SMAD4', 'SMAD5', 'SMAD7',
              'NOTCH1', 'NOTCH2', 'NOTCH3',
              'BCL6', 'BCL11A', 'BCL11B', 'TOX', 'TOX2', 'TOX3',
              'YBX1', 'YBX3', 'HMGA1', 'HMGA2', 'HMGB1', 'HMGB2',
              'MXD1', 'MXD3', 'MXD4', 'MGA', 'MNT',
              'ZFP36', 'ZFP36L1', 'ZFP36L2',
              'TFEB', 'TFE3', 'MITF', 'USF1', 'USF2',
              'XBP1', 'CREB1', 'CREB3', 'CREB5',
              'EBF1', 'MEF2A', 'MEF2B', 'MEF2C', 'MEF2D',
              'TEAD1', 'TEAD2', 'TEAD3', 'TEAD4',
              'WWTR1', 'YAP1', 'HIPK2',
              'SREBF1', 'SREBF2', 'MLX', 'MLXIP',
              'HSF1', 'HSF2'],
}

all_tfs = list(set(tf for genes in tf_families.values() for tf in genes))
print(f"Total TF genes defined: {len(all_tfs)}")

###############################################################################
# STEP 2: Load ALL cell types (including malignant)
###############################################################################
print("\n" + "=" * 60)
print("STEP 2: Loading all cell types from GSE115978")
print("=" * 60)

# Rebuild from TPM with all cells
annot = pd.read_csv(f'{DATA_DIR}/GSE115978_cell.annotations.csv.gz')
annot = annot.set_index('cells')

print("Loading TPM matrix...")
tpm = pd.read_csv(f'{DATA_DIR}/GSE115978_tpm.csv.gz', index_col=0)
tpm_T = tpm.T

adata_all = sc.AnnData(X=tpm_T.values.astype(np.float32))
adata_all.obs_names = tpm_T.index.tolist()
adata_all.var_names = tpm_T.columns.tolist()

common_cells = adata_all.obs_names.intersection(annot.index)
adata_all = adata_all[common_cells, :].copy()
for col in annot.columns:
    adata_all.obs[col] = annot.loc[adata_all.obs_names, col].values

# Exclude unclassified cells ('?')
adata_all = adata_all[adata_all.obs['cell.types'] != '?'].copy()

print(f"\nAll cells (excluding unclassified): {adata_all.n_obs}")
print(f"Cell types: {adata_all.obs['cell.types'].value_counts().to_dict()}")

# Log-transform
sc.pp.log1p(adata_all)

###############################################################################
# STEP 3: RF classifier using TF expression for ALL cell types
###############################################################################
print("\n" + "=" * 60)
print("STEP 3: Random Forest classification (all cell types)")
print("=" * 60)

tfs_in_data = [tf for tf in all_tfs if tf in adata_all.var_names]
print(f"TFs in dataset: {len(tfs_in_data)}")

tf_expr = pd.DataFrame(
    adata_all[:, tfs_in_data].X,
    index=adata_all.obs_names,
    columns=tfs_in_data
)

# Filter low-expression TFs
min_pct = 0.03
tf_pct = (tf_expr > 0).mean()
tfs_expressed = tf_pct[tf_pct >= min_pct].index.tolist()
print(f"TFs after filtering (>3% expression): {len(tfs_expressed)}")

tf_expr_filtered = tf_expr[tfs_expressed]
X = tf_expr_filtered.values

cell_types = adata_all.obs['cell.types'].values
le = LabelEncoder()
y = le.fit_transform(cell_types)

print(f"Feature matrix: {X.shape}")
print(f"Classes: {list(le.classes_)}")

# Cross-validation
rf = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_leaf=3,
                            random_state=42, n_jobs=-1, class_weight='balanced')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='f1_weighted')
print(f"\n5-fold CV F1 (weighted): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Train final model
rf.fit(X, y)

# Print report
y_pred = rf.predict(X)
report = classification_report(y, y_pred, target_names=le.classes_, output_dict=True)
print("\nClassification Report:")
for cls in le.classes_:
    r = report[cls]
    print(f"  {cls}: F1={r['f1-score']:.3f} (prec={r['precision']:.3f}, rec={r['recall']:.3f})")

###############################################################################
# STEP 4: SHAP analysis - Global and per-cell-type
###############################################################################
print("\n" + "=" * 60)
print("STEP 4: SHAP analysis for identity-determining TFs")
print("=" * 60)

explainer = shap.TreeExplainer(rf)

# Subsample for SHAP
np.random.seed(42)
n_sample = min(2000, X.shape[0])
sample_idx = np.random.choice(X.shape[0], n_sample, replace=False)
X_sample = X[sample_idx]
y_sample = y[sample_idx]

print(f"Computing SHAP values for {n_sample} cells...")
shap_values_raw = explainer.shap_values(X_sample)

# Handle different SHAP output formats
# shap_values can be: list of arrays (old API) or 3D array (samples x features x classes)
if isinstance(shap_values_raw, list):
    # Old API: list of (n_samples, n_features) arrays, one per class
    shap_values_list = shap_values_raw
    n_classes = len(shap_values_list)
elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3:
    # New API: (n_samples, n_features, n_classes)
    n_classes = shap_values_raw.shape[2]
    shap_values_list = [shap_values_raw[:, :, i] for i in range(n_classes)]
else:
    # Fallback: try treating as (n_samples, n_features) for binary
    shap_values_list = [shap_values_raw]
    n_classes = 1

print(f"  SHAP format: {n_classes} classes, shape per class: {shap_values_list[0].shape}")

# Global importance
shap_abs_mean = np.zeros(X.shape[1])
for class_shap in shap_values_list:
    shap_abs_mean += np.abs(class_shap).mean(axis=0)
shap_abs_mean /= n_classes

tf_importance = pd.DataFrame({
    'TF': tfs_expressed,
    'mean_abs_shap': shap_abs_mean,
    'rf_importance': rf.feature_importances_
}).sort_values('mean_abs_shap', ascending=False)
tf_importance['rank'] = range(1, len(tf_importance) + 1)
tf_importance.to_csv(f'{OUT_DIR}/tf_importance_global.csv', index=False)

print("\n=== Top 30 Global Identity-Determining TFs ===")
print(tf_importance.head(30)[['rank', 'TF', 'mean_abs_shap', 'rf_importance']].to_string(index=False))

# Per-cell-type importance
print("\n=== Cell-Type-Specific Top TFs ===")
celltype_top_tfs = {}
for i, cls in enumerate(le.classes_):
    if i < len(shap_values_list):
        class_shap = shap_values_list[i]
    else:
        continue
    class_imp = pd.DataFrame({
        'TF': tfs_expressed,
        'mean_shap': class_shap.mean(axis=0),
        'mean_abs_shap': np.abs(class_shap).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    celltype_top_tfs[cls] = class_imp
    class_imp.to_csv(f'{OUT_DIR}/tf_importance_{cls.replace(".", "_")}.csv', index=False)

    print(f"\n--- {cls} ---")
    for j, (_, row) in enumerate(class_imp.head(10).iterrows()):
        direction = "+" if row['mean_shap'] > 0 else "-"
        marker = " *** FOXP1" if row['TF'] == 'FOXP1' else ""
        print(f"  {j+1:2d}. {row['TF']:10s} SHAP={row['mean_abs_shap']:.4f} ({direction}){marker}")

###############################################################################
# STEP 5: Treatment-associated TF changes (all cells & CD8+ T cells)
###############################################################################
print("\n" + "=" * 60)
print("STEP 5: Treatment-associated TF expression changes")
print("=" * 60)

top_tfs = tf_importance.head(30)['TF'].tolist()

# All cells
treatment_changes_all = []
for tf in top_tfs:
    if tf in adata_all.var_names:
        naive_expr = adata_all[adata_all.obs['treatment.group'] == 'treatment.naive'][:, tf].X.flatten()
        treated_expr = adata_all[adata_all.obs['treatment.group'] == 'post.treatment'][:, tf].X.flatten()
        stat, pval = stats.mannwhitneyu(naive_expr, treated_expr, alternative='two-sided')
        # log2FC = (mean_log1p_post - mean_log1p_pre) / ln(2)
        # Equivalent to log2(geometric_mean_post / geometric_mean_pre) for TPM+1
        log2_fc = (treated_expr.mean() - naive_expr.mean()) / np.log(2)
        treatment_changes_all.append({
            'TF': tf, 'naive_mean': naive_expr.mean(), 'treated_mean': treated_expr.mean(),
            'log2_fc': log2_fc, 'pvalue': pval
        })

df_treat_all = pd.DataFrame(treatment_changes_all)
if len(df_treat_all) > 0:
    _, padj, _, _ = multipletests(df_treat_all['pvalue'], method='fdr_bh')
    df_treat_all['padj'] = padj
    df_treat_all = df_treat_all.sort_values('pvalue')
    df_treat_all.to_csv(f'{OUT_DIR}/tf_treatment_changes_all.csv', index=False)
    print("\nTreatment-associated TF changes (all cells, top 15):")
    print(df_treat_all.head(15).to_string(index=False))

# CD8+ T cells specifically
adata_cd8 = adata_all[adata_all.obs['cell.types'] == 'T.CD8'].copy()
cd8_changes = []
for tf in top_tfs:
    if tf in adata_cd8.var_names:
        naive_expr = adata_cd8[adata_cd8.obs['treatment.group'] == 'treatment.naive'][:, tf].X.flatten()
        treated_expr = adata_cd8[adata_cd8.obs['treatment.group'] == 'post.treatment'][:, tf].X.flatten()
        if len(naive_expr) > 10 and len(treated_expr) > 10:
            stat, pval = stats.mannwhitneyu(naive_expr, treated_expr, alternative='two-sided')
            # Correct log2FC: difference of log1p means divided by ln(2)
            log2_fc = (treated_expr.mean() - naive_expr.mean()) / np.log(2)
            cd8_changes.append({
                'TF': tf, 'naive_mean': naive_expr.mean(), 'treated_mean': treated_expr.mean(),
                'log2_fc': log2_fc, 'pvalue': pval
            })

df_cd8_changes = pd.DataFrame(cd8_changes)
if len(df_cd8_changes) > 0:
    _, padj, _, _ = multipletests(df_cd8_changes['pvalue'], method='fdr_bh')
    df_cd8_changes['padj'] = padj
    df_cd8_changes = df_cd8_changes.sort_values('pvalue')
    df_cd8_changes.to_csv(f'{OUT_DIR}/cd8_tf_treatment_changes.csv', index=False)
    print("\nCD8+ T cell TF changes (top 15):")
    print(df_cd8_changes.head(15).to_string(index=False))

print("\n=== ANALYSIS COMPLETE ===")
