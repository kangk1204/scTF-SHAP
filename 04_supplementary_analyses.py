"""
04_supplementary_analyses.py
Supplementary analyses to strengthen the paper:
1. IRF8-high CD8+ T cell doublet/macrophage contamination check
2. Patient-level pseudoreplication analysis for IRF8-high enrichment
3. Both analyses performed in discovery (GSE115978) and validation (GSE120575) cohorts
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import mannwhitneyu, fisher_exact
import os
import gzip

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)
DATA_DIR = os.path.join(BASE_DIR, 'data')

###############################################################################
# PART 1: DISCOVERY COHORT (GSE115978) - Doublet/Contamination Check
###############################################################################
print("=" * 70)
print("PART 1: DISCOVERY COHORT - IRF8-high Doublet/Contamination Check")
print("=" * 70)

# Load discovery data
print("\nLoading discovery cohort data...")
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

# Load CD8 subtypes from h5ad
adata_cd8_sub = sc.read_h5ad(f'{OUT_DIR}/adata_cd8_subtypes.h5ad')

# Re-cluster CD8 T cells with same parameters as 07 script
tf_cd8 = pd.read_csv(f'{OUT_DIR}/tf_importance_T_CD8.csv')
top_cd8_tfs = tf_cd8.head(15)['TF'].tolist()
available_cd8_tfs = [tf for tf in top_cd8_tfs if tf in adata_cd8_sub.var_names]

adata_cd8_tf = adata_cd8_sub[:, available_cd8_tfs].copy()
sc.pp.scale(adata_cd8_tf, max_value=10)
sc.tl.pca(adata_cd8_tf, n_comps=10, random_state=42)
sc.pp.neighbors(adata_cd8_tf, n_pcs=10, n_neighbors=15, random_state=42)
sc.tl.leiden(adata_cd8_tf, resolution=0.25, key_added='tf_subtype_v2', random_state=42)

adata_cd8_sub.obs['tf_subtype_v2'] = adata_cd8_tf.obs['tf_subtype_v2'].values

# Assign labels (same logic as 07)
cd8_sub_stats = {}
for st in sorted(adata_cd8_sub.obs['tf_subtype_v2'].unique()):
    mask = adata_cd8_sub.obs['tf_subtype_v2'] == st
    cd8_sub_stats[st] = {
        'n': mask.sum(),
        'tox': adata_cd8_sub[mask, 'TOX'].X.mean() if 'TOX' in adata_cd8_sub.var_names else 0,
        'irf8': adata_cd8_sub[mask, 'IRF8'].X.mean() if 'IRF8' in adata_cd8_sub.var_names else 0,
        'tcf7': adata_cd8_sub[mask, 'TCF7'].X.mean() if 'TCF7' in adata_cd8_sub.var_names else 0,
        'prdm1': adata_cd8_sub[mask, 'PRDM1'].X.mean() if 'PRDM1' in adata_cd8_sub.var_names else 0,
        'id2': adata_cd8_sub[mask, 'ID2'].X.mean() if 'ID2' in adata_cd8_sub.var_names else 0,
        'post_pct': (adata_cd8_sub.obs.loc[mask, 'treatment.group'] != 'treatment.naive').mean() * 100,
    }
    # Signature scores
    exhaustion_genes = ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'TOX', 'ENTPD1']
    effector_genes = ['GZMA', 'GZMB', 'GZMK', 'PRF1', 'IFNG', 'NKG7', 'GNLY']
    memory_genes = ['TCF7', 'LEF1', 'CCR7', 'SELL', 'IL7R']
    for sig_name, sig_genes in [('exhaust', exhaustion_genes), ('effector', effector_genes), ('memory', memory_genes)]:
        avail = [g for g in sig_genes if g in adata_cd8_sub.var_names]
        if avail:
            cd8_sub_stats[st][sig_name] = adata_cd8_sub[mask][:, avail].X.mean()

cd8_labels_v2 = {}
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
    cd8_labels_v2[st] = label

adata_cd8_sub.obs['subtype_label'] = adata_cd8_sub.obs['tf_subtype_v2'].map(cd8_labels_v2)

# Now check macrophage/myeloid markers in IRF8-high vs other CD8+ subtypes
# We need the full expression data for these markers
# adata_cd8_sub was saved from 01 script with all genes
print("\n--- Macrophage/Myeloid Marker Check in CD8+ T cell subtypes ---")

# Key markers to check
mac_markers = ['CD68', 'CD163', 'CSF1R', 'LYZ', 'AIF1', 'ITGAM', 'CD14', 'FCGR3A']
t_cell_markers = ['CD8A', 'CD8B', 'CD3D', 'CD3E', 'CD3G', 'CD2', 'CD7']
all_markers = mac_markers + t_cell_markers

available_markers = [m for m in all_markers if m in adata_cd8_sub.var_names]
missing_markers = [m for m in all_markers if m not in adata_cd8_sub.var_names]
print(f"Available markers: {available_markers}")
print(f"Missing markers: {missing_markers}")

# For each subtype, compute mean expression and % expressing
irf8_mask = adata_cd8_sub.obs['subtype_label'] == 'IRF8-high'
other_mask = ~irf8_mask

print(f"\nIRF8-high cells: n={irf8_mask.sum()}")
print(f"Other CD8+ T cells: n={other_mask.sum()}")

print(f"\n{'Marker':<10} {'IRF8-hi mean':>13} {'IRF8-hi %>0':>12} {'Other mean':>12} {'Other %>0':>10} {'p-value':>12} {'Interp.':>10}")
print("-" * 85)

contamination_results = {}
for marker in available_markers:
    irf8_expr = adata_cd8_sub[irf8_mask, marker].X.flatten()
    other_expr = adata_cd8_sub[other_mask, marker].X.flatten()

    irf8_mean = irf8_expr.mean()
    other_mean = other_expr.mean()
    irf8_pct = (irf8_expr > 0).mean() * 100
    other_pct = (other_expr > 0).mean() * 100

    stat, pval = mannwhitneyu(irf8_expr, other_expr, alternative='two-sided')

    marker_type = 'Mac' if marker in mac_markers else 'T cell'

    print(f"{marker:<10} {irf8_mean:>13.4f} {irf8_pct:>11.1f}% {other_mean:>12.4f} {other_pct:>9.1f}% {pval:>12.2e} {marker_type:>10}")

    contamination_results[marker] = {
        'irf8_mean': irf8_mean, 'other_mean': other_mean,
        'irf8_pct': irf8_pct, 'other_pct': other_pct,
        'pval': pval, 'type': marker_type
    }

# Doublet assessment summary
print("\n--- Doublet Assessment Summary (Discovery) ---")
mac_detected = False
for marker in mac_markers:
    if marker in contamination_results:
        r = contamination_results[marker]
        if r['irf8_pct'] > 10:
            mac_detected = True
            print(f"  WARNING: {marker} expressed in {r['irf8_pct']:.1f}% of IRF8-high cells")

if not mac_detected:
    print("  No significant macrophage marker expression detected in IRF8-high cells")

# CD8/T-cell marker check
print("\n  T-cell marker verification:")
for marker in t_cell_markers:
    if marker in contamination_results:
        r = contamination_results[marker]
        if r['irf8_pct'] > 80:
            status = "CONFIRMED"
        elif r['irf8_pct'] > 50:
            status = "present"
        else:
            status = "LOW"
        print(f"  {marker}: IRF8-hi={r['irf8_pct']:.1f}% expressing (mean={r['irf8_mean']:.3f}) → {status}")


###############################################################################
# PART 2: DISCOVERY COHORT - Patient-level Pseudoreplication Analysis
###############################################################################
print("\n" + "=" * 70)
print("PART 2: DISCOVERY COHORT - Patient-level Analysis")
print("=" * 70)

# Get patient info for CD8+ T cells
# patient column may be in different names
print("\nAvailable obs columns:", list(adata_cd8_sub.obs.columns[:20]))

# Check for patient column
patient_col = None
for col in ['patients', 'patient', 'samples', 'sample']:
    if col in adata_cd8_sub.obs.columns:
        patient_col = col
        break

if patient_col is None:
    # Try from the original annotation
    print("Patient column not in CD8 subtypes, checking original annotations...")
    # We need to match cells back to adata_all which has patient annotations
    cd8_cells = adata_cd8_sub.obs_names
    # adata_all has annot columns
    if 'patients' in adata_all.obs.columns:
        patient_col_src = 'patients'
    elif 'patient' in adata_all.obs.columns:
        patient_col_src = 'patient'
    elif 'samples' in adata_all.obs.columns:
        patient_col_src = 'samples'
    else:
        # Check what columns exist
        print("  adata_all columns:", list(adata_all.obs.columns))
        patient_col_src = None

    if patient_col_src:
        common_cd8 = cd8_cells.intersection(adata_all.obs_names)
        adata_cd8_sub.obs['patient'] = 'Unknown'
        adata_cd8_sub.obs.loc[common_cd8, 'patient'] = adata_all.obs.loc[common_cd8, patient_col_src].values
        patient_col = 'patient'
        print(f"  Mapped {len(common_cd8)} cells to patient IDs from '{patient_col_src}'")

if patient_col and patient_col in adata_cd8_sub.obs.columns:
    patients = adata_cd8_sub.obs[patient_col].unique()
    print(f"\nTotal patients with CD8+ T cells: {len(patients)}")

    # Get treatment group for each patient
    treat_col = 'treatment.group'

    # Compute per-patient IRF8-high proportion
    patient_data = []
    for pt in patients:
        pt_mask = adata_cd8_sub.obs[patient_col] == pt
        n_total = pt_mask.sum()
        if n_total < 5:  # Skip patients with very few cells
            continue
        n_irf8 = (adata_cd8_sub.obs.loc[pt_mask, 'subtype_label'] == 'IRF8-high').sum()
        prop_irf8 = n_irf8 / n_total

        # Get treatment group
        treat_vals = adata_cd8_sub.obs.loc[pt_mask, treat_col].unique()
        treat = 'Post' if any('naive' not in str(t) for t in treat_vals if 'naive' not in str(t)) else 'Naive'
        # Better: check majority
        is_naive = (adata_cd8_sub.obs.loc[pt_mask, treat_col] == 'treatment.naive').mean()
        treat = 'Naive' if is_naive > 0.5 else 'Post'

        patient_data.append({
            'patient': pt,
            'treatment': treat,
            'n_cd8': n_total,
            'n_irf8': n_irf8,
            'prop_irf8': prop_irf8,
        })

    pt_df = pd.DataFrame(patient_data)
    print(f"\nPatients with ≥5 CD8+ T cells: {len(pt_df)}")
    print(f"  Naive: {(pt_df['treatment'] == 'Naive').sum()}")
    print(f"  Post-tx: {(pt_df['treatment'] == 'Post').sum()}")

    # Patient-level comparison
    naive_props = pt_df[pt_df['treatment'] == 'Naive']['prop_irf8'].values
    post_props = pt_df[pt_df['treatment'] == 'Post']['prop_irf8'].values

    print(f"\nIRF8-high proportion per patient:")
    print(f"  Naive (n={len(naive_props)}): mean={naive_props.mean():.4f}, median={np.median(naive_props):.4f}")
    print(f"  Post-tx (n={len(post_props)}): mean={post_props.mean():.4f}, median={np.median(post_props):.4f}")

    if len(naive_props) >= 3 and len(post_props) >= 3:
        stat_pt, pval_pt = mannwhitneyu(naive_props, post_props, alternative='two-sided')
        print(f"  Wilcoxon rank-sum (patient-level): p = {pval_pt:.4e}")

        # Also try Fisher's exact on patient counts: patients with >0 IRF8-high cells
        naive_has_irf8 = (naive_props > 0).sum()
        naive_no_irf8 = (naive_props == 0).sum()
        post_has_irf8 = (post_props > 0).sum()
        post_no_irf8 = (post_props == 0).sum()

        table_pt = [[naive_has_irf8, post_has_irf8], [naive_no_irf8, post_no_irf8]]
        odds_pt, pval_pt_fisher = fisher_exact(table_pt)
        print(f"  Fisher's exact (patients with IRF8-high cells): OR={odds_pt:.3f}, p={pval_pt_fisher:.4e}")
        print(f"    Naive: {naive_has_irf8}/{len(naive_props)} patients have IRF8-high cells")
        print(f"    Post-tx: {post_has_irf8}/{len(post_props)} patients have IRF8-high cells")
    else:
        print("  Too few patients in one group for statistical testing")

    # Print per-patient details
    print(f"\nPer-patient details:")
    print(f"{'Patient':<20} {'Treatment':<8} {'N_CD8':>6} {'N_IRF8':>6} {'Prop':>7}")
    print("-" * 55)
    for _, row in pt_df.sort_values(['treatment', 'prop_irf8']).iterrows():
        print(f"{str(row['patient']):<20} {row['treatment']:<8} {row['n_cd8']:>6} {row['n_irf8']:>6} {row['prop_irf8']:>7.3f}")
else:
    print("  WARNING: Could not find patient column")


###############################################################################
# PART 3: VALIDATION COHORT (GSE120575) - Doublet Check + Patient-level
###############################################################################
print("\n" + "=" * 70)
print("PART 3: VALIDATION COHORT (GSE120575) - Doublet & Patient-level")
print("=" * 70)

# Load validation data
print("\nLoading validation cohort data...")
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

# Build annotation
annot_data = []
for i, cid in enumerate(cell_ids_val):
    pt_info = patient_treat_info[i] if i < len(patient_treat_info) else 'Unknown'
    treatment = 'Pre' if str(pt_info).startswith('Pre') else ('Post' if str(pt_info).startswith('Post') else 'Unknown')
    # Extract patient ID from pt_info (format: Pre_PatientID or Post_PatientID)
    parts = str(pt_info).split('_', 1)
    patient_id = parts[1] if len(parts) > 1 else pt_info
    annot_data.append({
        'cell_id': cid,
        'patient_treatment': pt_info,
        'treatment': treatment,
        'patient_id': patient_id,
    })

annot_val = pd.DataFrame(annot_data, index=cell_ids_val)

# Build AnnData
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
print(f"CD8+ T cells in validation: {adata_cd8_val.n_obs}")

# TF-based subclustering
tf_cd8_discovery = pd.read_csv(f'{OUT_DIR}/tf_importance_T_CD8.csv')
top_cd8_tfs_disc = tf_cd8_discovery.head(15)['TF'].tolist()
available_tfs_val = [tf for tf in top_cd8_tfs_disc if tf in adata_cd8_val.var_names]

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

adata_cd8_val.obs['subtype_label'] = adata_cd8_val.obs['tf_subtype'].map(val_labels)

# Doublet check in validation
print("\n--- Macrophage Marker Check in Validation CD8+ T cells ---")

irf8_mask_val = adata_cd8_val.obs['subtype_label'] == 'IRF8-high'
other_mask_val = ~irf8_mask_val

print(f"IRF8-high cells (validation): n={irf8_mask_val.sum()}")
print(f"Other CD8+ T cells (validation): n={other_mask_val.sum()}")

mac_markers_val = [m for m in mac_markers if m in adata_cd8_val.var_names]
t_cell_markers_val = [m for m in t_cell_markers if m in adata_cd8_val.var_names]
all_markers_val = mac_markers_val + t_cell_markers_val

print(f"\n{'Marker':<10} {'IRF8-hi mean':>13} {'IRF8-hi %>0':>12} {'Other mean':>12} {'Other %>0':>10} {'p-value':>12}")
print("-" * 75)

contamination_results_val = {}
for marker in all_markers_val:
    irf8_expr_v = adata_cd8_val[irf8_mask_val, marker].X.flatten()
    other_expr_v = adata_cd8_val[other_mask_val, marker].X.flatten()

    irf8_mean_v = irf8_expr_v.mean()
    other_mean_v = other_expr_v.mean()
    irf8_pct_v = (irf8_expr_v > 0).mean() * 100
    other_pct_v = (other_expr_v > 0).mean() * 100

    stat_v, pval_v = mannwhitneyu(irf8_expr_v, other_expr_v, alternative='two-sided')

    print(f"{marker:<10} {irf8_mean_v:>13.4f} {irf8_pct_v:>11.1f}% {other_mean_v:>12.4f} {other_pct_v:>9.1f}% {pval_v:>12.2e}")

    contamination_results_val[marker] = {
        'irf8_mean': irf8_mean_v, 'other_mean': other_mean_v,
        'irf8_pct': irf8_pct_v, 'other_pct': other_pct_v,
        'pval': pval_v,
    }

# Patient-level analysis for validation
print("\n--- Patient-level Analysis (Validation) ---")

val_patients = adata_cd8_val.obs['patient_id'].unique()
print(f"\nTotal patients in validation with CD8+ T cells: {len(val_patients)}")

patient_data_val = []
for pt in val_patients:
    pt_mask = adata_cd8_val.obs['patient_id'] == pt
    n_total = pt_mask.sum()
    if n_total < 5:
        continue
    n_irf8 = (adata_cd8_val.obs.loc[pt_mask, 'subtype_label'] == 'IRF8-high').sum()
    prop_irf8 = n_irf8 / n_total

    is_pre = (adata_cd8_val.obs.loc[pt_mask, 'treatment'] == 'Pre').mean()
    treat = 'Pre' if is_pre > 0.5 else 'Post'

    patient_data_val.append({
        'patient': pt,
        'treatment': treat,
        'n_cd8': n_total,
        'n_irf8': n_irf8,
        'prop_irf8': prop_irf8,
    })

pt_val_df = pd.DataFrame(patient_data_val)
print(f"Patients with ≥5 CD8+ T cells: {len(pt_val_df)}")
print(f"  Pre: {(pt_val_df['treatment'] == 'Pre').sum()}")
print(f"  Post: {(pt_val_df['treatment'] == 'Post').sum()}")

pre_props_val = pt_val_df[pt_val_df['treatment'] == 'Pre']['prop_irf8'].values
post_props_val = pt_val_df[pt_val_df['treatment'] == 'Post']['prop_irf8'].values

print(f"\nIRF8-high proportion per patient (validation):")
print(f"  Pre (n={len(pre_props_val)}): mean={pre_props_val.mean():.4f}, median={np.median(pre_props_val):.4f}")
print(f"  Post (n={len(post_props_val)}): mean={post_props_val.mean():.4f}, median={np.median(post_props_val):.4f}")

if len(pre_props_val) >= 3 and len(post_props_val) >= 3:
    stat_pt_val, pval_pt_val = mannwhitneyu(pre_props_val, post_props_val, alternative='two-sided')
    print(f"  Wilcoxon rank-sum (patient-level): p = {pval_pt_val:.4e}")

    # Fisher's exact
    pre_has = (pre_props_val > 0).sum()
    pre_no = (pre_props_val == 0).sum()
    post_has = (post_props_val > 0).sum()
    post_no = (post_props_val == 0).sum()

    table_val = [[pre_has, post_has], [pre_no, post_no]]
    odds_val, pval_val_fisher = fisher_exact(table_val)
    print(f"  Fisher's exact (patients with IRF8-high): OR={odds_val:.3f}, p={pval_val_fisher:.4e}")
    print(f"    Pre: {pre_has}/{len(pre_props_val)} patients have IRF8-high cells")
    print(f"    Post: {post_has}/{len(post_props_val)} patients have IRF8-high cells")

print(f"\nPer-patient details (validation):")
print(f"{'Patient':<30} {'Treatment':<8} {'N_CD8':>6} {'N_IRF8':>6} {'Prop':>7}")
print("-" * 65)
for _, row in pt_val_df.sort_values(['treatment', 'prop_irf8']).iterrows():
    print(f"{str(row['patient']):<30} {row['treatment']:<8} {row['n_cd8']:>6} {row['n_irf8']:>6} {row['prop_irf8']:>7.3f}")


###############################################################################
# FINAL SUMMARY
###############################################################################
print("\n" + "=" * 70)
print("FINAL SUMMARY FOR MANUSCRIPT")
print("=" * 70)

print("\n1. DOUBLET/CONTAMINATION CHECK:")
print("   Discovery cohort:")
for marker in ['CD68', 'CD163', 'CSF1R', 'LYZ']:
    if marker in contamination_results:
        r = contamination_results[marker]
        print(f"     {marker}: IRF8-hi {r['irf8_pct']:.1f}% expressing (mean={r['irf8_mean']:.3f}) vs Other {r['other_pct']:.1f}% (mean={r['other_mean']:.3f})")
for marker in ['CD8A', 'CD3D']:
    if marker in contamination_results:
        r = contamination_results[marker]
        print(f"     {marker}: IRF8-hi {r['irf8_pct']:.1f}% expressing (mean={r['irf8_mean']:.3f}) vs Other {r['other_pct']:.1f}% (mean={r['other_mean']:.3f})")

print("\n   Validation cohort:")
for marker in ['CD68', 'CD163', 'CSF1R', 'LYZ']:
    if marker in contamination_results_val:
        r = contamination_results_val[marker]
        print(f"     {marker}: IRF8-hi {r['irf8_pct']:.1f}% expressing (mean={r['irf8_mean']:.3f}) vs Other {r['other_pct']:.1f}% (mean={r['other_mean']:.3f})")
for marker in ['CD8A', 'CD3D']:
    if marker in contamination_results_val:
        r = contamination_results_val[marker]
        print(f"     {marker}: IRF8-hi {r['irf8_pct']:.1f}% expressing (mean={r['irf8_mean']:.3f}) vs Other {r['other_pct']:.1f}% (mean={r['other_mean']:.3f})")

print("\n2. PATIENT-LEVEL ANALYSIS:")
if len(naive_props) >= 3 and len(post_props) >= 3:
    print(f"   Discovery (n={len(pt_df)} patients): Naive mean prop={naive_props.mean():.4f}, Post mean prop={post_props.mean():.4f}, p={pval_pt:.4e}")
if len(pre_props_val) >= 3 and len(post_props_val) >= 3:
    print(f"   Validation (n={len(pt_val_df)} patients): Pre mean prop={pre_props_val.mean():.4f}, Post mean prop={post_props_val.mean():.4f}, p={pval_pt_val:.4e}")

print("\n=== ANALYSIS COMPLETE ===")
