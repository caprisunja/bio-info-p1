"""Compare expression of immune genes between AD and Healthy Control.

Creates `immune_gene_comparison.csv` with per-gene statistics and saves
boxplots for each gene to `immune_figures/`.

Run: python3 immune_gene_expression_compare.py
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt


GENES = ["TNF", "IL1B", "IL6", "IL10", "IFNG", "NLRP3", "TLR4", "MYD88", "NFKB1"]


def compare_genes(data_path="Alzheimer_dataset.csv", meta_path="Alzheimer_metadata.csv",
                  out_csv="immune_gene_comparison.csv", out_dir="immune_figures"):
    p_data = Path(data_path)
    p_meta = Path(meta_path)
    if not p_data.exists() or not p_meta.exists():
        raise FileNotFoundError("Missing data or metadata CSV in repository root.")

    expr = pd.read_csv(p_data, sep=';', index_col=0, decimal=',')
    meta = pd.read_csv(p_meta, sep=';')

    # Align samples - use intersection to be safe
    sample_names = list(meta['individual'].astype(str))
    common_samples = [s for s in sample_names if s in expr.columns]
    if len(common_samples) < 1:
        raise ValueError('No matching samples between metadata and expression data.')
    if len(common_samples) < len(sample_names):
        print(f"Warning: {len(sample_names)-len(common_samples)} samples in metadata missing from expression matrix; using {len(common_samples)} samples.")

    expr = expr[common_samples]

    # Case-insensitive mapping of gene index
    index_map = {str(g).strip().upper(): g for g in expr.index}

    results = []
    os.makedirs(out_dir, exist_ok=True)

    # Per-gene comparison
    for gene in GENES:
        key = gene.strip().upper()
        present = key in index_map
        row = {
            'gene': gene,
            'present': present,
            'n_AD': 0,
            'n_HC': 0,
            'mean_AD': np.nan,
            'mean_HC': np.nan,
            'log2FC_AD_vs_HC': np.nan,
            't_stat': np.nan,
            'p_value': np.nan
        }

        if not present:
            print(f"{gene} is NOT in data set")
            results.append(row)
            continue

        print(f"{gene} is in data set")
        gene_name = index_map[key]
        vals = expr.loc[gene_name]

        ad_samples = meta[meta['group'] == "Alzheimer's Disease"]['individual'].astype(str).tolist()
        hc_samples = meta[meta['group'] == 'Healthy Control']['individual'].astype(str).tolist()

        # keep only those present in common_samples
        ad_samples = [s for s in ad_samples if s in common_samples]
        hc_samples = [s for s in hc_samples if s in common_samples]

        ad_vals = vals[ad_samples].dropna().astype(float)
        hc_vals = vals[hc_samples].dropna().astype(float)

        row['n_AD'] = len(ad_vals)
        row['n_HC'] = len(hc_vals)
        if len(ad_vals) > 0:
            row['mean_AD'] = float(ad_vals.mean())
        if len(hc_vals) > 0:
            row['mean_HC'] = float(hc_vals.mean())

        if len(ad_vals) >= 2 and len(hc_vals) >= 2:
            tstat, pval = ttest_ind(ad_vals, hc_vals, equal_var=False, nan_policy='omit')
            row['t_stat'] = float(tstat)
            row['p_value'] = float(pval)
            # compute log2 fold change AD / HC (add small constant to avoid div by zero)
            row['log2FC_AD_vs_HC'] = float(np.log2((ad_vals.mean() + 1e-8) / (hc_vals.mean() + 1e-8)))

        # Save small boxplot for this gene (if samples exist)
        if (len(ad_vals) + len(hc_vals)) > 0:
            plot_df = pd.DataFrame({
                'expression': np.concatenate([ad_vals.values, hc_vals.values]),
                'group': ['Alzheimer'] * len(ad_vals) + ['Healthy'] * len(hc_vals)
            })
            plt.figure(figsize=(6, 4))
            sns.boxplot(data=plot_df, x='group', y='expression', palette='Set2')
            sns.stripplot(data=plot_df, x='group', y='expression', color='k', alpha=0.6, jitter=True)
            plt.title(f"{gene}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{gene}_boxplot.png"), dpi=150, bbox_inches='tight')
            plt.close()

        results.append(row)

    dfres = pd.DataFrame(results).set_index('gene')
    dfres.to_csv(out_csv)
    print(f"Results written to {out_csv}; figures (if any) in {out_dir}/")
    return dfres


if __name__ == '__main__':
    try:
        compare_genes()
    except Exception as e:
        print('Error:', e)
