import numpy as np
import pandas as pd

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

print("start")

# -----------------------------
# Helpers
# -----------------------------

def find_group_columns(groups: list[str]) -> tuple[str, str]:
    """
    Vind automatisch de AD- en Healthy groepsnaam, ook als de exacte spelling
    verschilt (quotes/spaties).
    """
    ad = [g for g in groups if "Alzheimer" in g]
    hc = [g for g in groups if "Healthy" in g]

    if len(ad) != 1 or len(hc) != 1:
        raise ValueError(f"Kan AD/Healthy niet uniek vinden in groepen: {groups}")

    return ad[0], hc[0]


def top5_by_effect(df: pd.DataFrame) -> pd.DataFrame:
    """
    Top 5 genen op grootste absolute log2FC.
    """
    return (
        df.reindex(df["log2FC_AD_vs_Healthy"].abs().sort_values(ascending=False).index)
          .head(5)[["gene", "region", "log2FC_AD_vs_Healthy", "p_value", "q_value_FDR"]]
    )


# -----------------------------
# Differential Expression
# -----------------------------

def differential_expression(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    region: str | None = None,
    group_col: str = "group",
    region_col: str = "brainRegion",
    individual_col: str = "individual",
    auto_log2p1: bool = True,
    min_n_per_group: int = 3,
) -> pd.DataFrame:
    """
    Per gen: mean/sd/n per groep, log2FC, Welch t-test p-value, FDR (BH),
    Cohen's d. Werkt voor hele brein (region=None) en per regio.

    data: index = gene, columns = individuals
    metadata: columns o.a. [individual, group, brainRegion]
    """

    # --- metadata checks + cleaning ---
    meta = metadata.copy()
    for c in [group_col, region_col, individual_col]:
        if c not in meta.columns:
            raise ValueError(f"metadata mist kolom '{c}'. Kolommen: {list(meta.columns)}")
    meta[individual_col] = meta[individual_col].astype(str).str.strip()

    # --- data column cleaning ---
    data = data.copy()
    data.columns = pd.Index([str(c).strip() for c in data.columns])

    # --- filter region if requested ---
    if region is not None:
        meta = meta[meta[region_col] == region].copy()
        if meta.empty:
            raise ValueError(
                f"Geen samples gevonden voor region='{region}'. "
                f"Unieke regions: {metadata[region_col].unique()}"
            )

    # --- overlap individuals ---
    overlap = sorted(set(meta[individual_col]).intersection(set(data.columns)))
    if len(overlap) == 0:
        raise ValueError("Geen overlap tussen data columns en metadata individuals.")

    meta = meta[meta[individual_col].isin(overlap)].copy()
    data = data[overlap]

    # --- optional log2(+1) transform ---
    # Heuristiek: als max waarde heel hoog is, dan is log transform vaak zinvol.
    if auto_log2p1:
        mx = np.nanmax(data.to_numpy())
        if mx > 50:
            data = np.log2(data + 1)

    # --- group names ---
    groups = meta[group_col].unique().tolist()
    ad_name, hc_name = find_group_columns(groups)

    ad_inds = meta.loc[meta[group_col] == ad_name, individual_col].tolist()
    hc_inds = meta.loc[meta[group_col] == hc_name, individual_col].tolist()

    if len(ad_inds) < min_n_per_group or len(hc_inds) < min_n_per_group:
        raise ValueError(
            f"Te weinig samples per groep. AD={len(ad_inds)}, Healthy={len(hc_inds)}, min={min_n_per_group}"
        )

    X_ad = data[ad_inds]
    X_hc = data[hc_inds]

    # --- summary stats ---
    mean_ad = X_ad.mean(axis=1)
    sd_ad   = X_ad.std(axis=1)
    n_ad    = X_ad.count(axis=1)

    mean_hc = X_hc.mean(axis=1)
    sd_hc   = X_hc.std(axis=1)
    n_hc    = X_hc.count(axis=1)

    # In log-space is log2FC = mean difference
    log2fc = mean_ad - mean_hc

    # --- Welch t-test (per gene) ---
    t_res = ttest_ind(
        X_ad.to_numpy(),
        X_hc.to_numpy(),
        axis=1,
        equal_var=False,
        nan_policy="omit",
    )
    pvals = t_res.pvalue

    # --- FDR correction (Benjamini-Hochberg) ---
    qvals = multipletests(pvals, alpha=0.05, method="fdr_bh")[1]

    # --- Cohen's d ---
    pooled_sd = np.sqrt(((n_ad - 1) * (sd_ad ** 2) + (n_hc - 1) * (sd_hc ** 2)) / (n_ad + n_hc - 2))
    cohens_d = (mean_ad - mean_hc) / pooled_sd.replace(0, np.nan)

    print("middle  ")

    # --- result dataframe ---
    res = pd.DataFrame({
        "gene": data.index,
        "region": region if region is not None else "ALL",
        "group_AD": ad_name,
        "group_Healthy": hc_name,
        "mean_AD": mean_ad.values,
        "sd_AD": sd_ad.values,
        "n_AD": n_ad.values.astype(int),
        "mean_Healthy": mean_hc.values,
        "sd_Healthy": sd_hc.values,
        "n_Healthy": n_hc.values.astype(int),
        "log2FC_AD_vs_Healthy": log2fc.values,
        "p_value": pvals,
        "q_value_FDR": qvals,
        "cohens_d": cohens_d.values,
    })

    # sort: most significant first
    res = res.sort_values(["q_value_FDR", "p_value"], ascending=True).reset_index(drop=True)
    return res

print("end")




# -----------------------------
# Main: load data + run analyses
# -----------------------------

if __name__ == "__main__":
    # Load
    data = pd.read_csv("Alzheimer_dataset.csv", delimiter=";", index_col=0, decimal=",")
    metadata = pd.read_csv("Alzheimer_metadata.csv", delimiter=";")

    # Check regions
    regions = metadata["brainRegion"].unique()
    print("Regions:", regions)

    # Whole brain
    print("\nRunning whole-brain differential expression...")
    de_all = differential_expression(data, metadata, region=None, auto_log2p1=True)

    # Per region
    de_by_region = {}
    for r in regions:
        print(f"\nRunning region: {r}")
        de_by_region[r] = differential_expression(data, metadata, region=r, auto_log2p1=True)

    # Top 5 effects
    print("\n=== Top 5 by absolute effect (Whole Brain) ===")
    print(top5_by_effect(de_all))

    for r in regions:
        print(f"\n=== Top 5 by absolute effect ({r}) ===")
        print(top5_by_effect(de_by_region[r]))

    # Significant counts
    print("\n=== # Significant genes (FDR < 0.05) ===")
    print("Whole brain:", (de_all["q_value_FDR"] < 0.05).sum())

    for r in regions:
        print(f"{r}:", (de_by_region[r]["q_value_FDR"] < 0.05).sum())

    # (Optional) Save results
    '''de_all.to_csv("DE_whole_brain.csv", index=False)
    for r in regions:
        safe_name = r.replace(" ", "_")
        de_by_region[r].to_csv(f"DE_{safe_name}.csv", index=False) '''

    print("\nDone. Files saved: DE_whole_brain.csv and DE_<region>.csv")

import numpy as np
import matplotlib.pyplot as plt

def volcano_plot(
    df,
    title="Volcano plot",
    p_col="p_value",
    fc_col="log2FC_AD_vs_Healthy",
    q_col="q_value_FDR",
    fdr_thresh=0.05,
    p_thresh=0.05,
    fc_thresh=1.0,
    use_fdr_for_highlight=True,
    top_n_labels=10
):
    """
    Volcano plot: x = log2FC, y = -log10(p)
    Highlight: (FDR < fdr_thresh) AND (abs(log2FC) >= fc_thresh)  [default]
    Labels: top_n_labels most significant highlighted points (lowest q or p)
    """

    x = df[fc_col].to_numpy()
    p = df[p_col].to_numpy()

    # bescherm tegen p=0
    p = np.clip(p, 1e-300, 1.0)
    y = -np.log10(p)

    plt.figure(figsize=(9, 6))
    plt.scatter(x, y, s=10)

    # drempellijnen
    plt.axvline(fc_thresh, linestyle="--")
    plt.axvline(-fc_thresh, linestyle="--")
    plt.axhline(-np.log10(p_thresh), linestyle="--")

    # highlight mask
    if use_fdr_for_highlight and (q_col in df.columns):
        mask = (df[q_col] < fdr_thresh) & (np.abs(df[fc_col]) >= fc_thresh)
    else:
        mask = (df[p_col] < p_thresh) & (np.abs(df[fc_col]) >= fc_thresh)

    # highlight punten bovenop tekenen (zelfde kleur is oké; als je kleur wil, zeg het)
    plt.scatter(df.loc[mask, fc_col], -np.log10(np.clip(df.loc[mask, p_col], 1e-300, 1.0)), s=18)

    # labels (optioneel)
    if top_n_labels and top_n_labels > 0:
        if use_fdr_for_highlight and (q_col in df.columns):
            label_df = df.loc[mask].sort_values(q_col).head(top_n_labels)
        else:
            label_df = df.loc[mask].sort_values(p_col).head(top_n_labels)

        for _, row in label_df.iterrows():
            plt.text(row[fc_col], -np.log10(max(row[p_col], 1e-300)), str(row["gene"]), fontsize=8)

    plt.title(title)
    plt.xlabel("log2 Fold Change (AD vs Healthy)")
    plt.ylabel("-log10(p-value)")
    plt.show()

volcano_plot(de_all, title="Whole brain: AD vs Healthy", fc_thresh=1.0, fdr_thresh=0.05)

for r, df in de_by_region.items():
    volcano_plot(df, title=f"{r}: AD vs Healthy", fc_thresh=1.0, fdr_thresh=0.05)
