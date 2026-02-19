import pandas as pd
import numpy as np


def gene_stats_by_group_and_region(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    individual_col: str = "individual",
    group_col: str = "group",
    region_col: str = "brainRegion",
    require_all_individuals: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    data:     index = genen, columns = individuals, values = expressie (float)
    metadata: per individual minstens kolommen: individual, group, brainRegion

    returns:
      (stats_group, stats_group_region)
    """

    # --- checks ---
    for c in [individual_col, group_col, region_col]:
        if c not in metadata.columns:
            raise ValueError(f"metadata mist kolom '{c}'. Gevonden: {list(metadata.columns)}")

    # zorg dat individual strings zijn (voor matchen)
    meta = metadata[[individual_col, group_col, region_col]].copy()
    meta[individual_col] = meta[individual_col].astype(str).str.strip()

    # data columns ook strings
    data_cols = pd.Index([str(c).strip() for c in data.columns])
    data = data.copy()
    data.columns = data_cols

    # match individuals
    meta_inds = set(meta[individual_col])
    data_inds = set(data.columns)

    missing_in_data = sorted(meta_inds - data_inds)
    missing_in_meta = sorted(data_inds - meta_inds)

    if require_all_individuals and missing_in_data:
        raise ValueError(f"Individuals wel in metadata maar niet in data: {missing_in_data[:10]}{'...' if len(missing_in_data)>10 else ''}")

    if require_all_individuals and missing_in_meta:
        raise ValueError(f"Individuals wel in data maar niet in metadata: {missing_in_meta[:10]}{'...' if len(missing_in_meta)>10 else ''}")

    # Neem alleen de kolommen die ook in metadata staan (veilig, ook als require_all_individuals=False)
    use_cols = [c for c in data.columns if c in meta_inds]
    data_use = data[use_cols]

    # --- long format (robust) ---
    long = data_use.stack().rename("expression").reset_index()

    # long heeft nu meestal kolommen: [<gene-index-col>, <individual-col>, 'expression']
    # Hernoem altijd de eerste twee kolommen expliciet:
    long = long.rename(columns={long.columns[0]: "gene", long.columns[1]: individual_col})

    # merge metadata
    long = long.merge(meta, on=individual_col, how="left")

    if long[group_col].isna().any() or long[region_col].isna().any():
        raise ValueError("Sommige individuals uit data konden niet gematcht worden met metadata (group/brainRegion is NaN).")

    # --- aggregaties ---
    stats_group = (
        long.groupby(["gene", group_col])["expression"]
            .agg(mean="mean", sd="std", n="count")
            .reset_index()
            .sort_values(["gene", group_col])
    )

    stats_group_region = (
        long.groupby(["gene", group_col, region_col])["expression"]
            .agg(mean="mean", sd="std", n="count")
            .reset_index()
            .sort_values(["gene", group_col, region_col])
    )

    return stats_group, stats_group_region


data = pd.read_csv("Alzheimer_dataset.csv", delimiter=";", index_col=0, decimal=",")
metadata = pd.read_csv("Alzheimer_metadata.csv", delimiter=";")

stats_group, stats_group_region = gene_stats_by_group_and_region(data, metadata)

stats_group.head()  #gemiddelde expressie per groep (AD vs HC)
stats_group_region.head() #gemiddelde expressie per groep en brain region (AD vs HC, per brain region)

#print(stats_group.head())
# Maak wide tabel per gene, met kolommen voor elke groep
wide = stats_group.pivot(index="gene", columns="group", values="mean")
wide["difference"] = (
    wide["Alzheimer's Disease"] - wide["Healthy Control"]
)
# Bereken absolute verschil
wide["abs_difference"] = wide["difference"].abs()

top5 = (
    wide.sort_values("abs_difference", ascending=False)
        .head(5)
)



# Maak wide tabel per gene, met kolommen voor elke groep, maar nu alleen voor de hippocampus
hippo = stats_group_region[
    stats_group_region["brainRegion"] == "hippocampal formation"
]
wide_hippo = hippo.pivot(
    index="gene",
    columns="group",
    values="mean"
)

wide_hippo["difference"] = (
    wide_hippo["Alzheimer's Disease"] -
    wide_hippo["Healthy Control"]
)

wide_hippo["abs_difference"] = wide_hippo["difference"].abs()

top5_hippo = (
    wide_hippo
        .sort_values("abs_difference", ascending=False)
        .head(5)
)

#print(top5_hippo)

metadata["brainRegion"].unique()


#print(top5)

#statistiek en cijfers over aantal samples per groep en brain region
#print(stats_group["group"].value_counts())
#print(metadata["group"].value_counts())
#print(metadata["brainRegion"].value_counts())
#print(stats_group_region.groupby(["group","brainRegion"])["n"].max())
