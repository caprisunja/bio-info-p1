"""Simple gene-presence checker for Alzheimer_dataset.csv

Usage: run `python3 immune.py` from the repository root. Prints for each
gene whether it is present in the dataset index (case-insensitive).
"""

from pathlib import Path
import pandas as pd


GENES = ["TNF", "IL1B", "IL6", "IL10", "IFNG", "NLRP3", "TLR4", "MYD88", "NFKB1"]


def check_genes(data_path: str = "Alzheimer_dataset.csv"):
    """Check presence of genes in the dataset index and print status.

    Prints lines like: "TNF is in data set" or "TNF is NOT in data set".
    Returns a dict mapping gene -> bool.
    """
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Read only the index (first column) to avoid loading full matrix when large
    df = pd.read_csv(p, sep=";", index_col=0, decimal=",")

    # Index may contain whitespace/case differences — compare case-insensitive
    index_map = {str(i).strip().upper(): i for i in df.index}

    results = {}
    for g in GENES:
        present = g.strip().upper() in index_map
        results[g] = present
        if present:
            print(f"{g} is in data set")
        else:
            print(f"{g} is NOT in data set")

    return results


if __name__ == "__main__":
    try:
        check_genes()
    except Exception as e:
        print("Error:", e)