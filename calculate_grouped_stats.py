import pandas as pd
import numpy as np

# Read the data files
data = pd.read_csv("Alzheimer_dataset.csv", delimiter=";", index_col=0, decimal=",")
metadata = pd.read_csv("Alzheimer_metadata.csv", delimiter=";")

# Get unique brain regions and groups
brain_regions = metadata['brainRegion'].unique()
groups = metadata['group'].unique()

# Create a results dataframe to store statistics
results_list = []

# For each combination of brain region and group
for brain_region in brain_regions:
    for group in groups:
        # Filter metadata for this combination
        subset_metadata = metadata[(metadata['brainRegion'] == brain_region) & (metadata['group'] == group)]
        
        # Get the individuals in this subset
        individuals = subset_metadata['individual'].tolist()
        
        # Skip if no individuals in this combination
        if not individuals:
            continue
        
        # Select data columns for these individuals
        subset_data = data[individuals]
        
        # Calculate mean and std dev for each gene (row)
        means = subset_data.mean(axis=1)
        stds = subset_data.std(axis=1)
        
        # Create a dataframe for this combination
        for gene in data.index:
            results_list.append({
                'Gene': gene,
                'Brain_Region': brain_region,
                'Group': group,
                'Average': means[gene],
                'Std_Dev': stds[gene]
            })

# Create final results dataframe
results_df = pd.DataFrame(results_list)

# Save to CSV
results_df.to_csv('statistics_by_group_location.csv', index=False, sep=';', decimal=',')

print("Statistics calculated and saved to 'statistics_by_group_location.csv'")
print(f"\nDataset shape: {results_df.shape}")
print(f"\nBrain regions: {list(brain_regions)}")
print(f"Groups: {list(groups)}")
print("\nFirst few rows of results:")
print(results_df.head(10))
print("\nSummary of combinations:")
summary = results_df.groupby(['Brain_Region', 'Group']).size()
print(summary)
