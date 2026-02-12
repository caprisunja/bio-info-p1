import pandas as pd
import numpy as np

# Read the data file
df = pd.read_csv('Alzheimer_dataset.csv', index_col=0, sep=';', decimal=',')

# Calculate mean and standard deviation for each row
results = pd.DataFrame({
    'Gene': df.index,
    'Average': df.mean(axis=1),
    'Std_Dev': df.std(axis=1)
})

# Set Gene as the index
results.set_index('Gene', inplace=True)

# Save to a new file
results.to_csv('statistics_results.csv', decimal=',')

print("Statistics calculated and saved to 'statistics_results.csv'")
print("\nFirst few rows of results:")
print(results.head())
