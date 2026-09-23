import pandas as pd
import numpy as np

def main():
    print("Loading raw dataset...")
    df = pd.read_csv("dataset/GDSC_DATASET.csv")

    print("Processing features...")
    # Select useful columns
    df_subset = df[['COSMIC_ID', 'DRUG_ID', 'Cancer Type (matching TCGA label)', 'TARGET_PATHWAY', 'LN_IC50']].copy()

    # Drop missing values
    df_subset = df_subset.dropna()

    # Convert categorical to boolean dummies, then multiply by 1 to make them float32 as expected by FNN
    cancer_type_dummies = pd.get_dummies(df_subset['Cancer Type (matching TCGA label)'], prefix='CancerType').astype('float32')
    pathway_dummies = pd.get_dummies(df_subset['TARGET_PATHWAY'], prefix='Pathway').astype('float32')

    # Combine features
    features_df = pd.concat([df_subset[['COSMIC_ID', 'DRUG_ID']], cancer_type_dummies, pathway_dummies, df_subset['LN_IC50']], axis=1)

    # We need to make sure the indices are strings or integers properly, but pandas to_csv handles it.
    features_df.set_index(['COSMIC_ID', 'DRUG_ID'], inplace=True)
    
    # Subsample to make training feasible for a toy example. Otherwise it's hundreds of thousands of rows.
    # Let's just keep the full dataset, FNN.py uses batch sizes, it might take a while though. 
    # Let's take a 10% sample so it runs quickly for the user to monitor loss.
    print(f"Original shape: {features_df.shape}")
    features_df = features_df.sample(frac=0.1, random_state=42)
    print(f"Sampled shape: {features_df.shape}")

    # Save to TSV
    output_path = "dataset/processed_features.txt"
    print(f"Saving to {output_path}...")
    features_df.to_csv(output_path, sep='\t')
    print("Done!")

if __name__ == '__main__':
    main()
