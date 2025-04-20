import pandas as pd
import os

# Load the dataset safely
df = pd.read_csv("val.csv", low_memory=False)

# Check if 'Category' column exists
if 'Category' not in df.columns:
    print("❌ 'Category' column not found in CSV.")
    print("Available columns:", df.columns.tolist())
    exit()

# Group by 'Category' and pick 5 samples from each
df_sampled = df.groupby('Category').head(5)

# Save the output
output_file = "val_sampled_5_per_category.csv"
df_sampled.to_csv(output_file, index=False)

# Confirm save
print(f"✅ Output saved to: {os.path.abspath(output_file)}")
print(f"Total sampled rows: {len(df_sampled)}")
