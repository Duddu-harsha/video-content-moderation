import pandas as pd

df = pd.read_csv("val.csv", low_memory=False)

# Count unique categories
category_count = df['Category'].nunique()
print(f"Number of different categories: {category_count}")
