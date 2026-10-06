import pandas as pd

input_file = "data/raw/breast_cppo.xlsx"
output_file = "data/processed/cancer_qol_raw.csv"

df = pd.read_excel(input_file)

df.to_csv(output_file, index=False)

print("=" * 60)
print("MAIN DATASET CREATED")
print("=" * 60)
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Saved to:", output_file)
print("=" * 60)
