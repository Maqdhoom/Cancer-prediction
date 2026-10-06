import pandas as pd

df = pd.read_csv("data/processed/cancer_qol_raw.csv")

print("=" * 70)
print("HEALTH STATUS END - TARGET")
print("=" * 70)

print("\nUnique values:")
print(df["Health_status_end"].unique())

print("\nValue counts:")
print(df["Health_status_end"].value_counts(dropna=False))

print("\nBaseline EORTC statistics:")
print(df["EORTC_baseline"].describe())

print("\nTarget missing values:")
print(df["Health_status_end"].isna().sum())

print("\nBaseline missing values:")
print(df["EORTC_baseline"].isna().sum())

