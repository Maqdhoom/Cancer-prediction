import pandas as pd

file = "data/processed/cancer_qol_raw.csv"

df = pd.read_csv(file)

print("=" * 70)
print("TARGET VARIABLE INSPECTION")
print("=" * 70)

print("\nALL COLUMNS:")
for i, column in enumerate(df.columns, 1):
    print(f"{i}. {column}")

print("\n" + "=" * 70)
print("HEALTH STATUS END")
print("=" * 70)

if "Health_status_end" in df.columns:
    print("Data type:", df["Health_status_end"].dtype)
    print("Missing values:", df["Health_status_end"].isna().sum())
    print("\nUnique values:")
    print(df["Health_status_end"].value_counts(dropna=False))
else:
    print("Health_status_end column not found.")

print("\n" + "=" * 70)
print("FIRST 10 ROWS")
print("=" * 70)

print(df.head(10).to_string())

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(df.describe(include="all").T.to_string())

