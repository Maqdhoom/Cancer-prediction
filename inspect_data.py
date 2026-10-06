import pandas as pd
from pathlib import Path

DATA_FOLDER = Path("data/raw")

files = [
    "breast_cppo.xlsx",
    "breast_qlq_c30.xlsx",
    "breast_scores.xlsx"
]

print("=" * 70)
print("DATASET INSPECTION")
print("=" * 70)

for file in files:
    path = DATA_FOLDER / file

    print("\n" + "=" * 70)
    print(f"FILE: {file}")
    print("=" * 70)

    df = pd.read_excel(path)

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\nColumn names:")
    for column in df.columns:
        print(" -", column)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)
