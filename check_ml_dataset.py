import pandas as pd

# Load final features and target
X = pd.read_csv("data/processed/X_features.csv")
y = pd.read_csv("data/processed/y_target.csv")

print("=" * 70)
print("STEP 2I - ML DATASET CHECK")
print("=" * 70)

print("\nDataset shape:")
print("X:", X.shape)
print("y:", y.shape)

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)
print(X.dtypes)

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)
print(X.isnull().sum())

print("\nTotal missing values:", X.isnull().sum().sum())

print("\n" + "=" * 70)
print("NUMERIC FEATURES")
print("=" * 70)

numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()

for column in numeric_columns:
    print("-", column)

print("\nNumber of numeric features:", len(numeric_columns))

print("\n" + "=" * 70)
print("CATEGORICAL FEATURES")
print("=" * 70)

categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()

for column in categorical_columns:
    print("-", column)

print("\nNumber of categorical features:", len(categorical_columns))

print("\n" + "=" * 70)
print("UNIQUE VALUES OF CATEGORICAL FEATURES")
print("=" * 70)

for column in categorical_columns:
    print(f"\n{column}:")
    print(X[column].value_counts(dropna=False))

print("\n" + "=" * 70)
print("STEP 2I COMPLETE")
print("=" * 70)
