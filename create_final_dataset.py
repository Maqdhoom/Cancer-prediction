import pandas as pd

# Load processed dataset
df = pd.read_csv("data/processed/cancer_qol_raw.csv")

# Target variable
target = "Health_status_end"

# Features = everything except target
X = df.drop(columns=[target])

# Target
y = df[target]

# Save features and target
X.to_csv("data/processed/X_features.csv", index=False)
y.to_csv("data/processed/y_target.csv", index=False)

print("=" * 70)
print("STEP 2H - FINAL ML DATASET")
print("=" * 70)

print("\nFeatures (X):")
print("Rows:", X.shape[0])
print("Columns:", X.shape[1])

print("\nTarget (y):")
print("Rows:", y.shape[0])
print("Target name:", target)

print("\nFeature columns:")
for i, column in enumerate(X.columns, start=1):
    print(f"{i}. {column}")

print("\nTarget values:")
print(y.value_counts().sort_index())

print("\nFiles created:")
print("data/processed/X_features.csv")
print("data/processed/y_target.csv")
