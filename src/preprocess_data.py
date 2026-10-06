import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

print("=" * 70)
print("STEP 2J - DATA PREPROCESSING")
print("=" * 70)

# Load features and target
X = pd.read_csv("data/processed/X_features.csv")
y = pd.read_csv("data/processed/y_target.csv").squeeze()

print("\nOriginal X shape:", X.shape)

# Identify numeric and categorical columns
numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()

print("\nNumeric columns:")
for column in numeric_columns:
    print("-", column)

print("\nCategorical columns:")
for column in categorical_columns:
    print("-", column)

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns)
    ]
)

# Transform features
X_processed = preprocessor.fit_transform(X)

# Get feature names after encoding
feature_names = preprocessor.get_feature_names_out()

# Convert to DataFrame
X_processed = pd.DataFrame(
    X_processed,
    columns=feature_names
)

# Save processed dataset
X_processed.to_csv(
    "data/processed/X_processed.csv",
    index=False
)

# Save target again
y.to_csv(
    "data/processed/y_final.csv",
    index=False,
    header=["Health_status_end"]
)

# Save preprocessing object
joblib.dump(
    preprocessor,
    "data/processed/preprocessor.pkl"
)

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print("\nOriginal features:", X.shape[1])
print("Processed features:", X_processed.shape[1])
print("Rows:", X_processed.shape[0])

print("\nProcessed dataset preview:")
print(X_processed.head())

print("\nFiles created:")
print("1. data/processed/X_processed.csv")
print("2. data/processed/y_final.csv")
print("3. data/processed/preprocessor.pkl")
