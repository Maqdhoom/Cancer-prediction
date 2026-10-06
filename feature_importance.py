import pandas as pd
import joblib

print("=" * 70)
print("STEP 2P - FEATURE IMPORTANCE")
print("=" * 70)

# Load tuned pipeline
pipeline = joblib.load(
    "data/processed/tuned_random_forest_pipeline.pkl"
)

# Get preprocessing and model
preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Get Random Forest importance
importances = model.feature_importances_

# Create feature importance table
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Sort from highest to lowest
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("TOP 15 MOST IMPORTANT FEATURES")
print("=" * 70)

print(
    importance_df.head(15).to_string(index=False)
)

# Save complete feature importance
importance_df.to_csv(
    "data/processed/feature_importance.csv",
    index=False
)

# Save top 15 separately
importance_df.head(15).to_csv(
    "data/processed/top_15_features.csv",
    index=False
)

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print("1. data/processed/feature_importance.csv")
print("2. data/processed/top_15_features.csv")

print("\n" + "=" * 70)
print("STEP 2P COMPLETE")
print("=" * 70)
