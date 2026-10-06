import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("STEP 2AF - FINAL FEATURE IMPORTANCE")
print("=" * 70)

# Load feature importance
df = pd.read_csv(
    "data/processed/feature_importance.csv"
)

# Sort by importance
df = df.sort_values(
    "Importance",
    ascending=False
)

# Select top 15 features
top_features = df.head(15)

print("\nTop 15 Important Features:")
print(top_features.to_string(index=False))

# Save top 15
top_features.to_csv(
    "data/processed/top_15_features_final.csv",
    index=False
)

# ---------------------------------------------------------
# Create graph
# ---------------------------------------------------------

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1],
    edgecolor="black"
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Feature Importances - Random Forest")

plt.tight_layout()

# Save graph
plt.savefig(
    "data/processed/final_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGraph saved:")
print("data/processed/final_feature_importance.png")

print("\nTop feature:")
print(top_features.iloc[0]["Feature"])

print("\n" + "=" * 70)
print("STEP 2AF COMPLETE")
print("=" * 70)
