import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("STEP 2AE - PREDICTION ERROR ANALYSIS")
print("=" * 70)

# Load final predictions
df = pd.read_csv("data/processed/final_predictions.csv")

# Calculate errors
df["Error"] = df["Actual"] - df["Predicted"]
df["Absolute_Error"] = df["Error"].abs()

print("\nPrediction Error Statistics:")
print(df["Error"].describe())

print("\nMean Absolute Error:")
print(df["Absolute_Error"].mean())

# ---------------------------------------------------------
# Largest prediction errors
# ---------------------------------------------------------

largest_errors = df.sort_values(
    "Absolute_Error",
    ascending=False
).head(10)

print("\nTop 10 Largest Prediction Errors:")
print(largest_errors.to_string(index=False))

# Save error analysis
df.to_csv(
    "data/processed/error_analysis_final.csv",
    index=False
)

largest_errors.to_csv(
    "data/processed/largest_prediction_errors_final.csv",
    index=False
)

# ---------------------------------------------------------
# Error Distribution
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.hist(
    df["Error"],
    bins=10,
    edgecolor="black"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Number of Patients")
plt.title("Distribution of Prediction Errors")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "data/processed/prediction_error_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFiles saved:")
print("data/processed/error_analysis_final.csv")
print("data/processed/largest_prediction_errors_final.csv")
print("data/processed/prediction_error_distribution.png")

print("\n" + "=" * 70)
print("STEP 2AE COMPLETE")
print("=" * 70)
