import pandas as pd
import numpy as np

print("=" * 70)
print("STEP 2S - ERROR ANALYSIS")
print("=" * 70)

# Load final predictions
results = pd.read_csv(
    "data/processed/final_predictions.csv"
)

# Calculate errors
results["Error"] = results["Actual"] - results["Predicted"]

results["Absolute_Error"] = np.abs(
    results["Error"]
)

results["Squared_Error"] = (
    results["Error"] ** 2
)

# Sort by largest absolute error
largest_errors = results.sort_values(
    by="Absolute_Error",
    ascending=False
)

print("\n" + "=" * 70)
print("TOP 10 LARGEST PREDICTION ERRORS")
print("=" * 70)

print(
    largest_errors[
        [
            "Actual",
            "Predicted",
            "Error",
            "Absolute_Error"
        ]
    ].head(10).to_string(index=False)
)

# Error statistics
print("\n" + "=" * 70)
print("ERROR STATISTICS")
print("=" * 70)

print(
    f"Mean Absolute Error : "
    f"{results['Absolute_Error'].mean():.4f}"
)

print(
    f"Maximum Error       : "
    f"{results['Absolute_Error'].max():.4f}"
)

print(
    f"Minimum Error       : "
    f"{results['Absolute_Error'].min():.4f}"
)

print(
    f"Median Error        : "
    f"{results['Absolute_Error'].median():.4f}"
)

# Count predictions by error range
print("\n" + "=" * 70)
print("ERROR RANGES")
print("=" * 70)

print(
    "Error <= 5:",
    (results["Absolute_Error"] <= 5).sum()
)

print(
    "Error 5-10:",
    (
        (results["Absolute_Error"] > 5) &
        (results["Absolute_Error"] <= 10)
    ).sum()
)

print(
    "Error 10-20:",
    (
        (results["Absolute_Error"] > 10) &
        (results["Absolute_Error"] <= 20)
    ).sum()
)

print(
    "Error > 20:",
    (results["Absolute_Error"] > 20).sum()
)

# Save complete error analysis
results.to_csv(
    "data/processed/error_analysis.csv",
    index=False
)

# Save top 10
largest_errors.head(10).to_csv(
    "data/processed/largest_prediction_errors.csv",
    index=False
)

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "1. data/processed/error_analysis.csv"
)

print(
    "2. data/processed/largest_prediction_errors.csv"
)

print("\n" + "=" * 70)
print("STEP 2S COMPLETE")
print("=" * 70)
