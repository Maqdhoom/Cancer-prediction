import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("STEP 2R - ACTUAL VS PREDICTED VISUALIZATION")
print("=" * 70)

# Load final predictions
results = pd.read_csv(
    "data/processed/final_predictions.csv"
)

actual = results["Actual"]
predicted = results["Predicted"]

# Create plot
plt.figure(figsize=(8, 6))

plt.scatter(
    actual,
    predicted,
    alpha=0.7
)

# Perfect prediction reference line
minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Health Status")
plt.ylabel("Predicted Health Status")

plt.title(
    "Actual vs Predicted Health Status"
)

plt.grid(True, alpha=0.3)

# Save figure
plt.savefig(
    "data/processed/actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGraph saved to:")
print("data/processed/actual_vs_predicted.png")

print("\n" + "=" * 70)
print("STEP 2R COMPLETE")
print("=" * 70)
