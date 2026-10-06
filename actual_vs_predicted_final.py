import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("STEP 2AD - ACTUAL VS PREDICTED")
print("=" * 70)

# Load final predictions
df = pd.read_csv("data/processed/final_predictions.csv")

print("\nPrediction Data:")
print(df.head())

print("\nNumber of predictions:", len(df))

# ---------------------------------------------------------
# Create Actual vs Predicted graph
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Actual"],
    df["Predicted"],
    edgecolor="black"
)

# Perfect prediction line
minimum = min(
    df["Actual"].min(),
    df["Predicted"].min()
)

maximum = max(
    df["Actual"].max(),
    df["Predicted"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Health Status")
plt.ylabel("Predicted Health Status")
plt.title("Actual vs Predicted Health Status")

plt.grid(
    alpha=0.3
)

plt.tight_layout()

# Save graph
plt.savefig(
    "data/processed/actual_vs_predicted_final.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGraph saved:")
print("data/processed/actual_vs_predicted_final.png")

print("\n" + "=" * 70)
print("STEP 2AD COMPLETE")
print("=" * 70)
