import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("STEP 2AC - FINAL MODEL COMPARISON")
print("=" * 70)

# Model results
results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Voting Regressor"
    ],
    "MAE": [
        17.5670,
        16.0936,
        16.1924
    ],
    "RMSE": [
        22.8075,
        20.4322,
        20.5568
    ],
    "R2": [
        -0.2046,
        0.0333,
        0.0214
    ]
})

print("\nFINAL MODEL COMPARISON")
print("-" * 70)
print(results.to_string(index=False))

# Save comparison table
results.to_csv(
    "data/processed/final_model_comparison.csv",
    index=False
)

# Find best model
best_mae = results.loc[results["MAE"].idxmin(), "Model"]
best_rmse = results.loc[results["RMSE"].idxmin(), "Model"]
best_r2 = results.loc[results["R2"].idxmax(), "Model"]

print("\nBest Model by MAE  :", best_mae)
print("Best Model by RMSE :", best_rmse)
print("Best Model by R²   :", best_r2)

# ---------------------------------------------------------
# Create comparison graph
# ---------------------------------------------------------
x = range(len(results))

plt.figure(figsize=(9, 6))

plt.bar(
    [i - 0.25 for i in x],
    results["MAE"],
    width=0.25,
    label="MAE",
    edgecolor="black"
)

plt.bar(
    x,
    results["RMSE"],
    width=0.25,
    label="RMSE",
    edgecolor="black"
)

plt.bar(
    [i + 0.25 for i in x],
    results["R2"],
    width=0.25,
    label="R²",
    edgecolor="black"
)

plt.xticks(
    x,
    results["Model"],
    rotation=15
)

plt.xlabel("Model")
plt.ylabel("Score")
plt.title("Final Model Comparison")

plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "data/processed/final_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFiles saved:")
print("data/processed/final_model_comparison.csv")
print("data/processed/final_model_comparison.png")

print("\n" + "=" * 70)
print("STEP 2AC COMPLETE")
print("=" * 70)
