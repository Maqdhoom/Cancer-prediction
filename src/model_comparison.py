import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("STEP 2AA - MODEL COMPARISON")
print("=" * 70)

# Load final evaluation results
metrics_file = "data/processed/final_metrics.csv"

try:
    df = pd.read_csv(metrics_file)

    print("\nModel Comparison:")
    print(df.to_string(index=False))

except FileNotFoundError:
    print("\nfinal_metrics.csv not found.")
    print("Creating comparison using the available model results...")

    df = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest"
        ],
        "MAE": [
            17.5670,
            16.0936
        ],
        "RMSE": [
            22.8075,
            20.4322
        ],
        "R2": [
            -0.2046,
            0.0333
        ]
    })

    print("\nAvailable Model Results:")
    print(df.to_string(index=False))

# -------------------------------
# Save comparison table
# -------------------------------
df.to_csv(
    "data/processed/model_comparison.csv",
    index=False
)

# -------------------------------
# Plot MAE
# -------------------------------
if "MAE" in df.columns:

    plt.figure(figsize=(8, 6))

    plt.bar(
        df["Model"],
        df["MAE"],
        edgecolor="black"
    )

    plt.xlabel("Model")
    plt.ylabel("MAE")
    plt.title("Model Comparison - MAE")

    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "data/processed/model_comparison_mae.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

# -------------------------------
# Plot RMSE
# -------------------------------
if "RMSE" in df.columns:

    plt.figure(figsize=(8, 6))

    plt.bar(
        df["Model"],
        df["RMSE"],
        edgecolor="black"
    )

    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.title("Model Comparison - RMSE")

    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "data/processed/model_comparison_rmse.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

# -------------------------------
# Plot R²
# -------------------------------
if "R2" in df.columns:

    plt.figure(figsize=(8, 6))

    plt.bar(
        df["Model"],
        df["R2"],
        edgecolor="black"
    )

    plt.xlabel("Model")
    plt.ylabel("R² Score")
    plt.title("Model Comparison - R²")

    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        "data/processed/model_comparison_r2.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

print("\nFiles saved:")
print("data/processed/model_comparison.csv")
print("data/processed/model_comparison_mae.png")
print("data/processed/model_comparison_rmse.png")
print("data/processed/model_comparison_r2.png")

print("\n" + "=" * 70)
print("STEP 2AA COMPLETE")
print("=" * 70)
