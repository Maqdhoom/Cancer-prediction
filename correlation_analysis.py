import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("STEP 2Z - CORRELATION ANALYSIS")
print("=" * 70)

# Load dataset
df = pd.read_csv("data/processed/cancer_qol_raw.csv")

# Select numerical columns
numeric_df = df.select_dtypes(include=["number"])

print("\nNumerical Features:")
print(list(numeric_df.columns))

# Calculate correlation matrix
correlation = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation.round(2))

# Save correlation matrix
correlation.to_csv(
    "data/processed/correlation_matrix.csv"
)

# Create heatmap
plt.figure(figsize=(10, 8))

plt.imshow(
    correlation,
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Matrix of Numerical Features")

plt.tight_layout()

# Save graph
plt.savefig(
    "data/processed/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFiles saved:")
print("data/processed/correlation_matrix.csv")
print("data/processed/correlation_heatmap.png")

print("\n" + "=" * 70)
print("STEP 2Z COMPLETE")
print("=" * 70)
