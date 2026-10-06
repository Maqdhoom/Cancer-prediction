import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("STEP 2Y - MISSING VALUES ANALYSIS")
print("=" * 70)

# Load dataset
df = pd.read_csv("data/processed/cancer_qol_raw.csv")

# Calculate missing values
missing = df.isnull().sum()

# Calculate percentage
missing_percent = (missing / len(df)) * 100

# Create table
missing_table = pd.DataFrame({
    "Missing Values": missing,
    "Percentage": missing_percent
})

# Keep only columns with missing values
missing_table = missing_table[missing_table["Missing Values"] > 0]

print("\nMissing Value Summary:")
print(missing_table)

# Check if there are any missing values
if missing_table.empty:
    print("\nNo missing values found in the dataset.")
else:
    # Create graph
    plt.figure(figsize=(10, 6))

    plt.bar(
        missing_table.index,
        missing_table["Missing Values"],
        edgecolor="black"
    )

    plt.xlabel("Features")
    plt.ylabel("Number of Missing Values")
    plt.title("Missing Values by Feature")

    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # Save graph
    plt.savefig(
        "data/processed/missing_values.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("\nGraph saved:")
    print("data/processed/missing_values.png")

# Save table
missing_table.to_csv(
    "data/processed/missing_values.csv"
)

print("\nMissing-value table saved:")
print("data/processed/missing_values.csv")

print("\n" + "=" * 70)
print("STEP 2Y COMPLETE")
print("=" * 70)
