import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("STEP 2W - AGE DISTRIBUTION")
print("=" * 70)

# Load dataset
df = pd.read_csv("data/processed/cancer_qol_raw.csv")

# Check Age column
print("\nAge statistics:")
print(df["Age"].describe())

# Create histogram
plt.figure(figsize=(8, 6))

plt.hist(
    df["Age"].dropna(),
    bins=10,
    edgecolor="black"
)

plt.xlabel("Age")
plt.ylabel("Number of Patients")
plt.title("Age Distribution of Cancer Patients")

plt.grid(
    axis="y",
    alpha=0.3
)

# Save graph
plt.savefig(
    "data/processed/age_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGraph saved:")
print("data/processed/age_distribution.png")

print("\n" + "=" * 70)
print("STEP 2W COMPLETE")
print("=" * 70)
