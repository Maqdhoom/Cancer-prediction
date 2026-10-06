import pandas as pd
import matplotlib.pyplot as plt

print("=" * 70)
print("STEP 2X - HEALTH STATUS DISTRIBUTION")
print("=" * 70)

# Load dataset
df = pd.read_csv("data/processed/cancer_qol_raw.csv")

# -------------------------------
# Health Status at End
# -------------------------------
print("\nHealth Status at End statistics:")
print(df["Health_status_end"].describe())

plt.figure(figsize=(8, 6))

plt.hist(
    df["Health_status_end"].dropna(),
    bins=10,
    edgecolor="black"
)

plt.xlabel("Health Status at End of Chemotherapy")
plt.ylabel("Number of Patients")
plt.title("Distribution of Health Status at End of Chemotherapy")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.savefig(
    "data/processed/health_status_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -------------------------------
# Baseline EORTC
# -------------------------------
print("\nBaseline EORTC statistics:")
print(df["EORTC_baseline"].describe())

plt.figure(figsize=(8, 6))

plt.hist(
    df["EORTC_baseline"].dropna(),
    bins=10,
    edgecolor="black"
)

plt.xlabel("Baseline EORTC Score")
plt.ylabel("Number of Patients")
plt.title("Distribution of Baseline EORTC Scores")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.savefig(
    "data/processed/eortc_baseline_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGraphs saved:")
print("data/processed/health_status_distribution.png")
print("data/processed/eortc_baseline_distribution.png")

print("\n" + "=" * 70)
print("STEP 2X COMPLETE")
print("=" * 70)
