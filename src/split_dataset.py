import pandas as pd
from sklearn.model_selection import train_test_split

print("=" * 70)
print("STEP 2K - TRAIN / TEST SPLIT")
print("=" * 70)

X = pd.read_csv("data/processed/X_processed.csv")
y = pd.read_csv("data/processed/y_final.csv").squeeze()

print("\nOriginal dataset:")
print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 70)
print("SPLIT RESULTS")
print("=" * 70)

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)

y_train.to_csv(
    "data/processed/y_train.csv",
    index=False,
    header=["Health_status_end"]
)

y_test.to_csv(
    "data/processed/y_test.csv",
    index=False,
    header=["Health_status_end"]
)

print("\nFiles created:")
print("X_train.csv")
print("X_test.csv")
print("y_train.csv")
print("y_test.csv")

print("\n" + "=" * 70)
print("STEP 2K COMPLETE")
print("=" * 70)
