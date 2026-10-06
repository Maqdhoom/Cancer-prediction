import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 70)
print("STEP 2M - RANDOM FOREST REGRESSION")
print("=" * 70)

# Load training and testing data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Create Random Forest model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
)

# Train model
model.fit(X_train, y_train)

print("\nRandom Forest training completed.")

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Training metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

# Testing metrics
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print("\n" + "=" * 70)
print("TRAINING PERFORMANCE")
print("=" * 70)

print(f"MAE  : {train_mae:.4f}")
print(f"RMSE : {train_rmse:.4f}")
print(f"R2   : {train_r2:.4f}")

print("\n" + "=" * 70)
print("TESTING PERFORMANCE")
print("=" * 70)

print(f"MAE  : {test_mae:.4f}")
print(f"RMSE : {test_rmse:.4f}")
print(f"R2   : {test_r2:.4f}")

# Actual vs predicted
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_test_pred
})

print("\n" + "=" * 70)
print("ACTUAL VS PREDICTED - TEST DATA")
print("=" * 70)

print(results.head(10))

# Save predictions
results.to_csv(
    "data/processed/random_forest_predictions.csv",
    index=False
)

# Save trained model
joblib.dump(
    model,
    "data/processed/random_forest_model.pkl"
)

print("\nFiles created:")
print("1. data/processed/random_forest_predictions.csv")
print("2. data/processed/random_forest_model.pkl")

print("\n" + "=" * 70)
print("STEP 2M COMPLETE")
print("=" * 70)
