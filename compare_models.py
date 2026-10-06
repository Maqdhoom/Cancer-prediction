import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 70)
print("STEP 2N - MODEL COMPARISON")
print("=" * 70)

# Load data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# --------------------------------------------------
# Linear Regression
# --------------------------------------------------

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_train_pred = linear_model.predict(X_train)
linear_test_pred = linear_model.predict(X_test)

linear_train_mae = mean_absolute_error(y_train, linear_train_pred)
linear_train_rmse = np.sqrt(mean_squared_error(y_train, linear_train_pred))
linear_train_r2 = r2_score(y_train, linear_train_pred)

linear_test_mae = mean_absolute_error(y_test, linear_test_pred)
linear_test_rmse = np.sqrt(mean_squared_error(y_test, linear_test_pred))
linear_test_r2 = r2_score(y_test, linear_test_pred)

# --------------------------------------------------
# Random Forest
# --------------------------------------------------

rf_model = joblib.load(
    "data/processed/random_forest_model.pkl"
)

rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

rf_train_mae = mean_absolute_error(y_train, rf_train_pred)
rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
rf_train_r2 = r2_score(y_train, rf_train_pred)

rf_test_mae = mean_absolute_error(y_test, rf_test_pred)
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
rf_test_r2 = r2_score(y_test, rf_test_pred)

# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n" + "=" * 70)
print("LINEAR REGRESSION")
print("=" * 70)

print("\nTraining:")
print(f"MAE  : {linear_train_mae:.4f}")
print(f"RMSE : {linear_train_rmse:.4f}")
print(f"R2   : {linear_train_r2:.4f}")

print("\nTesting:")
print(f"MAE  : {linear_test_mae:.4f}")
print(f"RMSE : {linear_test_rmse:.4f}")
print(f"R2   : {linear_test_r2:.4f}")

print("\n" + "=" * 70)
print("RANDOM FOREST")
print("=" * 70)

print("\nTraining:")
print(f"MAE  : {rf_train_mae:.4f}")
print(f"RMSE : {rf_train_rmse:.4f}")
print(f"R2   : {rf_train_r2:.4f}")

print("\nTesting:")
print(f"MAE  : {rf_test_mae:.4f}")
print(f"RMSE : {rf_test_rmse:.4f}")
print(f"R2   : {rf_test_r2:.4f}")

# --------------------------------------------------
# Comparison table
# --------------------------------------------------

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "Train MAE": [
        linear_train_mae,
        rf_train_mae
    ],
    "Test MAE": [
        linear_test_mae,
        rf_test_mae
    ],
    "Train RMSE": [
        linear_train_rmse,
        rf_train_rmse
    ],
    "Test RMSE": [
        linear_test_rmse,
        rf_test_rmse
    ],
    "Train R2": [
        linear_train_r2,
        rf_train_r2
    ],
    "Test R2": [
        linear_test_r2,
        rf_test_r2
    ]
})

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(comparison.round(4).to_string(index=False))

# Save comparison
comparison.to_csv(
    "data/processed/model_comparison.csv",
    index=False
)

print("\nComparison saved to:")
print("data/processed/model_comparison.csv")

print("\n" + "=" * 70)
print("STEP 2N COMPLETE")
print("=" * 70)
