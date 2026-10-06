import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 70)
print("STEP 2Q - FINAL MODEL EVALUATION")
print("=" * 70)

# Load original features and target
X = pd.read_csv("data/processed/X_features.csv")
y = pd.read_csv("data/processed/y_final.csv").squeeze()

# Same split used during tuning
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Load complete tuned pipeline
model = joblib.load(
    "data/processed/tuned_random_forest_pipeline.pkl"
)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 70)
print("FINAL TEST PERFORMANCE")
print("=" * 70)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

# Actual vs predicted
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Absolute_Error": np.abs(y_test.values - y_pred)
})

print("\n" + "=" * 70)
print("ACTUAL VS PREDICTED")
print("=" * 70)

print(results.to_string(index=False))

# Save results
results.to_csv(
    "data/processed/final_predictions.csv",
    index=False
)

# Save metrics
metrics = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2"],
    "Value": [mae, rmse, r2]
})

metrics.to_csv(
    "data/processed/final_metrics.csv",
    index=False
)

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print("1. data/processed/final_predictions.csv")
print("2. data/processed/final_metrics.csv")

print("\n" + "=" * 70)
print("STEP 2Q COMPLETE")
print("=" * 70)
