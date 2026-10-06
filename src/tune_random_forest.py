import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 70)
print("STEP 2O - RANDOM FOREST CROSS-VALIDATION")
print("=" * 70)

# --------------------------------------------------
# Load ORIGINAL features
# --------------------------------------------------

X = pd.read_csv("data/processed/X_features.csv")
y = pd.read_csv("data/processed/y_final.csv").squeeze()

print("\nFull dataset:")
print("X:", X.shape)
print("y:", y.shape)

# --------------------------------------------------
# Use the same 80/20 split as before
# --------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)

# --------------------------------------------------
# Identify feature types
# --------------------------------------------------

numeric_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumeric features:", len(numeric_columns))
print("Categorical features:", len(categorical_columns))

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            numeric_columns
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_columns
        )
    ]
)

# --------------------------------------------------
# Random Forest
# --------------------------------------------------

rf = RandomForestRegressor(
    random_state=42
)

# --------------------------------------------------
# Pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf)
    ]
)

# --------------------------------------------------
# Hyperparameters to test
# --------------------------------------------------

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5, 10],
    "model__min_samples_leaf": [2, 5, 10]
}

# --------------------------------------------------
# 5-fold cross-validation
# --------------------------------------------------

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    return_train_score=True
)

print("\nStarting hyperparameter search...")
print("Number of parameter combinations:",
      len(list(
          __import__("sklearn").model_selection.ParameterGrid(param_grid)
      )))

# Train/search
grid_search.fit(X_train, y_train)

print("\n" + "=" * 70)
print("BEST PARAMETERS")
print("=" * 70)

print(grid_search.best_params_)

print("\nBest cross-validation MAE:")
print(f"{-grid_search.best_score_:.4f}")

# --------------------------------------------------
# Best model
# --------------------------------------------------

best_model = grid_search.best_estimator_

# Predictions
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# --------------------------------------------------
# Training metrics
# --------------------------------------------------

train_mae = mean_absolute_error(
    y_train,
    y_train_pred
)

train_rmse = np.sqrt(
    mean_squared_error(
        y_train,
        y_train_pred
    )
)

train_r2 = r2_score(
    y_train,
    y_train_pred
)

# --------------------------------------------------
# Testing metrics
# --------------------------------------------------

test_mae = mean_absolute_error(
    y_test,
    y_test_pred
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_test_pred
    )
)

test_r2 = r2_score(
    y_test,
    y_test_pred
)

print("\n" + "=" * 70)
print("TUNED RANDOM FOREST - TRAINING")
print("=" * 70)

print(f"MAE  : {train_mae:.4f}")
print(f"RMSE : {train_rmse:.4f}")
print(f"R2   : {train_r2:.4f}")

print("\n" + "=" * 70)
print("TUNED RANDOM FOREST - TESTING")
print("=" * 70)

print(f"MAE  : {test_mae:.4f}")
print(f"RMSE : {test_rmse:.4f}")
print(f"R2   : {test_r2:.4f}")

# --------------------------------------------------
# Actual vs predicted
# --------------------------------------------------

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_test_pred
})

print("\n" + "=" * 70)
print("ACTUAL VS PREDICTED")
print("=" * 70)

print(results.head(10))

# Save predictions
results.to_csv(
    "data/processed/tuned_random_forest_predictions.csv",
    index=False
)

# Save complete pipeline
joblib.dump(
    best_model,
    "data/processed/tuned_random_forest_pipeline.pkl"
)

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print("1. data/processed/tuned_random_forest_predictions.csv")
print("2. data/processed/tuned_random_forest_pipeline.pkl")

print("\n" + "=" * 70)
print("STEP 2O COMPLETE")
print("=" * 70)
