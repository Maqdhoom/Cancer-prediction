import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 70)
print("STEP 2AB - ENSEMBLE VOTING")
print("=" * 70)

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv("data/processed/cancer_qol_raw.csv")

target = "Health_status_end"

X = df.drop(columns=[target])
y = df[target]

# ---------------------------------------------------------
# Train-test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ---------------------------------------------------------
# Identify numerical and categorical features
# ---------------------------------------------------------
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

# ---------------------------------------------------------
# Create three different models
# ---------------------------------------------------------

ridge = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        Ridge(alpha=10)
    )
])

random_forest = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42
        )
    )
])

gradient_boosting = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )
    )
])

# ---------------------------------------------------------
# Voting Regressor
# ---------------------------------------------------------
voting_model = VotingRegressor(
    estimators=[
        ("ridge", ridge),
        ("random_forest", random_forest),
        ("gradient_boosting", gradient_boosting)
    ]
)

# ---------------------------------------------------------
# Train ensemble
# ---------------------------------------------------------
print("\nTraining Voting Regressor...")

voting_model.fit(
    X_train,
    y_train
)

# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------
predictions = voting_model.predict(X_test)

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n" + "-" * 50)
print("VOTING REGRESSOR RESULTS")
print("-" * 50)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# ---------------------------------------------------------
# Save predictions
# ---------------------------------------------------------
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

results.to_csv(
    "data/processed/ensemble_predictions.csv",
    index=False
)

# ---------------------------------------------------------
# Save metrics
# ---------------------------------------------------------
metrics = pd.DataFrame({
    "Model": ["Voting Regressor"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2]
})

metrics.to_csv(
    "data/processed/ensemble_metrics.csv",
    index=False
)

# ---------------------------------------------------------
# Save model
# ---------------------------------------------------------
import joblib

joblib.dump(
    voting_model,
    "data/processed/voting_regressor.pkl"
)

print("\nFiles saved:")
print("data/processed/ensemble_predictions.csv")
print("data/processed/ensemble_metrics.csv")
print("data/processed/voting_regressor.pkl")

print("\n" + "=" * 70)
print("STEP 2AB COMPLETE")
print("=" * 70)
