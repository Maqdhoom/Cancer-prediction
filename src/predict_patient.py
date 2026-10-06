import pandas as pd
import joblib

print("=" * 70)

print("CANCER PATIENT HEALTH STATUS PREDICTION")
print("=" * 70)

# Load trained pipeline
model = joblib.load(
    "models/tuned_random_forest_pipeline.pkl"
)

print("\nEnter patient information.\n")

# Patient inputs
age = float(input("Age: "))

educational_level = input(
    "Educational level (Secondary or higher education / Other): "
)

chemotherapy_regimen = input(
    "Chemotherapy regimen (Taxane-based / Other): "
)

radiotherapy = input(
    "Radiotherapy (Yes / No): "
)

occupational_status = input(
    "Occupational status (Not employed / Employed): "
)

children = input(
    "Number of children: "
)

tnm_stage = input(
    "TNM stage (I / II / III): "
)

eortc_baseline = float(
    input("Baseline EORTC score: ")
)

ecog = float(
    input("ECOG score (0 / 1 / 2): ")
)

her2 = input(
    "HER2 (Negative / Positive): "
)

marital_status = input(
    "Marital status (Married/partnered / Single): "
)

perceived_risk = input(
    "Perceived risk of recurrence (Low / Medium / High / Very high): "
)

surgery = input(
    "Surgery (Mastectomy / BSC): "
)

axillary = input(
    "Axillary lymphadenectomy (Yes / No): "
)

# Create patient using EXACT training column names
patient = pd.DataFrame({
    "Axillary_lymphadenectomy": [axillary],
    "Surgery": [surgery],
    "Age": [age],
    "Educational_level": [educational_level],
    "Chemotherapy_regimen": [chemotherapy_regimen],
    "Radiotherapy": [radiotherapy],
    "Occupational_status ": [occupational_status],
    "Children": [children],
    "TNM_stage": [tnm_stage],
    "EORTC_baseline": [eortc_baseline],
    "ECOG": [ecog],
    "HER2": [her2],
    "Marital_status": [marital_status],
    "Perceived_risk_recurrence": [perceived_risk]
})

# Make prediction
prediction = model.predict(patient)[0]

# Keep score within 0-100
prediction = max(0, min(100, prediction))

print("\n" + "=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

print(f"\nPredicted Health Status: {prediction:.2f}")

print("\nThe model predicts the patient's")
print("end-of-treatment health-status score on a 0-100 scale.")

print("\n" + "=" * 70)
print("PREDICTION COMPLETE")
print("=" * 70)
