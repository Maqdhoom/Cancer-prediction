import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Cancer PRO Prediction",
    page_icon="🩺",
    layout="centered"
)

# --------------------------------------------------
# Load model
# --------------------------------------------------

MODEL_PATH = "models/tuned_random_forest_pipeline.pkl"

model = joblib.load(MODEL_PATH)

# Load project results
metrics = pd.read_csv(
     "results/final_metrics.csv"
)

features =pd.read_csv("results/final_metrics.csv")

# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🩺 Cancer Patient Health Status Prediction")

st.write(
    "Machine-learning prediction of end-of-treatment "
    "health status using patient and treatment information."
)

st.info(
    "This application is a machine-learning project demonstration "
    "and is not a clinical diagnosis or medical recommendation."
)

# --------------------------------------------------
# Prediction section
# --------------------------------------------------

st.header("Patient Information")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=50
)

educational_level = st.selectbox(
    "Educational Level",
    ["Secondary or higher education", "Other"]
)

chemotherapy_regimen = st.selectbox(
    "Chemotherapy Regimen",
    ["Taxane-based", "Other"]
)

radiotherapy = st.selectbox(
    "Radiotherapy",
    ["Yes", "No"]
)

occupational_status = st.selectbox(
    "Occupational Status",
    ["Not employed", "Employed"]
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=10,
    value=2
)

tnm_stage = st.selectbox(
    "TNM Stage",
    ["I", "II", "III"]
)

eortc_baseline = st.number_input(
    "Baseline EORTC Score",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=0.1
)

ecog = st.selectbox(
    "ECOG Score",
    [0, 1, 2]
)

her2 = st.selectbox(
    "HER2",
    ["Negative", "Positive"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Married/partnered", "Single"]
)

perceived_risk = st.selectbox(
    "Perceived Risk of Recurrence",
    ["Low", "Medium", "High", "Very high"]
)

surgery = st.selectbox(
    "Surgery",
    ["Mastectomy", "BSC"]
)

axillary = st.selectbox(
    "Axillary Lymphadenectomy",
    ["Yes", "No"]
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "🔮 Predict Health Status",
    use_container_width=True
):

    patient = pd.DataFrame({
        "Axillary_lymphadenectomy": [axillary],
        "Surgery": [surgery],
        "Age": [age],
        "Educational_level": [educational_level],
        "Chemotherapy_regimen": [chemotherapy_regimen],
        "Radiotherapy": [radiotherapy],
        "Occupational_status ": [occupational_status],
        "Children": [str(children)],
        "TNM_stage": [tnm_stage],
        "EORTC_baseline": [eortc_baseline],
        "ECOG": [ecog],
        "HER2": [her2],
        "Marital_status": [marital_status],
        "Perceived_risk_recurrence": [perceived_risk]
    })

    prediction = model.predict(patient)[0]

    prediction = max(
        0,
        min(100, prediction)
    )

    st.success("Prediction completed successfully!")

    st.metric(
        "Predicted Health Status",
        f"{prediction:.2f} / 100"
    )

# --------------------------------------------------
# Model information
# --------------------------------------------------

st.divider()

st.header("📊 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Patients", "219")

with col2:
    st.metric("Processed Features", "31")

with col3:
    st.metric("Model", "Random Forest")

# Metrics from final evaluation
mae = metrics.loc[
    metrics["Metric"] == "MAE",
    "Value"
].iloc[0]

rmse = metrics.loc[
    metrics["Metric"] == "RMSE",
    "Value"
].iloc[0]

r2 = metrics.loc[
    metrics["Metric"] == "R2",
    "Value"
].iloc[0]

st.subheader("Test Set Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", f"{mae:.2f}")

with col2:
    st.metric("RMSE", f"{rmse:.2f}")

with col3:
    st.metric("R²", f"{r2:.3f}")

# --------------------------------------------------
# Feature importance
# --------------------------------------------------
# --------------------------------------------------
# Top Important Features
# --------------------------------------------------

st.subheader("Top Important Features")

feature_importance_path = "results/final_feature_importance.png"

if os.path.exists(feature_importance_path):

    st.image(
        feature_importance_path,
        caption="Final Model Feature Importance",
        use_container_width=True
    )

else:
    st.warning("Feature importance graph not found.")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Cancer PRO Prediction | Machine Learning Mini-Project"
)
