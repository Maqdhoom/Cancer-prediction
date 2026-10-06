# Cancer PRO Prediction

A machine-learning project for predicting a cancer patient's end-of-treatment health status from clinical and demographic features. The workflow includes data preparation, exploratory analysis, model comparison, hyperparameter tuning, and a Streamlit web app for interactive prediction.

## Project overview

This project predicts a continuous health-status score from 0 to 100 using patient information such as age, education, treatment details, ECOG score, HER2 status, and perceived risk of recurrence. A tuned Random Forest regressor is used as the final model.

## Repository structure

- `app/app.py` – interactive Streamlit application
- `src/` – data preparation, preprocessing, training, and evaluation scripts
- `data/raw/` – raw patient datasets
- `data/processed/` – cleaned/processed dataset outputs
- `models/` – trained model artifacts
- `results/` – evaluation metrics and charts
- `requirements.txt` – Python dependencies

## Setup

1. Clone the repository.
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install Streamlit if it is not already available:
   ```bash
   pip install streamlit
   ```

## Run the app

Start the web application from the project root:

```bash
streamlit run app/app.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`).

## Main workflow

The project is organized around a typical ML pipeline:

1. Prepare the raw dataset using the scripts in `src/`
2. Preprocess and validate the target variable
3. Split and compare multiple models
4. Tune the best-performing model
5. Evaluate final performance and generate feature-importance plots
6. Use the Streamlit app for patient-level predictions

## Key outputs

- `results/final_metrics.csv` – MAE, RMSE, and R² metrics
- `results/final_feature_importance.png` – model feature importance chart
- `models/tuned_random_forest_pipeline.pkl` – saved trained pipeline
- `results/final_predictions.csv` – model predictions on the test set

## Notes

- This is a research/demo project and should not be used as a clinical diagnosis tool.
- The app is intended for demonstration and exploratory prediction rather than medical decision-making.

## License

This project is provided for educational and research use.
