# Cervical Cancer Risk Prediction – CIS 508 Final Project

This project uses supervised machine learning to predict the likelihood of cervical cancer based on patient demographic, behavioral, and clinical risk factors.

It demonstrates the **full ML lifecycle**:
- Data loading & cleaning
- Model training & evaluation (Logistic Regression and Random Forest)
- Experiment tracking with **Databricks MLflow**
- Model export for deployment in a Streamlit app

---

## Dataset

- **Source:** Kaggle – *Cervical Cancer Risk Classification*  
- **File:** `risk_factors_cervical_cancer.csv`  
- **Target variable:** `Biopsy` (1 = cancer present, 0 = no cancer)  
- **Features:** Age, number of pregnancies, smoking, hormonal contraceptives, STDs, etc.

---

## Files in This Repo

- `cervical_cancer.py`  
  End-to-end Python script that:
  - Loads and cleans the dataset
  - Trains Random Forest and Logistic Regression models
  - Evaluates accuracy, precision, recall, and F1-score
  - Optionally logs runs to **Databricks MLflow**
  - Saves the best model to `cervical_rf_model.pkl`

- `app/streamlit_app.py` *(recommended for deployment)*  
  Streamlit app that uses `cervical_rf_model.pkl` to provide an interactive risk prediction UI.

- `requirements.txt`  
  Python dependencies for running the project.

---

## Installation

Create and activate a virtual environment (recommended), then install requirements:

```bash
pip install -r requirements.txt
```

If you are not using MLflow or Databricks, you can omit those packages.

---

## Usage

### 1️⃣ Local Python (no Colab)

Make sure you have the dataset locally, then run:

```bash
python cervical_cancer.py --csv_path path/to/risk_factors_cervical_cancer.csv
```

The script will:

- Train both models
- Print metrics (accuracy, precision, recall, F1)
- Save the best model as `cervical_rf_model.pkl` in the current directory

### 2️⃣ Google Colab

1. Upload `cervical_cancer.py` to your Colab environment.
2. In a Colab cell, run:

```python
%run cervical_cancer.py
```

You will be prompted to upload the CSV file if `--csv_path` is not provided.

Alternatively, you can give an explicit path:

```python
%run cervical_cancer.py --csv_path "/content/risk_factors_cervical_cancer.csv"
```

---

## Databricks MLflow Integration

The script can log runs to a Databricks MLflow experiment if:

- `mlflow` is installed, and  
- `DATABRICKS_HOST` and `DATABRICKS_TOKEN` are set in the environment, or  
- In Colab, you have stored them as `userdata` secrets (`DATABRICKS_HOST`, `DATABRICKS_TOKEN`).

By default, the experiment path is:

```text
/Users/agolabi3@asu.edu/CervicalCancer_FinalProject
```

You can override it:

```bash
python cervical_cancer.py \
  --csv_path path/to/risk_factors_cervical_cancer.csv \
  --experiment_name "/Users/you@example.com/YourExperiment"
```

Each run logs:

- Parameters: `model_type`, key hyperparameters  
- Metrics: `accuracy`, `precision`, `recall`, `f1_score`  
- Model artifact: serialized sklearn model (`model` folder in MLflow)

---

## Model Selection Criterion

In this healthcare screening context, **recall (sensitivity)** is the most important metric because it measures how many true cancer cases are correctly identified.

The script therefore chooses the “best” model as the one with **higher recall**, and saves that model to:

```text
cervical_rf_model.pkl
```

---

## Deployment (Streamlit)

To complete the deployment portion of the project, you can create a Streamlit app (in `app/streamlit_app.py`) that:

- Loads `cervical_rf_model.pkl`
- Lets a user input age, pregnancies, smoking status, contraceptive use, and STD history
- Returns a **“High risk”** or **“Low risk”** message with a short explanation

Example run:

```bash
cd app
streamlit run streamlit_app.py
```

This satisfies the **Deployment & User Interaction** requirement of the CIS 508 final project rubric.
