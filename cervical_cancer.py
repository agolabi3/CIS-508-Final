"""
CIS 508 Final Project - Cervical Cancer Risk Prediction
Python script version (works locally or in Colab, with optional Databricks MLflow logging)

Usage (local):
    python cervical_cancer.py --csv_path path/to/risk_factors_cervical_cancer.csv

Usage (Colab):
    1. Upload this script and the CSV to your Colab environment
    2. Run: %run cervical_cancer.py            # will prompt for file upload if no --csv_path is given

This script:
  - Loads and cleans the cervical cancer dataset
  - Trains Logistic Regression and Random Forest models
  - Evaluates accuracy, precision, recall, and F1
  - Optionally logs runs to Databricks MLflow if credentials are available
  - Saves the best model as cervical_rf_model.pkl
"""

import argparse
import os
import sys
from typing import Optional
from contextlib import nullcontext

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Try to import mlflow; if not available, we'll just skip logging gracefully
try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None  # type: ignore
    MLFLOW_AVAILABLE = False


def load_dataset(csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the cervical cancer dataset.

    If csv_path is None and we are in a Colab environment, prompt the user to upload a file.
    """
    if csv_path is None:
        # Detect Colab
        if "google.colab" in sys.modules:
            from google.colab import files  # type: ignore

            print("📂 No CSV path provided. Please upload 'risk_factors_cervical_cancer.csv'.")
            uploaded = files.upload()
            if not uploaded:
                raise ValueError("No file uploaded.")
            csv_path = list(uploaded.keys())[0]
            print(f"✅ Using uploaded file: {csv_path}")
        else:
            raise ValueError(
                "csv_path is required when not running in Google Colab. "
                "Use --csv_path /path/to/risk_factors_cervical_cancer.csv"
            )

    df = pd.read_csv(csv_path)
    return df


def clean_and_split(df: pd.DataFrame):
    """Clean the dataset and return train/test splits."""
    # Replace '?' with NaN and convert numeric columns
    df = df.copy()
    df.replace("?", np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors="ignore")

    # Drop rows without target
    if "Biopsy" not in df.columns:
        raise KeyError("Expected target column 'Biopsy' not found in dataset.")
    df.dropna(subset=["Biopsy"], inplace=True)

    X = df.drop(columns=["Biopsy"])
    y = df["Biopsy"]

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def evaluate_model(name: str, y_true, y_pred):
    """Compute and print standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n📊 {name} Results")
    print("-" * 40)
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}


def setup_databricks_mlflow(experiment_name: str):
    """
    Configure MLflow to log to Databricks, if mlflow is available.

    It will:
      - Try to load DATABRICKS_HOST and DATABRICKS_TOKEN from Colab userdata if present
      - Fall back to environment variables if not
    """
    if not MLFLOW_AVAILABLE:
        print("⚠️ MLflow not installed. Skipping MLflow logging.")
        return False

    # If running in Colab, we can optionally pull secrets from google.colab.userdata
    if "google.colab" in sys.modules:
        try:
            from google.colab import userdata  # type: ignore

            host = userdata.get("DATABRICKS_HOST")
            token = userdata.get("DATABRICKS_TOKEN")
            if host:
                os.environ["DATABRICKS_HOST"] = host
            if token:
                os.environ["DATABRICKS_TOKEN"] = token
        except Exception:
            # If this fails, we just rely on environment variables directly
            pass

    try:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(experiment_name)
        print("✅ MLflow tracking URI:", mlflow.get_tracking_uri())
        print("✅ MLflow experiment  :", experiment_name)
        return True
    except Exception as e:
        print(f"⚠️ Could not configure Databricks MLflow: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train cervical cancer risk models and (optionally) log to Databricks MLflow."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to risk_factors_cervical_cancer.csv (optional in Colab; will prompt upload if omitted).",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="/Users/agolabi3@asu.edu/CervicalCancer_FinalProject",
        help="Databricks MLflow experiment path.",
    )
    args = parser.parse_args()

    # 1) Load and split data
    df = load_dataset(args.csv_path)
    X_train, X_test, y_train, y_test = clean_and_split(df)

    # 2) Optionally set up MLflow
    mlflow_ok = setup_databricks_mlflow(args.experiment_name) if MLFLOW_AVAILABLE else False

    # 3) Train and evaluate Random Forest
    with (mlflow.start_run(run_name="RandomForest_CervicalCancer") if mlflow_ok else nullcontext()):
        rf = RandomForestClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=None,
        )
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)
        rf_metrics = evaluate_model("Random Forest", y_test, rf_preds)

        if mlflow_ok:
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", rf.n_estimators)
            mlflow.log_param("max_depth", rf.max_depth)
            for k, v in rf_metrics.items():
                mlflow.log_metric(k, v)
            mlflow.sklearn.log_model(rf, artifact_path="model")

    # 4) Train and evaluate Logistic Regression
    with (mlflow.start_run(run_name="LogisticRegression_CervicalCancer") if mlflow_ok else nullcontext()):
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(X_train, y_train)
        log_preds = logreg.predict(X_test)
        log_metrics = evaluate_model("Logistic Regression", y_test, log_preds)

        if mlflow_ok:
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("max_iter", logreg.max_iter)
            for k, v in log_metrics.items():
                mlflow.log_metric(k, v)
            mlflow.sklearn.log_model(logreg, artifact_path="model")

    # 5) Save best model locally (prioritize recall in this healthcare context)
    best_model = rf if rf_metrics["recall"] >= log_metrics["recall"] else logreg
    best_name = "Random Forest" if best_model is rf else "Logistic Regression"

    from joblib import dump

    dump(best_model, "cervical_rf_model.pkl")
    print(f"\n💾 Saved best model ({best_name}) to 'cervical_rf_model.pkl'.")

    print(
        """
✅ NOTE:
In this healthcare screening context, RECALL (sensitivity) is the most important metric.
We prefer the model that misses the fewest true cancer cases (minimizing false negatives),
even if it produces a few more false positives.
"""
    )


if __name__ == "__main__":
    main()
