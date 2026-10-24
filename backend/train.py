from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

FEATURES = [
    "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
]
TARGET = "Machine failure"
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download ai4i2020.csv from "
            "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset"
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Unexpected CSV schema; missing columns: {missing}. Found: {list(df.columns)}")

    # Deliberately exclude ID columns and failure-mode sublabels to prevent target leakage.
    X = df[FEATURES].copy()
    y = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    if set(y.unique()) - {0, 1}:
        raise ValueError("Target must contain only 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categorical = ["Type"]
    numeric = [c for c in FEATURES if c not in categorical]
    prep = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical),
        ],
        remainder="drop"
    )

    classifier = Pipeline([
        ("preprocess", prep),
        ("model", RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
    classifier.fit(X_train, y_train)

    probabilities = classifier.predict_proba(X_test)[:, 1]
    # The threshold is a tunable demo policy, not a safety-certified operational limit.
    threshold = 0.35
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "dataset": "UCI AI4I 2020 Predictive Maintenance Dataset (synthetic)",
        "rows": int(len(df)),
        "failure_rate": float(y.mean()),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_failure": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_failure": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_failure": float(f1_score(y_test, predictions, zero_division=0)),
        "average_precision_pr_auc": float(average_precision_score(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "features": FEATURES,
        "excluded_leakage_columns": LEAKAGE_COLUMNS,
        "notes": "Synthetic benchmark; metrics do not establish real-world factory performance."
    }

    # Fit anomaly detection on training sensor inputs only; no target is passed to the detector.
    anomaly_prep = prep
    transformed_train = anomaly_prep.fit_transform(X_train)
    anomaly = IsolationForest(
        n_estimators=300, contamination=0.03, random_state=42
    )
    anomaly.fit(transformed_train)

    joblib.dump(classifier, ARTIFACTS / "failure_model.joblib")
    joblib.dump({"preprocess": anomaly_prep, "model": anomaly}, ARTIFACTS / "anomaly_model.joblib")
    (ARTIFACTS / "evaluation.json").write_text(json.dumps(metrics, indent=2))
    (ARTIFACTS / "metadata.json").write_text(json.dumps({
        "features": FEATURES,
        "threshold": threshold,
        "model_name": "RandomForestClassifier",
        "anomaly_model": "IsolationForest",
        "dataset": "UCI AI4I 2020 (synthetic)"
    }, indent=2))

    out = X_test.copy()
    out["actual_failure"] = y_test.values
    out["failure_probability"] = probabilities
    out["predicted_failure"] = predictions
    out.to_csv(ARTIFACTS / "test_predictions.csv", index=False)

    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(f"\nSaved model artifacts to: {ARTIFACTS}")

if __name__ == "__main__":
    main()
