"""Retrain the logistic regression with a leakage-free pipeline.

Fixes vs. the original notebook:
  * scaler is fitted on the TRAINING split only (the notebook refit it on the
    test split, so the saved scaler.pkl does not match the trained model);
  * features are scaled once (the notebook standardized, then standardized again);
  * imputation means come from the training split and are saved to disk;
  * features are not truncated to int before scaling.

Run from the folder containing heart.csv:  python train_model.py
"""
import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from preprocessing import FEATURE_COLUMNS, build_features

df = pd.read_csv("heart.csv")
X_raw = df.drop(columns="HeartDisease")
y = df["HeartDisease"]

# Same split as the notebook (test_size=0.2, random_state=42).
X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_raw, y, test_size=0.20, random_state=42
)

# Imputation values from the training split only.
impute = {
    "Cholesterol": round(X_tr_raw.loc[X_tr_raw["Cholesterol"] != 0, "Cholesterol"].mean(), 2),
    "RestingBP": round(X_tr_raw.loc[X_tr_raw["RestingBP"] != 0, "RestingBP"].mean(), 2),
}

X_tr = build_features(X_tr_raw, impute)[FEATURE_COLUMNS]
X_te = build_features(X_te_raw, impute)[FEATURE_COLUMNS]

scaler = StandardScaler().fit(X_tr)
model = LogisticRegression(max_iter=1000)
model.fit(scaler.transform(X_tr), y_tr)

pred = model.predict(scaler.transform(X_te))
print("Test accuracy:", round(accuracy_score(y_te, pred), 4))
print("Test F1:      ", round(f1_score(y_te, pred), 4))

joblib.dump(model, "logistic_heart.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(FEATURE_COLUMNS, "columns.pkl")
with open("preprocessing.json", "w") as f:
    json.dump({"impute": impute, "feature_columns": FEATURE_COLUMNS}, f, indent=2)
print("Saved logistic_heart.pkl, scaler.pkl, columns.pkl, preprocessing.json")