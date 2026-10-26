"""Heart Disease Prediction (educational). Uses the shared preprocessing module."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from preprocessing import FEATURE_COLUMNS, build_features

BASE = Path(__file__).parent


@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE / "logistic_heart.pkl")
    scaler = joblib.load(BASE / "scaler.pkl")
    cfg = json.loads((BASE / "preprocessing.json").read_text())
    return model, scaler, cfg


st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("Heart Disease Prediction")
st.caption(
    "Educational machine-learning demo. Not a medical diagnosis. "
    "Consult a qualified healthcare professional for medical concerns."
)

try:
    model, scaler, cfg = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Model file missing ({e.filename}). Run train_model.py to generate "
        "logistic_heart.pkl, scaler.pkl, columns.pkl and preprocessing.json."
    )
    st.stop()

if cfg.get("feature_columns") != FEATURE_COLUMNS:
    st.error("Feature list in preprocessing.json does not match the code. Retrain the model.")
    st.stop()

st.markdown("Provide the following details")

with st.form("patient"):
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Gender", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    blood_pressure = st.number_input(
        "Resting Blood Pressure (mmHg)", min_value=1, max_value=250, value=120,
        help="Enter a measured value. Zero is not valid.",
    )
    cholesterol = st.number_input(
        "Cholesterol (mg/dL)", min_value=1, max_value=700, value=200,
        help="Enter a measured value. Zero is not valid.",
    )
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar", [0, 1],
        format_func=lambda x: "Yes (> 120 mg/dL)" if x == 1 else "No (≤ 120 mg/dL)",
    )
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["N", "Y"])
    oldpeak = st.number_input("Oldpeak", min_value=-3.0, max_value=7.0, value=0.0, step=0.1)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    submitted = st.form_submit_button("Predict")

if submitted:
    raw = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": blood_pressure,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }])

    try:
        features = build_features(raw, cfg["impute"])
        scaled = scaler.transform(features[FEATURE_COLUMNS])
        prediction = int(model.predict(scaled)[0])
        probability = float(model.predict_proba(scaled)[0][1])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.subheader("Prediction Result")
    if prediction == 1:
        st.warning("The model predicts a potential heart disease risk.")
    else:
        st.success("The model does not predict heart disease.")
    st.write(f"Model probability of the positive class: {probability * 100:.1f}%")
    st.caption(
        "This probability is a model output from a model trained on a small public dataset. "
        "It is not a clinically validated risk score."
    )