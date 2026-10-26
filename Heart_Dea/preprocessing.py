"""Single source of truth for heart-disease feature preprocessing.

Used by BOTH train_model.py and the Streamlit app, so training and
inference cannot drift apart.
"""
import numpy as np
import pandas as pd

# Column order the model and scaler were fitted on.
FEATURE_COLUMNS = [
    "Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak",
    "MaxHR_Age_Ratio", "BP_Cholesterol", "Exercise_Stress", "HeartRateReserve",
    "Cholesterol_Age_Ratio", "BP_Age_Ratio",
    "Sex_M",
    "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA",
    "RestingECG_Normal", "RestingECG_ST",
    "ExerciseAngina_Y",
    "ST_Slope_Flat", "ST_Slope_Up",
    "AgeGroup_Adult", "AgeGroup_Middle_Age", "AgeGroup_Senior",
    "Oldpeak_Level_Mild", "Oldpeak_Level_Moderate", "Oldpeak_Level_High",
]

RAW_COLUMNS = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]


def build_features(raw: pd.DataFrame, impute: dict) -> pd.DataFrame:
    """Turn raw patient rows into the 27 model features, in FEATURE_COLUMNS order.

    `impute` holds the training-set means used to replace zero Cholesterol and
    zero RestingBP (the notebook treated 0 as "missing"). It must come from
    preprocessing.json saved at training time, never recomputed at inference.
    """
    df = raw[RAW_COLUMNS].copy()

    # Same rule as the notebook: 0 means missing, replace with the training mean.
    df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan).fillna(impute["Cholesterol"])
    df["RestingBP"] = df["RestingBP"].replace(0, np.nan).fillna(impute["RestingBP"])

    out = pd.DataFrame(index=df.index)

    # Base numeric features
    out["Age"] = df["Age"].astype(float)
    out["RestingBP"] = df["RestingBP"].astype(float)
    out["Cholesterol"] = df["Cholesterol"].astype(float)
    out["FastingBS"] = df["FastingBS"].astype(int)
    out["MaxHR"] = df["MaxHR"].astype(float)
    out["Oldpeak"] = df["Oldpeak"].astype(float)

    # Engineered numeric features (notebook formulas, NOT truncated to int)
    out["MaxHR_Age_Ratio"] = df["MaxHR"] / df["Age"]
    out["BP_Cholesterol"] = df["RestingBP"] * df["Cholesterol"]
    out["Exercise_Stress"] = ((df["ExerciseAngina"] == "Y") & (df["Oldpeak"] > 1)).astype(int)
    out["HeartRateReserve"] = 220 - df["Age"] - df["MaxHR"]
    out["Cholesterol_Age_Ratio"] = df["Cholesterol"] / df["Age"]
    out["BP_Age_Ratio"] = df["RestingBP"] / df["Age"]

    # One-hot encodings, matching pd.get_dummies(drop_first=True) baselines:
    # Sex F, ChestPainType ASY, RestingECG LVH, ST_Slope Down, ExerciseAngina N.
    out["Sex_M"] = (df["Sex"] == "M").astype(int)
    out["ChestPainType_ATA"] = (df["ChestPainType"] == "ATA").astype(int)
    out["ChestPainType_NAP"] = (df["ChestPainType"] == "NAP").astype(int)
    out["ChestPainType_TA"] = (df["ChestPainType"] == "TA").astype(int)
    out["RestingECG_Normal"] = (df["RestingECG"] == "Normal").astype(int)
    out["RestingECG_ST"] = (df["RestingECG"] == "ST").astype(int)
    out["ExerciseAngina_Y"] = (df["ExerciseAngina"] == "Y").astype(int)
    out["ST_Slope_Flat"] = (df["ST_Slope"] == "Flat").astype(int)
    out["ST_Slope_Up"] = (df["ST_Slope"] == "Up").astype(int)

    # Binned features. pd.cut bins are right-inclusive, as in the notebook.
    # Baselines: AgeGroup Young, Oldpeak_Level Normal.
    age_group = pd.cut(df["Age"], bins=[0, 30, 45, 60, 100],
                       labels=["Young", "Adult", "Middle_Age", "Senior"])
    out["AgeGroup_Adult"] = (age_group == "Adult").astype(int)
    out["AgeGroup_Middle_Age"] = (age_group == "Middle_Age").astype(int)
    out["AgeGroup_Senior"] = (age_group == "Senior").astype(int)

    oldpeak_level = pd.cut(df["Oldpeak"], bins=[-np.inf, 0, 1, 2, np.inf],
                           labels=["Normal", "Mild", "Moderate", "High"])
    out["Oldpeak_Level_Mild"] = (oldpeak_level == "Mild").astype(int)
    out["Oldpeak_Level_Moderate"] = (oldpeak_level == "Moderate").astype(int)
    out["Oldpeak_Level_High"] = (oldpeak_level == "High").astype(int)

    return out[FEATURE_COLUMNS]