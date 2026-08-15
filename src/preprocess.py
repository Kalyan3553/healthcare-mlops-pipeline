"""
Preprocessing for the UCI Heart Disease dataset.
Handles missing values, encodes categoricals, binarizes target.
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

RAW_PATH = "data/raw/heart_disease_uci.csv"
PROCESSED_PATH = "data/processed/heart_disease_clean.csv"

# Columns we don't want as model features
DROP_COLS = ["id", "dataset"]

CATEGORICAL_COLS = ["sex", "cp", "restecg", "exang", "slope", "thal"]
NUMERIC_COLS = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Binarize target: 0 = no disease, 1 = disease (any severity 1-4)
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"] + [c for c in DROP_COLS if c in df.columns])

    # fbs is boolean-like but stored as True/False strings in this source -> map to 0/1
    if df["fbs"].dtype == object:
        df["fbs"] = df["fbs"].map({"TRUE": 1, "FALSE": 0, True: 1, False: 0})
    df["fbs"] = df["fbs"].astype(float)  # keep NaN for imputation

    # Impute numeric columns with median
    num_imputer = SimpleImputer(strategy="median")
    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]
    df[numeric_present] = num_imputer.fit_transform(df[numeric_present])

    # fbs imputed separately (mode, since it's binary)
    df["fbs"] = df["fbs"].fillna(df["fbs"].mode()[0])

    # Impute categoricals with mode, then label encode
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna(df[col].mode()[0])
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def run():
    df = load_raw()
    df_clean = clean(df)
    df_clean.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved cleaned data: {df_clean.shape} -> {PROCESSED_PATH}")
    print(df_clean.isna().sum().sum(), "missing values remaining")
    print(df_clean["target"].value_counts())
    return df_clean


if __name__ == "__main__":
    run()
