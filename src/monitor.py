"""
Data drift monitoring with Evidently.

Compares a reference dataset (the data the model was trained on) against a
"current" batch (simulating newly arrived patient sensor readings) and
generates an HTML drift report.

In production this would run on a schedule (e.g. daily/weekly) against
freshly collected patient data, and could trigger the retraining pipeline
if significant drift is detected.
"""
import pandas as pd
import numpy as np
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset
import os

DATA_PATH = "data/processed/heart_disease_clean.csv"
REPORT_DIR = "reports"
REPORT_PATH = f"{REPORT_DIR}/drift_report.html"

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]


def load_reference_and_current():
    """
    Reference = the data the model was trained on (first 70%, sorted by index
    as a stand-in for "older" data).
    Current   = simulated new incoming batch (remaining 30%), with a bit of
    synthetic drift injected on a couple of vitals so the report has
    something meaningful to show (real new patient data may skew this way
    e.g. due to an older/sicker patient cohort, a new sensor calibration, etc).
    """
    df = pd.read_csv(DATA_PATH)

    split_idx = int(len(df) * 0.7)
    reference = df.iloc[:split_idx].copy()
    current = df.iloc[split_idx:].copy()

    # Simulate realistic drift: slightly older patients, higher resting BP
    # (e.g. a new clinic/cohort or sensor recalibration)
    rng = np.random.default_rng(42)
    current["age"] = current["age"] + rng.normal(5, 2, size=len(current))
    current["trestbps"] = current["trestbps"] * 1.08

    return reference, current


def build_data_definition():
    return DataDefinition(
        numerical_columns=NUMERIC_FEATURES,
        categorical_columns=CATEGORICAL_FEATURES,
    )


def run_drift_report():
    os.makedirs(REPORT_DIR, exist_ok=True)

    reference, current = load_reference_and_current()
    data_definition = build_data_definition()

    reference_dataset = Dataset.from_pandas(reference, data_definition=data_definition)
    current_dataset = Dataset.from_pandas(current, data_definition=data_definition)

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference_dataset, current_data=current_dataset)

    result.save_html(REPORT_PATH)
    print(f"Drift report saved to {REPORT_PATH}")

    # Also print a quick summary to console / logs
    result_dict = result.dict()
    print("Drift report generated. Open the HTML file to view full details.")

    return result


if __name__ == "__main__":
    run_drift_report()