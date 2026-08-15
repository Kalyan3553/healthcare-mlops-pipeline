"""
Retraining trigger for the heart disease risk pipeline.

Logic:
1. Run a drift check (reference vs "current" data) using Evidently.
2. If the share of drifted columns exceeds DRIFT_THRESHOLD, trigger retraining.
3. Retrain using train.py's logic (logs a new run to MLflow).
4. Compare the new model's accuracy against the currently registered
   "Production" model. If it's better, promote the new version.

In production this script would run on a schedule (cron / Airflow / GitHub
Actions) against freshly collected patient data. Here we simulate that with
the same reference/current split used in monitor.py.
"""
import sys
import mlflow
from mlflow import MlflowClient
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

from monitor import load_reference_and_current, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from train import train as run_training, EXPERIMENT_NAME, REGISTERED_MODEL_NAME

DRIFT_THRESHOLD = 0.5  # retrain if >50% of columns show drift


def check_drift() -> float:
    """Returns the share of drifted columns (0.0 - 1.0)."""
    reference, current = load_reference_and_current()

    data_definition = DataDefinition(
        numerical_columns=NUMERIC_FEATURES,
        categorical_columns=CATEGORICAL_FEATURES,
    )
    reference_dataset = Dataset.from_pandas(reference, data_definition=data_definition)
    current_dataset = Dataset.from_pandas(current, data_definition=data_definition)

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=reference_dataset, current_data=current_dataset)

    result_dict = result.dict()
    drift_share = result_dict["metrics"][0]["value"]["share"]
    return drift_share


def get_production_accuracy(client: MlflowClient) -> float:
    """Fetch accuracy of the current Production-aliased model version, if any."""
    try:
        model_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "production")
        run = client.get_run(model_version.run_id)
        return run.data.metrics.get("accuracy", 0.0)
    except Exception:
        # No production model registered yet
        return 0.0


def promote_latest_model(client: MlflowClient):
    """Alias the most recently registered model version as 'production'."""
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    latest_version = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "production", latest_version.version)
    print(f"Promoted model version {latest_version.version} to 'production'")


def main():
    print("Checking for data drift...")
    drift_share = check_drift()
    print(f"Drift share: {drift_share:.3f} (threshold: {DRIFT_THRESHOLD})")

    if drift_share < DRIFT_THRESHOLD:
        print("Drift below threshold. No retraining needed.")
        return

    print("Drift threshold exceeded. Retraining model...")
    model, metrics = run_training()
    new_accuracy = metrics["accuracy"]
    print(f"New model accuracy: {new_accuracy:.4f}")

    client = MlflowClient()
    current_prod_accuracy = get_production_accuracy(client)
    print(f"Current production accuracy: {current_prod_accuracy:.4f}")

    if new_accuracy >= current_prod_accuracy:
        promote_latest_model(client)
        print("New model promoted to production.")
    else:
        print("New model did not outperform production. Not promoted.")


if __name__ == "__main__":
    main()