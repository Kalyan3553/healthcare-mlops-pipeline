"""
Training script with MLflow experiment tracking.
Logs params, metrics, and the model itself; registers best model in the
MLflow Model Registry under "heart_disease_risk_model".
"""
import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
import joblib

DATA_PATH = "data/processed/heart_disease_clean.csv"
MODEL_PATH = "models/risk_model.joblib"
EXPERIMENT_NAME = "heart_disease_risk"
REGISTERED_MODEL_NAME = "heart_disease_risk_model"

# Hyperparams pulled out so they're easy to sweep/log
PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "random_state": 42,
}


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def train():
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run():
        mlflow.log_params(PARAMS)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        model = XGBClassifier(**PARAMS, eval_metric="logloss")
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC:  {auc:.4f}")
        print("\n" + classification_report(y_test, preds))

        # Log model to MLflow (versioned artifact + registry entry)
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        # Also keep a local joblib copy for the FastAPI step later
        joblib.dump(model, MODEL_PATH)
        print(f"Model saved locally to {MODEL_PATH}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")

    return model, {"accuracy": acc, "f1": f1, "roc_auc": auc}


if __name__ == "__main__":
    train()