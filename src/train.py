import mlflow
import pandas as pd
import argparse
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
import joblib

# -----------------------------
# FIX: Set tracking URI
# -----------------------------
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data/data_v1.csv")
parser.add_argument("--model", type=str, default="rf")
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--features", type=str, default="all")
args = parser.parse_args()

# -----------------------------
# Experiment
# -----------------------------
mlflow.set_experiment("2022BCD0057_experiment")

def train():
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "model.pkl"
    metrics_path = artifacts_dir / "metrics.json"

    # -----------------------------
    # Load Data
    # -----------------------------
    df = pd.read_csv(args.data)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # -----------------------------
    # Feature Selection
    # -----------------------------
    if args.features == "reduced":
        selected_features = ["Glucose", "BMI", "Age"]
        X = X[selected_features]
    else:
        selected_features = list(X.columns)

    # -----------------------------
    # Train/Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # Model Selection
    # -----------------------------
    if args.model == "rf":
        model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42)
    else:
        model = LogisticRegression(max_iter=200)

    # -----------------------------
    # Train
    # -----------------------------
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # -----------------------------
    # Metrics
    # -----------------------------
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)

    # -----------------------------
    # MLflow Logging (WITH RUN NAME 🔥)
    # -----------------------------
    run_name = f"{args.model}_{args.data.split('/')[-1]}_{args.features}"

    with mlflow.start_run(run_name=run_name) as run:

        print("Run ID:", run.info.run_id)

        # Parameters
        mlflow.log_param("model", args.model)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("dataset", args.data)
        mlflow.log_param("features_used", selected_features)

        # Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)

        # Model artifact
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

    # -----------------------------
    # Save metrics JSON (MANDATORY)
    # -----------------------------
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "features_used": selected_features,
        "Name": "Priyanka Kumari",
        "Roll No": "2022BCD0057"
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Training completed ✅")

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    train()