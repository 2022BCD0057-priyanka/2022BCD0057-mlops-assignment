import mlflow
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data/data_v1.csv")
parser.add_argument("--model", type=str, default="rf")
parser.add_argument("--n_estimators", type=int, default=100)
args = parser.parse_args()

mlflow.set_experiment("2022BCD0057_experiment")

def train():
    df = pd.read_csv(args.data)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Feature selection (for one run)
    if "v2" in args.data:
        X = X[["Glucose", "BMI", "Age"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if args.model == "rf":
        model = RandomForestClassifier(n_estimators=args.n_estimators)
    else:
        model = LogisticRegression()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)

    with mlflow.start_run():
        mlflow.log_param("model", args.model)
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("dataset", args.data)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)

        joblib.dump(model, "models/model.pkl")
        mlflow.log_artifact("models/model.pkl")

    # Save metrics JSON (IMPORTANT requirement)
    with open("metrics.json", "w") as f:
        f.write(f"""{{
            "accuracy": {acc},
            "precision": {prec},
            "Name": "Priyanka Kumari",
            "Roll No": "2022BCD0057"
        }}""")

if __name__ == "__main__":
    train()
print("Training completed and model saved to models/model.pkl")