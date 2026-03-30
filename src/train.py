import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

# REQUIRED IDENTIFICATION
NAME = "Priyanka"
ROLL = "2022BCD0057"

# MLflow setup
mlflow.set_experiment("2022BCD0057_experiment")

# Load dataset
df = pd.read_csv("data/data.csv")

# Split
X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

# MLflow logging
with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "model")

# Save model
joblib.dump(model, "models/model.pkl")

# Save metrics (MANDATORY)
with open("metrics.json", "w") as f:
    json.dump({
        "mse": mse,
        "r2": r2,
        "name": NAME,
        "roll": ROLL
    }, f)