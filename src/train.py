import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

mlflow.set_experiment("2022BCD0057_experiment")

def train(data_path, model_type="lr", features=None):

    df = pd.read_csv(data_path)

    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    if features:
        X = X[features]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    if model_type == "lr":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100)

    with mlflow.start_run():
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model", model_type)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, "model")

        print("Run complete")