from fastapi import FastAPI
import joblib

app = FastAPI()

# REQUIRED IDENTIFICATION
NAME = "Priyanka"
ROLL = "2022BCD0057"

# Load model
model = joblib.load("models/model.pkl")

# Health Endpoint
@app.get("/")
def health():
    return {
        "name": NAME,
        "roll": ROLL
    }

# Prediction Endpoint
@app.post("/predict")
def predict(data: list):
    prediction = model.predict([data]).tolist()
    return {
        "prediction": prediction,
        "name": NAME,
        "roll": ROLL
    }