from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# -----------------------------
# Load model
# -----------------------------
model = joblib.load("artifacts/model.pkl")

# -----------------------------
# Define Input Schema (IMPORTANT 🔥)
# -----------------------------
class InputData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def health():
    return {
        "Name": "Priyanka Kumari",
        "Roll No": "2022BCD0057"
    }

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert to correct order
        input_data = np.array([[
            data.Pregnancies,
            data.Glucose,
            data.BloodPressure,
            data.SkinThickness,
            data.Insulin,
            data.BMI,
            data.DiabetesPedigreeFunction,
            data.Age
        ]])

        prediction = model.predict(input_data)

        return {
            "prediction": int(prediction[0]),
            "Name": "Priyanka Kumari",
            "Roll No": "2022BCD0057"
        }

    except Exception as e:
        return {"error": str(e)}