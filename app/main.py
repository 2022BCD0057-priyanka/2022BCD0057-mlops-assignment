from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load model
with open("artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def health():
    return {
        "name": "Priyanka Kumari",
        "roll_no": "2022BCD0057"
    }

@app.post("/predict")
def predict(features: list):
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)

    return {
        "prediction": prediction.tolist(),
        "name": "Priyanka Kumari",
        "roll_no": "2022BCD0057"
    }