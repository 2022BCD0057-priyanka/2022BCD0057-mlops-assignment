from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("models/model.pkl")

@app.get("/")
def health():
    return {
        "Name": "Priyanka Kumari",
        "Roll No": "2022BCD0057"
    }

@app.post("/predict")
def predict(data: list):
    pred = model.predict([data])
    return {
        "prediction": int(pred[0]),
        "Name": "Priyanka Kumari",
        "Roll No": "2022BCD0057"
    }