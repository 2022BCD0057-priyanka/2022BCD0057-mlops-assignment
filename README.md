# MLOps Diabetes Prediction System

A machine learning operations (MLOps) project implementing an end-to-end ML pipeline for diabetes prediction using FastAPI, DVC, and MLflow.

**Author:** Priyanka Kumari (Roll No: 2022BCD0057)

---

## 📋 Project Overview

This project demonstrates a complete MLOps workflow including:
- **Model Training**: Train multiple models (Logistic Regression, Random Forest)
- **Experiment Tracking**: MLflow integration for tracking experiments and metrics
- **Data Versioning**: DVC for managing dataset versions
- **Model Prediction API**: FastAPI-based REST API for serving predictions
- **Containerization**: Docker support for deployment

---

## 🎯 Features

✅ **Multiple Model Support**: Train with Logistic Regression or Random Forest  
✅ **Feature Engineering**: Support for all features or reduced feature set  
✅ **Experiment Tracking**: MLflow UI for comparing runs  
✅ **RESTful Prediction API**: FastAPI with input validation  
✅ **Data Versioning**: DVC for reproducible datasets  
✅ **Containerized**: Docker support for easy deployment  

---

## 📁 Project Structure

```
Assignment-final-mlops/
├── app/
│   └── main.py                 # FastAPI application & prediction endpoint
├── src/
│   └── train.py               # Model training script
├── data/
│   ├── data_v1.csv            # Training data version 1
│   ├── data_v2.csv            # Training data version 2
│   ├── diabetes.csv           # Original dataset
│   ├── data_v1.csv.dvc        # DVC version tracking
│   └── data_v2.csv.dvc        # DVC version tracking
├── artifacts/
│   ├── model.pkl              # Trained model (serialized)
│   └── metrics.json           # Training metrics
├── mlruns/                     # MLflow experiment runs
├── Dockerfile                 # Docker configuration
├── dvc.yaml                   # DVC pipeline configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Setup

1. **Clone/Navigate to project directory**
   ```bash
   cd Assignment-final-mlops
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   - **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### Train a Model

Train a model with specific parameters:

```bash
python src/train.py --data data/data_v2.csv --model logistic --features reduced
```

**Arguments:**
- `--data`: Path to training data (default: `data/data_v1.csv`)
- `--model`: Model type - `rf` (Random Forest) or `logistic` (default: `rf`)
- `--n_estimators`: Number of trees for Random Forest (default: `100`)
- `--features`: Feature set - `all` or `reduced` (default: `all`)

**Example outputs:**
- ✅ Trained model saved to `artifacts/model.pkl`
- ✅ Metrics saved to `artifacts/metrics.json`
- ✅ Experiment tracked in MLflow

### Start Prediction API

Launch the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

---

## 📡 API Endpoints

### 1. Health Check
**GET** `/`

Returns server status and author information.

**Response:**
```json
{
  "Name": "Priyanka Kumari",
  "Roll No": "2022BCD0057"
}
```

### 2. Prediction
**POST** `/predict`

Make predictions on diabetes likelihood.

**Request Body:**
```json
{
  "Pregnancies": 6,
  "Glucose": 148,
  "BloodPressure": 72,
  "SkinThickness": 35,
  "Insulin": 0,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.627,
  "Age": 50
}
```

**Response:**
```json
{
  "prediction": 1,
  "Name": "Priyanka Kumari",
  "Roll No": "2022BCD0057"
}
```

(Prediction: 0 = No Diabetes, 1 = Diabetes)

---

## 🧪 Testing with cURL

### Health Check
```bash
curl -X GET "http://127.0.0.1:8000/"
```

### Make Prediction
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }'
```

---

## 📊 MLflow Tracking

Track and compare your model experiments:

1. **Start MLflow UI**:
   ```bash
   mlflow ui
   ```

2. **View experiments** at `http://127.0.0.1:5000`

3. **Compare metrics** across different training runs

---

## 🐳 Docker Deployment

Build and run with Docker:

```bash
# Build image
docker build -t diabetes-prediction:latest .

# Run container
docker run -p 8000:8000 diabetes-prediction:latest
```

---

## 📚 Dependencies

- **fastapi**: Web framework for building APIs
- **uvicorn**: ASGI server
- **scikit-learn**: Machine learning library
- **joblib**: Model serialization
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **mlflow**: Experiment tracking
- **dvc**: Data versioning

---

## 📝 Notes

- Ensure `artifacts/model.pkl` exists before starting the API
- MLflow tracking URI is set to `http://127.0.0.1:5000`
- API validation requires all 8 input features
- Models are trained on the Diabetes dataset with 8 features

---

## 🤝 Contributing

This is an assignment project. For questions or improvements, please contact the author.

---

## 📄 License

Assignment project - Educational purposes only.

---

**Last Updated:** April 1, 2026
