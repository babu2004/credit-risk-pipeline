from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predict Default / Not Default using trained ML model",
    version="1.0"
)

# --------------------
# Load model at startup
# --------------------
model = None

@app.on_event("startup")
def load_model():
    global model
    model = xgb.Booster()
    model.load_model("models/xgboost.json")
    print("✅ XGBoost model loaded successfully")


# --------------------
# Input schema
# --------------------
class CreditApplication(BaseModel):
    Seniority: float
    Home: float
    Time: float
    Age: float
    Marital: float
    Records: float
    Job: float
    Expenses: float
    Income: float
    Assets: float
    Debt: float
    Amount: float
    Price: float


FEATURE_ORDER = [
    "Seniority", "Home", "Time", "Age", "Marital", "Records", "Job",
    "Expenses", "Income", "Assets", "Debt", "Amount", "Price"
]


# --------------------
# Health check
# --------------------
@app.get("/")
def root():
    return {"message": "Credit Risk API is running"}


# --------------------
# Prediction endpoint
# --------------------
@app.post("/predict")
def predict(application: CreditApplication):
    # Convert input to DataFrame (correct feature order)
    data = pd.DataFrame([[getattr(application, col) for col in FEATURE_ORDER]],
                        columns=FEATURE_ORDER)

    # Convert to DMatrix
    dmatrix = xgb.DMatrix(data)

    # Predict probability
    prob_default = model.predict(dmatrix)[0]

    # Decision threshold
    prediction = "NOT ELIGIBLE" if prob_default >= 0.5 else "ELIGIBLE"

    return {
    "prediction": prediction,
    "Probability of able to pay the loan:": f"{round((1 - 0.23) * 100, 2)}%"
}

