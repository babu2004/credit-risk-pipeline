import joblib
import xgboost as xgb
import pandas as pd

MODEL_PATH = "models/xgboost.json"   # best model (can switch if needed)

FEATURE_COLUMNS = [
    "Seniority", "Home", "Time", "Age", "Marital", "Records", "Job",
    "Expenses", "Income", "Assets", "Debt", "Amount", "Price"
]


def get_user_input():
    """
    Collect user input from CLI.
    """
    user_data = {}

    print("Enter applicant details:")

    for col in FEATURE_COLUMNS:
        value = float(input(f"{col}: "))
        user_data[col] = value

    return pd.DataFrame([user_data])


def predict():

    model = xgb.Booster()
    model.load_model(MODEL_PATH)

    user_df = get_user_input()

    dmatrix = xgb.DMatrix(user_df)

    # ---- Prediction ----
    prob_default = model.predict(dmatrix)[0]
    prediction = 1 if prob_default >= 0.5 else 0

    # ---- Output ----
    print("\nPrediction Result")
    print("-----------------")

    if prediction == 1:
        print("⚠️  NOT ELIGIBLE")
    else:
        print("✅ ELIGIBLE")

    print(f"Probability of able to pay the loan: {(1 - prob_default) * 100:.2f}%")


if __name__ == "__main__":
    predict()
