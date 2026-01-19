import os
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import load_data, preprocess_data


DATA_PATH = "data/credit_risk.csv"
MODEL_DIR = "models"


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)


    df = load_data(DATA_PATH)
    df = preprocess_data(df)

  
    X = df.drop("Status", axis=1)
    y = (df["Status"] == 2).astype(int)

    # ---- Train / validation split ----
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    # Decision Tree

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    joblib.dump(dt_model, f"{MODEL_DIR}/decision_tree.pkl")

    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, f"{MODEL_DIR}/random_forest.pkl")

    
    #XGBoost
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "seed": 42
    }

    xgb_model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=175
    )

    xgb_model.save_model(f"{MODEL_DIR}/xgboost.json")

    print("✅ Training complete. Models saved in /models")


if __name__ == "__main__":
    train()
