import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from data_preprocessing import preprocess_data, load_data

DATA_PATH = "data/credit_risk.csv"

def evaluate():
    #load and process data
    df = load_data(DATA_PATH)
    df = preprocess_data(df)

    #features X and target y
    X = df.drop("Status", axis=1)
    y = (df["Status"] == 2).astype(int)

    #train and val split

    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.2, random_state=42)

    #decision tree 
    dt_model = joblib.load("models/decision_tree.pkl")
    dt_preds = dt_model.predict_proba(X_val)[:, 1]
    dt_auc   = roc_auc_score(y_val,dt_preds)

    #Random forest
    rf_model = joblib.load("models/random_forest.pkl")
    rf_preds = rf_model.predict_proba(X_val)[:, 1]
    rf_auc   = roc_auc_score(y_val,rf_preds)

    #XGBoost
    xgb_model = xgb.Booster()
    xgb_model.load_model("models/xgboost.json") 

    dval = xgb.DMatrix(X_val)
    xgb_preds = xgb_model.predict(dval)
    xgb_auc = roc_auc_score(y_val,xgb_preds)


    # Results 

    print("ROC-AUC Scores")
    print("----------------")
    print(f"Decision Tree : {dt_auc:.4f}")
    print(f"Random Forest : {rf_auc:.4f}")
    print(f"XGBoost       : {xgb_auc:.4f}")

if __name__ == "__main__":
    evaluate()