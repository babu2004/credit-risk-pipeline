# Credit-Risk-Default-Prediction  
Predicting Loan Default Risk Using Machine Learning 💳📊

Hey there! 👋 Welcome to my machine learning project, where I focus on predicting whether a loan applicant is likely to **default or not default**. This project demonstrates how machine learning and modern MLOps practices can help financial institutions assess credit risk more accurately and consistently.

The entire pipeline is designed to be **reproducible, scalable, and production-ready**, so you can easily run it locally or deploy it using Docker. ✨  
The project takes a structured credit dataset and builds an **end-to-end ML system**, from training and evaluation to a REST API for real-time predictions.

The goal?  
Help financial institutions make **better, data-driven lending decisions**, reduce risk, and improve customer outcomes using AI.

---

## 📝 Problem Description

Credit risk assessment is a critical task in the financial industry. Incorrect decisions can lead to:
- Financial losses due to loan defaults
- Missed opportunities by rejecting low-risk applicants

Traditional rule-based systems struggle to capture complex patterns in customer data. This project shows how **machine learning models** can improve default prediction accuracy by learning from historical data.

---

## 🎯 Objective

The objective of this project is to build a machine learning model that predicts whether a customer will **default or not default** on a loan, based on financial and demographic attributes.

---

## 📊 Dataset

This project uses a **credit risk dataset** containing customer financial and demographic information.  
The data is **already numerically encoded**, making it suitable for tree-based models.

**Features include:**
- Seniority
- Home ownership
- Employment and marital status
- Income, expenses, assets, and debt
- Loan amount and price

**Target variable:**
- `Status`  
  - `1` → Default  
  - `0` → Not Default  

For more detailed information about the dataset and column definitions, please refer to the `data/` directory.

---

## 🧠 Models Used

The following models are trained and evaluated:

- Decision Tree Classifier  
- Random Forest Classifier  
- XGBoost (final deployed model)

Models are compared using **ROC-AUC**, and the best-performing model is used for inference.

---

## 🚀 Live Prediction via REST API

This project exposes predictions through a **FastAPI REST API**, making it easy to integrate with:
- Web applications
- Mobile apps
- Backend services
- Postman or curl

---

## 🔧 Tools & Techniques

- Machine Learning: scikit-learn, XGBoost  
- API Framework: FastAPI  
- Containerization: Docker  
- Data Processing: pandas  

---

## ✨ Setup

### Clone the Repository
```bash
git clone https://github.com/babu2004/credit-risk-pipeline.git
cd credit-risk-default-prediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the API
```bash
uvicorn src.api:app --reload
```

Open:
- http://127.0.0.1:8000/
- http://127.0.0.1:8000/docs

---

## 🚦 Get Going

Train models, start the API, and send prediction requests to get **DEFAULT / NOT DEFAULT** results.
