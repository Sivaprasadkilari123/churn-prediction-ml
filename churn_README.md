# Customer Churn Prediction using Machine Learning

Predicts telecom customer churn using Logistic Regression, Random Forest, and XGBoost, with SMOTE to handle class imbalance. Includes an interactive Streamlit app for live predictions.

## Overview

- Built and compared 3 classification models: Logistic Regression, Random Forest, XGBoost
- Evaluated using Accuracy, Precision, Recall, and ROC-AUC
- Applied SMOTE to correct class imbalance (dataset is 80% no-churn / 20% churn, realistic for telecom)
- Ran feature-importance analysis to identify key churn drivers and inform retention strategy
- Deployed as a live Streamlit app for interactive churn risk prediction

## Model Results

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 68.2% | 30.6% | 46.7% | 0.647 |
| Random Forest | 73.2% | 34.8% | 39.2% | 0.681 |
| **XGBoost** | **74.2%** | **36.0%** | 37.5% | 0.679 |

## Key Churn Drivers (Feature Importance)

1. Contract type (month-to-month customers churn far more)
2. Internet service type (fiber optic customers churn more)
3. Lack of tech support
4. Lack of online security add-on

**Business takeaway:** Month-to-month customers without tech support or security add-ons are the highest-risk segment — a strong target for retention offers (discounted annual contracts, free security add-on trials).

## Tech Stack

Python, scikit-learn, XGBoost, imbalanced-learn (SMOTE), Streamlit, Pandas, NumPy

## Files

- `telecom_churn.csv` — dataset (3,000 customers, synthetic but realistic)
- `generate_data.py` — dataset generation script
- `train_model.py` — preprocessing, SMOTE, model training, evaluation
- `app.py` — Streamlit app for live churn prediction
- `churn_model.pkl`, `scaler.pkl`, `encoders.pkl`, `feature_columns.pkl` — saved model artifacts
- `feature_importance.csv` — churn driver rankings
- `requirements.txt` — dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
