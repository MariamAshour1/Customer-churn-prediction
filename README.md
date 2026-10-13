# 📉 Customer Churn Prediction

A supervised machine learning project that predicts whether a telecom customer will cancel their subscription (churn) using Python and scikit-learn.

---

## 📌 Project Overview

Customer churn is when subscribers stop using a service. This project builds a binary classification model to identify **at-risk customers early**, so the company can take action before they leave.

**Target variable:** `Churn` (Yes / No)

---

## 📁 Files

| File | Description |
|---|---|
| `customer_churn_prediction.ipynb` | Main Jupyter Notebook with all code, charts, and explanations |
| `Telco-Customer-Churn.csv` | Dataset (7,043 customers, 21 features) |
| `best_churn_model.joblib` | Saved final model pipeline |
| `report.md` | Short business report (problem, findings, results, recommendation) |

---

## 🗂️ Notebook Structure

1. `01_business_understanding` — Problem description and business context
2. `02_load_and_eda` — Load data, explore columns, 4 EDA charts
3. `03_split_and_preprocessing` — Train/test split and feature engineering pipeline
4. `04_baseline` — Dummy classifier as a performance baseline
5. `05_model_comparison` — Compare Logistic Regression, Decision Tree, Random Forest
6. `06_tuning` — Hyperparameter tuning with GridSearchCV
7. `07_final_evaluation` — Final model metrics and confusion matrix
8. `08_business_recommendations` — Insights, actions, and limitations

---

## 📊 Dataset

**IBM Telco Customer Churn** — [Source](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- 7,043 rows, 21 columns
- Mix of numerical and categorical features
- Target class imbalance: 73.5% Stay / 26.5% Churn

---

## ⚙️ How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib
   ```

3. **Open the notebook**
   ```bash
   jupyter notebook customer_churn_prediction.ipynb
   ```

4. **Run all cells** from top to bottom.

---

## 📈 Results

| Model | Accuracy | Recall (Churn) | F1 (Churn) |
|---|:---:|:---:|:---:|
| Baseline (Dummy) | 73.46% | 0.00% | 0.00% |
| Logistic Regression | 80.55% | 55.88% | 60.40% |
| Decision Tree | 79.42% | 54.55% | 58.45% |
| Random Forest | 80.27% | 49.73% | 57.23% |

**Best model:** Logistic Regression (tuned with `C=0.1` via GridSearchCV)

---

## 🔮 Using the Saved Model

```python
import joblib
import pandas as pd

model = joblib.load('best_churn_model.joblib')

customer = pd.DataFrame([{
    'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'No',
    'Dependents': 'No', 'tenure': 2, 'PhoneService': 'Yes',
    'MultipleLines': 'No', 'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No', 'OnlineBackup': 'No',
    'DeviceProtection': 'No', 'TechSupport': 'No',
    'StreamingTV': 'No', 'StreamingMovies': 'No',
    'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 70.70, 'TotalCharges': 151.65
}])

print(model.predict(customer))         # [1] = Churn
print(model.predict_proba(customer))   # Churn probability
```

---

## 🛠️ Tech Stack

- Python 3.10
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- joblib

---

## 📄 License

This project is for educational purposes only.
