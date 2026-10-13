# Customer Churn Prediction — Short Report

**Course:** Supervised Machine Learning
**Dataset:** IBM Telco Customer Churn (7,043 customers, 21 columns)

---

## 1. Business Problem

A telecom company is losing customers who cancel their subscription (churn). Keeping existing customers is much cheaper than finding new ones, so the company wants to predict **which customers are likely to leave** before they do.

By building a machine learning model on customer data (account details, service type, monthly charges, etc.), the company can identify at-risk customers early and take action — like offering a discount or having support reach out — to keep them.

**Target variable:** `Churn` (Yes = customer left, No = customer stayed)
This is a **binary classification** problem.

---

## 2. Key Findings from Exploration

**Class Imbalance:**
73.5% of customers stayed and 26.5% churned. This means accuracy alone is misleading — a model that always predicts "Stay" would score 73.5% while catching zero churners. We focused on **Recall** and **F1-Score** instead.

**Three main patterns stood out:**

| Finding | Detail |
|---|---|
| **Contract Type** | Month-to-month customers churn at a much higher rate (>40%) vs. 1-year (~11%) and 2-year (~3%) contracts |
| **Tenure** | Most churn happens in the **first 12 months**. Customers who stay past 2 years rarely leave |
| **Monthly Charges** | Churning customers pay a higher median monthly bill (~$80) compared to customers who stay (~$65) |

> **Note:** These are correlations, not causes. For example, customers planning to leave tend to *choose* month-to-month plans so they can cancel easily — not the other way around.

---

## 3. Results

We trained and compared four models on an 80/20 train/test split:

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1 (Churn) |
|---|:---:|:---:|:---:|:---:|
| Baseline (Dummy) | 73.46% | 0.00% | 0.00% | 0.00% |
| Logistic Regression | 80.55% | 65.72% | 55.88% | 60.40% |
| Decision Tree | 79.42% | 62.96% | 54.55% | 58.45% |
| Random Forest | 80.27% | 67.39% | 49.73% | 57.23% |

**Logistic Regression** had the best Recall and F1-Score, so we selected it for tuning.

We used **GridSearchCV** (5-fold cross-validation) to tune the regularization parameter `C`. The best value was `C = 0.1`, which improved the cross-validation F1-Score from **0.5901 → 0.6337**.

**Final model performance on the test set:**

| Metric | Score |
|---|:---:|
| Accuracy | 80.55% |
| Precision (Churn) | 65.72% |
| Recall (Churn) | 55.88% |
| F1-Score (Churn) | 60.40% |

**Confusion Matrix breakdown:**
- **True Positives (TP):** Churners correctly identified — company can intervene.
- **True Negatives (TN):** Loyal customers correctly identified — no wasted outreach.
- **False Positives (FP):** Loyal customers flagged as at-risk — minor cost (small discount offer).
- **False Negatives (FN):** Churners missed by the model — highest cost, customer leaves with no action taken.

---

## 4. Recommendation

**We recommend deploying Logistic Regression** for this task because:

- It achieves a good balance between catching churners (Recall) and avoiding too many false alarms (Precision).
- It is simple and interpretable — business managers can understand *why* a customer was flagged.
- It is fast and easy to maintain in production.

**Two actions the business should take based on predictions:**

1. **Offer contract upgrade incentives** to month-to-month customers flagged as at-risk (e.g., a $10/month discount to switch to a 1-year plan).
2. **Provide dedicated onboarding support** to new customers in their first 6 months, since this is when churn risk is highest.

**Two limitations to keep in mind:**

1. The dataset is imbalanced (only 26.5% churn), making it harder to catch every churner.
2. The data is a static snapshot — it does not capture recent behavioral changes like spikes in customer complaints right before cancellation.
