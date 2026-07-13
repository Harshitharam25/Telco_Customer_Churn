# Telco Customer Churn — Analysis & Prediction

Analyzing 7,000+ telecom customer records to identify what drives churn, and predicting churn risk with a logistic regression model. Insights delivered through a 2-page Power BI dashboard.

## Dataset

- **Source:** [IBM Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** ~7,043 customers, 21 features (demographics, services, contract, billing)
- **Target:** `Churn` (Yes/No)

## Headline result

<!-- FILL THIS IN: e.g. "Logistic regression predicts churn with XX% accuracy. Month-to-month contracts and electronic check payments are the strongest churn drivers." -->
Logistic regression achieves **XX% accuracy** on the test set. EDA shows churn concentrates in **month-to-month contracts**, **fiber optic internet customers**, and **higher monthly charges**.

## Method & key decisions

1. **Cleaning:** converted `TotalCharges` to numeric (coercing blanks), dropped nulls and the `customerID` column
2. **EDA:** churn distribution vs. contract type, internet service, payment method, tenure, and monthly charges (seaborn)
3. **Modeling:** one-hot encoding (`drop_first=True`), 80/20 train-test split, `StandardScaler`, logistic regression
4. **Evaluation:** accuracy + full classification report (precision/recall per class)
5. **Dashboard:** Power BI report with churn KPIs and driver breakdowns (see screenshots below)

## Dashboard

![Dashboard page 1](dashboards/screenshots/page1.png)
![Dashboard page 2](dashboards/screenshots/page2.png)

Full interactive report: `dashboards/Telco_Churn_Dashboard.pbix`

## How to reproduce

```bash
pip install -r requirements.txt
jupyter notebook notebooks/telco_churn_analysis.ipynb
```

## Limitations — what this does NOT do

- Single model (logistic regression); no comparison against tree-based models
- No hyperparameter tuning or cross-validation
- No class-imbalance handling (churn is a minority class)
- Snapshot data — no time dimension, so no survival/retention modeling

## AI disclosure

AI tools were used to assist with documentation and debugging; the analysis, modeling decisions, and dashboard design are my own.
