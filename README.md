# Telco Customer Churn Prediction

Predicting which customers are likely to churn using service, contract, and billing data, so a retention team can proactively target at-risk customers.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — IBM sample dataset via Kaggle. 7,043 customers, 21 features (demographics, account info, services subscribed, and churn label).

## Approach

1. Cleaned and preprocessed the data (handled missing `TotalCharges`, dropped `customerID`, encoded categoricals, scaled numeric features)
2. Explored churn patterns across contract type, tenure, and monthly charges
3. Trained and compared three models: Logistic Regression (balanced), Random Forest, and Random Forest (balanced)
4. Evaluated on accuracy, precision, recall, F1, and ROC-AUC — with a focus on **recall for the churn class**, since missing an at-risk customer is costlier than a false alarm

## Results

| Model | Accuracy | Recall (Churn) | ROC-AUC |
|---|---|---|---|
| **Logistic Regression (balanced)** | 72.6% | **0.79** | **0.835** |
| Random Forest | 79.0% | 0.52 | 0.816 |
| Random Forest (balanced) | 78.7% | 0.49 | 0.817 |

**Chosen model: Logistic Regression (balanced).** It trades some raw accuracy for a much higher churn recall (79% vs ~50%), which better fits the business goal of flagging as many at-risk customers as possible.

### Key churn drivers
- Month-to-month contracts (vs. one/two-year)
- Fiber optic internet service
- Low tenure
- Lack of add-on services like online security and tech support

## Setup

```bash
pip install -r requirements.txt
jupyter notebook telco-customer-churn-prediction.ipynb
```

Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the same folder as the notebook before running.

## Future improvements

- Tune the classification threshold to adjust the precision/recall trade-off
- Try SMOTE as an alternative to class weighting
- Try XGBoost
- Engineer additional features (e.g., service-tenure ratios)
