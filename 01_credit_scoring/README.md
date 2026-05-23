# Case Study 01 — Credit Scoring: Retail Portfolio

## Overview

End-to-end credit scoring model for a retail credit card portfolio using the UCI Credit Default dataset (30,000 clients, Taiwan, 2005). The goal is to build a production-grade scorecard that assigns a numeric score to each client, reflecting their probability of defaulting in the next payment period.

## Business Problem

A retail bank needs to rank loan applicants by credit risk to:
1. Decide approve/reject at origination
2. Set credit limits proportional to risk
3. Monitor portfolio quality over time using PSI

## Methodology

| Stage | Technique |
|-------|-----------|
| EDA | Missing value analysis, class imbalance, bivariate plots |
| Feature Engineering | WOE/IV binning, fine/coarse classing |
| Modeling | Logistic Regression (interpretable, regulatory standard) |
| Scorecard Scaling | Points-to-double-odds methodology (PDO = 20, base score = 600) |
| Validation | KS, Gini, AUC-ROC, CAP curve, PSI |

## Dataset

- **Source:** UCI ML Repository — [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Observations:** 30,000 clients
- **Features:** 23 variables (demographics, payment history, bill amounts)
- **Target:** `default.payment.next.month` (1 = default, 0 = no default)
- **Class imbalance:** ~22% default rate

## Expected Results

| Metric | Target |
|--------|--------|
| KS | > 0.35 |
| Gini | > 0.45 |
| AUC-ROC | > 0.73 |
| PSI (OOT) | < 0.10 |

## Files

```
01_credit_scoring/
├── data/
│   └── download.sh          # Fetch UCI dataset
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_and_validation.ipynb
├── reports/
│   └── (exported HTML reports)
└── src/
    └── scorecard.py          # Scoring module
```

## References

- Yeh, I.C. & Lien, C. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications.
- Siddiqi, N. (2006). *Credit Risk Scorecards.* Wiley.
- Thomas, L.C. (2009). *Consumer Credit Models.* Oxford University Press.
