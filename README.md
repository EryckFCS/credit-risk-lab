# 📊 Credit Risk Lab

> Quantitative research on credit risk modeling | Python · scikit-learn · statsmodels

Erick Condoy | Economist (UNL) | Quant Researcher

---

## Case Studies

| # | Study | Methods | Dataset | Status |
|---|-------|---------|---------|--------|
| 01 | Credit Scoring — Retail Portfolio | Logistic Reg, Scorecard, ROC/KS/Gini | UCI Credit Default | ✅ Complete |
| 02 | PD Estimation — SME Lending | Panel Data, Logit, LGD proxy | SBS Ecuador | 🔄 In Progress |
| 03 | Concentration Risk — Sector Analysis | HHI, CR5, Stress Testing | Boletín SBS 2023-24 | 📋 Planned |
| 04 | Basel III Capital Requirement Simulator | RWA, Tier 1, SA Approach | Synthetic | 📋 Planned |

## Stack
Python · pandas · scikit-learn · statsmodels · matplotlib · seaborn · DuckDB

## Structure
credit-risk-lab/
├── 01_credit_scoring/
│   ├── data/
│   ├── notebooks/
│   ├── reports/          ← PDF/HTML para LinkedIn
│   └── README.md
├── 02_pd_estimation/
...
└── utils/                ← funciones reutilizables (métricas: KS, Gini, AUC)
