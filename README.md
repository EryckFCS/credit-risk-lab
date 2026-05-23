# 📊 Credit Risk Lab

> Quantitative research portfolio in credit risk modeling, scoring, and regulatory capital analytics.

**Erick Condoy** · Economist (UNL) · Quant Researcher · Loja, Ecuador

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.14-4B8BBE?style=flat)](https://www.statsmodels.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Objective

This repository contains end-to-end case studies on credit risk modeling, built to production-grade standards. Each case study covers the full analytical cycle: data ingestion → exploratory analysis → model development → validation with industry metrics (KS, Gini, AUC, CAP) → business interpretation.

The focus is on techniques directly applicable to banking operations in Latin America, with references to **Basel III/IV** frameworks and **SBS Ecuador** regulatory data where available.

---

## 📁 Case Studies

| # | Study | Methods | Dataset | Status |
|---|-------|---------|---------|--------|
| [01](./01_credit_scoring/) | **Credit Scoring — Retail Portfolio** | Logistic Regression, Scorecard, ROC · KS · Gini | UCI Credit Default | 🔄 In Progress |
| [02](./02_pd_lgd_estimation/) | **PD & LGD Estimation — SME Lending** | Panel Logit, Survival Analysis, Beta Regression | SBS Ecuador / Synthetic | 📋 Planned |
| [03](./03_concentration_risk/) | **Concentration Risk — Sector Analysis** | HHI, CR5, Stress Testing | SBS Boletín Financiero | 📋 Planned |
| [04](./04_basel_capital_simulator/) | **Basel III Capital Requirement Simulator** | RWA, Tier 1 Capital, Standardized Approach | Synthetic | 📋 Planned |
| [05](./05_eda_sbs_ecuador/) | **Exploratory Analysis — SBS Ecuador Credit Portfolio** | EDA, Data Quality, Sector Breakdown | SBS Ecuador Open Data | 📋 Planned |

---

## 🛠️ Tech Stack

```
Language  : Python 3.11+
Core libs : pandas · numpy · scikit-learn · statsmodels · scipy
Viz       : matplotlib · seaborn · plotly
Reporting : nbconvert (HTML/PDF) · Jinja2
DB        : DuckDB (local analytical queries)
Env       : pip + venv | Docker-ready
```

---

## 📂 Repository Structure

```
credit-risk-lab/
├── README.md
├── requirements.txt           # Reproducible environment
├── .gitignore
│
├── utils/                     # Shared modules across all case studies
│   ├── __init__.py
│   ├── metrics.py             # KS, Gini, AUC, CAP curve, PSI
│   ├── preprocessing.py       # WOE/IV, binning, missing-value handlers
│   └── plotting.py            # CAP curve, ROC, score distribution
│
├── 01_credit_scoring/
│   ├── README.md              # Case study description
│   ├── data/                  # Raw data (not tracked by git)
│   │   └── download.sh        # Script to fetch UCI dataset
│   ├── notebooks/
│   │   ├── 01_eda.ipynb
│   │   ├── 02_feature_engineering.ipynb
│   │   └── 03_model_and_validation.ipynb
│   ├── reports/               # Exported HTML/PDF for sharing
│   └── src/
│       └── scorecard.py       # Production-ready scoring module
│
├── 02_pd_lgd_estimation/
│   ├── README.md
│   ├── data/
│   ├── notebooks/
│   └── reports/
│
├── 03_concentration_risk/
│   ├── README.md
│   ├── data/
│   ├── notebooks/
│   └── reports/
│
├── 04_basel_capital_simulator/
│   ├── README.md
│   ├── data/
│   ├── notebooks/
│   └── reports/
│
└── 05_eda_sbs_ecuador/
    ├── README.md
    ├── data/
    ├── notebooks/
    └── reports/
```

---

## ⚙️ Setup

```bash
# Clone the repository
git clone https://github.com/EryckFCS/credit-risk-lab.git
cd credit-risk-lab

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

---

## 📐 Validation Framework

Every model in this lab is evaluated with the following industry-standard metrics:

| Metric | Threshold (Good) | Interpretation |
|--------|-----------------|----------------|
| **KS Statistic** | > 0.30 | Max separation between good/bad score distributions |
| **Gini Coefficient** | > 0.40 | Discriminatory power (2·AUC − 1) |
| **AUC-ROC** | > 0.70 | Area under the ROC curve |
| **PSI** | < 0.10 | Population Stability Index (model drift monitoring) |
| **CAP Ratio** | > 0.60 | Cumulative Accuracy Profile |

---

## 🔗 Data Sources

- [UCI ML Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- [SBS Ecuador — Boletín Financiero](https://www.superbancos.gob.ec/bancos/estadisticas/)
- [Banco Central del Ecuador — Open Data](https://www.bce.fin.ec/estadisticas/)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Built with rigorous methodology. Every model is validated before it's trusted.</i>
</p>
