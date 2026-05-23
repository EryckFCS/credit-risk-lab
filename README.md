# 📊 Credit Risk Lab

> Quantitative research portfolio in credit risk modeling, scoring, and regulatory capital analytics.

**Erick Condoy** · Economist (UNL) · Quant Researcher · Loja, Ecuador

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9?style=flat&logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
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

## ⚙️ Setup (uv)

This project uses **[uv](https://docs.astral.sh/uv/)** for environment and dependency management — consistent with the rest of the research infrastructure.

```bash
# 1. Clone
git clone https://github.com/EryckFCS/credit-risk-lab.git
cd credit-risk-lab

# 2. Create venv + install all deps (uv resolves & locks automatically)
uv sync

# 3. Install dev extras (jupyter, pytest, ruff)
uv sync --extra dev

# 4. Activate the environment
source .venv/bin/activate   # Linux/macOS (zsh/bash)

# 5. Register Jupyter kernel
uv run python -m ipykernel install --user --name credit-risk-lab
```

### Daily workflow
```bash
uv run python cli.py list                        # list case studies
uv run python cli.py run --case 01               # execute notebooks
uv run python cli.py report --case 01            # export HTML report
uv run pytest                                    # run test suite
```

> **Adding a package:** `uv add <package>` — updates `pyproject.toml` and `uv.lock` atomically.

---

## 🛠️ Tech Stack

```
Language  : Python 3.11+
Env mgmt  : uv (astral.sh)
Core libs : pandas · numpy · scikit-learn · statsmodels · scipy
Viz       : matplotlib · seaborn · plotly
Reporting : nbconvert (HTML/PDF) · Jinja2
DB        : DuckDB (local analytical queries)
Testing   : pytest + pytest-cov
Linting   : ruff
```

---

## 📂 Repository Structure

```
credit-risk-lab/
├── pyproject.toml             # uv project definition + deps
├── uv.lock                    # locked dependency graph (committed)
├── .python-version            # 3.11 (uv pin)
├── README.md
├── cli.py                     # pipeline CLI (uv run python cli.py)
├── .gitignore
├── LICENSE
│
├── utils/                     # shared modules
│   ├── metrics.py             # KS, Gini, AUC, PSI, CAP
│   ├── preprocessing.py       # WOE/IV binning
│   └── plotting.py            # CAP, ROC, score distribution
│
├── tests/                     # pytest test suite
│   ├── test_metrics.py
│   └── test_preprocessing.py
│
├── 01_credit_scoring/
│   ├── data/download.sh       # fetch UCI dataset
│   ├── notebooks/             # 01_eda · 02_features · 03_model
│   ├── reports/               # exported HTML reports
│   └── src/scorecard.py
│
├── 02_pd_lgd_estimation/
├── 03_concentration_risk/
├── 04_basel_capital_simulator/
└── 05_eda_sbs_ecuador/
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

## 🧪 Testing

```bash
uv run pytest                  # full suite
uv run pytest -v --tb=short    # verbose
uv run pytest --cov=utils      # with coverage
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Built with rigorous methodology. Every model is validated before it’s trusted.</i>
</p>
