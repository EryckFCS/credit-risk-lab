# 📊 Credit Risk Lab

> Quantitative research portfolio in credit risk modeling, scoring, and regulatory capital analytics.

**Erick Condoy** · Economist (UNL) · Quant Researcher · Loja, Ecuador

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![uv](https://img.shields.io/badge/uv-package%20manager-DE5FE9?style=flat&logo=astral&logoColor=white)](https://docs.astral.sh/uv/)
[![Marimo](https://img.shields.io/badge/Marimo-0.13+-E44C3B?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0wIDE4Yy00LjQyIDAtOC0zLjU4LTgtOHMzLjU4LTggOC04IDggMy41OCA4IDgtMy41OCA4LTggOHoiLz48L3N2Zz4=&logoColor=white)](https://marimo.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![statsmodels](https://img.shields.io/badge/statsmodels-0.14-4B8BBE?style=flat)](https://www.statsmodels.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Objective

End-to-end credit risk case studies built to production-grade standards. Each study covers the full analytical cycle: data ingestion → EDA → feature engineering → model development → validation (KS, Gini, AUC, CAP, PSI) → business interpretation.

All interactive analysis runs in **[Marimo](https://marimo.io)** — reactive Python notebooks with DAG execution, pure `.py` files, and immaculate Git diffs. Jupyter is permanently removed from this repository.

---

## 📁 Case Studies

| # | Study | Methods | Dataset | Status |
|---|-------|---------|---------|--------|
| [01](./01_credit_scoring/) | **Credit Scoring — Retail Portfolio** | Logistic Regression, Scorecard PDO, ROC·KS·Gini | UCI Credit Default (30k) | 🔄 In Progress |
| [02](./02_pd_lgd_estimation/) | **PD & LGD Estimation — SME Lending** | Panel Logit, Survival Analysis, Beta Regression | SBS Ecuador / Synthetic | 📋 Planned |
| [03](./03_concentration_risk/) | **Concentration Risk — Sector Analysis** | HHI, CR5, Stress Testing | SBS Boletín Financiero | 📋 Planned |
| [04](./04_basel_capital_simulator/) | **Basel III Capital Requirement Simulator** | RWA, Tier 1 Capital, Standardized Approach | Synthetic | 📋 Planned |
| [05](./05_eda_sbs_ecuador/) | **EDA — SBS Ecuador Credit Portfolio** | EDA, Data Quality, Sector Breakdown | SBS Ecuador Open Data | 📋 Planned |

---

## ⚙️ Setup (uv + Marimo)

```bash
# 1. Clone
git clone https://github.com/EryckFCS/credit-risk-lab.git
cd credit-risk-lab

# 2. Create venv + install all deps
uv sync

# 3. Install dev extras (pytest, ruff)
uv sync --extra dev

# 4. Activate
source .venv/bin/activate   # zsh/bash Linux
```

### Marimo workflow

```bash
# Edit a notebook interactively (reactive UI in browser)
uv run marimo edit 01_credit_scoring/notebooks/01_eda_scorecard.py

# Run as a read-only app
uv run marimo run 01_credit_scoring/notebooks/01_eda_scorecard.py

# Export to HTML (shareable report)
uv run marimo export html 01_credit_scoring/notebooks/01_eda_scorecard.py \
    -o 01_credit_scoring/reports/01_eda_scorecard.html
```

> **Why Marimo?** Pure `.py` files → clean `git diff`. DAG execution → no hidden state. Reactive UI → no manual re-runs. WASM export → shareable without a server.

### Daily CLI workflow

```bash
uv run python cli.py list          # list case studies
uv run pytest                      # run test suite
uv run ruff check .                # lint
```

---

## 🛠️ Tech Stack

```
Language   : Python 3.11+
Env mgmt   : uv (astral.sh)
Notebooks  : marimo 0.13+ (DAG-reactive .py, NOT Jupyter)
Core libs  : pandas · numpy · scikit-learn · statsmodels · scipy
Viz        : matplotlib · seaborn · plotly · altair
DB         : DuckDB (local analytical queries)
Testing    : pytest + pytest-cov
Linting    : ruff
```

---

## 📂 Repository Structure

```
credit-risk-lab/
├── pyproject.toml             # uv project definition + deps
├── .python-version            # 3.11 (uv pin)
├── README.md
├── MIGRATION.md               # Jupyter → Marimo migration log
├── cli.py                     # pipeline CLI
├── .gitignore                 # *.ipynb PERMANENTLY blocked
├── LICENSE
│
├── utils/                     # shared modules
│   ├── metrics.py             # KS, Gini, AUC, PSI, CAP
│   ├── preprocessing.py       # WOE/IV binning
│   └── plotting.py            # CAP, ROC, score distribution
│
├── tests/
│
├── 01_credit_scoring/
│   ├── data/download.sh
│   ├── notebooks/
│   │   └── 01_eda_scorecard.py    ← Marimo notebook
│   ├── reports/.gitkeep
│   └── src/scorecard.py
│
├── 02_pd_lgd_estimation/
│   └── notebooks/02_pd_lgd_panel.py
├── 03_concentration_risk/
│   └── notebooks/03_hhi_concentration.py
├── 04_basel_capital_simulator/
│   └── notebooks/04_rwa_simulator.py
└── 05_eda_sbs_ecuador/
    └── notebooks/05_sbs_eda.py
```

---

## 📐 Validation Framework

| Metric | Threshold (Good) | Interpretation |
|--------|-----------------|----------------|
| **KS Statistic** | > 0.30 | Max separation between good/bad distributions |
| **Gini Coefficient** | > 0.40 | Discriminatory power (2·AUC − 1) |
| **AUC-ROC** | > 0.70 | Area under ROC curve |
| **PSI** | < 0.10 | Population Stability Index (drift monitoring) |
| **CAP Ratio** | > 0.60 | Cumulative Accuracy Profile |

---

## 🔗 Data Sources

- [UCI — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
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
  <i>Built with rigorous methodology. Every model is validated before it's trusted.</i>
</p>
