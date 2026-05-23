# Case Study 02 — PD Estimation (SBS Ecuador)

> Probability of Default estimation with supervisory banking data from Ecuador

**Scope:** Ecuadorian banking system · supervisory / public aggregate data  
**Status:** 🚧 In Progress

---

## Objective

Estimate **Probability of Default (PD)** using Ecuadorian supervisory data, moving from a generic retail scorecard dataset to a **local macro-credit risk framework**. The point of this case study is not only predictive power, but **contextual relevance**: prudential regulation, macro sensitivity, concentration risk, and supervisory interpretability.

## Strategic Rationale

This case study is the real differentiator of the portfolio.

Most junior credit-risk portfolios stop at the UCI dataset. This one advances toward a **country-relevant framework** using Ecuadorian public financial data, which is more aligned with:

- local banks' internship and analyst roles,
- supervisory-style reporting,
- macro-financial credit deterioration analysis,
- Basel / IFRS 9 thinking in an emerging market context.

## Methodology Roadmap

```text
Supervisory Data Collection (SBS / BCE)
        ↓
Panel Construction by bank × period
        ↓
Target Definition: delinquency / impaired portfolio proxy
        ↓
Macroeconomic Feature Merge
        ↓
PD Model Estimation
    ├── Logistic / Fractional response baseline
    ├── Panel model (bank fixed effects)
    └── Macro stress sensitivity
        ↓
Validation
    ├── rank ordering
    ├── calibration
    ├── temporal stability
    └── scenario interpretation
```

## Planned Datasets

| Source | Use |
|---|---|
| SBS Ecuador | delinquency, cartera bruta, cartera improductiva, provisions, bank-level aggregates |
| BCE Ecuador | GDP proxy, inflation, liquidity / credit aggregates, external conditions |
| Internal synthetic panel transforms | lagged features, growth rates, stress variables |

## Planned Outputs

- panel dataset bank × month (or quarter)
- delinquency proxy and PD target construction note
- exploratory macro-credit dashboard
- baseline PD model
- challenger panel specification
- stress scenario note for portfolio deterioration
- final HTML report

## Folder Logic

```text
02_pd_estimation/
├── data/
├── notebooks/
├── reports/
└── README.md
```

## Planned Notebooks

| # | Notebook | Content |
|---|---|---|
| 01 | `01_data_collection_sbs.ipynb` | fetch, clean and standardize SBS series |
| 02 | `02_panel_construction.ipynb` | build bank × period panel and target |
| 03 | `03_eda_macro_credit.ipynb` | macro-credit diagnostics and segmentation |
| 04 | `04_pd_model_baseline.ipynb` | baseline PD estimation |
| 05 | `05_panel_and_stress_testing.ipynb` | panel sensitivity and stress scenarios |
| 06 | `06_final_report_export.ipynb` | HTML report + presentation assets |

## Modeling Philosophy

This is **not** a retail application scorecard. It is a **portfolio / supervisory PD framework**. Therefore the design priorities differ:

- temporal consistency over cross-sectional fit,
- explainability over black-box complexity,
- macro sensitivity over static discrimination,
- robustness to small-sample local data constraints.

## Initial Variables to Track

- non-performing loans ratio
- overdue portfolio ratio
- provision coverage
- credit growth
- portfolio composition
- bank size / concentration
- inflation
- domestic activity proxy
- interest-rate / liquidity proxy
- external shock controls

## Expected Technical Challenges

- public data may be fragmented across XLS/XLSX/PDF bulletins,
- series definitions may change through time,
- target PD may need to be proxied from delinquency deterioration,
- panel depth may be limited for smaller institutions,
- stationarity and structural breaks will matter more than in consumer datasets.

## Research Positioning

The end product should read like a junior **bank risk analytics note**, not a Kaggle notebook.
