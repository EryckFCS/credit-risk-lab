# Case Study 01 — Credit Scoring

> End-to-end credit scorecard following industry-standard methodology

**Dataset:** UCI Default of Credit Card Clients · N = 30,000 · Taiwan 2005  
**Status:** ✅ Complete

---

## Methodology

```
Raw Data → EDA & Profiling → Feature Engineering (WoE/IV)
       → Logistic Regression → PDO Scorecard Scaling
       → Benchmarking (RF · XGBoost) → SHAP Interpretability
       → Regulatory Decision Matrix → HTML Report
```

## Notebooks

| # | Notebook | Content |
|---|----------|---------|
| 01 | `01_eda_and_profiling.ipynb` | Distribution analysis, missing values, default rate by segment |
| 02 | `02_feature_engineering_woe.ipynb` | WoE binning, IV table, 7 engineered features, train/test split |
| 03 | `03_modeling_logistic_regression.ipynb` | LR pipeline, KS/Gini/AUC/CAP/PSI, calibration, scorecard scaling |
| 04 | `04_model_comparison.ipynb` | LR vs RF vs XGBoost · SHAP global/beeswarm/waterfall · regulatory matrix |
| 05 | `05_final_report_export.ipynb` | Self-contained HTML report + LinkedIn post generator |

## Results (Champion — LR Scorecard, test set)

| Metric | Value | Industrial Threshold | Status |
|--------|-------|---------------------|--------|
| KS | _run NB03_ | ≥ 0.30 | — |
| Gini | _run NB03_ | ≥ 0.40 | — |
| AUC | _run NB03_ | ≥ 0.70 | — |
| PSI | _run NB07_ | < 0.10 | — |
| Brier | _run NB03_ | < 0.15 | — |

> Run notebooks 01–05 in sequence to populate results.

## Scorecard Parameters

```
Base Score = 600  ·  Base Odds = 50:1  ·  PDO = 20
Factor = PDO / ln(2) = 28.85
Offset = Base Score - Factor × ln(Base Odds) = 487.12
Score = Offset + Factor × ln(Odds)  →  Higher score = lower risk
```

## Key Features (WoE ranked by IV)

- `PAY_0` — most recent payment status (strongest predictor)
- `utilization` — avg_bill / limit_bal (engineered)
- `n_delinquent_months` — count of months with delay ≥ 1 (engineered)
- `max_delay` — maximum delay severity in 6 months (engineered)
- `pay_ratio` — avg_payment / avg_bill (engineered)

## How to Run

```bash
# 1. Install dependencies
pip install -r ../../requirements.txt

# 2. Run notebooks in order
jupyter lab  # open and run 01 → 05 sequentially

# 3. View report
open reports/credit_scoring_report.html
```

## References

- Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley.
- Thomas, L., Edelman, D., Crook, J. (2002). *Credit Scoring and Its Applications*. SIAM.
- Yeh, I.C. & Lien, C. (2009). *The comparisons of data mining techniques*. Expert Systems with Applications.
- Basel Committee (2006). *International Convergence of Capital Measurement* (Basel II).
