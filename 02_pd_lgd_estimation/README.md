# Case Study 02 — PD & LGD Estimation: SME Lending

## Overview

Estimation of Probability of Default (PD) and Loss Given Default (LGD) for Small and Medium Enterprise (SME) lending portfolios, aligned with Basel III Internal Ratings-Based (IRB) approach requirements.

## Status

📋 **Planned** — Development starts after Case Study 01 is complete.

## Methodology (Planned)

| Component | Method |
|-----------|--------|
| PD Model | Panel Logit with time fixed effects, Survival Analysis (Cox PH) |
| LGD Model | Beta Regression, Tobit model |
| Validation | Brier Score, Hosmer-Lemeshow, Binomial test |
| Backtesting | Traffic-light approach (Basel III Annex) |

## Dataset (Planned)

- SBS Ecuador open credit data
- Synthetic SME portfolio calibrated to Ecuador market conditions

## References

- Basel Committee on Banking Supervision (2006). *International Convergence of Capital Measurement and Capital Standards (Basel II).* BIS.
- Schuermann, T. (2004). *What Do We Know About Loss Given Default?* Wharton Financial Institutions Center.
