# Case Study 04 — Basel III Capital Simulator

## Objetivo

Implementar una **calculadora interactiva de capital regulatorio** bajo el framework
Basel III, cubriendo los tres pilares cuantitativos principales:
- **Pilar 1:** Capital mínimo — RWA, Tier 1, Tier 2, CAR
- **Liquidez:** LCR (Liquidity Coverage Ratio) y NSFR (Net Stable Funding Ratio)
- **Stress Test:** Impacto en ratios bajo escenarios adversos

## Framework Regulatorio

### Capital Requirements (BCBS 2017 — Basel III Finalised)

| Ratio | Fórmula | Mínimo Regulatorio |
|-------|---------|--------------------|
| CET1 | CET1 Capital / RWA | ≥ 4.5% |
| Tier 1 | (CET1 + AT1) / RWA | ≥ 6.0% |
| Total Capital | (Tier1 + Tier2) / RWA | ≥ 8.0% |
| Capital Conservation Buffer | CET1 adicional / RWA | ≥ 2.5% |
| **CAR Total con buffers** | | **≥ 10.5%** |

### Liquidity Ratios

| Ratio | Fórmula | Mínimo |
|-------|---------|--------|
| LCR | HQLA / Net Cash Outflows (30d) | ≥ 100% |
| NSFR | Available Stable Funding / Required Stable Funding | ≥ 100% |

### Risk-Weighted Assets (Standardised Approach)

| Tipo exposición | Risk Weight |
|-----------------|-------------|
| Soberanos (AAA–AA) | 0% |
| Soberanos (A) | 20% |
| Bancos (A) | 50% |
| Corporativas (BBB+) | 100% |
| Retail / Consumo | 75% |
| Hipotecario (LTV ≤ 60%) | 35% |
| Hipotecario (LTV > 80%) | 100% |
| Default / Vencido | 150% |
| Equity | 250% |

## Pipeline

```
01_rwa_calculator.py    → RWA por tipo de exposición → capital_base.parquet
02_liquidity_ratios.py  → LCR, NSFR, HQLA → liquidity_metrics.parquet
03_capital_dashboard.py → Dashboard interactivo Basel III completo
```

## Referencias

- BCBS (2017). *Basel III: Finalising post-crisis reforms*. BIS.
- BCBS (2013). *Basel III: The Liquidity Coverage Ratio and liquidity risk monitoring tools*. BIS.
- BCBS (2014). *Basel III: the Net Stable Funding Ratio*. BIS.
- SBS Ecuador. *Normas de Solvencia y Patrimonio Técnico*. Resolución No. SB-2021-0565.
