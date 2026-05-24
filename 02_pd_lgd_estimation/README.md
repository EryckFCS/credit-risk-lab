# Case Study 02 — PD & LGD Estimation: Sector Financiero Ecuador

> Estimación de Probabilidad de Default (PD) y Loss Given Default (LGD) con datos macro-financieros públicos de Ecuador, replicando el enfoque IRB (Internal Ratings-Based) de Basilea II/III adaptado al sistema financiero ecuatoriano.

**Scope:** Sistema financiero ecuatoriano · datos supervisores / públicos · enfoque macro-credit  
**Status:** 📋 Planned

---

## Objetivo

Estimar la **Probabilidad de Default (PD)** y la **Loss Given Default (LGD)** en el sector financiero ecuatoriano usando datos de panel del Banco Mundial, SEPS y BCE. El caso avanza desde el scorecard retail (Case Study 01) hacia un **marco de riesgo de crédito macro-supervisorial**, relevante para:

- roles de analista en banca local,
- reporting de estilo supervisorial,
- análisis de deterioro crediticio macro-financiero,
- pensamiento Basilea / IFRS 9 en mercados emergentes.

---

## Datasets

| Dataset | Fuente | Variables clave | Descarga |
|---------|--------|-----------------|----------|
| **Crédito sector privado / PIB** | [Banco Mundial](https://datos.bancomundial.org/indicador/FS.AST.PRVT.GD.ZS) | `credit_pct_gdp` (1994-2024) | API automática |
| **NPL Ratio (cartera vencida)** | [Banco Mundial](https://datos.bancomundial.org/indicador/FB.AST.NPER.ZS) | `npl_ratio` — proxy de PD sectorial | API automática |
| **Tasa activa referencial** | [Banco Mundial](https://datos.bancomundial.org/indicador/FR.INR.LEND) | `lending_rate` — costo de crédito | API automática |
| **Colocaciones cooperativas** | [SEPS Ecuador](https://estadisticas.seps.gob.ec/index.php/estadisticas-sfps/) | Saldo cartera por segmento y tipo | Manual CSV |
| **MOA — Monto de Operaciones Activas** | [BCE 2024](https://contenido.bce.fin.ec/documentos/Estadisticas/SectorMonFin/TasasInteres/informe_moa_2024.pdf) | Tasas y volumen por segmento | PDF manual |

---

## Setup

```bash
bash 02_pd_lgd_estimation/data/download.sh   # descarga Banco Mundial vía API
# Luego descargar SEPS manualmente (ver instrucciones en el script)
```

---

## Methodology Roadmap

```text
Supervisory Data Collection (SBS / BCE / Banco Mundial)
        ↓
Panel Construction: banco × período
        ↓
Target Definition: proxy de delinquency / cartera improductiva
        ↓
Macroeconomic Feature Merge (GDP, inflation, tasa activa)
        ↓
PD Model Estimation
    ├── Logit(NPL_ratio) ~ crédito/PIB + tasa_activa + GDP_growth
    ├── Panel model (bank fixed effects)
    └── Macro stress sensitivity
        ↓
LGD Model
    └── Beta Regression sobre recovery rates implícitos
        ↓
Stress Testing
    └── shock tasa (+300bps) + caída PIB (-3%)
        ↓
Validation
    ├── rank ordering
    ├── calibration
    ├── temporal stability
    └── scenario interpretation
        ↓
Reporte HTML final
```

---

## Planned Notebooks

| # | Notebook | Contenido |
|---|---|---|
| 01 | `01_data_collection_sbs.py` | fetch, clean y standardize series SBS/BCE/BM |
| 02 | `02_panel_construction.py` | construir panel banco × período y target PD |
| 03 | `03_eda_macro_credit.py` | diagnósticos macro-credit y segmentación |
| 04 | `04_pd_model_baseline.py` | estimación PD baseline |
| 05 | `05_lgd_beta_regression.py` | Beta Regression sobre LGD |
| 06 | `06_panel_stress_testing.py` | sensibilidad panel y stress scenarios |
| 07 | `07_final_report_export.py` | reporte HTML + assets de presentación |

---

## Modeling Philosophy

Este no es un scorecard de aplicación retail. Es un **marco PD/LGD macro-supervisorial**. Las prioridades de diseño difieren:

- consistencia temporal > fit cross-sectional,
- explicabilidad > complejidad black-box,
- sensibilidad macro > discriminación estática,
- robustez ante restricciones de pequeña muestra local.

---

## Contexto Ecuador

El NPL ratio del sistema bancario ecuatoriano subió de ~2.5% (2019) a ~4.1% (2020) durante el shock COVID-19, estabilizándose en ~3.2% (2023). Contexto regulatorio: **Resolución SB-2022-0517** sobre gestión de riesgo de crédito.

Variables iniciales a rastrear: ratio NPL, cartera vencida, cobertura de provisiones, crecimiento del crédito, composición de cartera, tamaño banco / concentración, inflación, proxy de actividad doméstica, proxy de tasa/liquidez, controles de shock externo.

---

## Expected Technical Challenges

- datos públicos fragmentados en XLS/XLSX/PDF,
- definiciones de series pueden cambiar en el tiempo,
- la PD objetivo puede requerir proxy desde deterioro de delinquency,
- profundidad del panel limitada para instituciones pequeñas,
- estacionariedad y quiebres estructurales son más críticos que en datasets de consumo.

---

## Referencias

- Basel Committee on Banking Supervision (2005). *Guidance on Paragraph 468 of the Framework Document*.
- Banco Mundial — World Development Indicators: [FS.AST.PRVT.GD.ZS](https://datos.bancomundial.org/indicador/FS.AST.PRVT.GD.ZS)
- SEPS Ecuador (2024). *Estadísticas del Sistema Financiero Popular y Solidario*.
- BCE (2024). *Boletín Analítico Anual MOA*. [PDF](https://contenido.bce.fin.ec/documentos/Estadisticas/SectorMonFin/TasasInteres/informe_moa_2024.pdf)
