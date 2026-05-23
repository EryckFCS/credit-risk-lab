# Case Study 02 — PD & LGD Estimation: Sector Financiero Ecuador

## Objetivo

Estimar la **Probabilidad de Default (PD)** y la **Loss Given Default (LGD)** usando datos macro-financieros públicos de Ecuador, aplicando modelos de panel logit y regresión beta. El caso replica el enfoque IRB (Internal Ratings-Based) de Basilea II/III adaptado al contexto del sistema financiero ecuatoriano.

## Datasets

| Dataset | Fuente | Variables clave | Descarga |
|---------|--------|-----------------|----------|
| **Crédito sector privado / PIB** | [Banco Mundial](https://datos.bancomundial.org/indicador/FS.AST.PRVT.GD.ZS) | `credit_pct_gdp` (1994-2024) | API automática |
| **NPL Ratio (cartera vencida)** | [Banco Mundial](https://datos.bancomundial.org/indicador/FB.AST.NPER.ZS) | `npl_ratio` — proxy de PD sectorial | API automática |
| **Tasa activa referencial** | [Banco Mundial](https://datos.bancomundial.org/indicador/FR.INR.LEND) | `lending_rate` — costo de crédito | API automática |
| **Colocaciones cooperativas** | [SEPS Ecuador](https://estadisticas.seps.gob.ec/index.php/estadisticas-sfps/) | Saldo cartera por segmento y tipo | Manual CSV |
| **MOA — Monto de Operaciones Activas** | [BCE 2024](https://contenido.bce.fin.ec/documentos/Estadisticas/SectorMonFin/TasasInteres/informe_moa_2024.pdf) | Tasas y volumen por segmento | PDF manual |

## Setup

```bash
bash 02_pd_lgd_estimation/data/download.sh   # descarga Banco Mundial vía API
# Luego descargar SEPS manualmente (ver instrucciones en el script)
```

## Metodología

```
1. Construcción del panel macro     → merge Banco Mundial + SEPS + BCE
2. Análisis de estacionariedad      → ADF test, PP test sobre series NPL y crédito/PIB
3. Modelo PD sectorial              → Logit(NPL_ratio) ~ crédito/PIB + tasa_activa + GDP_growth
4. Modelo LGD                       → Beta Regression sobre recovery rates implícitos
5. Stress Testing                   → simulación shock de tasa (+300bps) y caída PIB (-3%)
6. Reporte                          → HTML con visualizaciones de serie temporal
```

## Relevancia Ecuador

El NPL ratio del sistema bancario ecuatoriano subió de ~2.5% (2019) a ~4.1% (2020) durante el shock COVID-19, y se ha estabilizado en ~3.2% (2023). Este caso estudia la dinámica de ese ciclo usando datos del Banco Mundial y SEPS. Contexto regulatorio: **Resolución SB-2022-0517** sobre gestión de riesgo de crédito.

## Referencias

- Basel Committee on Banking Supervision (2005). *Guidance on Paragraph 468 of the Framework Document*.
- Banco Mundial — World Development Indicators: [FS.AST.PRVT.GD.ZS](https://datos.bancomundial.org/indicador/FS.AST.PRVT.GD.ZS)
- SEPS Ecuador (2024). *Estadísticas del Sistema Financiero Popular y Solidario*.
- BCE (2024). *Boletín Analítico Anual MOA*. [PDF](https://contenido.bce.fin.ec/documentos/Estadisticas/SectorMonFin/TasasInteres/informe_moa_2024.pdf)
