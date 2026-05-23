# Case Study 02 — PD Estimation: Macro Panel Data (SBS Ecuador)

## Objetivo

Estimar la **Probabilidad de Default (PD)** a nivel de segmento de cartera usando datos
macro-financieros públicos de la Superintendencia de Bancos del Ecuador (SBS) y el
Banco Central del Ecuador (BCE), modelados como un **panel de datos longitudinal**.

## Hipótesis de investigación

> Los ciclos macroeconómicos (tasa activa, desempleo, crecimiento del PIB) y
> variables de composición de cartera (concentración sectorial, plazo promedio) son
> determinantes estadísticamente significativos de la morosidad sistémica en Ecuador.

## Dataset

| Fuente | Variable | Frecuencia | Período |
|--------|----------|------------|--------|
| SBS — Boletines mensuales | Índice de morosidad por tipo de crédito | Mensual | 2015–2024 |
| SBS — Volumen de crédito | Cartera bruta, cartera vencida, cartera en riesgo | Mensual | 2015–2024 |
| BCE — Estadísticas | Tasa activa referencial, Tasa pasiva, Inflación | Mensual | 2015–2024 |
| BCE — Cuentas Nacionales | Crecimiento PIB real (interpolado mensual) | Trimestral→Mensual | 2015–2024 |
| INEC | Tasa de desempleo nacional | Trimestral→Mensual | 2015–2024 |

**Segmentos SBS modelados:**
- Crédito Comercial Ordinario
- Crédito de Consumo Ordinario
- Crédito Inmobiliario
- Microcrédito

## Pipeline

```
01_data_ingestion.py   → descarga/parsea SBS+BCE → panel_raw.parquet
02_feature_panel.py    → lags, diferencias, Hodrick-Prescott, dummies → panel_features.parquet
03_panel_logit.py      → Logit pooled + FE + RE, Hausman test, IRF → outputs/
```

## Metodología

### Transformación de morosidad → PD binaria / continua

Dos especificaciones paralelas:
1. **PD continua**: `morosidad_t = f(X_{t-1}, ..., X_{t-k})` — OLS/GLS con efectos fijos
2. **PD binaria**: `default_t = 1 si morosidad_t > umbral_p75` — Logit panel con FE

### Variables explicativas clave

| Variable | Transformación | Hipótesis |
|----------|---------------|----------|
| Tasa activa | Nivel + 1er lag | ↑ tasa → ↑ carga financiera → ↑ default |
| Crecimiento PIB | Ciclo HP (gap) | Recesión → ↑ default |
| Desempleo | 1era diferencia | ↑ desempleo → ↑ default consumo |
| Inflación | Nivel | ↑ inflación → erosión ingreso real |
| Concentración sectorial | HHI cartera | ↑ concentración → ↑ riesgo sistémico |
| Cartera en riesgo rezagada | AR(1) | Persistencia del ciclo de crédito |

### Selección de modelo: Hausman test

```
H0: efectos individuales no correlacionados con regresores (RE consistente)
H1: correlación existe (FE consistente, RE inconsistente)
Decisión: p < 0.05 → usar Fixed Effects
```

## Targets de validación

| Métrica | Umbral mínimo | Referencia |
|---------|--------------|------------|
| Pseudo R² (McFadden) | ≥ 0.20 | Hosmer & Lemeshow (2000) |
| AUC-ROC | ≥ 0.70 | Basel BIS WP No.14 |
| Breusch-Godfrey (autocorr.) | p > 0.05 | Greene (2018) |
| Wooldridge test (serial corr.) | p > 0.05 | Wooldridge (2002) |

## Referencias

- Jiménez, G. & Saurina, J. (2006). *Credit Cycles, Credit Risk, and Prudential Regulation*. International Journal of Central Banking.
- Pesaran, M. H. (2015). *Time Series and Panel Data Econometrics*. Oxford University Press.
- BIS (2005). *Studies on the Validation of Internal Rating Systems*. Working Paper No. 14.
- SBS Ecuador. *Boletines Financieros Mensuales*. https://www.superbancos.gob.ec/estadisticas/portalestudios/
- BCE Ecuador. *Estadísticas Macroeconómicas*. https://www.bce.fin.ec/
