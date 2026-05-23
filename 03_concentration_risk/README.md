# Case Study 03 — Riesgo de Concentración (SBS Ecuador)

## Objetivo

Medir y estressar el **riesgo de concentración crediticia** del sistema financiero ecuatoriano
usando métricas de supervisión bancaria internacional: HHI, CR5, Índice de Herfindahl
normalizado y el framework de capital adicional por concentración del BCBS.

## Hipótesis de investigación

> Los bancos privados ecuatorianos con alta concentración sectorial (HHI > 1800) exhiben
> mayor volatilidad de morosidad bajo escenarios de estrés que carteras diversificadas,
> consistente con las guías de concentración del Comité de Basilea (BCBS 2006).

## Dataset

| Fuente | Variable | Frecuencia | Período |
|--------|----------|------------|---------|
| SBS — Volumen de crédito | Cartera por sector económico (CIIU) | Mensual | 2018–2024 |
| SBS — Boletines mensuales | Cartera bruta por institución y segmento | Mensual | 2018–2024 |
| BCE — Cuentas Nacionales | Participación sectorial en PIB | Anual | 2018–2024 |
| Sintético calibrado | Distribución sectorial proxy SBS | Mensual | 2018–2024 |

**Sectores CIIU modelados (Top 8 por participación cartera SBS):**
- Comercio al por mayor y menor
- Agricultura, ganadería, silvicultura
- Industria manufacturera
- Construcción
- Transporte y logística
- Actividades financieras
- Servicios varios
- Consumo personas naturales

## Pipeline

```
01_data_sectors.py    → cartera por sector × institución → sector_panel.parquet
02_hhi_metrics.py     → HHI, CR5, HHI norm, índice Berry → concentration_metrics.parquet
03_stress_testing.py  → escenarios Basel, capital add-on, dashboard supervisorio
```

## Metodología

### Índice de Herfindahl-Hirschman (HHI)

\[
HHI = \sum_{i=1}^{N} s_i^2 \quad \text{donde } s_i = \frac{\text{cartera}_i}{\text{cartera total}}
\]

Interpretación regulatoria:
- HHI < 1000: mercado competitivo / cartera diversificada
- 1000 ≤ HHI < 1800: concentración moderada
- HHI ≥ 1800: concentración alta — capital add-on Basel

### CR5 (Concentration Ratio Top-5)

\[
CR5 = \sum_{i=1}^{5} s_i \quad \text{(top 5 sectores por participación)}
\]

### Stress Testing — 3 Escenarios Basel

| Escenario | Shock morosidad | PIB | Desempleo | Descripción |
|-----------|----------------|-----|-----------|-------------|
| Base | +0% | +2.0% | 4.8% | Condiciones actuales |
| Adverse | +150bps | -1.5% | +2.0pp | Recesión moderada |
| Severely Adverse | +400bps | -4.0% | +5.0pp | Crisis sistémica (COVID-like) |

### Capital Add-on por Concentración (BCBS 2006 §773)

\[
\Delta K_{conc} = K_{base} \times \left(\frac{HHI}{HHI_{ref}} - 1\right) \times \phi
\]

donde φ es el factor de ajuste sectorial (0.10–0.25 según el segmento).

## Referencias

- BCBS (2006). *International Convergence of Capital Measurement and Capital Standards* (Basel II). BIS.
- BCBS (2017). *Basel III: Finalising post-crisis reforms*. BIS.
- Düllmann, K. & Masschelein, N. (2007). *A Tractable Model to Measure Sector Concentration Risk in Credit Portfolios*. Journal of Financial Services Research.
- Gordy, M. (2003). *A Risk-Factor Model Foundation for Ratings-Based Bank Capital Rules*. Journal of Financial Intermediation.
- SBS Ecuador. *Volumen de Crédito*. https://www.superbancos.gob.ec/estadisticas/portalestudios/volumen-de-credito/
