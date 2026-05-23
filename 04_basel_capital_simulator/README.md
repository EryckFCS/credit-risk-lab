# Case Study 04 — Basel III Capital Requirement Simulator

## Objetivo

Simular el cálculo de **Requerimientos de Capital Mínimo** bajo el **Enfoque Estándar de Basilea III**, usando indicadores reales del sistema bancario ecuatoriano (Banco Mundial) y parámetros calibrados con la normativa de la SBS Ecuador. El simulador calcula RWA, Tier 1 Capital, CAR y lo compara con los mínimos regulatorios.

## Datasets

| Dataset | Fuente | Variables | Descarga |
|---------|--------|-----------|----------|
| **Capital/Activos** | [Banco Mundial FB.BNK.CAPA.ZS](https://datos.bancomundial.org/indicador/FB.BNK.CAPA.ZS) | Solvencia bancaria Ecuador | API automática |
| **ROA bancario** | [Banco Mundial FB.BNK.ROVG.ZS](https://datos.bancomundial.org/indicador/FB.BNK.ROVG.ZS) | Rentabilidad activos | API automática |
| **Provisiones / NPL** | [Banco Mundial FB.BNK.RESL.BS.ZS](https://datos.bancomundial.org/indicador/FB.BNK.RESL.BS.ZS) | Cobertura provisiones | API automática |
| **Parámetros RWA** | SBS Ecuador — Res. SB-2022-0517 | Ponderadores de riesgo | Normativa pública |

## Setup

```bash
bash 04_basel_capital_simulator/data/download.sh
```

## Metodología

```
1. Carga indicadores CAMEL Ecuador   → capital, ROA, provisiones (Banco Mundial API)
2. RWA Calculation                   → cartera crédito × ponderador SA (0-150%)
3. Capital mínimo requerido          → RWA × 8% (Basilea III) + colchón conservación 2.5%
4. CAR comparación                   → CAR observado vs. mínimo regulatorio (SBS: 9%)
5. Stress test                       → shock de crédito: default +5pp → impacto en CAR
6. Dashboard interactivo             → Plotly: evolución CAR Ecuador 2000-2023
```

## Marco Regulatorio Ecuador

La SBS Ecuador establece un **CAR mínimo del 9%** (superior al 8% de Basilea III), regulado por la **Resolución SB-2022-0517**. El sistema bancario ecuatoriano operó con CAR promedio de ~13-14% en 2023, pero con alta heterogeneidad entre entidades. Este caso analiza esa brecha.

## Referencias

- Basel Committee on Banking Supervision (2017). *Basel III: Finalising Post-Crisis Reforms*.
- SBS Ecuador (2022). *Resolución SB-2022-0517 — Gestión de Riesgo de Crédito*.
- Banco Mundial. [FB.BNK.CAPA.ZS](https://datos.bancomundial.org/indicador/FB.BNK.CAPA.ZS)
