# Case Study 05 — EDA: Sistema Financiero Nacional Ecuador

## Objetivo

Análisis exploratorio completo del **Sistema Financiero Nacional (SFN) de Ecuador** usando datos públicos del Banco Mundial (WDI), la SBS, la SEPS y el BCE. Produce un diagnóstico cuantitativo del estado del crédito, la morosidad, la solvencia y la inclusión financiera en Ecuador con contexto regional LATAM.

## Datasets (100% públicos, descargables automáticamente)

| Indicador | Fuente | Código | Período |
|-----------|--------|--------|---------|
| Crédito sector privado (% PIB) | [Banco Mundial](https://datos.bancomundial.org/indicador/FS.AST.PRVT.GD.ZS) | FS.AST.PRVT.GD.ZS | 1994–2024 |
| NPL ratio (cartera vencida %) | [Banco Mundial](https://datos.bancomundial.org/indicador/FB.AST.NPER.ZS) | FB.AST.NPER.ZS | 2000–2023 |
| Capital adequacy (capital/activos) | [Banco Mundial](https://datos.bancomundial.org/indicador/FB.BNK.CAPA.ZS) | FB.BNK.CAPA.ZS | 2000–2023 |
| ROA bancario | [Banco Mundial](https://datos.bancomundial.org/indicador/FB.BNK.ROVG.ZS) | FB.BNK.ROVG.ZS | 2000–2023 |
| Tasa activa referencial | [Banco Mundial](https://datos.bancomundial.org/indicador/FR.INR.LEND) | FR.INR.LEND | 2000–2023 |
| Tasa pasiva referencial | [Banco Mundial](https://datos.bancomundial.org/indicador/FR.INR.DPST) | FR.INR.DPST | 2000–2023 |
| Sucursales / 100k adultos | [Banco Mundial](https://datos.bancomundial.org/indicador/FB.CBK.BRCH.P5) | FB.CBK.BRCH.P5 | 2004–2022 |
| Estadísticas cooperativas (SFPS) | [SEPS Ecuador](https://estadisticas.seps.gob.ec/index.php/estadisticas-sfps/) | — | Manual CSV |

## Setup

```bash
bash 05_eda_sbs_ecuador/data/download.sh   # descarga 7 indicadores WB + construye panel
```

## Metodología

```
1. Panel construction     → merge 7 indicadores WB para Ecuador (1994–2024)
2. Data quality audit     → missing values, outliers, breaks de serie (crisis 1999, 2008, COVID)
3. Descriptive stats      → evolución crédito/PIB, NPL, spread activo-pasivo
4. Crisis analysis        → marcar períodos: crisis 1999, 2008, pandemia 2020
5. LATAM benchmark        → comparar Ecuador vs. CO, PE, CL, MX en NPL y crédito/PIB
6. Inclusión financiera   → sucursales por 100k adultos: Ecuador vs. región
7. Informe HTML           → reporte ejecutivo listo para LinkedIn
```

## Hallazgos Esperados

- El crédito al sector privado en Ecuador (~40% PIB) está por debajo del promedio LATAM (~55% PIB), indicando espacio de profundización financiera.
- El NPL ratio post-COVID (2020: ~4.1%) fue contenido vs. la crisis 1999 (estimado >15%).
- El spread tasa activa–pasiva en Ecuador es alto (~7-9pp) comparado con Chile (~3pp), reflejando menor competencia y mayor riesgo percibido.

## Referencias

- Banco Mundial — World Development Indicators (WDI): [datos.bancomundial.org](https://datos.bancomundial.org)
- SEPS Ecuador (2024). *Estadísticas del SFPS*. [estadisticas.seps.gob.ec](https://estadisticas.seps.gob.ec)
- BCE Ecuador. *Estadísticas Monetarias y Financieras*. [bce.fin.ec](https://contenido.bce.fin.ec/documentos/informacioneconomica/MonetarioFinanciero/ix_MonetariasFinancierasPrin.html)
- Carvajal, A. & Wynter, B. (2004). *Ecuador: Financial System Stability Assessment*. IMF Country Report 04/190.
