# Case Study 03 — Concentration Risk: Sistema Bancario Ecuador

## Objetivo

Medir y analizar el **riesgo de concentración** del sistema bancario ecuatoriano usando el **Índice Herfindahl-Hirschman (HHI)**, el ratio CR5 (concentración de los 5 mayores bancos) y el indicador del Banco Mundial de activos de los 3 mayores bancos sobre activos totales. Se compara con el panel LATAM (Colombia, Perú, Chile, México, Brasil).

## Datasets

| Dataset | Fuente | Variables | Descarga |
|---------|--------|-----------|----------|
| **Top-3 bank assets (% total)** | [Banco Mundial FB.BNK.CAPA.ZS](https://datos.bancomundial.org/indicador/FB.BNK.CAPA.ZS) | Concentración bancaria Ecuador 1996-2023 | API automática |
| **Comparación LATAM** | [Banco Mundial — panel EC,CO,PE,CL,MX,BR](https://datos.bancomundial.org/indicador/FB.BNK.CAPA.ZS) | Benchmarking regional | API automática |
| **Ranking empresas 2024** | [SuperCías Ecuador](https://appscvsmovil.supercias.gob.ec/ranking/reporte.html) | Activos/patrimonio por sector CIIU | Manual CSV |
| **Boletín financiero SBS** | [SBS Portal Estadístico](https://www.superbancos.gob.ec/estadisticas/portalestudios/) | Cartera por entidad | Manual |

## Setup

```bash
bash 03_concentration_risk/data/download.sh
```

## Metodología

```
1. HHI sectorial      → Σ (participación_i)² por segmento de crédito
2. CR5 bancario       → suma de cuotas de los 5 mayores bancos
3. Serie temporal     → evolución concentración Ecuador 2000-2023
4. Benchmarking LATAM → comparación regional con panel Banco Mundial
5. Stress test        → impacto de quiebra del banco más grande sobre el sistema
6. Interpretación     → contexto regulatorio SBS / resolución de crisis 1999
```

## Relevancia Ecuador

Ecuador vivió una crisis bancaria sistémica en **1999** con el congelamiento de depósitos. La concentración bancaria actual (los 3 mayores bancos concentran ~60% de activos) es un factor de riesgo sistémico monitoreado por la SBS. Este caso contextualiza esos datos con metodología cuantitativa.

## Referencias

- Rhoades, S.A. (1993). *The Herfindahl-Hirschman Index*. Federal Reserve Bulletin.
- Basel Committee (2019). *Supervisory Framework for Measuring and Controlling Large Exposures*.
- SBS Ecuador (2024). [Portal Estadístico](https://www.superbancos.gob.ec/estadisticas/portalestudios/)
- Banco Mundial. [Indicador FB.BNK.CAPA.ZS](https://datos.bancomundial.org/indicador/FB.BNK.CAPA.ZS)
