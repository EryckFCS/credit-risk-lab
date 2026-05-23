# Jupyter → Marimo Migration Log

## Estado: COMPLETADO

Este repositorio fue construido **nativamente en Marimo** desde el inicio.
Nunca existieron archivos `.ipynb` — no hubo conversión, solo adopción directa del estándar correcto.

---

## Por qué Marimo

| Criterio | Jupyter `.ipynb` | Marimo `.py` |
|---|---|---|---|
| **Git diff** | JSON binario ilegible | Diff limpio de Python puro |
| **Estado oculto** | Celdas fuera de orden producen bugs silenciosos | DAG garantiza ejecución determinista |
| **Reprodución** | Requiere kernel activo + re-run manual | `marimo run` es un comando único |
| **Versionado** | `.ipynb` en Git es ruido | `.py` es código soberano |
| **Compartir** | Requiere servidor Jupyter | Export WASM → HTML estático, sin servidor |

---

## Blindaje permanente en `.gitignore`

```gitignore
# JUPYTER — PERMANENTLY BLOCKED
*.ipynb
.ipynb_checkpoints/
**/.ipynb_checkpoints/
```

Cualquier `.ipynb` que aparezca en un PR es un error de proceso.

---

## Dependencias eliminadas

En `v0.3.0` se purgaron las siguientes dependencias legacy del `pyproject.toml`:

```
nbconvert   ✔ ELIMINADO
nbformat    ✔ ELIMINADO
jinja2      ✔ ELIMINADO  (era dependencia de nbconvert)
```

Reemplazadas por:

```
marimo>=0.13.0   ✔ ACTIVO
altair>=5.3.0    ✔ ACTIVO  (viz nativa Marimo sin %magic)
```

---

## Notebooks Marimo activos

```
01_credit_scoring/notebooks/01_eda_scorecard.py
02_pd_lgd_estimation/notebooks/02_pd_lgd_panel.py
03_concentration_risk/notebooks/03_hhi_concentration.py
04_basel_capital_simulator/notebooks/04_rwa_simulator.py
05_eda_sbs_ecuador/notebooks/05_sbs_eda.py
```

---

## Comandos ZSH reproducibles

```zsh
# Instalar entorno limpio desde cero
uv sync

# Editar notebook en modo reactivo (UI en browser)
uv run marimo edit 01_credit_scoring/notebooks/01_eda_scorecard.py

# Ejecutar como app read-only
uv run marimo run 01_credit_scoring/notebooks/01_eda_scorecard.py

# Exportar reporte HTML (para LinkedIn / portafolio)
uv run marimo export html 01_credit_scoring/notebooks/01_eda_scorecard.py \
    -o 01_credit_scoring/reports/01_eda_scorecard.html

# Purgar cualquier .ipynb que haya entrado por error
find . -name '*.ipynb' -not -path './.git/*' -delete
find . -name '.ipynb_checkpoints' -type d -exec rm -rf {} + 2>/dev/null || true
```

---

*Migrado en `v0.3.0` — Mayo 2026*
