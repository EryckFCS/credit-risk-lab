# Migration: Jupyter → Marimo

## Por qué se migró

| Problema Jupyter | Solución Marimo |
|-----------------|----------------|
| Estado oculto: las celdas se ejecutan out-of-order, produciendo resultados no reproducibles | DAG reactivo: Marimo construye un grafo acíclico dirigido (DAG) de dependencias entre celdas. El orden de ejecución es determinístico |
| `.ipynb` = JSON con outputs embebidos: diffs ilegibles en Git, conflictos de merge permanentes | `.py` puro: diff legible, `git blame` funciona, PR reviews son posibles |
| `%matplotlib inline`, `%load_ext`, comandos mágicos: no son Python estándar | Marimo usa Python puro: ninguna dependencia oculta del kernel |
| Checkpoints (`.ipynb_checkpoints/`): contaminan el árbol de Git | Eliminados permanentemente vía `.gitignore` |
| JupyterLab = dependencia pesada en el entorno de producción | Marimo es liviano y puede correr como app WASM o como script |

## Estructura post-migración

```
01_credit_scoring/notebooks/
├── 01_eda.py                    ← marimo (era 01_eda.ipynb)
├── 02_feature_engineering.py    ← marimo (era 02_feature_engineering.ipynb)
└── 03_model_and_validation.py   ← marimo (era 03_model_and_validation.ipynb)
```

Los archivos `.ipynb` han sido eliminados del repositorio y bloqueados en `.gitignore`.

## Cómo ejecutar los notebooks

```bash
# Instalar marimo (ya incluido en pyproject.toml)
uv sync

# Lanzar en modo interactivo (browser, reactive DAG)
uv run marimo edit 01_credit_scoring/notebooks/01_eda.py

# Ejecutar como script (sin browser, modo producción)
uv run python 01_credit_scoring/notebooks/01_eda.py

# Secuencia completa case study 01:
uv run marimo edit 01_credit_scoring/notebooks/01_eda.py
uv run marimo edit 01_credit_scoring/notebooks/02_feature_engineering.py
uv run marimo edit 01_credit_scoring/notebooks/03_model_and_validation.py
```

## Comandos de purga (ya ejecutados)

```bash
# Purgar notebooks Jupyter del repo
find . -name '*.ipynb' -not -path './.git/*' -delete
find . -name '.ipynb_checkpoints' -type d -exec rm -rf {} +

# Verificar que no quedan rastros
git status --short | grep ipynb  # debe retornar vacío
```

## Principios DAG en Marimo

Cada celda en Marimo declara explícitamente sus dependencias a través de los parámetros de la función decorada con `@app.cell`. Esto significa:

1. **No hay estado global oculto**: si una celda necesita `df`, lo recibe como argumento.
2. **Reactividad automática**: modificar `DATA_PATH` en la celda 02 invalida automáticamente todas las celdas downstream.
3. **Soberanía de datos**: los datos entran a las celdas como argumentos inmutables, la lógica es versionable.
4. **Rutas relativas**: `Path(__file__).resolve().parent` ancla siempre al repositorio, nunca al SO.
