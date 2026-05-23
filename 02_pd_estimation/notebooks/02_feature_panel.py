# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "pandas>=2.0",
#   "numpy>=1.26",
#   "pyarrow>=14.0",
#   "scipy>=1.11",
#   "statsmodels>=0.14",
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
# ]
# ///_

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="02 · Feature Engineering — Panel Econométrico")


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # 🔧 Feature Engineering — Panel Econométrico
        Construir las variables explicativas del modelo de PD:
        lags, primeras diferencias, ciclo HP, dummies de crisis y variables de interacción.
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import warnings
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.tsa.filters.hp_filter import hpfilter

    warnings.filterwarnings("ignore")
    ROOT = Path(".").resolve().parents[1] if "notebooks" in str(Path(".").resolve()) else Path(".")
    DATA_DIR = ROOT / "02_pd_estimation" / "data"
    REPORTS_DIR = ROOT / "02_pd_estimation" / "reports"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    PALETTE = {"primary": "#01696f", "secondary": "#437a22", "warning": "#d19900", "error": "#a12c7b"}
    print("✅ Imports OK")
    return DATA_DIR, PALETTE, REPORTS_DIR, ROOT, hpfilter, mo, np, pd, plt, sns, warnings


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 1 · Cargar panel raw")
    return


@app.cell
def __(DATA_DIR, pd):
    df = pd.read_parquet(DATA_DIR / "panel_raw.parquet")
    df = df.sort_values(["segmento", "fecha"]).reset_index(drop=True)
    print(f"Panel raw: {df.shape} | Períodos: {df['fecha'].nunique()} | Segmentos: {df['segmento'].nunique()}")
    return (df,)


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 2 · Lags y primeras diferencias")
    return


@app.cell
def __(df, pd):
    df2 = df.copy()
    macro_cols = ["tasa_activa", "tasa_pasiva", "inflacion", "desempleo", "pib_crecimiento"]

    for seg in df2["segmento"].unique():
        mask = df2["segmento"] == seg
        for col in macro_cols + ["morosidad"]:
            # Lags 1, 3, 6, 12
            for lag in [1, 3, 6, 12]:
                df2.loc[mask, f"{col}_lag{lag}"] = df2.loc[mask, col].shift(lag)
            # Primera diferencia
            df2.loc[mask, f"{col}_d1"] = df2.loc[mask, col].diff(1)

    print(f"Features tras lags+diff: {df2.shape[1]} columnas")
    return df2, lag, macro_cols, mask, seg


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 3 · Ciclo Hodrick-Prescott (componente cíclico del PIB)")
    return


@app.cell
def __(df2, hpfilter, np, pd):
    # HP filter sobre serie de PIB (lambda=14400 para datos mensuales — Ravn & Uhlig 2002)
    pib_series = df2[["fecha", "pib_crecimiento"]].drop_duplicates("fecha").set_index("fecha")["pib_crecimiento"]
    _, pib_cycle = hpfilter(pib_series, lamb=14400)

    hp_map = pib_cycle.to_dict()
    df2["pib_gap_hp"] = df2["fecha"].map(hp_map)
    print("✅ PIB gap HP (ciclo) calculado — lambda=14400 (Ravn & Uhlig 2002)")
    print(f"  Media ciclo: {pib_cycle.mean():.4f} | Std: {pib_cycle.std():.4f}")
    return hp_map, pib_cycle, pib_series


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 4 · Dummies de crisis y régimen")
    return


@app.cell
def __(df2, pd):
    df2["dummy_covid"]    = (df2["fecha"].between("2020-03", "2020-12")).astype(int)
    df2["dummy_oilshock"] = (df2["fecha"].between("2016-01", "2016-06")).astype(int)
    df2["dummy_recesion"] = ((df2["pib_crecimiento"] < 0)).astype(int)

    # Interacciones: crisis × tasa_activa
    df2["interact_covid_tasa"] = df2["dummy_covid"] * df2["tasa_activa"]
    df2["interact_covid_desempleo"] = df2["dummy_covid"] * df2["desempleo"]

    print("✅ Dummies creadas:")
    print(df2[["dummy_covid", "dummy_oilshock", "dummy_recesion"]].sum())
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 5 · Variable dependiente binaria (PD binaria)")
    return


@app.cell
def __(df2):
    # Umbral: percentil 75 de morosidad por segmento (within-segment stress)
    p75 = df2.groupby("segmento")["morosidad"].transform(lambda x: x.quantile(0.75))
    df2["default_bin"] = (df2["morosidad"] >= p75).astype(int)

    print("✅ Variable PD binaria (default_bin):")
    print(df2.groupby("segmento")["default_bin"].mean().rename("Tasa default binaria").round(3))
    return (p75,)


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 6 · Segmento dummies (Fixed Effects encoding)")
    return


@app.cell
def __(df2, pd):
    seg_dummies = pd.get_dummies(df2["segmento"], prefix="seg", drop_first=True)
    df2 = pd.concat([df2, seg_dummies], axis=1)
    df2 = df2.dropna().reset_index(drop=True)
    print(f"Panel final tras dropna: {df2.shape}")
    return df2, seg_dummies


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 7 · Visualización: distribución features clave")
    return


@app.cell
def __(PALETTE, df2, plt, sns):
    key_feats = [
        "morosidad", "tasa_activa", "desempleo",
        "pib_crecimiento", "pib_gap_hp", "tasa_activa_lag1"
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="white")
    fig.suptitle("Distribución de Features Clave — Panel SBS+BCE", fontsize=13, fontweight="bold")

    for ax, feat in zip(axes.flatten(), key_feats):
        sns.histplot(df2[feat], bins=30, kde=True, color=PALETTE["primary"], ax=ax)
        ax.set_title(feat.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("02_pd_estimation/reports/04_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Figura guardada")
    return ax, axes, feat, fig


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 8 · Exportar panel de features")
    return


@app.cell
def __(DATA_DIR, df2):
    out = DATA_DIR / "panel_features.parquet"
    df2.to_parquet(out, index=False)
    print(f"✅ panel_features.parquet exportado → {out}")
    print(f"Shape: {df2.shape}")
    print(f"Columnas totales: {len(df2.columns)}")
    print("\n🔜 Siguiente: ejecutar 03_panel_logit.py")
    return (out,)


if __name__ == "__main__":
    app.run()
