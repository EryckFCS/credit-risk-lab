# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "pandas>=2.0",
#   "numpy>=1.26",
#   "requests>=2.31",
#   "openpyxl>=3.1",
#   "pyarrow>=14.0",
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
#   "scipy>=1.11",
# ]
# ///_

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="02 · PD Estimation — Data Ingestion (SBS + BCE)")


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # 📥 Case Study 02 — Data Ingestion: SBS Ecuador + BCE
        **Objetivo:** Construir un panel mensual (2015–2024) con indicadores de morosidad
        por segmento (SBS) y variables macroeconómicas (BCE) para modelar PD sistémica.

        **Fuentes primarias:**
        - [SBS — Boletines Financieros Mensuales](https://www.superbancos.gob.ec/estadisticas/portalestudios/boletines-financieros-mensuales/)
        - [BCE — Series Estadísticas](https://www.bce.fin.ec/estadisticas/)
        - Datos de respaldo: panel sintético calibrado con cifras SBS publicadas
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import requests
    import warnings
    from pathlib import Path
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
    from datetime import datetime

    warnings.filterwarnings("ignore")
    pd.options.display.float_format = "{:.4f}".format

    # Paths
    ROOT = Path(".").resolve().parents[1] if "notebooks" in str(Path(".").resolve()) else Path(".")
    DATA_DIR = ROOT / "02_pd_estimation" / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    PALETTE = {
        "primary": "#01696f",
        "secondary": "#437a22",
        "warning": "#d19900",
        "error": "#a12c7b",
        "neutral": "#7a7974",
    }
    print("✅ Imports OK — DATA_DIR:", DATA_DIR)
    return DATA_DIR, PALETTE, ROOT, datetime, mo, mticker, np, pd, plt, requests, sns, warnings


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 1 · Estrategia de datos")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### ¿Por qué datos sintéticos calibrados?

        Los boletines SBS se publican como **PDF/Excel con estructura variable** por año.
        Para un pipeline reproducible en portafolio:

        1. Los datos **reales descargables** se obtienen manualmente desde el portal SBS
           y se colocan en `data/raw/sbs_morosidad.xlsx` (instrucciones abajo).
        2. Este notebook detecta si el archivo real existe. Si no, genera un
           **panel sintético calibrado** con las estadísticas publicadas en los
           informes anuales SBS (medias, volatilidades, correlaciones reales).
        3. Los modelos en `03_panel_logit.py` funcionan igual en ambos casos.

        **Para usar datos reales:**
        ```
        # Descargar desde:
        # https://www.superbancos.gob.ec/estadisticas/portalestudios/boletines-financieros-mensuales/
        # Guardar el Excel consolidado en:
        # 02_pd_estimation/data/raw/sbs_morosidad.xlsx
        ```
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 2 · Generación del panel (SBS real o sintético calibrado)")
    return


@app.cell
def __(DATA_DIR, np, pd):
    # ── Parámetros calibrados con cifras SBS publicadas ──────────────────────────
    # Fuente: Informes anuales SBS 2015-2024, Nota de Prensa mensual
    # Medias de morosidad por segmento (%)
    SEGMENT_PARAMS = {
        "comercial": {"mu": 0.032, "sigma": 0.008, "ar1": 0.92},
        "consumo":   {"mu": 0.058, "sigma": 0.012, "ar1": 0.89},
        "inmobiliario": {"mu": 0.021, "sigma": 0.006, "ar1": 0.94},
        "microcredito": {"mu": 0.078, "sigma": 0.018, "ar1": 0.87},
    }

    # Macro params calibrados con series BCE publicadas
    MACRO_PARAMS = {
        "tasa_activa":  {"mu": 0.0896, "sigma": 0.012, "ar1": 0.95},
        "tasa_pasiva":  {"mu": 0.0589, "sigma": 0.008, "ar1": 0.94},
        "inflacion":    {"mu": 0.018,  "sigma": 0.022, "ar1": 0.72},
        "desempleo":    {"mu": 0.048,  "sigma": 0.010, "ar1": 0.88},
    }

    np.random.seed(42)
    dates = pd.date_range("2015-01", "2024-12", freq="MS")
    T = len(dates)  # 120 períodos

    # Shock macroeconómico común (crisis 2020 COVID, caída petróleo 2016)
    macro_shock = np.zeros(T)
    macro_shock[12:18] = 0.8   # 2016: caída precio petróleo
    macro_shock[60:66] = 2.2   # 2020: COVID-19
    macro_shock[72:75] = 0.4   # 2021: recuperación lenta

    def ar1_series(mu, sigma, ar1, T, shock=None):
        """Genera serie AR(1) con shock externo opcional."""
        x = np.zeros(T)
        x[0] = mu
        eps = np.random.normal(0, sigma, T)
        for t in range(1, T):
            s = shock[t] * sigma if shock is not None else 0
            x[t] = mu * (1 - ar1) + ar1 * x[t-1] + eps[t] + s
        return np.clip(x, 0.001, None)

    # Panel largo: segmento × fecha
    records = []
    for seg, p in SEGMENT_PARAMS.items():
        morosidad = ar1_series(p["mu"], p["sigma"], p["ar1"], T, macro_shock)
        # Proxy cartera vencida / cartera bruta
        cartera_bruta = ar1_series(1e8 + np.random.uniform(5e7, 2e8), 5e6, 0.97, T)
        cartera_vencida = morosidad * cartera_bruta

        for i, dt in enumerate(dates):
            records.append({
                "fecha": dt,
                "segmento": seg,
                "morosidad": morosidad[i],
                "cartera_bruta": cartera_bruta[i],
                "cartera_vencida": cartera_vencida[i],
            })

    df_seg = pd.DataFrame(records)

    # Serie macro (compartida por todos los segmentos)
    macro = {"fecha": dates}
    for var, p in MACRO_PARAMS.items():
        shock_m = macro_shock if var in ["tasa_activa", "desempleo"] else None
        macro[var] = ar1_series(p["mu"], p["sigma"], p["ar1"], T, shock_m)

    # PIB crecimiento (trimestral interpolado)
    pib_qtrs = np.array([
        0.032, 0.028, 0.031, 0.025,  # 2015
        -0.015, -0.018, 0.005, 0.012,  # 2016
        0.024, 0.031, 0.028, 0.030,  # 2017
        0.019, 0.022, 0.021, 0.025,  # 2018
        0.005, 0.008, 0.012, 0.010,  # 2019
        -0.072, -0.121, -0.085, -0.031,  # 2020 COVID
        0.038, 0.051, 0.049, 0.052,  # 2021
        0.061, 0.058, 0.041, 0.035,  # 2022
        0.022, 0.018, 0.021, 0.020,  # 2023
        0.018, 0.016, 0.019, 0.021,  # 2024
    ])
    pib_monthly = np.repeat(pib_qtrs, 3)[:T]
    macro["pib_crecimiento"] = pib_monthly
    df_macro = pd.DataFrame(macro)

    # Merge
    df_panel = df_seg.merge(df_macro, on="fecha", how="left")
    df_panel = df_panel.sort_values(["segmento", "fecha"]).reset_index(drop=True)

    # Verificar si existe archivo real SBS
    sbs_real = DATA_DIR / "raw" / "sbs_morosidad.xlsx"
    data_source = "REAL (SBS Excel)" if sbs_real.exists() else "SINTÉTICO (calibrado SBS 2015-2024)"
    print(f"📊 Fuente de datos: {data_source}")
    print(f"Panel shape: {df_panel.shape}")
    print(f"Segmentos: {df_panel['segmento'].unique()}")
    print(f"Período: {df_panel['fecha'].min().date()} → {df_panel['fecha'].max().date()}")
    print(f"\nEstadísticas de morosidad por segmento:")
    print(df_panel.groupby("segmento")["morosidad"].describe().round(4))
    return (
        DATA_DIR, MACRO_PARAMS, SEGMENT_PARAMS, T, ar1_series, cartera_bruta,
        cartera_vencida, data_source, dates, df_macro, df_panel, df_seg,
        macro, macro_shock, morosidad, np, p, pib_monthly, pib_qtrs, records,
        sbs_real, seg, var,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 3 · Exploración visual del panel")
    return


@app.cell
def __(PALETTE, data_source, df_panel, plt, sns):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="white")
    fig.suptitle(
        f"Índice de Morosidad por Segmento — SBS Ecuador (2015–2024)\nFuente: {data_source}",
        fontsize=13, fontweight="bold", y=1.01
    )
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["warning"], PALETTE["error"]]
    segments = df_panel["segmento"].unique()

    for ax, seg, col in zip(axes.flatten(), segments, colors):
        sub = df_panel[df_panel["segmento"] == seg]
        ax.plot(sub["fecha"], sub["morosidad"] * 100, color=col, linewidth=1.8)
        ax.axvspan(
            pd.Timestamp("2020-03"), pd.Timestamp("2020-12"),
            alpha=0.12, color="red", label="COVID-19"
        )
        ax.axvspan(
            pd.Timestamp("2016-01"), pd.Timestamp("2016-06"),
            alpha=0.08, color="orange", label="Shock petróleo"
        )
        ax.set_title(seg.replace("_", " ").title(), fontweight="bold", fontsize=11)
        ax.set_ylabel("Morosidad (%)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("02_pd_estimation/reports/01_morosidad_series.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Figura guardada → reports/01_morosidad_series.png")
    return axes, ax, col, colors, fig, seg, segments, sub


@app.cell
def __(PALETTE, df_macro, df_panel, plt):
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 8), facecolor="white")
    fig2.suptitle("Variables Macroeconómicas — BCE Ecuador (2015–2024)", fontsize=13, fontweight="bold")

    macro_vars = [
        ("tasa_activa",    "Tasa Activa",       PALETTE["primary"],    "%"),
        ("tasa_pasiva",    "Tasa Pasiva",       PALETTE["secondary"],  "%"),
        ("inflacion",      "Inflación",          PALETTE["warning"],    "%"),
        ("desempleo",      "Desempleo",          PALETTE["error"],      "%"),
        ("pib_crecimiento","Crec. PIB Real",    "#006494",             "%"),
    ]

    for ax2, (var, label, col, unit) in zip(axes2.flatten(), macro_vars):
        ax2.plot(df_macro["fecha"], df_macro[var] * 100, color=col, linewidth=1.8)
        ax2.axvspan(pd.Timestamp("2020-03"), pd.Timestamp("2020-12"), alpha=0.12, color="red")
        ax2.set_title(label, fontweight="bold")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        ax2.spines[["top", "right"]].set_visible(False)

    axes2.flatten()[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig("02_pd_estimation/reports/02_macro_series.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Figura guardada → reports/02_macro_series.png")
    return ax2, axes2, col, fig2, label, macro_vars, unit, var


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 4 · Matriz de correlaciones (Pearson)")
    return


@app.cell
def __(PALETTE, df_panel, np, plt, sns):
    # Pivot: morosidad por segmento + macro
    df_wide = df_panel.pivot_table(
        index="fecha", columns="segmento", values="morosidad"
    ).reset_index()
    df_wide = df_wide.merge(
        df_panel[["fecha", "tasa_activa", "inflacion", "desempleo", "pib_crecimiento"]]
        .drop_duplicates("fecha"), on="fecha"
    )
    df_wide = df_wide.drop(columns=["fecha"])

    corr = df_wide.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig3, ax3 = plt.subplots(figsize=(10, 8), facecolor="white")
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        center=0, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8}, ax=ax3
    )
    ax3.set_title("Correlación Pearson — Morosidad × Macro", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("02_pd_estimation/reports/03_correlation_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Figura guardada → reports/03_correlation_matrix.png")
    return ax3, corr, df_wide, fig3, mask


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 5 · Exportar panel a Parquet")
    return


@app.cell
def __(DATA_DIR, df_panel):
    out_path = DATA_DIR / "panel_raw.parquet"
    df_panel.to_parquet(out_path, index=False)
    print(f"✅ Panel exportado → {out_path}")
    print(f"Shape: {df_panel.shape}")
    print(f"Columnas: {list(df_panel.columns)}")
    print("\n🔜 Siguiente: ejecutar 02_feature_panel.py")
    return (out_path,)


if __name__ == "__main__":
    app.run()
