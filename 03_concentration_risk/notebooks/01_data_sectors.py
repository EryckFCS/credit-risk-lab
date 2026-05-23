# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "pandas>=2.0",
#   "numpy>=1.26",
#   "pyarrow>=14.0",
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
# ]
# ///_

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="03 · Concentration Risk — Data Ingestion")


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # 📥 Case Study 03 — Data Ingestion: Cartera Sectorial SBS Ecuador
        **Objetivo:** Construir el panel de distribución de cartera por sector económico (CIIU)
        para calcular métricas de concentración (HHI, CR5) y aplicar stress testing Basel.

        **Fuente:** SBS Ecuador — Volumen de Crédito por Sector Económico (2018–2024).
        Datos sintéticos calibrados con la composición sectorial publicada en boletines SBS.
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

    warnings.filterwarnings("ignore")
    ROOT = Path(".").resolve().parents[1] if "notebooks" in str(Path(".").resolve()) else Path(".")
    DATA_DIR = ROOT / "03_concentration_risk" / "data"
    REPORTS_DIR = ROOT / "03_concentration_risk" / "reports"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    PALETTE = {
        "primary": "#01696f", "secondary": "#437a22",
        "warning": "#d19900", "error": "#a12c7b",
        "blue": "#006494", "purple": "#7a39bb",
        "orange": "#da7101", "neutral": "#7a7974",
    }
    print("✅ Imports OK")
    return DATA_DIR, PALETTE, REPORTS_DIR, ROOT, mo, np, pd, plt, sns, warnings


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 1 · Composición sectorial calibrada (proxy SBS 2018–2024)")
    return


@app.cell
def __(np, pd):
    # Participaciones medias calibradas con boletines SBS (cartera bruta por destino)
    # Fuente: Notas de Prensa SBS, Informes Anuales 2018-2024
    SECTORS = {
        "comercio":        {"share_mu": 0.22, "share_sigma": 0.025, "morosidad_mu": 0.042, "morosidad_sigma": 0.010},
        "agricultura":     {"share_mu": 0.14, "share_sigma": 0.018, "morosidad_mu": 0.051, "morosidad_sigma": 0.013},
        "manufactura":     {"share_mu": 0.11, "share_sigma": 0.015, "morosidad_mu": 0.035, "morosidad_sigma": 0.009},
        "construccion":    {"share_mu": 0.09, "share_sigma": 0.020, "morosidad_mu": 0.048, "morosidad_sigma": 0.015},
        "transporte":      {"share_mu": 0.07, "share_sigma": 0.012, "morosidad_mu": 0.063, "morosidad_sigma": 0.018},
        "financiero":      {"share_mu": 0.06, "share_sigma": 0.010, "morosidad_mu": 0.019, "morosidad_sigma": 0.006},
        "servicios":       {"share_mu": 0.12, "share_sigma": 0.014, "morosidad_mu": 0.044, "morosidad_sigma": 0.011},
        "consumo_pn":      {"share_mu": 0.19, "share_sigma": 0.022, "morosidad_mu": 0.058, "morosidad_sigma": 0.012},
    }

    # 10 bancos privados ficticios (representativos del sistema SBS)
    BANKS = [f"Banco_{chr(65+i)}" for i in range(10)]
    # Tamaños relativos de mercado (Banco_A = grande, Banco_J = pequeño)
    BANK_SIZES = np.array([0.22, 0.18, 0.14, 0.11, 0.09, 0.08, 0.07, 0.05, 0.04, 0.02])
    BANK_SIZES = BANK_SIZES / BANK_SIZES.sum()

    np.random.seed(2024)
    dates = pd.date_range("2018-01", "2024-12", freq="MS")
    T = len(dates)

    # Shock COVID
    covid_shock = np.zeros(T)
    covid_shock[26:32] = 1.0   # Mar–Sep 2020

    records = []
    for bank_idx, bank in enumerate(BANKS):
        bank_size = BANK_SIZES[bank_idx]
        # Cada banco tiene concentración sectorial diferente (más o menos diversificado)
        concentration_bias = np.random.dirichlet(np.ones(len(SECTORS)) * (3 + bank_idx * 0.5))

        for t_idx, dt in enumerate(dates):
            # Shares sectoriales con ruido
            raw_shares = np.array([
                max(0.01, np.random.normal(p["share_mu"], p["share_sigma"])) * (1 + concentration_bias[i] * 0.3)
                for i, p in enumerate(SECTORS.values())
            ])
            shares = raw_shares / raw_shares.sum()  # normalizar a 1

            for sec_idx, (sector, params) in enumerate(SECTORS.items()):
                mor = max(0.001, np.random.normal(
                    params["morosidad_mu"] + covid_shock[t_idx] * 0.025,
                    params["morosidad_sigma"]
                ))
                cartera = bank_size * 1e9 * shares[sec_idx] * (1 + np.random.normal(0, 0.02))
                records.append({
                    "fecha": dt,
                    "banco": bank,
                    "sector": sector,
                    "cartera": max(0, cartera),
                    "morosidad": mor,
                    "share_sector": shares[sec_idx],
                })

    df = pd.DataFrame(records)
    print(f"Panel shape: {df.shape}")
    print(f"Bancos: {df['banco'].nunique()} | Sectores: {df['sector'].nunique()} | Períodos: {df['fecha'].nunique()}")
    print(f"\nCartera total promedio por banco (USD millones):")
    print((df.groupby("banco")["cartera"].sum() / T / 1e6).round(1).to_string())
    return (
        BANK_SIZES, BANKS, SECTORS, T, bank, bank_idx, bank_size,
        cartera, concentration_bias, covid_shock, dates, df, mor,
        raw_shares, records, sec_idx, sector, shares, t_idx,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 2 · Composición sectorial media del sistema")
    return


@app.cell
def __(PALETTE, df, plt):
    sector_share = (
        df.groupby("sector")["cartera"].sum()
        / df["cartera"].sum()
    ).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    colors = list(PALETTE.values())[:len(sector_share)]

    # Pie chart
    wedges, texts, autotexts = axes[0].pie(
        sector_share.values, labels=sector_share.index,
        autopct="%1.1f%%", colors=colors,
        startangle=90, pctdistance=0.82,
        wedgeprops={"linewidth": 1.5, "edgecolor": "white"}
    )
    for at in autotexts:
        at.set_fontsize(9)
    axes[0].set_title("Composición Sectorial Media\nCartera SBS Ecuador 2018–2024",
                      fontweight="bold", fontsize=11)

    # Bar chart morosidad
    sector_mor = df.groupby("sector")["morosidad"].mean().sort_values(ascending=True)
    axes[1].barh(sector_mor.index, sector_mor.values * 100,
                 color=[PALETTE["primary"] if v < 0.05 else PALETTE["error"]
                        for v in sector_mor.values], alpha=0.85)
    axes[1].axvline(5.0, color=PALETTE["warning"], lw=1.5, linestyle="--", label="5% benchmark")
    axes[1].set_xlabel("Morosidad media (%)")
    axes[1].set_title("Morosidad Media por Sector", fontweight="bold", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    plt.tight_layout()
    plt.savefig("03_concentration_risk/reports/01_sector_composition.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Figura guardada → reports/01_sector_composition.png")
    return axes, autotexts, colors, fig, sector_mor, sector_share, texts, wedges, at


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 3 · Evolución temporal cartera por sector")
    return


@app.cell
def __(PALETTE, df, pd, plt):
    ts = df.groupby(["fecha", "sector"])["cartera"].sum().reset_index()
    ts_pivot = ts.pivot(index="fecha", columns="sector", values="cartera") / 1e9  # bn USD

    fig2, ax = plt.subplots(figsize=(14, 6), facecolor="white")
    colors_list = list(PALETTE.values())
    ts_pivot.plot(ax=ax, colormap="tab10", linewidth=1.6, alpha=0.85)
    ax.axvspan(pd.Timestamp("2020-03"), pd.Timestamp("2020-12"),
               alpha=0.12, color="red", label="COVID-19")
    ax.set_title("Cartera Bruta por Sector — Sistema Financiero SBS (2018–2024)",
                 fontweight="bold", fontsize=12)
    ax.set_ylabel("Cartera (USD Billones)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.1f}B"))
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("03_concentration_risk/reports/02_cartera_evolution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Figura guardada → reports/02_cartera_evolution.png")
    return ax, fig2, ts, ts_pivot


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 4 · Exportar panel")
    return


@app.cell
def __(DATA_DIR, df):
    out = DATA_DIR / "sector_panel.parquet"
    df.to_parquet(out, index=False)
    print(f"✅ sector_panel.parquet → {out}")
    print(f"Shape: {df.shape}")
    print("\n🔜 Siguiente: 02_hhi_metrics.py")
    return (out,)


if __name__ == "__main__":
    app.run()
