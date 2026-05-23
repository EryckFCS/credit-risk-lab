# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "pandas>=2.0",
#   "numpy>=1.26",
#   "pyarrow>=14.0",
#   "matplotlib>=3.8",
#   "seaborn>=0.13",
#   "scipy>=1.11",
# ]
# ///_

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="03 · HHI & Concentration Metrics")


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # 📊 Métricas de Concentración — HHI, CR5, Berry Index
        Cálculo de índices de concentración crediticia a nivel sistema y banco.
        Umbrales regulatorios según BCBS (2006) y guías de supervisión SBS Ecuador.
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
    import matplotlib.ticker as mticker
    import seaborn as sns
    from scipy.stats import pearsonr

    warnings.filterwarnings("ignore")
    ROOT = Path(".").resolve().parents[1] if "notebooks" in str(Path(".").resolve()) else Path(".")
    DATA_DIR  = ROOT / "03_concentration_risk" / "data"
    REPORTS_DIR = ROOT / "03_concentration_risk" / "reports"

    PALETTE = {
        "primary": "#01696f", "secondary": "#437a22",
        "warning": "#d19900", "error": "#a12c7b",
        "blue": "#006494", "neutral": "#7a7974",
    }
    THRESHOLDS = {"hhi_low": 1000, "hhi_high": 1800, "cr5_high": 0.70}
    print("✅ Imports OK")
    return DATA_DIR, PALETTE, REPORTS_DIR, ROOT, THRESHOLDS, mo, mticker, np, pd, pearsonr, plt, sns, warnings


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 1 · Cargar panel sectorial")
    return


@app.cell
def __(DATA_DIR, pd):
    df = pd.read_parquet(DATA_DIR / "sector_panel.parquet")
    print(f"Panel: {df.shape}")
    return (df,)


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 2 · HHI sectorial por banco × mes")
    return


@app.cell
def __(df):
    def compute_hhi(series):
        """HHI = sum(si^2) donde si = participación sector i. Rango [0, 10000]."""
        total = series.sum()
        if total == 0:
            return np.nan
        shares = series / total
        return (shares ** 2).sum() * 10000

    def compute_cr5(series):
        """CR5 = suma top-5 participaciones."""
        total = series.sum()
        if total == 0:
            return np.nan
        shares = (series / total).sort_values(ascending=False)
        return shares.iloc[:5].sum()

    def compute_berry(series):
        """Berry Index = 1 - HHI/10000 (diversificación 0→1)."""
        return 1 - compute_hhi(series) / 10000

    # HHI por banco × mes
    hhi_banco = (
        df.groupby(["fecha", "banco"])["cartera"]
        .apply(compute_hhi)
        .reset_index(name="hhi")
    )
    hhi_banco["cr5"]   = df.groupby(["fecha", "banco"])["cartera"].apply(compute_cr5).values
    hhi_banco["berry"] = 1 - hhi_banco["hhi"] / 10000

    # HHI sistema (agregado)
    hhi_sistema = (
        df.groupby("fecha")["cartera"]
        .apply(compute_hhi)
        .reset_index(name="hhi_sistema")
    )
    hhi_sistema["cr5_sistema"] = df.groupby("fecha")["cartera"].apply(compute_cr5).values
    hhi_sistema["berry_sistema"] = 1 - hhi_sistema["hhi_sistema"] / 10000

    print("=== HHI Sistema (últimos 6 meses) ===")
    print(hhi_sistema.tail(6).round(2).to_string(index=False))
    print(f"\nHHI sistema media: {hhi_sistema['hhi_sistema'].mean():.1f}")
    print(f"HHI sistema actual: {hhi_sistema['hhi_sistema'].iloc[-1]:.1f}")
    return compute_berry, compute_cr5, compute_hhi, hhi_banco, hhi_sistema


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 3 · Clasificación regulatoria por banco")
    return


@app.cell
def __(THRESHOLDS, hhi_banco, pd):
    def hhi_category(hhi):
        if hhi < THRESHOLDS["hhi_low"]:  return "🟢 Diversificado"
        elif hhi < THRESHOLDS["hhi_high"]: return "🟡 Concentración moderada"
        else:                               return "🔴 Concentración alta"

    latest = hhi_banco[hhi_banco["fecha"] == hhi_banco["fecha"].max()].copy()
    latest["categoria"] = latest["hhi"].apply(hhi_category)
    latest = latest.sort_values("hhi", ascending=False)

    print("=== CLASIFICACIÓN REGULATORIA HHI (dic-2024) ===")
    print(latest[["banco", "hhi", "cr5", "berry", "categoria"]].to_string(index=False))
    return hhi_category, latest


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 4 · Visualización — Dashboard de concentración")
    return


@app.cell
def __(PALETTE, THRESHOLDS, hhi_banco, hhi_sistema, latest, pd, plt, sns):
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(16, 12), facecolor="white")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Dashboard de Concentración Crediticia — SBS Ecuador",
                 fontsize=14, fontweight="bold")

    # ── A) HHI sistema en el tiempo
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(hhi_sistema["fecha"], hhi_sistema["hhi_sistema"],
             color=PALETTE["primary"], lw=2, label="HHI Sistema")
    ax1.axhline(THRESHOLDS["hhi_low"],  color=PALETTE["secondary"], lw=1.5,
                ls="--", label=f"Diversificado < {THRESHOLDS['hhi_low']}")
    ax1.axhline(THRESHOLDS["hhi_high"], color=PALETTE["error"], lw=1.5,
                ls="--", label=f"Alta concentración ≥ {THRESHOLDS['hhi_high']}")
    ax1.axvspan(pd.Timestamp("2020-03"), pd.Timestamp("2020-12"),
                alpha=0.12, color="red", label="COVID-19")
    ax1.fill_between(hhi_sistema["fecha"], THRESHOLDS["hhi_low"], THRESHOLDS["hhi_high"],
                     alpha=0.06, color=PALETTE["warning"], label="Zona moderada")
    ax1.set_ylabel("HHI")
    ax1.set_title("Índice HHI Sistema Financiero 2018–2024", fontweight="bold")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # ── B) HHI por banco (barras horizontales, último período)
    ax2 = fig.add_subplot(gs[1, 0])
    colors_bar = [
        PALETTE["error"] if h >= THRESHOLDS["hhi_high"]
        else PALETTE["warning"] if h >= THRESHOLDS["hhi_low"]
        else PALETTE["secondary"]
        for h in latest["hhi"]
    ]
    ax2.barh(latest["banco"], latest["hhi"], color=colors_bar, alpha=0.85)
    ax2.axvline(THRESHOLDS["hhi_low"],  color=PALETTE["secondary"], lw=1.5, ls="--")
    ax2.axvline(THRESHOLDS["hhi_high"], color=PALETTE["error"],     lw=1.5, ls="--")
    ax2.set_title("HHI por Banco (dic-2024)", fontweight="bold")
    ax2.set_xlabel("HHI")
    ax2.spines[["top", "right"]].set_visible(False)

    # ── C) CR5 en el tiempo por banco
    ax3 = fig.add_subplot(gs[1, 1])
    for banco in hhi_banco["banco"].unique():
        sub = hhi_banco[hhi_banco["banco"] == banco]
        ax3.plot(sub["fecha"], sub["cr5"], alpha=0.6, lw=1.2)
    ax3.plot(hhi_sistema["fecha"], hhi_sistema["cr5_sistema"],
             color=PALETTE["primary"], lw=2.5, label="CR5 Sistema", zorder=5)
    ax3.axhline(THRESHOLDS["cr5_high"], color=PALETTE["error"], lw=1.5,
                ls="--", label=f"Umbral CR5={THRESHOLDS['cr5_high']:.0%}")
    ax3.set_title("CR5 por Banco vs Sistema", fontweight="bold")
    ax3.set_ylabel("CR5")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax3.legend(fontsize=8)
    ax3.spines[["top", "right"]].set_visible(False)

    # ── D) Berry Index (diversificación)
    ax4 = fig.add_subplot(gs[1, 2])
    berry_latest = latest.sort_values("berry", ascending=True)
    ax4.barh(berry_latest["banco"], berry_latest["berry"],
             color=[PALETTE["primary"] if b > 0.85 else PALETTE["warning"] for b in berry_latest["berry"]],
             alpha=0.85)
    ax4.set_title("Berry Index (Diversificación)", fontweight="bold")
    ax4.set_xlabel("Berry Index (0=monopolio, 1=diversificado)")
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax4.spines[["top", "right"]].set_visible(False)

    plt.savefig("03_concentration_risk/reports/03_hhi_dashboard.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Dashboard guardado → reports/03_hhi_dashboard.png")
    return (
        ax1, ax2, ax3, ax4, banco, berry_latest, colors_bar,
        fig, gridspec, gs, sub,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 5 · Correlación HHI ↔ Morosidad")
    return


@app.cell
def __(PALETTE, df, hhi_banco, pd, pearsonr, plt):
    # Morosidad sistema por banco × mes
    mor_banco = (
        df.groupby(["fecha", "banco"])
        .apply(lambda g: np.average(g["morosidad"], weights=g["cartera"]))
        .reset_index(name="morosidad_w")
    )
    import numpy as np
    merged = hhi_banco.merge(mor_banco, on=["fecha", "banco"])
    r, p = pearsonr(merged["hhi"], merged["morosidad_w"])

    fig3, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.scatter(merged["hhi"], merged["morosidad_w"] * 100,
               alpha=0.25, color=PALETTE["primary"], s=15)
    # Línea de regresión
    z = np.polyfit(merged["hhi"], merged["morosidad_w"] * 100, 1)
    p_fn = np.poly1d(z)
    x_line = np.linspace(merged["hhi"].min(), merged["hhi"].max(), 100)
    ax.plot(x_line, p_fn(x_line), color=PALETTE["error"], lw=2, label=f"r={r:.3f}, p={p:.4f}")
    ax.set_xlabel("HHI")
    ax.set_ylabel("Morosidad ponderada (%)")
    ax.set_title("Correlación: Concentración (HHI) ↔ Morosidad", fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("03_concentration_risk/reports/04_hhi_vs_morosidad.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Correlación Pearson r={r:.4f}, p-value={p:.4f}")
    return ax, fig3, merged, mor_banco, np, p, p_fn, r, x_line, z


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 6 · Exportar métricas")
    return


@app.cell
def __(DATA_DIR, hhi_banco, hhi_sistema):
    hhi_banco.to_parquet(DATA_DIR / "concentration_metrics.parquet", index=False)
    hhi_sistema.to_parquet(DATA_DIR / "hhi_sistema.parquet", index=False)
    print("✅ concentration_metrics.parquet exportado")
    print("✅ hhi_sistema.parquet exportado")
    print("\n🔜 Siguiente: 03_stress_testing.py")
    return


if __name__ == "__main__":
    app.run()
