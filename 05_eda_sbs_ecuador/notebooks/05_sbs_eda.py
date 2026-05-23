# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.13.0",
#   "pandas>=2.2.0",
#   "numpy>=1.26.0",
#   "matplotlib>=3.8.0",
#   "seaborn>=0.13.0",
#   "plotly>=5.20.0",
#   "requests>=2.31.0",
# ]
# //
import marimo

__generated_with = "0.13.0"
app = marimo.App(
    width="full",
    app_title="EDA — SBS Ecuador Credit Portfolio",
)


@app.cell
def _():
    import marimo as mo
    mo.md("""
    # 🇪🇨 Case Study 05 — EDA: SBS Ecuador Credit Portfolio

    **Objective:** Exploratory analysis of the Ecuadorian banking system's credit portfolio
    using publicly available data from the Superintendencia de Bancos y Seguros (SBS).

    **Data sources:**
    - [SBS Boletín Financiero](https://www.superbancos.gob.ec/bancos/estadisticas/)
    - [BCE Operaciones Crediticias](https://www.bce.fin.ec/estadisticas/)

    **Scope:** Portfolio quality, sector distribution, NPL evolution, bank-level comparison.
    """
    )
    return (mo,)


@app.cell
def _(mo):
    import numpy as np
    import pandas as pd
    import sys
    from pathlib import Path
    import warnings
    warnings.filterwarnings("ignore")

    NOTEBOOK_DIR = Path(__file__).parent
    REPO_ROOT    = NOTEBOOK_DIR.parent.parent
    DATA_DIR     = NOTEBOOK_DIR.parent / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    mo.md(f"**Data dir:** `{DATA_DIR}`  \n*Place SBS Excel files here to activate real-data mode.*")
    return DATA_DIR, NOTEBOOK_DIR, REPO_ROOT, np, pd, sys, warnings


@app.cell
def _(DATA_DIR, mo, np, pd):
    # Auto-healing: detect real SBS files, fallback to synthetic
    sbs_files = list(DATA_DIR.glob("*.xlsx")) + list(DATA_DIR.glob("*.xls"))

    if sbs_files:
        try:
            df_sbs = pd.read_excel(sbs_files[0], sheet_name=0)
            data_source = f"Real SBS data: `{sbs_files[0].name}`"
        except Exception as e:
            df_sbs = None
            data_source = f"Failed to load `{sbs_files[0].name}`: {e}"
    else:
        # Synthetic SBS-style data
        np.random.seed(2024)
        quarters = pd.date_range("2019-03", periods=20, freq="QE")
        banks = ["Pichincha", "Guayaquil", "Pacífico", "Produbanco",
                 "Internacional", "Bolivariano", "Austro", "Loja"]

        records = []
        for bank in banks:
            base_portfolio = np.random.uniform(0.5e9, 8e9)
            for i, q in enumerate(quarters):
                trend = 1 + 0.015 * i + np.random.normal(0, 0.02)
                npl_base = np.random.uniform(0.02, 0.07)
                records.append({
                    "bank": bank,
                    "quarter": q,
                    "portfolio_usd": base_portfolio * trend,
                    "npl_ratio": np.clip(npl_base + np.random.normal(0, 0.008), 0.005, 0.15),
                    "coverage_ratio": np.clip(np.random.normal(1.8, 0.4), 0.8, 3.5),
                    "capital_ratio": np.clip(np.random.normal(0.14, 0.025), 0.08, 0.25),
                })
        df_sbs = pd.DataFrame(records)
        data_source = "Synthetic SBS-style data (place real .xlsx files in `data/`)"

    mo.callout(mo.md(f"**Data source:** {data_source}"), kind="info")
    return banks, base_portfolio, data_source, df_sbs, quarters, records, sbs_files


@app.cell
def _(df_sbs, mo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.05)

    # NPL evolution over time
    npl_pivot = df_sbs.pivot_table(
        index="quarter", columns="bank", values="npl_ratio", aggfunc="mean"
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="white")

    # 1. NPL ratio by bank over time
    palette = ["#01696f","#4f98a3","#a12c7b","#d163a7",
               "#bb653b","#fdab43","#6daa45","#5591c7"]
    for i, bank in enumerate(npl_pivot.columns):
        axes[0,0].plot(npl_pivot.index, npl_pivot[bank],
                       label=bank, color=palette[i % len(palette)], linewidth=1.8)
    axes[0,0].set_title("NPL Ratio Evolution by Bank", fontweight="bold")
    axes[0,0].set_ylabel("NPL Ratio")
    axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    axes[0,0].legend(fontsize=7)

    # 2. Portfolio size distribution
    latest = df_sbs[df_sbs["quarter"] == df_sbs["quarter"].max()]
    axes[0,1].bar(latest["bank"], latest["portfolio_usd"] / 1e9,
                  color=palette[:len(latest)], edgecolor="white")
    axes[0,1].set_title(f"Portfolio Size by Bank (Latest Quarter)", fontweight="bold")
    axes[0,1].set_ylabel("Portfolio (USD billions)")
    axes[0,1].tick_params(axis="x", rotation=30)

    # 3. Capital ratio vs NPL scatter
    axes[1,0].scatter(latest["npl_ratio"], latest["capital_ratio"],
                       c=palette[:len(latest)], s=120, zorder=3)
    for _, row in latest.iterrows():
        axes[1,0].annotate(row["bank"],
                            (row["npl_ratio"], row["capital_ratio"]),
                            textcoords="offset points", xytext=(5, 4), fontsize=8)
    axes[1,0].axvline(0.05, color="#a12c7b", linestyle="--", alpha=0.6, label="NPL=5% threshold")
    axes[1,0].set_title("Capital Ratio vs NPL (Latest)", fontweight="bold")
    axes[1,0].set_xlabel("NPL Ratio")
    axes[1,0].set_ylabel("Capital Ratio")
    axes[1,0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    axes[1,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    axes[1,0].legend()

    # 4. System NPL trend (average)
    sys_npl = df_sbs.groupby("quarter")["npl_ratio"].mean()
    axes[1,1].fill_between(sys_npl.index, sys_npl.values, alpha=0.25, color="#a12c7b")
    axes[1,1].plot(sys_npl.index, sys_npl.values, color="#a12c7b", linewidth=2.5)
    axes[1,1].axhline(0.05, color="#01696f", linestyle="--", label="5% regulatory reference")
    axes[1,1].set_title("System-Wide Average NPL Trend", fontweight="bold")
    axes[1,1].set_ylabel("NPL Ratio")
    axes[1,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
    axes[1,1].legend()

    plt.suptitle("SBS Ecuador — Credit Portfolio Dashboard", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    mo.mpl.interactive(fig)
    return (
        axes, fig, latest, matplotlib, npl_pivot,
        palette, plt, sns, sys_npl,
    )


@app.cell
def _(mo):
    mo.md("""
    ---
    ## 🗓️ Next Steps
    - [ ] Download real SBS Boletín Financiero (`.xlsx`) and place in `data/`
    - [ ] BCE operaciones crediticias: volume by credit type (consumo, vivienda, microempresa)
    - [ ] Correlation analysis: NPL vs macro indicators (PIB growth, inflation, unemployment)
    - [ ] Geographic breakdown by province (if microdatos available)
    """)
    return


if __name__ == "__main__":
    app.run()
