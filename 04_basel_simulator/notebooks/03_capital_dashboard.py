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
app = marimo.App(width="medium", app_title="04 · Basel III — Capital Dashboard")

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    # 📊 Basel III — Capital & Liquidity Dashboard
    Dashboard supervisorio integrado: capital, liquidez y stress test bajo Basel III.
    Replica el tipo de reporte que usa un oficial de riesgo en un banco regulado.
    """)
    return

@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import warnings
    from pathlib import Path
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    from datetime import datetime

    warnings.filterwarnings("ignore")
    ROOT = Path(".").resolve().parents[1] if "notebooks" in str(Path(".").resolve()) else Path(".")
    DATA_DIR    = ROOT / "04_basel_simulator" / "data"
    REPORTS_DIR = ROOT / "04_basel_simulator" / "reports"

    PALETTE = {"primary": "#01696f", "secondary": "#437a22",
               "warning": "#d19900", "error": "#a12c7b", "blue": "#006494"}
    print("✅ Imports OK")
    return DATA_DIR, PALETTE, REPORTS_DIR, ROOT, datetime, gridspec, mo, np, pd, plt, sns, warnings

@app.cell(hide_code=True)
def __(mo):
    mo.md("## 1 · Cargar datos")
    return

@app.cell
def __(DATA_DIR, pd):
    cap = pd.read_parquet(DATA_DIR / "capital_base.parquet").iloc[0]
    liq = pd.read_parquet(DATA_DIR / "liquidity_metrics.parquet").iloc[0]
    print("Capital:", cap.to_dict())
    print("Liquidez:", liq.to_dict())
    return cap, liq

@app.cell(hide_code=True)
def __(mo):
    mo.md("## 2 · Stress Test — Impacto en Capital")
    return

@app.cell
def __(cap, np, pd):
    SCENARIOS_STRESS = {
        "Base":             {"rwa_delta": 0.00,  "capital_loss": 0.000, "color": "#437a22"},
        "Adverse":          {"rwa_delta": 0.12,  "capital_loss": 0.050, "color": "#d19900"},
        "Severely Adverse": {"rwa_delta": 0.28,  "capital_loss": 0.140, "color": "#a12c7b"},
    }

    stress_results = []
    for scen, params in SCENARIOS_STRESS.items():
        rwa_stressed     = cap["total_rwa"] * (1 + params["rwa_delta"])
        capital_stressed = cap["total_capital"] * (1 - params["capital_loss"])
        tier1_stressed   = cap["tier1"] * (1 - params["capital_loss"])
        cet1_stressed    = cap["cet1"] * (1 - params["capital_loss"])

        car_stressed   = capital_stressed / rwa_stressed
        tier1_stressed_ratio = tier1_stressed / rwa_stressed
        cet1_stressed_ratio  = cet1_stressed  / rwa_stressed

        stress_results.append({
            "escenario": scen,
            "rwa_stressed": rwa_stressed,
            "capital_stressed": capital_stressed,
            "car": car_stressed,
            "tier1_ratio": tier1_stressed_ratio,
            "cet1_ratio": cet1_stressed_ratio,
            "breach_car":   car_stressed < 0.105,
            "breach_tier1": tier1_stressed_ratio < 0.060,
            "breach_cet1":  cet1_stressed_ratio < 0.045,
            "color": params["color"],
        })

    df_stress = pd.DataFrame(stress_results)
    print("=== STRESS TEST — CAPITAL RATIOS ===")
    print(df_stress[["escenario", "car", "tier1_ratio", "cet1_ratio",
                     "breach_car", "breach_tier1", "breach_cet1"]]
          .round(4).to_string(index=False))
    return SCENARIOS_STRESS, df_stress, stress_results

@app.cell(hide_code=True)
def __(mo):
    mo.md("## 3 · Dashboard Supervisorio Integrado")
    return

@app.cell
def __(PALETTE, cap, datetime, df_stress, gridspec, liq, plt):
    fig = plt.figure(figsize=(18, 14), facecolor="white")
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.40)
    fig.suptitle(
        f"Basel III — Capital & Liquidity Dashboard\nReporte Supervisorio · {datetime.now().strftime('%Y-%m-%d')}",
        fontsize=14, fontweight="bold"
    )

    def ratio_bar(ax, value, minimum, label, color):
        """Mini gauge de ratio vs mínimo regulatorio."""
        bar_col = color if value >= minimum else PALETTE["error"]
        ax.barh([label], [value * 100], color=bar_col, alpha=0.85, height=0.5)
        ax.axvline(minimum * 100, color="black", lw=2, ls="--", label=f"Min {minimum:.1%}")
        ax.set_xlim(0, max(value * 100 * 1.3, minimum * 100 * 1.5))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax.legend(fontsize=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_yticks([])
        status = "✅ CUMPLE" if value >= minimum else "🔴 BREACH"
        ax.set_title(f"{label}: {value:.2%}  {status}", fontweight="bold", fontsize=10,
                     color=color if value >= minimum else PALETTE["error"])

    # ── Fila 0: KPIs de capital
    ratio_bar(fig.add_subplot(gs[0,0]), cap["cet1_ratio"], 0.070, "CET1 Ratio", PALETTE["primary"])
    ratio_bar(fig.add_subplot(gs[0,1]), cap["tier1_ratio"], 0.085, "Tier 1 Ratio", PALETTE["secondary"])
    ratio_bar(fig.add_subplot(gs[0,2]), cap["car"], 0.105, "CAR Total", PALETTE["blue"])

    # ── RWA Density
    ax_rwa = fig.add_subplot(gs[0,3])
    density = cap["total_rwa"] / cap["total_assets"]
    ax_rwa.pie(
        [density, 1 - density],
        labels=[f"RWA\n{density:.1%}", f"No RWA\n{1-density:.1%}"],
        colors=[PALETTE["warning"], "#e6e4df"],
        startangle=90,
        wedgeprops={"linewidth": 2, "edgecolor": "white"}
    )
    ax_rwa.set_title("RWA Density", fontweight="bold")

    # ── Fila 1: Liquidez
    ratio_bar(fig.add_subplot(gs[1,0]), liq["lcr"], 1.00, "LCR", PALETTE["primary"])
    ratio_bar(fig.add_subplot(gs[1,1]), liq["nsfr"], 1.00, "NSFR", PALETTE["secondary"])

    # HQLA Breakdown (mini)
    ax_hq = fig.add_subplot(gs[1,2:])
    hqla_labels = ["L1 Cash", "L1 Sovereign", "L2A Sov", "L2A Covered", "L2B RMBS", "L2B Equity"]
    hqla_vals   = [180, 320, 119, 68, 41.25, 15]  # post-haircut
    colors_hq   = [PALETTE["primary"], PALETTE["secondary"], PALETTE["blue"],
                   PALETTE["blue"], PALETTE["warning"], PALETTE["warning"]]
    ax_hq.bar(hqla_labels, hqla_vals, color=colors_hq, alpha=0.85)
    ax_hq.set_title("HQLA por Categoría (USD M, post-haircut)", fontweight="bold")
    ax_hq.set_ylabel("USD M")
    ax_hq.spines[["top", "right"]].set_visible(False)
    ax_hq.tick_params(axis="x", rotation=30, labelsize=9)

    # ── Fila 2: Stress Test — CAR bajo escenarios
    ax_st = fig.add_subplot(gs[2,:])
    scen_names = df_stress["escenario"].tolist()
    ratios_to_plot = {
        "CAR Total":   (df_stress["car"],         0.105, "o-"),
        "Tier 1":      (df_stress["tier1_ratio"],  0.085, "s--"),
        "CET1":        (df_stress["cet1_ratio"],   0.070, "^:"),
    }
    colors_st = [PALETTE["primary"], PALETTE["secondary"], PALETTE["blue"]]
    for (name, (vals, thresh, ls)), col in zip(ratios_to_plot.items(), colors_st):
        ax_st.plot(scen_names, vals * 100, ls, color=col, lw=2.5,
                   markersize=10, label=name)
        ax_st.axhline(thresh * 100, color=col, lw=1, ls=":", alpha=0.5)
    ax_st.set_title("Stress Test — Capital Ratios bajo Escenarios Basel", fontweight="bold")
    ax_st.set_ylabel("%")
    ax_st.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax_st.legend(fontsize=10)
    ax_st.fill_between(scen_names, 0, 10.5, alpha=0.05, color=PALETTE["error"],
                       label="Zona breach CAR")
    ax_st.grid(axis="y", linestyle="--", alpha=0.3)
    ax_st.spines[["top", "right"]].set_visible(False)

    plt.savefig("04_basel_simulator/reports/03_capital_dashboard.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Dashboard guardado → reports/03_capital_dashboard.png")
    return ax_hq, ax_rwa, ax_st, col, colors_hq, colors_st, density, fig, gs, hqla_labels, hqla_vals, ls, name, ratios_to_plot, scen_names, thresh, vals

@app.cell(hide_code=True)
def __(mo):
    mo.md("## 4 · Model Card")
    return

@app.cell
def __(REPORTS_DIR, cap, datetime, df_stress, liq):
    card = f"""# Basel III Capital & Liquidity Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Capital Adequacy (Baseline)

| Ratio | Valor | Mínimo | Status |
|-------|-------|--------|--------|
| CET1 Ratio | {cap['cet1_ratio']:.2%} | 7.0% | {'✅' if cap['cet1_ratio']>=0.07 else '🔴'} |
| Tier 1 Ratio | {cap['tier1_ratio']:.2%} | 8.5% | {'✅' if cap['tier1_ratio']>=0.085 else '🔴'} |
| CAR Total | {cap['car']:.2%} | 10.5% | {'✅' if cap['car']>=0.105 else '🔴'} |
| RWA Density | {cap['rwa_density']:.1%} | — | — |

## Liquidity (Baseline)

| Ratio | Valor | Mínimo | Status |
|-------|-------|--------|--------|
| LCR | {liq['lcr']:.1%} | 100% | {'✅' if liq['lcr']>=1 else '🔴'} |
| NSFR | {liq['nsfr']:.1%} | 100% | {'✅' if liq['nsfr']>=1 else '🔴'} |

## Stress Test — CAR bajo Escenarios

| Escenario | CAR | Tier1 | CET1 | Breach CAR |
|-----------|-----|-------|------|------------|
"""
    for _, r in df_stress.iterrows():
        card += f"| {r['escenario']} | {r['car']:.2%} | {r['tier1_ratio']:.2%} | {r['cet1_ratio']:.2%} | {'🔴 SÍ' if r['breach_car'] else '✅ NO'} |\n"

    card += """
## Supuestos
- Portafolio proxy de banco mediano Ecuador (activos ~USD 2,940M)
- Standardised Approach para RWA (sin modelos internos IRB)
- LGD y correlaciones uniformes por clase de activo
- Escenarios de estrés calibrados con BCBS DFAST framework

## Referencias
- BCBS (2017). Basel III: Finalising post-crisis reforms. BIS.
- BCBS (2013). Basel III: LCR and liquidity risk monitoring tools. BIS.
- BCBS (2014). Basel III: Net Stable Funding Ratio. BIS.
- SBS Ecuador. Resolución No. SB-2021-0565 (Patrimonio Técnico).
"""
    card_path = REPORTS_DIR / "basel3_report.md"
    card_path.write_text(card)
    print(f"✅ Report exportado → {card_path}")
    print("\n🏁 Case Study 04 completado.")
    return card, card_path

if __name__ == "__main__":
    app.run()
