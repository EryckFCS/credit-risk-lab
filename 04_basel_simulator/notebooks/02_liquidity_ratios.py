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
app = marimo.App(width="medium", app_title="04 · Basel III — LCR & NSFR")

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    # 💧 Basel III — Liquidity Coverage Ratio (LCR) & NSFR
    Cálculo de los dos ratios de liquidez de Basel III:
    - **LCR**: cubre el estrés de liquidez a 30 días (BCBS 2013)
    - **NSFR**: estructura de fondeo estable a 1 año (BCBS 2014)
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
    import matplotlib.patches as mpatches
    import seaborn as sns

    warnings.filterwarnings("ignore")
    ROOT = Path(".").resolve().parents[1] if "notebooks" in str(Path(".").resolve()) else Path(".")
    DATA_DIR    = ROOT / "04_basel_simulator" / "data"
    REPORTS_DIR = ROOT / "04_basel_simulator" / "reports"

    PALETTE = {"primary": "#01696f", "secondary": "#437a22",
               "warning": "#d19900", "error": "#a12c7b", "blue": "#006494"}
    print("✅ Imports OK")
    return DATA_DIR, PALETTE, REPORTS_DIR, ROOT, mo, mpatches, np, pd, plt, sns, warnings

@app.cell(hide_code=True)
def __(mo):
    mo.md("## 1 · LCR — High Quality Liquid Assets & Net Cash Outflows")
    return

@app.cell
def __(pd):
    # HQLA (High Quality Liquid Assets)
    HQLA = {
        "L1_cash":          {"valor": 180.0, "haircut": 0.00, "descripcion": "Caja y reservas BCE"},
        "L1_sovereign":     {"valor": 320.0, "haircut": 0.00, "descripcion": "Bonos soberanos 0% RW"},
        "L2A_sovereign":    {"valor": 140.0, "haircut": 0.15, "descripcion": "Bonos soberanos 20% RW (L2A)"},
        "L2A_covered":      {"valor": 80.0,  "haircut": 0.15, "descripcion": "Cédulas hipotecarias (L2A)"},
        "L2B_rmbs":         {"valor": 55.0,  "haircut": 0.25, "descripcion": "RMBS elegibles (L2B)"},
        "L2B_equity":       {"valor": 30.0,  "haircut": 0.50, "descripcion": "Equity en índice (L2B)"},
    }

    # Cash Outflows (30 días)
    OUTFLOWS = {
        "retail_deposits_stable":   {"valor": 850.0, "rate": 0.03, "descripcion": "Depósitos retail estables"},
        "retail_deposits_less":     {"valor": 320.0, "rate": 0.10, "descripcion": "Depósitos retail menos estables"},
        "wholesale_operat":         {"valor": 280.0, "rate": 0.25, "descripcion": "Depósitos wholesale operacionales"},
        "wholesale_non_operat":     {"valor": 190.0, "rate": 0.40, "descripcion": "Depósitos wholesale no operacionales"},
        "secured_funding":          {"valor": 120.0, "rate": 0.00, "descripcion": "Financiamiento garantizado HQLA"},
        "credit_facilities":        {"valor": 95.0,  "rate": 0.10, "descripcion": "Líneas de crédito comprometidas"},
        "liquidity_facilities":     {"valor": 45.0,  "rate": 0.30, "descripcion": "Líneas de liquidez comprometidas"},
    }

    # Cash Inflows (30 días) — cap al 75% de outflows
    INFLOWS = {
        "loans_performing":  {"valor": 180.0, "rate": 0.50, "descripcion": "Cobros préstamos al día"},
        "securities_maturing":{"valor": 90.0,  "rate": 1.00, "descripcion": "Vto. valores en 30d"},
        "other_inflows":     {"valor": 40.0,  "rate": 0.50, "descripcion": "Otros ingresos"},
    }

    df_hqla = pd.DataFrame(HQLA).T.reset_index().rename(columns={"index": "item"})
    df_hqla[["valor", "haircut"]] = df_hqla[["valor", "haircut"]].astype(float)
    df_hqla["hqla_adj"] = df_hqla["valor"] * (1 - df_hqla["haircut"])

    df_out = pd.DataFrame(OUTFLOWS).T.reset_index().rename(columns={"index": "item"})
    df_out[["valor", "rate"]] = df_out[["valor", "rate"]].astype(float)
    df_out["outflow"] = df_out["valor"] * df_out["rate"]

    df_in = pd.DataFrame(INFLOWS).T.reset_index().rename(columns={"index": "item"})
    df_in[["valor", "rate"]] = df_in[["valor", "rate"]].astype(float)
    df_in["inflow"] = df_in["valor"] * df_in["rate"]

    total_hqla      = df_hqla["hqla_adj"].sum()
    total_outflows  = df_out["outflow"].sum()
    total_inflows   = min(df_in["inflow"].sum(), 0.75 * total_outflows)  # cap 75%
    net_outflows    = total_outflows - total_inflows
    lcr             = total_hqla / net_outflows if net_outflows > 0 else float("inf")

    print(f"Total HQLA (ajustado): USD {total_hqla:,.1f}M")
    print(f"Total Outflows:        USD {total_outflows:,.1f}M")
    print(f"Total Inflows (cap):   USD {total_inflows:,.1f}M")
    print(f"Net Cash Outflows:     USD {net_outflows:,.1f}M")
    print(f"\n{'==='*10}")
    print(f"✅ LCR = {lcr:.1%}  (mínimo regulatorio: 100%)")
    flag = "✅ CUMPLE" if lcr >= 1.0 else "🔴 INCUMPLE"
    print(f"   {flag}  |  Excedente HQLA: USD {total_hqla - net_outflows:,.1f}M")
    return (
        HQLA, INFLOWS, OUTFLOWS, df_hqla, df_in, df_out,
        lcr, net_outflows, total_hqla, total_inflows, total_outflows,
    )

@app.cell(hide_code=True)
def __(mo):
    mo.md("## 2 · NSFR — Net Stable Funding Ratio")
    return

@app.cell
def __(pd):
    # Available Stable Funding (ASF)
    ASF_ITEMS = {
        "tier1_capital":     {"valor": 945.0,  "factor": 1.00, "descripcion": "Capital Tier 1"},
        "tier2_lt":          {"valor": 120.0,  "factor": 1.00, "descripcion": "Tier 2 plazo > 1 año"},
        "deposits_retail_lt":{"valor": 620.0,  "factor": 0.95, "descripcion": "Depósitos retail > 1 año"},
        "deposits_retail_st":{"valor": 550.0,  "factor": 0.90, "descripcion": "Depósitos retail < 1 año estables"},
        "wholesale_nfc_lt":  {"valor": 240.0,  "factor": 0.50, "descripcion": "Wholesale no financiero < 1 año"},
        "wholesale_other":   {"valor": 180.0,  "factor": 0.00, "descripcion": "Otros wholesale < 6m"},
    }

    # Required Stable Funding (RSF)
    RSF_ITEMS = {
        "l1_hqla":           {"valor": 500.0,  "factor": 0.00, "descripcion": "HQLA Nivel 1"},
        "l2a_hqla":          {"valor": 220.0,  "factor": 0.15, "descripcion": "HQLA Nivel 2A"},
        "unencumbered_loans_1y":{"valor": 380.0,"factor": 0.50, "descripcion": "Préstamos < 1 año sin garantizar"},
        "loans_retail_lt1y": {"valor": 520.0,  "factor": 0.85, "descripcion": "Préstamos retail > 1 año"},
        "loans_corp_lt1y":   {"valor": 640.0,  "factor": 1.00, "descripcion": "Préstamos corporativos > 1 año"},
        "other_assets":      {"valor": 95.0,   "factor": 1.00, "descripcion": "Otros activos"},
    }

    df_asf = pd.DataFrame(ASF_ITEMS).T.reset_index().rename(columns={"index": "item"})
    df_asf[["valor", "factor"]] = df_asf[["valor", "factor"]].astype(float)
    df_asf["asf"] = df_asf["valor"] * df_asf["factor"]

    df_rsf = pd.DataFrame(RSF_ITEMS).T.reset_index().rename(columns={"index": "item"})
    df_rsf[["valor", "factor"]] = df_rsf[["valor", "factor"]].astype(float)
    df_rsf["rsf"] = df_rsf["valor"] * df_rsf["factor"]

    ASF = df_asf["asf"].sum()
    RSF = df_rsf["rsf"].sum()
    nsfr = ASF / RSF if RSF > 0 else float("inf")

    print(f"Available Stable Funding (ASF): USD {ASF:,.1f}M")
    print(f"Required Stable Funding  (RSF): USD {RSF:,.1f}M")
    print(f"\n{'==='*10}")
    print(f"✅ NSFR = {nsfr:.1%}  (mínimo regulatorio: 100%)")
    flag = "✅ CUMPLE" if nsfr >= 1.0 else "🔴 INCUMPLE"
    print(f"   {flag}  |  Excedente ASF: USD {ASF - RSF:,.1f}M")
    return (ASF, ASF_ITEMS, RSF, RSF_ITEMS, df_asf, df_rsf, nsfr)

@app.cell(hide_code=True)
def __(mo):
    mo.md("## 3 · Visualización LCR + NSFR")
    return

@app.cell
def __(ASF, PALETTE, RSF, df_asf, df_hqla, df_out, df_rsf, lcr, net_outflows, nsfr, plt, total_hqla):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")
    fig.suptitle("Basel III Liquidity Ratios — LCR & NSFR", fontsize=13, fontweight="bold")

    # A) HQLA waterfall
    items_a = df_hqla.sort_values("hqla_adj", ascending=True)
    axes[0,0].barh(items_a["item"], items_a["hqla_adj"],
                   color=PALETTE["primary"], alpha=0.85)
    axes[0,0].set_title("HQLA por Categoría (ajustado)", fontweight="bold")
    axes[0,0].set_xlabel("USD M")
    axes[0,0].spines[["top", "right"]].set_visible(False)
    axes[0,0].tick_params(axis="y", labelsize=8)

    # B) LCR gauge visual
    lcr_pct = min(lcr * 100, 250)  # cap visual en 250%
    ax_lcr = axes[0,1]
    bar_color = PALETTE["primary"] if lcr >= 1.0 else PALETTE["error"]
    ax_lcr.barh(["HQLA", "Net Outflows"],
                [total_hqla, net_outflows],
                color=[PALETTE["primary"], PALETTE["warning"]], alpha=0.85)
    ax_lcr.set_title(f"LCR = {lcr:.1%}  ({'CUMPLE ✅' if lcr>=1 else 'INCUMPLE 🔴'})",
                     fontweight="bold",
                     color=PALETTE["primary"] if lcr >= 1 else PALETTE["error"])
    ax_lcr.set_xlabel("USD M")
    ax_lcr.spines[["top", "right"]].set_visible(False)

    # C) ASF vs RSF waterfall
    asf_sorted = df_asf.sort_values("asf", ascending=True)
    axes[1,0].barh(asf_sorted["item"], asf_sorted["asf"],
                   color=PALETTE["secondary"], alpha=0.85, label="ASF")
    axes[1,0].set_title("Available Stable Funding (ASF)", fontweight="bold")
    axes[1,0].set_xlabel("USD M")
    axes[1,0].spines[["top", "right"]].set_visible(False)
    axes[1,0].tick_params(axis="y", labelsize=8)

    # D) NSFR gauge
    ax_nsfr = axes[1,1]
    ax_nsfr.barh(["ASF", "RSF"],
                 [ASF, RSF],
                 color=[PALETTE["secondary"], PALETTE["warning"]], alpha=0.85)
    ax_nsfr.set_title(f"NSFR = {nsfr:.1%}  ({'CUMPLE ✅' if nsfr>=1 else 'INCUMPLE 🔴'})",
                      fontweight="bold",
                      color=PALETTE["primary"] if nsfr >= 1 else PALETTE["error"])
    ax_nsfr.set_xlabel("USD M")
    ax_nsfr.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("04_basel_simulator/reports/02_liquidity_ratios.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Figura guardada → reports/02_liquidity_ratios.png")
    return ax_lcr, ax_nsfr, axes, asf_sorted, bar_color, fig, items_a, lcr_pct

@app.cell(hide_code=True)
def __(mo):
    mo.md("## 4 · Exportar métricas de liquidez")
    return

@app.cell
def __(ASF, DATA_DIR, RSF, lcr, net_outflows, nsfr, pd, total_hqla, total_outflows):
    liq = pd.DataFrame([{
        "total_hqla": total_hqla,
        "net_cash_outflows": net_outflows,
        "lcr": lcr,
        "lcr_pass": lcr >= 1.0,
        "asf": ASF,
        "rsf": RSF,
        "nsfr": nsfr,
        "nsfr_pass": nsfr >= 1.0,
    }])
    liq.to_parquet(DATA_DIR / "liquidity_metrics.parquet", index=False)
    print("✅ liquidity_metrics.parquet exportado")
    print("\n🔜 Siguiente: 03_capital_dashboard.py")
    return (liq,)

if __name__ == "__main__":
    app.run()
