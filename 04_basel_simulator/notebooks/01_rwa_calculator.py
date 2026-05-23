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
app = marimo.App(width="medium", app_title="04 · Basel III — RWA Calculator")

@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""
    # 🏦 Basel III — RWA Calculator (Standardised Approach)
    Cálculo de **Risk-Weighted Assets** por tipo de exposición
    usando el Standardised Approach de Basel III (BCBS 2017).
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
    import seaborn as sns

    warnings.filterwarnings("ignore")
    ROOT = Path(".").resolve().parents[1] if "notebooks" in str(Path(".").resolve()) else Path(".")
    DATA_DIR    = ROOT / "04_basel_simulator" / "data"
    REPORTS_DIR = ROOT / "04_basel_simulator" / "reports"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    PALETTE = {"primary": "#01696f", "secondary": "#437a22",
               "warning": "#d19900", "error": "#a12c7b", "blue": "#006494"}
    print("✅ Imports OK")
    return DATA_DIR, PALETTE, REPORTS_DIR, ROOT, mo, np, pd, plt, sns, warnings

@app.cell(hide_code=True)
def __(mo):
    mo.md("## 1 · Risk Weights — Standardised Approach BCBS 2017")
    return

@app.cell
def __(pd):
    # Risk Weights por clase de activo (BCBS 2017 — SA)
    RISK_WEIGHTS = {
        "soberano_aaa":     {"rw": 0.00, "descripcion": "Soberano AAA-AA (ej. US Treasury)"},
        "soberano_a":       {"rw": 0.20, "descripcion": "Soberano A (ej. Chile, Perú)"},
        "soberano_bbb":     {"rw": 0.50, "descripcion": "Soberano BBB (ej. Ecuador)"},
        "banco_a":          {"rw": 0.50, "descripcion": "Banco contraparte rating A"},
        "banco_bbb":        {"rw": 1.00, "descripcion": "Banco contraparte BBB"},
        "corporativo_aaa":  {"rw": 0.65, "descripcion": "Corporativo AAA-AA (BCBS 2017)"},
        "corporativo_a":    {"rw": 1.00, "descripcion": "Corporativo A-BBB"},
        "corporativo_bb":   {"rw": 1.50, "descripcion": "Corporativo BB y below"},
        "retail":           {"rw": 0.75, "descripcion": "Retail regulatorio (consumo)"},
        "hipotecario_ltv60":{"rw": 0.35, "descripcion": "Hipotecario LTV ≤ 60%"},
        "hipotecario_ltv80":{"rw": 0.75, "descripcion": "Hipotecario 60% < LTV ≤ 80%"},
        "hipotecario_ltv90":{"rw": 1.00, "descripcion": "Hipotecario LTV > 80%"},
        "default":          {"rw": 1.50, "descripcion": "Exposición en default (vencida)"},
        "equity_listed":    {"rw": 2.50, "descripcion": "Equity cotizado en bolsa"},
        "equity_unlisted":  {"rw": 4.00, "descripcion": "Equity no cotizado"},
    }
    df_rw = pd.DataFrame(RISK_WEIGHTS).T.reset_index().rename(columns={"index": "clase"})
    df_rw["rw"] = df_rw["rw"].astype(float)
    print(df_rw[["clase", "rw", "descripcion"]].to_string(index=False))
    return (RISK_WEIGHTS, df_rw)

@app.cell(hide_code=True)
def __(mo):
    mo.md("## 2 · Portafolio de exposiciones — Banco ejemplo (proxy SBS Ecuador)")
    return

@app.cell
def __(RISK_WEIGHTS, np, pd):
    np.random.seed(42)
    # Composición típica de un banco mediano ecuatoriano (USD millones)
    PORTFOLIO = [
        {"clase": "soberano_bbb",     "exposicion": 420.0,  "descripcion": "Bonos BCE / T-bills Ecuador"},
        {"clase": "banco_a",          "exposicion": 180.0,  "descripcion": "Depósitos interbancarios"},
        {"clase": "corporativo_a",    "exposicion": 650.0,  "descripcion": "Crédito comercial ordinario"},
        {"clase": "corporativo_bb",   "exposicion": 120.0,  "descripcion": "Crédito PYME alto riesgo"},
        {"clase": "retail",           "exposicion": 890.0,  "descripcion": "Crédito consumo personas"},
        {"clase": "hipotecario_ltv60","exposicion": 310.0,  "descripcion": "Hipotecario LTV bajo"},
        {"clase": "hipotecario_ltv80","exposicion": 195.0,  "descripcion": "Hipotecario LTV medio"},
        {"clase": "hipotecario_ltv90","exposicion": 85.0,   "descripcion": "Hipotecario LTV alto"},
        {"clase": "default",          "exposicion": 62.0,   "descripcion": "Cartera vencida / en riesgo"},
        {"clase": "equity_listed",    "exposicion": 28.0,   "descripcion": "Inversiones en equity BVQ"},
    ]
    df_port = pd.DataFrame(PORTFOLIO)
    df_port["rw"]  = df_port["clase"].map({k: v["rw"] for k, v in RISK_WEIGHTS.items()})
    df_port["rwa"] = df_port["exposicion"] * df_port["rw"]

    TOTAL_ASSETS = df_port["exposicion"].sum()
    TOTAL_RWA    = df_port["rwa"].sum()
    print(f"Total Activos:  USD {TOTAL_ASSETS:,.1f}M")
    print(f"Total RWA:      USD {TOTAL_RWA:,.1f}M")
    print(f"RWA Density:    {TOTAL_RWA/TOTAL_ASSETS:.1%}")
    print(f"\nDescomposición RWA:")
    print(df_port[["clase", "exposicion", "rw", "rwa"]].to_string(index=False))
    return (PORTFOLIO, TOTAL_ASSETS, TOTAL_RWA, df_port)

@app.cell(hide_code=True)
def __(mo):
    mo.md("## 3 · Capital Requirements — Basel III")
    return

@app.cell
def __(TOTAL_RWA, pd):
    # Capital structure (supuesto banco mediano Ecuador)
    CAPITAL = {
        "CET1":           850.0,   # Common Equity Tier 1
        "AT1":            95.0,    # Additional Tier 1 (bonos AT1)
        "Tier2":          120.0,   # Tier 2 (deuda subordinada)
    }
    CAPITAL["Tier1"]  = CAPITAL["CET1"] + CAPITAL["AT1"]
    CAPITAL["Total"]  = CAPITAL["Tier1"] + CAPITAL["Tier2"]

    REQS = {
        "CET1":                   {"valor": CAPITAL["CET1"]   / TOTAL_RWA, "minimo": 0.045, "buffer": 0.025},
        "Tier1":                  {"valor": CAPITAL["Tier1"]  / TOTAL_RWA, "minimo": 0.060, "buffer": 0.025},
        "Total Capital":          {"valor": CAPITAL["Total"]  / TOTAL_RWA, "minimo": 0.080, "buffer": 0.025},
        "CAR (con conservation)": {"valor": CAPITAL["Total"]  / TOTAL_RWA, "minimo": 0.105, "buffer": 0.000},
    }

    print("\n=== CAPITAL ADEQUACY RATIOS ===")
    for name, d in REQS.items():
        req = d["minimo"] + d["buffer"]
        flag = "✅" if d["valor"] >= req else "🔴"
        surplus = (d["valor"] - req) * TOTAL_RWA
        print(f"  {flag} {name:28s}: {d['valor']:.2%}  (req={req:.1%}, surplus=USD {surplus:,.1f}M)")

    df_cap = pd.DataFrame(REQS).T.reset_index().rename(columns={"index": "ratio"})
    return (CAPITAL, REQS, df_cap)

@app.cell(hide_code=True)
def __(mo):
    mo.md("## 4 · Visualización RWA + Capital")
    return

@app.cell
def __(CAPITAL, PALETTE, REQS, TOTAL_RWA, df_port, plt):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="white")
    fig.suptitle("Basel III Capital Simulator — Banco Ejemplo SBS Ecuador",
                 fontsize=13, fontweight="bold")

    # A) RWA por clase de activo
    df_sorted = df_port.sort_values("rwa", ascending=True)
    colors_bar = [PALETTE["error"] if rw >= 1.5
                  else PALETTE["warning"] if rw >= 0.75
                  else PALETTE["primary"] for rw in df_sorted["rw"]]
    axes[0].barh(df_sorted["clase"], df_sorted["rwa"], color=colors_bar, alpha=0.85)
    axes[0].set_xlabel("RWA (USD M)")
    axes[0].set_title("RWA por Clase de Activo", fontweight="bold")
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].tick_params(axis="y", labelsize=8)

    # B) Composición de capital
    cap_labels = ["CET1", "AT1", "Tier2"]
    cap_vals   = [CAPITAL["CET1"], CAPITAL["AT1"], CAPITAL["Tier2"]]
    cap_colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["warning"]]
    wedges, texts, autotexts = axes[1].pie(
        cap_vals, labels=cap_labels, autopct="%1.1f%%",
        colors=cap_colors, startangle=90,
        wedgeprops={"linewidth": 2, "edgecolor": "white"}
    )
    axes[1].set_title(f"Composición Capital\nTotal: USD {sum(cap_vals):,.0f}M",
                      fontweight="bold")

    # C) Gauge: Capital Ratios vs Thresholds
    ratios = [d["valor"] for d in REQS.values()]
    thresholds = [d["minimo"] + d["buffer"] for d in REQS.values()]
    ratio_names = list(REQS.keys())
    x = range(len(ratio_names))
    bars = axes[2].bar(x, [r * 100 for r in ratios],
                       color=[PALETTE["primary"] if r >= t else PALETTE["error"]
                              for r, t in zip(ratios, thresholds)],
                       alpha=0.85, width=0.5)
    for xi, t in zip(x, thresholds):
        axes[2].axhline(y=t * 100, xmin=(xi - 0.3) / len(x), xmax=(xi + 0.8) / len(x),
                        color="black", lw=2, linestyle="--")
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels([r.replace(" ", "\n") for r in ratio_names], fontsize=8)
    axes[2].set_ylabel("%")
    axes[2].set_title("Capital Ratios vs Mínimos Basel III", fontweight="bold")
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    axes[2].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("04_basel_simulator/reports/01_rwa_capital.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Figura guardada → reports/01_rwa_capital.png")
    return axes, autotexts, bars, cap_colors, cap_labels, cap_vals, colors_bar, fig, ratio_names, ratios, texts, thresholds, wedges, x, xi

@app.cell(hide_code=True)
def __(mo):
    mo.md("## 5 · Exportar capital base")
    return

@app.cell
def __(CAPITAL, DATA_DIR, REQS, TOTAL_ASSETS, TOTAL_RWA, pd):
    summary = pd.DataFrame([{
        "total_assets": TOTAL_ASSETS,
        "total_rwa": TOTAL_RWA,
        "rwa_density": TOTAL_RWA / TOTAL_ASSETS,
        "cet1": CAPITAL["CET1"],
        "tier1": CAPITAL["Tier1"],
        "total_capital": CAPITAL["Total"],
        "cet1_ratio": REQS["CET1"]["valor"],
        "tier1_ratio": REQS["Tier1"]["valor"],
        "car": REQS["CAR (con conservation)"]["valor"],
    }])
    summary.to_parquet(DATA_DIR / "capital_base.parquet", index=False)
    print("✅ capital_base.parquet exportado")
    print("\n🔜 Siguiente: 02_liquidity_ratios.py")
    return (summary,)

if __name__ == "__main__":
    app.run()
