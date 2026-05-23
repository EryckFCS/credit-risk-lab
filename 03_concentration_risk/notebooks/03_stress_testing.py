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
app = marimo.App(width="medium", app_title="03 · Stress Testing & Capital Add-on Basel")


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # 🔴 Stress Testing — Concentración y Capital Basel
        Aplicación de tres escenarios de estrés (Base / Adverse / Severely Adverse)
        y cálculo del **capital add-on por concentración** según BCBS §773.

        \[
        \Delta K_{conc} = K_{base} \times \left(\frac{HHI}{HHI_{ref}} - 1\right) \times \phi
        \]
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
    import matplotlib.gridspec as gridspec
    import seaborn as sns

    warnings.filterwarnings("ignore")
    ROOT = Path(".").resolve().parents[1] if "notebooks" in str(Path(".").resolve()) else Path(".")
    DATA_DIR  = ROOT / "03_concentration_risk" / "data"
    REPORTS_DIR = ROOT / "03_concentration_risk" / "reports"

    PALETTE = {
        "primary": "#01696f", "secondary": "#437a22",
        "warning": "#d19900", "error": "#a12c7b",
        "blue": "#006494", "neutral": "#7a7974",
    }
    print("✅ Imports OK")
    return DATA_DIR, PALETTE, REPORTS_DIR, ROOT, gridspec, mo, np, pd, plt, sns, warnings


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 1 · Cargar datos")
    return


@app.cell
def __(DATA_DIR, pd):
    df_panel   = pd.read_parquet(DATA_DIR / "sector_panel.parquet")
    df_hhi     = pd.read_parquet(DATA_DIR / "concentration_metrics.parquet")
    df_sistema = pd.read_parquet(DATA_DIR / "hhi_sistema.parquet")
    print(f"Panel: {df_panel.shape} | HHI banco: {df_hhi.shape} | HHI sistema: {df_sistema.shape}")
    return df_hhi, df_panel, df_sistema


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 2 · Definición de escenarios de estrés (BCBS / Fed DFAST framework)")
    return


@app.cell
def __(mo):
    SCENARIOS = {
        "Base": {
            "delta_morosidad": 0.000,
            "pib": 0.020,
            "desempleo_delta": 0.000,
            "color": "#437a22",
            "descripcion": "Condiciones macroeconómicas actuales (2024)",
        },
        "Adverse": {
            "delta_morosidad": 0.015,
            "pib": -0.015,
            "desempleo_delta": 0.020,
            "color": "#d19900",
            "descripcion": "Recesión moderada — caída commodities, contracción crédito",
        },
        "Severely Adverse": {
            "delta_morosidad": 0.040,
            "pib": -0.040,
            "desempleo_delta": 0.050,
            "color": "#a12c7b",
            "descripcion": "Crisis sistémica (escala COVID-19 2020)",
        },
    }
    mo.md(f"""
    | Escenario | Δ Morosidad | PIB | Δ Desempleo | Descripción |
    |-----------|-------------|-----|-------------|-------------|
    | Base | +{SCENARIOS['Base']['delta_morosidad']:.0%} | +{SCENARIOS['Base']['pib']:.1%} | +{SCENARIOS['Base']['desempleo_delta']:.0%}pp | {SCENARIOS['Base']['descripcion']} |
    | Adverse | +{SCENARIOS['Adverse']['delta_morosidad']:.0%} | {SCENARIOS['Adverse']['pib']:.1%} | +{SCENARIOS['Adverse']['desempleo_delta']:.0%}pp | {SCENARIOS['Adverse']['descripcion']} |
    | Severely Adverse | +{SCENARIOS['Severely Adverse']['delta_morosidad']:.0%} | {SCENARIOS['Severely Adverse']['pib']:.1%} | +{SCENARIOS['Severely Adverse']['desempleo_delta']:.0%}pp | {SCENARIOS['Severely Adverse']['descripcion']} |
    """)
    return (SCENARIOS,)


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 3 · EL y capital requirements bajo cada escenario")
    return


@app.cell
def __(SCENARIOS, df_hhi, df_panel, np, pd):
    # Baseline: métricas actuales (dic-2024)
    latest_date = df_hhi["fecha"].max()
    df_base = df_hhi[df_hhi["fecha"] == latest_date].copy()

    # Cartera total por banco
    cartera_banco = (
        df_panel[df_panel["fecha"] == latest_date]
        .groupby("banco")["cartera"].sum().reset_index(name="cartera_total")
    )
    morosidad_banco = (
        df_panel[df_panel["fecha"] == latest_date]
        .groupby("banco")
        .apply(lambda g: np.average(g["morosidad"], weights=g["cartera"]))
        .reset_index(name="morosidad_base")
    )
    df_base = df_base.merge(cartera_banco, on="banco").merge(morosidad_banco, on="banco")

    # Parámetros Basel para Ecuador (proxy Standardised Approach)
    LGD = 0.45        # LGD asumido (45% — BCBS SA para exposiciones corporativas)
    RW  = 1.00        # Risk Weight (100% SA)
    CAR = 0.105       # Capital Adequacy Ratio mínimo Basel III (8% Tier1 + 2.5% buffer)
    HHI_REF = 1000.0  # HHI referencia (cartera perfectamente diversificada)
    PHI_MAP = {       # Factor ajuste φ por segmento (BCBS 2006 §773 proxy)
        "Banco_A": 0.12, "Banco_B": 0.12, "Banco_C": 0.14,
        "Banco_D": 0.16, "Banco_E": 0.18, "Banco_F": 0.20,
        "Banco_G": 0.20, "Banco_H": 0.22, "Banco_I": 0.23, "Banco_J": 0.25,
    }
    df_base["phi"] = df_base["banco"].map(PHI_MAP)

    results = []
    for _, row in df_base.iterrows():
        for scen_name, scen in SCENARIOS.items():
            mor_stress = row["morosidad_base"] + scen["delta_morosidad"]
            pd_stress  = min(mor_stress, 0.99)  # PD proxy

            # Expected Loss
            EL = pd_stress * LGD * row["cartera_total"]

            # Capital base (8% RWA)
            K_base = row["cartera_total"] * RW * CAR

            # Capital add-on por concentración (BCBS §773)
            hhi_ratio = max(0, row["hhi"] / HHI_REF - 1)
            K_addon = K_base * hhi_ratio * row["phi"]

            # Capital total requerido
            K_total = K_base + K_addon

            # Capital ratio bajo estrés
            capital_ratio_stress = max(0, (K_base - EL) / row["cartera_total"])

            results.append({
                "banco": row["banco"],
                "escenario": scen_name,
                "hhi": row["hhi"],
                "morosidad_stress": mor_stress,
                "EL_USD_mn": EL / 1e6,
                "K_base_USD_mn": K_base / 1e6,
                "K_addon_USD_mn": K_addon / 1e6,
                "K_total_USD_mn": K_total / 1e6,
                "capital_ratio_stress": capital_ratio_stress,
                "breach_CAR": capital_ratio_stress < CAR,
            })

    df_stress = pd.DataFrame(results)
    print("=== STRESS TEST RESULTS (Severely Adverse) ===")
    sa = df_stress[df_stress["escenario"] == "Severely Adverse"].sort_values("K_addon_USD_mn", ascending=False)
    print(sa[["banco", "hhi", "EL_USD_mn", "K_addon_USD_mn", "capital_ratio_stress", "breach_CAR"]]
          .round(2).to_string(index=False))
    print(f"\nBancos con breach CAR: {sa['breach_CAR'].sum()} / {len(sa)}")
    return (
        CAR, HHI_REF, K_addon, K_base, K_total, LGD, PHI_MAP, RW,
        cartera_banco, df_base, df_stress, latest_date, morosidad_banco,
        pd_stress, results, row, sa, scen, scen_name,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 4 · Dashboard de resultados — Stress Test")
    return


@app.cell
def __(CAR, PALETTE, SCENARIOS, df_stress, gridspec, np, pd, plt):
    fig = plt.figure(figsize=(18, 14), facecolor="white")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)
    fig.suptitle("Stress Testing — Concentración Crediticia · SBS Ecuador",
                 fontsize=15, fontweight="bold")

    bancos = df_stress["banco"].unique()
    scen_names = ["Base", "Adverse", "Severely Adverse"]
    scen_colors = [SCENARIOS[s]["color"] for s in scen_names]

    # ── A) EL por escenario (barras agrupadas)
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(bancos))
    width = 0.25
    for i, (scen, col) in enumerate(zip(scen_names, scen_colors)):
        sub = df_stress[df_stress["escenario"] == scen].sort_values("banco")
        ax1.bar(x + i * width, sub["EL_USD_mn"], width, label=scen, color=col, alpha=0.85)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(bancos, fontsize=9)
    ax1.set_ylabel("Expected Loss (USD Millones)")
    ax1.set_title("Expected Loss por Banco y Escenario", fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # ── B) Capital add-on vs Capital base
    ax2 = fig.add_subplot(gs[1, :2])
    sa_sub = df_stress[df_stress["escenario"] == "Severely Adverse"].sort_values("banco")
    ax2.bar(sa_sub["banco"], sa_sub["K_base_USD_mn"],
            label="K Base (8% RWA)", color=PALETTE["primary"], alpha=0.85)
    ax2.bar(sa_sub["banco"], sa_sub["K_addon_USD_mn"],
            bottom=sa_sub["K_base_USD_mn"],
            label="K Add-on Concentración", color=PALETTE["error"], alpha=0.85)
    ax2.set_title("Capital Requerido: Base + Add-on Concentración\n(Escenario Severely Adverse)",
                  fontweight="bold")
    ax2.set_ylabel("Capital (USD Millones)")
    ax2.legend(fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    # ── C) Capital ratio bajo estrés vs threshold
    ax3 = fig.add_subplot(gs[1, 2])
    for scen, col in zip(scen_names, scen_colors):
        sub = df_stress[df_stress["escenario"] == scen].sort_values("banco")
        ax3.plot(sub["banco"], sub["capital_ratio_stress"] * 100,
                 marker="o", color=col, label=scen, lw=2)
    ax3.axhline(CAR * 100, color="black", lw=1.5, ls="--", label=f"Min CAR {CAR:.1%}")
    ax3.set_title("Capital Ratio bajo Estrés", fontweight="bold")
    ax3.set_ylabel("%")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax3.legend(fontsize=8)
    ax3.tick_params(axis="x", rotation=45)
    ax3.spines[["top", "right"]].set_visible(False)

    # ── D) Mapa de calor: HHI × Escenario × Capital add-on
    ax4 = fig.add_subplot(gs[2, :])
    pivot = df_stress.pivot_table(
        index="banco", columns="escenario", values="K_addon_USD_mn"
    )[scen_names]
    import seaborn as sns
    sns.heatmap(
        pivot, annot=True, fmt=".1f", ax=ax4,
        cmap="RdYlGn_r",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Capital Add-on (USD Mn)"}
    )
    ax4.set_title("Capital Add-on por Concentración × Escenario (USD Millones)",
                  fontweight="bold")
    ax4.set_xlabel("")
    ax4.set_ylabel("")

    plt.savefig("03_concentration_risk/reports/05_stress_test_dashboard.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Dashboard guardado → reports/05_stress_test_dashboard.png")
    return (
        ax1, ax2, ax3, ax4, col, fig, gs, i, pivot,
        sa_sub, scen, scen_colors, scen_names, sns, sub, width, x,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md("## 5 · Model Card — Concentration Risk Report")
    return


@app.cell
def __(CAR, LGD, REPORTS_DIR, df_stress, np):
    from datetime import datetime

    sa_data = df_stress[df_stress["escenario"] == "Severely Adverse"]
    adv_data = df_stress[df_stress["escenario"] == "Adverse"]

    card = f"""# Concentration Risk Report — SBS Ecuador

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Resumen Ejecutivo

Análisis de concentración crediticia del sistema bancario privado ecuatoriano
bajo el framework BCBS (2006) §773. Panel 10 bancos × 84 meses (2018–2024).

## Métricas de Concentración (dic-2024)

| Banco | HHI | Clasificación |
|-------|-----|---------------|
| Ver `03_hhi_dashboard.png` | | |

## Resultados Stress Test

### Escenario Adverse (+150bps morosidad, PIB -1.5%)
- Total EL sistema: **USD {adv_data['EL_USD_mn'].sum():.1f}M**
- Capital add-on total: **USD {adv_data['K_addon_USD_mn'].sum():.1f}M**
- Bancos con breach CAR ({CAR:.1%}): **{adv_data['breach_CAR'].sum()}**

### Escenario Severely Adverse (+400bps morosidad, PIB -4.0%)
- Total EL sistema: **USD {sa_data['EL_USD_mn'].sum():.1f}M**
- Capital add-on total: **USD {sa_data['K_addon_USD_mn'].sum():.1f}M**
- Bancos con breach CAR ({CAR:.1%}): **{sa_data['breach_CAR'].sum()}**
- Banco más vulnerable: **{sa_data.loc[sa_data['K_addon_USD_mn'].idxmax(), 'banco']}** (HHI={sa_data.loc[sa_data['K_addon_USD_mn'].idxmax(), 'hhi']:.0f})

## Supuestos

| Parámetro | Valor | Fuente |
|-----------|-------|--------|
| LGD | {LGD:.0%} | BCBS SA — exposiciones corporativas |
| Risk Weight | 100% | BCBS Standardised Approach |
| CAR mínimo | {CAR:.1%} | Basel III (8% Tier1 + 2.5% conservation buffer) |
| HHI referencia | 1000 | Cartera perfectamente diversificada |
| φ factor | 0.12–0.25 | BCBS §773 proxy por tamaño banco |

## Limitaciones

- Datos sintéticos calibrados: no reflejan la composición real de cartera individual.
- LGD uniforme: en la práctica varía por garantía, segmento y contraparte.
- Sin modelamiento de correlaciones entre sectores bajo estrés (copula).
- Capital add-on es una aproximación al Pilar 2 Basel; no reemplaza el ICAAP.
- Escenarios calibrados con datos históricos SBS; out-of-distribution events no cubiertos.

## Referencias

- BCBS (2006). Basel II §773. Capital add-on for sector concentration. BIS.
- Düllmann & Masschelein (2007). Sector Concentration Risk. JFSR.
- Gordy (2003). Risk-Factor Model for Ratings-Based Capital Rules. JFI.
"""

    card_path = REPORTS_DIR / "concentration_risk_report.md"
    card_path.write_text(card)
    print(f"✅ Report exportado → {card_path}")
    print("\n🏁 Case Study 03 completado.")
    return card, card_path, datetime, sa_data, adv_data


if __name__ == "__main__":
    app.run()
