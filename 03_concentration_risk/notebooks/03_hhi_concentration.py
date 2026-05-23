# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.13.0",
#   "pandas>=2.2.0",
#   "numpy>=1.26.0",
#   "matplotlib>=3.8.0",
#   "plotly>=5.20.0",
# ]
# //
import marimo

__generated_with = "0.13.0"
app = marimo.App(
    width="full",
    app_title="Concentration Risk — HHI & Sector Analysis",
)


@app.cell
def _():
    import marimo as mo
    mo.md("""
    # 🏛️ Case Study 03 — Concentration Risk: HHI & Sector Analysis

    **Objective:** Measure credit portfolio concentration by sector, counterparty,
    and geography using regulatory-grade indices, then simulate stress scenarios.

    **Methods:**
    - Herfindahl-Hirschman Index (HHI)
    - Concentration Ratio CR5 / CR10
    - Gini concentration curve
    - Single-name stress test (top-5 obligor shock)

    **Data:** SBS Ecuador Boletín Financiero (sector breakdown) / Synthetic fallback  
    **Reference:** BCBS (2006). *Studies on credit concentration risk*. BIS Working Paper.
    """)
    return (mo,)


@app.cell
def _(mo):
    import numpy as np
    import pandas as pd
    import sys
    from pathlib import Path

    NOTEBOOK_DIR = Path(__file__).parent
    REPO_ROOT    = NOTEBOOK_DIR.parent.parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # Synthetic sector exposure (SBS-style categories)
    np.random.seed(7)
    sectors = [
        "Comercio", "Manufactura", "Servicios", "Agricultura",
        "Construcción", "Transporte", "Minería", "Turismo",
        "Educación", "Salud"
    ]
    raw_exposures = np.random.exponential(scale=1.5, size=len(sectors))
    exposures     = raw_exposures / raw_exposures.sum()

    df_sectors = pd.DataFrame({
        "sector":       sectors,
        "exposure_pct": (exposures * 100).round(2),
        "exposure_usd": (exposures * 1_000_000_000).round(0),  # synthetic $1B portfolio
    }).sort_values("exposure_pct", ascending=False).reset_index(drop=True)

    mo.vstack([
        mo.md("## 🏦 Portfolio Sector Breakdown (Synthetic — $1B)"),
        mo.ui.table(df_sectors)
    ])
    return NOTEBOOK_DIR, REPO_ROOT, df_sectors, exposures, np, pd, raw_exposures, sectors, sys


@app.cell
def _(df_sectors, mo, np):
    # ── Concentration indices ────────────────────────────────────────────────
    w = df_sectors["exposure_pct"].values / 100  # weights sum to 1

    hhi      = float(np.sum(w ** 2))
    hhi_norm = (hhi - 1/len(w)) / (1 - 1/len(w))  # normalized [0,1]
    cr5      = float(w[:5].sum())
    cr10     = float(w[:10].sum())

    # Thresholds per BCBS guidelines
    hhi_label = "Low" if hhi < 0.10 else ("Moderate" if hhi < 0.18 else "High")

    mo.callout(
        mo.md(f"""
        ## 📊 Concentration Indices

        | Index | Value | Interpretation |
        |---|---|---|
        | **HHI** | `{hhi:.4f}` | {hhi_label} concentration |
        | **HHI Normalized** | `{hhi_norm:.4f}` | Adjusted for n={len(w)} sectors |
        | **CR5** | `{cr5:.2%}` | Top 5 sectors share |
        | **CR10** | `{cr10:.2%}` | Top 10 sectors share |

        > BCBS thresholds: HHI < 0.10 = low, 0.10-0.18 = moderate, > 0.18 = high
        """),
        kind="success" if hhi < 0.10 else ("warn" if hhi < 0.18 else "danger")
    )
    return cr10, cr5, hhi, hhi_label, hhi_norm, w


@app.cell
def _(df_sectors, mo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    # Lorenz / Gini concentration curve
    w_sorted  = np.sort(df_sectors["exposure_pct"].values / 100)
    cum_share  = np.cumsum(w_sorted) / w_sorted.sum()
    pop_share  = np.arange(1, len(w_sorted) + 1) / len(w_sorted)
    gini_conc  = 1 - 2 * np.trapz(cum_share, pop_share)

    axes[0].plot([0] + list(pop_share), [0] + list(cum_share),
                 color="#a12c7b", linewidth=2.5, label=f"Lorenz (Gini={gini_conc:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#bab9b4", linewidth=1.5, label="Perfect equality")
    axes[0].fill_between([0] + list(pop_share), [0] + list(cum_share),
                          [0] + list(pop_share), alpha=0.15, color="#a12c7b")
    axes[0].set_title("Lorenz Curve — Sector Concentration", fontweight="bold")
    axes[0].set_xlabel("Cumulative share of sectors")
    axes[0].set_ylabel("Cumulative share of exposure")
    axes[0].legend()

    # Bar chart
    colors = ["#01696f" if i < 3 else "#4f98a3" if i < 6 else "#cedcd8"
               for i in range(len(df_sectors))]
    axes[1].barh(df_sectors["sector"], df_sectors["exposure_pct"],
                  color=colors, edgecolor="white")
    axes[1].set_title("Sector Exposure (%)", fontweight="bold")
    axes[1].set_xlabel("Portfolio weight (%)")
    axes[1].invert_yaxis()

    plt.tight_layout()
    mo.mpl.interactive(fig)
    return axes, colors, cum_share, fig, gini_conc, matplotlib, plt, pop_share, w_sorted


@app.cell
def _(df_sectors, mo, np):
    # ── Single-name stress test ──────────────────────────────────────────────
    SHOCK_LGD = 0.45  # assume 45% LGD on shocked sector
    top5 = df_sectors.head(5).copy()
    top5["stressed_loss_pct"] = (top5["exposure_pct"] / 100 * SHOCK_LGD * 100).round(3)

    total_portfolio_usd = 1_000_000_000
    top5["stressed_loss_usd"] = (top5["exposure_pct"] / 100 * SHOCK_LGD * total_portfolio_usd).round(0)

    mo.vstack([
        mo.md(f"## ⚠️ Stress Test: Top-5 Sector Default (LGD = {SHOCK_LGD:.0%})"),
        mo.ui.table(top5[["sector", "exposure_pct", "stressed_loss_pct", "stressed_loss_usd"]]),
        mo.md(f"**Total stressed loss (top 5 simultaneous default):** `${top5['stressed_loss_usd'].sum():,.0f}` ({top5['stressed_loss_pct'].sum():.2f}% of portfolio)")
    ])
    return SHOCK_LGD, top5, total_portfolio_usd


@app.cell
def _(mo):
    mo.md("""
    ---
    ## 🗓️ Next Steps
    - [ ] Source SBS Ecuador sector breakdown (Boletín Financiero)
    - [ ] Geographic concentration (province-level HHI)
    - [ ] Multi-period trend: HHI evolution 2019-2024
    - [ ] Export stress test report
    """)
    return


if __name__ == "__main__":
    app.run()
