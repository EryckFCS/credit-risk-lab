# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.13.0",
#   "pandas>=2.2.0",
#   "numpy>=1.26.0",
#   "matplotlib>=3.8.0",
# ]
# //
import marimo

__generated_with = "0.13.0"
app = marimo.App(
    width="full",
    app_title="Basel III Capital Requirement Simulator",
)


@app.cell
def _():
    import marimo as mo
    mo.md("""
    # ⚖️ Case Study 04 — Basel III Capital Requirement Simulator

    **Objective:** Simulate Risk-Weighted Assets (RWA) and minimum capital requirements
    under the Basel III Standardized Approach for a synthetic credit portfolio.

    **Framework:**
    - Basel III Standardized Approach (SA) risk weights by exposure class
    - Minimum Total Capital Ratio: 8% of RWA
    - Tier 1 Capital Ratio: 6% of RWA
    - CET1 Ratio: 4.5% of RWA
    - Capital Conservation Buffer: +2.5%

    **Reference:** BCBS (2017). *Basel III: Finalising post-crisis reforms*. BIS.
    """)
    return (mo,)


@app.cell
def _(mo):
    import numpy as np
    import pandas as pd

    # Basel III SA risk weights by exposure class
    RISK_WEIGHTS = {
        "Sovereign (AAA-AA)": 0.00,
        "Sovereign (A)": 0.20,
        "Sovereign (BBB-BB)": 0.50,
        "Bank (AAA-AA)": 0.20,
        "Bank (A)": 0.50,
        "Corporate (AAA-AA)": 0.20,
        "Corporate (A)": 0.50,
        "Corporate (BBB-BB)": 1.00,
        "Corporate (unrated)": 1.00,
        "Retail (regulatory)": 0.75,
        "Residential Mortgage (LTV<80%)": 0.35,
        "Residential Mortgage (LTV>80%)": 0.75,
        "Commercial Real Estate": 1.00,
        "Past Due (>90 days)": 1.50,
        "SME (regulatory retail)": 0.75,
    }

    # Synthetic portfolio: exposure by class
    np.random.seed(99)
    classes = list(RISK_WEIGHTS.keys())
    raw_exp = np.random.exponential(2.0, len(classes))
    exposures_usd = (raw_exp / raw_exp.sum() * 500_000_000).round(0)  # $500M portfolio

    df_portfolio = pd.DataFrame({
        "exposure_class":  classes,
        "risk_weight":     [RISK_WEIGHTS[c] for c in classes],
        "exposure_usd":    exposures_usd,
    })
    df_portfolio["rwa_usd"] = (df_portfolio["exposure_usd"] * df_portfolio["risk_weight"]).round(0)

    total_exposure = df_portfolio["exposure_usd"].sum()
    total_rwa      = df_portfolio["rwa_usd"].sum()
    avg_rw         = total_rwa / total_exposure

    mo.vstack([
        mo.md("## 🏦 Portfolio RWA — Basel III Standardized Approach"),
        mo.ui.table(df_portfolio.assign(
            exposure_usd=df_portfolio["exposure_usd"].apply(lambda x: f"${x:,.0f}"),
            rwa_usd=df_portfolio["rwa_usd"].apply(lambda x: f"${x:,.0f}"),
            risk_weight=df_portfolio["risk_weight"].apply(lambda x: f"{x:.0%}")
        ))
    ])
    return (
        RISK_WEIGHTS, avg_rw, classes, df_portfolio, exposures_usd,
        np, pd, raw_exp, total_exposure, total_rwa,
    )


@app.cell
def _(avg_rw, mo, total_exposure, total_rwa):
    # Capital requirements
    cet1_req   = total_rwa * 0.045
    tier1_req  = total_rwa * 0.060
    total_req  = total_rwa * 0.080
    buffer_req = total_rwa * 0.025
    combined   = total_rwa * 0.105  # Total + conservation buffer

    mo.callout(
        mo.md(f"""
        ## 💰 Capital Requirements Summary

        | Component | Rate | Required Capital |
        |---|---|---|
        | **Total Exposure** | — | `${total_exposure:,.0f}` |
        | **Total RWA** | — | `${total_rwa:,.0f}` |
        | **Avg Risk Weight** | — | `{avg_rw:.2%}` |
        | **CET1 Minimum** | 4.5% | `${cet1_req:,.0f}` |
        | **Tier 1 Minimum** | 6.0% | `${tier1_req:,.0f}` |
        | **Total Capital Min** | 8.0% | `${total_req:,.0f}` |
        | **Conservation Buffer** | 2.5% | `${buffer_req:,.0f}` |
        | **Combined Buffer** | 10.5% | `${combined:,.0f}` |
        """),
        kind="info"
    )
    return buffer_req, cet1_req, combined, tier1_req, total_req


@app.cell
def _(df_portfolio, mo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    df_plot = df_portfolio.sort_values("rwa_usd", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    # RWA by class
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(df_plot)))
    axes[0].barh(df_plot["exposure_class"], df_plot["rwa_usd"] / 1e6,
                  color=colors, edgecolor="white")
    axes[0].set_title("RWA by Exposure Class ($M)", fontweight="bold")
    axes[0].set_xlabel("RWA ($ millions)")

    # Exposure vs RWA comparison
    x = np.arange(len(df_plot))
    axes[1].bar(x - 0.2, df_plot["exposure_usd"] / 1e6, 0.4,
                label="Exposure", color="#4f98a3", alpha=0.85)
    axes[1].bar(x + 0.2, df_plot["rwa_usd"] / 1e6, 0.4,
                label="RWA", color="#a12c7b", alpha=0.85)
    axes[1].set_title("Exposure vs RWA ($M)", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c.split("(")[0].strip() for c in df_plot["exposure_class"]],
                             rotation=45, ha="right", fontsize=8)
    axes[1].legend()

    plt.tight_layout()
    mo.mpl.interactive(fig)
    return axes, colors, df_plot, fig, matplotlib, plt, x


@app.cell
def _(mo):
    mo.md("""
    ---
    ## 🗓️ Next Steps
    - [ ] IRB approach: PD × LGD × EAD × M capital formula
    - [ ] Sensitivity analysis: capital vs. PD migration
    - [ ] ICAAP stress scenario (100bp rate shock)
    - [ ] Connect with Case Study 02 (PD estimates → IRB weights)
    """)
    return


if __name__ == "__main__":
    app.run()
