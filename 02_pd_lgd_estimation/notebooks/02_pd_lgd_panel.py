# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.13.0",
#   "pandas>=2.2.0",
#   "numpy>=1.26.0",
#   "statsmodels>=0.14.0",
#   "scipy>=1.12.0",
#   "matplotlib>=3.8.0",
#   "seaborn>=0.13.0",
# ]
# //
import marimo

__generated_with = "0.13.0"
app = marimo.App(
    width="full",
    app_title="PD & LGD Estimation — SME Panel",
)


@app.cell
def _():
    import marimo as mo
    mo.md("""
    # 🏦 Case Study 02 — PD & LGD Estimation: SME Lending

    **Objective:** Estimate Probability of Default (PD) and Loss Given Default (LGD)
    for a Small & Medium Enterprise (SME) credit portfolio using panel data methods.

    **Methods:**
    - PD: Panel Logit with time fixed effects (Logistic Regression on longitudinal data)
    - LGD: Beta Regression (bounded [0,1] response, mixture model for boundary observations)
    - Survival Analysis: Cox Proportional Hazards for time-to-default

    **Data target:** SBS Ecuador Microdatos de Cartera / Synthetic fallback  
    **Reference:** Schuermann, T. (2004). *What Do We Know About Loss Given Default?* Wharton.
    """)
    return (mo,)


@app.cell
def _(mo):
    import sys
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")

    NOTEBOOK_DIR = Path(__file__).parent
    CASE_DIR     = NOTEBOOK_DIR.parent
    REPO_ROOT    = CASE_DIR.parent
    DATA_DIR     = CASE_DIR / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    mo.md(f"**Repo root:** `{REPO_ROOT}`")
    return CASE_DIR, DATA_DIR, NOTEBOOK_DIR, Path, REPO_ROOT, np, pd, sys, warnings


@app.cell
def _(mo, np, pd):
    # Synthetic SME panel data (until SBS data is sourced)
    np.random.seed(42)
    N_FIRMS   = 500
    N_PERIODS = 8  # quarters

    firm_ids  = np.repeat(np.arange(N_FIRMS), N_PERIODS)
    periods   = np.tile(np.arange(N_PERIODS), N_FIRMS)

    df_panel = pd.DataFrame({
        "firm_id":        firm_ids,
        "period":         periods,
        "leverage":       np.clip(np.random.normal(0.55, 0.20, N_FIRMS * N_PERIODS), 0.05, 0.99),
        "current_ratio":  np.clip(np.random.lognormal(0.3, 0.4, N_FIRMS * N_PERIODS), 0.2, 8.0),
        "roa":            np.random.normal(0.04, 0.06, N_FIRMS * N_PERIODS),
        "loan_size_log":  np.random.normal(11.5, 1.2, N_FIRMS * N_PERIODS),
        "sector":         np.random.choice(["commerce","manufacturing","services","agriculture"], N_FIRMS * N_PERIODS),
    })

    # PD: logistic DGP
    log_odds = (-3.5 + 2.1 * df_panel["leverage"]
                - 0.8 * df_panel["current_ratio"]
                - 4.0 * df_panel["roa"]
                + 0.3 * np.random.normal(0, 1, N_FIRMS * N_PERIODS))
    df_panel["default"] = (np.random.uniform(size=N_FIRMS * N_PERIODS) < 1 / (1 + np.exp(-log_odds))).astype(int)

    # LGD: beta-distributed [0,1]
    df_panel["lgd"] = np.where(
        df_panel["default"] == 1,
        np.clip(np.random.beta(2, 5, N_FIRMS * N_PERIODS), 0.01, 0.99),
        np.nan
    )

    default_rate = df_panel["default"].mean()
    n_defaults   = df_panel["default"].sum()

    mo.callout(
        mo.md(f"**Synthetic panel generated** — {N_FIRMS} firms × {N_PERIODS} periods = `{len(df_panel):,}` obs  \nDefault rate: `{default_rate:.2%}` ({n_defaults} events)"),
        kind="info"
    )
    return df_panel, default_rate, log_odds, n_defaults


@app.cell
def _(df_panel, mo):
    import statsmodels.api as sm
    import pandas as pd

    # Panel PD model (pooled logit with sector dummies)
    FEATURES_PD = ["leverage", "current_ratio", "roa", "loan_size_log"]
    df_model = pd.get_dummies(df_panel[FEATURES_PD + ["sector", "default"]],
                               columns=["sector"], drop_first=True, dtype=float)
    df_model = df_model.dropna()

    X_pd = sm.add_constant(df_model.drop(columns=["default"]))
    y_pd = df_model["default"]

    logit_model = sm.Logit(y_pd, X_pd).fit(disp=False)

    summary_df = pd.DataFrame({
        "coef":     logit_model.params.round(4),
        "std_err":  logit_model.bse.round(4),
        "z":        logit_model.tvalues.round(3),
        "p_value":  logit_model.pvalues.round(4),
        "sig":      logit_model.pvalues.apply(lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else "")))
    })

    mo.vstack([
        mo.md("## 📊 Pooled Logit — PD Estimation"),
        mo.md(f"**Pseudo-R² (McFadden):** `{logit_model.prsquared:.4f}`  |  **AIC:** `{logit_model.aic:.1f}`  |  **Obs:** `{int(logit_model.nobs):,}`"),
        mo.ui.table(summary_df.reset_index().rename(columns={"index": "variable"}))
    ])
    return X_pd, df_model, logit_model, sm, summary_df, y_pd


@app.cell
def _(df_panel, mo, np):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    lgd_obs = df_panel["lgd"].dropna()

    fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
    ax.hist(lgd_obs, bins=40, color="#a12c7b", alpha=0.8, edgecolor="white", linewidth=0.5)
    ax.axvline(lgd_obs.mean(), color="#01696f", linewidth=2,
               linestyle="--", label=f"Mean LGD = {lgd_obs.mean():.3f}")
    ax.set_title("LGD Distribution (defaulted exposures)", fontweight="bold")
    ax.set_xlabel("Loss Given Default")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    mo.mpl.interactive(fig)
    return ax, fig, lgd_obs, matplotlib, plt, sns


@app.cell
def _(mo):
    mo.md("""
    ---
    ## 🗓️ Next Steps
    - [ ] Source real SBS Ecuador microdatos de cartera
    - [ ] Cox PH model for time-to-default (survival analysis)
    - [ ] Beta regression for LGD with zero-one inflation
    - [ ] Through-the-cycle vs point-in-time PD adjustment
    """)
    return


if __name__ == "__main__":
    app.run()
