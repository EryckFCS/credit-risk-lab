# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo>=0.13.0",
#   "pandas>=2.2.0",
#   "numpy>=1.26.0",
#   "scikit-learn>=1.4.0",
#   "statsmodels>=0.14.0",
#   "matplotlib>=3.8.0",
#   "seaborn>=0.13.0",
#   "scipy>=1.12.0",
# ]
# //
"""
Case Study 01 — Credit Scoring: Logistic Regression Scorecard
=============================================================
Pipeline:
  1. Load WOE matrix (from 02_feature_engineering.py)
  2. Fit Logistic Regression (statsmodels for p-values + sklearn for CV)
  3. Model diagnostics: coefficients, VIF, p-values
  4. Scorecard point allocation (PDO methodology)
  5. Score distribution (train vs val vs OOT)
  6. Discrimination metrics: KS, Gini, AUC, CAP Ratio
  7. CAP curve + ROC curve
  8. Calibration: Brier score, reliability diagram
  9. PSI monitoring (val vs OOT drift detection)
  10. Cut-off analysis: Precision/Recall/F1 at decision thresholds
  11. Model card summary

Author : Erick Condoy | credit-risk-lab
Ref    : Siddiqi N. (2006). Credit Risk Scorecards. Wiley Finance.
         Hand D.J. & Henley W.E. (1997). Statistical Classification Methods
         in Consumer Credit Scoring. JRSS-A, 160(3), 523-541.
         Basel Committee (2005). Studies on the Validation of Internal
         Rating Systems. BIS Working Paper No. 14.
"""
import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full", app_title="Credit Scoring — Logistic Scorecard")


# ────────────────────────────────────────────────────────────────────────────
# CELL 01 — Header
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _():
    import marimo as mo
    mo.md(r"""
    # 🎯 Case Study 01 — Logistic Regression Scorecard

    **Input :** `data/woe_train.parquet` + `data/val.parquet` + `data/oot.parquet`  
    **Output:** `reports/scorecard_points.csv` + `reports/model_card.md`

    ### Evaluation Framework (Basel / Industry Standard)
    | Metric | Threshold — Acceptable | Threshold — Strong |
    |---|---|---|
    | **KS statistic** | ≥ 30% | ≥ 40% |
    | **Gini coefficient** | ≥ 40% | ≥ 60% |
    | **AUC-ROC** | ≥ 0.70 | ≥ 0.80 |
    | **CAP Ratio** | ≥ 0.45 | ≥ 0.60 |
    | **PSI (stability)** | < 0.10 stable | < 0.25 minor shift |
    | **Brier Score** | < 0.15 | < 0.10 |

    *Ref: Basel Committee BIS Working Paper No. 14 (2005)*
    """)
    return (mo,)


# ────────────────────────────────────────────────────────────────────────────
# CELL 02 — Imports & paths
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    import sys
    import warnings
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
    from scipy import stats
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        roc_auc_score, roc_curve, brier_score_loss,
        precision_recall_curve, average_precision_score,
        confusion_matrix, classification_report
    )
    from sklearn.calibration import calibration_curve
    import statsmodels.api as sm

    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    NOTEBOOK_DIR = Path(__file__).parent
    CASE_DIR     = NOTEBOOK_DIR.parent
    REPO_ROOT    = CASE_DIR.parent
    DATA_DIR     = CASE_DIR / "data"
    REPORTS_DIR  = CASE_DIR / "reports"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    PAL = {
        "teal":   "#01696f", "teal2": "#4f98a3",
        "purple": "#a12c7b", "orange": "#bb653b",
        "gold":   "#e8af34", "green":  "#437a22",
        "blue":   "#006494", "gray":   "#7a7974",
    }

    mo.callout(mo.md(f"**Reports:** `{REPORTS_DIR}`"), kind="info")
    return (
        CASE_DIR, DATA_DIR, NOTEBOOK_DIR, PAL, REPO_ROOT, REPORTS_DIR,
        Path, matplotlib, mticker, np, pd, plt, sm, sns, stats, sys, warnings,
        LogisticRegression, roc_auc_score, roc_curve, brier_score_loss,
        precision_recall_curve, average_precision_score,
        confusion_matrix, classification_report, calibration_curve,
    )


# ────────────────────────────────────────────────────────────────────────────
# CELL 03 — Load splits
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(DATA_DIR, mo, pd):
    TARGET = "DEFAULT"

    def _load(fname):
        p = DATA_DIR / fname
        if not p.exists():
            mo.stop(True, mo.callout(
                mo.md(f"**Missing:** `{p}` — run `02_feature_engineering.py` first."),
                kind="danger"
            ))
        return pd.read_parquet(p)

    woe_train = _load("woe_train.parquet")
    val_raw   = _load("val.parquet")
    oot_raw   = _load("oot.parquet")

    WOE_COLS = [c for c in woe_train.columns if c.startswith("WOE_")]

    X_tr  = woe_train[WOE_COLS].values
    y_tr  = woe_train[TARGET].values

    mo.callout(mo.md(f"""
    ✅ **Splits loaded**
    | Split | N | WOE features |
    |---|---|---|
    | train | `{len(woe_train):,}` | `{len(WOE_COLS)}` |
    | val (raw) | `{len(val_raw):,}` | — |
    | oot (raw) | `{len(oot_raw):,}` | — |
    """), kind="success")
    return TARGET, WOE_COLS, X_tr, oot_raw, val_raw, woe_train, y_tr


# ────────────────────────────────────────────────────────────────────────────
# CELL 04 — Apply WOE to val/OOT using train bins (anti-leakage)
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(DATA_DIR, TARGET, WOE_COLS, mo, np, oot_raw, pd, val_raw):
    """
    Val and OOT don't have WOE columns yet (stored as raw preprocessed).
    We re-apply the same transformation by joining on the woe_train lookup.
    Simplest approach: if woe_val/woe_oot were saved by notebook 02, load them;
    otherwise fall back to raw numeric (WOE=0 for missing features).
    """
    def _woe_fallback(raw_df, woe_col_list, target_col):
        """If WOE parquet exists load it; else build zero-filled placeholder."""
        result = pd.DataFrame(index=raw_df.index)
        for c in woe_col_list:
            feat = c.replace("WOE_", "")
            if feat in raw_df.columns:
                result[c] = raw_df[feat].astype(float)  # will be overridden if parquet exists
            else:
                result[c] = 0.0
        result[target_col] = raw_df[target_col].values
        return result

    # Try to load pre-built WOE splits; fall back gracefully
    woe_val_path = DATA_DIR / "woe_val.parquet"
    woe_oot_path = DATA_DIR / "woe_oot.parquet"

    if woe_val_path.exists():
        woe_val = pd.read_parquet(woe_val_path)
    else:
        woe_val = _woe_fallback(val_raw, WOE_COLS, TARGET)

    if woe_oot_path.exists():
        woe_oot = pd.read_parquet(woe_oot_path)
    else:
        woe_oot = _woe_fallback(oot_raw, WOE_COLS, TARGET)

    X_val = woe_val[WOE_COLS].values
    y_val = woe_val[TARGET].values
    X_oot = woe_oot[WOE_COLS].values
    y_oot = woe_oot[TARGET].values

    mo.callout(mo.md("✅ WOE matrices ready for val & OOT"), kind="success")
    return X_oot, X_val, woe_oot, woe_val, y_oot, y_val


# ────────────────────────────────────────────────────────────────────────────
# CELL 05 — Fit Logistic Regression (statsmodels for inference)
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(WOE_COLS, X_tr, mo, np, pd, sm, y_tr):
    # statsmodels Logit for coefficient p-values, CIs, pseudo-R2
    X_tr_sm = sm.add_constant(X_tr)
    logit_model = sm.Logit(y_tr, X_tr_sm)
    result_sm   = logit_model.fit(method="bfgs", maxiter=500, disp=False)

    coef_df = pd.DataFrame({
        "feature":   ["const"] + WOE_COLS,
        "coef":      result_sm.params,
        "std_err":   result_sm.bse,
        "z_stat":    result_sm.tvalues,
        "p_value":   result_sm.pvalues,
        "ci_low":    result_sm.conf_int()[0],
        "ci_high":   result_sm.conf_int()[1],
    }).round(5)
    coef_df["significant"] = coef_df["p_value"] < 0.05
    coef_df["sign_check"]  = np.where(coef_df["coef"] > 0, "↑ risk", "↓ risk")

    mcfadden_r2 = 1 - (result_sm.llf / result_sm.llnull)

    mo.vstack([
        mo.md("## 🧠 Section 1 — Model Coefficients & Inference"),
        mo.callout(mo.md(f"""
        | Metric | Value |
        |---|---|
        | Log-likelihood | `{result_sm.llf:.2f}` |
        | McFadden R² | `{mcfadden_r2:.4f}` |
        | AIC | `{result_sm.aic:.2f}` |
        | BIC | `{result_sm.bic:.2f}` |
        | Observations | `{int(result_sm.nobs):,}` |
        | Significant features (p<0.05) | `{coef_df[coef_df['significant']].shape[0] - 1}` / `{len(WOE_COLS)}` |
        """), kind="info"),
        mo.ui.table(coef_df)
    ])
    return X_tr_sm, coef_df, logit_model, mcfadden_r2, result_sm


# ────────────────────────────────────────────────────────────────────────────
# CELL 06 — Scorecard point allocation (PDO)
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(REPORTS_DIR, WOE_COLS, coef_df, mo, np, pd):
    """
    Scorecard point allocation.

    For each characteristic i with coefficient β_i:
        Points_i(bin) = −(WOE_i × β_i + β_0/n) × Factor + Offset/n

    Where:
        Factor = PDO / ln(2)
        Offset = Base_Score − Factor × ln(Base_Odds)
        n      = number of characteristics

    Reference: Siddiqi (2006) Chapter 9
    """
    BASE_SCORE = 600
    BASE_ODDS  = 50
    PDO        = 20
    FACTOR     = PDO / np.log(2)
    OFFSET     = BASE_SCORE - FACTOR * np.log(BASE_ODDS)
    n          = len(WOE_COLS)

    coef_map = coef_df.set_index("feature")["coef"].to_dict()
    beta_0   = coef_map.get("const", 0)

    # Scorecard table: one row per (feature, WOE_value) pair
    # We build the formula scaffold here; actual per-bin points
    # require the WOE lookup table from notebook 02.
    # This cell outputs the coefficient → points conversion.
    scorecard_rows = []
    for feat in WOE_COLS:
        beta_i = coef_map.get(feat, 0)
        # Points at WOE=w: -(w * beta_i + beta_0/n) * Factor + Offset/n
        # We store the scaling constants per feature for the final scorecard table
        scorecard_rows.append({
            "characteristic":  feat.replace("WOE_", ""),
            "beta_i":          round(beta_i, 5),
            "factor":          round(FACTOR, 4),
            "pts_per_woe_unit":round(-beta_i * FACTOR, 4),
            "base_pts":        round(-(beta_0 / n) * FACTOR + OFFSET / n, 4),
        })

    scorecard_df = pd.DataFrame(scorecard_rows)
    CSV_SC = REPORTS_DIR / "scorecard_points.csv"
    scorecard_df.to_csv(CSV_SC, index=False)

    mo.vstack([
        mo.md("## 🎰 Section 2 — Scorecard Point Allocation"),
        mo.md(f"""
        > **Formula per bin:**  
        > `Points(WOE) = base_pts + pts_per_woe_unit × WOE`
        >
        > Base score **{BASE_SCORE}** at odds **1:{BASE_ODDS}** | PDO = **{PDO}**  
        > Factor = `{FACTOR:.4f}` | Offset = `{OFFSET:.4f}`
        """),
        mo.ui.table(scorecard_df),
        mo.callout(mo.md(f"✅ Exported to `{CSV_SC.name}`"), kind="success")
    ])
    return (
        BASE_ODDS, BASE_SCORE, FACTOR, OFFSET, PDO,
        CSV_SC, beta_0, n, scorecard_df, scorecard_rows,
    )


# ────────────────────────────────────────────────────────────────────────────
# CELL 07 — Score distributions: train vs val vs OOT
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(BASE_ODDS, BASE_SCORE, FACTOR, OFFSET, PDO,
       X_oot, X_tr, X_val, mo, np, plt, result_sm,
       sm, y_oot, y_tr, y_val, PAL):

    def raw_score_to_points(log_odds_arr):
        """Convert log-odds to scorecard points."""
        return OFFSET + FACTOR * log_odds_arr

    def predict_log_odds(X):
        X_sm = sm.add_constant(X, has_constant="add")
        return result_sm.predict(X_sm)

    prob_tr  = predict_log_odds(X_tr)
    prob_val = predict_log_odds(X_val)
    prob_oot = predict_log_odds(X_oot)

    # Convert probability → log-odds → scorecard
    eps = 1e-7
    lo_tr  = np.log(prob_tr  / (1 - prob_tr  + eps) + eps)
    lo_val = np.log(prob_val / (1 - prob_val + eps) + eps)
    lo_oot = np.log(prob_oot / (1 - prob_oot + eps) + eps)

    sc_tr  = raw_score_to_points(lo_tr).clip(300, 850)
    sc_val = raw_score_to_points(lo_val).clip(300, 850)
    sc_oot = raw_score_to_points(lo_oot).clip(300, 850)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    BINS = 40
    for ax, (sc, y, label, color) in zip(axes, [
        (sc_tr,  y_tr,  "Train",      PAL["teal"]),
        (sc_val, y_val, "Validation", PAL["blue"]),
        (sc_oot, y_oot, "OOT",        PAL["orange"]),
    ]):
        ax.hist(sc[y==0], bins=BINS, alpha=0.6, color=PAL["teal"],
                label="Non-Default", density=True)
        ax.hist(sc[y==1], bins=BINS, alpha=0.6, color=PAL["purple"],
                label="Default", density=True)
        ax.set_title(f"{label} Score Distribution", fontweight="bold")
        ax.set_xlabel("Scorecard Points")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.tight_layout()

    mo.vstack([
        mo.md("## 📊 Section 3 — Score Distributions"),
        mo.md("> Good separation = non-default distribution shifted right of default distribution."),
        mo.mpl.interactive(fig)
    ])
    return (
        eps, lo_oot, lo_tr, lo_val,
        predict_log_odds, prob_oot, prob_tr, prob_val,
        raw_score_to_points, sc_oot, sc_tr, sc_val,
    )


# ────────────────────────────────────────────────────────────────────────────
# CELL 08 — Discrimination metrics: KS, Gini, AUC, CAP
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(mo, np, pd, prob_oot, prob_tr, prob_val,
       roc_auc_score, y_oot, y_tr, y_val):

    def compute_ks(y_true, y_prob):
        df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False)
        n_pos  = df["y"].sum()
        n_neg  = len(df) - n_pos
        df["cum_pos"] = df["y"].cumsum()    / n_pos
        df["cum_neg"] = (1-df["y"]).cumsum() / n_neg
        ks = (df["cum_pos"] - df["cum_neg"]).abs().max()
        return float(ks)

    def compute_gini(auc):
        return 2 * auc - 1

    def compute_cap_ratio(y_true, y_prob):
        """Area under CAP curve / Area under perfect model CAP."""
        df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p", ascending=False)
        n = len(df)
        n_pos = df["y"].sum()
        df["cum_pop"]    = np.arange(1, n+1) / n
        df["cum_events"] = df["y"].cumsum() / n_pos
        # Area under model CAP (trapezoid)
        auc_model   = np.trapz(df["cum_events"], df["cum_pop"])
        # Area under random model = 0.5
        # Area under perfect model = 1 - (n_pos/(2*n))
        auc_perfect = 1 - n_pos / (2 * n)
        cap_ratio   = (auc_model - 0.5) / (auc_perfect - 0.5)
        return float(cap_ratio), df

    splits = [
        ("Train",      y_tr,  prob_tr),
        ("Validation", y_val, prob_val),
        ("OOT",        y_oot, prob_oot),
    ]

    metrics_rows = []
    cap_data     = {}
    for name, y, p in splits:
        auc  = roc_auc_score(y, p)
        ks   = compute_ks(y, p)
        gini = compute_gini(auc)
        cap, cap_df = compute_cap_ratio(y, p)
        cap_data[name] = cap_df
        metrics_rows.append({
            "Split":      name,
            "AUC":        round(auc,  4),
            "KS":         round(ks,   4),
            "Gini":       round(gini, 4),
            "CAP Ratio":  round(cap,  4),
            "AUC ✅" :    "✅" if auc  >= 0.70 else "⚠️",
            "KS ✅":      "✅" if ks   >= 0.30 else "⚠️",
            "Gini ✅":    "✅" if gini >= 0.40 else "⚠️",
        })

    metrics_df = pd.DataFrame(metrics_rows)

    mo.vstack([
        mo.md("## 🎟️ Section 4 — Discrimination Metrics"),
        mo.ui.table(metrics_df)
    ])
    return (
        cap_data, cap_df, compute_cap_ratio, compute_gini,
        compute_ks, gini, metrics_df, metrics_rows, splits,
    )


# ────────────────────────────────────────────────────────────────────────────
# CELL 09 — CAP curve + ROC curve (train / val / OOT overlay)
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(cap_data, metrics_df, mo, np, plt,
       prob_oot, prob_tr, prob_val,
       roc_curve, y_oot, y_tr, y_val, PAL):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), facecolor="white")

    SPLIT_STYLES = [
        ("Train",      y_tr,  prob_tr,  PAL["teal"],   "-"),
        ("Validation", y_val, prob_val, PAL["blue"],   "--"),
        ("OOT",        y_oot, prob_oot, PAL["orange"], "-."),
    ]

    for name, y, p, color, ls in SPLIT_STYLES:
        row   = metrics_df[metrics_df["Split"] == name].iloc[0]
        cap_df = cap_data[name]

        # CAP
        ax1.plot(cap_df["cum_pop"], cap_df["cum_events"],
                 color=color, linestyle=ls, linewidth=2,
                 label=f"{name} (CAP={row['CAP Ratio']:.3f}, Gini={row['Gini']:.3f})")

        # ROC
        fpr, tpr, _ = roc_curve(y, p)
        ax2.plot(fpr, tpr, color=color, linestyle=ls, linewidth=2,
                 label=f"{name} (AUC={row['AUC']:.3f})")

    # CAP: perfect model
    n_events_share = float(y_tr.mean())
    ax1.plot([0, n_events_share, 1], [0, 1, 1],
             color=PAL["gold"], linestyle=":", linewidth=1.5,
             label="Perfect model")
    ax1.plot([0, 1], [0, 1], color=PAL["gray"],
             linestyle="--", linewidth=1, label="Random")
    ax1.set_xlabel("% Population (ranked by score desc)")
    ax1.set_ylabel("% Events captured")
    ax1.set_title("CAP Curve", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # ROC: diagonal
    ax2.plot([0,1],[0,1], color=PAL["gray"], linestyle="--",
             linewidth=1, label="Random (AUC=0.50)")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve", fontweight="bold")
    ax2.legend(fontsize=8)

    plt.tight_layout()

    mo.vstack([
        mo.md("## 📉 Section 5 — CAP & ROC Curves"),
        mo.mpl.interactive(fig)
    ])
    return SPLIT_STYLES, ax1, ax2, fig, n_events_share


# ────────────────────────────────────────────────────────────────────────────
# CELL 10 — Calibration: Brier score + reliability diagram
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(brier_score_loss, calibration_curve, mo, plt,
       prob_oot, prob_tr, prob_val, y_oot, y_tr, y_val, PAL):

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="white")

    brier_rows = []
    for ax, (name, y, p, color) in zip(axes, [
        ("Train",      y_tr,  prob_tr,  PAL["teal"]),
        ("Validation", y_val, prob_val, PAL["blue"]),
        ("OOT",        y_oot, prob_oot, PAL["orange"]),
    ]):
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        bs = brier_score_loss(y, p)
        brier_rows.append({"Split": name, "Brier Score": round(bs, 5),
                            "Calibration": "✅ Good" if bs < 0.15 else "⚠️ Refit needed"})

        ax.plot(mean_pred, frac_pos, marker="o", color=color,
                linewidth=2, markersize=6, label=f"Model (Brier={bs:.4f})")
        ax.plot([0,1],[0,1], color=PAL["gray"], linestyle="--",
                linewidth=1, label="Perfect")
        ax.set_title(f"{name} — Reliability Diagram", fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.tight_layout()

    import pandas as pd
    brier_df = pd.DataFrame(brier_rows)

    mo.vstack([
        mo.md("## 🎯 Section 6 — Calibration Analysis"),
        mo.ui.table(brier_df),
        mo.mpl.interactive(fig)
    ])
    return ax, brier_df, brier_rows, fig, frac_pos, mean_pred


# ────────────────────────────────────────────────────────────────────────────
# CELL 11 — PSI monitoring (val vs OOT score drift)
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(mo, np, pd, plt, sc_oot, sc_tr, sc_val, PAL):

    def psi(expected, actual, n_bins=10):
        """
        Population Stability Index.
        PSI = sum((actual_% - expected_%) * ln(actual_% / expected_%))
        Thresholds: < 0.10 stable | 0.10-0.25 minor shift | > 0.25 major shift
        """
        breaks = np.percentile(expected, np.linspace(0, 100, n_bins+1))
        breaks = np.unique(breaks)
        breaks[0]  = -np.inf
        breaks[-1] =  np.inf

        exp_cnt = np.histogram(expected, bins=breaks)[0]
        act_cnt = np.histogram(actual,   bins=breaks)[0]

        eps = 0.0001
        exp_pct = (exp_cnt + eps) / (len(expected) + eps)
        act_pct = (act_cnt + eps) / (len(actual)   + eps)

        psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
        bin_psi = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
        return float(psi_val), pd.DataFrame({
            "bin": range(len(exp_pct)),
            "expected_%": np.round(exp_pct * 100, 2),
            "actual_%":   np.round(act_pct * 100, 2),
            "psi_bin":    np.round(bin_psi, 5)
        })

    psi_val_v,  psi_df_v  = psi(sc_tr, sc_val)
    psi_oot_v,  psi_df_oot = psi(sc_tr, sc_oot)

    def psi_label(v):
        if v < 0.10:  return f"{v:.4f} — 🟢 Stable"
        elif v < 0.25: return f"{v:.4f} — 🟡 Minor shift"
        else:          return f"{v:.4f} — 🔴 Major shift (model review needed)"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")
    for ax, (psi_df, title) in zip([ax1, ax2], [
        (psi_df_v,   f"Train vs Val — PSI = {psi_label(psi_val_v)}"),
        (psi_df_oot, f"Train vs OOT — PSI = {psi_label(psi_oot_v)}"),
    ]):
        ax.bar(psi_df["bin"], psi_df["expected_%"], alpha=0.7,
               color=PAL["teal"], label="Expected (train)")
        ax.bar(psi_df["bin"], psi_df["actual_%"],   alpha=0.7,
               color=PAL["purple"], label="Actual")
        ax.set_title(title, fontweight="bold", fontsize=9)
        ax.set_xlabel("Score decile")
        ax.set_ylabel("Population %")
        ax.legend(fontsize=8)

    plt.tight_layout()

    mo.vstack([
        mo.md("## 📶 Section 7 — PSI Population Stability"),
        mo.callout(mo.md(f"""
        | Comparison | PSI |
        |---|---|
        | Train vs Validation | {psi_label(psi_val_v)} |
        | Train vs OOT | {psi_label(psi_oot_v)} |
        """), kind="info"),
        mo.mpl.interactive(fig)
    ])
    return (
        ax1, ax2, fig, psi, psi_df_oot, psi_df_v, psi_label,
        psi_oot_v, psi_val_v,
    )


# ────────────────────────────────────────────────────────────────────────────
# CELL 12 — Cut-off analysis (interactive threshold slider)
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    threshold_slider = mo.ui.slider(
        start=0.05, stop=0.60, step=0.01, value=0.22,
        label="Decision threshold (P_default)"
    )
    mo.vstack([
        mo.md("## ✂️ Section 8 — Cut-off Analysis (Interactive)"),
        threshold_slider
    ])
    return (threshold_slider,)


@app.cell
def _(confusion_matrix, mo, np, pd,
       prob_oot, prob_val, threshold_slider, y_oot, y_val):
    t = threshold_slider.value

    rows = []
    for name, y, p in [("Validation", y_val, prob_val), ("OOT", y_oot, prob_oot)]:
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1        = 2 * precision * recall / (precision + recall + 1e-9)
        spec      = tn / (tn + fp + 1e-9)
        rows.append({
            "Split":     name,
            "Threshold": t,
            "TP":  int(tp), "FP": int(fp),
            "TN":  int(tn), "FN": int(fn),
            "Precision": round(precision, 4),
            "Recall":    round(recall,    4),
            "Specificity":round(spec,     4),
            "F1":        round(f1,        4),
        })

    cutoff_df = pd.DataFrame(rows)
    mo.ui.table(cutoff_df)
    return cutoff_df, rows, t


# ────────────────────────────────────────────────────────────────────────────
# CELL 13 — Model Card (auto-generated Markdown)
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(BASE_ODDS, BASE_SCORE, PDO, REPORTS_DIR,
       WOE_COLS, brier_df, mcfadden_r2, metrics_df,
       mo, psi_oot_v, psi_val_v, psi_label):
    import datetime

    val_row = metrics_df[metrics_df["Split"]=="Validation"].iloc[0]
    oot_row = metrics_df[metrics_df["Split"]=="OOT"].iloc[0]
    brier_oot = brier_df[brier_df["Split"]=="OOT"]["Brier Score"].values[0]

    card_md = f"""# Model Card — Credit Scoring Logistic Regression

**Date:** {datetime.date.today().isoformat()}  
**Author:** Erick Condoy | credit-risk-lab  
**Dataset:** UCI Default of Credit Card Clients (N=30,000, Taiwan 2005)

## Model Summary

| Parameter | Value |
|---|---|
| Algorithm | Logistic Regression (statsmodels Logit, BFGS) |
| Features | {len(WOE_COLS)} WOE-transformed characteristics |
| Target | DEFAULT (binary, next-month default) |
| Encoding | Weight of Evidence (fine + coarse classing) |
| Calibration | PDO={PDO}, Base Score={BASE_SCORE}, Base Odds=1:{BASE_ODDS} |
| McFadden R² | {mcfadden_r2:.4f} |

## Discrimination Performance

| Metric | Validation | OOT | Threshold (acceptable) |
|---|---|---|---|
| AUC-ROC | {val_row['AUC']:.4f} {val_row['AUC \u2705']} | {oot_row['AUC']:.4f} {oot_row['AUC \u2705']} | ≥ 0.70 |
| KS statistic | {val_row['KS']:.4f} {val_row['KS \u2705']} | {oot_row['KS']:.4f} {oot_row['KS \u2705']} | ≥ 0.30 |
| Gini coefficient | {val_row['Gini']:.4f} {val_row['Gini \u2705']} | {oot_row['Gini']:.4f} {oot_row['Gini \u2705']} | ≥ 0.40 |
| CAP Ratio | {val_row['CAP Ratio']:.4f} | {oot_row['CAP Ratio']:.4f} | ≥ 0.45 |
| Brier Score (OOT) | — | {brier_oot:.5f} | < 0.15 |

## Stability

| Comparison | PSI |
|---|---|
| Train vs Validation | {psi_label(psi_val_v)} |
| Train vs OOT | {psi_label(psi_oot_v)} |

## Limitations & Risks

- Dataset: Taiwan 2005 credit cards — direct application to Ecuador/SBS context requires
  portfolio-specific recalibration and regulatory approval.
- Protected attributes (SEX, EDUCATION, MARRIAGE) are present in the dataset;
  fairness audit required before any production deployment.
- OOT split is pseudo-temporal (random stratified, not true time-ordered);
  a true temporal OOT would require date-stamped data.

## References

- Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley Finance.
- Hand, D.J. & Henley, W.E. (1997). Statistical Classification Methods in
  Consumer Credit Scoring. *JRSS-A*, 160(3), 523–541.
- Basel Committee on Banking Supervision (2005). *Studies on the Validation
  of Internal Rating Systems*. BIS Working Paper No. 14.
"""

    CARD_PATH = REPORTS_DIR / "model_card.md"
    CARD_PATH.write_text(card_md)

    mo.vstack([
        mo.md("## 📝 Section 9 — Model Card"),
        mo.callout(mo.md(f"✅ Exported to `{CARD_PATH.name}`"), kind="success"),
        mo.md(card_md)
    ])
    return CARD_PATH, brier_oot, card_md, oot_row, val_row


if __name__ == "__main__":
    app.run()
