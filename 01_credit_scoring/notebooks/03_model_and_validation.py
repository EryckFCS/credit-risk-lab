# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "pyarrow"]
# ///
"""
Case Study 01 — Credit Scoring | Model Development & Validation
DAG: processed_woe.parquet → train/test split → logistic regression
     → scorecard (PDO) → validation metrics (KS, Gini, AUC, CAP, PSI) → figures
Author: Erick Condoy | credit-risk-lab
"""
import marimo as mo

app = mo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    import marimo as mo
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore")
    matplotlib.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    PALETTE = ["#01696f", "#964219", "#006494", "#437a22", "#7a39bb", "#d19900"]
    return (
        LogisticRegression, PALETTE, Path, StandardScaler, StratifiedKFold,
        cross_val_score, mo, np, pd, plt, roc_auc_score, roc_curve,
        sys, warnings, matplotlib,
    )


@app.cell
def _(Path, mo, sys):
    REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
    DATA_DIR   = REPO_ROOT / "01_credit_scoring" / "data"
    REPORT_DIR = REPO_ROOT / "01_credit_scoring" / "reports"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from utils.metrics import ks_statistic, gini_coefficient, psi, cap_ratio
    from utils.plotting import plot_cap_curve, plot_roc_curve, plot_score_distribution
    mo.md(f"✅ Repo root resolved | utils.metrics + utils.plotting loaded")


@app.cell
def _(DATA_DIR, mo, pd):
    woe_path = DATA_DIR / "processed_woe.parquet"
    if not woe_path.exists():
        mo.stop(True, mo.md("❌ Run notebook `02_feature_engineering.py` first"))
    df_woe = pd.read_parquet(woe_path)
    FEATURE_COLS = [c for c in df_woe.columns if c.endswith("_WOE")]
    mo.md(f"✅ WOE data: `{len(df_woe):,} rows` | Features: `{len(FEATURE_COLS)}`")


@app.cell
def _(FEATURE_COLS, LogisticRegression, StandardScaler, StratifiedKFold,
      cross_val_score, df_woe, mo, np, pd):
    from sklearn.model_selection import train_test_split

    X = df_woe[FEATURE_COLS].fillna(0).values
    y = df_woe["DEFAULT"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, C=0.5)
    clf.fit(X_train_s, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(clf, X_train_s, y_train, cv=cv, scoring="roc_auc")

    mo.md(f"""
    ### Model Training Complete
    | Metric | Value |
    |--------|-------|
    | Algorithm | Logistic Regression (`class_weight='balanced'`) |
    | Training set | {len(y_train):,} obs ({y_train.mean():.2%} default) |
    | Test set | {len(y_test):,} obs ({y_test.mean():.2%} default) |
    | CV AUC (5-fold) | **{cv_auc.mean():.4f} ± {cv_auc.std():.4f}** |
    """)


@app.cell
def _(cap_ratio, clf, gini_coefficient, ks_statistic, mo, np,
      pd, psi, roc_auc_score, scaler, X_test_s, X_train_s, y_test, y_train):
    p_train = clf.predict_proba(X_train_s)[:, 1]
    p_test  = clf.predict_proba(X_test_s)[:, 1]

    metrics = {
        "KS":        (ks_statistic(y_test, p_test),       0.30, ">"),
        "Gini":      (gini_coefficient(y_test, p_test),   0.40, ">"),
        "AUC":       (roc_auc_score(y_test, p_test),      0.70, ">"),
        "CAP Ratio": (cap_ratio(y_test, p_test),          0.60, ">"),
        "PSI":       (psi(p_train, p_test),               0.10, "<"),
    }

    rows = []
    for name, (val, thr, direction) in metrics.items():
        passed = (val > thr) if direction == ">" else (val < thr)
        rows.append({
            "Metric": name,
            "Value": round(val, 4),
            "Threshold": f"{direction} {thr}",
            "Status": "✅ PASS" if passed else "❌ FAIL",
        })
    val_df = pd.DataFrame(rows)

    all_pass = all("✅" in r["Status"] for r in rows)
    mo.vstack([
        mo.md(f"### Validation Report — {'All metrics PASSED ✅' if all_pass else 'Some metrics FAILED ❌'}"),
        mo.ui.table(val_df, selection=None, label="Model Validation"),
    ])


@app.cell
def _(PALETTE, REPORT_DIR, clf, gini_coefficient, mo, np, pd,
      plot_cap_curve, plot_roc_curve, plot_score_distribution,
      p_test, p_train, plt, roc_auc_score, y_test, y_train):
    # Three figures: CAP, ROC, Score distribution
    fig_cap = plot_cap_curve(y_test, p_test, save_path=REPORT_DIR / "fig08_cap_curve.png")
    fig_roc = plot_roc_curve(y_test, p_test, save_path=REPORT_DIR / "fig09_roc_curve.png")
    fig_sco = plot_score_distribution(
        p_train, y_train, p_test, y_test,
        save_path=REPORT_DIR / "fig10_score_distribution.png"
    )
    mo.hstack([fig_cap, fig_roc])


@app.cell
def _(mo, REPORT_DIR, p_test, p_train):
    # Scorecard scaling: PDO methodology
    # Score = Offset + Factor * log(p_good / p_bad)
    import numpy as np

    BASE_SCORE = 600
    PDO = 20          # Points to Double Odds
    BASE_ODDS = 50    # Good:Bad ratio at base score

    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)

    # Convert probabilities to score
    def prob_to_score(p_bad):
        p_good = np.clip(1 - p_bad, 1e-7, 1 - 1e-7)
        p_bad_c = np.clip(p_bad, 1e-7, 1 - 1e-7)
        return offset + factor * np.log(p_good / p_bad_c)

    scores_test = prob_to_score(p_test)

    score_stats = {
        "min":    scores_test.min(),
        "p25":    np.percentile(scores_test, 25),
        "median": np.median(scores_test),
        "mean":   scores_test.mean(),
        "p75":    np.percentile(scores_test, 75),
        "max":    scores_test.max(),
    }

    mo.md(f"""
    ### Scorecard — PDO Scaling
    | Parameter | Value |
    |-----------|-------|
    | Base score | {BASE_SCORE} |
    | PDO | {PDO} (points to double odds) |
    | Base odds | {BASE_ODDS} (good:bad) |
    | Factor | {factor:.4f} |
    | Offset | {offset:.4f} |

    **Score distribution (test set):**
    | Min | P25 | Median | Mean | P75 | Max |
    |-----|-----|--------|------|-----|-----|
    | {score_stats['min']:.0f} | {score_stats['p25']:.0f} | {score_stats['median']:.0f} | {score_stats['mean']:.0f} | {score_stats['p75']:.0f} | {score_stats['max']:.0f} |

    *Higher score = lower risk (conventional credit scoring direction)*
    """)


if __name__ == "__main__":
    app.run()
