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
#   "ucimlrepo>=0.0.7",
# ]
# //
import marimo

__generated_with = "0.13.0"
app = marimo.App(
    width="full",
    app_title="Credit Scoring — EDA & Scorecard",
)


@app.cell
def _():
    import marimo as mo
    mo.md("""
    # 📊 Case Study 01 — Credit Scoring: EDA & Scorecard

    **Objective:** Build a production-grade binary classifier to predict default probability
    on a retail credit portfolio, then calibrate it into an industry-standard scorecard
    using the PDO (Points-to-Double-Odds) methodology.

    **Dataset:** UCI Default of Credit Card Clients — 30,000 observations, Taiwan 2005.  
    **Reference:** Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley.
    """)
    return (mo,)


@app.cell
def _(mo):
    import sys
    from pathlib import Path

    # Soberanía de rutas: ancla a la raíz del repositorio
    NOTEBOOK_DIR = Path(__file__).parent
    CASE_DIR = NOTEBOOK_DIR.parent
    REPO_ROOT = CASE_DIR.parent
    DATA_DIR = CASE_DIR / "data"
    REPORTS_DIR = CASE_DIR / "reports"

    # Auto-healing: añade utils al path sin hardcoding de SO
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # Crear directorios si no existen
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mo.md(f"""
    **Paths resolved:**
    - Repo root: `{REPO_ROOT}`
    - Data dir:  `{DATA_DIR}`
    - Reports:   `{REPORTS_DIR}`
    """)
    return CASE_DIR, DATA_DIR, NOTEBOOK_DIR, REPO_ROOT, REPORTS_DIR, Path, sys


@app.cell
def _(mo):
    import pandas as pd
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")
    mo.md("### 📦 Dependencies loaded")
    return np, pd, warnings


@app.cell
def _(DATA_DIR, mo, pd):
    from pathlib import Path

    PARQUET_CACHE = DATA_DIR / "uci_credit_default.parquet"

    def load_data() -> pd.DataFrame:
        """Load UCI dataset. Uses local Parquet cache if available."""
        if PARQUET_CACHE.exists():
            df = pd.read_parquet(PARQUET_CACHE)
            source = "Parquet cache"
        else:
            try:
                from ucimlrepo import fetch_ucirepo
                dataset = fetch_ucirepo(id=350)
                df = pd.concat([
                    dataset.data.features,
                    dataset.data.targets.rename(columns={"Y": "DEFAULT"})
                ], axis=1)
                df.columns = [c.upper() for c in df.columns]
                df.to_parquet(PARQUET_CACHE, index=False)
                source = "UCI API (cached locally)"
            except Exception as e:
                return None, str(e)
        return df, source

    df_raw, _source = load_data()

    if df_raw is None:
        mo.callout(
            mo.md(f"**Data load failed:** {_source}. Run `data/download.sh` first."),
            kind="danger"
        )
    else:
        mo.md(f"""
        ✅ **Dataset loaded** — source: *{_source}*

        - Shape: `{df_raw.shape}`
        - Default rate: `{df_raw['DEFAULT'].mean():.2%}`
        """)
    return PARQUET_CACHE, df_raw, load_data


@app.cell
def _(df_raw, mo, pd):
    if df_raw is None:
        mo.stop(True, mo.md("**Stop:** no data available."))

    # ── Data quality introspection ──────────────────────────────────────────
    quality = pd.DataFrame({
        "dtype":        df_raw.dtypes,
        "nulls":        df_raw.isnull().sum(),
        "null_pct":     (df_raw.isnull().mean() * 100).round(2),
        "unique":       df_raw.nunique(),
        "mean":         df_raw.select_dtypes("number").mean().round(4),
        "std":          df_raw.select_dtypes("number").std().round(4),
    })

    mo.vstack([
        mo.md("## 🔍 Data Quality Report"),
        mo.ui.table(quality.reset_index().rename(columns={"index": "column"}))
    ])
    return (quality,)


@app.cell
def _(df_raw, mo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    # ── Target distribution ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="white")

    counts = df_raw["DEFAULT"].value_counts()
    axes[0].bar(["Non-Default", "Default"], counts.values,
                color=["#4f98a3", "#a12c7b"], edgecolor="white", linewidth=1.2)
    axes[0].set_title("Target Distribution", fontweight="bold")
    axes[0].set_ylabel("Count")
    for bar, val in zip(axes[0].patches, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
                     f"{val:,}", ha="center", fontsize=10)

    # Age distribution by default
    df_raw.groupby("DEFAULT")["AGE"].plot.hist(
        ax=axes[1], bins=30, alpha=0.6,
        color=["#4f98a3", "#a12c7b"], label=["Non-Default", "Default"]
    )
    axes[1].set_title("Age Distribution by Default", fontweight="bold")
    axes[1].set_xlabel("Age")
    axes[1].legend()

    plt.tight_layout()
    mo.mpl.interactive(fig)
    return axes, fig, matplotlib, plt, sns


@app.cell
def _(df_raw, mo, np, pd):
    # ── Correlation matrix (numeric features only) ──────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    numeric_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    corr_matrix = df_raw[numeric_cols].corr()

    fig_corr, ax_corr = plt.subplots(figsize=(14, 10), facecolor="white")
    import seaborn as sns
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, ax=ax_corr,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=False, linewidths=0.3,
        cbar_kws={"shrink": 0.8}
    )
    ax_corr.set_title("Correlation Matrix — Numeric Features", fontweight="bold", pad=12)
    plt.tight_layout()
    mo.mpl.interactive(fig_corr)
    return ax_corr, corr_matrix, fig_corr, numeric_cols


@app.cell
def _(df_raw, mo, np, pd):
    # ── WOE / IV analysis (top features) ────────────────────────────────────
    import sys
    from pathlib import Path
    _root = Path(__file__).parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    try:
        from utils.preprocessing import compute_iv_table
        candidate_features = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2",
                               "BILL_AMT1", "PAY_AMT1"]
        iv_results = []
        for feat in candidate_features:
            try:
                tbl = compute_iv_table(df_raw, feat, "DEFAULT", bins=10)
                iv_results.append({"feature": feat, "IV": tbl["IV"].sum().round(4)})
            except Exception:
                iv_results.append({"feature": feat, "IV": None})

        iv_df = pd.DataFrame(iv_results).sort_values("IV", ascending=False)
        mo.vstack([
            mo.md("## 📊 Information Value (IV) — Feature Predictive Power"),
            mo.md("""
            | IV Range | Predictive Power |
            |---|---|
            | < 0.02 | Useless |
            | 0.02 – 0.10 | Weak |
            | 0.10 – 0.30 | Medium |
            | 0.30 – 0.50 | Strong |
            | > 0.50 | Suspicious (data leakage?) |
            """),
            mo.ui.table(iv_df)
        ])
    except ImportError as e:
        mo.callout(mo.md(f"utils not available: `{e}`"), kind="warn")
    return


@app.cell
def _(df_raw, mo, np, pd):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    TARGET = "DEFAULT"
    FEATURES = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3",
                "BILL_AMT1", "BILL_AMT2", "PAY_AMT1", "PAY_AMT2"]

    X = df_raw[FEATURES].copy()
    y = df_raw[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logit",  LogisticRegression(max_iter=500, C=1.0, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    mo.callout(
        mo.md(f"## 🤖 Logistic Regression Baseline\n\n**AUC-ROC:** `{auc:.4f}`  \nTrain: `{len(X_train):,}` obs — Test: `{len(X_test):,}` obs"),
        kind="success"
    )
    return (
        FEATURES, TARGET, X, X_test, X_train,
        Pipeline, StandardScaler, LogisticRegression,
        auc, pipe, roc_auc_score, train_test_split, y, y_prob,
        y_test, y_train,
    )


@app.cell
def _(auc, mo, np, pipe, y_prob, y_test):
    import sys
    from pathlib import Path
    _root = Path(__file__).parent.parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from utils.metrics import ks_statistic, gini_coefficient, cap_ratio
        ks  = ks_statistic(y_test.values, y_prob)
        gini = gini_coefficient(y_test.values, y_prob)
        cap  = cap_ratio(y_test.values, y_prob)

        metrics_summary = {
            "AUC-ROC": f"{auc:.4f}  {'\u2705' if auc > 0.70 else '\u26a0\ufe0f'}",
            "KS Statistic": f"{ks:.4f}  {'\u2705' if ks > 0.30 else '\u26a0\ufe0f'}",
            "Gini Coefficient": f"{gini:.4f}  {'\u2705' if gini > 0.40 else '\u26a0\ufe0f'}",
            "CAP Ratio": f"{cap:.4f}  {'\u2705' if cap > 0.60 else '\u26a0\ufe0f'}",
        }

        import pandas as pd
        metrics_df = pd.DataFrame(list(metrics_summary.items()),
                                   columns=["Metric", "Value"])
        mo.vstack([
            mo.md("## 📐 Validation Metrics"),
            mo.ui.table(metrics_df)
        ])
    except ImportError:
        mo.md(f"**AUC:** `{auc:.4f}` — install utils for full metrics")
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## 🗓️ Next Steps

    - [ ] WOE transformation for all candidate features
    - [ ] Fine/coarse classing for scorecard bins
    - [ ] PDO scorecard calibration (base score=600, PDO=20, odds=1:1)
    - [ ] PSI monitoring on hold-out set
    - [ ] Export report: `marimo export html notebooks/01_eda_scorecard.py`
    """)
    return


if __name__ == "__main__":
    app.run()
