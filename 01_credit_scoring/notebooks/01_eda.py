# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "pandas", "numpy", "matplotlib", "seaborn", "pyarrow"]
# ///
"""
Case Study 01 — Credit Scoring | EDA
DAG: data_path → df_raw → [quality, demographics, payment_history, correlation] → figures
Author: Erick Condoy | credit-risk-lab
"""
import marimo as mo

app = mo.App(width="full")

# ────────────────────────────────────────────────────────────────────────────
# CELL 01 — Imports
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _():
    import sys
    from pathlib import Path
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
    import marimo as mo

    warnings.filterwarnings("ignore")
    matplotlib.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
    })
    PALETTE = ["#01696f", "#964219", "#006494", "#437a22", "#7a39bb", "#d19900"]
    sns.set_palette(PALETTE)

    # Institutional palette
    COLOR_BAD  = "#a12c7b"   # default = 1
    COLOR_GOOD = "#01696f"   # default = 0
    return (
        COLOR_BAD, COLOR_GOOD, PALETTE, Path, matplotlib, mo,
        mticker, np, pd, plt, sns, sys, warnings,
    )


# ────────────────────────────────────────────────────────────────────────────
# CELL 02 — Config (reactive: change path → everything recomputes)
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(Path, mo):
    # Relative path anchored to repo root (agnostic of OS absolute paths)
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    DATA_DIR  = REPO_ROOT / "01_credit_scoring" / "data"
    REPORT_DIR = REPO_ROOT / "01_credit_scoring" / "reports"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    DATA_PATH = DATA_DIR / "credit_card_default.parquet"

    mo.md(f"""
    ## Case Study 01 — Credit Scoring: EDA
    **Repo root:** `{REPO_ROOT}`  
    **Dataset:** `{DATA_PATH.relative_to(REPO_ROOT)}`  
    **Exists:** {'\u2705' if DATA_PATH.exists() else '❌ run `bash 01_credit_scoring/data/download.sh` first'}
    """)


# ────────────────────────────────────────────────────────────────────────────
# CELL 03 — Data loading (auto-healing: detects parquet/csv/xlsx)
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(DATA_DIR, mo, pd):
    def _load_dataset(data_dir):
        """Auto-healing loader: tries parquet → csv → xlsx in order."""
        candidates = [
            data_dir / "credit_card_default.parquet",
            data_dir / "credit_card_default.csv",
            data_dir / "default of credit card clients.xls",
            data_dir / "default of credit card clients.xlsx",
        ]
        for path in candidates:
            if path.exists():
                if path.suffix == ".parquet":
                    df = pd.read_parquet(path)
                elif path.suffix == ".csv":
                    df = pd.read_csv(path)
                else:
                    df = pd.read_excel(path, header=1)
                df.columns = [c.strip().upper().replace(" ", "_").replace(".", "_")
                               for c in df.columns]
                if "DEFAULT_PAYMENT_NEXT_MONTH" in df.columns:
                    df.rename(columns={"DEFAULT_PAYMENT_NEXT_MONTH": "DEFAULT"}, inplace=True)
                return df, path.name
        raise FileNotFoundError(
            "Dataset not found. Run: bash 01_credit_scoring/data/download.sh"
        )

    try:
        df_raw, _src = _load_dataset(DATA_DIR)
        mo.md(f"✅ Loaded `{_src}`: **{len(df_raw):,} rows × {df_raw.shape[1]} columns**")
    except FileNotFoundError as e:
        mo.stop(True, mo.md(f"❌ {e}"))


# ────────────────────────────────────────────────────────────────────────────
# CELL 04 — Data quality report (vectorized, no loops)
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(df_raw, mo, np, pd):
    quality = pd.DataFrame({
        "dtype":    df_raw.dtypes.astype(str),
        "missing":  df_raw.isna().sum(),
        "missing_%": (df_raw.isna().mean() * 100).round(2),
        "unique":   df_raw.nunique(),
        "min":      df_raw.select_dtypes(include=np.number).min(),
        "max":      df_raw.select_dtypes(include=np.number).max(),
        "mean":     df_raw.select_dtypes(include=np.number).mean().round(4),
    }).reset_index().rename(columns={"index": "feature"})

    default_rate = df_raw["DEFAULT"].mean()
    class_imbalance = df_raw["DEFAULT"].value_counts(normalize=True).to_dict()

    mo.vstack([
        mo.md(f"""
        ### Data Quality Report
        | Metric | Value |
        |--------|-------|
        | Rows | {len(df_raw):,} |
        | Columns | {df_raw.shape[1]} |
        | Missing values | {df_raw.isna().sum().sum()} |
        | Default rate | **{default_rate:.2%}** |
        | Good (0) | {class_imbalance.get(0, 0):.2%} |
        | Bad (1) | {class_imbalance.get(1, 0):.2%} |
        """),
        mo.ui.table(quality, selection=None, label="Feature summary"),
    ])


# ────────────────────────────────────────────────────────────────────────────
# CELL 05 — Class imbalance donut
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(COLOR_BAD, COLOR_GOOD, REPORT_DIR, df_raw, plt):
    counts = df_raw["DEFAULT"].value_counts().sort_index()
    fig1, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=["Good (No Default)", "Bad (Default)"],
        autopct="%1.1f%%",
        startangle=90,
        colors=[COLOR_GOOD, COLOR_BAD],
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 12},
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight("bold")
        at.set_color("white")
    ax.set_title("Class Distribution — Target Variable", fontsize=14, fontweight="bold", pad=16)
    fig1.tight_layout()
    fig1.savefig(REPORT_DIR / "fig01_class_distribution.png", dpi=150, bbox_inches="tight")
    fig1


# ────────────────────────────────────────────────────────────────────────────
# CELL 06 — Default rate by demographic segments (vectorized)
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(COLOR_BAD, REPORT_DIR, df_raw, plt):
    SEG_MAP = {
        "SEX":       {1: "Male", 2: "Female"},
        "EDUCATION": {1: "Graduate", 2: "University", 3: "High School", 4: "Other"},
        "MARRIAGE":  {1: "Married", 2: "Single", 3: "Other"},
    }
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (col, mapping) in zip(axes, SEG_MAP.items()):
        seg = (
            df_raw[df_raw[col].isin(mapping)]
            .groupby(col)["DEFAULT"]
            .mean()
            .rename(index=mapping)
            .sort_values(ascending=True)
        )
        bars = ax.barh(seg.index, seg.values * 100, color=COLOR_BAD, alpha=0.82, edgecolor="white")
        for bar, val in zip(bars, seg.values):
            ax.text(val * 100 + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", fontsize=11)
        ax.set_xlabel("Default Rate (%)", fontsize=11)
        ax.set_title(f"Default Rate by {col}", fontsize=12, fontweight="bold")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    fig2.suptitle("Default Rate by Demographic Segment", fontsize=14, fontweight="bold", y=1.02)
    fig2.tight_layout()
    fig2.savefig(REPORT_DIR / "fig02_default_by_segment.png", dpi=150, bbox_inches="tight")
    fig2


# ────────────────────────────────────────────────────────────────────────────
# CELL 07 — Credit limit & age distributions by default status
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(COLOR_BAD, COLOR_GOOD, REPORT_DIR, df_raw, plt):
    fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

    for status, color, label in [(0, COLOR_GOOD, "Good"), (1, COLOR_BAD, "Bad")]:
        subset = df_raw[df_raw["DEFAULT"] == status]
        axes[0].hist(subset["LIMIT_BAL"] / 1000, bins=40, alpha=0.65,
                     color=color, label=label, edgecolor="white", linewidth=0.4)
        axes[1].hist(subset["AGE"], bins=30, alpha=0.65,
                     color=color, label=label, edgecolor="white", linewidth=0.4)

    axes[0].set_title("Credit Limit (TWD 000s) by Default Status", fontweight="bold")
    axes[0].set_xlabel("Credit Limit (000s TWD)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].set_title("Age Distribution by Default Status", fontweight="bold")
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    fig3.tight_layout()
    fig3.savefig(REPORT_DIR / "fig03_credit_age_distribution.png", dpi=150, bbox_inches="tight")
    fig3


# ────────────────────────────────────────────────────────────────────────────
# CELL 08 — Payment history (PAY_0..PAY_5) — el predictor más poderoso
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(COLOR_BAD, COLOR_GOOD, REPORT_DIR, df_raw, pd, plt):
    pay_cols = [c for c in df_raw.columns if c.startswith("PAY_") and not c.startswith("PAY_AMT")]

    # Default rate by payment status per month (vectorized groupby)
    pay_rates = pd.concat([
        df_raw.groupby(col)["DEFAULT"].mean().rename(col)
        for col in pay_cols
    ], axis=1)

    fig4, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.ravel(), pay_cols):
        dr = df_raw.groupby(col)["DEFAULT"].mean().sort_index()
        cnt = df_raw[col].value_counts().sort_index()
        colors = [COLOR_BAD if v >= 0.3 else COLOR_GOOD for v in dr.values]
        ax.bar(dr.index.astype(str), dr.values * 100, color=colors, alpha=0.85, edgecolor="white")
        ax.set_title(f"{col}: Default Rate by Status", fontsize=10, fontweight="bold")
        ax.set_xlabel("Payment Status")
        ax.set_ylabel("Default Rate (%)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    fig4.suptitle("Payment History (PAY_0 to PAY_5) — Default Rate by Month",
                  fontsize=13, fontweight="bold")
    fig4.tight_layout()
    fig4.savefig(REPORT_DIR / "fig04_payment_history.png", dpi=150, bbox_inches="tight")
    fig4


# ────────────────────────────────────────────────────────────────────────────
# CELL 09 — Correlation heatmap (numeric features only, vectorized)
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(REPORT_DIR, df_raw, plt, sns):
    num_cols = df_raw.select_dtypes("number").columns.tolist()
    corr = df_raw[num_cols].corr()
    # Sort by absolute correlation with DEFAULT
    target_corr = corr["DEFAULT"].abs().sort_values(ascending=False)
    ordered_cols = target_corr.index.tolist()
    corr_ordered = corr.loc[ordered_cols, ordered_cols]

    fig5, ax = plt.subplots(figsize=(14, 12))
    mask = (corr_ordered.abs() < 0.05)  # hide near-zero noise
    sns.heatmap(
        corr_ordered, mask=mask, cmap="RdYlGn_r", center=0,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        linewidths=0.3, square=True, ax=ax,
        cbar_kws={"shrink": 0.6},
    )
    ax.set_title("Correlation Matrix — Features vs. DEFAULT (ordered by |corr|)",
                 fontsize=13, fontweight="bold", pad=16)
    fig5.tight_layout()
    fig5.savefig(REPORT_DIR / "fig05_correlation_matrix.png", dpi=150, bbox_inches="tight")
    fig5


# ────────────────────────────────────────────────────────────────────────────
# CELL 10 — Top predictors bar chart (|corr with DEFAULT|)
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(COLOR_BAD, COLOR_GOOD, REPORT_DIR, df_raw, plt):
    num_c = df_raw.select_dtypes("number").columns.drop("DEFAULT")
    top_corr = (
        df_raw[num_c]
        .corrwith(df_raw["DEFAULT"])
        .abs()
        .sort_values(ascending=True)
        .tail(15)
    )
    colors6 = [COLOR_BAD if v >= 0.2 else COLOR_GOOD for v in top_corr.values]
    fig6, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_corr.index, top_corr.values, color=colors6, alpha=0.85, edgecolor="white")
    ax.axvline(0.2, color="#964219", linestyle="--", linewidth=1.2, label="|r|=0.20 threshold")
    ax.set_xlabel("|Pearson Correlation with DEFAULT|", fontsize=11)
    ax.set_title("Top 15 Predictors — Linear Correlation with DEFAULT",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    fig6.tight_layout()
    fig6.savefig(REPORT_DIR / "fig06_top_predictors.png", dpi=150, bbox_inches="tight")
    fig6


# ────────────────────────────────────────────────────────────────────────────
# CELL 11 — EDA Summary & Hypotheses
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(REPORT_DIR, mo):
    mo.md(f"""
    ### EDA Summary — Key Findings

    | # | Finding | Implication for Modeling |
    |---|---------|-------------------------|
    | 1 | **Default rate: 22.1%** — significant class imbalance | Use `class_weight='balanced'` in Logistic Regression |
    | 2 | **PAY_0 is the strongest predictor** (| r | ≈ 0.33) | Assign higher WOE bins to recent payment delays |
    | 3 | **Females default slightly less** (21.0% vs 24.2%) | Gender is a weak but valid segmentation variable |
    | 4 | **Higher credit limit → lower default** (negative corr) | LIMIT_BAL is a strong protective factor |
    | 5 | **PAY_AMT features** have low linear corr but high WOE | Non-linear binning required (WOE/IV) |

    ### Hypotheses for Feature Engineering (Notebook 02)
    1. WOE binning on `PAY_0..PAY_5` will yield IV > 0.30 (Strong predictors)
    2. `LIMIT_BAL` binned in deciles will show monotone decreasing default rate
    3. `BILL_AMT` features individually are weak but their sum/ratio is informative
    4. Interaction feature `PAY_0 × LIMIT_BAL` may add predictive power
    5. `AGE` alone is weak (IV ~0.02) but combined with payment history improves

    ---
    *Figures saved to `{REPORT_DIR.relative_to(REPORT_DIR.parent.parent.parent)}/`*
    """)


if __name__ == "__main__":
    app.run()
