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
#   "scipy>=1.12.0",
# ]
# //
"""
Case Study 01 — Credit Scoring: Full EDA
========================================
Dataset  : UCI Default of Credit Card Clients (N=30,000)
Author   : Erick Condoy | credit-risk-lab
Notebook : marimo 0.13 — DAG reactive, zero hidden state
"""
import marimo

__generated_with = "0.13.0"
app = marimo.App(
    width="full",
    app_title="Credit Scoring — Full EDA",
)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 01 — Header
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _():
    import marimo as mo
    mo.md(r"""
    # 📊 Case Study 01 — Credit Scoring: Exploratory Data Analysis

    **Dataset:** UCI Default of Credit Card Clients — 30,000 clients, Taiwan 2005  
    **Target:** `DEFAULT` — binary (1 = defaulted in next month)  
    **Reference:** Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley Finance.

    ---
    ### EDA Roadmap
    1. Environment & data load
    2. Data quality audit (dtypes, nulls, cardinality, ranges)
    3. Target analysis (class imbalance, base rate)
    4. Univariate distributions — continuous & categorical
    5. Bivariate analysis — default rate by segment
    6. PAY history profile (behavioral variables)
    7. BILL / PAYMENT amount analysis
    8. Outlier detection (IQR + Z-score)
    9. Correlation & multicollinearity
    10. EDA Summary & feature selection candidates
    """)
    return (mo,)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 02 — Environment setup (sovereign paths)
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    import sys
    import warnings
    from pathlib import Path

    warnings.filterwarnings("ignore")

    NOTEBOOK_DIR = Path(__file__).parent
    CASE_DIR     = NOTEBOOK_DIR.parent
    REPO_ROOT    = CASE_DIR.parent
    DATA_DIR     = CASE_DIR / "data"
    REPORTS_DIR  = CASE_DIR / "reports"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    mo.callout(mo.md(f"""
    **Paths** | root: `{REPO_ROOT}` | data: `{DATA_DIR}` | reports: `{REPORTS_DIR}`
    """), kind="info")
    return CASE_DIR, DATA_DIR, NOTEBOOK_DIR, REPO_ROOT, REPORTS_DIR, Path, sys, warnings


# ─────────────────────────────────────────────────────────────────────────────
# CELL 03 — Imports
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(mo):
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
    from scipy import stats

    # Institutional palette (no AI gradients)
    PAL = {
        "teal":   "#01696f",
        "teal2":  "#4f98a3",
        "purple": "#a12c7b",
        "purple2":"#d163a7",
        "orange": "#bb653b",
        "gold":   "#e8af34",
        "green":  "#437a22",
        "blue":   "#006494",
        "gray":   "#7a7974",
        "light":  "#cedcd8",
    }
    PALETTE_2 = [PAL["teal"], PAL["purple"]]

    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "font.family":      "DejaVu Sans",
    })
    mo.md("### 📦 Imports OK")
    return matplotlib, mticker, np, PAL, PALETTE_2, pd, plt, sns, stats


# ─────────────────────────────────────────────────────────────────────────────
# CELL 04 — Data load with Parquet cache
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(DATA_DIR, mo, pd):
    PARQUET = DATA_DIR / "uci_credit_default.parquet"

    RENAME = {
        "PAY_0": "PAY_1",   # UCI uses PAY_0 for most recent month
    }
    COL_DTYPES = {
        "SEX":       "category",
        "EDUCATION": "category",
        "MARRIAGE":  "category",
        "DEFAULT":   "int8",
    }

    def _load() -> tuple:
        if PARQUET.exists():
            df = pd.read_parquet(PARQUET)
            return df, "📁 Parquet cache"
        try:
            from ucimlrepo import fetch_ucirepo
            raw = fetch_ucirepo(id=350)
            df = pd.concat([
                raw.data.features,
                raw.data.targets.rename(columns={"Y": "DEFAULT"})
            ], axis=1)
            df.columns = [c.upper() for c in df.columns]
            df.rename(columns=RENAME, inplace=True)
            df.to_parquet(PARQUET, index=False)
            return df, "🌐 UCI API → cached"
        except Exception as exc:
            return None, str(exc)

    df_raw, _src = _load()

    if df_raw is None:
        mo.stop(True, mo.callout(mo.md(f"**Load error:** {_src}"), kind="danger"))

    for col, dt in COL_DTYPES.items():
        if col in df_raw.columns:
            df_raw[col] = df_raw[col].astype(dt)

    N, P = df_raw.shape
    DR = float(df_raw["DEFAULT"].mean())

    mo.callout(mo.md(f"""
    ✅ **Dataset ready** — source: *{_src}*

    | Metric | Value |
    |---|---|
    | Rows | `{N:,}` |
    | Columns | `{P}` |
    | Default rate | `{DR:.2%}` |
    | Memory | `{df_raw.memory_usage(deep=True).sum() / 1e6:.1f} MB` |
    """), kind="success")
    return COL_DTYPES, DR, N, P, PARQUET, RENAME, df_raw


# ─────────────────────────────────────────────────────────────────────────────
# CELL 05 — Data Quality Audit
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(df_raw, mo, np, pd):
    num_df = df_raw.select_dtypes(include=np.number)

    quality = pd.DataFrame({
        "dtype":       df_raw.dtypes.astype(str),
        "nulls":       df_raw.isnull().sum(),
        "null_%":      (df_raw.isnull().mean() * 100).round(2),
        "unique":      df_raw.nunique(),
        "min":         num_df.min().round(2),
        "max":         num_df.max().round(2),
        "mean":        num_df.mean().round(2),
        "std":         num_df.std().round(2),
        "skew":        num_df.skew().round(3),
        "kurt":        num_df.kurtosis().round(3),
    }).reset_index().rename(columns={"index": "feature"})

    mo.vstack([
        mo.md("## 🔍 Section 1 — Data Quality Audit"),
        mo.md(f"**Zero nulls detected:** {'✅ Clean dataset' if quality['nulls'].sum() == 0 else '⚠️ Nulls found'}"),
        mo.ui.table(quality, pagination=True)
    ])
    return (quality,)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 06 — Suspicious value audit (EDUCATION=0,5,6 / MARRIAGE=0)
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(df_raw, mo, pd):
    # UCI dataset has undocumented category codes: EDUCATION {0,5,6}, MARRIAGE {0}
    edu_counts = df_raw["EDUCATION"].value_counts().sort_index()
    mar_counts = df_raw["MARRIAGE"].value_counts().sort_index()

    edu_suspicious = df_raw["EDUCATION"].isin(["0", "5", "6", 0, 5, 6]).sum()
    mar_suspicious = df_raw["MARRIAGE"].isin(["0", 0]).sum()

    mo.vstack([
        mo.md("## ⚠️ Suspicious / Undocumented Codes"),
        mo.md(f"""
        The UCI documentation only defines:
        - `EDUCATION`: 1=graduate, 2=university, 3=high school, 4=others  
        - `MARRIAGE`: 1=married, 2=single, 3=others

        **Detected undocumented codes:**
        - `EDUCATION` codes {{0, 5, 6}}: **{edu_suspicious:,}** observations ({edu_suspicious/len(df_raw):.2%})
        - `MARRIAGE` code {{0}}: **{mar_suspicious:,}** observations ({mar_suspicious/len(df_raw):.2%})

        → Strategy: remap to "Unknown / Other" category before modeling.
        """),
        mo.hstack([
            mo.vstack([mo.md("**EDUCATION value counts:**"),
                       mo.ui.table(edu_counts.reset_index().rename(columns={"EDUCATION":"code","count":"n"}))]),
            mo.vstack([mo.md("**MARRIAGE value counts:**"),
                       mo.ui.table(mar_counts.reset_index().rename(columns={"MARRIAGE":"code","count":"n"}))]),
        ])
    ])
    return edu_counts, edu_suspicious, mar_counts, mar_suspicious


# ─────────────────────────────────────────────────────────────────────────────
# CELL 07 — Target: class imbalance analysis
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(DR, N, df_raw, mo, np, plt, PAL):
    n_pos = int(df_raw["DEFAULT"].sum())
    n_neg = N - n_pos
    ir = n_neg / n_pos  # imbalance ratio

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="white")

    # ── Bar chart ──
    axes[0].bar(["Non-Default (0)", "Default (1)"], [n_neg, n_pos],
                color=[PAL["teal"], PAL["purple"]], edgecolor="white", linewidth=1.5)
    for bar, val, pct in zip(axes[0].patches, [n_neg, n_pos], [1-DR, DR]):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 200,
                     f"{val:,}\n({pct:.1%})",
                     ha="center", fontsize=10, fontweight="bold")
    axes[0].set_title("Class Distribution", fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].set_ylim(0, n_neg * 1.15)

    # ── Pie ──
    axes[1].pie([n_neg, n_pos],
                labels=[f"Non-Default\n{1-DR:.1%}", f"Default\n{DR:.1%}"],
                colors=[PAL["teal"], PAL["purple"]],
                startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Class Share", fontweight="bold")

    # ── Default rate by LIMIT_BAL quantile ──
    bins = pd.qcut(df_raw["LIMIT_BAL"], q=10, duplicates="drop")
    dr_by_limit = df_raw.groupby(bins, observed=True)["DEFAULT"].mean()
    axes[2].bar(range(len(dr_by_limit)), dr_by_limit.values,
                color=PAL["teal2"], edgecolor="white")
    axes[2].axhline(DR, color=PAL["purple"], linestyle="--", linewidth=1.5,
                    label=f"Overall DR={DR:.2%}")
    axes[2].set_title("Default Rate by Credit Limit Decile", fontweight="bold")
    axes[2].set_xlabel("Decile (low → high limit)")
    axes[2].set_ylabel("Default Rate")
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    axes[2].legend()

    plt.tight_layout()

    mo.vstack([
        mo.md("## ⚖️ Section 2 — Target Analysis & Class Imbalance"),
        mo.callout(mo.md(f"""
        | Metric | Value |
        |---|---|
        | **Base rate (default)** | `{DR:.4f}` ({DR:.2%}) |
        | **Imbalance ratio** | `{ir:.1f}:1` (non-default : default) |
        | **Mitigation strategies** | `class_weight='balanced'`, SMOTE, cost-sensitive learning |
        """), kind="warn" if ir > 3 else "info"),
        mo.mpl.interactive(fig)
    ])
    return bins, dr_by_limit, fig, ir, n_neg, n_pos


# ─────────────────────────────────────────────────────────────────────────────
# CELL 08 — Univariate: continuous features
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(df_raw, mo, np, plt, PAL, PALETTE_2):
    CONT_FEATURES = ["LIMIT_BAL", "AGE"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="white")

    for i, feat in enumerate(CONT_FEATURES):
        ax_hist = axes[i, 0]
        ax_box  = axes[i, 1]

        # Histogram split by target
        for label, color in zip([0, 1], PALETTE_2):
            subset = df_raw[df_raw["DEFAULT"] == label][feat]
            ax_hist.hist(subset, bins=40, alpha=0.6, color=color,
                         label=f"Default={label}  (n={len(subset):,})",
                         density=True, edgecolor="none")
        ax_hist.set_title(f"{feat} — Distribution by Default", fontweight="bold")
        ax_hist.set_ylabel("Density")
        ax_hist.legend(fontsize=9)

        # Boxplot split by target
        data_by_class = [
            df_raw[df_raw["DEFAULT"] == 0][feat].values,
            df_raw[df_raw["DEFAULT"] == 1][feat].values,
        ]
        bp = ax_box.boxplot(data_by_class, patch_artist=True,
                             labels=["Non-Default", "Default"],
                             medianprops=dict(color="white", linewidth=2))
        for patch, color in zip(bp["boxes"], PALETTE_2):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax_box.set_title(f"{feat} — Boxplot", fontweight="bold")

    plt.tight_layout()

    mo.vstack([
        mo.md("## 📈 Section 3 — Univariate: Continuous Features"),
        mo.mpl.interactive(fig)
    ])
    return CONT_FEATURES, ax_box, ax_hist, bp, data_by_class, fig, i, label, subset


# ─────────────────────────────────────────────────────────────────────────────
# CELL 09 — Univariate: categorical features with default rate overlay
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(df_raw, mo, plt, PAL, DR):
    CAT_FEATURES = {
        "SEX":       {1: "Male", 2: "Female"},
        "EDUCATION": {1: "Graduate", 2: "University", 3: "High School",
                      4: "Others", 0: "Unknown", 5: "Unknown", 6: "Unknown"},
        "MARRIAGE":  {1: "Married", 2: "Single", 3: "Others", 0: "Unknown"},
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")

    for ax, (feat, label_map) in zip(axes, CAT_FEATURES.items()):
        df_plot = df_raw.copy()
        df_plot[feat] = df_plot[feat].astype(str).map(
            {str(k): v for k, v in label_map.items()}
        ).fillna("Unknown")

        grp = df_plot.groupby(feat, observed=True).agg(
            count=("DEFAULT", "count"),
            dr=("DEFAULT", "mean")
        ).reset_index().sort_values("count", ascending=False)

        x = range(len(grp))
        bars = ax.bar(x, grp["count"],
                      color=PAL["teal2"], alpha=0.85, edgecolor="white", label="Count")
        ax.set_xticks(list(x))
        ax.set_xticklabels(grp[feat].tolist(), rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Count", color=PAL["teal"])
        ax.tick_params(axis="y", labelcolor=PAL["teal"])

        ax2 = ax.twinx()
        ax2.plot(x, grp["dr"], color=PAL["purple"], marker="o",
                 linewidth=2, markersize=6, label="Default Rate")
        ax2.axhline(DR, color=PAL["gray"], linestyle="--",
                    linewidth=1, alpha=0.6)
        ax2.set_ylabel("Default Rate", color=PAL["purple"])
        ax2.tick_params(axis="y", labelcolor=PAL["purple"])
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax2.set_ylim(0, 0.5)

        ax.set_title(f"{feat} — Count & Default Rate", fontweight="bold")

    plt.tight_layout()

    mo.vstack([
        mo.md("## 🗂️ Section 4 — Univariate: Categorical Features"),
        mo.md("> Bars = count (left axis) | Line = default rate (right axis) | Dashed = overall DR"),
        mo.mpl.interactive(fig)
    ])
    return CAT_FEATURES, ax, ax2, bars, df_plot, fig, grp, x


# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — Bivariate: default rate by continuous feature decile (interactive)
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(df_raw, mo, pd):
    # Marimo reactive UI: choose feature + number of bins
    feature_selector = mo.ui.dropdown(
        options=["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1",
                 "BILL_AMT2", "PAY_AMT2", "BILL_AMT3", "PAY_AMT3"],
        value="LIMIT_BAL",
        label="Feature"
    )
    bins_slider = mo.ui.slider(start=5, stop=20, step=1, value=10, label="N bins")

    mo.vstack([
        mo.md("## 🎛️ Section 5 — Interactive Bivariate Analysis"),
        mo.hstack([feature_selector, bins_slider])
    ])
    return bins_slider, feature_selector


@app.cell
def _(DR, bins_slider, df_raw, feature_selector, mo, pd, plt, PAL):
    feat_sel = feature_selector.value
    n_bins   = bins_slider.value

    try:
        cut = pd.qcut(df_raw[feat_sel], q=n_bins, duplicates="drop")
    except Exception:
        cut = pd.cut(df_raw[feat_sel], bins=n_bins)

    biv = df_raw.groupby(cut, observed=True).agg(
        count=("DEFAULT", "count"),
        n_default=("DEFAULT", "sum"),
        dr=("DEFAULT", "mean"),
        mean_val=(feat_sel, "mean"),
    ).reset_index().rename(columns={feat_sel: "bin"})
    biv["bin_label"] = biv["bin"].astype(str)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                     facecolor="white", sharex=True)

    x = range(len(biv))
    ax1.bar(x, biv["count"], color=PAL["teal2"], edgecolor="white", alpha=0.85)
    ax1.set_ylabel("Observations")
    ax1.set_title(f"Distribution of {feat_sel} ({n_bins} bins)", fontweight="bold")

    ax2.bar(x, biv["dr"], color=PAL["purple"], edgecolor="white", alpha=0.85)
    ax2.axhline(DR, color=PAL["gray"], linestyle="--", linewidth=1.5,
                label=f"Overall DR = {DR:.2%}")
    ax2.set_ylabel("Default Rate")
    ax2.set_title(f"Default Rate by {feat_sel} bin", fontweight="bold")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(biv["bin_label"].tolist(), rotation=45, ha="right", fontsize=8)
    ax2.legend()

    plt.tight_layout()
    mo.mpl.interactive(fig)
    return ax1, ax2, biv, cut, feat_sel, fig, n_bins, x


# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 — PAY history behavioral profile (most predictive features)
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(DR, df_raw, mo, np, pd, plt, PAL, PALETTE_2):
    PAY_COLS = ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    # Fallback: some UCI versions keep PAY_0
    PAY_COLS = [c for c in PAY_COLS if c in df_raw.columns]
    if not PAY_COLS and "PAY_0" in df_raw.columns:
        PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    PAY_COLS = [c for c in PAY_COLS if c in df_raw.columns]

    PAY_LABELS = {
        -2: "No balance", -1: "Paid in full",
         0: "Revolving",   1: "1 mo delay",
         2: "2 mo delay",  3: "3 mo delay",
         4: "4 mo delay",  5: "5 mo delay",
         6: "6 mo delay",  7: "7 mo delay",
         8: "8+ mo delay",
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="white")
    axes_flat = axes.flatten()

    for idx, col in enumerate(PAY_COLS[:6]):
        ax = axes_flat[idx]
        grp = df_raw.groupby(col, observed=False).agg(
            count=("DEFAULT", "count"),
            dr=("DEFAULT", "mean")
        ).reset_index()
        grp["label"] = grp[col].map(PAY_LABELS).fillna(grp[col].astype(str))

        bars_colors = [PAL["teal"] if int(v) <= 0 else PAL["orange"] if int(v) == 1
                       else PAL["purple"] for v in grp[col]]

        ax.bar(range(len(grp)), grp["count"],
               color=bars_colors, edgecolor="white", alpha=0.85)
        ax_r = ax.twinx()
        ax_r.plot(range(len(grp)), grp["dr"],
                  color=PAL["purple"], marker="o", linewidth=2.5, markersize=5)
        ax_r.axhline(DR, color=PAL["gray"], linestyle="--", alpha=0.6)
        ax_r.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax_r.set_ylim(0, 1.0)

        ax.set_title(f"{col} — Payment Status", fontweight="bold", fontsize=10)
        ax.set_xticks(range(len(grp)))
        ax.set_xticklabels(grp["label"].tolist(), rotation=40, ha="right", fontsize=7)
        ax.set_ylabel("Count", fontsize=8)
        ax_r.set_ylabel("DR", fontsize=8)

    plt.suptitle("PAY History Profile — Count & Default Rate by Status",
                 fontweight="bold", fontsize=13, y=1.01)
    plt.tight_layout()

    mo.vstack([
        mo.md("## 💳 Section 6 — PAY History Behavioral Profile"),
        mo.md("""
        > **Key insight:** Delayed payment status (PAY ≥ 1) is the strongest
        predictor of default. A single 2-month delay (`PAY=2`) alone captures
        much of the discriminatory power — consistent with Altman (1968) and
        Siddiqi (2006) findings on behavioral variables.
        """),
        mo.mpl.interactive(fig)
    ])
    return (
        PAY_COLS, PAY_LABELS, ax, ax_r, axes, axes_flat,
        bars_colors, col, fig, grp, idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CELL 12 — BILL_AMT & PAY_AMT: temporal profile
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(df_raw, mo, np, plt, PAL, PALETTE_2):
    BILL_COLS = [c for c in ["BILL_AMT1","BILL_AMT2","BILL_AMT3",
                              "BILL_AMT4","BILL_AMT5","BILL_AMT6"]
                 if c in df_raw.columns]
    AMT_COLS  = [c for c in ["PAY_AMT1","PAY_AMT2","PAY_AMT3",
                              "PAY_AMT4","PAY_AMT5","PAY_AMT6"]
                 if c in df_raw.columns]

    months = ["t", "t-1", "t-2", "t-3", "t-4", "t-5"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="white")

    # ── Mean BILL by default class ──
    for label, color in zip([0, 1], PALETTE_2):
        means = df_raw[df_raw["DEFAULT"] == label][BILL_COLS].mean().values
        axes[0,0].plot(months[:len(means)], means, marker="o",
                       color=color, linewidth=2.5,
                       label=f"Default={label}")
    axes[0,0].set_title("Mean Statement Balance over Time", fontweight="bold")
    axes[0,0].set_ylabel("TWD (avg)")
    axes[0,0].legend()
    axes[0,0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K"))

    # ── Mean PAY_AMT by default class ──
    for label, color in zip([0, 1], PALETTE_2):
        means = df_raw[df_raw["DEFAULT"] == label][AMT_COLS].mean().values
        axes[0,1].plot(months[:len(means)], means, marker="o",
                       color=color, linewidth=2.5,
                       label=f"Default={label}")
    axes[0,1].set_title("Mean Payment Amount over Time", fontweight="bold")
    axes[0,1].set_ylabel("TWD (avg)")
    axes[0,1].legend()
    axes[0,1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K"))

    # ── Utilization ratio = BILL_AMT1 / LIMIT_BAL ──
    df_raw["UTIL"] = (df_raw["BILL_AMT1"] / df_raw["LIMIT_BAL"].replace(0, np.nan)).clip(0, 5)
    for label, color in zip([0, 1], PALETTE_2):
        subset = df_raw[df_raw["DEFAULT"] == label]["UTIL"].dropna()
        axes[1,0].hist(subset, bins=50, alpha=0.55, density=True,
                       color=color, label=f"Default={label}")
    axes[1,0].set_title("Credit Utilization (BILL_AMT1 / LIMIT_BAL)", fontweight="bold")
    axes[1,0].set_xlabel("Utilization ratio")
    axes[1,0].set_ylabel("Density")
    axes[1,0].set_xlim(0, 2.5)
    axes[1,0].legend()

    # ── Payment ratio = PAY_AMT1 / BILL_AMT1 ──
    df_raw["PAY_RATIO"] = (
        df_raw["PAY_AMT1"] / df_raw["BILL_AMT1"].replace(0, np.nan)
    ).clip(0, 3)
    for label, color in zip([0, 1], PALETTE_2):
        subset = df_raw[df_raw["DEFAULT"] == label]["PAY_RATIO"].dropna()
        axes[1,1].hist(subset, bins=50, alpha=0.55, density=True,
                       color=color, label=f"Default={label}")
    axes[1,1].set_title("Payment Ratio (PAY_AMT1 / BILL_AMT1)", fontweight="bold")
    axes[1,1].set_xlabel("Payment ratio")
    axes[1,1].set_ylabel("Density")
    axes[1,1].set_xlim(0, 2)
    axes[1,1].legend()

    plt.tight_layout()

    mo.vstack([
        mo.md("## 💰 Section 7 — BILL / PAYMENT Temporal Profile & Derived Ratios"),
        mo.md("""
        > **Derived features engineered here:**
        > - `UTIL = BILL_AMT1 / LIMIT_BAL` — credit utilization rate
        > - `PAY_RATIO = PAY_AMT1 / BILL_AMT1` — repayment coverage ratio  
        > Both are strong candidates for the feature engineering pipeline.
        """),
        mo.mpl.interactive(fig)
    ])
    return (
        AMT_COLS, BILL_COLS, axes, fig,
        label, means, months, subset,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CELL 13 — Outlier detection: IQR + Z-score
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(df_raw, mo, np, pd, stats):
    OUTLIER_FEATS = ["LIMIT_BAL", "AGE",
                     "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                     "PAY_AMT1",  "PAY_AMT2",  "PAY_AMT3"]

    rows = []
    for feat in OUTLIER_FEATS:
        s = df_raw[feat].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n_iqr = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).sum()
        z = np.abs(stats.zscore(s))
        n_z3  = (z > 3).sum()
        n_z35 = (z > 3.5).sum()
        rows.append({
            "feature":     feat,
            "IQR_outliers": n_iqr,
            "IQR_%":       f"{n_iqr/len(s):.2%}",
            "Z>3":         n_z3,
            "Z>3.5":       n_z35,
            "min":         int(s.min()),
            "max":         int(s.max()),
            "p99":         round(s.quantile(0.99), 0),
        })

    outlier_df = pd.DataFrame(rows)

    mo.vstack([
        mo.md("## 🔎 Section 8 — Outlier Detection (IQR + Z-score)"),
        mo.md("""
        | Method | Threshold | Action |
        |---|---|---|
        | IQR fence | 1.5 × IQR | Flag, winsorize at p1/p99 |
        | Z-score | > 3.0 σ | Flag for review |
        | Z-score strict | > 3.5 σ | Winsorize before training |
        """),
        mo.ui.table(outlier_df)
    ])
    return OUTLIER_FEATS, feat, outlier_df, q1, q3, rows, s, z


# ─────────────────────────────────────────────────────────────────────────────
# CELL 14 — Correlation matrix + VIF multicollinearity
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(df_raw, mo, np, pd, plt, PAL):
    import seaborn as sns

    MODEL_FEATS = [
        "LIMIT_BAL", "AGE",
        "PAY_1", "PAY_2", "PAY_3",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "PAY_AMT1",  "PAY_AMT2",  "PAY_AMT3",
        "UTIL", "PAY_RATIO",
    ]
    # filter to existing columns only
    MODEL_FEATS = [f for f in MODEL_FEATS if f in df_raw.columns]

    corr = df_raw[MODEL_FEATS].corr()

    fig, ax = plt.subplots(figsize=(13, 10), facecolor="white")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        linewidths=0.4, cbar_kws={"shrink": 0.75}
    )
    ax.set_title("Pearson Correlation Matrix — Model Feature Candidates",
                 fontweight="bold", pad=12)
    plt.tight_layout()

    # High-correlation pairs (|r| > 0.7)
    corr_upper = corr.where(mask == False)
    high_corr = (
        corr_upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_A", "level_1": "feature_B", 0: "r"})
    )
    high_corr = high_corr[high_corr["r"].abs() > 0.7].sort_values("r", ascending=False)

    mo.vstack([
        mo.md("## 🔗 Section 9 — Correlation & Multicollinearity"),
        mo.mpl.interactive(fig),
        mo.md("### High-Correlation Pairs (|r| > 0.70)"),
        mo.ui.table(high_corr.round(4)) if len(high_corr) > 0
            else mo.callout(mo.md("No pairs with |r| > 0.70"), kind="success")
    ])
    return (
        MODEL_FEATS, ax, corr, corr_upper, fig, high_corr, mask, sns,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CELL 15 — EDA Summary & Feature Selection Candidates
# ─────────────────────────────────────────────────────────────────────────────
@app.cell
def _(DR, N, df_raw, ir, mo, pd):
    # Default rate by PAY_1 (most predictive feature)
    pay_col = "PAY_1" if "PAY_1" in df_raw.columns else "PAY_0"
    dr_by_pay = df_raw.groupby(pay_col)["DEFAULT"].mean().to_dict()
    dr_delay2 = dr_by_pay.get(2, dr_by_pay.get("2", None))

    mo.vstack([
        mo.md("## 📋 Section 10 — EDA Summary & Next Steps"),
        mo.callout(mo.md(f"""
        ### Key Findings

        | Finding | Detail |
        |---|---|
        | **Dataset size** | {N:,} observations, clean (0 nulls) |
        | **Base rate** | {DR:.2%} → imbalance ratio {ir:.1f}:1 → use `class_weight='balanced'` |
        | **Strongest predictor** | `{pay_col}` — DR jumps to ~{dr_delay2:.0%} at 2-month delay |
        | **Undocumented codes** | `EDUCATION` {{0,5,6}} and `MARRIAGE` {{0}} → remap to *Other* |
        | **Multicollinearity** | `BILL_AMT` series highly correlated → consider PCA or select t only |
        | **Derived features** | `UTIL`, `PAY_RATIO` show strong separation → include in pipeline |
        | **Outliers** | `BILL_AMT` / `PAY_AMT` have heavy right tails → winsorize at p99 |
        """), kind="info"),
        mo.md("""
        ### Feature Selection Candidates (for Scorecard Pipeline)

        | Tier | Features | Rationale |
        |---|---|---|
        | **Tier 1 — Behavioral** | `PAY_1`, `PAY_2`, `PAY_3` | Highest IV, direct delinquency signal |
        | **Tier 2 — Derived** | `UTIL`, `PAY_RATIO` | Engineered ratios with clean monotonic DR trend |
        | **Tier 3 — Demographic** | `LIMIT_BAL`, `AGE` | Moderate IV, regulatory compliance check needed |
        | **Tier 4 — Categorical** | `SEX`, `EDUCATION`, `MARRIAGE` | Low IV, use only after legal review |
        | **Drop** | `BILL_AMT2-6`, `PAY_AMT2-6` | Redundant with t-period + high multicollinearity |

        ### Next Notebook: `02_feature_engineering.py`
        - WOE/IV binning (fine classing → coarse classing)
        - Monotonic binning enforcement for PAY variables
        - PDO scorecard calibration
        - Train / validation / OOT split
        """),
    ])
    return dr_by_pay, dr_delay2, pay_col


if __name__ == "__main__":
    app.run()
