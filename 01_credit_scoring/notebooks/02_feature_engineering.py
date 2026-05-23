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
#   "ucimlrepo>=0.0.7",
# ]
# //
"""
Case Study 01 — Credit Scoring: Feature Engineering
=====================================================
Pipeline:
  1. Data load + preprocessing (remap undocumented codes, winsorize, derive ratios)
  2. Train / Validation / OOT split (stratified, temporal proxy)
  3. Fine classing  → WOE/IV per feature
  4. Coarse classing → monotonic bins enforced for PAY variables
  5. IV table & predictor strength ranking
  6. WOE transformation → model-ready matrix
  7. Scorecard calibration (PDO methodology)
  8. Export artifacts: woe_iv_summary.csv, split parquets

Author : Erick Condoy | credit-risk-lab
Ref    : Siddiqi N. (2006). Credit Risk Scorecards. Wiley Finance.
         Thomas L.C. (2009). Consumer Credit Models. OUP.
"""
import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full", app_title="Credit Scoring — Feature Engineering")


# ────────────────────────────────────────────────────────────────────────────
# CELL 01 — Header
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _():
    import marimo as mo
    mo.md(r"""
    # ⚙️ Case Study 01 — Feature Engineering & WOE/IV Pipeline

    **Inputs :** `data/uci_credit_default.parquet` (from 01_eda_scorecard.py)  
    **Outputs:** `data/train.parquet`, `data/val.parquet`, `data/oot.parquet`,
                 `data/woe_iv_summary.csv`  

    ### Pipeline Steps
    1. Preprocessing — recode + winsorize + derive ratios
    2. Train / Val / OOT split (60 / 20 / 20 stratified)
    3. Fine classing (20 equal-frequency bins per feature)
    4. WOE / IV calculation with monotonicity check
    5. Coarse classing — manual merge to enforce monotonic WOE
    6. IV ranking & feature selection decision
    7. WOE transformation → model matrix
    8. Scorecard PDO calibration scaffold
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
    from sklearn.model_selection import train_test_split

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

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    PAL = {
        "teal": "#01696f", "teal2": "#4f98a3", "purple": "#a12c7b",
        "orange": "#bb653b", "gold": "#e8af34", "green": "#437a22",
        "blue": "#006494", "gray": "#7a7974", "light": "#cedcd8",
    }

    mo.callout(mo.md(f"**Root:** `{REPO_ROOT}` | **Data:** `{DATA_DIR}`"), kind="info")
    return (
        CASE_DIR, DATA_DIR, NOTEBOOK_DIR, PAL, REPO_ROOT,
        Path, matplotlib, mticker, np, pd, plt, sns, stats,
        sys, train_test_split, warnings,
    )


# ────────────────────────────────────────────────────────────────────────────
# CELL 03 — Load raw data (requires 01_eda to have run first)
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(DATA_DIR, mo, np, pd):
    PARQUET_RAW = DATA_DIR / "uci_credit_default.parquet"

    if not PARQUET_RAW.exists():
        mo.stop(True, mo.callout(mo.md(
            "**Run `01_eda_scorecard.py` first** to download and cache the dataset."
        ), kind="danger"))

    df = pd.read_parquet(PARQUET_RAW)
    df.columns = [c.upper() for c in df.columns]

    # Normalise PAY_0 → PAY_1 if needed
    if "PAY_0" in df.columns and "PAY_1" not in df.columns:
        df.rename(columns={"PAY_0": "PAY_1"}, inplace=True)

    N0 = len(df)
    mo.callout(mo.md(f"✅ Loaded `{PARQUET_RAW.name}` — **{N0:,}** rows, **{df.shape[1]}** cols"), kind="success")
    return N0, PARQUET_RAW, df


# ────────────────────────────────────────────────────────────────────────────
# CELL 04 — Preprocessing: recode + winsorize + engineer ratios
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(df, mo, np, pd):
    df_p = df.copy()

    # 1. Drop ID (non-predictive)
    df_p.drop(columns=["ID"], errors="ignore", inplace=True)

    # 2. Recode undocumented categorical codes → 'other'
    edu_map = {0: 4, 5: 4, 6: 4}   # 4 = others
    mar_map = {0: 3}                 # 3 = others
    df_p["EDUCATION"] = df_p["EDUCATION"].replace(edu_map).astype(int)
    df_p["MARRIAGE"]  = df_p["MARRIAGE"].replace(mar_map).astype(int)

    # 3. Winsorize heavy-tailed continuous features at [p1, p99]
    WINSORIZE_COLS = [
        "LIMIT_BAL",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1",  "PAY_AMT2",  "PAY_AMT3",
        "PAY_AMT4",  "PAY_AMT5",  "PAY_AMT6",
    ]
    for col in [c for c in WINSORIZE_COLS if c in df_p.columns]:
        lo, hi = df_p[col].quantile(0.01), df_p[col].quantile(0.99)
        df_p[col] = df_p[col].clip(lo, hi)

    # 4. Derived features (from EDA Section 7)
    df_p["UTIL"] = (
        df_p["BILL_AMT1"] / df_p["LIMIT_BAL"].replace(0, np.nan)
    ).clip(0, 5).fillna(0)

    df_p["PAY_RATIO"] = (
        df_p["PAY_AMT1"] / df_p["BILL_AMT1"].replace(0, np.nan)
    ).clip(0, 5).fillna(0)

    # 5. Max delinquency across PAY history
    pay_hist_cols = [c for c in ["PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
                     if c in df_p.columns]
    df_p["MAX_DELAY"] = df_p[pay_hist_cols].clip(lower=0).max(axis=1)

    # 6. Number of months with any delay (PAY > 0)
    df_p["N_DELAYS"] = (df_p[pay_hist_cols] > 0).sum(axis=1)

    N_final = len(df_p)
    new_feats = ["UTIL", "PAY_RATIO", "MAX_DELAY", "N_DELAYS"]

    mo.vstack([
        mo.md("## 🧹 Section 1 — Preprocessing"),
        mo.callout(mo.md(f"""
        | Step | Detail |
        |---|---|
        | Dropped | `ID` |
        | Recoded | `EDUCATION` {{0,5,6}} → 4 (other) · `MARRIAGE` {{0}} → 3 (other) |
        | Winsorized | {len(WINSORIZE_COLS)} columns at p1/p99 |
        | Engineered | `UTIL`, `PAY_RATIO`, `MAX_DELAY`, `N_DELAYS` |
        | Final shape | `{df_p.shape[0]:,}` × `{df_p.shape[1]}` |
        """), kind="success")
    ])
    return WINSORIZE_COLS, N_final, df_p, new_feats, pay_hist_cols


# ────────────────────────────────────────────────────────────────────────────
# CELL 05 — Train / Val / OOT split (stratified, 60/20/20)
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(DATA_DIR, df_p, mo, pd, train_test_split):
    TARGET = "DEFAULT"
    SEED   = 42

    X = df_p.drop(columns=[TARGET])
    y = df_p[TARGET]

    # 60% train | 40% temp
    X_tr, X_temp, y_tr, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=SEED
    )
    # 40% temp → 50/50 val / OOT  → each = 20% of total
    X_val, X_oot, y_val, y_oot = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    train_df = X_tr.assign(DEFAULT=y_tr.values)
    val_df   = X_val.assign(DEFAULT=y_val.values)
    oot_df   = X_oot.assign(DEFAULT=y_oot.values)

    # Persist splits
    train_df.to_parquet(DATA_DIR / "train.parquet", index=False)
    val_df.to_parquet(DATA_DIR   / "val.parquet",   index=False)
    oot_df.to_parquet(DATA_DIR   / "oot.parquet",   index=False)

    mo.vstack([
        mo.md("## ✂️ Section 2 — Data Splits"),
        mo.callout(mo.md(f"""
        | Split | N | Default Rate | Purpose |
        |---|---|---|---|
        | **Train** | {len(train_df):,} ({len(train_df)/len(df_p):.0%}) | {y_tr.mean():.2%} | WOE fitting + model training |
        | **Validation** | {len(val_df):,} ({len(val_df)/len(df_p):.0%}) | {y_val.mean():.2%} | Hyperparameter selection |
        | **OOT** | {len(oot_df):,} ({len(oot_df)/len(df_p):.0%}) | {y_oot.mean():.2%} | Out-of-time hold-out (final eval) |

        > **OOT discipline:** WOE bins are fitted on train ONLY.
        > Val and OOT receive the transformation, never influence it.
        """), kind="info")
    ])
    return (
        SEED, TARGET, X, X_oot, X_temp, X_tr, X_val,
        oot_df, train_df, val_df, y, y_oot, y_temp, y_tr, y_val,
    )


# ────────────────────────────────────────────────────────────────────────────
# CELL 06 — WOE / IV engine (fitted on train only)
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(mo, np, pd):
    """Pure-function WOE/IV engine — no side effects, fully vectorized."""

    def woe_iv_binning(
        series: pd.Series,
        target: pd.Series,
        n_bins: int = 20,
        min_bin_pct: float = 0.02,
    ) -> pd.DataFrame:
        """
        Fine-classing WOE/IV table for a single feature.

        Returns DataFrame with columns:
            bin, n, n_events, n_non_events, pct, event_rate,
            woe, iv_bin, cumulative_iv

        WOE formula  : ln(Distribution_Events / Distribution_Non_Events)
        IV  formula  : (Dist_Events - Dist_Non_Events) × WOE
        IV thresholds (Siddiqi 2006):
            < 0.02  → Useless
            0.02–0.10 → Weak
            0.10–0.30 → Medium
            0.30–0.50 → Strong
            > 0.50  → Suspicious (possible data leakage)
        """
        df = pd.DataFrame({"x": series, "y": target}).dropna()
        total_events     = df["y"].sum()
        total_non_events = len(df) - total_events

        if total_events == 0 or total_non_events == 0:
            raise ValueError(f"Target has only one class — cannot compute WOE.")

        # Fine classing: equal-frequency bins
        try:
            bins = pd.qcut(df["x"], q=n_bins, duplicates="drop")
        except Exception:
            bins = pd.cut(df["x"], bins=n_bins)

        grp = (
            df.assign(bin=bins)
            .groupby("bin", observed=True)
            .agg(n=("y", "count"), n_events=("y", "sum"))
            .reset_index()
        )
        grp["n_non_events"] = grp["n"] - grp["n_events"]
        grp["pct"]          = grp["n"] / len(df)

        # Remove bins below min_bin_pct (merge into adjacent — simple drop for fine classing)
        grp = grp[grp["pct"] >= min_bin_pct].copy()

        # Laplace smoothing to avoid log(0)
        eps = 0.5
        grp["dist_events"]     = (grp["n_events"]     + eps) / (total_events     + eps)
        grp["dist_non_events"] = (grp["n_non_events"] + eps) / (total_non_events + eps)

        grp["event_rate"] = grp["n_events"] / grp["n"]
        grp["woe"]        = np.log(grp["dist_events"] / grp["dist_non_events"])
        grp["iv_bin"]     = (grp["dist_events"] - grp["dist_non_events"]) * grp["woe"]
        grp["cumulative_iv"] = grp["iv_bin"].cumsum()

        return grp[["bin", "n", "n_events", "n_non_events",
                    "pct", "event_rate", "woe", "iv_bin", "cumulative_iv"]]


    def check_monotonicity(woe_series: pd.Series) -> str:
        """Returns 'monotonic_inc', 'monotonic_dec', or 'non_monotonic'."""
        diffs = woe_series.diff().dropna()
        if (diffs >= 0).all():
            return "monotonic_inc ✅"
        elif (diffs <= 0).all():
            return "monotonic_dec ✅"
        else:
            return "non_monotonic ⚠️"


    mo.md("### ⚙️ WOE/IV engine loaded — `woe_iv_binning()` + `check_monotonicity()`")
    return check_monotonicity, woe_iv_binning


# ────────────────────────────────────────────────────────────────────────────
# CELL 07 — Compute WOE/IV for all candidate features (train only)
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(check_monotonicity, mo, pd, train_df, woe_iv_binning):
    TARGET = "DEFAULT"

    CANDIDATE_FEATURES = [
        "PAY_1", "PAY_2", "PAY_3",         # Behavioral — expected highest IV
        "MAX_DELAY", "N_DELAYS",            # Derived behavioral
        "UTIL", "PAY_RATIO",               # Derived ratios
        "LIMIT_BAL", "AGE",                # Demographic
        "BILL_AMT1", "PAY_AMT1",           # Financial t
        "EDUCATION", "MARRIAGE", "SEX",   # Categorical
    ]
    CANDIDATE_FEATURES = [f for f in CANDIDATE_FEATURES if f in train_df.columns]

    woe_tables  = {}   # {feature: DataFrame}
    iv_summary  = []   # [{feature, IV, monotonicity, strength}]

    IV_LABELS = [
        (0.50, "🔴 Suspicious (leakage?)"),
        (0.30, "🟢 Strong"),
        (0.10, "🟡 Medium"),
        (0.02, "🟠 Weak"),
        (0.00, "⚫ Useless"),
    ]

    def iv_label(iv: float) -> str:
        for threshold, label in IV_LABELS:
            if iv >= threshold:
                return label
        return "⚫ Useless"

    for feat in CANDIDATE_FEATURES:
        try:
            tbl = woe_iv_binning(
                train_df[feat], train_df[TARGET],
                n_bins=20, min_bin_pct=0.01
            )
            woe_tables[feat] = tbl
            total_iv = tbl["iv_bin"].sum()
            mono = check_monotonicity(tbl["woe"])
            iv_summary.append({
                "feature":      feat,
                "IV":           round(total_iv, 4),
                "bins":         len(tbl),
                "monotonicity": mono,
                "strength":     iv_label(total_iv),
            })
        except Exception as e:
            iv_summary.append({
                "feature": feat, "IV": None,
                "bins": 0, "monotonicity": "error",
                "strength": str(e)
            })

    iv_df = (
        pd.DataFrame(iv_summary)
        .sort_values("IV", ascending=False)
        .reset_index(drop=True)
    )

    mo.vstack([
        mo.md("## 📊 Section 3 — Information Value (IV) Ranking"),
        mo.md("""
        > **IV interpretation (Siddiqi, 2006):**
        > `< 0.02` Useless | `0.02–0.10` Weak | `0.10–0.30` Medium |
        > `0.30–0.50` Strong | `> 0.50` Suspicious
        """),
        mo.ui.table(iv_df)
    ])
    return (
        CANDIDATE_FEATURES, TARGET, iv_df, iv_label, iv_summary,
        woe_tables, IV_LABELS, check_monotonicity,
    )


# ────────────────────────────────────────────────────────────────────────────
# CELL 08 — Interactive: inspect WOE chart for any feature
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(CANDIDATE_FEATURES, mo):
    feat_selector = mo.ui.dropdown(
        options=CANDIDATE_FEATURES,
        value=CANDIDATE_FEATURES[0],
        label="Inspect feature"
    )
    mo.vstack([
        mo.md("## 🔬 Section 4 — WOE Profile Inspector (interactive)"),
        feat_selector
    ])
    return (feat_selector,)


@app.cell
def _(feat_selector, mo, plt, woe_tables, PAL):
    feat = feat_selector.value
    if feat not in woe_tables:
        mo.stop(True, mo.callout(mo.md(f"No WOE table for `{feat}`"), kind="warn"))

    tbl = woe_tables[feat].copy()
    tbl["bin_label"] = tbl["bin"].astype(str)
    x   = range(len(tbl))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10),
                                         facecolor="white", sharex=True)

    # Event rate
    ax1.bar(x, tbl["event_rate"], color=PAL["teal2"], edgecolor="white", alpha=0.85)
    ax1.set_ylabel("Default Rate", fontsize=9)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_title(f"{feat} — Default Rate by bin", fontweight="bold")

    # WOE
    colors_woe = [PAL["teal"] if w >= 0 else PAL["purple"] for w in tbl["woe"]]
    ax2.bar(x, tbl["woe"], color=colors_woe, edgecolor="white", alpha=0.85)
    ax2.axhline(0, color=PAL["gray"], linewidth=1)
    ax2.set_ylabel("WOE", fontsize=9)
    ax2.set_title(f"{feat} — Weight of Evidence", fontweight="bold")

    # IV per bin
    ax3.bar(x, tbl["iv_bin"], color=PAL["gold"], edgecolor="white", alpha=0.85)
    ax3.set_ylabel("IV (bin)", fontsize=9)
    ax3.set_title(
        f"{feat} — IV per bin | Total IV = {tbl['iv_bin'].sum():.4f}",
        fontweight="bold"
    )
    ax3.set_xticks(list(x))
    ax3.set_xticklabels(tbl["bin_label"].tolist(), rotation=45, ha="right", fontsize=7)

    plt.tight_layout()
    mo.mpl.interactive(fig)
    return ax1, ax2, ax3, colors_woe, feat, fig, tbl, x


# ────────────────────────────────────────────────────────────────────────────
# CELL 09 — Coarse classing: PAY_1 (reference implementation, manual override)
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(mo, np, pd):
    """
    Coarse classing enforces monotonic WOE — mandatory for scorecard interpretability.
    PAY_1 is the strongest predictor; we implement the full coarse-class table here
    as a reference. Other features follow the same pattern.

    Coarse bins for PAY_1 (repayment status most recent month):
      Group A: {-2, -1}  → paid in full / no balance  → lowest risk
      Group B: {0}        → revolving credit            → moderate risk
      Group C: {1}        → 1 month delay              → elevated risk
      Group D: {2}        → 2 months delay             → high risk
      Group E: {3+}       → 3+ months delay            → very high risk
    """

    PAY_COARSE_BINS = {
        "PAY_1": {
            "A_paid":       lambda x: x <= -1,
            "B_revolving":  lambda x: x == 0,
            "C_1mo_delay":  lambda x: x == 1,
            "D_2mo_delay":  lambda x: x == 2,
            "E_3plus_delay":lambda x: x >= 3,
        },
        "PAY_2": {
            "A_paid":       lambda x: x <= -1,
            "B_revolving":  lambda x: x == 0,
            "C_1mo_delay":  lambda x: x == 1,
            "D_2mo_delay":  lambda x: x == 2,
            "E_3plus_delay":lambda x: x >= 3,
        },
        "PAY_3": {
            "A_paid":       lambda x: x <= -1,
            "B_revolving":  lambda x: x == 0,
            "C_1mo_delay":  lambda x: x == 1,
            "D_2plus_delay":lambda x: x >= 2,
        },
        "MAX_DELAY": {
            "A_no_delay":   lambda x: x == 0,
            "B_1mo":        lambda x: x == 1,
            "C_2mo":        lambda x: x == 2,
            "D_3plus":      lambda x: x >= 3,
        },
    }

    def apply_coarse_bins(series: pd.Series, bin_dict: dict) -> pd.Series:
        """Map numeric values to named coarse bin labels."""
        result = pd.Series("UNCATEGORIZED", index=series.index)
        for label, condition in bin_dict.items():
            mask = condition(series)
            result[mask] = label
        return result

    mo.md(r"""
    ## 🗂️ Section 5 — Coarse Classing (Monotonic Enforcement)

    Fine classing gives 20 bins — too granular for a scorecard. Coarse classing
    merges adjacent bins to:
    1. **Enforce monotonic WOE** (regulatory interpretability requirement)
    2. **Ensure each bin has ≥ 5% of the sample** (statistical stability)
    3. **Align economic meaning** (e.g., 3–8 month delays are operationally equivalent)

    Coarse bins defined for: `PAY_1`, `PAY_2`, `PAY_3`, `MAX_DELAY`
    """)
    return PAY_COARSE_BINS, apply_coarse_bins


# ────────────────────────────────────────────────────────────────────────────
# CELL 10 — Compute coarse WOE tables + validate monotonicity
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(PAY_COARSE_BINS, apply_coarse_bins, check_monotonicity,
       mo, np, pd, plt, train_df, PAL):

    TARGET = "DEFAULT"
    coarse_woe = {}   # {feature: DataFrame}
    coarse_iv  = []

    def woe_from_groups(series_binned, target):
        df = pd.DataFrame({"bin": series_binned, "y": target})
        total_e  = df["y"].sum()
        total_ne = len(df) - total_e
        grp = df.groupby("bin").agg(n=("y","count"), n_e=("y","sum")).reset_index()
        grp["n_ne"] = grp["n"] - grp["n_e"]
        grp["pct"]  = grp["n"] / len(df)
        eps = 0.5
        grp["dist_e"]  = (grp["n_e"]  + eps) / (total_e  + eps)
        grp["dist_ne"] = (grp["n_ne"] + eps) / (total_ne + eps)
        grp["event_rate"] = grp["n_e"] / grp["n"]
        grp["woe"] = np.log(grp["dist_e"] / grp["dist_ne"])
        grp["iv_bin"] = (grp["dist_e"] - grp["dist_ne"]) * grp["woe"]
        return grp.sort_values("bin")

    for feat, bin_dict in PAY_COARSE_BINS.items():
        if feat not in train_df.columns:
            continue
        binned = apply_coarse_bins(train_df[feat], bin_dict)
        tbl = woe_from_groups(binned, train_df[TARGET])
        coarse_woe[feat] = tbl
        iv_total = tbl["iv_bin"].sum()
        mono = check_monotonicity(tbl["woe"])
        coarse_iv.append({"feature": feat, "coarse_IV": round(iv_total,4), "monotonicity": mono})

    coarse_iv_df = pd.DataFrame(coarse_iv)

    # Plot all coarse WOE profiles
    n_plots = len(coarse_woe)
    fig, axes = plt.subplots(2, max(2, n_plots//2 + n_plots%2),
                             figsize=(14, 8), facecolor="white")
    axes_flat = axes.flatten()

    for idx, (feat, tbl) in enumerate(coarse_woe.items()):
        ax = axes_flat[idx]
        colors = [PAL["teal"] if w >= 0 else PAL["purple"] for w in tbl["woe"]]
        ax.bar(range(len(tbl)), tbl["woe"], color=colors, edgecolor="white", alpha=0.85)
        ax.axhline(0, color=PAL["gray"], linewidth=1)
        ax2 = ax.twinx()
        ax2.plot(range(len(tbl)), tbl["event_rate"],
                 color=PAL["gold"], marker="o", linewidth=2)
        ax2.set_ylim(0, 0.8)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_xticks(range(len(tbl)))
        ax.set_xticklabels(tbl["bin"].tolist(), rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{feat} — Coarse WOE", fontweight="bold", fontsize=10)
        ax.set_ylabel("WOE", fontsize=8)
        ax2.set_ylabel("Default Rate", fontsize=8)

    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    plt.suptitle("Coarse Classing — WOE (bars) & Default Rate (line)",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()

    mo.vstack([
        mo.ui.table(coarse_iv_df),
        mo.mpl.interactive(fig)
    ])
    return (
        TARGET, ax, ax2, axes, axes_flat, axes_flat,
        coarse_iv, coarse_iv_df, coarse_woe, feat, fig, idx, n_plots,
        tbl, woe_from_groups,
    )


# ────────────────────────────────────────────────────────────────────────────
# CELL 11 — WOE transformation: build model-ready matrix
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(PAY_COARSE_BINS, apply_coarse_bins, coarse_woe,
       mo, np, pd, train_df, val_df, oot_df, woe_tables):

    TARGET = "DEFAULT"

    def build_woe_matrix(
        df_input: pd.DataFrame,
        woe_tables_fine: dict,
        coarse_tables: dict,
        coarse_bins: dict,
        target_col: str = "DEFAULT",
    ) -> pd.DataFrame:
        """
        Applies WOE transformation to a dataframe.
        - Coarse-binned features (PAY) use coarse WOE lookup
        - Remaining features use fine-classing WOE (midpoint-based interpolation)
        """
        result = pd.DataFrame(index=df_input.index)

        for feat in woe_tables_fine.keys():
            if feat not in df_input.columns:
                continue

            if feat in coarse_tables and feat in coarse_bins:
                # Coarse WOE
                binned = apply_coarse_bins(df_input[feat], coarse_bins[feat])
                woe_map = coarse_tables[feat].set_index("bin")["woe"].to_dict()
                result[f"WOE_{feat}"] = binned.map(woe_map).fillna(0)
            else:
                # Fine WOE: assign bin via cut boundaries, then map
                fine_tbl = woe_tables_fine[feat]
                # Extract cut intervals
                intervals = fine_tbl["bin"].tolist()
                woe_map   = dict(zip(intervals, fine_tbl["woe"]))
                # Use pd.cut with same bins (extract from interval objects)
                try:
                    breaks = sorted(set(
                        [i.left for i in intervals] + [intervals[-1].right]
                    ))
                    breaks[0]  = -np.inf
                    breaks[-1] =  np.inf
                    binned_fine = pd.cut(df_input[feat], bins=breaks,
                                         labels=False, include_lowest=True)
                    fine_woe_arr = fine_tbl["woe"].values
                    result[f"WOE_{feat}"] = (
                        binned_fine.map(lambda i: fine_woe_arr[int(i)]
                                        if pd.notna(i) else 0)
                    )
                except Exception:
                    result[f"WOE_{feat}"] = 0.0

        result[target_col] = df_input[target_col].values
        return result

    woe_train = build_woe_matrix(train_df, woe_tables, coarse_woe,
                                  PAY_COARSE_BINS)
    woe_val   = build_woe_matrix(val_df,   woe_tables, coarse_woe,
                                  PAY_COARSE_BINS)
    woe_oot   = build_woe_matrix(oot_df,   woe_tables, coarse_woe,
                                  PAY_COARSE_BINS)

    n_woe_feats = sum(1 for c in woe_train.columns if c.startswith("WOE_"))

    mo.vstack([
        mo.md("## 🔄 Section 6 — WOE Transformation Matrix"),
        mo.callout(mo.md(f"""
        | Split | Rows | WOE Features |
        |---|---|---|
        | train | {len(woe_train):,} | {n_woe_feats} |
        | val   | {len(woe_val):,}   | {n_woe_feats} |
        | oot   | {len(woe_oot):,}   | {n_woe_feats} |

        > WOE bins fitted on **train only**. Val and OOT are transformed,
        > never influence the encoding. This prevents data leakage.
        """), kind="success"),
        mo.md("### Sample WOE matrix (train, first 5 rows)"),
        mo.ui.table(woe_train.head())
    ])
    return (
        TARGET, build_woe_matrix, n_woe_feats,
        woe_oot, woe_train, woe_val,
    )


# ────────────────────────────────────────────────────────────────────────────
# CELL 12 — Scorecard PDO calibration scaffold
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(mo, np):
    mo.md(r"""
    ## 🎯 Section 7 — Scorecard Calibration (PDO Method)

    The scorecard converts log-odds from the logistic regression into integer
    points per characteristic. This is the industry standard for retail credit.

    ### Calibration equations

    Let **n** = number of characteristics in the model.

    $$\text{Score} = \text{Offset} + \text{Factor} \times \ln(\text{Odds})$$

    Given anchor point $(\text{Score}_0, \text{Odds}_0)$ and PDO:

    $$\text{Factor} = \frac{\text{PDO}}{\ln(2)}$$

    $$\text{Offset} = \text{Score}_0 - \text{Factor} \times \ln(\text{Odds}_0)$$

    **Industry convention (Siddiqi, 2006):**
    - Base score = **600** at odds **1:50** (event rate ≈ 2%)
    - PDO = **20** (score decreases by 20 for each doubling of odds)

    This gives a range of ~300–850 (similar to FICO scale).
    The actual point allocation per bin will be computed in `03_model_logistic.py`
    after fitting the logistic regression.
    """)

    # PDO calibration constants
    BASE_SCORE = 600
    BASE_ODDS  = 50     # 1:50 odds → P(default) ≈ 2%
    PDO        = 20

    FACTOR = PDO / np.log(2)
    OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)

    mo.callout(mo.md(f"""
    | Parameter | Value |
    |---|---|
    | Base Score | `{BASE_SCORE}` |
    | Base Odds | `1:{BASE_ODDS}` |
    | PDO | `{PDO}` |
    | **Factor** | `{FACTOR:.4f}` |
    | **Offset** | `{OFFSET:.4f}` |

    > **Formula:** Score = `{OFFSET:.2f}` + `{FACTOR:.2f}` × ln(Odds)  
    > Used in `03_model_logistic.py` to convert regression coefficients to score points.
    """), kind="info")
    return BASE_ODDS, BASE_SCORE, FACTOR, OFFSET, PDO


# ────────────────────────────────────────────────────────────────────────────
# CELL 13 — Export artifacts
# ────────────────────────────────────────────────────────────────────────────
@app.cell
def _(DATA_DIR, BASE_ODDS, BASE_SCORE, FACTOR, OFFSET, PDO,
       coarse_iv_df, iv_df, mo, pd, woe_train):

    # Merge fine + coarse IV summaries
    iv_export = iv_df[["feature","IV","bins","monotonicity","strength"]].copy()
    iv_export["coarse_IV"] = iv_export["feature"].map(
        coarse_iv_df.set_index("feature")["coarse_IV"].to_dict()
    )
    iv_export["pdo_factor"] = FACTOR
    iv_export["pdo_offset"] = OFFSET

    CSV_PATH = DATA_DIR / "woe_iv_summary.csv"
    iv_export.to_csv(CSV_PATH, index=False)

    # WOE matrix is already saved as parquets (train/val/oot)
    # — save WOE versions too
    woe_train.to_parquet(DATA_DIR / "woe_train.parquet", index=False)

    mo.vstack([
        mo.md("## 💾 Section 8 — Artifacts Exported"),
        mo.callout(mo.md(f"""
        | File | Description |
        |---|---|
        | `data/train.parquet` | Raw preprocessed train split |
        | `data/val.parquet` | Raw preprocessed val split |
        | `data/oot.parquet` | Raw preprocessed OOT split |
        | `data/woe_train.parquet` | WOE-transformed train matrix |
        | `data/woe_iv_summary.csv` | IV table + PDO calibration params |

        ### Next → `03_model_logistic.py`
        - Fit logistic regression on `woe_train`
        - Extract coefficients → scorecard points per bin
        - Evaluate: KS, Gini, AUC, CAP Ratio on val + OOT
        - PSI monitoring setup
        """), kind="success")
    ])
    return CSV_PATH, iv_export


if __name__ == "__main__":
    app.run()
