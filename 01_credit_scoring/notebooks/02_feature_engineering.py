# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo", "pandas", "numpy", "matplotlib", "seaborn", "pyarrow", "scikit-learn"]
# ///
"""
Case Study 01 — Credit Scoring | Feature Engineering
DAG: df_raw → iv_screen → selected_features → woe_encoded → processed_woe.parquet
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
    import seaborn as sns
    import marimo as mo

    warnings.filterwarnings("ignore")
    matplotlib.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    PALETTE = ["#01696f", "#964219", "#006494", "#437a22", "#7a39bb", "#d19900"]
    return Path, matplotlib, mo, np, pd, plt, sns, sys, warnings, PALETTE


@app.cell
def _(Path, mo):
    REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
    DATA_DIR   = REPO_ROOT / "01_credit_scoring" / "data"
    REPORT_DIR = REPO_ROOT / "01_credit_scoring" / "reports"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Dynamic sys.path injection so utils is always found
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from utils.preprocessing import woe_binning, iv_summary
    mo.md(f"✅ Repo root: `{REPO_ROOT}` | utils loaded")


@app.cell
def _(DATA_DIR, mo, pd):
    DATA_PATH = DATA_DIR / "credit_card_default.parquet"
    if not DATA_PATH.exists():
        mo.stop(True, mo.md("❌ Run `bash 01_credit_scoring/data/download.sh` first"))
    df = pd.read_parquet(DATA_PATH)
    mo.md(f"✅ Loaded: `{len(df):,} rows × {df.shape[1]} cols` | Default rate: `{df['DEFAULT'].mean():.2%}`")


@app.cell
def _(df, iv_summary, mo):
    iv_df = iv_summary(df, "DEFAULT")
    IV_THRESHOLD = 0.02
    selected = iv_df[iv_df["iv"] >= IV_THRESHOLD]["feature"].tolist()
    mo.vstack([
        mo.md(f"### IV Screen (threshold = {IV_THRESHOLD})\n"
              f"**{len(selected)}/{len(iv_df)} features** passed IV ≥ {IV_THRESHOLD}"),
        mo.ui.table(iv_df, selection=None, label="IV Summary"),
    ])


@app.cell
def _(PALETTE, REPORT_DIR, iv_df, plt):
    IV_COLORS = {
        "Strong":  PALETTE[0],
        "Medium":  PALETTE[2],
        "Weak":    PALETTE[5],
        "Useless": "#bab9b4",
    }
    colors_bar = [IV_COLORS.get(s, "#bab9b4") for s in iv_df["strength"]]
    fig, ax = plt.subplots(figsize=(10, max(6, len(iv_df) * 0.35)))
    ax.barh(iv_df["feature"], iv_df["iv"], color=colors_bar, alpha=0.88, edgecolor="white")
    ax.axvline(0.02, color="#bab9b4",  linestyle="--", linewidth=1.0, label="Useless (0.02)")
    ax.axvline(0.10, color=PALETTE[5], linestyle="--", linewidth=1.0, label="Weak (0.10)")
    ax.axvline(0.30, color=PALETTE[2], linestyle="--", linewidth=1.0, label="Medium (0.30)")
    ax.axvline(0.50, color=PALETTE[0], linestyle="--", linewidth=1.0, label="Strong (0.50)")
    ax.set_xlabel("Information Value (IV)", fontsize=11)
    ax.set_title("Feature IV Screen — Predictive Power", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "fig07_iv_screen.png", dpi=150, bbox_inches="tight")
    fig


@app.cell
def _(DATA_DIR, df, mo, pd, selected, woe_binning):
    # WOE encode all selected features
    woe_frames = []
    for feat in selected:
        try:
            woe_tbl = woe_binning(df, feat, "DEFAULT")
            # Build WOE mapping: original value → WOE score
            bin_col  = pd.cut(df[feat], bins=10, duplicates="drop", labels=False)
            woe_map  = dict(zip(woe_tbl.index, woe_tbl["woe"]))
            woe_vals = bin_col.map(woe_map).fillna(0)
            woe_frames.append(woe_vals.rename(f"{feat}_WOE"))
        except Exception:
            pass

    df_woe = pd.concat([df[["DEFAULT"]], *woe_frames], axis=1)
    out_path = DATA_DIR / "processed_woe.parquet"
    df_woe.to_parquet(out_path, index=False)

    mo.md(f"""
    ### WOE Encoding Complete
    - Features encoded: **{len(woe_frames)}**
    - Output shape: **{df_woe.shape[0]:,} × {df_woe.shape[1]}**
    - Saved to: `{out_path.relative_to(out_path.parent.parent.parent)}`
    """)


if __name__ == "__main__":
    app.run()
