"""
Feature Engineering for Credit Risk Models
===========================================
Weight of Evidence (WOE), Information Value (IV), binning utilities.
"""

import numpy as np
import pandas as pd
from typing import Optional


def woe_binning(
    df: pd.DataFrame,
    feature: str,
    target: str,
    bins: int = 10,
    min_bin_pct: float = 0.05,
    eps: float = 1e-4,
) -> pd.DataFrame:
    """
    Compute WOE and IV for a numeric feature against a binary target.

    WOE_i = ln(Distribution_Bad_i / Distribution_Good_i)
    IV     = sum[(Distribution_Bad_i - Distribution_Good_i) * WOE_i]

    IV < 0.02  → Useless predictor
    IV 0.02–0.10 → Weak
    IV 0.10–0.30 → Medium
    IV > 0.30  → Strong

    Parameters
    ----------
    df      : DataFrame with feature and target columns
    feature : name of the numeric predictor
    target  : binary target (1 = bad/default, 0 = good)
    bins    : number of quantile bins
    min_bin_pct : minimum fraction of observations per bin (merges small bins)

    Returns
    -------
    pd.DataFrame with columns: bin, n_obs, n_bad, n_good, pct_bad, pct_good, woe, iv_bin
    """
    total_bad = df[target].sum()
    total_good = len(df) - total_bad

    df_tmp = df[[feature, target]].copy().dropna(subset=[feature])
    df_tmp["bin"] = pd.qcut(df_tmp[feature], q=bins, duplicates="drop")

    grouped = (
        df_tmp.groupby("bin")[target]
        .agg(n_obs="count", n_bad="sum")
        .reset_index()
    )
    grouped["n_good"] = grouped["n_obs"] - grouped["n_bad"]
    grouped["pct_bad"] = np.clip(grouped["n_bad"] / total_bad, eps, None)
    grouped["pct_good"] = np.clip(grouped["n_good"] / total_good, eps, None)
    grouped["woe"] = np.log(grouped["pct_bad"] / grouped["pct_good"])
    grouped["iv_bin"] = (grouped["pct_bad"] - grouped["pct_good"]) * grouped["woe"]

    return grouped


def iv_summary(df: pd.DataFrame, target: str, bins: int = 10) -> pd.DataFrame:
    """
    Compute IV for all numeric columns in a DataFrame.
    Useful for quick feature selection screening.

    Returns
    -------
    pd.DataFrame sorted by IV descending, with columns: feature, iv, strength
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)

    records = []
    for col in numeric_cols:
        try:
            woe_df = woe_binning(df, col, target, bins=bins)
            iv = woe_df["iv_bin"].sum()
            records.append({"feature": col, "iv": iv})
        except Exception:
            records.append({"feature": col, "iv": np.nan})

    result = pd.DataFrame(records).sort_values("iv", ascending=False).reset_index(drop=True)

    def _strength(iv):
        if iv < 0.02:
            return "Useless"
        elif iv < 0.10:
            return "Weak"
        elif iv < 0.30:
            return "Medium"
        else:
            return "Strong"

    result["strength"] = result["iv"].apply(_strength)
    return result
