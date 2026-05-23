"""
Credit Risk Validation Metrics
================================
KS Statistic, Gini Coefficient, AUC, PSI, CAP Ratio.
All functions accept numpy arrays or pandas Series.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic: maximum separation between
    cumulative distributions of good (0) and bad (1) accounts.

    Returns
    -------
    float : KS value in [0, 1]. Industry threshold: > 0.30
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    df = pd.DataFrame({"score": y_score, "label": y_true})
    df_sorted = df.sort_values("score", ascending=False).reset_index(drop=True)

    total_bad = y_true.sum()
    total_good = len(y_true) - total_bad

    df_sorted["cum_bad"] = (df_sorted["label"] == 1).cumsum() / total_bad
    df_sorted["cum_good"] = (df_sorted["label"] == 0).cumsum() / total_good
    df_sorted["ks"] = np.abs(df_sorted["cum_bad"] - df_sorted["cum_good"])

    return df_sorted["ks"].max()


def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Gini coefficient derived from AUC-ROC: Gini = 2 * AUC - 1.

    Returns
    -------
    float : Gini in [-1, 1]. Industry threshold: > 0.40
    """
    auc = roc_auc_score(y_true, y_score)
    return 2 * auc - 1


def psi(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10,
    eps: float = 1e-4,
) -> float:
    """
    Population Stability Index (PSI).
    Measures distributional shift between training and scoring populations.

    PSI < 0.10 → Stable
    PSI 0.10–0.25 → Minor shift, monitor
    PSI > 0.25 → Major shift, recalibrate model

    Parameters
    ----------
    expected : array-like, scores from training/reference period
    actual   : array-like, scores from current period
    buckets  : number of quantile bins (default 10 = deciles)
    eps      : small constant to avoid log(0)

    Returns
    -------
    float : PSI value
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)

    def _bucket_pcts(arr):
        counts, _ = np.histogram(arr, bins=breakpoints)
        pcts = counts / len(arr)
        return np.clip(pcts, eps, None)

    exp_pct = _bucket_pcts(expected)
    act_pct = _bucket_pcts(actual)

    # Align lengths after np.unique may reduce bins
    min_len = min(len(exp_pct), len(act_pct))
    exp_pct, act_pct = exp_pct[:min_len], act_pct[:min_len]

    psi_value = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi_value)


def cap_ratio(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    CAP Ratio (Accuracy Ratio): area under CAP curve relative to perfect model.

    Returns
    -------
    float : CAP ratio in [0, 1]. Industry threshold: > 0.60
    """
    n = len(y_true)
    n_bad = y_true.sum()

    sorted_idx = np.argsort(y_score)[::-1]
    sorted_labels = y_true[sorted_idx]

    cum_bad = np.cumsum(sorted_labels) / n_bad
    cum_total = np.arange(1, n + 1) / n

    area_model = np.trapz(cum_bad, cum_total)
    area_perfect = 1 - (n_bad / (2 * n))
    area_random = 0.5

    return (area_model - area_random) / (area_perfect - area_random)
