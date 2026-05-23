"""
Standard Credit Risk Plots
===========================
CAP curve, ROC curve, score distribution.
Institutional style: white background, muted palette.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Optional

# Institutional color palette
COLOR_MODEL = "#01696f"   # Teal — model
COLOR_RANDOM = "#bab9b4"  # Gray — random
COLOR_PERFECT = "#964219" # Orange — perfect
COLOR_BAD = "#a12c7b"     # Maroon — bad accounts
COLOR_GOOD = "#437a22"    # Green — good accounts


def _base_style(ax: plt.Axes) -> None:
    """Apply clean institutional styling to an axes object."""
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#dcd9d5")
    ax.tick_params(colors="#7a7974", labelsize=9)
    ax.xaxis.label.set_color("#28251d")
    ax.yaxis.label.set_color("#28251d")
    ax.title.set_color("#28251d")


def plot_cap_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "CAP Curve",
    figsize: tuple = (6, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Cumulative Accuracy Profile (CAP) curve."""
    n = len(y_true)
    n_bad = y_true.sum()

    sorted_idx = np.argsort(y_score)[::-1]
    sorted_labels = np.asarray(y_true)[sorted_idx]

    cum_bad = np.concatenate([[0], np.cumsum(sorted_labels) / n_bad])
    cum_total = np.concatenate([[0], np.arange(1, n + 1) / n])

    fig, ax = plt.subplots(figsize=figsize, dpi=120, facecolor="white")
    ax.plot([0, 1], [0, 1], color=COLOR_RANDOM, lw=1.5, linestyle="--", label="Random model")
    ax.plot(
        [0, n_bad / n, 1], [0, 1, 1],
        color=COLOR_PERFECT, lw=1.5, linestyle=":", label="Perfect model"
    )
    ax.plot(cum_total, cum_bad, color=COLOR_MODEL, lw=2.0, label="Predictive model")
    ax.fill_between(cum_total, cum_bad, alpha=0.08, color=COLOR_MODEL)

    _base_style(ax)
    ax.set_xlabel("% of Population (sorted by score)")
    ax.set_ylabel("% of Bad Accounts Captured")
    ax.set_title(title, fontsize=11, fontweight="semibold")
    ax.legend(fontsize=8, framealpha=0.5)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "ROC Curve",
    figsize: tuple = (6, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=figsize, dpi=120, facecolor="white")
    ax.plot([0, 1], [0, 1], color=COLOR_RANDOM, lw=1.5, linestyle="--", label="Random")
    ax.plot(fpr, tpr, color=COLOR_MODEL, lw=2.0, label=f"Model (AUC = {auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.08, color=COLOR_MODEL)

    _base_style(ax)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=11, fontweight="semibold")
    ax.legend(fontsize=8, framealpha=0.5)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_score_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "Score Distribution by Class",
    figsize: tuple = (7, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Overlapping score histograms for good/bad accounts."""
    scores_bad = y_score[np.asarray(y_true) == 1]
    scores_good = y_score[np.asarray(y_true) == 0]

    fig, ax = plt.subplots(figsize=figsize, dpi=120, facecolor="white")
    ax.hist(scores_good, bins=40, alpha=0.55, color=COLOR_GOOD, label="Good (0)", density=True)
    ax.hist(scores_bad, bins=40, alpha=0.55, color=COLOR_BAD, label="Bad (1)", density=True)

    _base_style(ax)
    ax.set_xlabel("Predicted Probability of Default")
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=11, fontweight="semibold")
    ax.legend(fontsize=8, framealpha=0.5)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
