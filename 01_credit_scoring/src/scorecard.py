"""
Scorecard Scaling Module
=========================
Converts logistic regression log-odds into a numeric credit score
using the standard Points-to-Double-Odds (PDO) methodology.

Reference:
    Siddiqi, N. (2006). Credit Risk Scorecards. Wiley.
    Score = Offset + Factor * log-odds
    Factor = PDO / ln(2)
    Offset = base_score - Factor * ln(base_odds)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import Optional


class Scorecard:
    """
    Wraps a fitted LogisticRegression and scales predictions
    to a credit score range using PDO methodology.

    Parameters
    ----------
    model : fitted sklearn LogisticRegression
    pdo   : Points to Double the Odds (typically 20)
    base_score : score at base_odds (typically 600)
    base_odds  : good:bad ratio at base_score (typically 50)
    """

    def __init__(
        self,
        model: LogisticRegression,
        pdo: float = 20.0,
        base_score: float = 600.0,
        base_odds: float = 50.0,
    ):
        self.model = model
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds

        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of default (class 1)."""
        return self.model.predict_proba(X)[:, 1]

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return scaled credit score. Higher score = lower risk.
        Score = Offset + Factor * ln(p_good / p_bad)
        """
        p_default = self.predict_proba(X)
        p_good = 1 - p_default
        # Avoid log(0)
        p_good = np.clip(p_good, 1e-6, 1 - 1e-6)
        p_bad = 1 - p_good
        log_odds = np.log(p_good / p_bad)
        return self.offset + self.factor * log_odds

    def score_summary(self, X: pd.DataFrame, y_true: Optional[np.ndarray] = None) -> pd.DataFrame:
        """DataFrame with probability, score, and optional true label."""
        result = pd.DataFrame({
            "p_default": self.predict_proba(X),
            "credit_score": self.predict_score(X),
        })
        if y_true is not None:
            result["default"] = np.asarray(y_true)
        return result
