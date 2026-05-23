"""
Test suite for utils/metrics.py
Run with: uv run pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest
from utils.metrics import ks_statistic, gini_coefficient, psi, cap_ratio


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def perfect_predictions():
    """Perfect classifier: score=1 for all bad, score=0 for all good."""
    rng = np.random.default_rng(42)
    n = 1000
    y_true = np.array([1] * 220 + [0] * 780)  # ~22% default rate
    y_score = np.where(y_true == 1, 0.95 + rng.uniform(0, 0.05, n), rng.uniform(0, 0.05, n))
    return y_true, y_score


@pytest.fixture
def random_predictions():
    """Random classifier: scores uncorrelated with labels."""
    rng = np.random.default_rng(0)
    n = 1000
    y_true = rng.integers(0, 2, n)
    y_score = rng.uniform(0, 1, n)
    return y_true, y_score


@pytest.fixture
def good_predictions():
    """Good classifier: AUC ~0.75, consistent with industry credit model."""
    rng = np.random.default_rng(7)
    n = 3000
    y_true = (rng.uniform(0, 1, n) < 0.22).astype(int)
    # Bad accounts get higher scores on average
    y_score = np.where(y_true == 1,
                       rng.beta(5, 2, n),
                       rng.beta(2, 5, n))
    return y_true, y_score


# ── KS Statistic ──────────────────────────────────────────────────────────────
class TestKSStatistic:
    def test_perfect_classifier_ks_near_1(self, perfect_predictions):
        y_true, y_score = perfect_predictions
        ks = ks_statistic(y_true, y_score)
        assert ks > 0.85, f"Perfect classifier KS should be > 0.85, got {ks:.4f}"

    def test_random_classifier_ks_near_0(self, random_predictions):
        y_true, y_score = random_predictions
        ks = ks_statistic(y_true, y_score)
        assert ks < 0.10, f"Random classifier KS should be < 0.10, got {ks:.4f}"

    def test_good_model_ks_above_threshold(self, good_predictions):
        y_true, y_score = good_predictions
        ks = ks_statistic(y_true, y_score)
        assert ks > 0.30, f"Good model KS should be > 0.30 (industry threshold), got {ks:.4f}"

    def test_ks_range(self, good_predictions):
        y_true, y_score = good_predictions
        ks = ks_statistic(y_true, y_score)
        assert 0.0 <= ks <= 1.0, f"KS must be in [0, 1], got {ks}"

    def test_accepts_pandas_series(self):
        import pandas as pd
        y_true = pd.Series([1, 0, 1, 0, 1, 0, 0, 0, 1, 0])
        y_score = pd.Series([0.9, 0.1, 0.8, 0.2, 0.85, 0.15, 0.3, 0.05, 0.75, 0.4])
        ks = ks_statistic(y_true, y_score)
        assert isinstance(ks, float)


# ── Gini Coefficient ───────────────────────────────────────────────────────────
class TestGiniCoefficient:
    def test_perfect_classifier_gini_near_1(self, perfect_predictions):
        y_true, y_score = perfect_predictions
        gini = gini_coefficient(y_true, y_score)
        assert gini > 0.85, f"Perfect classifier Gini should be > 0.85, got {gini:.4f}"

    def test_random_classifier_gini_near_0(self, random_predictions):
        y_true, y_score = random_predictions
        gini = gini_coefficient(y_true, y_score)
        assert abs(gini) < 0.10, f"Random Gini should be near 0, got {gini:.4f}"

    def test_good_model_gini_above_threshold(self, good_predictions):
        y_true, y_score = good_predictions
        gini = gini_coefficient(y_true, y_score)
        assert gini > 0.40, f"Good model Gini should be > 0.40, got {gini:.4f}"

    def test_gini_equals_2auc_minus_1(self, good_predictions):
        from sklearn.metrics import roc_auc_score
        y_true, y_score = good_predictions
        gini = gini_coefficient(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        assert abs(gini - (2 * auc - 1)) < 1e-10, "Gini must equal 2*AUC - 1"


# ── PSI ─────────────────────────────────────────────────────────────────────────
class TestPSI:
    def test_identical_distributions_psi_near_0(self):
        rng = np.random.default_rng(1)
        arr = rng.uniform(0, 1, 2000)
        result = psi(arr, arr)
        assert result < 0.01, f"Identical distributions PSI should be ~0, got {result:.4f}"

    def test_stable_distribution_psi_below_threshold(self):
        rng = np.random.default_rng(2)
        expected = rng.beta(2, 5, 3000)
        actual = rng.beta(2.1, 5.1, 3000)  # minor shift
        result = psi(expected, actual)
        assert result < 0.10, f"Stable PSI should be < 0.10, got {result:.4f}"

    def test_major_shift_psi_above_threshold(self):
        rng = np.random.default_rng(3)
        expected = rng.beta(2, 5, 3000)
        actual = rng.beta(5, 2, 3000)   # distribution flipped
        result = psi(expected, actual)
        assert result > 0.25, f"Major shift PSI should be > 0.25, got {result:.4f}"

    def test_psi_non_negative(self):
        rng = np.random.default_rng(4)
        a = rng.uniform(0, 1, 500)
        b = rng.uniform(0.2, 0.8, 500)
        result = psi(a, b)
        assert result >= 0.0, "PSI must be non-negative"


# ── CAP Ratio ─────────────────────────────────────────────────────────────────
class TestCAPRatio:
    def test_perfect_classifier_cap_near_1(self, perfect_predictions):
        y_true, y_score = perfect_predictions
        cap = cap_ratio(y_true, y_score)
        assert cap > 0.85, f"Perfect CAP should be > 0.85, got {cap:.4f}"

    def test_random_classifier_cap_near_0(self, random_predictions):
        y_true, y_score = random_predictions
        cap = cap_ratio(y_true, y_score)
        assert abs(cap) < 0.10, f"Random CAP should be near 0, got {cap:.4f}"

    def test_good_model_cap_above_threshold(self, good_predictions):
        y_true, y_score = good_predictions
        cap = cap_ratio(y_true, y_score)
        assert cap > 0.50, f"Good model CAP should be > 0.50, got {cap:.4f}"
