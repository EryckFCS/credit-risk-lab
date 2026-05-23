"""
Test suite for utils/preprocessing.py
Run with: uv run pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pandas as pd
import pytest
from utils.preprocessing import woe_binning, iv_summary


@pytest.fixture
def sample_df():
    """Synthetic credit dataset: 2000 obs, 22% default rate."""
    rng = np.random.default_rng(99)
    n = 2000
    age = rng.integers(21, 70, n)
    limit_bal = rng.integers(10_000, 800_000, n)
    pay_0 = rng.integers(-2, 9, n)

    # Default correlated with pay_0 and limit_bal
    logit = -2.5 + 0.3 * (pay_0 > 1).astype(float) - 0.001 * limit_bal / 10_000
    p_default = 1 / (1 + np.exp(-logit))
    default = (rng.uniform(0, 1, n) < p_default).astype(int)

    return pd.DataFrame({
        "age": age,
        "limit_bal": limit_bal,
        "pay_0": pay_0,
        "default": default,
    })


class TestWOEBinning:
    def test_returns_dataframe(self, sample_df):
        result = woe_binning(sample_df, "age", "default")
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self, sample_df):
        result = woe_binning(sample_df, "limit_bal", "default")
        required = {"bin", "n_obs", "n_bad", "n_good", "woe", "iv_bin"}
        assert required.issubset(set(result.columns))

    def test_iv_bin_sum_equals_iv(self, sample_df):
        result = woe_binning(sample_df, "pay_0", "default")
        iv_total = result["iv_bin"].sum()
        assert iv_total >= 0, "IV must be non-negative"

    def test_n_obs_sums_to_non_null_count(self, sample_df):
        result = woe_binning(sample_df, "age", "default")
        n_total = result["n_obs"].sum()
        expected = sample_df["age"].notna().sum()
        assert n_total == expected

    def test_strong_predictor_has_high_iv(self, sample_df):
        """pay_0 is the strongest predictor in the synthetic dataset."""
        result = woe_binning(sample_df, "pay_0", "default")
        iv = result["iv_bin"].sum()
        assert iv > 0.10, f"Strong predictor IV should be > 0.10, got {iv:.4f}"


class TestIVSummary:
    def test_returns_dataframe_sorted_by_iv(self, sample_df):
        result = iv_summary(sample_df, "default")
        assert isinstance(result, pd.DataFrame)
        assert result["iv"].is_monotonic_decreasing

    def test_strength_column_values(self, sample_df):
        result = iv_summary(sample_df, "default")
        valid = {"Useless", "Weak", "Medium", "Strong"}
        assert set(result["strength"].unique()).issubset(valid)

    def test_excludes_target_column(self, sample_df):
        result = iv_summary(sample_df, "default")
        assert "default" not in result["feature"].values

    def test_all_numeric_features_present(self, sample_df):
        result = iv_summary(sample_df, "default")
        expected_features = {"age", "limit_bal", "pay_0"}
        assert expected_features.issubset(set(result["feature"].values))
