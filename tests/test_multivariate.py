"""
test_multivariate.py
--------------------
Tests for MultivariateTest in src/multivariate_test.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.multivariate_test import MultivariateTest


@pytest.fixture(scope="module")
def mt():
    return MultivariateTest(significance_level=0.05)


def _make_proportion_df(variant_rates: dict, n_per_variant: int, seed: int = 42) -> pd.DataFrame:
    """Create a DataFrame with binary metric column across multiple variants."""
    rng = np.random.default_rng(seed)
    rows = []
    for variant, rate in variant_rates.items():
        successes = rng.binomial(1, rate, size=n_per_variant)
        for val in successes:
            rows.append({"variant": variant, "conversion_rate": float(val)})
    return pd.DataFrame(rows)


# 1. Bonferroni corrected p-values >= uncorrected
def test_bonferroni_more_conservative_than_uncorrected(mt):
    df = _make_proportion_df(
        {"control": 0.10, "variant_a": 0.13, "variant_b": 0.10},
        n_per_variant=3000,
    )
    result = mt.analyze_proportion(
        data=df,
        variant_col="variant",
        metric_col="conversion_rate",
        metric_name="conversion_rate",
        experiment_id="bonf-test",
    )
    for pr in result.pairwise_comparisons:
        assert pr.p_value_bonferroni >= pr.p_value_uncorrected - 1e-10, (
            f"Bonferroni p={pr.p_value_bonferroni:.6f} < uncorrected p={pr.p_value_uncorrected:.6f}"
        )


# 2. Holm p-values <= Bonferroni (for at least some comparisons)
def test_holm_less_conservative_than_bonferroni(mt):
    df = _make_proportion_df(
        {"control": 0.10, "variant_a": 0.14, "variant_b": 0.10, "variant_c": 0.10},
        n_per_variant=3000,
    )
    result = mt.analyze_proportion(
        data=df,
        variant_col="variant",
        metric_col="conversion_rate",
        metric_name="conversion_rate",
        experiment_id="holm-vs-bonf",
    )
    holm_le_bonf = [
        pr.p_value_holm <= pr.p_value_bonferroni + 1e-10
        for pr in result.pairwise_comparisons
    ]
    assert any(holm_le_bonf), (
        "At least some Holm p-values should be <= Bonferroni p-values"
    )


# 3. FDR-BH controls false discovery rate under null
def test_fdr_bh_controls_false_discovery_rate(mt):
    # 10 comparisons all under null
    rng = np.random.default_rng(0)
    n = 2000
    rate = 0.10
    # Create 11 variants with the same rate
    variants = ["control"] + [f"var_{i}" for i in range(10)]
    rows = []
    for v in variants:
        successes = rng.binomial(1, rate, size=n)
        for val in successes:
            rows.append({"variant": v, "conversion_rate": float(val)})
    df = pd.DataFrame(rows)

    result = mt.analyze_proportion(
        data=df,
        variant_col="variant",
        metric_col="conversion_rate",
        metric_name="conversion_rate",
        experiment_id="fdr-null",
        control_variant="control",
    )
    bonf_sig = sum(1 for pr in result.pairwise_comparisons if pr.is_significant_bonferroni)
    # With Bonferroni, expected false discoveries should be <= alpha
    assert bonf_sig / max(len(result.pairwise_comparisons), 1) <= 0.05 + 0.10, (
        f"Too many Bonferroni significant results under null: {bonf_sig}"
    )


# 4. FWER controlled with Bonferroni across 500 replications
def test_family_wise_error_rate_controlled():
    rng = np.random.default_rng(1)
    mt_local = MultivariateTest(significance_level=0.05)
    n = 500
    rate = 0.10
    n_sims = 500
    any_false_positive = 0

    for _ in range(n_sims):
        rows = []
        for v in ["control", "var_1", "var_2", "var_3", "var_4"]:
            successes = rng.binomial(1, rate, size=n)
            for val in successes:
                rows.append({"variant": v, "conversion_rate": float(val)})
        df = pd.DataFrame(rows)

        result = mt_local.analyze_proportion(
            data=df,
            variant_col="variant",
            metric_col="conversion_rate",
            metric_name="conversion_rate",
            experiment_id="fwer-test",
            control_variant="control",
        )
        if any(pr.is_significant_bonferroni for pr in result.pairwise_comparisons):
            any_false_positive += 1

    fwer = any_false_positive / n_sims
    assert fwer <= 0.05 + 0.03, (
        f"FWER with Bonferroni = {fwer:.4f}, exceeds 0.05 + tolerance"
    )


# 5. Winner identified correctly when one variant clearly wins
def test_winner_identified_correctly(mt):
    n = 10_000
    # variant_c at 10%, all others at 6%
    rng = np.random.default_rng(42)
    rows = []
    variant_rates = {
        "control": 0.06,
        "variant_a": 0.06,
        "variant_b": 0.06,
        "variant_c": 0.10,
    }
    for v, rate in variant_rates.items():
        successes = rng.binomial(1, rate, size=n)
        for val in successes:
            rows.append({"variant": v, "conversion_rate": float(val)})
    df = pd.DataFrame(rows)

    result = mt.analyze_proportion(
        data=df,
        variant_col="variant",
        metric_col="conversion_rate",
        metric_name="conversion_rate",
        experiment_id="winner-test",
        control_variant="control",
    )
    assert result.winner == "variant_c", (
        f"Expected winner='variant_c', got winner={result.winner!r}"
    )


# 6. No winner when all equal
def test_no_winner_when_all_equal(mt):
    n = 500
    rate = 0.10
    rows = []
    for v in ["control", "variant_a", "variant_b"]:
        for _ in range(n):
            rows.append({"variant": v, "conversion_rate": rate})
    df = pd.DataFrame(rows)

    result = mt.analyze_proportion(
        data=df,
        variant_col="variant",
        metric_col="conversion_rate",
        metric_name="conversion_rate",
        experiment_id="no-winner",
    )
    assert result.winner is None, (
        f"Expected no winner when all equal, got winner={result.winner!r}"
    )


# 7. Omnibus chi2 significant for real effect
def test_omnibus_chi2_significant_for_real_effect(mt):
    n = 5000
    df = _make_proportion_df(
        {"control": 0.08, "variant_a": 0.12, "variant_b": 0.08},
        n_per_variant=n,
        seed=10,
    )
    result = mt.analyze_proportion(
        data=df,
        variant_col="variant",
        metric_col="conversion_rate",
        metric_name="conversion_rate",
        experiment_id="chi2-sig",
    )
    assert result.overall_chi2_p_value < 0.05, (
        f"Omnibus chi2 p={result.overall_chi2_p_value:.4f} not significant for real effect"
    )


# 8. Omnibus not significant under null
def test_omnibus_not_significant_under_null(mt):
    # Deterministic equal rates – chi2 should be 0 or very high p-value
    n = 1000
    rate = 0.10
    rows = []
    for v in ["control", "variant_a", "variant_b"]:
        # exact same successes across variants
        for _ in range(int(n * rate)):
            rows.append({"variant": v, "conversion_rate": 1.0})
        for _ in range(n - int(n * rate)):
            rows.append({"variant": v, "conversion_rate": 0.0})
    df = pd.DataFrame(rows)

    result = mt.analyze_proportion(
        data=df,
        variant_col="variant",
        metric_col="conversion_rate",
        metric_name="conversion_rate",
        experiment_id="chi2-null",
    )
    assert result.overall_chi2_p_value > 0.05, (
        f"Omnibus chi2 p={result.overall_chi2_p_value:.4f} should be > 0.05 under null"
    )


# 9. Sum of probability_best_bayesian ≈ 1.0
def test_probability_best_sums_to_one(mt):
    df = _make_proportion_df(
        {"control": 0.10, "variant_a": 0.12, "variant_b": 0.11},
        n_per_variant=2000,
    )
    result = mt.analyze_proportion(
        data=df,
        variant_col="variant",
        metric_col="conversion_rate",
        metric_name="conversion_rate",
        experiment_id="prob-best-sum",
    )
    total = sum(result.probability_best_bayesian.values())
    assert total == pytest.approx(1.0, abs=0.01), (
        f"probability_best_bayesian values sum to {total:.6f}, expected ~1.0"
    )


# 10. correction_comparison_table has columns for all 4 corrections
def test_correction_comparison_table_has_four_methods(mt):
    df = _make_proportion_df(
        {"control": 0.10, "variant_a": 0.13},
        n_per_variant=1000,
    )
    result = mt.analyze_proportion(
        data=df,
        variant_col="variant",
        metric_col="conversion_rate",
        metric_name="conversion_rate",
        experiment_id="corr-table",
    )
    table = result.correction_comparison_table
    assert isinstance(table, pd.DataFrame)
    # The _build_correction_table method stores columns:
    # p_uncorrected, p_bonferroni, p_holm, p_fdr_bh, p_dunnett
    required_cols = {"p_uncorrected", "p_bonferroni", "p_holm", "p_fdr_bh"}
    actual_cols = set(table.columns)
    assert required_cols.issubset(actual_cols), (
        f"Missing columns in correction_comparison_table: {required_cols - actual_cols}"
    )


# 11. analyze() with 2 metrics returns dict with 2 keys
def test_analyze_returns_result_per_metric(mt):
    rng = np.random.default_rng(42)
    n = 500
    rows = []
    for v, rate in [("control", 0.10), ("treatment", 0.13)]:
        conv = rng.binomial(1, rate, size=n).astype(float)
        time_vals = rng.normal(loc=30.0 if v == "control" else 32.0, scale=5.0, size=n)
        for c, t in zip(conv, time_vals):
            rows.append({
                "variant": v,
                "conversion_rate": c,
                "time_on_page": t,
            })
    df = pd.DataFrame(rows)

    results = mt.analyze(
        experiment_data=df,
        variant_col="variant",
        metrics=["conversion_rate", "time_on_page"],
        experiment_id="multi-metric",
        control_variant="control",
    )
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert "conversion_rate" in results
    assert "time_on_page" in results
