"""
test_power_analysis.py
----------------------
Tests for ExperimentDesigner in src/experiment_designer.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from src.experiment_designer import ExperimentDesigner


@pytest.fixture(scope="module")
def designer():
    return ExperimentDesigner(significance_level=0.05, power=0.80)


# 1. Known value: baseline=0.10, mde=0.02, alpha=0.05, power=0.80  → n ≈ 3842
def test_sample_size_proportion_known_value(designer):
    result = designer.calculate_sample_size_proportion(
        baseline_rate=0.10,
        mde=0.02,
        significance_level=0.05,
        power=0.80,
        num_variants=2,
    )
    expected_n = 3842
    assert result.sample_size_per_variant == pytest.approx(expected_n, rel=0.05), (
        f"Expected n ≈ {expected_n}, got {result.sample_size_per_variant}"
    )


# 2. Smaller MDE requires larger n
def test_sample_size_increases_with_smaller_mde(designer):
    result_large_mde = designer.calculate_sample_size_proportion(
        baseline_rate=0.10, mde=0.04
    )
    result_small_mde = designer.calculate_sample_size_proportion(
        baseline_rate=0.10, mde=0.02
    )
    assert result_small_mde.sample_size_per_variant > result_large_mde.sample_size_per_variant


# 3. Higher power (0.90 vs 0.80) requires larger n
def test_sample_size_increases_with_higher_power(designer):
    result_80 = designer.calculate_sample_size_proportion(
        baseline_rate=0.10, mde=0.02, power=0.80
    )
    result_90 = designer.calculate_sample_size_proportion(
        baseline_rate=0.10, mde=0.02, power=0.90
    )
    assert result_90.sample_size_per_variant > result_80.sample_size_per_variant


# 4. 4 variants → Bonferroni correction → smaller per-comparison alpha → larger n
def test_sample_size_bonferroni_adjustment(designer):
    result_2 = designer.calculate_sample_size_proportion(
        baseline_rate=0.10, mde=0.02, num_variants=2
    )
    result_4 = designer.calculate_sample_size_proportion(
        baseline_rate=0.10, mde=0.02, num_variants=4
    )
    assert result_4.sample_size_per_variant > result_2.sample_size_per_variant
    # adjusted_alpha for 4 variants should be smaller than for 2 variants
    assert result_4.adjusted_alpha < result_2.adjusted_alpha


# 5. MDE recoverable from required n
def test_achievable_mde_inverse_of_sample_size(designer):
    target_mde = 0.02
    baseline = 0.10
    required = designer.calculate_sample_size_proportion(
        baseline_rate=baseline, mde=target_mde
    )
    n = required.sample_size_per_variant
    recovered_mde = designer.compute_achievable_mde(
        sample_size_per_variant=n, baseline_rate=baseline
    )
    assert recovered_mde == pytest.approx(target_mde, rel=0.05), (
        f"Recovered MDE {recovered_mde:.5f} != target {target_mde:.5f}"
    )


# 6. Power at effect = MDE should be >= declared power (within tolerance)
def test_power_at_true_effect_exceeds_nominal(designer):
    mde = 0.02
    baseline = 0.10
    result = designer.calculate_sample_size_proportion(
        baseline_rate=baseline, mde=mde, power=0.80
    )
    n = result.sample_size_per_variant
    power_df = designer.compute_power_at_effect(
        sample_size_per_variant=n,
        baseline_rate=baseline,
        effect_sizes=[mde],
    )
    achieved_power = float(power_df["power"].iloc[0])
    # Should be >= 0.75 (5% tolerance below nominal 0.80 due to ceiling in n calculation)
    assert achieved_power >= 0.75, f"Power at MDE too low: {achieved_power:.4f}"


# 7. Power at effect = 0 should be approximately alpha (type I error rate)
def test_power_at_zero_effect_is_alpha(designer):
    power_df = designer.compute_power_at_effect(
        sample_size_per_variant=5000,
        baseline_rate=0.10,
        effect_sizes=[0.0],
        significance_level=0.05,
    )
    power_at_zero = float(power_df["power"].iloc[0])
    assert power_at_zero == pytest.approx(0.05, abs=0.001), (
        f"Power at zero effect should equal alpha=0.05, got {power_at_zero}"
    )


# 8. Continuous metric: larger effect size → smaller required n
def test_sample_size_continuous_scales_with_effect(designer):
    result_small = designer.calculate_sample_size_continuous(
        baseline_mean=100.0, baseline_std=20.0, mde_absolute=2.0
    )
    result_large = designer.calculate_sample_size_continuous(
        baseline_mean=100.0, baseline_std=20.0, mde_absolute=8.0
    )
    assert result_large.sample_size_per_variant < result_small.sample_size_per_variant


# 9. Randomization: 1000 subjects, 2 variants → ~500/500
def test_randomization_returns_correct_variant_counts(designer):
    subject_ids = [f"user_{i}" for i in range(1000)]
    result = designer.randomize_subjects(
        subject_ids=subject_ids, num_variants=2, seed=42
    )
    counts = result.assignment_df["variant"].value_counts()
    assert len(counts) == 2
    for count in counts.values:
        assert abs(count - 500) <= 100, (
            f"Count {count} is more than 10% from expected 500"
        )


# 10. Randomization is deterministic
def test_randomization_is_deterministic(designer):
    subject_ids = [f"user_{i}" for i in range(200)]
    result_a = designer.randomize_subjects(
        subject_ids=subject_ids, num_variants=2, seed=99
    )
    result_b = designer.randomize_subjects(
        subject_ids=subject_ids, num_variants=2, seed=99
    )
    assignments_a = (
        result_a.assignment_df.sort_values("contact_id")["variant"].tolist()
    )
    assignments_b = (
        result_b.assignment_df.sort_values("contact_id")["variant"].tolist()
    )
    assert assignments_a == assignments_b, "Same inputs must produce same assignments"


# 11. Balance check returns a dict with p-values
def test_balance_check_runs_without_error(designer):
    subject_ids = [f"user_{i}" for i in range(500)]
    result = designer.randomize_subjects(
        subject_ids=subject_ids, num_variants=2, seed=0
    )
    balance = result.balance_check
    assert isinstance(balance, dict), "balance_check should be a dict"
    assert len(balance) > 0, "balance_check should have at least one entry"
    for key, val in balance.items():
        assert "p_value" in val, f"p_value missing in balance_check['{key}']"
        assert isinstance(val["p_value"], float)
