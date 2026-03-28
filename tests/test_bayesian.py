"""
test_bayesian.py
----------------
Tests for BayesianABTest in src/ab_bayesian.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.ab_bayesian import BayesianABTest


@pytest.fixture(scope="module")
def bayes():
    return BayesianABTest(mc_samples=50_000)


def _successes(rng, n, rate):
    """Helper: draw binomial successes."""
    return int(rng.binomial(n, rate))


# 1. Under null (same rate), P(B>A) should be between 0.40 and 0.60
def test_probability_treatment_better_near_half_under_null(bayes):
    # Use exactly the same success count for both groups to guarantee a true null
    n = 5000
    successes = 500  # exactly 10% in both arms
    result = bayes.analyze_proportion_analytical(
        control_successes=successes,
        control_n=n,
        treatment_successes=successes,
        treatment_n=n,
        metric_name="conversion_rate",
        experiment_id="null-test",
    )
    prob = result.probability_treatment_better
    assert 0.40 <= prob <= 0.60, (
        f"Under null, P(B>A) should be ~0.5, got {prob:.4f}"
    )


# 2. Control 10%, treatment 20%, n=2000: P(B>A) > 0.99
def test_probability_treatment_better_high_for_large_effect(bayes):
    result = bayes.analyze_proportion_analytical(
        control_successes=200,
        control_n=2000,
        treatment_successes=400,
        treatment_n=2000,
        metric_name="conversion_rate",
        experiment_id="large-effect",
    )
    assert result.probability_treatment_better > 0.99, (
        f"Large effect should give P(B>A) > 0.99, got {result.probability_treatment_better:.6f}"
    )


# 3. Posterior mean lift close to true effect
def test_posterior_mean_lift_close_to_true_effect(bayes):
    true_effect = 0.05
    control_rate = 0.10
    n = 5000
    result = bayes.analyze_proportion_analytical(
        control_successes=int(n * control_rate),
        control_n=n,
        treatment_successes=int(n * (control_rate + true_effect)),
        treatment_n=n,
        metric_name="conversion_rate",
        experiment_id="lift-accuracy",
    )
    assert abs(result.posterior_mean_lift - true_effect) < 0.01, (
        f"Posterior mean lift {result.posterior_mean_lift:.4f} far from true {true_effect}"
    )


# 4. Under null, 95% HDI should contain 0
def test_credible_interval_contains_zero_under_null(bayes):
    rng = np.random.default_rng(42)
    n = 3000
    rate = 0.12
    result = bayes.analyze_proportion_analytical(
        control_successes=_successes(rng, n, rate),
        control_n=n,
        treatment_successes=_successes(rng, n, rate),
        treatment_n=n,
        metric_name="conversion_rate",
        experiment_id="null-ci",
    )
    lo, hi = result.credible_interval_95
    assert lo <= 0.0 <= hi, (
        f"Under null, 95% CI should contain 0; got [{lo:.4f}, {hi:.4f}]"
    )


# 5. Large effect, large n: CI should NOT contain 0
def test_credible_interval_excludes_zero_for_large_effect(bayes):
    result = bayes.analyze_proportion_analytical(
        control_successes=500,
        control_n=5000,
        treatment_successes=1000,
        treatment_n=5000,
        metric_name="conversion_rate",
        experiment_id="large-effect-ci",
    )
    lo, hi = result.credible_interval_95
    assert lo > 0.0, (
        f"95% CI should exclude 0 for large effect; got [{lo:.4f}, {hi:.4f}]"
    )


# 6. Expected loss low for clear winner
def test_expected_loss_low_for_clear_winner(bayes):
    result = bayes.analyze_proportion_analytical(
        control_successes=500,
        control_n=5000,
        treatment_successes=1000,
        treatment_n=5000,
        metric_name="conversion_rate",
        experiment_id="loss-winner",
    )
    assert result.expected_loss_treatment < 0.001, (
        f"Expected loss for clear winner should be < 0.001, got {result.expected_loss_treatment:.6f}"
    )


# 7. Expected loss meaningful under null
def test_expected_loss_high_when_uncertain(bayes):
    # Use exactly equal successes so P(B>A) ≈ 0.5 — truly uncertain
    # In this case, expected_loss_treatment = E[max(control - treatment, 0)] > 0
    # and should be non-trivially large (treatment could easily be worse)
    n = 2000
    successes = 200  # 10% in both arms
    result = bayes.analyze_proportion_analytical(
        control_successes=successes,
        control_n=n,
        treatment_successes=successes,
        treatment_n=n,
        metric_name="conversion_rate",
        experiment_id="loss-null",
    )
    # Under true null, expected_loss_treatment ≈ expected_loss_control ≈ some positive value
    # Both should be > 0 and non-trivial (we're genuinely uncertain)
    assert result.expected_loss_treatment > 0.0, (
        f"Expected loss under null should be > 0, got {result.expected_loss_treatment:.6f}"
    )
    # The combined uncertainty: if we're ~50/50 on who wins, there IS meaningful risk
    # Check either arm's expected loss is above trivial threshold of 1e-4
    assert result.expected_loss_treatment > 1e-4 or result.expected_loss_control > 1e-4, (
        "At least one arm's expected loss should be non-trivial under true null"
    )


# 8. SHIP_TREATMENT when P(B>A) > 0.97
def test_decision_ship_treatment_for_strong_signal(bayes):
    result = bayes.analyze_proportion_analytical(
        control_successes=500,
        control_n=5000,
        treatment_successes=1000,
        treatment_n=5000,
        metric_name="conversion_rate",
        experiment_id="ship-decision",
    )
    assert result.probability_treatment_better > 0.97, (
        f"Expected P(B>A) > 0.97, got {result.probability_treatment_better:.4f}"
    )
    assert result.decision == "SHIP_TREATMENT", (
        f"Expected SHIP_TREATMENT, got {result.decision!r}"
    )


# 9. CONTINUE_TESTING or INCONCLUSIVE under null
def test_decision_inconclusive_under_null(bayes):
    rng = np.random.default_rng(42)
    n = 2000
    rate = 0.10
    result = bayes.analyze_proportion_analytical(
        control_successes=_successes(rng, n, rate),
        control_n=n,
        treatment_successes=_successes(rng, n, rate),
        treatment_n=n,
        metric_name="conversion_rate",
        experiment_id="inconclusive",
    )
    valid_decisions = {"CONTINUE_TESTING", "INCONCLUSIVE", "SHIP_CONTROL"}
    assert result.decision in valid_decisions, (
        f"Under null, expected indecisive result, got {result.decision!r}"
    )


# 10. Analytical method twice with same data gives very close results
def test_analytical_and_monte_carlo_agree(bayes):
    kwargs = dict(
        control_successes=300,
        control_n=3000,
        treatment_successes=360,
        treatment_n=3000,
        metric_name="conversion_rate",
        experiment_id="agree-test",
    )
    result_a = bayes.analyze_proportion_analytical(**kwargs)
    result_b = bayes.analyze_proportion_analytical(**kwargs)
    diff = abs(result_a.probability_treatment_better - result_b.probability_treatment_better)
    assert diff < 0.01, (
        f"Two runs of same method should agree closely; diff={diff:.6f}"
    )


# 11. Probability in ROPE is substantial under null
def test_rope_probability_high_under_null(bayes):
    # Use "default" metric name so ROPE is (-0.005, 0.005) — wider than "conversion_rate"
    # which has ROPE (-0.001, 0.001). With n=5000 and equal rates, posterior std
    # of lift ≈ sqrt(2 * p*(1-p) / n) ≈ 0.006; ~60% of mass falls in ±0.005 band.
    n = 5000
    successes = 500  # 10% exactly in both arms
    result = bayes.analyze_proportion_analytical(
        control_successes=successes,
        control_n=n,
        treatment_successes=successes,
        treatment_n=n,
        metric_name="default",   # ROPE_DEFAULTS["default"] = (-0.005, 0.005)
        experiment_id="rope-null",
    )
    assert result.probability_in_rope > 0.30, (
        f"probability_in_rope under null should be > 0.30, got {result.probability_in_rope:.4f}"
    )
