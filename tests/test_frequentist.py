"""
test_frequentist.py
-------------------
Tests for FrequentistABTest in src/ab_frequentist.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.ab_frequentist import FrequentistABTest


@pytest.fixture(scope="module")
def tester():
    return FrequentistABTest(significance_level=0.05, n_bootstrap=500)


# 1. Large effect – significant
def test_proportion_significant_large_effect(tester):
    result = tester.analyze_proportion(
        control_successes=1000,
        control_n=5000,
        treatment_successes=1500,
        treatment_n=5000,
        metric_name="conversion_rate",
        experiment_id="test-001",
    )
    assert result.is_significant, "Large effect should be significant"
    assert result.p_value < 0.05


# 2. Null scenario – not significant
def test_proportion_not_significant_null(tester):
    result = tester.analyze_proportion(
        control_successes=500,
        control_n=5000,
        treatment_successes=510,
        treatment_n=5000,
        metric_name="conversion_rate",
        experiment_id="test-002",
    )
    assert not result.is_significant, "Near-null effect should not be significant"
    assert result.p_value > 0.05


# 3. p-values uniform under null (type I error check)
def test_p_values_uniform_under_null():
    # Under the null, p-values from a two-proportion z-test on large samples
    # should be approximately uniform.  For discrete data at alpha=0.05, we
    # verify the empirical rejection rate is within [0.02, 0.10] — a sensible
    # range that avoids both anti-conservatism and excessive conservatism.
    # We do NOT use a KS test because discrete binomial data produces slight
    # departures from uniformity that KS can detect spuriously.
    rng = np.random.default_rng(42)
    tester = FrequentistABTest(significance_level=0.05, n_bootstrap=100)
    n = 2000  # larger n → better Normal approximation, more uniform p-values
    rate = 0.20  # moderate rate → good approximation
    p_values = []
    for _ in range(2000):
        c_succ = int(rng.binomial(n, rate))
        t_succ = int(rng.binomial(n, rate))
        result = tester.analyze_proportion(
            control_successes=c_succ,
            control_n=n,
            treatment_successes=t_succ,
            treatment_n=n,
            metric_name="metric",
            experiment_id="null-exp",
        )
        p_values.append(result.p_value)

    # Check type I error rate is close to nominal 5%
    type_i_rate = sum(p < 0.05 for p in p_values) / len(p_values)
    assert 0.02 <= type_i_rate <= 0.10, (
        f"Type I error rate {type_i_rate:.4f} should be close to 0.05 under null"
    )

    # Additionally verify p-value distribution is approximately uniform via
    # checking that the 25th, 50th, 75th percentiles are in reasonable ranges
    p_arr = sorted(p_values)
    q25 = p_arr[int(0.25 * len(p_arr))]
    q50 = p_arr[int(0.50 * len(p_arr))]
    q75 = p_arr[int(0.75 * len(p_arr))]
    assert 0.15 <= q25 <= 0.35, f"25th percentile {q25:.4f} far from 0.25"
    assert 0.40 <= q50 <= 0.60, f"50th percentile {q50:.4f} far from 0.50"
    assert 0.65 <= q75 <= 0.85, f"75th percentile {q75:.4f} far from 0.75"


# 4. Power detection at nominal power
def test_power_detection_at_nominal_power():
    from src.experiment_designer import ExperimentDesigner

    rng = np.random.default_rng(0)
    tester = FrequentistABTest(significance_level=0.05, n_bootstrap=100)
    designer = ExperimentDesigner(significance_level=0.05, power=0.80)

    baseline = 0.10
    mde = 0.03
    ss = designer.calculate_sample_size_proportion(
        baseline_rate=baseline, mde=mde, num_variants=2
    )
    n = ss.sample_size_per_variant

    detections = 0
    n_sims = 500
    for _ in range(n_sims):
        c_succ = int(rng.binomial(n, baseline))
        t_succ = int(rng.binomial(n, baseline + mde))
        result = tester.analyze_proportion(
            control_successes=c_succ,
            control_n=n,
            treatment_successes=t_succ,
            treatment_n=n,
            metric_name="metric",
            experiment_id="power-check",
        )
        if result.is_significant:
            detections += 1

    detection_rate = detections / n_sims
    assert detection_rate >= 0.70, (
        f"Detection rate {detection_rate:.2f} < 0.70 (conservative check for 80% power)"
    )


# 5. Relative lift calculation
def test_relative_lift_calculation(tester):
    result = tester.analyze_proportion(
        control_successes=200,
        control_n=1000,
        treatment_successes=240,
        treatment_n=1000,
        metric_name="conversion_rate",
        experiment_id="lift-test",
    )
    control_rate = 200 / 1000
    treatment_rate = 240 / 1000
    expected_rel_lift = (treatment_rate - control_rate) / control_rate
    assert result.relative_lift == pytest.approx(expected_rel_lift, rel=1e-6)
    assert result.absolute_lift == pytest.approx(treatment_rate - control_rate, rel=1e-6)


# 6. 95% CI contains true effect ~95% of the time
def test_ci_contains_true_effect():
    rng = np.random.default_rng(7)
    tester = FrequentistABTest(significance_level=0.05, n_bootstrap=100)
    true_effect = 0.03
    baseline = 0.10
    n = 2000
    coverage_count = 0
    n_sims = 200
    for _ in range(n_sims):
        c_succ = int(rng.binomial(n, baseline))
        t_succ = int(rng.binomial(n, baseline + true_effect))
        result = tester.analyze_proportion(
            control_successes=c_succ,
            control_n=n,
            treatment_successes=t_succ,
            treatment_n=n,
            metric_name="metric",
            experiment_id="ci-coverage",
        )
        lo, hi = result.confidence_interval_absolute
        if lo <= true_effect <= hi:
            coverage_count += 1

    coverage = coverage_count / n_sims
    assert coverage >= 0.90, (
        f"CI coverage {coverage:.2f} < 0.90 for 95% CI"
    )


# 7. Continuous Welch's t-test significant
def test_continuous_welch_t_significant(tester):
    rng = np.random.default_rng(42)
    control_vals = rng.normal(loc=10.0, scale=2.0, size=200)
    treatment_vals = rng.normal(loc=12.0, scale=2.0, size=200)
    result = tester.analyze_continuous(
        control_values=control_vals,
        treatment_values=treatment_vals,
        metric_name="time_on_page",
        experiment_id="t-test-001",
    )
    assert result.is_significant, "Large continuous effect should be significant"
    assert result.p_value < 0.05
    assert result.test_type == "welch_t"


# 8. Bootstrap CI has reasonable width
def test_continuous_bootstrap_ci_width(tester):
    rng = np.random.default_rng(42)
    control_vals = rng.normal(loc=10.0, scale=2.0, size=300)
    treatment_vals = rng.normal(loc=10.5, scale=2.0, size=300)
    result = tester.analyze_continuous(
        control_values=control_vals,
        treatment_values=treatment_vals,
        metric_name="time_on_page",
        experiment_id="boot-ci-test",
    )
    lo, hi = result.confidence_interval_absolute
    width = hi - lo
    assert width > 0, "CI width must be positive"
    assert width < 5.0, f"CI width {width:.4f} seems unreasonably large"


# 9. Guardrail degradation is flagged
def test_guardrail_degradation_flagged(tester):
    # unsubscribe_rate: control=0.01, treatment=0.025 → >10% relative increase
    control_data = {"unsubscribe_rate": {"successes": 100, "n": 10000}}
    treatment_data = {"unsubscribe_rate": {"successes": 250, "n": 10000}}
    results = tester.run_guardrail_checks(
        control_data=control_data,
        treatment_data=treatment_data,
        guardrail_metrics=["unsubscribe_rate"],
        metric_type_map={"unsubscribe_rate": "proportion"},
    )
    assert len(results) == 1
    assert results[0]["is_degraded"], (
        "Unsubscribe rate increased by 150% – should be flagged as degraded"
    )


# 10. Stable guardrail metric not flagged
def test_guardrail_stable_not_flagged(tester):
    control_data = {"unsubscribe_rate": {"successes": 100, "n": 10000}}
    treatment_data = {"unsubscribe_rate": {"successes": 100, "n": 10000}}
    results = tester.run_guardrail_checks(
        control_data=control_data,
        treatment_data=treatment_data,
        guardrail_metrics=["unsubscribe_rate"],
        metric_type_map={"unsubscribe_rate": "proportion"},
    )
    assert len(results) == 1
    assert not results[0]["is_degraded"], (
        "Identical guardrail metric should not be flagged"
    )


# 11. analyze() processes a DataFrame
def test_analyze_method_processes_dataframe(tester):
    rng = np.random.default_rng(42)
    n = 500
    control_data = pd.DataFrame({
        "variant": ["control"] * n,
        "conversion_rate": rng.binomial(1, 0.10, size=n).astype(float),
    })
    treatment_data = pd.DataFrame({
        "variant": ["treatment"] * n,
        "conversion_rate": rng.binomial(1, 0.15, size=n).astype(float),
    })
    df = pd.concat([control_data, treatment_data], ignore_index=True)

    result_dict = tester.analyze(
        experiment_data=df,
        variants=["control", "treatment"],
        metrics=["conversion_rate"],
        guardrail_metrics=[],
        experiment_id="analyze-test",
        control_variant="control",
    )

    assert "results" in result_dict
    assert "conversion_rate" in result_dict["results"]
    r = result_dict["results"]["conversion_rate"]
    assert hasattr(r, "p_value")
    assert hasattr(r, "is_significant")
