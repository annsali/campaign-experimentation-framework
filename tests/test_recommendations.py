"""
test_recommendations.py
-----------------------
Tests for RecommendationEngine in src/recommendation_engine.py.
"""

import sys
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.recommendation_engine import RecommendationEngine, ExperimentReport


# ─────────────────────────────────────────────────────────────────────────────
# Mock result helpers (duck-typed to mimic real dataclass fields)
# ─────────────────────────────────────────────────────────────────────────────

def make_frequentist_result(
    p_value: float = 0.03,
    relative_lift: float = 0.15,
    absolute_lift: float = 0.03,
    control_metric: float = 0.20,
    treatment_metric: float = 0.23,
    is_significant: bool = True,
    sample_size_control: int = 5000,
    sample_size_treatment: int = 5000,
    confidence_interval_absolute: tuple = (-0.01, 0.07),
    confidence_interval_relative: tuple = (-0.05, 0.35),
    significance_level: float = 0.05,
) -> SimpleNamespace:
    """Creates a SimpleNamespace mimicking FrequentistResult fields."""
    return SimpleNamespace(
        experiment_id="test-exp",
        metric_name="conversion_rate",
        control_metric=control_metric,
        treatment_metric=treatment_metric,
        absolute_lift=absolute_lift,
        relative_lift=relative_lift,
        confidence_interval_absolute=confidence_interval_absolute,
        confidence_interval_relative=confidence_interval_relative,
        p_value=p_value,
        is_significant=is_significant,
        significance_level=significance_level,
        test_statistic=2.5,
        test_type="two_proportion_z",
        effect_size=0.08,
        effect_size_type="cohens_h",
        sample_size_control=sample_size_control,
        sample_size_treatment=sample_size_treatment,
        power_achieved=0.83,
        n_bootstrap=0,
        # Note: no probability_treatment_better → identified as frequentist
    )


def make_bayesian_result(
    probability_treatment_better: float = 0.97,
    expected_loss_treatment: float = 0.0005,
    expected_loss_control: float = 0.05,
    posterior_mean_lift: float = 0.03,
    posterior_std_lift: float = 0.005,
    control_metric: float = 0.20,
    treatment_metric: float = 0.23,
    credible_interval_95: tuple = (0.01, 0.05),
    probability_in_rope: float = 0.05,
    decision: str = "SHIP_TREATMENT",
    sample_size_control: int = 5000,
    sample_size_treatment: int = 5000,
) -> SimpleNamespace:
    """Creates a SimpleNamespace mimicking BayesianResult fields."""
    return SimpleNamespace(
        experiment_id="test-exp",
        metric_name="conversion_rate",
        control_metric=control_metric,
        treatment_metric=treatment_metric,
        probability_treatment_better=probability_treatment_better,
        expected_loss_treatment=expected_loss_treatment,
        expected_loss_control=expected_loss_control,
        posterior_mean_lift=posterior_mean_lift,
        posterior_std_lift=posterior_std_lift,
        credible_interval_95=credible_interval_95,
        rope=(-0.005, 0.005),
        probability_in_rope=probability_in_rope,
        decision=decision,
        method="beta_binomial_analytical",
        sample_size_control=sample_size_control,
        sample_size_treatment=sample_size_treatment,
        n_samples=100000,
        bayes_factor=19.0,
        diagnostics={},
    )


@pytest.fixture(scope="module")
def engine():
    return RecommendationEngine(
        prob_ship_threshold=0.95,
        prob_extend_threshold=0.80,
        expected_loss_threshold=0.005,
    )


# 1. Frequentist SHIP_TREATMENT
def test_verdict_ship_treatment_frequentist(engine):
    result = make_frequentist_result(p_value=0.001, relative_lift=0.15, is_significant=True)
    report = engine.generate_report(
        experiment_id="EXP-001",
        experiment_name="Test Ship",
        primary_metric="conversion_rate",
        primary_result=result,
        guardrail_results=[],
    )
    assert report.overall_verdict == "SHIP_TREATMENT", (
        f"Expected SHIP_TREATMENT, got {report.overall_verdict!r}"
    )


# 2. Frequentist SHIP_CONTROL
def test_verdict_ship_control_frequentist(engine):
    result = make_frequentist_result(
        p_value=0.001,
        relative_lift=-0.15,
        absolute_lift=-0.03,
        treatment_metric=0.17,
        is_significant=True,
    )
    report = engine.generate_report(
        experiment_id="EXP-002",
        experiment_name="Test Ship Control",
        primary_metric="conversion_rate",
        primary_result=result,
        guardrail_results=[],
    )
    assert report.overall_verdict == "SHIP_CONTROL", (
        f"Expected SHIP_CONTROL, got {report.overall_verdict!r}"
    )


# 3. Frequentist EXTEND_TEST (p_value just above 0.05)
def test_verdict_extend_test(engine):
    result = make_frequentist_result(
        p_value=0.08, relative_lift=0.05, is_significant=False
    )
    report = engine.generate_report(
        experiment_id="EXP-003",
        experiment_name="Test Extend",
        primary_metric="conversion_rate",
        primary_result=result,
        guardrail_results=[],
    )
    assert report.overall_verdict == "EXTEND_TEST", (
        f"Expected EXTEND_TEST, got {report.overall_verdict!r}"
    )


# 4. Frequentist NO_WINNER (p_value = 0.50)
def test_verdict_no_winner(engine):
    result = make_frequentist_result(
        p_value=0.50, relative_lift=0.01, is_significant=False
    )
    report = engine.generate_report(
        experiment_id="EXP-004",
        experiment_name="Test No Winner",
        primary_metric="conversion_rate",
        primary_result=result,
        guardrail_results=[],
    )
    assert report.overall_verdict == "NO_WINNER", (
        f"Expected NO_WINNER, got {report.overall_verdict!r}"
    )


# 5. Guardrail blocks verdict
def test_guardrail_blocks_verdict(engine):
    result = make_frequentist_result(p_value=0.001, relative_lift=0.15, is_significant=True)
    degraded_guardrail = [
        {
            "metric": "unsubscribe_rate",
            "metric_name": "unsubscribe_rate",
            "control_value": 0.01,
            "treatment_value": 0.03,
            "relative_change": 2.0,
            "p_value": 0.001,
            "is_degraded": True,
            "message": "GUARDRAIL VIOLATED",
        }
    ]
    report = engine.generate_report(
        experiment_id="EXP-005",
        experiment_name="Test Guardrail Block",
        primary_metric="conversion_rate",
        primary_result=result,
        guardrail_results=degraded_guardrail,
    )
    assert report.overall_verdict == "BLOCKED_BY_GUARDRAIL", (
        f"Expected BLOCKED_BY_GUARDRAIL, got {report.overall_verdict!r}"
    )


# 6. Bayesian SHIP_TREATMENT
def test_verdict_ship_treatment_bayesian(engine):
    result = make_bayesian_result(
        probability_treatment_better=0.97,
        expected_loss_treatment=0.001,
    )
    report = engine.generate_report(
        experiment_id="EXP-006",
        experiment_name="Bayesian Ship",
        primary_metric="conversion_rate",
        primary_result=result,
        guardrail_results=[],
    )
    assert report.overall_verdict == "SHIP_TREATMENT", (
        f"Expected SHIP_TREATMENT, got {report.overall_verdict!r}"
    )


# 7. Bayesian CONTINUE_TESTING
def test_verdict_continue_testing_bayesian(engine):
    result = make_bayesian_result(
        probability_treatment_better=0.85,
        expected_loss_treatment=0.01,
        decision="CONTINUE_TESTING",
    )
    report = engine.generate_report(
        experiment_id="EXP-007",
        experiment_name="Bayesian Extend",
        primary_metric="conversion_rate",
        primary_result=result,
        guardrail_results=[],
    )
    # Engine uses EXTEND_TEST for prob between prob_extend_threshold and prob_ship_threshold
    assert report.overall_verdict in ("EXTEND_TEST", "CONTINUE_TESTING"), (
        f"Expected EXTEND_TEST or CONTINUE_TESTING, got {report.overall_verdict!r}"
    )


# 8. generate_report() returns ExperimentReport with all required non-None fields
def test_report_has_all_required_fields(engine):
    result = make_frequentist_result(p_value=0.02, relative_lift=0.10, is_significant=True)
    report = engine.generate_report(
        experiment_id="EXP-008",
        experiment_name="Full Report Test",
        primary_metric="conversion_rate",
        primary_result=result,
    )
    assert isinstance(report, ExperimentReport)
    assert report.experiment_id == "EXP-008"
    assert report.experiment_name == "Full Report Test"
    assert report.overall_verdict is not None and len(report.overall_verdict) > 0
    assert report.verdict_confidence is not None
    assert report.primary_metric_name == "conversion_rate"
    assert isinstance(report.primary_metric_control, float)
    assert isinstance(report.primary_metric_treatment, float)
    assert isinstance(report.primary_metric_lift_absolute, float)
    assert isinstance(report.primary_metric_lift_relative, float)
    assert isinstance(report.primary_metric_confidence, float)
    assert isinstance(report.primary_metric_ci, tuple)
    assert len(report.primary_metric_ci) == 2
    assert isinstance(report.is_primary_significant, bool)
    assert report.generated_at is not None and len(report.generated_at) > 0


# 9. recommendation_text is non-empty string
def test_recommendation_text_not_empty(engine):
    result = make_frequentist_result(p_value=0.03, relative_lift=0.08, is_significant=True)
    report = engine.generate_report(
        experiment_id="EXP-009",
        experiment_name="Rec Text Test",
        primary_metric="conversion_rate",
        primary_result=result,
    )
    assert isinstance(report.recommendation_text, str)
    assert len(report.recommendation_text.strip()) > 0


# 10. generate_portfolio_summary() with 3 reports returns DataFrame with 3 rows
def test_portfolio_summary_has_correct_shape(engine):
    reports = []
    for i in range(3):
        result = make_frequentist_result(
            p_value=0.01 * (i + 1),
            relative_lift=0.05 * (i + 1),
            is_significant=(i < 2),
        )
        report = engine.generate_report(
            experiment_id=f"EXP-{i+1:03d}",
            experiment_name=f"Experiment {i+1}",
            primary_metric="conversion_rate",
            primary_result=result,
        )
        reports.append(report)

    summary = engine.generate_portfolio_summary(reports)
    assert len(summary) == 3, f"Expected 3 rows, got {len(summary)}"
    required_cols = {
        "experiment_id", "experiment_name", "verdict",
        "primary_metric", "lift_relative_pct", "is_primary_significant",
    }
    assert required_cols.issubset(set(summary.columns)), (
        f"Missing columns: {required_cols - set(summary.columns)}"
    )


# 11. save_report() writes JSON, reloading contains key fields
def test_save_and_reload_report(engine):
    result = make_frequentist_result(p_value=0.04, relative_lift=0.07, is_significant=True)
    report = engine.generate_report(
        experiment_id="EXP-SAVE-01",
        experiment_name="Save Test",
        primary_metric="conversion_rate",
        primary_result=result,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_path = engine.save_report(report, output_dir=tmpdir)
        assert Path(saved_path).exists(), f"Saved file not found at {saved_path}"

        with open(saved_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert data["experiment_id"] == "EXP-SAVE-01"
        assert data["experiment_name"] == "Save Test"
        assert "overall_verdict" in data
        assert "recommendation_text" in data
        assert "primary_metric_ci" in data
        assert isinstance(data["primary_metric_ci"], list)
        assert len(data["primary_metric_ci"]) == 2
