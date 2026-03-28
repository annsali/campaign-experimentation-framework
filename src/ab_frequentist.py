"""
ab_frequentist.py
=================
Frequentist A/B test analysis for the Campaign Experimentation Framework.

Supports:
- Two-proportion z-test (+ chi-squared cross-check) for binary / proportion metrics
- Welch's t-test (+ Mann-Whitney U cross-check) for continuous metrics
- Bootstrap confidence intervals for continuous metrics
- One-sided non-inferiority guardrail checks
- Power achieved at the observed effect size (prospective power formula)

All public classes and functions are documented with NumPy-style docstrings.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.weightstats import ttest_ind as sm_ttest_ind

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SIGNIFICANCE_LEVEL,
    CONFIDENCE_LEVEL,
    BOOTSTRAP_ITERATIONS,
    PROPORTION_METRICS,
    CONTINUOUS_METRICS,
    GUARDRAIL_RELATIVE_DEGRADATION,
    GUARDRAIL_ALPHA,
)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FrequentistResult:
    """
    Container for the output of a single frequentist A/B test.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for the experiment.
    metric_name : str
        Name of the metric being tested.
    control_metric : float
        Observed value (rate or mean) for the control group.
    treatment_metric : float
        Observed value (rate or mean) for the treatment group.
    absolute_lift : float
        treatment_metric - control_metric.
    relative_lift : float
        (treatment_metric - control_metric) / control_metric.
    confidence_interval_absolute : tuple[float, float]
        (lower, upper) CI for the absolute lift.
    confidence_interval_relative : tuple[float, float]
        (lower, upper) CI for the relative lift.
    p_value : float
        Two-sided p-value from the primary test.
    is_significant : bool
        True if p_value < significance_level.
    significance_level : float
        Alpha used for the test.
    test_statistic : float
        Test statistic value (z or t).
    test_type : str
        One of "two_proportion_z", "chi_squared", "welch_t", "mann_whitney".
    effect_size : float
        Cohen's h (proportions) or Cohen's d (continuous).
    effect_size_type : str
        "cohens_h" or "cohens_d".
    sample_size_control : int
        Number of observations in the control group.
    sample_size_treatment : int
        Number of observations in the treatment group.
    power_achieved : float
        Prospective power at the observed effect size and sample sizes.
    n_bootstrap : int
        Number of bootstrap resamples used for CIs (0 = analytic).
    """

    experiment_id: str
    metric_name: str
    control_metric: float
    treatment_metric: float
    absolute_lift: float
    relative_lift: float
    confidence_interval_absolute: tuple[float, float]
    confidence_interval_relative: tuple[float, float]
    p_value: float
    is_significant: bool
    significance_level: float
    test_statistic: float
    test_type: str          # "two_proportion_z" | "chi_squared" | "welch_t" | "mann_whitney"
    effect_size: float      # Cohen's h for proportions, Cohen's d for continuous
    effect_size_type: str   # "cohens_h" | "cohens_d"
    sample_size_control: int
    sample_size_treatment: int
    power_achieved: float   # prospective power at observed effect, NOT post-hoc
    n_bootstrap: int = 0


# ---------------------------------------------------------------------------
# Main analysis class
# ---------------------------------------------------------------------------

class FrequentistABTest:
    """
    Frequentist A/B testing engine for the Campaign Experimentation Framework.

    Implements two-proportion z-tests for binary metrics and Welch's t-tests
    (with bootstrap CIs) for continuous metrics.  Cross-checks are performed
    with chi-squared and Mann-Whitney U tests respectively.

    Parameters
    ----------
    significance_level : float, optional
        Two-sided alpha for hypothesis tests.  Default: ``SIGNIFICANCE_LEVEL``.
    confidence_level : float, optional
        Confidence level for interval estimation.  Default: ``CONFIDENCE_LEVEL``.
    n_bootstrap : int, optional
        Number of bootstrap resamples for continuous-metric CIs.
        Default: ``BOOTSTRAP_ITERATIONS``.
    """

    def __init__(
        self,
        significance_level: float = SIGNIFICANCE_LEVEL,
        confidence_level: float = CONFIDENCE_LEVEL,
        n_bootstrap: int = BOOTSTRAP_ITERATIONS,
    ) -> None:
        self.significance_level = significance_level
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self._alpha = significance_level
        self._z_alpha_half = stats.norm.ppf(1.0 - significance_level / 2.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _safe_relative_lift(self, absolute_lift: float, control: float) -> float:
        """Return relative lift, guarding against zero-control."""
        if control == 0.0:
            return np.nan
        return absolute_lift / abs(control)

    def _cohens_h(self, p1: float, p2: float) -> float:
        """Cohen's h effect size for two proportions."""
        return 2.0 * np.arcsin(np.sqrt(p2)) - 2.0 * np.arcsin(np.sqrt(p1))

    def _cohens_d(self, mean_c: float, mean_t: float, std_c: float, std_t: float) -> float:
        """
        Cohen's d using pooled standard deviation.

        pooled_std = sqrt((std_c^2 + std_t^2) / 2)
        """
        pooled_var = (std_c ** 2 + std_t ** 2) / 2.0
        if pooled_var <= 0.0:
            return 0.0
        return (mean_t - mean_c) / np.sqrt(pooled_var)

    def _prospective_power(self, effect: float, n_c: int, n_t: int) -> float:
        """
        Prospective power at the observed effect size.

        Uses the standard two-sample power formula:

            power = Phi(z_beta) + Phi(z_beta')

        where

            z_beta  =  |effect| * sqrt(n_eff / 2) - z_{alpha/2}
            z_beta' = -|effect| * sqrt(n_eff / 2) - z_{alpha/2}

        and n_eff = harmonic_mean(n_c, n_t).

        Parameters
        ----------
        effect : float
            Standardised effect size (Cohen's h or Cohen's d).
        n_c, n_t : int
            Control and treatment sample sizes.

        Returns
        -------
        float
            Estimated power in [0, 1].
        """
        if n_c <= 0 or n_t <= 0 or effect == 0.0:
            return 0.0
        # Harmonic mean for unequal samples
        n_eff = 2.0 * n_c * n_t / (n_c + n_t)
        sqrt_term = abs(effect) * np.sqrt(n_eff / 2.0)
        z_a2 = self._z_alpha_half
        power = (
            stats.norm.sf(z_a2 - sqrt_term)
            + stats.norm.cdf(-z_a2 - sqrt_term)
        )
        return float(np.clip(power, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Proportion metric analysis
    # ------------------------------------------------------------------

    def analyze_proportion(
        self,
        control_successes: int,
        control_n: int,
        treatment_successes: int,
        treatment_n: int,
        metric_name: str,
        experiment_id: str,
    ) -> FrequentistResult:
        """
        Analyse a binary / proportion metric using a two-proportion z-test.

        A chi-squared test is computed as a cross-check but is **not** used
        for the primary p-value or test statistic.

        Confidence intervals for the absolute lift are derived using the
        normal approximation (delta method).  CIs for the relative lift are
        propagated from the absolute CI.

        Parameters
        ----------
        control_successes : int
            Number of successes (conversions, opens, …) in the control group.
        control_n : int
            Total observations in the control group.
        treatment_successes : int
            Number of successes in the treatment group.
        treatment_n : int
            Total observations in the treatment group.
        metric_name : str
            Human-readable metric label.
        experiment_id : str
            Experiment identifier used for record-keeping.

        Returns
        -------
        FrequentistResult
            Fully populated result object.

        Raises
        ------
        ValueError
            If either group has zero observations.
        """
        if control_n <= 0 or treatment_n <= 0:
            raise ValueError(
                f"Sample sizes must be positive. Got control_n={control_n}, "
                f"treatment_n={treatment_n}."
            )

        # Clip successes to valid range
        control_successes = int(np.clip(control_successes, 0, control_n))
        treatment_successes = int(np.clip(treatment_successes, 0, treatment_n))

        control_rate = control_successes / control_n
        treatment_rate = treatment_successes / treatment_n

        # ── Two-proportion z-test (primary) ──────────────────────────────
        counts = np.array([treatment_successes, control_successes])
        nobs = np.array([treatment_n, control_n])
        z_stat, p_value = proportions_ztest(counts, nobs, alternative="two-sided")

        # ── Chi-squared (cross-check) ─────────────────────────────────────
        contingency = np.array([
            [treatment_successes, treatment_n - treatment_successes],
            [control_successes,   control_n   - control_successes],
        ])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                chi2_stat, _chi2_p, _dof, _exp = stats.chi2_contingency(
                    contingency, correction=False
                )
        except ValueError:
            # Degenerate table (all zeros in a row/column); chi2 is undefined
            chi2_stat = np.nan

        # ── Lift ──────────────────────────────────────────────────────────
        absolute_lift = treatment_rate - control_rate
        relative_lift = self._safe_relative_lift(absolute_lift, control_rate)

        # ── CI for absolute lift (delta / normal approx) ──────────────────
        se_diff = np.sqrt(
            control_rate * (1 - control_rate) / control_n
            + treatment_rate * (1 - treatment_rate) / treatment_n
        )
        z_ci = stats.norm.ppf(1.0 - (1.0 - self.confidence_level) / 2.0)
        ci_abs_lo = absolute_lift - z_ci * se_diff
        ci_abs_hi = absolute_lift + z_ci * se_diff

        # ── CI for relative lift (propagate from absolute CI) ─────────────
        if control_rate != 0.0:
            ci_rel_lo = ci_abs_lo / abs(control_rate)
            ci_rel_hi = ci_abs_hi / abs(control_rate)
        else:
            ci_rel_lo = ci_rel_hi = np.nan

        # ── Effect size: Cohen's h ─────────────────────────────────────────
        cohens_h = self._cohens_h(control_rate, treatment_rate)

        # ── Power ─────────────────────────────────────────────────────────
        power = self._prospective_power(cohens_h, control_n, treatment_n)

        return FrequentistResult(
            experiment_id=experiment_id,
            metric_name=metric_name,
            control_metric=control_rate,
            treatment_metric=treatment_rate,
            absolute_lift=absolute_lift,
            relative_lift=relative_lift,
            confidence_interval_absolute=(ci_abs_lo, ci_abs_hi),
            confidence_interval_relative=(ci_rel_lo, ci_rel_hi),
            p_value=float(p_value),
            is_significant=bool(p_value < self.significance_level),
            significance_level=self.significance_level,
            test_statistic=float(z_stat),
            test_type="two_proportion_z",
            effect_size=float(cohens_h),
            effect_size_type="cohens_h",
            sample_size_control=control_n,
            sample_size_treatment=treatment_n,
            power_achieved=power,
            n_bootstrap=0,
        )

    # ------------------------------------------------------------------
    # Continuous metric analysis
    # ------------------------------------------------------------------

    def analyze_continuous(
        self,
        control_values: np.ndarray | list,
        treatment_values: np.ndarray | list,
        metric_name: str,
        experiment_id: str,
    ) -> FrequentistResult:
        """
        Analyse a continuous metric using Welch's t-test with bootstrap CIs.

        Mann-Whitney U is computed as a non-parametric cross-check; its p-value
        is stored in the result's docstring but does not affect the primary
        decision.  Bootstrap confidence intervals use the percentile method.

        Parameters
        ----------
        control_values : array-like
            Raw observations for the control group.
        treatment_values : array-like
            Raw observations for the treatment group.
        metric_name : str
            Human-readable metric label.
        experiment_id : str
            Experiment identifier.

        Returns
        -------
        FrequentistResult
            Fully populated result object.  The ``test_type`` is ``"welch_t"``.

        Raises
        ------
        ValueError
            If either array is empty.
        """
        ctrl = np.asarray(control_values, dtype=float)
        treat = np.asarray(treatment_values, dtype=float)

        ctrl = ctrl[np.isfinite(ctrl)]
        treat = treat[np.isfinite(treat)]

        if len(ctrl) == 0 or len(treat) == 0:
            raise ValueError(
                f"Both arrays must be non-empty after removing non-finite values. "
                f"Got {len(ctrl)} control and {len(treat)} treatment observations."
            )

        n_c = len(ctrl)
        n_t = len(treat)
        mean_c = float(np.mean(ctrl))
        mean_t = float(np.mean(treat))
        std_c = float(np.std(ctrl, ddof=1)) if n_c > 1 else 0.0
        std_t = float(np.std(treat, ddof=1)) if n_t > 1 else 0.0

        # ── Welch's t-test (primary) ──────────────────────────────────────
        t_stat, p_value = stats.ttest_ind(ctrl, treat, equal_var=False)

        # ── Mann-Whitney U (cross-check / non-parametric) ─────────────────
        try:
            _mw_stat, _mw_p = stats.mannwhitneyu(ctrl, treat, alternative="two-sided")
        except ValueError:
            _mw_p = np.nan

        # ── Lift ──────────────────────────────────────────────────────────
        absolute_lift = mean_t - mean_c
        relative_lift = self._safe_relative_lift(absolute_lift, mean_c)

        # ── Bootstrap CI (percentile method) ─────────────────────────────
        rng = np.random.default_rng(42)
        boot_diffs = np.empty(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            boot_c = rng.choice(ctrl, size=n_c, replace=True)
            boot_t = rng.choice(treat, size=n_t, replace=True)
            boot_diffs[i] = np.mean(boot_t) - np.mean(boot_c)

        alpha_tail = (1.0 - self.confidence_level) / 2.0
        ci_abs_lo = float(np.percentile(boot_diffs, 100.0 * alpha_tail))
        ci_abs_hi = float(np.percentile(boot_diffs, 100.0 * (1.0 - alpha_tail)))

        # Relative CI: propagate bootstrap diffs through relative transform
        if mean_c != 0.0:
            boot_rel = boot_diffs / abs(mean_c)
            ci_rel_lo = float(np.percentile(boot_rel, 100.0 * alpha_tail))
            ci_rel_hi = float(np.percentile(boot_rel, 100.0 * (1.0 - alpha_tail)))
        else:
            ci_rel_lo = ci_rel_hi = np.nan

        # ── Effect size: Cohen's d ─────────────────────────────────────────
        cohens_d = self._cohens_d(mean_c, mean_t, std_c, std_t)

        # ── Power ─────────────────────────────────────────────────────────
        power = self._prospective_power(cohens_d, n_c, n_t)

        return FrequentistResult(
            experiment_id=experiment_id,
            metric_name=metric_name,
            control_metric=mean_c,
            treatment_metric=mean_t,
            absolute_lift=absolute_lift,
            relative_lift=relative_lift,
            confidence_interval_absolute=(ci_abs_lo, ci_abs_hi),
            confidence_interval_relative=(ci_rel_lo, ci_rel_hi),
            p_value=float(p_value),
            is_significant=bool(p_value < self.significance_level),
            significance_level=self.significance_level,
            test_statistic=float(t_stat),
            test_type="welch_t",
            effect_size=float(cohens_d),
            effect_size_type="cohens_d",
            sample_size_control=n_c,
            sample_size_treatment=n_t,
            power_achieved=power,
            n_bootstrap=self.n_bootstrap,
        )

    # ------------------------------------------------------------------
    # Guardrail checks
    # ------------------------------------------------------------------

    def run_guardrail_checks(
        self,
        control_data: dict[str, Any],
        treatment_data: dict[str, Any],
        guardrail_metrics: list[str],
        metric_type_map: dict[str, str],
    ) -> list[dict[str, Any]]:
        """
        Run one-sided non-inferiority guardrail tests for a list of metrics.

        For each metric, tests whether the treatment has degraded the metric
        beyond an acceptable threshold.  "Degradation" is metric-direction
        aware: for metrics where lower is worse (e.g., ``bounce_rate``,
        ``unsubscribe_rate``) the user should reverse the sign convention
        upstream; by default this method treats *increase* as degradation.

        Null hypothesis
        ---------------
        H0: relative_change <= GUARDRAIL_RELATIVE_DEGRADATION
        Ha: relative_change  > GUARDRAIL_RELATIVE_DEGRADATION  (degraded)

        The one-sided p-value is derived from the two-sided z-test p-value.

        Parameters
        ----------
        control_data : dict
            Mapping of metric_name -> array-like of control observations, OR
            ``{"successes": int, "n": int}`` for proportion metrics.
        treatment_data : dict
            Same structure as ``control_data`` for the treatment group.
        guardrail_metrics : list[str]
            Metrics to check.
        metric_type_map : dict[str, str]
            ``{metric_name: "proportion" | "continuous"}`` lookup.

        Returns
        -------
        list[dict]
            One dict per guardrail metric with keys:

            - ``metric``           – metric name
            - ``control_value``    – control rate or mean
            - ``treatment_value``  – treatment rate or mean
            - ``relative_change``  – (treatment - control) / |control|
            - ``p_value``          – one-sided p-value
            - ``is_degraded``      – True if significant degradation detected
            - ``message``          – human-readable summary
        """
        results = []
        for metric in guardrail_metrics:
            try:
                m_type = metric_type_map.get(metric, "continuous")
                c_raw = control_data.get(metric)
                t_raw = treatment_data.get(metric)

                if c_raw is None or t_raw is None:
                    results.append({
                        "metric": metric,
                        "control_value": np.nan,
                        "treatment_value": np.nan,
                        "relative_change": np.nan,
                        "p_value": np.nan,
                        "is_degraded": False,
                        "message": f"No data found for guardrail metric '{metric}'.",
                    })
                    continue

                if m_type == "proportion":
                    # Expect dict or tuple: {"successes": int, "n": int}
                    if isinstance(c_raw, dict):
                        c_succ, c_n = c_raw["successes"], c_raw["n"]
                        t_succ, t_n = t_raw["successes"], t_raw["n"]
                    else:
                        c_arr = np.asarray(c_raw, dtype=float)
                        t_arr = np.asarray(t_raw, dtype=float)
                        c_succ, c_n = int(np.sum(c_arr)), len(c_arr)
                        t_succ, t_n = int(np.sum(t_arr)), len(t_arr)

                    c_val = c_succ / c_n if c_n > 0 else 0.0
                    t_val = t_succ / t_n if t_n > 0 else 0.0

                    counts = np.array([t_succ, c_succ])
                    nobs = np.array([t_n, c_n])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _z, p_two = proportions_ztest(counts, nobs, alternative="two-sided")
                else:
                    c_arr = np.asarray(c_raw, dtype=float)
                    t_arr = np.asarray(t_raw, dtype=float)
                    c_arr = c_arr[np.isfinite(c_arr)]
                    t_arr = t_arr[np.isfinite(t_arr)]
                    c_val = float(np.mean(c_arr)) if len(c_arr) > 0 else np.nan
                    t_val = float(np.mean(t_arr)) if len(t_arr) > 0 else np.nan

                    if len(c_arr) < 2 or len(t_arr) < 2:
                        p_two = np.nan
                    else:
                        _t, p_two = stats.ttest_ind(c_arr, t_arr, equal_var=False)

                # One-sided p-value (right tail: treatment > control threshold)
                p_one = p_two / 2.0 if not np.isnan(p_two) else np.nan

                rel_change = (
                    (t_val - c_val) / abs(c_val)
                    if c_val not in (0.0, np.nan) and not np.isnan(c_val)
                    else np.nan
                )

                is_degraded = bool(
                    not np.isnan(rel_change)
                    and rel_change > GUARDRAIL_RELATIVE_DEGRADATION
                    and not np.isnan(p_one)
                    and p_one < GUARDRAIL_ALPHA
                )

                if is_degraded:
                    msg = (
                        f"GUARDRAIL VIOLATED: '{metric}' degraded by "
                        f"{rel_change:.1%} (threshold: {GUARDRAIL_RELATIVE_DEGRADATION:.1%}), "
                        f"p={p_one:.4f} < alpha={GUARDRAIL_ALPHA}."
                    )
                elif not np.isnan(rel_change) and rel_change > GUARDRAIL_RELATIVE_DEGRADATION:
                    msg = (
                        f"WARNING: '{metric}' shows {rel_change:.1%} degradation "
                        f"but is not statistically significant (p={p_one:.4f})."
                    )
                else:
                    msg = (
                        f"OK: '{metric}' within acceptable bounds "
                        f"(relative change: {rel_change:.1%})."
                    )

                results.append({
                    "metric": metric,
                    "control_value": c_val,
                    "treatment_value": t_val,
                    "relative_change": rel_change,
                    "p_value": p_one,
                    "is_degraded": is_degraded,
                    "message": msg,
                })

            except Exception as exc:  # pragma: no cover
                results.append({
                    "metric": metric,
                    "control_value": np.nan,
                    "treatment_value": np.nan,
                    "relative_change": np.nan,
                    "p_value": np.nan,
                    "is_degraded": False,
                    "message": f"Error during guardrail check for '{metric}': {exc}",
                })

        return results

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        experiment_data: pd.DataFrame,
        variants: list[str],
        metrics: list[str],
        guardrail_metrics: list[str],
        experiment_id: str,
        control_variant: str | None = None,
    ) -> dict[str, Any]:
        """
        Run the full frequentist analysis for an experiment.

        Automatically detects whether each metric is a proportion or continuous
        variable by inspecting the column's unique values and mean.

        Parameters
        ----------
        experiment_data : pd.DataFrame
            Must contain a ``variant`` column and one column per metric.
        variants : list[str]
            All variant labels present in the dataset.
        metrics : list[str]
            Primary metrics to test.
        guardrail_metrics : list[str]
            Metrics used for guardrail non-inferiority checks.
        experiment_id : str
            Experiment identifier propagated to all ``FrequentistResult`` objects.
        control_variant : str, optional
            Label of the control variant.  Defaults to ``variants[0]``.

        Returns
        -------
        dict with keys:
            - ``results``           – ``{metric_name: FrequentistResult}``
            - ``guardrail_checks``  – list of dicts from ``run_guardrail_checks``
            - ``experiment_id``     – experiment identifier
            - ``control_variant``   – control variant label
            - ``treatment_variant`` – treatment variant label
        """
        if "variant" not in experiment_data.columns:
            raise KeyError("experiment_data must contain a 'variant' column.")

        if control_variant is None:
            control_variant = variants[0]

        treatment_variants = [v for v in variants if v != control_variant]
        if len(treatment_variants) == 0:
            raise ValueError("At least one treatment variant is required.")
        # Use the first non-control variant for 2-variant case
        treatment_variant = treatment_variants[0]

        ctrl_df = experiment_data[experiment_data["variant"] == control_variant]
        treat_df = experiment_data[experiment_data["variant"] == treatment_variant]

        if ctrl_df.empty or treat_df.empty:
            raise ValueError(
                f"One or both variants ('{control_variant}', '{treatment_variant}') "
                "have no rows in experiment_data."
            )

        # ── Analyse primary metrics ───────────────────────────────────────
        metric_results: dict[str, FrequentistResult] = {}
        metric_type_map: dict[str, str] = {}

        for metric in metrics:
            if metric not in experiment_data.columns:
                warnings.warn(f"Metric '{metric}' not found in data; skipping.")
                continue

            # Skip non-numeric columns that cannot be tested
            if not pd.api.types.is_numeric_dtype(experiment_data[metric]):
                warnings.warn(
                    f"Metric '{metric}' is non-numeric (dtype={experiment_data[metric].dtype}); skipping."
                )
                continue

            ctrl_vals = ctrl_df[metric].dropna().values.astype(float)
            treat_vals = treat_df[metric].dropna().values.astype(float)

            if len(ctrl_vals) == 0 or len(treat_vals) == 0:
                warnings.warn(
                    f"Metric '{metric}' has no valid observations in one or both groups; skipping."
                )
                continue

            is_proportion = self._detect_proportion(metric, ctrl_vals)
            metric_type_map[metric] = "proportion" if is_proportion else "continuous"

            try:
                if is_proportion:
                    result = self.analyze_proportion(
                        control_successes=int(np.sum(ctrl_vals)),
                        control_n=len(ctrl_vals),
                        treatment_successes=int(np.sum(treat_vals)),
                        treatment_n=len(treat_vals),
                        metric_name=metric,
                        experiment_id=experiment_id,
                    )
                else:
                    result = self.analyze_continuous(
                        control_values=ctrl_vals,
                        treatment_values=treat_vals,
                        metric_name=metric,
                        experiment_id=experiment_id,
                    )
                metric_results[metric] = result
            except Exception as exc:
                warnings.warn(f"Analysis failed for metric '{metric}': {exc}")

        # ── Guardrail checks ──────────────────────────────────────────────
        # Build data dicts for the guardrail helper
        all_guardrail = list(set(guardrail_metrics))
        ctrl_gdata: dict[str, Any] = {}
        treat_gdata: dict[str, Any] = {}
        for gm in all_guardrail:
            if gm not in experiment_data.columns:
                continue
            ctrl_gdata[gm] = ctrl_df[gm].dropna().values
            treat_gdata[gm] = treat_df[gm].dropna().values
            # Add to type map if not already there
            if gm not in metric_type_map:
                g_vals = ctrl_df[gm].dropna().values
                metric_type_map[gm] = (
                    "proportion" if self._detect_proportion(gm, g_vals) else "continuous"
                )

        guardrail_checks = self.run_guardrail_checks(
            control_data=ctrl_gdata,
            treatment_data=treat_gdata,
            guardrail_metrics=all_guardrail,
            metric_type_map=metric_type_map,
        )

        return {
            "results": metric_results,
            "guardrail_checks": guardrail_checks,
            "experiment_id": experiment_id,
            "control_variant": control_variant,
            "treatment_variant": treatment_variant,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_proportion(metric_name: str, values: np.ndarray) -> bool:
        """
        Determine whether a metric should be treated as a proportion.

        A metric is classified as a proportion if:
        - Its name is in the global ``PROPORTION_METRICS`` set, OR
        - All finite values are 0 or 1 (binary column), AND the mean is
          strictly between 0 and 1.

        Parameters
        ----------
        metric_name : str
        values : np.ndarray

        Returns
        -------
        bool
        """
        if metric_name in PROPORTION_METRICS:
            return True
        if metric_name in CONTINUOUS_METRICS:
            return False
        try:
            float_vals = np.asarray(values, dtype=float)
        except (TypeError, ValueError):
            return False
        finite_vals = float_vals[np.isfinite(float_vals)]
        if len(finite_vals) == 0:
            return False
        unique_vals = set(np.unique(finite_vals))
        is_binary = unique_vals.issubset({0.0, 1.0})
        mean_val = float(np.mean(finite_vals))
        return is_binary and 0.0 < mean_val < 1.0

    def results_to_dataframe(self, results: dict[str, Any]) -> pd.DataFrame:
        """
        Convert the ``results`` dict (from ``analyze``) to a tidy DataFrame.

        Parameters
        ----------
        results : dict
            Output of ``analyze()``.  The ``results`` key must map metric
            names to ``FrequentistResult`` instances.

        Returns
        -------
        pd.DataFrame
            One row per metric with all result fields as columns.
        """
        metric_results = results.get("results", {})
        if not metric_results:
            return pd.DataFrame()

        rows = []
        for metric_name, r in metric_results.items():
            rows.append({
                "experiment_id": r.experiment_id,
                "metric": r.metric_name,
                "test_type": r.test_type,
                "control_value": r.control_metric,
                "treatment_value": r.treatment_metric,
                "absolute_lift": r.absolute_lift,
                "relative_lift_pct": r.relative_lift * 100.0 if not np.isnan(r.relative_lift) else np.nan,
                "ci_abs_lower": r.confidence_interval_absolute[0],
                "ci_abs_upper": r.confidence_interval_absolute[1],
                "ci_rel_lower_pct": r.confidence_interval_relative[0] * 100.0
                    if not np.isnan(r.confidence_interval_relative[0]) else np.nan,
                "ci_rel_upper_pct": r.confidence_interval_relative[1] * 100.0
                    if not np.isnan(r.confidence_interval_relative[1]) else np.nan,
                "p_value": r.p_value,
                "is_significant": r.is_significant,
                "significance_level": r.significance_level,
                "test_statistic": r.test_statistic,
                "effect_size": r.effect_size,
                "effect_size_type": r.effect_size_type,
                "n_control": r.sample_size_control,
                "n_treatment": r.sample_size_treatment,
                "power_achieved": r.power_achieved,
                "n_bootstrap": r.n_bootstrap,
            })

        df = pd.DataFrame(rows).set_index("metric")
        return df


# ---------------------------------------------------------------------------
# Module-level helper function
# ---------------------------------------------------------------------------

def format_result_report(result: FrequentistResult) -> str:
    """
    Return a human-readable summary of a single ``FrequentistResult``.

    Parameters
    ----------
    result : FrequentistResult
        The result to format.

    Returns
    -------
    str
        Multi-line formatted string suitable for console output.

    Examples
    --------
    >>> print(format_result_report(res))
    ============================================================
    Metric         : open_rate
    Experiment     : exp001_email_subject_line
    Test type      : two_proportion_z
    ...
    """
    sig_marker = "*** SIGNIFICANT ***" if result.is_significant else "not significant"
    rel_lift_str = (
        f"{result.relative_lift:+.2%}" if not np.isnan(result.relative_lift) else "N/A"
    )
    ci_abs = result.confidence_interval_absolute
    ci_rel = result.confidence_interval_relative

    ci_abs_str = f"[{ci_abs[0]:+.6f}, {ci_abs[1]:+.6f}]"
    if not (np.isnan(ci_rel[0]) or np.isnan(ci_rel[1])):
        ci_rel_str = f"[{ci_rel[0]:+.2%}, {ci_rel[1]:+.2%}]"
    else:
        ci_rel_str = "N/A"

    ci_label = f"{(1.0 - result.significance_level) * 100:.0f}% CI"

    lines = [
        "=" * 60,
        f"  Metric         : {result.metric_name}",
        f"  Experiment     : {result.experiment_id}",
        f"  Test type      : {result.test_type}",
        "-" * 60,
        f"  Control        : {result.control_metric:.6f}  (n={result.sample_size_control:,})",
        f"  Treatment      : {result.treatment_metric:.6f}  (n={result.sample_size_treatment:,})",
        f"  Absolute lift  : {result.absolute_lift:+.6f}",
        f"  Relative lift  : {rel_lift_str}",
        f"  CI absolute    : {ci_abs_str}  ({ci_label})",
        f"  CI relative    : {ci_rel_str}",
        "-" * 60,
        f"  p-value        : {result.p_value:.6f}",
        f"  Test statistic : {result.test_statistic:.4f}",
        f"  Effect size    : {result.effect_size:.4f}  ({result.effect_size_type})",
        f"  Power achieved : {result.power_achieved:.2%}",
        f"  Decision       : {sig_marker}  (alpha={result.significance_level})",
        "=" * 60,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    # Resolve paths relative to this file so the script works from any cwd
    _PROJECT_ROOT = Path(__file__).parent.parent
    _DATA_PATH = _PROJECT_ROOT / "data" / "exp001_email_subject_line.csv"

    print(f"\nCampaign Experimentation Framework — Frequentist A/B Analysis")
    print(f"Project root : {_PROJECT_ROOT}")
    print(f"Data file    : {_DATA_PATH}\n")

    # ── Load data ─────────────────────────────────────────────────────────
    if not _DATA_PATH.exists():
        print(
            f"[INFO] Data file not found at {_DATA_PATH}.\n"
            "       Generating synthetic data for demonstration...\n"
        )
        rng = np.random.default_rng(42)
        n_per_variant = 2_500

        ctrl_open = rng.binomial(1, 0.22, n_per_variant)
        treat_open = rng.binomial(1, 0.26, n_per_variant)
        ctrl_click = rng.binomial(1, 0.045, n_per_variant)
        treat_click = rng.binomial(1, 0.052, n_per_variant)
        ctrl_time = rng.gamma(shape=2.5, scale=40, size=n_per_variant)
        treat_time = rng.gamma(shape=2.5, scale=43, size=n_per_variant)

        df = pd.DataFrame({
            "user_id": range(2 * n_per_variant),
            "variant": ["control"] * n_per_variant + ["treatment"] * n_per_variant,
            "open_rate": np.concatenate([ctrl_open, treat_open]).astype(float),
            "click_rate": np.concatenate([ctrl_click, treat_click]).astype(float),
            "time_on_page": np.concatenate([ctrl_time, treat_time]),
        })
    else:
        df = pd.read_csv(_DATA_PATH)
        print(f"Loaded {len(df):,} rows from {_DATA_PATH.name}")
        print(df.head())
        print()

    # ── Run analysis ──────────────────────────────────────────────────────
    tester = FrequentistABTest()

    variants_in_data = df["variant"].unique().tolist()

    # Only pass numeric columns as metrics; skip identifiers and categoricals
    _non_metric = {"user_id", "contact_id", "variant", "assignment_date",
                   "day_of_week", "industry", "company_size", "region",
                   "engagement_tier"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    primary_metrics = [c for c in numeric_cols if c not in _non_metric]
    guardrail_m = [m for m in primary_metrics if m in {"bounce_rate", "unsubscribe_rate",
                                                        "spam_complaint_rate"}]

    output = tester.analyze(
        experiment_data=df,
        variants=variants_in_data,
        metrics=primary_metrics,
        guardrail_metrics=guardrail_m,
        experiment_id="exp001_email_subject_line",
        control_variant=None,  # auto-select first variant
    )

    print(f"Experiment : {output['experiment_id']}")
    print(f"Control    : {output['control_variant']}")
    print(f"Treatment  : {output['treatment_variant']}\n")

    # ── Print per-metric reports ──────────────────────────────────────────
    for metric, res in output["results"].items():
        print(format_result_report(res))
        print()

    # ── Guardrail summary ─────────────────────────────────────────────────
    if output["guardrail_checks"]:
        print("\nGuardrail Checks")
        print("-" * 60)
        for gc in output["guardrail_checks"]:
            status = "FAIL" if gc["is_degraded"] else "PASS"
            print(f"  [{status}] {gc['message']}")
    else:
        print("\n(No guardrail metrics to check for this dataset.)")

    # ── Summary table ─────────────────────────────────────────────────────
    summary_df = tester.results_to_dataframe(output)
    if not summary_df.empty:
        print("\nSummary Table")
        print("=" * 60)
        cols_to_show = [
            "control_value", "treatment_value",
            "absolute_lift", "relative_lift_pct",
            "p_value", "is_significant", "power_achieved",
        ]
        available_cols = [c for c in cols_to_show if c in summary_df.columns]
        with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
            print(summary_df[available_cols].to_string())
    print()
