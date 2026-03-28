"""
Bayesian A/B Test Analysis for the Campaign Experimentation Framework.

Supports two computation paths:
  1. Beta-Binomial analytical (fast, no PyMC required) — default for proportion metrics.
  2. PyMC MCMC (slow, full posterior) — optional; falls back to analytical when PyMC is
     not installed or MCMC fails to converge.

Decision logic follows a three-outcome rule:
  SHIP_TREATMENT | SHIP_CONTROL | CONTINUE_TESTING | INCONCLUSIVE
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    BAYESIAN_MC_SAMPLES,
    EXPECTED_LOSS_THRESHOLD,
    MCMC_CHAINS,
    MCMC_DRAWS,
    MCMC_TARGET_ACCEPT,
    MCMC_TUNE,
    MIN_ESS,
    PROB_TREATMENT_BETTER_EXTEND,
    PROB_TREATMENT_BETTER_SHIP,
    PROPORTION_METRICS,
    RHAT_THRESHOLD,
    ROPE_DEFAULTS,
    SIGNIFICANCE_LEVEL,
)

# ── optional heavy dependencies ───────────────────────────────────────────────
try:
    import pymc as pm
    import arviz as az

    _PYMC_AVAILABLE = True
except ImportError:
    _PYMC_AVAILABLE = False
    pm = None  # type: ignore[assignment]
    az = None  # type: ignore[assignment]

# ── try arviz standalone (in case only arviz is installed) ────────────────────
if not _PYMC_AVAILABLE:
    try:
        import arviz as az  # noqa: F811

        _ARVIZ_AVAILABLE = True
    except ImportError:
        _ARVIZ_AVAILABLE = False
else:
    _ARVIZ_AVAILABLE = True


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BayesianResult:
    """Container for all Bayesian A/B test outputs."""

    experiment_id: str
    metric_name: str

    # Posterior means for each variant
    control_metric: float
    treatment_metric: float

    # Core decision signals
    probability_treatment_better: float  # P(treatment > control)
    expected_loss_treatment: float       # expected loss if we choose treatment
    expected_loss_control: float         # expected loss if we choose control

    # Lift distribution
    posterior_mean_lift: float
    posterior_std_lift: float
    credible_interval_95: tuple[float, float]  # 95 % HDI

    # ROPE
    rope: tuple[float, float]
    probability_in_rope: float

    # Decision
    decision: str  # SHIP_TREATMENT | SHIP_CONTROL | CONTINUE_TESTING | INCONCLUSIVE

    # Method metadata
    method: str  # "beta_binomial_analytical" | "pymc_mcmc"
    sample_size_control: int
    sample_size_treatment: int
    n_samples: int

    # Bayesian evidence
    bayes_factor: float  # Savage-Dickey / ratio approximation for point null H0

    # Optional MCMC diagnostics (populated by PyMC methods)
    diagnostics: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Main analyser class
# ─────────────────────────────────────────────────────────────────────────────


class BayesianABTest:
    """
    Bayesian A/B test analyser.

    Parameters
    ----------
    mc_samples:
        Number of Monte Carlo draws for analytical Beta-Binomial integration.
    prob_ship_threshold:
        P(treatment > control) required to declare SHIP_TREATMENT.
    prob_extend_threshold:
        Lower bound of P(treatment > control) to recommend CONTINUE_TESTING.
    expected_loss_threshold:
        Maximum acceptable expected loss (absolute) to ship a variant.
    """

    def __init__(
        self,
        mc_samples: int = BAYESIAN_MC_SAMPLES,
        prob_ship_threshold: float = PROB_TREATMENT_BETTER_SHIP,
        prob_extend_threshold: float = PROB_TREATMENT_BETTER_EXTEND,
        expected_loss_threshold: float = EXPECTED_LOSS_THRESHOLD,
    ) -> None:
        self.mc_samples = mc_samples
        self.prob_ship_threshold = prob_ship_threshold
        self.prob_extend_threshold = prob_extend_threshold
        self.expected_loss_threshold = expected_loss_threshold
        self._rng = np.random.default_rng(seed=42)

    # ── decision logic ────────────────────────────────────────────────────────

    def _compute_decision(
        self,
        prob_better: float,
        expected_loss_treatment: float,
        expected_loss_control: float,
    ) -> str:
        """
        Map posterior summaries to a human-readable decision string.

        Rules (evaluated in order):
          1. SHIP_TREATMENT  — high confidence treatment wins AND low loss
          2. SHIP_CONTROL    — high confidence treatment loses AND low loss for control
          3. CONTINUE_TESTING — moderate confidence, more data may help
          4. INCONCLUSIVE    — low confidence, no clear direction
        """
        if (
            prob_better >= self.prob_ship_threshold
            and expected_loss_treatment <= self.expected_loss_threshold
        ):
            return "SHIP_TREATMENT"

        if (
            prob_better <= (1.0 - self.prob_ship_threshold)
            and expected_loss_control <= self.expected_loss_threshold
        ):
            return "SHIP_CONTROL"

        if prob_better >= self.prob_extend_threshold:
            return "CONTINUE_TESTING"

        return "INCONCLUSIVE"

    # ── shared helpers ────────────────────────────────────────────────────────

    def _hdi_95(self, samples: np.ndarray) -> tuple[float, float]:
        """Return the 95 % highest-density interval for a 1-D array of samples."""
        if _ARVIZ_AVAILABLE and az is not None:
            hdi = az.hdi(samples, hdi_prob=0.95)
            # arviz returns a numpy array [lower, upper]
            return float(hdi[0]), float(hdi[1])
        lo, hi = np.percentile(samples, [2.5, 97.5])
        return float(lo), float(hi)

    def _get_rope(self, metric_name: str) -> tuple[float, float]:
        """Look up ROPE bounds; fall back to 'default' key if metric not catalogued."""
        return ROPE_DEFAULTS.get(metric_name, ROPE_DEFAULTS["default"])

    def _bayes_factor_ratio(self, prob_better: float) -> float:
        """
        Savage-Dickey ratio approximation for BF10 (H1: delta != 0 vs H0: delta = 0).

        BF10 ≈ P(T > C) / P(T <= C)  (assumes equal prior odds)
        Clipped to avoid division-by-zero at the boundaries.
        """
        p = np.clip(prob_better, 1e-9, 1.0 - 1e-9)
        return float(p / (1.0 - p))

    def _lift_summaries(
        self,
        control_samples: np.ndarray,
        treatment_samples: np.ndarray,
        metric_name: str,
    ) -> dict:
        """Compute all lift-related quantities shared across methods."""
        lift = treatment_samples - control_samples
        prob_better = float(np.mean(treatment_samples > control_samples))

        rope = self._get_rope(metric_name)
        prob_in_rope = float(
            np.mean((lift >= rope[0]) & (lift <= rope[1]))
        )

        el_treatment = float(np.mean(np.maximum(control_samples - treatment_samples, 0.0)))
        el_control = float(np.mean(np.maximum(treatment_samples - control_samples, 0.0)))

        return dict(
            lift_samples=lift,
            prob_better=prob_better,
            posterior_mean_lift=float(np.mean(lift)),
            posterior_std_lift=float(np.std(lift)),
            credible_interval_95=self._hdi_95(lift),
            rope=rope,
            probability_in_rope=prob_in_rope,
            expected_loss_treatment=el_treatment,
            expected_loss_control=el_control,
        )

    # ── analytical Beta-Binomial ──────────────────────────────────────────────

    def analyze_proportion_analytical(
        self,
        control_successes: int,
        control_n: int,
        treatment_successes: int,
        treatment_n: int,
        metric_name: str,
        experiment_id: str,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> BayesianResult:
        """
        Closed-form Beta-Binomial Bayesian analysis via Monte Carlo integration.

        Posterior distributions:
          control   ~ Beta(prior_alpha + successes_c,  prior_beta + n_c - successes_c)
          treatment ~ Beta(prior_alpha + successes_t,  prior_beta + n_t - successes_t)

        Parameters
        ----------
        control_successes, control_n:
            Successes and total observations for the control variant.
        treatment_successes, treatment_n:
            Successes and total observations for the treatment variant.
        metric_name:
            Used for ROPE look-up and labelling.
        experiment_id:
            Identifier propagated into the result object.
        prior_alpha, prior_beta:
            Beta prior hyper-parameters (default: uniform Beta(1,1)).
        """
        # Posterior parameters
        a_c = prior_alpha + control_successes
        b_c = prior_beta + (control_n - control_successes)
        a_t = prior_alpha + treatment_successes
        b_t = prior_beta + (treatment_n - treatment_successes)

        # Draw MC samples
        control_samples = stats.beta.rvs(
            a_c, b_c, size=self.mc_samples, random_state=self._rng
        )
        treatment_samples = stats.beta.rvs(
            a_t, b_t, size=self.mc_samples, random_state=self._rng
        )

        summaries = self._lift_summaries(control_samples, treatment_samples, metric_name)
        prob_better = summaries["prob_better"]

        decision = self._compute_decision(
            prob_better,
            summaries["expected_loss_treatment"],
            summaries["expected_loss_control"],
        )

        return BayesianResult(
            experiment_id=experiment_id,
            metric_name=metric_name,
            control_metric=float(np.mean(control_samples)),
            treatment_metric=float(np.mean(treatment_samples)),
            probability_treatment_better=prob_better,
            expected_loss_treatment=summaries["expected_loss_treatment"],
            expected_loss_control=summaries["expected_loss_control"],
            posterior_mean_lift=summaries["posterior_mean_lift"],
            posterior_std_lift=summaries["posterior_std_lift"],
            credible_interval_95=summaries["credible_interval_95"],
            rope=summaries["rope"],
            probability_in_rope=summaries["probability_in_rope"],
            decision=decision,
            method="beta_binomial_analytical",
            sample_size_control=control_n,
            sample_size_treatment=treatment_n,
            n_samples=self.mc_samples,
            bayes_factor=self._bayes_factor_ratio(prob_better),
        )

    # ── PyMC Beta-Binomial (proportion) ───────────────────────────────────────

    def analyze_proportion_pymc(
        self,
        control_successes: int,
        control_n: int,
        treatment_successes: int,
        treatment_n: int,
        metric_name: str,
        experiment_id: str,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> BayesianResult:
        """
        PyMC Beta-Binomial model for proportion metrics.

        Falls back to ``analyze_proportion_analytical`` if PyMC is not available
        or if the sampler fails to converge (R-hat >= RHAT_THRESHOLD or ESS < MIN_ESS).

        Parameters
        ----------
        (Same as analyze_proportion_analytical)
        """
        if not _PYMC_AVAILABLE:
            warnings.warn(
                "PyMC is not installed — falling back to analytical Beta-Binomial.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.analyze_proportion_analytical(
                control_successes, control_n,
                treatment_successes, treatment_n,
                metric_name, experiment_id,
                prior_alpha, prior_beta,
            )

        try:
            with pm.Model() as _model:  # noqa: F841
                # Priors
                theta_c = pm.Beta("theta_c", alpha=prior_alpha, beta=prior_beta)
                theta_t = pm.Beta("theta_t", alpha=prior_alpha, beta=prior_beta)

                # Likelihoods
                pm.Binomial(
                    "obs_c",
                    n=control_n,
                    p=theta_c,
                    observed=control_successes,
                )
                pm.Binomial(
                    "obs_t",
                    n=treatment_n,
                    p=theta_t,
                    observed=treatment_successes,
                )

                # Derived quantity
                _delta = pm.Deterministic("delta", theta_t - theta_c)  # noqa: F841

                idata = pm.sample(
                    draws=MCMC_DRAWS,
                    tune=MCMC_TUNE,
                    chains=MCMC_CHAINS,
                    target_accept=MCMC_TARGET_ACCEPT,
                    progressbar=False,
                    random_seed=42,
                )

            # ── convergence diagnostics ───────────────────────────────────────
            rhat_vals = az.rhat(idata)
            ess_vals = az.ess(idata)

            diagnostics: dict = {}
            converged = True

            for var in ("theta_c", "theta_t"):
                rhat = float(rhat_vals[var].values)
                ess = float(ess_vals[var].values)
                diagnostics[f"rhat_{var}"] = rhat
                diagnostics[f"ess_{var}"] = ess
                if rhat >= RHAT_THRESHOLD:
                    warnings.warn(
                        f"R-hat for {var} = {rhat:.4f} >= {RHAT_THRESHOLD} "
                        f"(experiment={experiment_id}, metric={metric_name}). "
                        "Falling back to analytical method.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    converged = False
                if ess < MIN_ESS:
                    warnings.warn(
                        f"ESS for {var} = {ess:.1f} < {MIN_ESS} "
                        f"(experiment={experiment_id}, metric={metric_name}). "
                        "Falling back to analytical method.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    converged = False

            if not converged:
                return self.analyze_proportion_analytical(
                    control_successes, control_n,
                    treatment_successes, treatment_n,
                    metric_name, experiment_id,
                    prior_alpha, prior_beta,
                )

            # ── extract posterior samples ─────────────────────────────────────
            post = idata.posterior
            control_samples = post["theta_c"].values.flatten()
            treatment_samples = post["theta_t"].values.flatten()

            summaries = self._lift_summaries(control_samples, treatment_samples, metric_name)
            prob_better = summaries["prob_better"]

            decision = self._compute_decision(
                prob_better,
                summaries["expected_loss_treatment"],
                summaries["expected_loss_control"],
            )

            result = BayesianResult(
                experiment_id=experiment_id,
                metric_name=metric_name,
                control_metric=float(np.mean(control_samples)),
                treatment_metric=float(np.mean(treatment_samples)),
                probability_treatment_better=prob_better,
                expected_loss_treatment=summaries["expected_loss_treatment"],
                expected_loss_control=summaries["expected_loss_control"],
                posterior_mean_lift=summaries["posterior_mean_lift"],
                posterior_std_lift=summaries["posterior_std_lift"],
                credible_interval_95=summaries["credible_interval_95"],
                rope=summaries["rope"],
                probability_in_rope=summaries["probability_in_rope"],
                decision=decision,
                method="pymc_mcmc",
                sample_size_control=control_n,
                sample_size_treatment=treatment_n,
                n_samples=len(control_samples),
                bayes_factor=self._bayes_factor_ratio(prob_better),
                diagnostics=diagnostics,
            )
            return result

        except Exception as exc:  # pylint: disable=broad-except
            warnings.warn(
                f"PyMC sampling failed ({exc!r}). "
                "Falling back to analytical Beta-Binomial.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.analyze_proportion_analytical(
                control_successes, control_n,
                treatment_successes, treatment_n,
                metric_name, experiment_id,
                prior_alpha, prior_beta,
            )

    # ── PyMC Normal-Normal (continuous) ───────────────────────────────────────

    def analyze_continuous_pymc(
        self,
        control_values: np.ndarray | pd.Series,
        treatment_values: np.ndarray | pd.Series,
        metric_name: str,
        experiment_id: str,
    ) -> BayesianResult:
        """
        PyMC Normal-Normal model for continuous metrics.

        Model specification:
          mu_group    ~ Normal(pooled_mean, 2 * pooled_std)   [weakly informative]
          sigma_group ~ HalfNormal(pooled_std)
          y_group     ~ Normal(mu_group, sigma_group)

        Falls back to a frequentist-inspired Normal approximation (using sample means
        and the CLT) when PyMC is not available.

        Parameters
        ----------
        control_values, treatment_values:
            Raw observed values for each variant.
        metric_name:
            Used for ROPE look-up and labelling.
        experiment_id:
            Identifier propagated into the result object.
        """
        control_values = np.asarray(control_values, dtype=float)
        treatment_values = np.asarray(treatment_values, dtype=float)

        control_n = len(control_values)
        treatment_n = len(treatment_values)

        pooled = np.concatenate([control_values, treatment_values])
        pooled_mean = float(np.mean(pooled))
        pooled_std = float(np.std(pooled)) or 1.0  # guard against constant data

        if not _PYMC_AVAILABLE:
            warnings.warn(
                "PyMC is not installed — using Normal approximation for continuous metric.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._analyze_continuous_normal_approx(
                control_values, treatment_values,
                metric_name, experiment_id,
                pooled_mean, pooled_std,
            )

        try:
            with pm.Model() as _model:  # noqa: F841
                # Weakly informative priors
                mu_c = pm.Normal("mu_c", mu=pooled_mean, sigma=pooled_std * 2.0)
                mu_t = pm.Normal("mu_t", mu=pooled_mean, sigma=pooled_std * 2.0)
                sigma_c = pm.HalfNormal("sigma_c", sigma=pooled_std)
                sigma_t = pm.HalfNormal("sigma_t", sigma=pooled_std)

                # Likelihoods
                pm.Normal("obs_c", mu=mu_c, sigma=sigma_c, observed=control_values)
                pm.Normal("obs_t", mu=mu_t, sigma=sigma_t, observed=treatment_values)

                # Derived
                _delta = pm.Deterministic("delta", mu_t - mu_c)  # noqa: F841

                idata = pm.sample(
                    draws=MCMC_DRAWS,
                    tune=MCMC_TUNE,
                    chains=MCMC_CHAINS,
                    target_accept=MCMC_TARGET_ACCEPT,
                    progressbar=False,
                    random_seed=42,
                )

            # ── convergence check ─────────────────────────────────────────────
            rhat_vals = az.rhat(idata)
            ess_vals = az.ess(idata)

            diagnostics: dict = {}
            converged = True

            for var in ("mu_c", "mu_t"):
                rhat = float(rhat_vals[var].values)
                ess = float(ess_vals[var].values)
                diagnostics[f"rhat_{var}"] = rhat
                diagnostics[f"ess_{var}"] = ess
                if rhat >= RHAT_THRESHOLD or ess < MIN_ESS:
                    warnings.warn(
                        f"Convergence issue for {var}: R-hat={rhat:.4f}, ESS={ess:.1f}. "
                        "Falling back to Normal approximation.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    converged = False

            if not converged:
                return self._analyze_continuous_normal_approx(
                    control_values, treatment_values,
                    metric_name, experiment_id,
                    pooled_mean, pooled_std,
                )

            # ── extract posterior ─────────────────────────────────────────────
            post = idata.posterior
            control_samples = post["mu_c"].values.flatten()
            treatment_samples = post["mu_t"].values.flatten()

            summaries = self._lift_summaries(control_samples, treatment_samples, metric_name)
            prob_better = summaries["prob_better"]
            decision = self._compute_decision(
                prob_better,
                summaries["expected_loss_treatment"],
                summaries["expected_loss_control"],
            )

            return BayesianResult(
                experiment_id=experiment_id,
                metric_name=metric_name,
                control_metric=float(np.mean(control_samples)),
                treatment_metric=float(np.mean(treatment_samples)),
                probability_treatment_better=prob_better,
                expected_loss_treatment=summaries["expected_loss_treatment"],
                expected_loss_control=summaries["expected_loss_control"],
                posterior_mean_lift=summaries["posterior_mean_lift"],
                posterior_std_lift=summaries["posterior_std_lift"],
                credible_interval_95=summaries["credible_interval_95"],
                rope=summaries["rope"],
                probability_in_rope=summaries["probability_in_rope"],
                decision=decision,
                method="pymc_mcmc",
                sample_size_control=control_n,
                sample_size_treatment=treatment_n,
                n_samples=len(control_samples),
                bayes_factor=self._bayes_factor_ratio(prob_better),
                diagnostics=diagnostics,
            )

        except Exception as exc:  # pylint: disable=broad-except
            warnings.warn(
                f"PyMC sampling failed ({exc!r}). "
                "Falling back to Normal approximation.",
                RuntimeWarning,
                stacklevel=2,
            )
            return self._analyze_continuous_normal_approx(
                control_values, treatment_values,
                metric_name, experiment_id,
                pooled_mean, pooled_std,
            )

    def _analyze_continuous_normal_approx(
        self,
        control_values: np.ndarray,
        treatment_values: np.ndarray,
        metric_name: str,
        experiment_id: str,
        pooled_mean: float,
        pooled_std: float,
    ) -> BayesianResult:
        """
        CLT-based Normal approximation for continuous metrics (PyMC fallback).

        Posterior for each group mean:
          mu_group | data ~ Normal(x_bar_group, se_group)
        where se_group = sample_std / sqrt(n).

        Draws self.mc_samples from each posterior to produce the same output
        structure as the PyMC path.
        """
        control_n = len(control_values)
        treatment_n = len(treatment_values)

        mu_c = float(np.mean(control_values))
        mu_t = float(np.mean(treatment_values))
        se_c = float(np.std(control_values, ddof=1)) / np.sqrt(control_n)
        se_t = float(np.std(treatment_values, ddof=1)) / np.sqrt(treatment_n)

        control_samples = self._rng.normal(mu_c, se_c, size=self.mc_samples)
        treatment_samples = self._rng.normal(mu_t, se_t, size=self.mc_samples)

        summaries = self._lift_summaries(control_samples, treatment_samples, metric_name)
        prob_better = summaries["prob_better"]
        decision = self._compute_decision(
            prob_better,
            summaries["expected_loss_treatment"],
            summaries["expected_loss_control"],
        )

        return BayesianResult(
            experiment_id=experiment_id,
            metric_name=metric_name,
            control_metric=mu_c,
            treatment_metric=mu_t,
            probability_treatment_better=prob_better,
            expected_loss_treatment=summaries["expected_loss_treatment"],
            expected_loss_control=summaries["expected_loss_control"],
            posterior_mean_lift=summaries["posterior_mean_lift"],
            posterior_std_lift=summaries["posterior_std_lift"],
            credible_interval_95=summaries["credible_interval_95"],
            rope=summaries["rope"],
            probability_in_rope=summaries["probability_in_rope"],
            decision=decision,
            method="normal_approximation",
            sample_size_control=control_n,
            sample_size_treatment=treatment_n,
            n_samples=self.mc_samples,
            bayes_factor=self._bayes_factor_ratio(prob_better),
        )

    # ── high-level dispatcher ─────────────────────────────────────────────────

    def analyze(
        self,
        experiment_data: pd.DataFrame,
        variants: list[str],
        metrics: list[str],
        experiment_id: str,
        control_variant: Optional[str] = None,
        use_pymc: bool = False,
    ) -> dict:
        """
        Analyse one or more metrics for a two-variant experiment.

        Parameters
        ----------
        experiment_data:
            DataFrame with at least a ``variant`` column and one column per metric.
        variants:
            List of variant labels present in the ``variant`` column.
            Must contain exactly two elements (or ``control_variant`` plus one other).
        metrics:
            Metric column names to analyse.
        experiment_id:
            Identifier for labelling results.
        control_variant:
            Name of the control variant.  If None, the first element of ``variants``
            is used as control.
        use_pymc:
            If True, attempt to use PyMC MCMC for all analyses
            (proportion metrics included).  Falls back to analytical automatically.

        Returns
        -------
        dict with keys:
          "results"           : {metric_name: BayesianResult}
          "experiment_id"     : str
          "control_variant"   : str
          "treatment_variant" : str
        """
        if len(variants) < 2:
            raise ValueError("analyze() requires at least two variants.")

        control_variant = control_variant or variants[0]
        treatment_variants = [v for v in variants if v != control_variant]

        if len(treatment_variants) != 1:
            raise ValueError(
                "analyze() supports exactly one treatment variant per call. "
                f"Found: {treatment_variants}"
            )
        treatment_variant = treatment_variants[0]

        ctrl_df = experiment_data.loc[
            experiment_data["variant"] == control_variant
        ]
        treat_df = experiment_data.loc[
            experiment_data["variant"] == treatment_variant
        ]

        results: dict[str, BayesianResult] = {}

        for metric in metrics:
            if metric not in experiment_data.columns:
                warnings.warn(
                    f"Metric '{metric}' not found in experiment_data — skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            is_proportion = metric in PROPORTION_METRICS

            if is_proportion:
                ctrl_vals = ctrl_df[metric].dropna()
                treat_vals = treat_df[metric].dropna()

                ctrl_n = len(ctrl_vals)
                treat_n = len(treat_vals)
                ctrl_successes = int(np.round(ctrl_vals.mean() * ctrl_n))
                treat_successes = int(np.round(treat_vals.mean() * treat_n))

                if use_pymc:
                    results[metric] = self.analyze_proportion_pymc(
                        ctrl_successes, ctrl_n,
                        treat_successes, treat_n,
                        metric, experiment_id,
                    )
                else:
                    results[metric] = self.analyze_proportion_analytical(
                        ctrl_successes, ctrl_n,
                        treat_successes, treat_n,
                        metric, experiment_id,
                    )

            else:
                # Continuous metric
                ctrl_vals = ctrl_df[metric].dropna().to_numpy(dtype=float)
                treat_vals = treat_df[metric].dropna().to_numpy(dtype=float)

                if use_pymc:
                    results[metric] = self.analyze_continuous_pymc(
                        ctrl_vals, treat_vals, metric, experiment_id
                    )
                else:
                    # Non-PyMC path: use Normal approximation directly
                    pooled = np.concatenate([ctrl_vals, treat_vals])
                    pooled_mean = float(np.mean(pooled))
                    pooled_std = float(np.std(pooled)) or 1.0
                    results[metric] = self._analyze_continuous_normal_approx(
                        ctrl_vals, treat_vals, metric, experiment_id,
                        pooled_mean, pooled_std,
                    )

        return {
            "results": results,
            "experiment_id": experiment_id,
            "control_variant": control_variant,
            "treatment_variant": treatment_variant,
        }

    # ── reporting helpers ─────────────────────────────────────────────────────

    def results_to_dataframe(self, results: dict) -> pd.DataFrame:
        """
        Flatten a results dict (as returned by ``analyze()``) into a tidy DataFrame.

        Each row corresponds to one metric.
        """
        rows = []
        for metric_name, res in results["results"].items():
            rows.append(
                {
                    "experiment_id": res.experiment_id,
                    "metric_name": res.metric_name,
                    "control_variant": results["control_variant"],
                    "treatment_variant": results["treatment_variant"],
                    "control_metric": res.control_metric,
                    "treatment_metric": res.treatment_metric,
                    "probability_treatment_better": res.probability_treatment_better,
                    "expected_loss_treatment": res.expected_loss_treatment,
                    "expected_loss_control": res.expected_loss_control,
                    "posterior_mean_lift": res.posterior_mean_lift,
                    "posterior_std_lift": res.posterior_std_lift,
                    "ci_95_lower": res.credible_interval_95[0],
                    "ci_95_upper": res.credible_interval_95[1],
                    "rope_lower": res.rope[0],
                    "rope_upper": res.rope[1],
                    "probability_in_rope": res.probability_in_rope,
                    "bayes_factor": res.bayes_factor,
                    "decision": res.decision,
                    "method": res.method,
                    "sample_size_control": res.sample_size_control,
                    "sample_size_treatment": res.sample_size_treatment,
                    "n_samples": res.n_samples,
                }
            )
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone report formatter
# ─────────────────────────────────────────────────────────────────────────────


def format_bayesian_report(result: BayesianResult) -> str:
    """
    Return a human-readable string summarising a single ``BayesianResult``.

    Intended for console output, logging, or embedding in larger reports.
    """
    ci_lo, ci_hi = result.credible_interval_95
    rope_lo, rope_hi = result.rope

    lines = [
        "=" * 68,
        f"  Bayesian A/B Report — {result.experiment_id}  |  {result.metric_name}",
        "=" * 68,
        f"  Method            : {result.method}",
        f"  N (control)       : {result.sample_size_control:,}",
        f"  N (treatment)     : {result.sample_size_treatment:,}",
        f"  MC / MCMC samples : {result.n_samples:,}",
        "-" * 68,
        f"  Control mean      : {result.control_metric:.5f}",
        f"  Treatment mean    : {result.treatment_metric:.5f}",
        "-" * 68,
        f"  P(treatment > control) : {result.probability_treatment_better:.4f}",
        f"  Posterior lift mean    : {result.posterior_mean_lift:+.5f}",
        f"  Posterior lift std     : {result.posterior_std_lift:.5f}",
        f"  95% HDI                : [{ci_lo:+.5f}, {ci_hi:+.5f}]",
        "-" * 68,
        f"  ROPE                   : [{rope_lo:+.5f}, {rope_hi:+.5f}]",
        f"  P(lift in ROPE)        : {result.probability_in_rope:.4f}",
        "-" * 68,
        f"  Expected loss (treatment) : {result.expected_loss_treatment:.6f}",
        f"  Expected loss (control)   : {result.expected_loss_control:.6f}",
        f"  Bayes Factor (BF10)       : {result.bayes_factor:.3f}",
        "=" * 68,
        f"  DECISION: {result.decision}",
        "=" * 68,
    ]

    if result.diagnostics:
        lines.insert(-2, "-" * 68)
        lines.insert(-2, "  MCMC Diagnostics:")
        for k, v in result.diagnostics.items():
            lines.insert(-2, f"    {k:<30}: {v:.4f}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# __main__ — quick demo using synthetic EXP-001-style data
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Campaign Experimentation Framework — Bayesian A/B Analysis demo")
    print(f"PyMC available: {_PYMC_AVAILABLE}")
    print()

    # ── Simulate EXP-001: Email Subject Line Personalization ─────────────────
    # Primary metric: open_rate   (proportion)
    # Secondary:      click_rate, conversion_rate

    rng = np.random.default_rng(seed=42)

    N_PER_VARIANT = 25_000

    # Control: open_rate ~ 0.22, click_rate ~ 0.04, conversion_rate ~ 0.015
    # Treatment: +2 pp open, +0.5 pp click, +0.3 pp conversion
    ctrl_opens = rng.binomial(1, 0.22, size=N_PER_VARIANT)
    ctrl_clicks = rng.binomial(1, 0.040, size=N_PER_VARIANT)
    ctrl_conv = rng.binomial(1, 0.015, size=N_PER_VARIANT)

    trt_opens = rng.binomial(1, 0.24, size=N_PER_VARIANT)
    trt_clicks = rng.binomial(1, 0.045, size=N_PER_VARIANT)
    trt_conv = rng.binomial(1, 0.018, size=N_PER_VARIANT)

    ctrl_df = pd.DataFrame(
        {
            "variant": "control",
            "open_rate": ctrl_opens,
            "click_rate": ctrl_clicks,
            "conversion_rate": ctrl_conv,
        }
    )
    trt_df = pd.DataFrame(
        {
            "variant": "treatment",
            "open_rate": trt_opens,
            "click_rate": trt_clicks,
            "conversion_rate": trt_conv,
        }
    )
    exp_data = pd.concat([ctrl_df, trt_df], ignore_index=True)

    # ── Run analytical Bayesian analysis ─────────────────────────────────────
    analyser = BayesianABTest()

    output = analyser.analyze(
        experiment_data=exp_data,
        variants=["control", "treatment"],
        metrics=["open_rate", "click_rate", "conversion_rate"],
        experiment_id="EXP-001",
        control_variant="control",
        use_pymc=False,
    )

    for metric, result in output["results"].items():
        print(format_bayesian_report(result))
        print()

    # ── Tabular summary ───────────────────────────────────────────────────────
    df_summary = analyser.results_to_dataframe(output)
    print("Tabular summary:")
    print(
        df_summary[
            [
                "metric_name",
                "control_metric",
                "treatment_metric",
                "probability_treatment_better",
                "posterior_mean_lift",
                "bayes_factor",
                "decision",
            ]
        ].to_string(index=False)
    )
