"""
Holdout / Incrementality Analysis with Difference-in-Differences
for the Campaign Experimentation & Lift Measurement Framework.

This module provides:
  - IncrementalityResult  — dataclass capturing all lift, DID, and
                             cost-efficiency statistics for one metric
  - BalanceCheckResult    — dataclass for a single covariate balance test
  - HoldoutAnalyzer       — the primary analysis engine

Typical usage
-------------
    from src.holdout_analysis import HoldoutAnalyzer

    analyzer = HoldoutAnalyzer()
    result   = analyzer.analyze(
        data=df,
        variant_col="variant",
        metric_col="converted",
        experiment_id="holdout_q1_2026",
        period_col="period",
        campaign_cost=50_000.0,
        revenue_col="revenue",
    )
    print(analyzer.format_holdout_report(result))
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Config import — fall back to hard-coded defaults so the module can be used
# standalone during early development.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from config import (
        SIGNIFICANCE_LEVEL,
        CONFIDENCE_LEVEL,
        CONTAMINATION_FLAG_THRESHOLD,
        PARALLEL_TRENDS_ALPHA,
    )
except ImportError:  # pragma: no cover
    warnings.warn(
        "config.py not found — using built-in defaults for holdout_analysis.",
        ImportWarning,
        stacklevel=2,
    )
    SIGNIFICANCE_LEVEL = 0.05
    CONFIDENCE_LEVEL = 0.95
    CONTAMINATION_FLAG_THRESHOLD = 0.02
    PARALLEL_TRENDS_ALPHA = 0.10


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class IncrementalityResult:
    """All statistics produced by a single holdout/incrementality analysis."""

    experiment_id: str
    metric_name: str

    # Core rates
    exposed_conversion_rate: float
    holdout_conversion_rate: float

    # Lift statistics
    raw_lift: float                          # exposed_rate - holdout_rate
    relative_lift_pct: float                 # raw_lift / holdout_rate * 100
    incremental_conversions: int
    total_exposed: int
    total_holdout: int

    # Frequentist inference (simple difference)
    confidence_interval_lift: Tuple[float, float]   # 95 % CI on raw lift
    p_value: float
    is_significant: bool

    # Difference-in-Differences
    did_estimator: float
    did_confidence_interval: Tuple[float, float]
    did_p_value: float

    # Business metrics (None when inputs not supplied)
    cost_per_incremental_conversion: Optional[float]
    incremental_revenue: Optional[float]
    incremental_roas: Optional[float]

    # Treatment efficiency
    nnt: float          # Number Needed to Treat = 1 / raw_lift

    # Which estimator was primary
    method: str         # "simple_difference" | "did"


@dataclass
class BalanceCheckResult:
    """Result of one covariate balance check between exposed and holdout."""

    variable: str
    p_value: float
    test_statistic: float
    is_balanced: bool                        # p_value > 0.05
    exposed_distribution: Dict
    holdout_distribution: Dict


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _wilson_ci(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    half_width = (z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))) / denom
    return (float(centre - half_width), float(centre + half_width))


def _lift_ci(
    rate_a: float,
    n_a: int,
    rate_b: float,
    n_b: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Approximate CI for the difference in two proportions (exposed - holdout)."""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    se = np.sqrt(rate_a * (1 - rate_a) / n_a + rate_b * (1 - rate_b) / n_b)
    diff = rate_a - rate_b
    return (float(diff - z * se), float(diff + z * se))


def _is_binary(series: pd.Series) -> bool:
    """Return True if series contains only 0/1 (or bool) values."""
    unique = series.dropna().unique()
    return set(unique).issubset({0, 1, True, False})


# ---------------------------------------------------------------------------
# Main analyser class
# ---------------------------------------------------------------------------

class HoldoutAnalyzer:
    """
    Holdout / incrementality analyser supporting:
      - Simple difference-in-conversion-rates
      - Difference-in-Differences (DiD) via OLS
      - Covariate balance checks (chi-squared / Welch's t)
      - Parallel-trends test on pre-period panel data
      - Contamination detection
      - Segment-level lift decomposition
    """

    def __init__(
        self,
        significance_level: float = SIGNIFICANCE_LEVEL,
        confidence_level: float = CONFIDENCE_LEVEL,
    ) -> None:
        self.significance_level = significance_level
        self.confidence_level = confidence_level

    # ------------------------------------------------------------------
    # Balance check
    # ------------------------------------------------------------------

    def check_balance(
        self,
        data: pd.DataFrame,
        variant_col: str,
        covariates: List[str],
    ) -> List[BalanceCheckResult]:
        """
        Test whether covariates are balanced between exposed and holdout groups.

        Categorical covariates: chi-squared test of independence.
        Continuous covariates : Welch's independent-samples t-test.

        Returns
        -------
        List[BalanceCheckResult] — one entry per covariate.
        """
        results: List[BalanceCheckResult] = []

        exposed_mask = data[variant_col] == "exposed"
        holdout_mask = data[variant_col] == "holdout"

        for cov in covariates:
            col = data[cov].dropna()
            exposed_vals = data.loc[exposed_mask, cov].dropna()
            holdout_vals = data.loc[holdout_mask, cov].dropna()

            # Decide test type
            if (
                isinstance(col.dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(col)
                or col.nunique() <= 10
            ):
                # Chi-squared test
                ct = pd.crosstab(data[variant_col], data[cov])
                chi2, p_val, _, _ = stats.chi2_contingency(ct)
                test_stat = float(chi2)

                exposed_dist = (
                    exposed_vals.value_counts(normalize=True).to_dict()
                )
                holdout_dist = (
                    holdout_vals.value_counts(normalize=True).to_dict()
                )
            else:
                # Welch's t-test
                t_stat, p_val = stats.ttest_ind(
                    exposed_vals.astype(float),
                    holdout_vals.astype(float),
                    equal_var=False,
                )
                test_stat = float(t_stat)

                exposed_dist = {
                    "mean": float(exposed_vals.mean()),
                    "std": float(exposed_vals.std()),
                    "n": int(len(exposed_vals)),
                }
                holdout_dist = {
                    "mean": float(holdout_vals.mean()),
                    "std": float(holdout_vals.std()),
                    "n": int(len(holdout_vals)),
                }

            results.append(
                BalanceCheckResult(
                    variable=cov,
                    p_value=float(p_val),
                    test_statistic=test_stat,
                    is_balanced=float(p_val) > 0.05,
                    exposed_distribution=exposed_dist,
                    holdout_distribution=holdout_dist,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Parallel trends
    # ------------------------------------------------------------------

    def check_parallel_trends(
        self,
        panel_data: pd.DataFrame,
        variant_col: str,
        period_col: str,
        metric_col: str,
        pre_period_value: str = "baseline",
    ) -> Dict:
        """
        Test the parallel-trends assumption using pre-period data only.

        The regression model is:

            metric ~ time + variant + time * variant

        where *time* is a numeric encoding of the pre-period ordering.
        If the interaction p-value < PARALLEL_TRENDS_ALPHA the assumption
        is flagged as violated.

        Returns
        -------
        dict with keys:
          metric_name, p_value, coefficient, is_parallel, plot_data
        """
        pre_data = panel_data[panel_data[period_col] == pre_period_value].copy()

        if pre_data.empty:
            warnings.warn(
                f"No rows found for pre_period_value='{pre_period_value}' "
                f"in column '{period_col}'. Parallel-trends check skipped.",
                UserWarning,
                stacklevel=2,
            )
            return {
                "metric_name": metric_col,
                "p_value": np.nan,
                "coefficient": np.nan,
                "is_parallel": None,
                "plot_data": pd.DataFrame(),
            }

        # Encode time as numeric rank of unique period values
        time_order = sorted(pre_data[period_col].unique())
        time_map = {t: i for i, t in enumerate(time_order)}
        pre_data = pre_data.copy()
        pre_data["_time"] = pre_data[period_col].map(time_map)
        pre_data["_variant_binary"] = (
            pre_data[variant_col] == "exposed"
        ).astype(int)

        formula = f"{metric_col} ~ _time + _variant_binary + _time:_variant_binary"
        try:
            model = smf.ols(formula=formula, data=pre_data).fit()
            interaction_term = "_time:_variant_binary"
            p_value = float(model.pvalues.get(interaction_term, np.nan))
            coefficient = float(model.params.get(interaction_term, np.nan))
        except Exception as exc:
            warnings.warn(
                f"Parallel-trends OLS failed: {exc}",
                UserWarning,
                stacklevel=2,
            )
            p_value = np.nan
            coefficient = np.nan

        is_parallel: Optional[bool] = (
            None if np.isnan(p_value) else p_value >= PARALLEL_TRENDS_ALPHA
        )

        # Aggregate means per (period, variant) for plotting
        plot_data = (
            pre_data.groupby([period_col, variant_col])[metric_col]
            .mean()
            .reset_index()
        )

        return {
            "metric_name": metric_col,
            "p_value": p_value,
            "coefficient": coefficient,
            "is_parallel": is_parallel,
            "plot_data": plot_data,
        }

    # ------------------------------------------------------------------
    # Simple lift (cross-sectional)
    # ------------------------------------------------------------------

    def compute_simple_lift(
        self,
        data: pd.DataFrame,
        variant_col: str,
        metric_col: str,
        exposed_label: str = "exposed",
        holdout_label: str = "holdout",
    ) -> Dict:
        """
        Compute cross-sectional lift between exposed and holdout groups.

        For binary metrics: two-proportion z-test; Wilson CIs.
        For continuous metrics: Welch's t-test; normal approximation CI.

        Returns
        -------
        dict with keys:
          exposed_rate, holdout_rate, raw_lift, relative_lift_pct,
          p_value, ci_lower, ci_upper, is_binary, n_exposed, n_holdout
        """
        exposed_data = data.loc[data[variant_col] == exposed_label, metric_col].dropna()
        holdout_data = data.loc[data[variant_col] == holdout_label, metric_col].dropna()

        n_exposed = int(len(exposed_data))
        n_holdout = int(len(holdout_data))

        binary = _is_binary(exposed_data) and _is_binary(holdout_data)

        exposed_rate = float(exposed_data.mean())
        holdout_rate = float(holdout_data.mean())
        raw_lift = exposed_rate - holdout_rate
        relative_lift_pct = (
            (raw_lift / holdout_rate * 100) if holdout_rate != 0 else np.nan
        )

        if binary:
            # Two-proportion z-test
            count_exp = int(exposed_data.sum())
            count_hld = int(holdout_data.sum())

            # Pooled proportion under H0
            p_pool = (count_exp + count_hld) / (n_exposed + n_holdout)
            se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_exposed + 1 / n_holdout))
            z_stat = raw_lift / se_pool if se_pool > 0 else 0.0
            p_value = float(2 * stats.norm.sf(abs(z_stat)))

            # Wilson CI for the difference
            ci_exp = _wilson_ci(count_exp, n_exposed, self.confidence_level)
            ci_hld = _wilson_ci(count_hld, n_holdout, self.confidence_level)
            ci_lower, ci_upper = _lift_ci(
                exposed_rate, n_exposed, holdout_rate, n_holdout, self.confidence_level
            )
        else:
            # Welch's t-test for continuous metric
            t_stat, p_value = stats.ttest_ind(
                exposed_data.astype(float),
                holdout_data.astype(float),
                equal_var=False,
            )
            p_value = float(p_value)
            z = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
            se = np.sqrt(
                exposed_data.var(ddof=1) / n_exposed
                + holdout_data.var(ddof=1) / n_holdout
            )
            ci_lower = float(raw_lift - z * se)
            ci_upper = float(raw_lift + z * se)

        return {
            "exposed_rate": exposed_rate,
            "holdout_rate": holdout_rate,
            "raw_lift": raw_lift,
            "relative_lift_pct": relative_lift_pct,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "is_binary": binary,
            "n_exposed": n_exposed,
            "n_holdout": n_holdout,
        }

    # ------------------------------------------------------------------
    # Difference-in-Differences
    # ------------------------------------------------------------------

    def compute_did(
        self,
        panel_data: pd.DataFrame,
        variant_col: str,
        period_col: str,
        metric_col: str,
        exposed_label: str = "exposed",
        holdout_label: str = "holdout",
        pre_label: str = "baseline",
        post_label: str = "test",
    ) -> Dict:
        """
        Difference-in-Differences estimator via OLS.

        Model:
            metric ~ C(variant) + C(period) + C(variant):C(period)

        The interaction coefficient for (exposed × test) is the DiD estimate.

        Returns
        -------
        dict with keys:
          did_estimate, se, t_stat, p_value, ci_lower, ci_upper, means_table
        """
        df = panel_data[
            panel_data[variant_col].isin([exposed_label, holdout_label])
            & panel_data[period_col].isin([pre_label, post_label])
        ].copy()

        if df.empty:
            warnings.warn(
                "compute_did: no rows matching variant/period labels. "
                "Returning NaN DiD estimate.",
                UserWarning,
                stacklevel=2,
            )
            nan_result: Dict = dict(
                did_estimate=np.nan, se=np.nan, t_stat=np.nan,
                p_value=np.nan, ci_lower=np.nan, ci_upper=np.nan,
                means_table=pd.DataFrame(),
            )
            return nan_result

        # 2×2 means table
        means_table = (
            df.groupby([variant_col, period_col])[metric_col]
            .mean()
            .unstack(period_col)
            .reindex(index=[holdout_label, exposed_label])
        )
        # Ensure both period columns present
        for col in [pre_label, post_label]:
            if col not in means_table.columns:
                means_table[col] = np.nan

        # Manual DiD for reference
        try:
            manual_did = (
                (means_table.loc[exposed_label, post_label]
                 - means_table.loc[exposed_label, pre_label])
                - (means_table.loc[holdout_label, post_label]
                   - means_table.loc[holdout_label, pre_label])
            )
        except KeyError:
            manual_did = np.nan

        # OLS with interaction
        formula = (
            f"{metric_col} ~ C({variant_col}, Treatment('{holdout_label}'))"
            f" + C({period_col}, Treatment('{pre_label}'))"
            f" + C({variant_col}, Treatment('{holdout_label}')):C({period_col}, Treatment('{pre_label}'))"
        )
        try:
            model = smf.ols(formula=formula, data=df).fit()

            # Locate the interaction parameter — its name contains both
            # the exposed label and the post label
            interaction_params = [
                k for k in model.params.index
                if exposed_label in k and post_label in k
            ]
            if not interaction_params:
                raise ValueError(
                    f"Interaction term not found in model params: {list(model.params.index)}"
                )
            term = interaction_params[0]

            did_estimate = float(model.params[term])
            se = float(model.bse[term])
            t_stat = float(model.tvalues[term])
            p_value = float(model.pvalues[term])
            ci = model.conf_int(alpha=1 - self.confidence_level).loc[term]
            ci_lower = float(ci.iloc[0])
            ci_upper = float(ci.iloc[1])

        except Exception as exc:
            warnings.warn(
                f"compute_did OLS failed ({exc}). Using manual DiD; "
                "SE/t/p are unavailable.",
                UserWarning,
                stacklevel=2,
            )
            did_estimate = float(manual_did) if not np.isnan(manual_did) else np.nan
            se = np.nan
            t_stat = np.nan
            p_value = np.nan
            ci_lower = np.nan
            ci_upper = np.nan

        return {
            "did_estimate": did_estimate,
            "se": se,
            "t_stat": t_stat,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "means_table": means_table,
        }

    # ------------------------------------------------------------------
    # Master analyse method
    # ------------------------------------------------------------------

    def analyze(
        self,
        data: pd.DataFrame,
        variant_col: str,
        metric_col: str,
        experiment_id: str,
        period_col: Optional[str] = None,
        campaign_cost: Optional[float] = None,
        revenue_col: Optional[str] = None,
        covariates: Optional[List[str]] = None,
        exposed_label: str = "exposed",
        holdout_label: str = "holdout",
    ) -> IncrementalityResult:
        """
        Full holdout / incrementality analysis.

        Parameters
        ----------
        data           : DataFrame with one row per unit
        variant_col    : column name containing group labels
        metric_col     : outcome metric column
        experiment_id  : identifier stored in the result
        period_col     : if supplied, DiD is run in addition to simple lift
        campaign_cost  : total campaign spend (for CPiC calculation)
        revenue_col    : column containing per-unit revenue values
        covariates     : columns to include in balance check
        exposed_label  : label identifying the exposed group
        holdout_label  : label identifying the holdout group

        Returns
        -------
        IncrementalityResult
        """
        # ── Simple lift ───────────────────────────────────────────────
        simple = self.compute_simple_lift(
            data, variant_col, metric_col, exposed_label, holdout_label
        )

        exposed_rate = simple["exposed_rate"]
        holdout_rate = simple["holdout_rate"]
        raw_lift = simple["raw_lift"]
        relative_lift_pct = simple["relative_lift_pct"]
        n_exposed = simple["n_exposed"]
        n_holdout = simple["n_holdout"]
        p_value = simple["p_value"]
        ci_lower = simple["ci_lower"]
        ci_upper = simple["ci_upper"]
        is_significant = p_value < self.significance_level

        incremental_conversions = int(round(raw_lift * n_exposed))

        # NNT
        nnt = (1.0 / raw_lift) if raw_lift > 0 else float("inf")

        # ── DiD (optional) ────────────────────────────────────────────
        did_estimator = float("nan")
        did_ci = (float("nan"), float("nan"))
        did_p_value = float("nan")
        method = "simple_difference"

        if period_col is not None:
            did_result = self.compute_did(
                data, variant_col, period_col, metric_col,
                exposed_label, holdout_label,
            )
            did_estimator = did_result["did_estimate"]
            did_ci = (did_result["ci_lower"], did_result["ci_upper"])
            did_p_value = did_result["p_value"]
            method = "did"

            # Prefer DiD for is_significant when available
            if not np.isnan(did_p_value):
                is_significant = did_p_value < self.significance_level

        # ── Business metrics ──────────────────────────────────────────
        cost_per_incremental = None
        if campaign_cost is not None and incremental_conversions > 0:
            cost_per_incremental = campaign_cost / incremental_conversions

        incremental_revenue: Optional[float] = None
        incremental_roas: Optional[float] = None
        if revenue_col is not None:
            exposed_rev = data.loc[data[variant_col] == exposed_label, revenue_col].mean()
            holdout_rev = data.loc[data[variant_col] == holdout_label, revenue_col].mean()
            incremental_revenue = float((exposed_rev - holdout_rev) * n_exposed)
            if campaign_cost is not None and campaign_cost > 0:
                incremental_roas = incremental_revenue / campaign_cost

        # ── Balance check (side-effect: warnings, not stored in result) ──
        if covariates:
            balance_results = self.check_balance(data, variant_col, covariates)
            imbalanced = [b.variable for b in balance_results if not b.is_balanced]
            if imbalanced:
                warnings.warn(
                    f"Balance check failed for covariates: {imbalanced}. "
                    "Interpret lift estimates with caution.",
                    UserWarning,
                    stacklevel=2,
                )

        return IncrementalityResult(
            experiment_id=experiment_id,
            metric_name=metric_col,
            exposed_conversion_rate=exposed_rate,
            holdout_conversion_rate=holdout_rate,
            raw_lift=raw_lift,
            relative_lift_pct=float(relative_lift_pct) if not np.isnan(relative_lift_pct) else float("nan"),
            incremental_conversions=incremental_conversions,
            total_exposed=n_exposed,
            total_holdout=n_holdout,
            confidence_interval_lift=(ci_lower, ci_upper),
            p_value=p_value,
            is_significant=is_significant,
            did_estimator=did_estimator,
            did_confidence_interval=did_ci,
            did_p_value=did_p_value,
            cost_per_incremental_conversion=cost_per_incremental,
            incremental_revenue=incremental_revenue,
            incremental_roas=incremental_roas,
            nnt=nnt,
            method=method,
        )

    # ------------------------------------------------------------------
    # Contamination check
    # ------------------------------------------------------------------

    def check_contamination(
        self,
        data: pd.DataFrame,
        variant_col: str,
        exposure_indicator_col: str,
        holdout_label: str = "holdout",
    ) -> Dict:
        """
        Detect holdout contamination: individuals assigned to holdout who
        actually received campaign materials.

        Parameters
        ----------
        data                   : DataFrame with one row per unit
        variant_col            : column containing group assignments
        exposure_indicator_col : 1/True = unit received campaign materials
        holdout_label          : label for holdout group in variant_col

        Returns
        -------
        dict with keys:
          contamination_rate, contaminated_ids, is_contaminated, message
        """
        holdout_mask = data[variant_col] == holdout_label
        holdout_data = data[holdout_mask]
        total_holdout = int(len(holdout_data))

        if total_holdout == 0:
            return {
                "contamination_rate": 0.0,
                "contaminated_ids": [],
                "is_contaminated": False,
                "message": "No holdout units found.",
            }

        contaminated = holdout_data[holdout_data[exposure_indicator_col] == 1]
        contamination_rate = len(contaminated) / total_holdout
        is_contaminated = contamination_rate > CONTAMINATION_FLAG_THRESHOLD

        # Return index values as IDs when the DataFrame index is meaningful,
        # otherwise return positional indices
        contaminated_ids: list = (
            contaminated.index.tolist()
            if hasattr(contaminated.index, "name")
            else list(range(len(contaminated)))
        )

        if is_contaminated:
            message = (
                f"WARNING: {contamination_rate:.1%} of holdout units "
                f"({len(contaminated)}/{total_holdout}) received campaign exposure. "
                f"Threshold is {CONTAMINATION_FLAG_THRESHOLD:.1%}. "
                "Lift estimates are likely understated."
            )
        else:
            message = (
                f"Contamination rate {contamination_rate:.2%} is within "
                f"acceptable threshold ({CONTAMINATION_FLAG_THRESHOLD:.1%})."
            )

        return {
            "contamination_rate": float(contamination_rate),
            "contaminated_ids": contaminated_ids,
            "is_contaminated": is_contaminated,
            "message": message,
        }

    # ------------------------------------------------------------------
    # Formatted text report
    # ------------------------------------------------------------------

    def format_holdout_report(
        self,
        result: IncrementalityResult,
        balance_results: Optional[List[BalanceCheckResult]] = None,
    ) -> str:
        """
        Return a full-width human-readable report for an IncrementalityResult.
        """
        sep = "=" * 72
        thin = "-" * 72
        lines: List[str] = [
            sep,
            f"  HOLDOUT / INCREMENTALITY REPORT",
            f"  Experiment : {result.experiment_id}",
            f"  Metric     : {result.metric_name}",
            f"  Method     : {result.method}",
            sep,
            "",
            "SAMPLE SIZES",
            thin,
            f"  Exposed   : {result.total_exposed:>10,}",
            f"  Holdout   : {result.total_holdout:>10,}",
            "",
            "CONVERSION RATES",
            thin,
            f"  Exposed rate  : {result.exposed_conversion_rate:>8.4f}  "
            f"({result.exposed_conversion_rate * 100:.2f}%)",
            f"  Holdout rate  : {result.holdout_conversion_rate:>8.4f}  "
            f"({result.holdout_conversion_rate * 100:.2f}%)",
            "",
            "LIFT STATISTICS",
            thin,
            f"  Raw lift          : {result.raw_lift:>+.4f}",
            f"  Relative lift     : {result.relative_lift_pct:>+.2f}%",
            f"  {int(self.confidence_level * 100)}% CI (lift)  : "
            f"[{result.confidence_interval_lift[0]:+.4f}, "
            f"{result.confidence_interval_lift[1]:+.4f}]",
            f"  p-value           : {result.p_value:.4f}",
            f"  Significant       : {'YES  ***' if result.is_significant else 'no'}",
            f"  Incr. conversions : {result.incremental_conversions:>+,}",
            f"  NNT               : {result.nnt:.1f}" if np.isfinite(result.nnt)
            else "  NNT               : inf (no positive lift)",
        ]

        # DiD block
        if not np.isnan(result.did_estimator):
            lines += [
                "",
                "DIFFERENCE-IN-DIFFERENCES",
                thin,
                f"  DiD estimate : {result.did_estimator:>+.4f}",
                f"  DiD {int(self.confidence_level * 100)}% CI  : "
                f"[{result.did_confidence_interval[0]:+.4f}, "
                f"{result.did_confidence_interval[1]:+.4f}]",
                f"  DiD p-value  : {result.did_p_value:.4f}",
            ]

        # Cost / revenue block
        if any(
            v is not None
            for v in [
                result.cost_per_incremental_conversion,
                result.incremental_revenue,
                result.incremental_roas,
            ]
        ):
            lines += ["", "COST EFFICIENCY", thin]
            if result.cost_per_incremental_conversion is not None:
                lines.append(
                    f"  Cost per incr. conversion : "
                    f"${result.cost_per_incremental_conversion:>10,.2f}"
                )
            if result.incremental_revenue is not None:
                lines.append(
                    f"  Incremental revenue       : "
                    f"${result.incremental_revenue:>10,.2f}"
                )
            if result.incremental_roas is not None:
                lines.append(
                    f"  Incremental ROAS          : "
                    f"{result.incremental_roas:>8.2f}x"
                )

        # Balance check block
        if balance_results:
            lines += ["", "COVARIATE BALANCE", thin]
            for br in balance_results:
                status = "BALANCED" if br.is_balanced else "IMBALANCED !"
                lines.append(
                    f"  {br.variable:<28s}  p={br.p_value:.4f}  [{status}]"
                )

        lines += ["", sep, ""]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Segment lift decomposition
    # ------------------------------------------------------------------

    def compute_segment_lift(
        self,
        data: pd.DataFrame,
        variant_col: str,
        metric_col: str,
        segment_col: str,
        experiment_id: str,
    ) -> pd.DataFrame:
        """
        Compute per-segment lift and index vs. the overall experiment lift.

        Recommendations
        ---------------
        - INCREASE_INVESTMENT : segment lift  > 1.5x overall lift
        - MAINTAIN            : 0.5x <= segment lift <= 1.5x overall
        - REDUCE_INVESTMENT   : segment lift < 0.5x overall lift OR not significant
        - EXCLUDE             : negative segment lift

        Returns
        -------
        pd.DataFrame with columns:
            segment, n_exposed, n_holdout,
            segment_lift, segment_lift_ci_lower, segment_lift_ci_upper,
            p_value, is_significant, index_vs_overall, recommendation
        """
        # Compute overall lift for indexing
        overall = self.compute_simple_lift(data, variant_col, metric_col)
        overall_lift = overall["raw_lift"]

        rows: List[Dict] = []
        for segment_val in sorted(data[segment_col].dropna().unique()):
            seg_data = data[data[segment_col] == segment_val]

            n_exp = int((seg_data[variant_col] == "exposed").sum())
            n_hld = int((seg_data[variant_col] == "holdout").sum())

            if n_exp == 0 or n_hld == 0:
                continue

            seg_stats = self.compute_simple_lift(seg_data, variant_col, metric_col)
            seg_lift = seg_stats["raw_lift"]
            p_val = seg_stats["p_value"]
            sig = p_val < self.significance_level

            index_vs_overall = (
                (seg_lift / overall_lift) if overall_lift != 0 else np.nan
            )

            # Recommendation logic
            if seg_lift < 0:
                rec = "EXCLUDE"
            elif not sig or (
                not np.isnan(index_vs_overall) and index_vs_overall < 0.5
            ):
                rec = "REDUCE_INVESTMENT"
            elif not np.isnan(index_vs_overall) and index_vs_overall > 1.5:
                rec = "INCREASE_INVESTMENT"
            else:
                rec = "MAINTAIN"

            rows.append(
                {
                    "segment": segment_val,
                    "n_exposed": n_exp,
                    "n_holdout": n_hld,
                    "segment_lift": seg_lift,
                    "segment_lift_ci_lower": seg_stats["ci_lower"],
                    "segment_lift_ci_upper": seg_stats["ci_upper"],
                    "p_value": p_val,
                    "is_significant": sig,
                    "index_vs_overall": index_vs_overall,
                    "recommendation": rec,
                }
            )

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# __main__ demonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import textwrap

    rng = np.random.default_rng(42)

    # ── Generate synthetic panel data ─────────────────────────────────────
    n_units = 4_000
    holdout_frac = 0.20

    unit_ids = np.arange(n_units)
    variants = np.where(
        rng.random(n_units) < holdout_frac, "holdout", "exposed"
    )
    segments = rng.choice(["enterprise", "mid-market", "smb"], size=n_units, p=[0.2, 0.3, 0.5])
    industry = rng.choice(["tech", "finance", "healthcare", "retail"], size=n_units)

    # Baseline (pre) period
    baseline_rate_exposed = 0.08
    baseline_rate_holdout = 0.08
    pre_converted = np.where(
        variants == "exposed",
        rng.binomial(1, baseline_rate_exposed, n_units),
        rng.binomial(1, baseline_rate_holdout, n_units),
    )
    pre_revenue = pre_converted * rng.uniform(100, 500, n_units)

    # Test (post) period — campaign lifts exposed group by ~3 pp
    post_rate_exposed = 0.11
    post_rate_holdout = 0.082
    post_converted = np.where(
        variants == "exposed",
        rng.binomial(1, post_rate_exposed, n_units),
        rng.binomial(1, post_rate_holdout, n_units),
    )
    post_revenue = post_converted * rng.uniform(100, 500, n_units)

    # Tiny contamination: ~1% of holdout got ads
    holdout_idx = np.where(variants == "holdout")[0]
    contaminated_idx = rng.choice(holdout_idx, size=int(len(holdout_idx) * 0.01), replace=False)
    exposure_indicator = np.zeros(n_units, dtype=int)
    exposure_indicator[variants == "exposed"] = 1
    exposure_indicator[contaminated_idx] = 1

    # Build long (panel) and wide (post-only) DataFrames
    pre_df = pd.DataFrame(
        {
            "unit_id": unit_ids,
            "variant": variants,
            "period": "baseline",
            "converted": pre_converted,
            "revenue": pre_revenue,
            "segment": segments,
            "industry": industry,
            "exposure_indicator": exposure_indicator,
        }
    )
    post_df = pd.DataFrame(
        {
            "unit_id": unit_ids,
            "variant": variants,
            "period": "test",
            "converted": post_converted,
            "revenue": post_revenue,
            "segment": segments,
            "industry": industry,
            "exposure_indicator": exposure_indicator,
        }
    )
    panel_df = pd.concat([pre_df, post_df], ignore_index=True)
    post_only_df = post_df.copy()

    # ── Run analysis ──────────────────────────────────────────────────────
    analyzer = HoldoutAnalyzer()

    print("\n" + "=" * 72)
    print("  CAMPAIGN EXPERIMENTATION FRAMEWORK — Holdout Analysis Demo")
    print("=" * 72)

    # 1. Balance check
    print("\n[1] Pre-experiment balance check")
    balance = analyzer.check_balance(
        pre_df, "variant", covariates=["segment", "industry"]
    )
    for br in balance:
        status = "BALANCED" if br.is_balanced else "IMBALANCED !"
        print(f"    {br.variable:<12s}  p={br.p_value:.4f}  [{status}]")

    # 2. Parallel trends
    print("\n[2] Parallel trends check (pre-period only — trivially passes here)")
    pt = analyzer.check_parallel_trends(
        panel_df[panel_df["period"] == "baseline"].assign(
            _fake_time=rng.integers(0, 3, size=len(panel_df[panel_df["period"] == "baseline"]))
        ).rename(columns={"period": "wave"}),
        variant_col="variant",
        period_col="wave",
        metric_col="converted",
        pre_period_value="baseline",
    )
    print(f"    interaction p={pt['p_value']:.4f}  |  is_parallel={pt['is_parallel']}")

    # 3. Contamination check
    print("\n[3] Contamination check")
    cont = analyzer.check_contamination(post_only_df, "variant", "exposure_indicator")
    print(f"    {cont['message']}")

    # 4. Full analysis (DiD + business metrics)
    print("\n[4] Full incrementality analysis")
    CAMPAIGN_COST = 50_000.0
    result = analyzer.analyze(
        data=panel_df,
        variant_col="variant",
        metric_col="converted",
        experiment_id="holdout_q1_2026",
        period_col="period",
        campaign_cost=CAMPAIGN_COST,
        revenue_col="revenue",
        covariates=["segment", "industry"],
    )
    print(analyzer.format_holdout_report(result, balance_results=balance))

    # 5. Segment lift decomposition
    print("\n[5] Segment-level lift decomposition")
    seg_df = analyzer.compute_segment_lift(
        post_only_df, "variant", "converted", "segment",
        experiment_id="holdout_q1_2026",
    )
    print(seg_df.to_string(index=False))
    print()
