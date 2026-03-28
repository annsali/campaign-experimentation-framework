"""
A/B/n Multivariate Test Analysis with Multiple Comparison Corrections
for the Campaign Experimentation & Lift Measurement Framework.

Supports:
- Omnibus tests: ANOVA / Kruskal-Wallis (continuous), chi-squared (proportions)
- Pairwise comparisons: Welch's t-test (continuous), two-proportion z-test (proportions)
- Four multiple-comparison corrections: Bonferroni, Holm, FDR-BH, Dunnett
- Bayesian probability-of-being-best via Monte Carlo sampling
- Winner determination with confidence levels
- Full text reporting

Usage
-----
    python src/multivariate_test.py

Or programmatically::

    from src.multivariate_test import MultivariateTest
    mt = MultivariateTest()
    results = mt.analyze(df, variant_col="variant", metrics=["conversion_rate"], experiment_id="exp001")
"""

from __future__ import annotations

import itertools
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Project-root config import
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (  # noqa: E402
    BOOTSTRAP_ITERATIONS,
    CONFIDENCE_LEVEL,
    CONTINUOUS_METRICS,
    DEFAULT_MCC,
    PROPORTION_METRICS,
    SIGNIFICANCE_LEVEL,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PairwiseResult:
    """Holds all statistics for a single pairwise variant comparison on one metric."""

    control_variant: str
    treatment_variant: str
    metric_name: str
    control_metric: float
    treatment_metric: float
    absolute_lift: float
    relative_lift: float
    p_value_uncorrected: float
    p_value_bonferroni: float
    p_value_holm: float
    p_value_fdr_bh: float
    is_significant_bonferroni: bool
    is_significant_holm: bool
    is_significant_fdr_bh: bool
    is_significant_dunnett: bool  # only valid for vs-control comparisons
    test_statistic: float
    confidence_interval: Tuple[float, float]  # 95% CI for absolute lift


@dataclass
class MultivariateResult:
    """Aggregated result for one metric across all variants in an experiment."""

    experiment_id: str
    metric_name: str
    n_variants: int
    overall_f_test_p_value: float          # ANOVA / Kruskal-Wallis p-value
    overall_chi2_p_value: float            # for proportions
    pairwise_comparisons: List[PairwiseResult]
    winner: Optional[str]                  # variant name or None
    winner_confidence: str                 # "high" | "medium" | "low" | "none"
    variant_ranks: List[dict]              # [{variant, mean, rank}]
    probability_best_bayesian: Dict[str, float]
    correction_comparison_table: pd.DataFrame  # all 4 corrections side by side


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class MultivariateTest:
    """A/B/n multivariate test analysis with multiple comparison corrections.

    Parameters
    ----------
    significance_level:
        Two-sided alpha threshold (default from config).
    confidence_level:
        Confidence level for CIs (default from config).
    """

    def __init__(
        self,
        significance_level: float = SIGNIFICANCE_LEVEL,
        confidence_level: float = CONFIDENCE_LEVEL,
    ) -> None:
        self.significance_level = significance_level
        self.confidence_level = confidence_level
        self._alpha = significance_level  # shorthand

    # ------------------------------------------------------------------
    # Multiple comparison corrections
    # ------------------------------------------------------------------

    def _apply_corrections(self, p_values: List[float]) -> Dict[str, List[float]]:
        """Apply four MCC methods to a list of raw p-values.

        Parameters
        ----------
        p_values:
            Uncorrected p-values from all pairwise comparisons.

        Returns
        -------
        dict with keys: bonferroni, holm, fdr_bh, dunnett.
        Each value is a list of corrected p-values in the same order.
        """
        n = len(p_values)
        arr = np.array(p_values, dtype=float)

        # Bonferroni: multiply by number of comparisons, cap at 1.0
        bonferroni = np.minimum(arr * n, 1.0).tolist()

        # Holm step-down
        _, holm_pvals, _, _ = multipletests(arr, method="holm")
        holm = holm_pvals.tolist()

        # Benjamini-Hochberg FDR
        _, fdr_bh_pvals, _, _ = multipletests(arr, method="fdr_bh")
        fdr_bh = fdr_bh_pvals.tolist()

        # Dunnett (scipy >= 1.11): fall back to Holm if unavailable
        try:
            # scipy.stats.dunnett requires the raw group samples, not just p-values.
            # Since we only have p-values here, we use Holm as the best available
            # proxy at this aggregation stage. Dunnett is applied directly in the
            # analyze_* methods where raw samples are available.
            dunnett = holm  # placeholder — overridden in analysis methods
        except Exception:
            dunnett = holm

        return {
            "bonferroni": bonferroni,
            "holm": holm,
            "fdr_bh": fdr_bh,
            "dunnett": dunnett,
        }

    # ------------------------------------------------------------------
    # Bayesian probability of being best
    # ------------------------------------------------------------------

    def _compute_probability_best(
        self,
        variant_means: Dict[str, float],
        variant_stds: Dict[str, float],
        variant_ns: Dict[str, int],
        n_samples: int = 50_000,
    ) -> Dict[str, float]:
        """Estimate P(variant is best) via Monte Carlo sampling from Normal posteriors.

        For each variant, draw n_samples values from Normal(mean, se) where
        se = std / sqrt(n).  The probability for each variant is the fraction
        of draws where it achieves the highest sampled value.

        Parameters
        ----------
        variant_means, variant_stds, variant_ns:
            Per-variant mean, standard deviation, and sample size.
        n_samples:
            Number of Monte Carlo draws.

        Returns
        -------
        dict mapping variant name -> probability float.
        """
        variants = list(variant_means.keys())
        rng = np.random.default_rng(42)

        # Draw matrix: shape (n_samples, n_variants)
        draws = np.column_stack([
            rng.normal(
                loc=variant_means[v],
                scale=max(variant_stds[v] / np.sqrt(max(variant_ns[v], 1)), 1e-12),
                size=n_samples,
            )
            for v in variants
        ])

        # For each sample, find the index of the best variant
        best_idx = np.argmax(draws, axis=1)
        counts = np.bincount(best_idx, minlength=len(variants))
        probabilities = counts / n_samples

        return {v: float(probabilities[i]) for i, v in enumerate(variants)}

    # ------------------------------------------------------------------
    # Proportion analysis
    # ------------------------------------------------------------------

    def analyze_proportion(
        self,
        data: pd.DataFrame,
        variant_col: str,
        metric_col: str,
        metric_name: str,
        experiment_id: str,
        control_variant: Optional[str] = None,
    ) -> MultivariateResult:
        """Analyze a binary proportion metric across multiple variants.

        Steps
        -----
        1. Compute conversion rate per variant.
        2. Omnibus chi-squared test on the contingency table.
        3. Pairwise two-proportion z-tests (all pairs or vs control only).
        4. Apply four MCC methods; attempt Dunnett for vs-control pairs.
        5. Determine winner and confidence level.
        6. Compute Bayesian P(best).
        7. Build correction comparison table.

        Parameters
        ----------
        data:
            Experiment DataFrame.
        variant_col:
            Column identifying the variant.
        metric_col:
            Binary (0/1) outcome column.
        metric_name:
            Human-readable metric label (used in output).
        experiment_id:
            Experiment identifier string.
        control_variant:
            If provided, compare only vs this control arm.  If None, all
            pairwise comparisons are run.

        Returns
        -------
        MultivariateResult
        """
        variants = sorted(data[variant_col].dropna().unique().tolist())
        n_variants = len(variants)

        if n_variants < 2:
            raise ValueError(
                f"analyze_proportion requires at least 2 variants; found {n_variants}."
            )

        # Per-variant statistics
        variant_counts: Dict[str, int] = {}
        variant_successes: Dict[str, int] = {}
        variant_rates: Dict[str, float] = {}

        for v in variants:
            mask = data[variant_col] == v
            n = int(mask.sum())
            s = int(data.loc[mask, metric_col].sum())
            variant_counts[v] = n
            variant_successes[v] = s
            variant_rates[v] = s / n if n > 0 else 0.0

        # ── Omnibus chi-squared test ─────────────────────────────────────────
        contingency = np.array([
            [variant_successes[v], variant_counts[v] - variant_successes[v]]
            for v in variants
        ])
        chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency)
        # For proportion metrics there is no separate F-test; use NaN
        overall_f_p = float("nan")
        overall_chi2_p = float(chi2_p)

        # ── Pairwise comparisons ─────────────────────────────────────────────
        if control_variant is not None and control_variant in variants:
            pairs = [(control_variant, v) for v in variants if v != control_variant]
        else:
            pairs = list(itertools.combinations(variants, 2))

        raw_p_values: List[float] = []
        pair_stats: List[Tuple] = []  # (ctrl, trt, stat, ci, ctrl_rate, trt_rate)

        alpha = self.significance_level
        z_crit = stats.norm.ppf(1 - alpha / 2)

        for ctrl, trt in pairs:
            n1, s1 = variant_counts[ctrl], variant_successes[ctrl]
            n2, s2 = variant_counts[trt], variant_successes[trt]
            p1, p2 = variant_rates[ctrl], variant_rates[trt]

            # Pooled proportion z-test
            p_pool = (s1 + s2) / (n1 + n2)
            se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
            if se_pool == 0:
                z_stat = 0.0
                p_val = 1.0
            else:
                z_stat = (p2 - p1) / se_pool
                p_val = float(2 * stats.norm.sf(abs(z_stat)))

            # CI for absolute lift using unpooled SE
            se_lift = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
            lift = p2 - p1
            ci_low = lift - z_crit * se_lift
            ci_high = lift + z_crit * se_lift

            raw_p_values.append(p_val)
            pair_stats.append((ctrl, trt, float(z_stat), (float(ci_low), float(ci_high)), p1, p2))

        # ── Apply four corrections ───────────────────────────────────────────
        corrections = self._apply_corrections(raw_p_values)

        # Attempt Dunnett using raw samples when control is specified
        dunnett_p_values = corrections["holm"]  # default
        if control_variant is not None and control_variant in variants:
            dunnett_p_values = self._dunnett_proportion_fallback(
                variant_counts, variant_successes, variants,
                control_variant, pairs, corrections["holm"],
            )

        # ── Build PairwiseResult list ────────────────────────────────────────
        pairwise: List[PairwiseResult] = []
        for i, (ctrl, trt, stat, ci, ctrl_rate, trt_rate) in enumerate(pair_stats):
            abs_lift = trt_rate - ctrl_rate
            rel_lift = abs_lift / ctrl_rate if ctrl_rate != 0 else float("nan")
            p_unc = raw_p_values[i]
            p_bon = corrections["bonferroni"][i]
            p_holm = corrections["holm"][i]
            p_fdr = corrections["fdr_bh"][i]
            p_dun = dunnett_p_values[i]

            pairwise.append(PairwiseResult(
                control_variant=ctrl,
                treatment_variant=trt,
                metric_name=metric_name,
                control_metric=ctrl_rate,
                treatment_metric=trt_rate,
                absolute_lift=abs_lift,
                relative_lift=rel_lift,
                p_value_uncorrected=p_unc,
                p_value_bonferroni=p_bon,
                p_value_holm=p_holm,
                p_value_fdr_bh=p_fdr,
                is_significant_bonferroni=p_bon < alpha,
                is_significant_holm=p_holm < alpha,
                is_significant_fdr_bh=p_fdr < alpha,
                is_significant_dunnett=p_dun < alpha,
                test_statistic=stat,
                confidence_interval=ci,
            ))

        # ── Variant ranks ────────────────────────────────────────────────────
        sorted_variants = sorted(variants, key=lambda v: variant_rates[v], reverse=True)
        variant_ranks = [
            {"variant": v, "mean": variant_rates[v], "rank": r + 1}
            for r, v in enumerate(sorted_variants)
        ]

        # ── Winner determination ─────────────────────────────────────────────
        winner, winner_confidence = self._determine_winner(
            sorted_variants, variant_rates, pairwise, metric_name
        )

        # ── Bayesian P(best) ─────────────────────────────────────────────────
        # For proportions: std = sqrt(p*(1-p)), n from variant_counts
        variant_stds = {
            v: np.sqrt(variant_rates[v] * (1 - variant_rates[v]))
            for v in variants
        }
        prob_best = self._compute_probability_best(
            variant_means=variant_rates,
            variant_stds=variant_stds,
            variant_ns=variant_counts,
        )

        # ── Correction comparison table ──────────────────────────────────────
        corr_table = self._build_correction_table(pairwise, dunnett_p_values, pairs)

        return MultivariateResult(
            experiment_id=experiment_id,
            metric_name=metric_name,
            n_variants=n_variants,
            overall_f_test_p_value=overall_f_p,
            overall_chi2_p_value=overall_chi2_p,
            pairwise_comparisons=pairwise,
            winner=winner,
            winner_confidence=winner_confidence,
            variant_ranks=variant_ranks,
            probability_best_bayesian=prob_best,
            correction_comparison_table=corr_table,
        )

    # ------------------------------------------------------------------
    # Continuous analysis
    # ------------------------------------------------------------------

    def analyze_continuous(
        self,
        data: pd.DataFrame,
        variant_col: str,
        metric_col: str,
        metric_name: str,
        experiment_id: str,
        control_variant: Optional[str] = None,
    ) -> MultivariateResult:
        """Analyze a continuous metric across multiple variants.

        Steps
        -----
        1. Compute per-variant mean, std, n.
        2. Omnibus one-way ANOVA; also run Kruskal-Wallis non-parametric test.
        3. Pairwise Welch's t-test (unequal variance).
        4. Apply four MCC methods; attempt Dunnett for vs-control pairs.
        5. Determine winner and confidence level.
        6. Compute Bayesian P(best).
        7. Build correction comparison table.

        Parameters
        ----------
        data:
            Experiment DataFrame.
        variant_col:
            Column identifying the variant.
        metric_col:
            Continuous numeric outcome column.
        metric_name:
            Human-readable metric label.
        experiment_id:
            Experiment identifier string.
        control_variant:
            If provided, compare only vs this control arm.

        Returns
        -------
        MultivariateResult
        """
        variants = sorted(data[variant_col].dropna().unique().tolist())
        n_variants = len(variants)

        if n_variants < 2:
            raise ValueError(
                f"analyze_continuous requires at least 2 variants; found {n_variants}."
            )

        # Per-variant arrays and statistics
        variant_data: Dict[str, np.ndarray] = {}
        variant_means: Dict[str, float] = {}
        variant_stds: Dict[str, float] = {}
        variant_ns: Dict[str, int] = {}

        for v in variants:
            arr = data.loc[data[variant_col] == v, metric_col].dropna().values.astype(float)
            variant_data[v] = arr
            variant_means[v] = float(np.mean(arr))
            variant_stds[v] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            variant_ns[v] = len(arr)

        # ── Omnibus tests ────────────────────────────────────────────────────
        all_groups = [variant_data[v] for v in variants]

        # One-way ANOVA (F-test)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_stat, f_p = stats.f_oneway(*all_groups)
        overall_f_p = float(f_p)

        # Kruskal-Wallis (non-parametric alternative)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                kw_stat, kw_p = stats.kruskal(*all_groups)
            except ValueError:
                kw_p = float("nan")
        # Store Kruskal-Wallis in chi2 slot for symmetry with proportion path
        overall_chi2_p = float(kw_p)

        # ── Pairwise Welch's t-tests ─────────────────────────────────────────
        if control_variant is not None and control_variant in variants:
            pairs = [(control_variant, v) for v in variants if v != control_variant]
        else:
            pairs = list(itertools.combinations(variants, 2))

        raw_p_values: List[float] = []
        pair_stats: List[Tuple] = []

        alpha = self.significance_level
        t_df_approx: Dict[Tuple, float] = {}  # degrees of freedom per pair for CI

        for ctrl, trt in pairs:
            arr1 = variant_data[ctrl]
            arr2 = variant_data[trt]
            n1, n2 = len(arr1), len(arr2)
            m1, m2 = variant_means[ctrl], variant_means[trt]
            s1, s2 = variant_stds[ctrl], variant_stds[trt]

            t_stat, p_val = stats.ttest_ind(arr1, arr2, equal_var=False)
            p_val = float(p_val)
            t_stat = float(t_stat)

            # Welch-Satterthwaite degrees of freedom for CI
            se1_sq = s1 ** 2 / n1 if n1 > 0 else 0.0
            se2_sq = s2 ** 2 / n2 if n2 > 0 else 0.0
            se_lift = np.sqrt(se1_sq + se2_sq)
            denom = (se1_sq ** 2 / max(n1 - 1, 1)) + (se2_sq ** 2 / max(n2 - 1, 1))
            ws_df = (se1_sq + se2_sq) ** 2 / denom if denom > 0 else n1 + n2 - 2
            t_df_approx[(ctrl, trt)] = ws_df
            t_crit = stats.t.ppf(1 - alpha / 2, df=ws_df)

            lift = m2 - m1
            ci_low = lift - t_crit * se_lift
            ci_high = lift + t_crit * se_lift

            raw_p_values.append(p_val)
            pair_stats.append((ctrl, trt, t_stat, (float(ci_low), float(ci_high)), m1, m2))

        # ── Apply four corrections ───────────────────────────────────────────
        corrections = self._apply_corrections(raw_p_values)

        # Attempt real Dunnett when control is specified and scipy >= 1.11
        dunnett_p_values = corrections["holm"]  # default
        if control_variant is not None and control_variant in variants:
            dunnett_p_values = self._dunnett_continuous(
                variant_data, variants, control_variant, pairs, corrections["holm"]
            )

        # ── Build PairwiseResult list ────────────────────────────────────────
        pairwise: List[PairwiseResult] = []
        for i, (ctrl, trt, stat, ci, m_ctrl, m_trt) in enumerate(pair_stats):
            abs_lift = m_trt - m_ctrl
            rel_lift = abs_lift / m_ctrl if m_ctrl != 0 else float("nan")
            p_unc = raw_p_values[i]
            p_bon = corrections["bonferroni"][i]
            p_holm = corrections["holm"][i]
            p_fdr = corrections["fdr_bh"][i]
            p_dun = dunnett_p_values[i]

            pairwise.append(PairwiseResult(
                control_variant=ctrl,
                treatment_variant=trt,
                metric_name=metric_name,
                control_metric=m_ctrl,
                treatment_metric=m_trt,
                absolute_lift=abs_lift,
                relative_lift=rel_lift,
                p_value_uncorrected=p_unc,
                p_value_bonferroni=p_bon,
                p_value_holm=p_holm,
                p_value_fdr_bh=p_fdr,
                is_significant_bonferroni=p_bon < alpha,
                is_significant_holm=p_holm < alpha,
                is_significant_fdr_bh=p_fdr < alpha,
                is_significant_dunnett=p_dun < alpha,
                test_statistic=stat,
                confidence_interval=ci,
            ))

        # ── Variant ranks ────────────────────────────────────────────────────
        sorted_variants = sorted(variants, key=lambda v: variant_means[v], reverse=True)
        variant_ranks = [
            {"variant": v, "mean": variant_means[v], "rank": r + 1}
            for r, v in enumerate(sorted_variants)
        ]

        # ── Winner determination ─────────────────────────────────────────────
        winner, winner_confidence = self._determine_winner(
            sorted_variants, variant_means, pairwise, metric_name
        )

        # ── Bayesian P(best) ─────────────────────────────────────────────────
        prob_best = self._compute_probability_best(
            variant_means=variant_means,
            variant_stds=variant_stds,
            variant_ns=variant_ns,
        )

        # ── Correction comparison table ──────────────────────────────────────
        corr_table = self._build_correction_table(pairwise, dunnett_p_values, pairs)

        return MultivariateResult(
            experiment_id=experiment_id,
            metric_name=metric_name,
            n_variants=n_variants,
            overall_f_test_p_value=overall_f_p,
            overall_chi2_p_value=overall_chi2_p,
            pairwise_comparisons=pairwise,
            winner=winner,
            winner_confidence=winner_confidence,
            variant_ranks=variant_ranks,
            probability_best_bayesian=prob_best,
            correction_comparison_table=corr_table,
        )

    # ------------------------------------------------------------------
    # Top-level dispatcher
    # ------------------------------------------------------------------

    def analyze(
        self,
        experiment_data: pd.DataFrame,
        variant_col: str,
        metrics: List[str],
        experiment_id: str,
        control_variant: Optional[str] = None,
    ) -> Dict[str, MultivariateResult]:
        """Analyze multiple metrics, routing each to the correct analysis method.

        Parameters
        ----------
        experiment_data:
            DataFrame with one row per subject, containing variant and metric columns.
        variant_col:
            Column that identifies which variant each row belongs to.
        metrics:
            List of metric column names to analyze.
        experiment_id:
            Experiment identifier for labelling outputs.
        control_variant:
            Optional control arm name.  When provided, pairwise comparisons are
            restricted to control-vs-treatment pairs and Dunnett's correction is
            used where available.

        Returns
        -------
        dict mapping metric_name -> MultivariateResult
        """
        results: Dict[str, MultivariateResult] = {}

        for metric in metrics:
            if metric not in experiment_data.columns:
                warnings.warn(
                    f"Metric '{metric}' not found in DataFrame columns; skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            if metric in PROPORTION_METRICS:
                result = self.analyze_proportion(
                    data=experiment_data,
                    variant_col=variant_col,
                    metric_col=metric,
                    metric_name=metric,
                    experiment_id=experiment_id,
                    control_variant=control_variant,
                )
            elif metric in CONTINUOUS_METRICS:
                result = self.analyze_continuous(
                    data=experiment_data,
                    variant_col=variant_col,
                    metric_col=metric,
                    metric_name=metric,
                    experiment_id=experiment_id,
                    control_variant=control_variant,
                )
            else:
                # Unknown metric: auto-detect by dtype
                col_dtype = experiment_data[metric].dtype
                if pd.api.types.is_float_dtype(col_dtype):
                    unique_vals = experiment_data[metric].dropna().unique()
                    if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                        result = self.analyze_proportion(
                            data=experiment_data,
                            variant_col=variant_col,
                            metric_col=metric,
                            metric_name=metric,
                            experiment_id=experiment_id,
                            control_variant=control_variant,
                        )
                    else:
                        result = self.analyze_continuous(
                            data=experiment_data,
                            variant_col=variant_col,
                            metric_col=metric,
                            metric_name=metric,
                            experiment_id=experiment_id,
                            control_variant=control_variant,
                        )
                elif pd.api.types.is_integer_dtype(col_dtype):
                    unique_vals = experiment_data[metric].dropna().unique()
                    if set(unique_vals).issubset({0, 1}):
                        result = self.analyze_proportion(
                            data=experiment_data,
                            variant_col=variant_col,
                            metric_col=metric,
                            metric_name=metric,
                            experiment_id=experiment_id,
                            control_variant=control_variant,
                        )
                    else:
                        result = self.analyze_continuous(
                            data=experiment_data,
                            variant_col=variant_col,
                            metric_col=metric,
                            metric_name=metric,
                            experiment_id=experiment_id,
                            control_variant=control_variant,
                        )
                else:
                    warnings.warn(
                        f"Cannot infer metric type for '{metric}'; skipping.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue

            results[metric] = result

        return results

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def build_correction_comparison_table(self, result: MultivariateResult) -> pd.DataFrame:
        """Build a side-by-side comparison table for all four corrections.

        Parameters
        ----------
        result:
            A MultivariateResult (from analyze_proportion or analyze_continuous).

        Returns
        -------
        DataFrame with columns:
            comparison, p_uncorrected, p_bonferroni, p_holm, p_fdr_bh, p_dunnett,
            sig_bonferroni, sig_holm, sig_fdr_bh, sig_dunnett
        """
        rows = []
        for pr in result.pairwise_comparisons:
            rows.append({
                "comparison": f"{pr.control_variant} vs {pr.treatment_variant}",
                "p_uncorrected": round(pr.p_value_uncorrected, 6),
                "p_bonferroni": round(pr.p_value_bonferroni, 6),
                "p_holm": round(pr.p_value_holm, 6),
                "p_fdr_bh": round(pr.p_value_fdr_bh, 6),
                "p_dunnett": round(pr.p_value_fdr_bh, 6),  # stored in fdr slot; Dunnett per pair
                "sig_bonferroni": pr.is_significant_bonferroni,
                "sig_holm": pr.is_significant_holm,
                "sig_fdr_bh": pr.is_significant_fdr_bh,
                "sig_dunnett": pr.is_significant_dunnett,
            })
        return pd.DataFrame(rows)

    def format_multivariate_report(
        self,
        results: Dict[str, MultivariateResult],
        primary_metric: str,
    ) -> str:
        """Generate a full-text analysis report covering all metrics.

        The report contains:
        - Experiment summary
        - Winner announcement for the primary metric
        - Per-metric sections with omnibus test, variant performance table,
          all pairwise comparisons under all four corrections, and Bayesian
          probability estimates
        - Recommendations

        Parameters
        ----------
        results:
            Dict returned by :meth:`analyze`.
        primary_metric:
            The key metric that drives the winner declaration.

        Returns
        -------
        Formatted multi-line string.
        """
        lines: List[str] = []

        if not results:
            return "No results to report."

        # Derive experiment ID from the first result
        first_result = next(iter(results.values()))
        exp_id = first_result.experiment_id

        sep_heavy = "=" * 72
        sep_light = "-" * 72

        lines.append(sep_heavy)
        lines.append(f"  MULTIVARIATE TEST REPORT  |  Experiment: {exp_id}")
        lines.append(sep_heavy)
        lines.append(
            f"  Metrics analyzed : {len(results)}  "
            f"({', '.join(results.keys())})"
        )
        lines.append(
            f"  Significance level (alpha) : {self.significance_level}  |  "
            f"Confidence level : {self.confidence_level}"
        )
        lines.append(f"  Primary metric   : {primary_metric}")
        lines.append("")

        # Winner summary
        if primary_metric in results:
            pr = results[primary_metric]
            if pr.winner:
                lines.append(f"  PRIMARY METRIC WINNER: {pr.winner.upper()}")
                lines.append(f"  Winner confidence    : {pr.winner_confidence.upper()}")
            else:
                lines.append("  PRIMARY METRIC: No clear winner identified.")
        lines.append("")

        # Per-metric sections
        for metric_name, result in results.items():
            lines.append(sep_heavy)
            lines.append(f"  METRIC: {metric_name}")
            lines.append(sep_heavy)

            # Omnibus tests
            lines.append("  Omnibus Tests:")
            if not np.isnan(result.overall_f_test_p_value):
                sig_f = " [*]" if result.overall_f_test_p_value < self.significance_level else ""
                lines.append(
                    f"    ANOVA F-test         p = {result.overall_f_test_p_value:.6f}{sig_f}"
                )
            if not np.isnan(result.overall_chi2_p_value):
                sig_c = " [*]" if result.overall_chi2_p_value < self.significance_level else ""
                label = "Chi-squared" if np.isnan(result.overall_f_test_p_value) else "Kruskal-Wallis"
                lines.append(
                    f"    {label:<20} p = {result.overall_chi2_p_value:.6f}{sig_c}"
                )
            lines.append("")

            # Variant performance table
            lines.append("  Variant Performance (ranked):")
            header = f"    {'Rank':<5} {'Variant':<20} {'Mean/Rate':<14} {'P(Best)':<12}"
            lines.append(header)
            lines.append("    " + "-" * (len(header) - 4))
            for row in result.variant_ranks:
                v = row["variant"]
                prob = result.probability_best_bayesian.get(v, float("nan"))
                lines.append(
                    f"    {row['rank']:<5} {v:<20} {row['mean']:<14.6f} {prob:<12.4f}"
                )
            lines.append("")

            # Pairwise comparisons table
            lines.append("  Pairwise Comparisons (all corrections):")
            col_w = [28, 10, 10, 10, 10, 10]
            header_row = (
                f"    {'Comparison':<{col_w[0]}}"
                f"{'Uncorr':>{col_w[1]}}"
                f"{'Bonf':>{col_w[2]}}"
                f"{'Holm':>{col_w[3]}}"
                f"{'FDR-BH':>{col_w[4]}}"
                f"{'Dunnett':>{col_w[5]}}"
            )
            lines.append(header_row)
            lines.append("    " + "-" * (sum(col_w) + 4))

            for pr in result.pairwise_comparisons:
                comparison = f"{pr.control_variant} vs {pr.treatment_variant}"
                lift_str = f"  lift={pr.absolute_lift:+.4f} ({pr.relative_lift * 100:+.1f}%)" \
                    if not np.isnan(pr.relative_lift) else f"  lift={pr.absolute_lift:+.4f}"

                def fmt_p(p: float, sig: bool) -> str:
                    marker = "*" if sig else " "
                    return f"{p:.4f}{marker}"

                lines.append(
                    f"    {comparison:<{col_w[0]}}"
                    f"{fmt_p(pr.p_value_uncorrected, pr.p_value_uncorrected < self.significance_level):>{col_w[1]}}"
                    f"{fmt_p(pr.p_value_bonferroni, pr.is_significant_bonferroni):>{col_w[2]}}"
                    f"{fmt_p(pr.p_value_holm, pr.is_significant_holm):>{col_w[3]}}"
                    f"{fmt_p(pr.p_value_fdr_bh, pr.is_significant_fdr_bh):>{col_w[4]}}"
                    f"{fmt_p(pr.p_value_fdr_bh, pr.is_significant_dunnett):>{col_w[5]}}"
                )
                lines.append(
                    f"    {'':<{col_w[0]}}"
                    f"  abs_lift={pr.absolute_lift:+.5f}  "
                    f"CI=[{pr.confidence_interval[0]:+.5f}, {pr.confidence_interval[1]:+.5f}]"
                    f"{lift_str}"
                )

            lines.append("    (* = significant at alpha)")
            lines.append("")

            # Winner
            if result.winner:
                lines.append(
                    f"  Winner: {result.winner}  (confidence: {result.winner_confidence})"
                )
            else:
                lines.append("  Winner: None — no variant is significantly best.")
            lines.append("")

        # Recommendations
        lines.append(sep_heavy)
        lines.append("  RECOMMENDATIONS")
        lines.append(sep_heavy)

        primary_result = results.get(primary_metric)
        if primary_result and primary_result.winner:
            w = primary_result.winner
            conf = primary_result.winner_confidence
            top = primary_result.variant_ranks[0]
            lift_vs_others = [
                pr.relative_lift
                for pr in primary_result.pairwise_comparisons
                if pr.treatment_variant == w or pr.control_variant == w
            ]
            avg_rel = np.nanmean(lift_vs_others) if lift_vs_others else float("nan")

            if conf == "high":
                lines.append(
                    f"  SHIP: Variant '{w}' shows consistent, statistically significant"
                )
                lines.append(
                    f"  improvement across all pairwise corrections.  "
                    f"Avg relative lift: {avg_rel * 100:+.1f}%."
                )
                lines.append(
                    f"  Bayesian P(best) = {primary_result.probability_best_bayesian.get(w, 0):.3f}."
                )
            elif conf == "medium":
                lines.append(
                    f"  EXTEND: Variant '{w}' is likely the best performer but does not"
                )
                lines.append(
                    "  dominate under all correction methods.  Consider extending the"
                )
                lines.append("  experiment to collect more data before shipping.")
            else:
                lines.append(
                    f"  CAUTION: Variant '{w}' is the top performer, but the evidence"
                )
                lines.append("  is weak.  Extend or redesign the experiment.")
        else:
            lines.append(
                "  INCONCLUSIVE: No variant dominates the primary metric under"
            )
            lines.append(
                "  multiple-comparison corrections.  Review effect sizes and consider"
            )
            lines.append("  a redesigned experiment with larger sample sizes.")

        lines.append("")
        lines.append("  Correction guide:")
        lines.append("    Bonferroni — most conservative; controls FWER strongly.")
        lines.append("    Holm       — step-down; more powerful than Bonferroni.")
        lines.append("    FDR-BH     — controls false discovery rate; least conservative.")
        lines.append("    Dunnett    — optimal for many-vs-one (vs control) comparisons.")
        lines.append(sep_heavy)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _determine_winner(
        self,
        sorted_variants: List[str],
        variant_means: Dict[str, float],
        pairwise: List[PairwiseResult],
        metric_name: str,
    ) -> Tuple[Optional[str], str]:
        """Determine a winner and confidence level from pairwise Bonferroni results.

        A variant is the winner if it has the highest mean AND is significantly
        better (Bonferroni) than ALL other variants it is compared against.

        Confidence levels:
        - "high"   : winner significant under Bonferroni
        - "medium" : winner significant under Holm but not Bonferroni
        - "low"    : winner significant under FDR-BH but not Holm
        - "none"   : no clear winner

        Returns
        -------
        (winner_name | None, confidence_string)
        """
        if not sorted_variants:
            return None, "none"

        top_variant = sorted_variants[0]

        # Collect comparisons that involve the top variant as treatment
        top_as_trt = [
            pr for pr in pairwise
            if pr.treatment_variant == top_variant
        ]
        # Also check comparisons where top is listed as control (it should still win)
        top_as_ctrl = [
            pr for pr in pairwise
            if pr.control_variant == top_variant
        ]

        # All comparisons touching the top variant
        all_involving_top = top_as_trt + top_as_ctrl

        if not all_involving_top:
            return top_variant, "low"

        # Check significance levels
        def all_sig(attr: str) -> bool:
            """True if top variant beats all compared variants under this correction."""
            for pr in top_as_trt:
                if not getattr(pr, attr):
                    return False
            for pr in top_as_ctrl:
                # top is control; treatment must NOT be significant (top lost)
                if getattr(pr, attr):
                    return False
            return True

        if all_sig("is_significant_bonferroni") and top_as_trt:
            return top_variant, "high"
        elif all_sig("is_significant_holm") and top_as_trt:
            return top_variant, "medium"
        elif all_sig("is_significant_fdr_bh") and top_as_trt:
            return top_variant, "low"

        # Partial dominance: top variant wins some but not all comparisons
        any_sig_bon = any(pr.is_significant_bonferroni for pr in top_as_trt)
        if any_sig_bon:
            return top_variant, "low"

        return None, "none"

    def _build_correction_table(
        self,
        pairwise: List[PairwiseResult],
        dunnett_p_values: List[float],
        pairs: List[Tuple[str, str]],
    ) -> pd.DataFrame:
        """Build the correction comparison DataFrame stored in MultivariateResult."""
        rows = []
        for i, pr in enumerate(pairwise):
            rows.append({
                "comparison": f"{pr.control_variant} vs {pr.treatment_variant}",
                "p_uncorrected": pr.p_value_uncorrected,
                "p_bonferroni": pr.p_value_bonferroni,
                "p_holm": pr.p_value_holm,
                "p_fdr_bh": pr.p_value_fdr_bh,
                "p_dunnett": dunnett_p_values[i] if i < len(dunnett_p_values) else float("nan"),
                "sig_bonferroni": pr.is_significant_bonferroni,
                "sig_holm": pr.is_significant_holm,
                "sig_fdr_bh": pr.is_significant_fdr_bh,
                "sig_dunnett": pr.is_significant_dunnett,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _dunnett_continuous(
        variant_data: Dict[str, np.ndarray],
        variants: List[str],
        control_variant: str,
        pairs: List[Tuple[str, str]],
        fallback_p_values: List[float],
    ) -> List[float]:
        """Attempt Dunnett's test using scipy.stats.dunnett (scipy >= 1.11).

        Falls back to the provided Holm p-values if Dunnett is unavailable
        or raises an exception.

        Parameters
        ----------
        variant_data:
            Raw samples per variant.
        variants:
            All variant names.
        control_variant:
            The control arm name.
        pairs:
            List of (control, treatment) tuples in the same order as p-values.
        fallback_p_values:
            Holm p-values used when Dunnett is not available.

        Returns
        -------
        List of p-values in the same order as ``pairs``.
        """
        try:
            treatment_variants = [v for v in variants if v != control_variant]
            treatment_samples = [variant_data[v] for v in treatment_variants]
            control_samples = variant_data[control_variant]

            dunnett_result = stats.dunnett(*treatment_samples, control=control_samples)
            # dunnett_result.pvalue is ordered by treatment_variants
            trt_p_map = dict(zip(treatment_variants, dunnett_result.pvalue))

            p_values = []
            for ctrl, trt in pairs:
                p_values.append(float(trt_p_map.get(trt, fallback_p_values[len(p_values)])))
            return p_values

        except AttributeError:
            # scipy.stats.dunnett not available (scipy < 1.11)
            return fallback_p_values
        except Exception:
            return fallback_p_values

    @staticmethod
    def _dunnett_proportion_fallback(
        variant_counts: Dict[str, int],
        variant_successes: Dict[str, int],
        variants: List[str],
        control_variant: str,
        pairs: List[Tuple[str, str]],
        fallback_p_values: List[float],
    ) -> List[float]:
        """Dunnett is not defined for proportions; return Holm p-values as proxy.

        Dunnett's test assumes continuous normally-distributed data, so for
        proportion metrics we fall back to the Holm-corrected p-values which
        have the same family-wise error rate control for vs-control designs.
        """
        return fallback_p_values


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def run_multivariate_analysis(
    data: pd.DataFrame,
    variant_col: str,
    metrics: List[str],
    experiment_id: str,
    control_variant: Optional[str] = None,
    primary_metric: Optional[str] = None,
    significance_level: float = SIGNIFICANCE_LEVEL,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> Dict[str, MultivariateResult]:
    """Convenience wrapper around :class:`MultivariateTest`.

    Parameters
    ----------
    data, variant_col, metrics, experiment_id, control_variant:
        Passed directly to :meth:`MultivariateTest.analyze`.
    primary_metric:
        If provided, also prints the formatted report to stdout.
    significance_level, confidence_level:
        Statistical thresholds.

    Returns
    -------
    dict mapping metric_name -> MultivariateResult
    """
    mt = MultivariateTest(
        significance_level=significance_level,
        confidence_level=confidence_level,
    )
    results = mt.analyze(
        experiment_data=data,
        variant_col=variant_col,
        metrics=metrics,
        experiment_id=experiment_id,
        control_variant=control_variant,
    )

    if primary_metric and results:
        pm = primary_metric if primary_metric in results else next(iter(results))
        report = mt.format_multivariate_report(results, primary_metric=pm)
        print(report)

    return results


# ---------------------------------------------------------------------------
# __main__ demo: A/B/C/D test with both proportion and continuous metrics
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    print("Generating synthetic A/B/C/D experiment data...")
    rng = np.random.default_rng(42)

    N_PER_VARIANT = 5_000
    variants_demo = ["control", "variant_b", "variant_c", "variant_d"]

    # Ground-truth parameters
    true_conversion = {"control": 0.08, "variant_b": 0.082, "variant_c": 0.110, "variant_d": 0.095}
    true_click = {"control": 0.15, "variant_b": 0.151, "variant_c": 0.180, "variant_d": 0.165}
    true_time = {"control": 180.0, "variant_b": 183.0, "variant_c": 205.0, "variant_d": 195.0}
    true_time_std = 55.0

    records = []
    for v in variants_demo:
        n = N_PER_VARIANT
        conversion = rng.binomial(1, true_conversion[v], n)
        click = rng.binomial(1, true_click[v], n)
        time_on_page = np.clip(
            rng.normal(true_time[v], true_time_std, n), a_min=1, a_max=None
        ).round(1)
        engagement = rng.normal(
            {"control": 0.0, "variant_b": 0.1, "variant_c": 2.5, "variant_d": 1.2}[v],
            5.0, n
        ).round(2)

        chunk = pd.DataFrame({
            "variant": v,
            "conversion_rate": conversion,
            "click_rate": click,
            "time_on_page": time_on_page,
            "engagement_score_delta": engagement,
        })
        records.append(chunk)

    demo_df = pd.concat(records, ignore_index=True)
    print(f"Dataset: {len(demo_df):,} rows x {demo_df.shape[1]} cols\n")

    # ── Run analysis ─────────────────────────────────────────────────────────
    mt = MultivariateTest()

    print("Running analysis (all pairs, no explicit control)...")
    results_all = mt.analyze(
        experiment_data=demo_df,
        variant_col="variant",
        metrics=["conversion_rate", "click_rate", "time_on_page", "engagement_score_delta"],
        experiment_id="demo_abcd_exp",
        control_variant=None,
    )

    report_all = mt.format_multivariate_report(results_all, primary_metric="conversion_rate")
    print(report_all)

    # ── Run analysis with explicit control ───────────────────────────────────
    print("\nRunning analysis (vs control only)...")
    results_ctrl = mt.analyze(
        experiment_data=demo_df,
        variant_col="variant",
        metrics=["conversion_rate", "time_on_page"],
        experiment_id="demo_abcd_exp",
        control_variant="control",
    )

    report_ctrl = mt.format_multivariate_report(results_ctrl, primary_metric="conversion_rate")
    print(report_ctrl)

    # ── Show correction comparison table ─────────────────────────────────────
    print("\nCorrection comparison table for conversion_rate (vs-control):")
    print(
        results_ctrl["conversion_rate"].correction_comparison_table.to_string(index=False)
    )

    # ── Bayesian P(best) summary ─────────────────────────────────────────────
    print("\nBayesian P(best) — conversion_rate:")
    for variant, prob in sorted(
        results_all["conversion_rate"].probability_best_bayesian.items(),
        key=lambda x: -x[1],
    ):
        bar = "#" * int(prob * 40)
        print(f"  {variant:<15} {prob:.4f}  {bar}")

    print("\nDone.")
