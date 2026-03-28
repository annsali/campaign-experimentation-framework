"""
Segment Optimization Module
============================
Heterogeneous Treatment Effect (HTE) analysis, segment ranking, budget
allocation, and follow-up experiment recommendations for campaign
experimentation.

Analyzes sub-group performance to identify where a campaign over- or
under-performs relative to the overall lift, and translates those findings
into actionable budget and targeting recommendations.
"""

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import itertools

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SIGNIFICANCE_LEVEL, MIN_SEGMENT_SIZE, SEGMENT_ALPHA


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class SegmentResult:
    """
    Statistical and strategic summary for a single segment × variant slice.
    """
    segment_name: str
    segment_value: str
    n_exposed: int
    n_holdout: int                      # control observations in segment
    segment_lift: float                 # absolute lift within segment
    segment_lift_ci: tuple              # (lower, upper) 95% CI on lift
    p_value: float
    is_significant: bool
    index_vs_overall: float             # segment_lift / overall_lift (1.0 = average)
    recommendation: str                 # see _get_recommendation


# ── Main class ────────────────────────────────────────────────────────────────

class SegmentOptimizer:
    """
    Heterogeneous Treatment Effect (HTE) analysis and segment budget optimizer.

    Parameters
    ----------
    significance_level : float
        Family-wise alpha (before per-segment Bonferroni correction).
    min_segment_size : int
        Minimum observations per arm within a segment to run analysis.
    """

    def __init__(
        self,
        significance_level: float = SIGNIFICANCE_LEVEL,
        min_segment_size: int = MIN_SEGMENT_SIZE,
    ):
        self.significance_level = significance_level
        self.min_segment_size = min_segment_size

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_proportion_metric(self, series: pd.Series) -> bool:
        """True if all values are 0 or 1 (binary metric)."""
        unique_vals = series.dropna().unique()
        return set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False})

    def _compute_lift_and_stats(
        self,
        control_vals: pd.Series,
        treatment_vals: pd.Series,
        alpha: float,
    ) -> tuple:
        """
        Compute lift, 95% CI, and p-value for control vs treatment.

        Returns (lift, ci_lower, ci_upper, p_value).
        """
        n_c = len(control_vals)
        n_t = len(treatment_vals)

        if n_c < 2 or n_t < 2:
            return (0.0, -np.inf, np.inf, 1.0)

        is_prop = self._is_proportion_metric(
            pd.concat([control_vals, treatment_vals], ignore_index=True)
        )

        if is_prop:
            p_c = control_vals.mean()
            p_t = treatment_vals.mean()
            lift = p_t - p_c

            p_pool = (control_vals.sum() + treatment_vals.sum()) / (n_c + n_t)
            se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))

            if se_pool == 0:
                return (lift, lift, lift, 1.0)

            z = lift / se_pool
            p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

            # CI using unpooled SE
            se_unpooled = np.sqrt(
                p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t
            )
            z_crit = stats.norm.ppf(1 - alpha / 2)
            ci_lower = lift - z_crit * se_unpooled
            ci_upper = lift + z_crit * se_unpooled
        else:
            mean_c = control_vals.mean()
            mean_t = treatment_vals.mean()
            lift = mean_t - mean_c

            t_stat, p_value = stats.ttest_ind(
                treatment_vals.values,
                control_vals.values,
                equal_var=False,
            )

            se = np.sqrt(
                control_vals.var(ddof=1) / n_c
                + treatment_vals.var(ddof=1) / n_t
            )
            df = n_c + n_t - 2
            t_crit = stats.t.ppf(1 - alpha / 2, df=df)
            ci_lower = lift - t_crit * se
            ci_upper = lift + t_crit * se

        return (
            round(lift, 6),
            round(ci_lower, 6),
            round(ci_upper, 6),
            round(float(p_value), 6),
        )

    def _get_recommendation(
        self,
        segment_lift: float,
        overall_lift: float,
        p_value: float,
        alpha: float,
        n: int,
    ) -> str:
        """
        Map segment statistics to an investment recommendation.

        Rules (evaluated in priority order):
        1. INSUFFICIENT_DATA  — n < min_segment_size
        2. EXCLUDE            — significant negative lift
        3. INCREASE_INVESTMENT — index > 1.5 and significant
        4. MAINTAIN           — 0.75 <= index <= 1.5 and significant
        5. REDUCE_INVESTMENT  — (index < 0.75 and significant) OR
                                 (not significant and index < 0.5)
        6. MONITOR            — positive but not yet significant

        Parameters
        ----------
        segment_lift : float
        overall_lift : float
        p_value : float
        alpha : float
            Bonferroni-corrected significance threshold.
        n : int
            Minimum arm size within segment.

        Returns
        -------
        str
        """
        if n < self.min_segment_size:
            return "INSUFFICIENT_DATA"

        is_sig = p_value < alpha
        index = (segment_lift / overall_lift) if overall_lift != 0 else 0.0

        if segment_lift < 0 and is_sig:
            return "EXCLUDE"
        if index > 1.5 and is_sig:
            return "INCREASE_INVESTMENT"
        if 0.75 <= index <= 1.5 and is_sig:
            return "MAINTAIN"
        if (index < 0.75 and is_sig) or (not is_sig and index < 0.5):
            return "REDUCE_INVESTMENT"
        return "MONITOR"

    # ── HTE analysis ──────────────────────────────────────────────────────────

    def analyze_hte(
        self,
        data: pd.DataFrame,
        variant_col: str,
        metric_col: str,
        segment_cols: list,
        control_variant: str,
        treatment_variant: str,
        experiment_id: str,
    ) -> dict:
        """
        Perform HTE analysis across all specified segmentation dimensions.

        For each segment column and each unique value within it:
        - Filters data to that segment slice.
        - Skips slices with fewer than MIN_SEGMENT_SIZE observations per arm.
        - Computes lift, CI, p-value, and investment recommendation.
        - Runs an interaction test (OLS: metric ~ treatment + segment +
          treatment*segment) to detect significant treatment heterogeneity.

        Parameters
        ----------
        data : pd.DataFrame
            Full experiment dataset containing variant, metric, and segment cols.
        variant_col : str
            Column identifying control vs treatment (e.g. 'variant').
        metric_col : str
            Primary outcome column.
        segment_cols : list[str]
            Columns by which to segment (e.g. ['industry', 'region']).
        control_variant : str
            Value in variant_col that identifies the control arm.
        treatment_variant : str
            Value in variant_col that identifies the treatment arm.
        experiment_id : str
            Used for labeling in output.

        Returns
        -------
        dict[str, list[SegmentResult]]
            Keys are segment column names; values are lists of SegmentResult.
        """
        # Filter to only control and treatment rows
        subset = data[data[variant_col].isin([control_variant, treatment_variant])].copy()
        subset["_is_treatment"] = (subset[variant_col] == treatment_variant).astype(int)

        control_all = subset.loc[subset["_is_treatment"] == 0, metric_col]
        treatment_all = subset.loc[subset["_is_treatment"] == 1, metric_col]

        # Overall lift
        overall_lift, *_ = self._compute_lift_and_stats(
            control_all, treatment_all, alpha=self.significance_level
        )

        output: dict = {}

        for seg_col in segment_cols:
            if seg_col not in data.columns:
                warnings.warn(f"Segment column '{seg_col}' not found in data. Skipping.")
                continue

            unique_vals = subset[seg_col].dropna().unique()
            n_segments = len(unique_vals)

            # Bonferroni-corrected alpha for this dimension
            bonferroni_alpha = min(1.0, SEGMENT_ALPHA / max(n_segments, 1))

            seg_results: list[SegmentResult] = []

            for val in unique_vals:
                seg_mask = subset[seg_col] == val
                seg_data = subset[seg_mask]

                c_vals = seg_data.loc[seg_data["_is_treatment"] == 0, metric_col].dropna()
                t_vals = seg_data.loc[seg_data["_is_treatment"] == 1, metric_col].dropna()

                n_c = len(c_vals)
                n_t = len(t_vals)
                min_n = min(n_c, n_t)

                if min_n < self.min_segment_size:
                    seg_results.append(
                        SegmentResult(
                            segment_name=seg_col,
                            segment_value=str(val),
                            n_exposed=n_t,
                            n_holdout=n_c,
                            segment_lift=0.0,
                            segment_lift_ci=(0.0, 0.0),
                            p_value=1.0,
                            is_significant=False,
                            index_vs_overall=0.0,
                            recommendation="INSUFFICIENT_DATA",
                        )
                    )
                    continue

                lift, ci_lo, ci_hi, p_val = self._compute_lift_and_stats(
                    c_vals, t_vals, alpha=bonferroni_alpha
                )
                is_sig = p_val < bonferroni_alpha
                index = (lift / overall_lift) if overall_lift != 0 else 0.0

                recommendation = self._get_recommendation(
                    segment_lift=lift,
                    overall_lift=overall_lift,
                    p_value=p_val,
                    alpha=bonferroni_alpha,
                    n=min_n,
                )

                seg_results.append(
                    SegmentResult(
                        segment_name=seg_col,
                        segment_value=str(val),
                        n_exposed=n_t,
                        n_holdout=n_c,
                        segment_lift=lift,
                        segment_lift_ci=(ci_lo, ci_hi),
                        p_value=p_val,
                        is_significant=is_sig,
                        index_vs_overall=round(index, 4),
                        recommendation=recommendation,
                    )
                )

            # ── Interaction test ───────────────────────────────────────────────
            interaction_p = None
            try:
                from scipy.stats import f_oneway

                # Encode segment as integer categories
                seg_encoded = pd.Categorical(subset[seg_col]).codes
                treat = subset["_is_treatment"].values
                metric = subset[metric_col].values

                # Interaction column
                interaction = treat * seg_encoded.astype(float)

                # Build design matrix (no intercept removal needed; use OLS via numpy)
                X = np.column_stack([
                    np.ones(len(treat)),
                    treat.astype(float),
                    seg_encoded.astype(float),
                    interaction,
                ])
                y = metric.astype(float)
                valid = ~np.isnan(y)
                X, y = X[valid], y[valid]

                if X.shape[0] > X.shape[1]:
                    try:
                        coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
                        y_hat = X @ coeffs
                        ss_res = np.sum((y - y_hat) ** 2)
                        ss_tot = np.sum((y - y.mean()) ** 2)
                        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

                        # F-test for interaction term (column index 3)
                        n_obs = len(y)
                        p_full = X.shape[1]
                        X_reduced = X[:, :3]
                        coeffs_r, *_ = np.linalg.lstsq(X_reduced, y, rcond=None)
                        y_hat_r = X_reduced @ coeffs_r
                        ss_res_r = np.sum((y - y_hat_r) ** 2)
                        df_num = 1
                        df_den = n_obs - p_full
                        if df_den > 0 and ss_res > 0:
                            f_stat = ((ss_res_r - ss_res) / df_num) / (ss_res / df_den)
                            interaction_p = float(1.0 - stats.f.cdf(f_stat, df_num, df_den))
                    except np.linalg.LinAlgError:
                        interaction_p = None
            except Exception:
                interaction_p = None

            output[seg_col] = {
                "results": seg_results,
                "overall_lift": round(overall_lift, 6),
                "interaction_p_value": (
                    round(interaction_p, 4) if interaction_p is not None else None
                ),
                "n_segments_tested": n_segments,
                "bonferroni_alpha": round(bonferroni_alpha, 6),
                "experiment_id": experiment_id,
            }

        return output

    # ── Segment ranking ───────────────────────────────────────────────────────

    def rank_segments(self, segment_results: dict) -> pd.DataFrame:
        """
        Flatten all SegmentResult objects into a single ranked DataFrame.

        Parameters
        ----------
        segment_results : dict
            Output of analyze_hte().

        Returns
        -------
        pd.DataFrame
            Columns: segment_col, segment_value, n_exposed, n_holdout, lift,
                     lift_ci_lower, lift_ci_upper, p_value, significant,
                     index, recommendation, budget_index
        """
        rows = []

        for seg_col, payload in segment_results.items():
            results = payload["results"] if isinstance(payload, dict) else payload
            for r in results:
                rows.append({
                    "segment_col": r.segment_name,
                    "segment_value": r.segment_value,
                    "n_exposed": r.n_exposed,
                    "n_holdout": r.n_holdout,
                    "lift": r.segment_lift,
                    "lift_ci_lower": r.segment_lift_ci[0],
                    "lift_ci_upper": r.segment_lift_ci[1],
                    "p_value": r.p_value,
                    "significant": r.is_significant,
                    "index": r.index_vs_overall,
                    "recommendation": r.recommendation,
                })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)

        # Budget index: proportional to lift, floored at 0
        positive_lifts = df["lift"].clip(lower=0)
        total_positive = positive_lifts.sum()
        if total_positive > 0:
            df["budget_index"] = (positive_lifts / total_positive).round(4)
        else:
            df["budget_index"] = 0.0

        return df

    # ── Budget allocation ─────────────────────────────────────────────────────

    def optimize_budget_allocation(
        self,
        segment_results: list,
        total_budget: float,
        min_segment_budget: float = 0.05,
    ) -> pd.DataFrame:
        """
        Allocate budget across segments proportional to marginal returns.

        Marginal return proxy: segment_lift * (n_exposed + n_holdout).
        Floor: each included segment receives at least min_segment_budget
        fraction of total_budget.

        Segments with recommendation EXCLUDE or INSUFFICIENT_DATA are omitted.

        Parameters
        ----------
        segment_results : list[SegmentResult]
        total_budget : float
            Total budget in monetary units.
        min_segment_budget : float
            Minimum fraction of total_budget per segment (default 5%).

        Returns
        -------
        pd.DataFrame
            Columns: segment_col, segment_value, recommended_budget,
                     pct_of_total, expected_incremental_conversions
        """
        exclude_recs = {"EXCLUDE", "INSUFFICIENT_DATA"}
        eligible = [
            r for r in segment_results
            if r.recommendation not in exclude_recs and r.segment_lift > 0
        ]

        if not eligible:
            return pd.DataFrame(columns=[
                "segment_col", "segment_value", "recommended_budget",
                "pct_of_total", "expected_incremental_conversions",
            ])

        n_eligible = len(eligible)
        min_abs = min_segment_budget * total_budget
        remaining_budget = total_budget - min_abs * n_eligible

        # Marginal return score
        scores = np.array([
            max(0.0, r.segment_lift) * (r.n_exposed + r.n_holdout)
            for r in eligible
        ], dtype=float)
        total_score = scores.sum()

        if total_score > 0:
            proportional = scores / total_score * remaining_budget
        else:
            proportional = np.full(n_eligible, remaining_budget / n_eligible)

        rows = []
        for r, prop_budget in zip(eligible, proportional):
            budget = min_abs + prop_budget
            pct = budget / total_budget
            # Expected incremental conversions given budget
            # Proxy: lift * (n_exposed/total_exposed) * (budget/total_budget)
            total_exposed = sum(x.n_exposed for x in eligible)
            reach_fraction = r.n_exposed / total_exposed if total_exposed > 0 else 0.0
            exp_conv = r.segment_lift * reach_fraction * (budget / total_budget) * r.n_exposed

            rows.append({
                "segment_col": r.segment_name,
                "segment_value": r.segment_value,
                "recommended_budget": round(budget, 2),
                "pct_of_total": round(pct, 4),
                "expected_incremental_conversions": round(exp_conv, 2),
            })

        return pd.DataFrame(rows).sort_values("recommended_budget", ascending=False).reset_index(drop=True)

    # ── Next-experiment recommendations ───────────────────────────────────────

    def generate_next_experiment_recommendations(
        self,
        segment_results: dict,
        overall_result_dict: dict,
    ) -> list:
        """
        Identify inconclusive segments and generate follow-up experiment specs.

        Inconclusive = not significant but positive trend (MONITOR recommendation).
        Also flags segments with high index but insufficient data.

        Parameters
        ----------
        segment_results : dict
            Output of analyze_hte().
        overall_result_dict : dict
            {'overall_lift': float, 'mde': float, 'max_n': int, ...}

        Returns
        -------
        list[dict]
            Each dict: {segment, segment_value, reason,
                        recommended_mde, recommended_design}
        """
        recommendations = []
        overall_lift = overall_result_dict.get("overall_lift", 0.0)
        overall_max_n = overall_result_dict.get("max_n", 10_000)

        for seg_col, payload in segment_results.items():
            results = payload["results"] if isinstance(payload, dict) else payload

            for r in results:
                if r.recommendation == "MONITOR":
                    # Recommend follow-up targeted at this segment
                    # Required N for 80% power at half the observed lift
                    if r.segment_lift > 0:
                        mde = r.segment_lift * 0.5
                        baseline = abs(r.segment_lift) / 2.0 + 0.05
                        p2 = baseline + mde
                        p1 = baseline
                        z_alpha = stats.norm.ppf(1 - 0.05 / 2)
                        z_beta = stats.norm.ppf(0.80)
                        p_bar = (p1 + p2) / 2
                        n_required = int(
                            np.ceil(
                                2 * p_bar * (1 - p_bar) * (z_alpha + z_beta) ** 2 / mde ** 2
                            )
                        )
                    else:
                        mde = 0.01
                        n_required = overall_max_n

                    recommendations.append({
                        "segment": seg_col,
                        "segment_value": r.segment_value,
                        "reason": (
                            f"Positive trend (lift={r.segment_lift:.4f}, "
                            f"p={r.p_value:.3f}) but not yet significant "
                            f"at Bonferroni-corrected alpha. "
                            f"N={r.n_exposed + r.n_holdout} was likely underpowered."
                        ),
                        "recommended_mde": round(mde, 5),
                        "recommended_n_per_arm": n_required,
                        "recommended_design": (
                            "Targeted A/B test restricted to this segment. "
                            "Use sequential testing with O'Brien-Fleming boundaries "
                            f"to detect MDE={mde:.4f} with 80% power."
                        ),
                    })

                elif r.recommendation == "INSUFFICIENT_DATA" and r.index_vs_overall > 1.2:
                    recommendations.append({
                        "segment": seg_col,
                        "segment_value": r.segment_value,
                        "reason": (
                            f"Segment has high index ({r.index_vs_overall:.2f}x overall) "
                            f"but only {r.n_exposed + r.n_holdout} observations — "
                            f"below minimum of {self.min_segment_size * 2}."
                        ),
                        "recommended_mde": round(overall_lift * 0.8, 5),
                        "recommended_n_per_arm": self.min_segment_size * 3,
                        "recommended_design": (
                            "Increase traffic allocation to this segment in next "
                            "experiment cycle to achieve minimum segment N of "
                            f"{self.min_segment_size * 2} per arm."
                        ),
                    })

        return recommendations

    # ── Report formatting ─────────────────────────────────────────────────────

    def format_segment_report(self, ranked_df: pd.DataFrame, top_n: int = 10) -> str:
        """
        Produce a formatted text report showing top and bottom segments.

        Parameters
        ----------
        ranked_df : pd.DataFrame
            Output of rank_segments().
        top_n : int
            Number of top and bottom segments to show.

        Returns
        -------
        str
        """
        if ranked_df.empty:
            return "No segment results to display."

        lines = []
        lines.append("=" * 90)
        lines.append("SEGMENT OPTIMIZATION REPORT")
        lines.append("=" * 90)

        col_hdr = (
            f"{'Segment':<18}  {'Value':<20}  {'N Exp':>7}  {'Lift':>8}  "
            f"{'CI Lower':>9}  {'CI Upper':>9}  {'P-Val':>7}  "
            f"{'Index':>6}  {'Bgt Idx':>7}  {'Recommendation':<22}"
        )
        sep = "-" * 90

        # Top segments
        lines.append(f"\nTOP {min(top_n, len(ranked_df))} SEGMENTS BY LIFT")
        lines.append(sep)
        lines.append(col_hdr)
        lines.append(sep)
        for _, row in ranked_df.head(top_n).iterrows():
            lines.append(
                f"{str(row['segment_col']):<18}  {str(row['segment_value']):<20}  "
                f"{int(row['n_exposed']):>7,}  {row['lift']:>8.4f}  "
                f"{row['lift_ci_lower']:>9.4f}  {row['lift_ci_upper']:>9.4f}  "
                f"{row['p_value']:>7.4f}  {row['index']:>6.2f}  "
                f"{row['budget_index']:>7.4f}  {str(row['recommendation']):<22}"
            )
        lines.append(sep)

        # Bottom segments (excluding top already shown)
        if len(ranked_df) > top_n:
            bottom = ranked_df.tail(min(top_n, len(ranked_df) - top_n))
            lines.append(f"\nBOTTOM {len(bottom)} SEGMENTS BY LIFT")
            lines.append(sep)
            lines.append(col_hdr)
            lines.append(sep)
            for _, row in bottom.iterrows():
                lines.append(
                    f"{str(row['segment_col']):<18}  {str(row['segment_value']):<20}  "
                    f"{int(row['n_exposed']):>7,}  {row['lift']:>8.4f}  "
                    f"{row['lift_ci_lower']:>9.4f}  {row['lift_ci_upper']:>9.4f}  "
                    f"{row['p_value']:>7.4f}  {row['index']:>6.2f}  "
                    f"{row['budget_index']:>7.4f}  {str(row['recommendation']):<22}"
                )
            lines.append(sep)

        # Recommendation summary
        rec_counts = ranked_df["recommendation"].value_counts()
        lines.append("\nRECOMMENDATION SUMMARY")
        lines.append(sep)
        for rec, cnt in rec_counts.items():
            lines.append(f"  {str(rec):<28} {cnt:>4} segment(s)")
        lines.append(sep)

        return "\n".join(lines)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Segment Optimization — Demonstration")
    print("=" * 70)

    rng = np.random.default_rng(42)
    N = 6_000

    # Synthetic experiment dataset
    industries = ["Technology", "Finance", "Healthcare", "Retail", "Manufacturing"]
    regions    = ["AMER", "EMEA", "APAC"]
    company_sizes = ["SMB", "Mid-Market", "Enterprise"]

    data = pd.DataFrame({
        "variant": rng.choice(["control", "treatment"], size=N),
        "industry": rng.choice(industries, size=N),
        "region": rng.choice(regions, size=N),
        "company_size": rng.choice(company_sizes, size=N),
    })

    # Baseline conversion rate
    base_rate = 0.08
    treatment_mask = data["variant"] == "treatment"

    # Industry-specific uplifts
    industry_uplift = {
        "Technology": 0.04,
        "Finance": 0.02,
        "Healthcare": 0.01,
        "Retail": -0.01,
        "Manufacturing": 0.005,
    }

    data["converted"] = 0.0
    for idx, row in data.iterrows():
        rate = base_rate
        if treatment_mask.iloc[idx if isinstance(idx, int) else data.index.get_loc(idx)]:
            rate += industry_uplift.get(row["industry"], 0.02)
        data.at[idx, "converted"] = int(rng.random() < rate)

    data["converted"] = data["converted"].astype(int)

    optimizer = SegmentOptimizer(
        significance_level=SIGNIFICANCE_LEVEL,
        min_segment_size=100,
    )

    print("\n--- Running HTE Analysis ---")
    hte_results = optimizer.analyze_hte(
        data=data,
        variant_col="variant",
        metric_col="converted",
        segment_cols=["industry", "region", "company_size"],
        control_variant="control",
        treatment_variant="treatment",
        experiment_id="demo_campaign_001",
    )

    print("\n--- Ranking Segments ---")
    ranked = optimizer.rank_segments(hte_results)
    print(optimizer.format_segment_report(ranked, top_n=5))

    print("\n--- Budget Allocation (Total Budget: $500,000) ---")
    all_results = []
    for seg_col, payload in hte_results.items():
        all_results.extend(payload["results"])

    budget_df = optimizer.optimize_budget_allocation(
        segment_results=all_results,
        total_budget=500_000.0,
        min_segment_budget=0.05,
    )
    print(budget_df.to_string(index=False))

    print("\n--- Next Experiment Recommendations ---")
    next_exp = optimizer.generate_next_experiment_recommendations(
        segment_results=hte_results,
        overall_result_dict={"overall_lift": 0.02, "max_n": N},
    )
    for rec in next_exp:
        print(f"\n  Segment  : {rec['segment']} = {rec['segment_value']}")
        print(f"  Reason   : {rec['reason']}")
        print(f"  MDE      : {rec['recommended_mde']:.5f}")
        print(f"  N/arm    : {rec['recommended_n_per_arm']:,}")
        print(f"  Design   : {rec['recommended_design']}")
