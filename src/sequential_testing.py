"""
Sequential Testing Module
=========================
Implements group-sequential hypothesis testing with O'Brien-Fleming and
Lan-DeMets alpha-spending functions, always-valid confidence intervals
(mSPRT-inspired), and full simulation utilities.

Used for continuous monitoring of campaign experiments without inflating
Type-I error rates beyond the nominal significance level.
"""

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import SIGNIFICANCE_LEVEL, POWER, MAX_INTERIM_LOOKS


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class SequentialBoundary:
    """Boundary values for a single interim look in a sequential test."""
    look_number: int
    cumulative_n: int
    alpha_spent: float
    critical_value: float          # nominal z-score critical value at this look
    efficacy_boundary: float       # reject null if |z| > efficacy_boundary
    futility_boundary: float       # stop for futility if |z| < futility_boundary


@dataclass
class SequentialDecision:
    """Outcome of evaluating a single interim look."""
    look_number: int
    cumulative_n: int
    current_z: float
    current_p_value: float
    current_lift: float
    decision: str                  # CONTINUE | STOP_FOR_EFFICACY | STOP_FOR_FUTILITY
    efficacy_boundary: float
    futility_boundary: float
    message: str


# ── Main class ────────────────────────────────────────────────────────────────

class SequentialTester:
    """
    Group-sequential hypothesis tester for campaign A/B experiments.

    Supports O'Brien-Fleming and Lan-DeMets alpha-spending functions.
    Provides always-valid (anytime-valid) confidence intervals via a
    simplified mSPRT-inspired approach.

    Parameters
    ----------
    alpha : float
        Two-sided family-wise Type-I error rate.
    power : float
        Desired power (1 - beta).
    n_looks : int
        Number of pre-planned interim looks (including final).
    spending_function : str
        'obrien_fleming' or 'lan_demets'.
    """

    def __init__(
        self,
        alpha: float = SIGNIFICANCE_LEVEL,
        power: float = POWER,
        n_looks: int = 5,
        spending_function: str = "obrien_fleming",
    ):
        self.alpha = alpha
        self.power = power
        self.n_looks = n_looks
        self.spending_function = spending_function

        # Derived constants
        self._z_alpha_final = stats.norm.ppf(1 - alpha / 2)
        self._z_beta = stats.norm.ppf(power)

    # ── Boundary calculation ──────────────────────────────────────────────────

    def _obrien_fleming_boundary(self, t: float) -> float:
        """
        O'Brien-Fleming efficacy boundary at information fraction t.

        The OBF boundary is conservative (wide) early in the trial and
        converges to the nominal critical value z_{alpha/2} at t = 1.

        Parameters
        ----------
        t : float
            Information fraction: current_n / max_n.  Must be in (0, 1].

        Returns
        -------
        float
            Critical z-score at this information fraction.
        """
        if t <= 0:
            return np.inf
        return self._z_alpha_final * np.sqrt(1.0 / t)

    def _lan_demets_boundary(self, t: float, alpha_spent_so_far: float) -> float:
        """
        Lan-DeMets alpha-spending function (O'Brien-Fleming shape).

        Cumulative alpha spent through information fraction t:
            alpha_spent(t) = 2 * (1 - Phi(z_alpha / sqrt(t)))

        Returns the *incremental* alpha available at this look.

        Parameters
        ----------
        t : float
            Current information fraction.
        alpha_spent_so_far : float
            Cumulative alpha already spent at previous looks.

        Returns
        -------
        float
            Incremental alpha to spend at this look.
        """
        if t <= 0:
            return 0.0
        cumulative = 2.0 * (1.0 - stats.norm.cdf(self._z_alpha_final / np.sqrt(t)))
        incremental = max(0.0, cumulative - alpha_spent_so_far)
        return incremental

    def compute_boundaries(
        self,
        max_sample_size: int,
        looks: Optional[list] = None,
    ) -> list:
        """
        Pre-compute efficacy and futility boundaries for all planned looks.

        Parameters
        ----------
        max_sample_size : int
            Total planned sample size (both arms combined).
        looks : list[int], optional
            Cumulative sample sizes at each look.  If None, evenly spaced.

        Returns
        -------
        list[SequentialBoundary]
        """
        if looks is None:
            looks = [
                int(round(max_sample_size * (k + 1) / self.n_looks))
                for k in range(self.n_looks)
            ]

        boundaries: list[SequentialBoundary] = []
        alpha_spent_so_far = 0.0

        for i, n in enumerate(looks):
            look_number = i + 1
            t = n / max_sample_size  # information fraction

            # ── Efficacy boundary ─────────────────────────────────────────────
            if self.spending_function == "obrien_fleming":
                efficacy_z = self._obrien_fleming_boundary(t)
                # alpha spent is cumulative area beyond OBF boundary
                alpha_spent_cum = 2.0 * (1.0 - stats.norm.cdf(efficacy_z))
                incremental_alpha = max(0.0, alpha_spent_cum - alpha_spent_so_far)
            else:  # lan_demets
                incremental_alpha = self._lan_demets_boundary(t, alpha_spent_so_far)
                if incremental_alpha > 0:
                    efficacy_z = stats.norm.ppf(1 - incremental_alpha / 2)
                else:
                    efficacy_z = np.inf

            alpha_spent_so_far += incremental_alpha

            # ── Futility (beta-spending) boundary ─────────────────────────────
            # Non-binding futility: conditional power < 0.20 => stop for futility
            # Simplified: futility z = z_beta * sqrt(t)
            # (At early looks, this is near 0; rises toward z_beta at end)
            beta = 1.0 - self.power
            futility_z = self._z_beta * np.sqrt(t)

            boundaries.append(
                SequentialBoundary(
                    look_number=look_number,
                    cumulative_n=n,
                    alpha_spent=alpha_spent_so_far,
                    critical_value=self._z_alpha_final,
                    efficacy_boundary=round(efficacy_z, 4),
                    futility_boundary=round(futility_z, 4),
                )
            )

        return boundaries

    # ── Single-look evaluation ────────────────────────────────────────────────

    def evaluate_look(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        look_number: int,
        boundaries: list,
        metric_type: str = "proportion",
    ) -> SequentialDecision:
        """
        Evaluate a single interim look and return a decision.

        Parameters
        ----------
        control_data : np.ndarray
            Observations in the control arm (0/1 for proportion, float for continuous).
        treatment_data : np.ndarray
            Observations in the treatment arm.
        look_number : int
            Which planned look this is (1-indexed).
        boundaries : list[SequentialBoundary]
            Pre-computed boundaries from compute_boundaries().
        metric_type : str
            'proportion' or 'continuous'.

        Returns
        -------
        SequentialDecision
        """
        n_c = len(control_data)
        n_t = len(treatment_data)
        cumulative_n = n_c + n_t

        # Find matching boundary (use last if look_number exceeds list)
        idx = min(look_number - 1, len(boundaries) - 1)
        boundary = boundaries[idx]

        # ── Compute z-statistic ───────────────────────────────────────────────
        if metric_type == "proportion":
            p_c = np.mean(control_data)
            p_t = np.mean(treatment_data)
            p_pool = (np.sum(control_data) + np.sum(treatment_data)) / cumulative_n
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
            if se == 0:
                z = 0.0
            else:
                z = (p_t - p_c) / se
            lift = (p_t - p_c) / p_c if p_c > 0 else 0.0
        else:
            mean_c = np.mean(control_data)
            mean_t = np.mean(treatment_data)
            se = np.sqrt(np.var(control_data, ddof=1) / n_c + np.var(treatment_data, ddof=1) / n_t)
            if se == 0:
                z = 0.0
            else:
                z = (mean_t - mean_c) / se
            lift = (mean_t - mean_c) / abs(mean_c) if mean_c != 0 else 0.0

        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

        # ── Decision logic ────────────────────────────────────────────────────
        eff_b = boundary.efficacy_boundary
        fut_b = boundary.futility_boundary

        if abs(z) >= eff_b:
            decision = "STOP_FOR_EFFICACY"
            direction = "positive" if z > 0 else "negative"
            message = (
                f"Look {look_number}: |z|={abs(z):.3f} exceeds efficacy boundary "
                f"{eff_b:.3f}. Statistically significant {direction} effect detected. "
                f"Lift = {lift:.2%}, p = {p_value:.4f}."
            )
        elif abs(z) < fut_b and look_number > 1:
            decision = "STOP_FOR_FUTILITY"
            message = (
                f"Look {look_number}: |z|={abs(z):.3f} is below futility boundary "
                f"{fut_b:.3f}. Insufficient evidence of a meaningful effect; "
                f"stopping to conserve resources. Lift = {lift:.2%}."
            )
        else:
            decision = "CONTINUE"
            message = (
                f"Look {look_number}: |z|={abs(z):.3f} is between boundaries "
                f"[{fut_b:.3f}, {eff_b:.3f}]. Continue experiment. "
                f"Current lift estimate = {lift:.2%}."
            )

        return SequentialDecision(
            look_number=look_number,
            cumulative_n=cumulative_n,
            current_z=round(z, 4),
            current_p_value=round(p_value, 6),
            current_lift=round(lift, 6),
            decision=decision,
            efficacy_boundary=round(eff_b, 4),
            futility_boundary=round(fut_b, 4),
            message=message,
        )

    # ── Simulation ────────────────────────────────────────────────────────────

    def simulate_sequential_test(
        self,
        true_effect: float,
        baseline_rate: float,
        max_n_per_group: int,
        daily_n: int = 100,
        n_looks: int = 5,
        seed: int = 42,
    ) -> dict:
        """
        Simulate a sequential experiment with data arriving in daily batches.

        Parameters
        ----------
        true_effect : float
            Absolute difference added to baseline_rate for treatment arm.
        baseline_rate : float
            Conversion rate in control arm (for proportion metric).
        max_n_per_group : int
            Maximum observations per arm before forced stop.
        daily_n : int
            Number of new observations per arm per day.
        n_looks : int
            Number of pre-planned interim looks.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            {
                'trace': list of per-look dicts,
                'final_decision': str,
                'stopped_early': bool,
                'stopping_day': int or None,
                'final_n_per_group': int,
                'true_effect': float,
                'baseline_rate': float,
            }
        """
        rng = np.random.default_rng(seed)
        treatment_rate = baseline_rate + true_effect

        max_total_n = 2 * max_n_per_group
        boundaries = self.compute_boundaries(
            max_sample_size=max_total_n,
            looks=[
                int(round(max_total_n * (k + 1) / n_looks))
                for k in range(n_looks)
            ],
        )

        # Determine days at which interim looks occur
        look_ns = [b.cumulative_n for b in boundaries]  # total (both arms)

        control_obs: list[float] = []
        treatment_obs: list[float] = []
        trace: list[dict] = []

        day = 0
        look_idx = 0
        final_decision = "CONTINUE"
        stopping_day: Optional[int] = None

        while len(control_obs) < max_n_per_group and look_idx < len(boundaries):
            day += 1
            batch_c = rng.binomial(1, baseline_rate, size=daily_n).tolist()
            batch_t = rng.binomial(1, treatment_rate, size=daily_n).tolist()
            control_obs.extend(batch_c)
            treatment_obs.extend(batch_t)

            # Cap at max_n_per_group
            if len(control_obs) > max_n_per_group:
                control_obs = control_obs[:max_n_per_group]
                treatment_obs = treatment_obs[:max_n_per_group]

            current_total = len(control_obs) + len(treatment_obs)

            # Check if we've reached or passed the next look threshold
            if current_total >= look_ns[look_idx] or len(control_obs) >= max_n_per_group:
                decision_obj = self.evaluate_look(
                    control_data=np.array(control_obs),
                    treatment_data=np.array(treatment_obs),
                    look_number=look_idx + 1,
                    boundaries=boundaries,
                    metric_type="proportion",
                )

                trace.append({
                    "day": day,
                    "look_number": look_idx + 1,
                    "cumulative_n_per_group": len(control_obs),
                    "cumulative_n_total": current_total,
                    "z_stat": decision_obj.current_z,
                    "p_value": decision_obj.current_p_value,
                    "lift_estimate": decision_obj.current_lift,
                    "efficacy_boundary": decision_obj.efficacy_boundary,
                    "futility_boundary": decision_obj.futility_boundary,
                    "decision": decision_obj.decision,
                    "message": decision_obj.message,
                })

                look_idx += 1
                final_decision = decision_obj.decision

                if decision_obj.decision in ("STOP_FOR_EFFICACY", "STOP_FOR_FUTILITY"):
                    stopping_day = day
                    break

        return {
            "trace": trace,
            "final_decision": final_decision,
            "stopped_early": stopping_day is not None and len(control_obs) < max_n_per_group,
            "stopping_day": stopping_day,
            "final_n_per_group": len(control_obs),
            "true_effect": true_effect,
            "baseline_rate": baseline_rate,
        }

    # ── Always-valid confidence intervals ────────────────────────────────────

    def compute_always_valid_ci(
        self,
        control_cumulative: list,
        treatment_cumulative: list,
        alpha: Optional[float] = None,
    ) -> list:
        """
        Compute mSPRT-inspired always-valid (anytime-valid) confidence intervals.

        At each timepoint t, the CI is widened by a log-correction factor so
        that it maintains coverage simultaneously across all t.  This is a
        simplified asymptotically-valid approximation (not a full mSPRT).

        Parameters
        ----------
        control_cumulative : list of float
            Running list of control observations (individual values).
        treatment_cumulative : list of float
            Running list of treatment observations (individual values).
        alpha : float, optional
            Significance level; defaults to self.alpha.

        Returns
        -------
        list[dict]
            Each dict: {t, n, mean_diff, ci_lower, ci_upper, is_significant}
        """
        if alpha is None:
            alpha = self.alpha

        control_arr = np.array(control_cumulative, dtype=float)
        treatment_arr = np.array(treatment_cumulative, dtype=float)
        n = min(len(control_arr), len(treatment_arr))

        results = []

        for t in range(1, n + 1):
            c_slice = control_arr[:t]
            tr_slice = treatment_arr[:t]

            mean_diff = np.mean(tr_slice) - np.mean(c_slice)

            # Variance of difference of means
            var_c = np.var(c_slice, ddof=1) if t > 1 else 0.25
            var_t = np.var(tr_slice, ddof=1) if t > 1 else 0.25
            se = np.sqrt((var_c + var_t) / t)

            # Anytime-valid inflation: multiply by sqrt(log(t+1) + 1)
            # This is a heuristic correction that bounds the coverage probability
            # uniformly over time (approximate, not exact).
            if t > 1:
                inflation = np.sqrt(np.log(t + 1) + 1.0)
            else:
                inflation = 1.0

            z_crit = stats.norm.ppf(1 - alpha / 2) * inflation
            margin = z_crit * se

            ci_lower = mean_diff - margin
            ci_upper = mean_diff + margin
            is_significant = (ci_lower > 0) or (ci_upper < 0)

            results.append({
                "t": t,
                "n": t * 2,
                "mean_diff": round(mean_diff, 6),
                "ci_lower": round(ci_lower, 6),
                "ci_upper": round(ci_upper, 6),
                "is_significant": is_significant,
            })

        return results

    # ── Reporting ─────────────────────────────────────────────────────────────

    def format_sequential_report(self, decisions: list) -> str:
        """
        Produce a formatted ASCII table summarising all interim looks.

        Parameters
        ----------
        decisions : list[SequentialDecision]

        Returns
        -------
        str
        """
        if not decisions:
            return "No interim looks to report."

        header = (
            f"{'Look':>4}  {'N':>8}  {'Z-Stat':>8}  {'P-Value':>9}  "
            f"{'Lift':>8}  {'Eff.Bnd':>8}  {'Fut.Bnd':>8}  {'Decision':<24}"
        )
        sep = "-" * len(header)

        rows = [sep, header, sep]
        for d in decisions:
            rows.append(
                f"{d.look_number:>4}  {d.cumulative_n:>8,}  {d.current_z:>8.3f}  "
                f"{d.current_p_value:>9.4f}  {d.current_lift:>8.2%}  "
                f"{d.efficacy_boundary:>8.3f}  {d.futility_boundary:>8.3f}  "
                f"{d.decision:<24}"
            )
        rows.append(sep)

        final = decisions[-1]
        rows.append(f"\nFinal decision: {final.decision}")
        rows.append(f"Final message : {final.message}")

        return "\n".join(rows)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Sequential Testing — Demonstration")
    print("=" * 70)

    tester = SequentialTester(
        alpha=0.05,
        power=0.80,
        n_looks=5,
        spending_function="obrien_fleming",
    )

    MAX_N = 5_000  # total (both arms)

    print("\n--- Pre-computed O'Brien-Fleming Boundaries (max N=5,000) ---")
    boundaries = tester.compute_boundaries(max_sample_size=MAX_N)
    print(
        f"{'Look':>4}  {'Cum N':>6}  {'Alpha Spent':>11}  "
        f"{'Efficacy Z':>10}  {'Futility Z':>10}"
    )
    for b in boundaries:
        print(
            f"{b.look_number:>4}  {b.cumulative_n:>6,}  {b.alpha_spent:>11.4f}  "
            f"{b.efficacy_boundary:>10.4f}  {b.futility_boundary:>10.4f}"
        )

    print("\n--- Simulation: true_effect=+0.02, baseline=0.10 ---")
    result = tester.simulate_sequential_test(
        true_effect=0.02,
        baseline_rate=0.10,
        max_n_per_group=2_500,
        daily_n=200,
        n_looks=5,
        seed=42,
    )
    decisions = [
        SequentialDecision(
            look_number=row["look_number"],
            cumulative_n=row["cumulative_n_total"],
            current_z=row["z_stat"],
            current_p_value=row["p_value"],
            current_lift=row["lift_estimate"],
            decision=row["decision"],
            efficacy_boundary=row["efficacy_boundary"],
            futility_boundary=row["futility_boundary"],
            message=row["message"],
        )
        for row in result["trace"]
    ]
    print(tester.format_sequential_report(decisions))
    print(f"\nStopped early: {result['stopped_early']}")
    print(f"Stopping day : {result['stopping_day']}")
    print(f"Final N/group: {result['final_n_per_group']:,}")

    print("\n--- Always-Valid CIs (first 10 timepoints shown) ---")
    rng = np.random.default_rng(99)
    ctrl = rng.binomial(1, 0.10, 300).tolist()
    trt  = rng.binomial(1, 0.12, 300).tolist()
    ci_results = tester.compute_always_valid_ci(ctrl, trt)
    print(
        f"{'t':>5}  {'N':>6}  {'Mean Diff':>10}  {'CI Lower':>10}  "
        f"{'CI Upper':>10}  {'Sig?':>5}"
    )
    for row in ci_results[::30]:  # every 30th point
        print(
            f"{row['t']:>5}  {row['n']:>6}  {row['mean_diff']:>10.4f}  "
            f"{row['ci_lower']:>10.4f}  {row['ci_upper']:>10.4f}  "
            f"{'Yes' if row['is_significant'] else 'No':>5}"
        )
