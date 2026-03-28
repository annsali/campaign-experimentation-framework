"""
experiment_designer.py
----------------------
Pre-experiment planning tools for the Campaign Experimentation & Lift
Measurement Framework.

Provides:
  - SampleSizeResult  : dataclass for sample-size calculation outputs
  - RandomizationResult : dataclass for subject-assignment outputs
  - ExperimentDesigner  : orchestrates all design-phase decisions
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats

# ---------------------------------------------------------------------------
# Path setup – allow running as a script from any cwd
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    POWER,
    RANDOM_SEED,
    SIGNIFICANCE_LEVEL,
    VISUALS_DIR,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SampleSizeResult:
    """Encapsulates the output of a sample-size power calculation."""

    baseline_rate: float
    mde: float                    # minimum detectable effect (absolute)
    significance_level: float
    power: float
    num_variants: int
    sample_size_per_variant: int
    total_sample_size: int
    metric_type: str              # "proportion" | "continuous" | "ratio"
    adjusted_alpha: float         # after Bonferroni correction when > 2 variants

    def __str__(self) -> str:
        lines = [
            f"SampleSizeResult({self.metric_type})",
            f"  baseline            : {self.baseline_rate:.4f}",
            f"  MDE (absolute)      : {self.mde:.4f}",
            f"  significance level  : {self.significance_level:.4f}",
            f"  adjusted alpha      : {self.adjusted_alpha:.4f}",
            f"  power               : {self.power:.2f}",
            f"  variants            : {self.num_variants}",
            f"  n per variant       : {self.sample_size_per_variant:,}",
            f"  total n             : {self.total_sample_size:,}",
        ]
        return "\n".join(lines)


@dataclass
class RandomizationResult:
    """Encapsulates the output of a subject-randomization step."""

    assignment_df: pd.DataFrame              # contact_id, variant, assignment_hash, strat cols
    balance_check: Dict[str, dict]           # chi-squared results per stratification variable
    is_balanced: bool
    balance_summary: str

    def __str__(self) -> str:
        return (
            f"RandomizationResult("
            f"n={len(self.assignment_df)}, "
            f"is_balanced={self.is_balanced})\n"
            f"{self.balance_summary}"
        )


# ---------------------------------------------------------------------------
# ExperimentDesigner
# ---------------------------------------------------------------------------

class ExperimentDesigner:
    """
    Pre-experiment planning toolkit.

    Handles sample-size calculations (proportion & continuous), power curves,
    deterministic hash-based randomization with optional stratification, and
    experiment duration estimation.

    Parameters
    ----------
    significance_level : float
        Default two-sided alpha (default: config.SIGNIFICANCE_LEVEL).
    power : float
        Default statistical power 1 - beta (default: config.POWER).
    seed : int
        Default random seed for deterministic randomization (default: config.RANDOM_SEED).
    """

    def __init__(
        self,
        significance_level: float = SIGNIFICANCE_LEVEL,
        power: float = POWER,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.significance_level = significance_level
        self.power = power
        self.seed = seed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_params(
        self,
        significance_level: Optional[float],
        power: Optional[float],
    ):
        """Return (alpha, power) using instance defaults when None is passed."""
        alpha = significance_level if significance_level is not None else self.significance_level
        pwr = power if power is not None else self.power
        return alpha, pwr

    @staticmethod
    def _bonferroni_alpha(significance_level: float, num_variants: int) -> float:
        """
        Return Bonferroni-corrected alpha for family-wise error control.

        For a standard A/B test (2 variants) no correction is applied.
        For k variants there are (k-1) treatment-vs-control comparisons.
        """
        if num_variants <= 2:
            return significance_level
        return significance_level / (num_variants - 1)

    # ------------------------------------------------------------------
    # Sample-size: proportion metric
    # ------------------------------------------------------------------

    def calculate_sample_size_proportion(
        self,
        baseline_rate: float,
        mde: float,
        significance_level: Optional[float] = None,
        power: Optional[float] = None,
        num_variants: int = 2,
    ) -> SampleSizeResult:
        """
        Calculate the per-variant sample size for a two-proportion z-test.

        Uses the exact formula with separate variance terms for each proportion
        (Fleiss et al.) rather than the pooled approximation.

        Parameters
        ----------
        baseline_rate : float
            Observed conversion/open/click rate in the control group (0 < p < 1).
        mde : float
            Minimum detectable effect as an *absolute* change (e.g. 0.03 = 3 pp).
        significance_level : float, optional
            Two-sided alpha; falls back to instance default.
        power : float, optional
            Desired power; falls back to instance default.
        num_variants : int
            Total number of variants including control (>= 2).

        Returns
        -------
        SampleSizeResult
        """
        # --- edge-case guards ---
        if not (0.0 < baseline_rate < 1.0):
            raise ValueError(
                f"baseline_rate must be strictly between 0 and 1, got {baseline_rate}"
            )
        if mde == 0.0:
            raise ValueError("mde must be non-zero; a zero effect is not detectable.")
        if abs(mde) >= 1.0:
            raise ValueError(f"|mde| must be < 1.0, got {mde}")

        alpha, pwr = self._resolve_params(significance_level, power)
        adjusted_alpha = self._bonferroni_alpha(alpha, num_variants)

        p1 = baseline_rate
        p2 = baseline_rate + mde

        # Clamp p2 to valid range
        p2 = max(1e-6, min(1 - 1e-6, p2))

        z_alpha = scipy.stats.norm.ppf(1 - adjusted_alpha / 2)
        z_beta = scipy.stats.norm.ppf(pwr)

        pooled_p = (p1 + p2) / 2
        numerator = (
            z_alpha * math.sqrt(2 * pooled_p * (1 - pooled_p))
            + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        ) ** 2
        denominator = (p2 - p1) ** 2

        n = math.ceil(numerator / denominator)

        return SampleSizeResult(
            baseline_rate=baseline_rate,
            mde=mde,
            significance_level=alpha,
            power=pwr,
            num_variants=num_variants,
            sample_size_per_variant=n,
            total_sample_size=n * num_variants,
            metric_type="proportion",
            adjusted_alpha=adjusted_alpha,
        )

    # ------------------------------------------------------------------
    # Sample-size: continuous metric
    # ------------------------------------------------------------------

    def calculate_sample_size_continuous(
        self,
        baseline_mean: float,
        baseline_std: float,
        mde_absolute: float,
        significance_level: Optional[float] = None,
        power: Optional[float] = None,
        num_variants: int = 2,
    ) -> SampleSizeResult:
        """
        Calculate the per-variant sample size for an independent two-sample t-test.

        Parameters
        ----------
        baseline_mean : float
            Mean of the metric in control (used as baseline_rate in the result).
        baseline_std : float
            Standard deviation of the metric in control (must be > 0).
        mde_absolute : float
            Minimum detectable absolute difference between means.
        significance_level : float, optional
        power : float, optional
        num_variants : int

        Returns
        -------
        SampleSizeResult
            ``baseline_rate`` is set to ``baseline_mean`` for reference;
            ``mde`` is the absolute effect size.
        """
        if baseline_std <= 0:
            raise ValueError(f"baseline_std must be > 0, got {baseline_std}")
        if mde_absolute == 0.0:
            raise ValueError("mde_absolute must be non-zero.")

        alpha, pwr = self._resolve_params(significance_level, power)
        adjusted_alpha = self._bonferroni_alpha(alpha, num_variants)

        z_alpha = scipy.stats.norm.ppf(1 - adjusted_alpha / 2)
        z_beta = scipy.stats.norm.ppf(pwr)

        d = abs(mde_absolute) / baseline_std          # Cohen's d
        n = math.ceil(2 * ((z_alpha + z_beta) / d) ** 2)

        return SampleSizeResult(
            baseline_rate=baseline_mean,
            mde=mde_absolute,
            significance_level=alpha,
            power=pwr,
            num_variants=num_variants,
            sample_size_per_variant=n,
            total_sample_size=n * num_variants,
            metric_type="continuous",
            adjusted_alpha=adjusted_alpha,
        )

    # ------------------------------------------------------------------
    # Achievable MDE given a fixed sample size
    # ------------------------------------------------------------------

    def compute_achievable_mde(
        self,
        sample_size_per_variant: int,
        baseline_rate: float,
        significance_level: Optional[float] = None,
        power: Optional[float] = None,
    ) -> float:
        """
        Find the smallest absolute MDE detectable with the given sample size.

        Uses Brent's method (``scipy.optimize.brentq``) on the proportion
        sample-size formula, solving for mde such that the required n equals
        ``sample_size_per_variant``.

        Parameters
        ----------
        sample_size_per_variant : int
            Available sample per variant arm.
        baseline_rate : float
            Control conversion rate (0 < p < 1).
        significance_level : float, optional
        power : float, optional

        Returns
        -------
        float
            Minimum detectable effect (absolute).
        """
        alpha, pwr = self._resolve_params(significance_level, power)

        upper_bound = min(baseline_rate, 1.0 - baseline_rate) - 1e-6
        if upper_bound <= 0:
            raise ValueError(
                "baseline_rate is too close to 0 or 1 to compute an MDE."
            )

        def _residual(mde: float) -> float:
            result = self.calculate_sample_size_proportion(
                baseline_rate=baseline_rate,
                mde=mde,
                significance_level=alpha,
                power=pwr,
            )
            return result.sample_size_per_variant - sample_size_per_variant

        try:
            achievable_mde = scipy.optimize.brentq(
                _residual,
                a=1e-6,
                b=upper_bound,
                xtol=1e-6,
                maxiter=500,
            )
        except ValueError:
            # If even the smallest effect requires fewer subjects, return near-zero
            achievable_mde = 1e-6

        return float(achievable_mde)

    # ------------------------------------------------------------------
    # Power at various effect sizes
    # ------------------------------------------------------------------

    def compute_power_at_effect(
        self,
        sample_size_per_variant: int,
        baseline_rate: float,
        effect_sizes: List[float],
        significance_level: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Compute achieved statistical power for a range of effect sizes.

        Parameters
        ----------
        sample_size_per_variant : int
            Fixed sample per arm.
        baseline_rate : float
            Control rate (0 < p < 1).
        effect_sizes : list of float
            Absolute changes in rate to evaluate.
        significance_level : float, optional

        Returns
        -------
        pd.DataFrame
            Columns: effect_size, power, is_detectable (power >= self.power).
        """
        alpha, _ = self._resolve_params(significance_level, None)
        z_alpha = scipy.stats.norm.ppf(1 - alpha / 2)
        n = sample_size_per_variant

        records = []
        for effect in effect_sizes:
            if effect == 0.0:
                records.append(
                    {"effect_size": effect, "power": alpha, "is_detectable": False}
                )
                continue

            p = baseline_rate
            p2 = max(1e-9, min(1 - 1e-9, p + effect))
            pooled_p = (p + p2) / 2
            se = math.sqrt(2 * pooled_p * (1 - pooled_p) / n)

            if se == 0:
                pwr = 0.0
            else:
                pwr = (
                    scipy.stats.norm.sf(z_alpha - abs(effect) / se)
                    + scipy.stats.norm.cdf(-z_alpha - abs(effect) / se)
                )

            records.append(
                {
                    "effect_size": effect,
                    "power": float(pwr),
                    "is_detectable": pwr >= self.power,
                }
            )

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Power curve plot
    # ------------------------------------------------------------------

    def plot_power_curve(
        self,
        sample_size_per_variant: int,
        baseline_rate: float,
        significance_level: Optional[float] = None,
        mde: Optional[float] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """
        Generate and save a power-vs-effect-size curve.

        A horizontal reference line is drawn at power = 0.80.  If ``mde`` is
        supplied, a vertical line marks that effect.

        Parameters
        ----------
        sample_size_per_variant : int
        baseline_rate : float
        significance_level : float, optional
        mde : float, optional
            If provided, draw a vertical line and annotate the MDE.
        save_path : str, optional
            Full file path (including extension) for the saved figure.
            Defaults to ``VISUALS_DIR/power_curve_n{n}.png``.

        Returns
        -------
        str
            Absolute path to the saved figure.
        """
        alpha, _ = self._resolve_params(significance_level, None)

        upper = 3 * mde if mde else 0.10
        upper = min(upper, min(baseline_rate, 1.0 - baseline_rate) - 1e-4)
        n_points = 200
        effects = np.linspace(1e-5, upper, n_points)

        df = self.compute_power_at_effect(
            sample_size_per_variant=sample_size_per_variant,
            baseline_rate=baseline_rate,
            effect_sizes=effects.tolist(),
            significance_level=alpha,
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(df["effect_size"], df["power"], color="#1f77b4", linewidth=2.5, label="Power")
        ax.axhline(0.80, color="#d62728", linestyle="--", linewidth=1.5, label="80% power threshold")

        if mde is not None:
            ax.axvline(
                mde,
                color="#2ca02c",
                linestyle=":",
                linewidth=2,
                label=f"MDE = {mde:.4f}",
            )
            # Annotate MDE on curve
            mde_power = df.loc[
                (df["effect_size"] - mde).abs().idxmin(), "power"
            ]
            ax.annotate(
                f"MDE={mde:.3f}\npower={mde_power:.2f}",
                xy=(mde, mde_power),
                xytext=(mde + upper * 0.05, mde_power - 0.08),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=9,
                color="#2ca02c",
            )

        ax.fill_between(
            df["effect_size"],
            df["power"],
            0.80,
            where=df["power"] >= 0.80,
            alpha=0.10,
            color="#1f77b4",
            label="Adequately powered region",
        )

        ax.set_xlabel("Absolute Effect Size (proportion difference)", fontsize=11)
        ax.set_ylabel("Statistical Power (1 - β)", fontsize=11)
        ax.set_title(
            f"Power Curve  |  n={sample_size_per_variant:,} per variant  |  "
            f"α={alpha:.3f}  |  baseline={baseline_rate:.3f}",
            fontsize=11,
        )
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()

        if save_path:
            out_path = Path(save_path)
        else:
            visuals = Path(VISUALS_DIR)
            visuals.mkdir(parents=True, exist_ok=True)
            out_path = visuals / f"power_curve_n{sample_size_per_variant}.png"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return str(out_path.resolve())

    # ------------------------------------------------------------------
    # Randomization
    # ------------------------------------------------------------------

    def randomize_subjects(
        self,
        subject_ids: List,
        num_variants: int,
        variant_names: Optional[List[str]] = None,
        stratify_by: Optional[Dict] = None,
        seed: Optional[int] = None,
    ) -> RandomizationResult:
        """
        Assign subjects to variants using deterministic MD5-hash randomization.

        Hash-based assignment ensures reproducibility: the same subject will
        always land in the same variant for a given seed, regardless of order.

        If ``stratify_by`` is provided, assignment is done within each stratum
        to ensure proportional representation across variants.

        Parameters
        ----------
        subject_ids : list
            Unique identifiers for subjects (any hashable type).
        num_variants : int
            Number of experimental arms (>= 2).
        variant_names : list of str, optional
            Labels for each arm.  Defaults to ["control", "variant_1", ...].
        stratify_by : dict, optional
            Mapping ``{subject_id: stratum_value}``.  May contain only a subset
            of subject_ids; unmatched subjects are placed in stratum "unknown".
        seed : int, optional
            Override instance seed for this call.

        Returns
        -------
        RandomizationResult
        """
        if num_variants < 2:
            raise ValueError("num_variants must be >= 2.")

        _seed = seed if seed is not None else self.seed

        if variant_names is None:
            variant_names = ["control"] + [f"variant_{i}" for i in range(1, num_variants)]

        if len(variant_names) != num_variants:
            raise ValueError(
                f"len(variant_names)={len(variant_names)} != num_variants={num_variants}"
            )

        rows = []

        if stratify_by:
            # Group subjects by stratum
            strata: Dict[str, List] = {}
            for sid in subject_ids:
                stratum = str(stratify_by.get(sid, "unknown"))
                strata.setdefault(stratum, []).append(sid)

            for stratum, ids in strata.items():
                # Within each stratum, deterministically order by hash then assign
                # sequentially to ensure near-equal allocation
                hashed = []
                for sid in ids:
                    digest = hashlib.md5(f"{sid}_{_seed}".encode()).hexdigest()
                    hashed.append((sid, digest))
                hashed.sort(key=lambda x: x[1])  # stable ordering within stratum

                for rank, (sid, digest) in enumerate(hashed):
                    variant_idx = rank % num_variants
                    rows.append(
                        {
                            "contact_id": sid,
                            "variant": variant_names[variant_idx],
                            "assignment_hash": digest[:16],
                            "stratum": stratum,
                        }
                    )
        else:
            for sid in subject_ids:
                digest = hashlib.md5(f"{sid}_{_seed}".encode()).hexdigest()
                variant_idx = int(digest[:8], 16) % num_variants
                rows.append(
                    {
                        "contact_id": sid,
                        "variant": variant_names[variant_idx],
                        "assignment_hash": digest[:16],
                    }
                )

        assignment_df = pd.DataFrame(rows)

        # ------------------------------------------------------------------
        # Balance check via chi-squared test of independence
        # ------------------------------------------------------------------
        balance_check: Dict[str, dict] = {}
        all_p_values: List[float] = []

        strat_cols = [c for c in assignment_df.columns if c not in ("contact_id", "variant", "assignment_hash")]

        if strat_cols:
            for col in strat_cols:
                ct = pd.crosstab(assignment_df[col], assignment_df["variant"])
                chi2, p_value, dof, expected = scipy.stats.chi2_contingency(ct)
                balance_check[col] = {
                    "chi2_statistic": float(chi2),
                    "p_value": float(p_value),
                    "degrees_of_freedom": int(dof),
                    "is_balanced": bool(p_value > 0.05),
                }
                all_p_values.append(p_value)
        else:
            # No stratification columns – check overall variant counts
            counts = assignment_df["variant"].value_counts()
            expected_n = len(assignment_df) / num_variants
            chi2 = float(sum((c - expected_n) ** 2 / expected_n for c in counts))
            dof = num_variants - 1
            p_value = float(1 - scipy.stats.chi2.cdf(chi2, dof))
            balance_check["variant_counts"] = {
                "chi2_statistic": chi2,
                "p_value": p_value,
                "degrees_of_freedom": dof,
                "is_balanced": bool(p_value > 0.05),
            }
            all_p_values.append(p_value)

        is_balanced = all(p > 0.05 for p in all_p_values)

        # Summary string
        dist = assignment_df["variant"].value_counts().to_dict()
        lines = ["Balance check results:"]
        for col, res in balance_check.items():
            status = "BALANCED" if res["is_balanced"] else "IMBALANCED"
            lines.append(
                f"  {col}: chi2={res['chi2_statistic']:.3f}, "
                f"p={res['p_value']:.4f} [{status}]"
            )
        lines.append(f"Variant distribution: {dist}")
        balance_summary = "\n".join(lines)

        return RandomizationResult(
            assignment_df=assignment_df,
            balance_check=balance_check,
            is_balanced=is_balanced,
            balance_summary=balance_summary,
        )

    # ------------------------------------------------------------------
    # Duration estimation
    # ------------------------------------------------------------------

    def estimate_duration(
        self,
        required_sample_size_per_variant: int,
        daily_traffic: float,
        num_variants: int,
        novelty_adjustment_days: int = 7,
        effect_ramp_up_days: int = 7,
    ) -> dict:
        """
        Estimate how long the experiment needs to run.

        Parameters
        ----------
        required_sample_size_per_variant : int
        daily_traffic : float
            Total daily eligible subjects (across all variants).
        num_variants : int
        novelty_adjustment_days : int
            Days to discard at the start for novelty-effect bias.
        effect_ramp_up_days : int
            Days for the treatment effect to fully manifest.

        Returns
        -------
        dict with keys: raw_days, adjusted_days, recommended_runtime_days,
            daily_traffic, note.
        """
        if daily_traffic <= 0:
            raise ValueError("daily_traffic must be positive.")

        raw_days = (required_sample_size_per_variant * num_variants) / daily_traffic
        adjusted_days = raw_days + novelty_adjustment_days + effect_ramp_up_days

        return {
            "raw_days": float(raw_days),
            "adjusted_days": math.ceil(adjusted_days),
            "recommended_runtime_days": max(14, math.ceil(adjusted_days)),
            "daily_traffic": daily_traffic,
            "note": "First 7 days discounted for novelty effect",
        }

    # ------------------------------------------------------------------
    # Full design orchestration
    # ------------------------------------------------------------------

    def design_experiment(
        self,
        experiment_name: str,
        hypothesis: str,
        primary_metric: str,
        baseline_rate: float,
        mde: float,
        daily_traffic: float,
        num_variants: int = 2,
        metric_type: str = "proportion",
    ) -> dict:
        """
        Produce a complete experiment design document.

        Orchestrates sample-size calculation and duration estimation and
        returns a structured dictionary suitable for serialization to the
        experiment catalog.

        Parameters
        ----------
        experiment_name : str
        hypothesis : str
        primary_metric : str
        baseline_rate : float
            Baseline conversion rate (proportion) or mean (continuous).
        mde : float
            Absolute minimum detectable effect.
        daily_traffic : float
            Total daily eligible subjects.
        num_variants : int
        metric_type : str
            "proportion" | "continuous"

        Returns
        -------
        dict
        """
        if metric_type == "proportion":
            ss_result = self.calculate_sample_size_proportion(
                baseline_rate=baseline_rate,
                mde=mde,
                num_variants=num_variants,
            )
        elif metric_type == "continuous":
            # For continuous, treat baseline_rate as baseline_mean and mde as absolute.
            # A std of 1.0 is a placeholder; callers should pass explicit std.
            ss_result = self.calculate_sample_size_continuous(
                baseline_mean=baseline_rate,
                baseline_std=max(baseline_rate * 0.5, 1e-3),
                mde_absolute=mde,
                num_variants=num_variants,
            )
        else:
            raise ValueError(f"Unsupported metric_type: {metric_type!r}")

        duration = self.estimate_duration(
            required_sample_size_per_variant=ss_result.sample_size_per_variant,
            daily_traffic=daily_traffic,
            num_variants=num_variants,
        )

        design = {
            "experiment_name": experiment_name,
            "hypothesis": hypothesis,
            "primary_metric": primary_metric,
            "metric_type": metric_type,
            "num_variants": num_variants,
            "baseline_rate": baseline_rate,
            "minimum_detectable_effect": mde,
            "significance_level": self.significance_level,
            "adjusted_alpha": ss_result.adjusted_alpha,
            "power": self.power,
            "sample_size_per_variant": ss_result.sample_size_per_variant,
            "total_sample_size": ss_result.total_sample_size,
            "daily_traffic": daily_traffic,
            "raw_days_required": duration["raw_days"],
            "adjusted_days_required": duration["adjusted_days"],
            "recommended_runtime_days": duration["recommended_runtime_days"],
            "duration_note": duration["note"],
        }
        return design

    # ------------------------------------------------------------------
    # Pretty-print checklist
    # ------------------------------------------------------------------

    def print_design_checklist(self, design: dict) -> None:
        """
        Print a human-readable experiment design checklist.

        Parameters
        ----------
        design : dict
            Output of :meth:`design_experiment`.
        """
        sep = "=" * 65

        print(sep)
        print(f"  EXPERIMENT DESIGN CHECKLIST")
        print(f"  {design.get('experiment_name', 'Unnamed')}")
        print(sep)

        sections = {
            "HYPOTHESIS & METRIC": [
                ("Hypothesis", "hypothesis"),
                ("Primary metric", "primary_metric"),
                ("Metric type", "metric_type"),
            ],
            "STATISTICAL PARAMETERS": [
                ("Significance level (alpha)", "significance_level"),
                ("Adjusted alpha (Bonferroni)", "adjusted_alpha"),
                ("Power (1 - beta)", "power"),
                ("Number of variants", "num_variants"),
            ],
            "EFFECT SIZE & BASELINE": [
                ("Baseline rate / mean", "baseline_rate"),
                ("Minimum detectable effect", "minimum_detectable_effect"),
            ],
            "SAMPLE SIZE": [
                ("Sample size per variant", "sample_size_per_variant"),
                ("Total sample size", "total_sample_size"),
            ],
            "DURATION PLANNING": [
                ("Daily traffic", "daily_traffic"),
                ("Raw days required", "raw_days_required"),
                ("Adjusted days (+ novelty + ramp)", "adjusted_days_required"),
                ("Recommended runtime (days)", "recommended_runtime_days"),
                ("Note", "duration_note"),
            ],
        }

        for section_title, fields in sections.items():
            print(f"\n  [{section_title}]")
            for label, key in fields:
                value = design.get(key, "N/A")
                if isinstance(value, float):
                    formatted = f"{value:,.4f}" if value < 10 else f"{value:,.1f}"
                elif isinstance(value, int):
                    formatted = f"{value:,}"
                else:
                    formatted = str(value)
                print(f"    {'[x]':<5} {label:<40} {formatted}")

        print(f"\n{sep}")


# ---------------------------------------------------------------------------
# __main__ – demonstrate designs for all 5 catalog experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    designer = ExperimentDesigner()

    print("\n" + "#" * 70)
    print("# Campaign Experimentation Framework – Pre-Experiment Planning Demo")
    print("#" * 70)

    # ------------------------------------------------------------------
    # EXP-001: Email Subject Line Personalization (2-variant A/B)
    # baseline open rate ~22%, MDE = 3 pp, ~5 k emails/day
    # ------------------------------------------------------------------
    print("\n\n=== EXP-001: Email Subject Line Personalization ===")
    d1 = designer.design_experiment(
        experiment_name="Email Subject Line Personalization",
        hypothesis="Personalized subject lines will increase open rates by 15%+",
        primary_metric="open_rate",
        baseline_rate=0.22,
        mde=0.03,
        daily_traffic=5_000,
        num_variants=2,
        metric_type="proportion",
    )
    designer.print_design_checklist(d1)
    ss1 = designer.calculate_sample_size_proportion(
        baseline_rate=0.22, mde=0.03, num_variants=2
    )
    print(f"\n  Detailed SampleSizeResult:\n{ss1}")

    # ------------------------------------------------------------------
    # EXP-002: Landing Page Multivariate Test (4 variants, Bonferroni)
    # baseline form-submit ~8%, MDE = 2 pp, ~3 k visits/day
    # ------------------------------------------------------------------
    print("\n\n=== EXP-002: Landing Page Multivariate Test ===")
    d2 = designer.design_experiment(
        experiment_name="Landing Page Multivariate Test",
        hypothesis="Simplified form (Variant C) will increase form submission rate by 20%+",
        primary_metric="form_submit_rate",
        baseline_rate=0.08,
        mde=0.02,
        daily_traffic=3_000,
        num_variants=4,
        metric_type="proportion",
    )
    designer.print_design_checklist(d2)
    ss2 = designer.calculate_sample_size_proportion(
        baseline_rate=0.08, mde=0.02, num_variants=4
    )
    print(f"\n  Detailed SampleSizeResult:\n{ss2}")

    # ------------------------------------------------------------------
    # EXP-003: High-Intent Segment Holdout Test (2-variant)
    # baseline opportunity-creation ~12%, MDE = 5 pp, ~500 accounts/day
    # ------------------------------------------------------------------
    print("\n\n=== EXP-003: High-Intent Segment Holdout Test ===")
    d3 = designer.design_experiment(
        experiment_name="High-Intent Segment Holdout Test",
        hypothesis="Campaign drives 15%+ incremental opportunity creation in High-Intent segment",
        primary_metric="opportunity_created",
        baseline_rate=0.12,
        mde=0.05,
        daily_traffic=500,
        num_variants=2,
        metric_type="proportion",
    )
    designer.print_design_checklist(d3)
    ss3 = designer.calculate_sample_size_proportion(
        baseline_rate=0.12, mde=0.05, num_variants=2
    )
    print(f"\n  Detailed SampleSizeResult:\n{ss3}")

    # ------------------------------------------------------------------
    # EXP-004: Channel Mix A/B Test (2-variant)
    # baseline MQL-conversion ~5%, MDE = 2 pp, ~2 k contacts/day
    # ------------------------------------------------------------------
    print("\n\n=== EXP-004: Channel Mix A/B Test ===")
    d4 = designer.design_experiment(
        experiment_name="Channel Mix A/B Test",
        hypothesis="Email + Paid Social retargeting increases MQL conversion rate vs email-only",
        primary_metric="mql_conversion_rate",
        baseline_rate=0.05,
        mde=0.02,
        daily_traffic=2_000,
        num_variants=2,
        metric_type="proportion",
    )
    designer.print_design_checklist(d4)
    ss4 = designer.calculate_sample_size_proportion(
        baseline_rate=0.05, mde=0.02, num_variants=2
    )
    print(f"\n  Detailed SampleSizeResult:\n{ss4}")

    # ------------------------------------------------------------------
    # EXP-005: Send Time Optimization (3 variants, Bonferroni)
    # baseline open rate ~22%, MDE = 2 pp, ~6 k emails/day
    # ------------------------------------------------------------------
    print("\n\n=== EXP-005: Send Time Optimization ===")
    d5 = designer.design_experiment(
        experiment_name="Send Time Optimization",
        hypothesis="Send time significantly affects open and click rates",
        primary_metric="open_rate",
        baseline_rate=0.22,
        mde=0.02,
        daily_traffic=6_000,
        num_variants=3,
        metric_type="proportion",
    )
    designer.print_design_checklist(d5)
    ss5 = designer.calculate_sample_size_proportion(
        baseline_rate=0.22, mde=0.02, num_variants=3
    )
    print(f"\n  Detailed SampleSizeResult:\n{ss5}")

    # ------------------------------------------------------------------
    # Bonus: achievable MDE and power-curve generation for EXP-001
    # ------------------------------------------------------------------
    print("\n\n=== Bonus: Achievable MDE given n=25,000 (EXP-001 baseline) ===")
    achievable = designer.compute_achievable_mde(
        sample_size_per_variant=25_000,
        baseline_rate=0.22,
    )
    print(f"  Achievable MDE with n=25,000 per variant: {achievable:.4f} ({achievable*100:.2f} pp)")

    print("\n=== Bonus: Power at various effect sizes (EXP-001, n=25,000) ===")
    effects = [0.01, 0.02, 0.025, 0.03, 0.04, 0.05]
    power_df = designer.compute_power_at_effect(
        sample_size_per_variant=25_000,
        baseline_rate=0.22,
        effect_sizes=effects,
    )
    print(power_df.to_string(index=False))

    print("\n=== Bonus: Generating power curve plot for EXP-001 ===")
    curve_path = designer.plot_power_curve(
        sample_size_per_variant=25_000,
        baseline_rate=0.22,
        mde=0.03,
    )
    print(f"  Power curve saved to: {curve_path}")

    # ------------------------------------------------------------------
    # Bonus: randomization demo (small subset)
    # ------------------------------------------------------------------
    print("\n=== Bonus: Randomization demo (100 subjects, 2 variants) ===")
    ids = [f"contact_{i}" for i in range(100)]
    strata = {f"contact_{i}": ("enterprise" if i % 3 == 0 else "smb") for i in range(100)}
    rand_result = designer.randomize_subjects(
        subject_ids=ids,
        num_variants=2,
        variant_names=["control", "treatment"],
        stratify_by={"stratum": strata},
    )
    print(rand_result)
    print(rand_result.assignment_df.head(10).to_string(index=False))
