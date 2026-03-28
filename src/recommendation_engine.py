"""
recommendation_engine.py
========================
Translates statistical A/B test results into plain-English marketing decisions
for the Campaign Experimentation & Lift Measurement Framework.

Supports both Bayesian (BayesianResult) and Frequentist (FrequentistResult) inputs
and generates structured ExperimentReport objects with actionable next-steps.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    PROB_TREATMENT_BETTER_SHIP,
    PROB_TREATMENT_BETTER_EXTEND,
    EXPECTED_LOSS_THRESHOLD,
    GUARDRAIL_RELATIVE_DEGRADATION,
    SIGNIFICANCE_LEVEL,
    PROPORTION_METRICS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Report dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentReport:
    """Fully self-contained experiment report produced by RecommendationEngine."""

    experiment_id: str
    experiment_name: str

    # Top-level verdict
    overall_verdict: str          # SHIP_TREATMENT | SHIP_CONTROL | EXTEND_TEST | NO_WINNER | BLOCKED_BY_GUARDRAIL
    verdict_confidence: str       # "high" | "medium" | "low"

    # Primary metric summary
    primary_metric_name: str
    primary_metric_control: float
    primary_metric_treatment: float
    primary_metric_lift_absolute: float
    primary_metric_lift_relative: float
    primary_metric_confidence: float      # P(treatment better) for Bayesian; (1 - p_value) for frequentist
    primary_metric_ci: tuple[float, float]
    is_primary_significant: bool

    # Supporting analyses
    secondary_metric_results: list[dict] = field(default_factory=list)
    guardrail_results: list[dict] = field(default_factory=list)
    blocked_guardrails: list[str] = field(default_factory=list)
    segment_insights: list[dict] = field(default_factory=list)   # top-3 segments

    # Plain-English outputs
    recommendation_text: str = ""
    estimated_annual_impact: str = ""
    next_steps: list[str] = field(default_factory=list)

    # Metadata
    generated_at: str = ""   # ISO 8601 timestamp


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class RecommendationEngine:
    """
    Translate raw statistical results into actionable ExperimentReports.

    Parameters
    ----------
    prob_ship_threshold : float
        P(treatment > control) required to recommend shipping treatment.
        Default: PROB_TREATMENT_BETTER_SHIP (0.95).
    prob_extend_threshold : float
        P(treatment > control) above which we suggest extending the test rather
        than declaring no winner.  Default: PROB_TREATMENT_BETTER_EXTEND (0.80).
    expected_loss_threshold : float
        Maximum expected loss acceptable when shipping treatment.
        Default: EXPECTED_LOSS_THRESHOLD (0.005).
    """

    def __init__(
        self,
        prob_ship_threshold: float = PROB_TREATMENT_BETTER_SHIP,
        prob_extend_threshold: float = PROB_TREATMENT_BETTER_EXTEND,
        expected_loss_threshold: float = EXPECTED_LOSS_THRESHOLD,
    ) -> None:
        self.prob_ship_threshold = prob_ship_threshold
        self.prob_extend_threshold = prob_extend_threshold
        self.expected_loss_threshold = expected_loss_threshold

    # ── internal helpers ──────────────────────────────────────────────────────

    def _is_bayesian(self, result) -> bool:
        """Return True when *result* is a BayesianResult (duck-typed)."""
        return hasattr(result, "probability_treatment_better")

    def _is_frequentist(self, result) -> bool:
        """Return True when *result* is a FrequentistResult (duck-typed)."""
        return hasattr(result, "p_value") and not self._is_bayesian(result)

    # ── verdict logic ─────────────────────────────────────────────────────────

    def _determine_verdict(
        self,
        primary_result,
        guardrail_results: list,
    ) -> tuple[str, str]:
        """
        Derive the overall verdict and confidence level.

        Returns
        -------
        (verdict, confidence)
            verdict    : one of SHIP_TREATMENT | SHIP_CONTROL | EXTEND_TEST |
                         NO_WINNER | BLOCKED_BY_GUARDRAIL
            confidence : "high" | "medium" | "low"
        """
        # 1. Guardrails take precedence over everything else.
        if guardrail_results:
            for g in guardrail_results:
                if g.get("is_degraded", False):
                    return ("BLOCKED_BY_GUARDRAIL", "high")

        # 2. Bayesian path.
        if self._is_bayesian(primary_result):
            prob = primary_result.probability_treatment_better
            loss = primary_result.expected_loss_treatment

            if prob >= self.prob_ship_threshold and loss <= self.expected_loss_threshold:
                return ("SHIP_TREATMENT", "high")
            if prob >= self.prob_extend_threshold:
                return ("EXTEND_TEST", "medium")
            if prob <= (1.0 - self.prob_ship_threshold):
                return ("SHIP_CONTROL", "high")
            return ("NO_WINNER", "low")

        # 3. Frequentist path.
        if self._is_frequentist(primary_result):
            p = primary_result.p_value
            lift = primary_result.relative_lift

            if p < SIGNIFICANCE_LEVEL and lift > 0:
                return ("SHIP_TREATMENT", "high")
            if p < SIGNIFICANCE_LEVEL and lift < 0:
                return ("SHIP_CONTROL", "high")
            if 0.05 <= p < 0.15:
                return ("EXTEND_TEST", "medium")
            return ("NO_WINNER", "low")

        # 4. Fallback for unknown result types.
        return ("NO_WINNER", "low")

    # ── metric formatting ─────────────────────────────────────────────────────

    def _format_metric_value(self, metric_name: str, value: float) -> str:
        """
        Return a human-readable string for *value* appropriate to the metric type.

        Proportion metrics (open_rate, click_rate, …) are formatted as percentages
        with one decimal place; all other metrics use two decimal places.

        Examples
        --------
        >>> engine._format_metric_value("open_rate", 0.221)
        '22.1%'
        >>> engine._format_metric_value("pipeline_value", 14250.75)
        '14250.75'
        """
        if metric_name in PROPORTION_METRICS:
            return f"{value * 100:.1f}%"
        return f"{value:.2f}"

    # ── annual impact estimate ────────────────────────────────────────────────

    def _estimate_annual_impact(
        self,
        metric_name: str,
        absolute_lift: float,
        sample_size: int,
        test_duration_days: int = 30,
    ) -> str:
        """
        Extrapolate the observed lift to a monthly volume estimate.

        Formula
        -------
        incremental_per_month = absolute_lift
                                * (sample_size / test_duration_days * 30)

        Returns a human-readable string, e.g.
        "+4,200 additional email opens per month" or
        "+$285,000 incremental pipeline per month".
        """
        monthly_volume = sample_size / max(test_duration_days, 1) * 30
        incremental = absolute_lift * monthly_volume

        sign = "+" if incremental >= 0 else ""

        # Dollar metrics — pipeline_value, cost_per_mql
        dollar_metrics = {"pipeline_value", "cost_per_mql"}
        if metric_name in dollar_metrics:
            return f"{sign}${incremental:,.0f} incremental {metric_name.replace('_', ' ')} per month"

        # Proportion metrics — express as additional events
        if metric_name in PROPORTION_METRICS:
            friendly = metric_name.replace("_", " ").replace("rate", "").strip()
            return f"{sign}{incremental:,.0f} additional {friendly}s per month"

        # Generic continuous
        friendly = metric_name.replace("_", " ")
        return f"{sign}{incremental:,.2f} incremental {friendly} per month"

    # ── segment insight extraction ────────────────────────────────────────────

    def _extract_segment_insights(
        self, segment_results: list | None
    ) -> list[dict]:
        """
        Return the top-3 segments sorted by absolute relative lift.

        Each element in *segment_results* is expected to be a dict with at
        least: segment_name, relative_lift, is_significant.
        """
        if not segment_results:
            return []

        sorted_segs = sorted(
            segment_results,
            key=lambda s: abs(s.get("relative_lift", 0.0)),
            reverse=True,
        )
        return sorted_segs[:3]

    # ── recommendation text generator ────────────────────────────────────────

    def _build_recommendation_text(
        self,
        verdict: str,
        primary_metric: str,
        primary_result,
        secondary_results: dict | None,
        segment_insights: list[dict],
        test_duration_days: int,
    ) -> str:
        """
        Produce a concise, plain-English recommendation paragraph.
        """
        ctrl_fmt = self._format_metric_value(
            primary_metric, primary_result.control_metric
        )
        trt_fmt = self._format_metric_value(
            primary_metric, primary_result.treatment_metric
        )

        # Pull CI from whichever result type is present.
        if self._is_bayesian(primary_result):
            ci = primary_result.credible_interval_95
            lift_abs = primary_result.posterior_mean_lift
            lift_rel = (
                primary_result.posterior_mean_lift / primary_result.control_metric
                if primary_result.control_metric != 0
                else 0.0
            )
        else:
            ci = primary_result.confidence_interval_absolute
            lift_abs = primary_result.absolute_lift
            lift_rel = primary_result.relative_lift

        ci_lo_fmt = self._format_metric_value(primary_metric, ci[0])
        ci_hi_fmt = self._format_metric_value(primary_metric, ci[1])
        lift_rel_pct = f"{lift_rel * 100:+.1f}%"
        lift_abs_fmt = self._format_metric_value(primary_metric, abs(lift_abs))

        # Build secondary context.
        sec_context = ""
        if secondary_results:
            sig_names = [
                m for m, r in secondary_results.items()
                if r.get("is_significant", False)
            ]
            if sig_names:
                sec_context = (
                    f" Secondary metrics also significant: {', '.join(sig_names)}."
                )

        # Segment insight.
        seg_context = ""
        if segment_insights:
            top = segment_insights[0]
            seg_name = top.get("segment_name", "unknown segment")
            seg_lift = top.get("relative_lift", 0.0)
            seg_context = (
                f" Strongest lift in segment '{seg_name}' ({seg_lift * 100:+.1f}%)."
            )

        # ── SHIP_TREATMENT ────────────────────────────────────────────────────
        if verdict == "SHIP_TREATMENT":
            return (
                f"Ship treatment. {primary_metric.replace('_', ' ').title()} improved "
                f"from {ctrl_fmt} to {trt_fmt} ({lift_rel_pct} relative lift). "
                f"95% CI: [{ci_lo_fmt}, {ci_hi_fmt}]."
                f"{sec_context}{seg_context}"
            )

        # ── SHIP_CONTROL ──────────────────────────────────────────────────────
        if verdict == "SHIP_CONTROL":
            return (
                f"Ship control. Treatment degraded {primary_metric.replace('_', ' ')} "
                f"from {ctrl_fmt} to {trt_fmt} ({lift_rel_pct} relative change). "
                f"95% CI: [{ci_lo_fmt}, {ci_hi_fmt}]. "
                f"Retain the current version."
            )

        # ── EXTEND_TEST ───────────────────────────────────────────────────────
        if verdict == "EXTEND_TEST":
            if self._is_bayesian(primary_result):
                conf_pct = primary_result.probability_treatment_better * 100
            else:
                conf_pct = (1.0 - primary_result.p_value) * 100

            # Rough guidance: run until we reach roughly 2× current N.
            n_current = (
                primary_result.sample_size_control
                + primary_result.sample_size_treatment
            )
            additional_days = max(7, test_duration_days)
            additional_n = int(n_current * 0.5)

            return (
                f"Continue test for approximately {additional_days} more days "
                f"to collect ~{additional_n:,} additional observations. "
                f"Currently {conf_pct:.0f}% confident treatment is better, "
                f"but this falls below the {self.prob_ship_threshold * 100:.0f}% "
                f"threshold required to ship."
            )

        # ── NO_WINNER ─────────────────────────────────────────────────────────
        if verdict == "NO_WINNER":
            return (
                "No significant difference detected between control and treatment. "
                "Consider testing a larger effect size, revising the hypothesis, "
                "or running a follow-up experiment with a more targeted audience."
            )

        # ── BLOCKED_BY_GUARDRAIL ──────────────────────────────────────────────
        if verdict == "BLOCKED_BY_GUARDRAIL":
            return (
                "DO NOT SHIP. A guardrail metric has degraded beyond the acceptable "
                f"threshold of {GUARDRAIL_RELATIVE_DEGRADATION * 100:.0f}%. "
                "Investigate the root cause before making any rollout decisions. "
                "Consider pausing the experiment to prevent further harm."
            )

        return "No recommendation available."

    # ── next-steps generator ──────────────────────────────────────────────────

    def _build_next_steps(
        self,
        verdict: str,
        blocked_guardrails: list[str],
        segment_insights: list[dict],
        experiment_id: str,
    ) -> list[str]:
        """
        Return 3-5 concrete next-step bullet points appropriate to the verdict.
        """
        steps: list[str] = []

        if verdict == "SHIP_TREATMENT":
            steps = [
                "Coordinate with engineering to roll out treatment to 100% of eligible audience.",
                "Monitor primary and guardrail metrics for 2 weeks post-full-launch.",
                "Document experiment learnings in the experiment catalog.",
                "Schedule a post-launch readout to share results with stakeholders.",
            ]
            if segment_insights:
                top_seg = segment_insights[0].get("segment_name", "top segment")
                steps.append(
                    f"Consider a follow-up experiment targeting '{top_seg}' "
                    "where the largest lift was observed."
                )

        elif verdict == "SHIP_CONTROL":
            steps = [
                "Retain the current control version; do not deploy treatment.",
                "Conduct a root-cause analysis to understand why treatment underperformed.",
                "Revisit creative or copy strategy before designing the next iteration.",
                "Document findings in the experiment catalog to inform future tests.",
                "Consider exploring an alternative hypothesis with the insights gained.",
            ]

        elif verdict == "EXTEND_TEST":
            steps = [
                "Continue running the experiment without changes until the required sample size is reached.",
                "Do not peek at results daily — set a calendar reminder for the next planned analysis date.",
                "Ensure no external changes (promotions, product changes) affect either variant.",
                "Re-evaluate results once the target sample size is achieved.",
                "If the test is cost-prohibitive to extend, consider a smaller MDE or a power re-calculation.",
            ]

        elif verdict == "NO_WINNER":
            steps = [
                "Do not ship treatment — no evidence of improvement.",
                "Archive experiment results and document the null finding.",
                "Re-examine the minimum detectable effect and consider a larger variation.",
                "Explore qualitative research (surveys, session recordings) to generate new hypotheses.",
                "Review targeting criteria — a broader audience may dilute a real effect.",
            ]

        elif verdict == "BLOCKED_BY_GUARDRAIL":
            steps = [
                f"URGENT: Investigate degradation in guardrail metric(s): {', '.join(blocked_guardrails)}.",
                "Pause experiment traffic if degradation is ongoing.",
                "Engage engineering to identify any bugs or unintended side-effects in the treatment.",
                "Do not proceed with any rollout until guardrail metrics are restored to baseline.",
                "Schedule a post-mortem to prevent similar issues in future experiments.",
            ]

        return steps

    # ── public API ────────────────────────────────────────────────────────────

    def generate_report(
        self,
        experiment_id: str,
        experiment_name: str,
        primary_metric: str,
        primary_result,
        secondary_results: dict | None = None,
        guardrail_results: list | None = None,
        segment_results: list | None = None,
        campaign_cost: float | None = None,
        test_duration_days: int = 30,
    ) -> ExperimentReport:
        """
        Orchestrate all sub-methods to produce a complete ExperimentReport.

        Parameters
        ----------
        experiment_id : str
            Unique identifier for the experiment (e.g. "EXP-2024-042").
        experiment_name : str
            Human-readable experiment title.
        primary_metric : str
            Name of the primary success metric.
        primary_result : BayesianResult | FrequentistResult
            Statistical result for the primary metric.
        secondary_results : dict, optional
            Mapping of metric_name -> dict with keys is_significant, absolute_lift,
            p_value (frequentist) or probability_treatment_better (Bayesian).
        guardrail_results : list of dict, optional
            Each dict must contain: metric_name, is_degraded, relative_change, p_value.
        segment_results : list of dict, optional
            HTE segment results; each dict needs segment_name, relative_lift,
            is_significant.
        campaign_cost : float, optional
            Monthly cost of campaign (not used currently, reserved for ROI calc).
        test_duration_days : int
            Duration of the experiment for annualised impact projection.

        Returns
        -------
        ExperimentReport
        """
        guardrail_results = guardrail_results or []
        secondary_results = secondary_results or {}

        # ── verdict ──────────────────────────────────────────────────────────
        verdict, confidence = self._determine_verdict(primary_result, guardrail_results)

        # ── blocked guardrails list ───────────────────────────────────────────
        blocked = [
            g["metric_name"]
            for g in guardrail_results
            if g.get("is_degraded", False)
        ]

        # ── primary metric signals ────────────────────────────────────────────
        if self._is_bayesian(primary_result):
            ci = primary_result.credible_interval_95
            lift_abs = primary_result.posterior_mean_lift
            lift_rel = (
                primary_result.posterior_mean_lift / primary_result.control_metric
                if primary_result.control_metric != 0
                else 0.0
            )
            confidence_val = primary_result.probability_treatment_better
            is_significant = (
                primary_result.probability_treatment_better >= self.prob_ship_threshold
                or primary_result.probability_treatment_better
                <= (1.0 - self.prob_ship_threshold)
            )
        else:
            ci = primary_result.confidence_interval_absolute
            lift_abs = primary_result.absolute_lift
            lift_rel = primary_result.relative_lift
            confidence_val = 1.0 - primary_result.p_value
            is_significant = primary_result.is_significant

        # ── sample size for impact estimate ───────────────────────────────────
        total_n = (
            primary_result.sample_size_control + primary_result.sample_size_treatment
        )

        # ── segments ──────────────────────────────────────────────────────────
        segment_insights = self._extract_segment_insights(segment_results)

        # ── recommendation text ───────────────────────────────────────────────
        rec_text = self._build_recommendation_text(
            verdict,
            primary_metric,
            primary_result,
            secondary_results,
            segment_insights,
            test_duration_days,
        )

        # ── annual impact ─────────────────────────────────────────────────────
        impact_str = self._estimate_annual_impact(
            primary_metric, lift_abs, total_n, test_duration_days
        )

        # ── next steps ────────────────────────────────────────────────────────
        next_steps = self._build_next_steps(
            verdict, blocked, segment_insights, experiment_id
        )

        # ── secondary metric dicts ────────────────────────────────────────────
        sec_list: list[dict] = []
        for mname, mdata in secondary_results.items():
            sec_list.append(
                {
                    "metric_name": mname,
                    "absolute_lift": mdata.get("absolute_lift", 0.0),
                    "p_value": mdata.get("p_value", None),
                    "probability_treatment_better": mdata.get(
                        "probability_treatment_better", None
                    ),
                    "is_significant": mdata.get("is_significant", False),
                }
            )

        return ExperimentReport(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            overall_verdict=verdict,
            verdict_confidence=confidence,
            primary_metric_name=primary_metric,
            primary_metric_control=primary_result.control_metric,
            primary_metric_treatment=primary_result.treatment_metric,
            primary_metric_lift_absolute=lift_abs,
            primary_metric_lift_relative=lift_rel,
            primary_metric_confidence=confidence_val,
            primary_metric_ci=tuple(ci),
            is_primary_significant=is_significant,
            secondary_metric_results=sec_list,
            guardrail_results=guardrail_results,
            blocked_guardrails=blocked,
            segment_insights=segment_insights,
            recommendation_text=rec_text,
            estimated_annual_impact=impact_str,
            next_steps=next_steps,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    # ── console report formatter ──────────────────────────────────────────────

    def format_console_report(self, report: ExperimentReport) -> str:
        """
        Return a fully formatted, fixed-width console report string.

        Example output
        --------------
        ==================================================
        EXPERIMENT REPORT: Subject Line Test (EXP-2024-001)
        ==================================================

        VERDICT: SHIP TREATMENT  [confidence: high]
        ...
        """
        sep = "=" * 60
        lines: list[str] = [
            sep,
            f"EXPERIMENT REPORT: {report.experiment_name} ({report.experiment_id})",
            sep,
            "",
        ]

        # ── Verdict ───────────────────────────────────────────────────────────
        verdict_display = report.overall_verdict.replace("_", " ")
        lines += [
            f"VERDICT: {verdict_display}  [confidence: {report.verdict_confidence}]",
            "",
        ]

        # ── Primary metric ────────────────────────────────────────────────────
        ctrl_fmt = self._format_metric_value(
            report.primary_metric_name, report.primary_metric_control
        )
        trt_fmt = self._format_metric_value(
            report.primary_metric_name, report.primary_metric_treatment
        )
        lift_abs_fmt = self._format_metric_value(
            report.primary_metric_name, abs(report.primary_metric_lift_absolute)
        )
        lift_sign = "+" if report.primary_metric_lift_absolute >= 0 else "-"
        lift_rel_pct = f"{report.primary_metric_lift_relative * 100:+.1f}%"
        ci_lo = self._format_metric_value(
            report.primary_metric_name, report.primary_metric_ci[0]
        )
        ci_hi = self._format_metric_value(
            report.primary_metric_name, report.primary_metric_ci[1]
        )
        conf_pct = report.primary_metric_confidence * 100

        lines += [
            f"PRIMARY METRIC ({report.primary_metric_name}):",
            f"  Control: {ctrl_fmt}  |  Treatment: {trt_fmt}",
            f"  Absolute Lift: {lift_sign}{lift_abs_fmt}  |  Relative Lift: {lift_rel_pct}",
            f"  Confidence: {conf_pct:.1f}% probability treatment is better",
            f"  95% CI: [{ci_lo}, {ci_hi}]",
            "",
        ]

        # ── Secondary metrics ─────────────────────────────────────────────────
        if report.secondary_metric_results:
            lines.append("SECONDARY METRICS:")
            for m in report.secondary_metric_results:
                mname = m["metric_name"]
                lift = m.get("absolute_lift", 0.0)
                p = m.get("p_value")
                sig_label = "Significant" if m.get("is_significant") else "Not Significant"
                p_str = f"p={p:.3f}" if p is not None else "Bayesian"
                lift_fmt = self._format_metric_value(mname, abs(lift))
                lift_sign = "+" if lift >= 0 else "-"
                lines.append(
                    f"  {mname}: {lift_sign}{lift_fmt} lift ({p_str}) -- {sig_label}"
                )
            lines.append("")

        # ── Guardrails ────────────────────────────────────────────────────────
        if report.guardrail_results:
            lines.append("GUARDRAILS:")
            for g in report.guardrail_results:
                gname = g.get("metric_name", "unknown")
                gchange = g.get("relative_change", 0.0)
                gp = g.get("p_value")
                degraded = g.get("is_degraded", False)
                status = "DEGRADED" if degraded else "No degradation"
                p_str = f"p={gp:.3f}" if gp is not None else "n/a"
                change_sign = "+" if gchange >= 0 else ""
                lines.append(
                    f"  {gname}: {change_sign}{gchange * 100:.1f}% ({p_str}) -- {status}"
                )
            lines.append("")

        # ── Segment insights ──────────────────────────────────────────────────
        if report.segment_insights:
            lines.append("SEGMENT INSIGHTS:")
            for i, seg in enumerate(report.segment_insights):
                seg_name = seg.get("segment_name", "unknown")
                seg_lift = seg.get("relative_lift", 0.0)
                label = (
                    "Strongest lift"
                    if i == 0
                    else ("Weakest lift" if i == len(report.segment_insights) - 1 else "Notable segment")
                )
                lines.append(f"  {label}: {seg_name} ({seg_lift * 100:+.1f}% relative)")
            lines.append("")

        # ── Recommendation ────────────────────────────────────────────────────
        lines += [
            "RECOMMENDATION:",
            f"  {report.recommendation_text}",
            "",
            f"  Estimated impact: {report.estimated_annual_impact}",
            "",
        ]

        # ── Next steps ────────────────────────────────────────────────────────
        lines.append("NEXT STEPS:")
        for i, step in enumerate(report.next_steps, start=1):
            lines.append(f"  {i}. {step}")

        lines += ["", sep]
        return "\n".join(lines)

    # ── persistence ───────────────────────────────────────────────────────────

    def save_report(
        self, report: ExperimentReport, output_dir: str = "experiments"
    ) -> str:
        """
        Persist *report* as a JSON file.

        Parameters
        ----------
        report : ExperimentReport
        output_dir : str
            Directory path (relative or absolute) to write the file into.

        Returns
        -------
        str
            Absolute path to the saved JSON file.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        file_path = out_path / f"{report.experiment_id}_report.json"

        # Convert dataclass -> dict and handle tuple serialisation.
        data = asdict(report)
        # tuple[float, float] is not JSON-serialisable by default — convert to list.
        data["primary_metric_ci"] = list(data["primary_metric_ci"])

        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)

        return str(file_path.resolve())

    # ── portfolio summary ─────────────────────────────────────────────────────

    def generate_portfolio_summary(
        self, reports: list[ExperimentReport]
    ) -> pd.DataFrame:
        """
        Build a summary DataFrame across multiple experiments.

        Returns
        -------
        pd.DataFrame
            Columns: experiment_id, experiment_name, verdict, primary_metric,
            lift_relative_pct, is_primary_significant, blocked_guardrails_count
        """
        rows = []
        for r in reports:
            rows.append(
                {
                    "experiment_id": r.experiment_id,
                    "experiment_name": r.experiment_name,
                    "verdict": r.overall_verdict,
                    "primary_metric": r.primary_metric_name,
                    "lift_relative_pct": round(r.primary_metric_lift_relative * 100, 2),
                    "is_primary_significant": r.is_primary_significant,
                    "blocked_guardrails_count": len(r.blocked_guardrails),
                }
            )
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Demo / smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dataclasses import dataclass as _dc

    # ── Minimal stub dataclasses that mimic BayesianResult / FrequentistResult ─

    @_dc
    class _FakeFreqResult:
        experiment_id: str
        metric_name: str
        control_metric: float
        treatment_metric: float
        absolute_lift: float
        relative_lift: float
        confidence_interval_absolute: tuple
        confidence_interval_relative: tuple
        p_value: float
        is_significant: bool
        significance_level: float
        test_statistic: float
        test_type: str
        effect_size: float
        effect_size_type: str
        sample_size_control: int
        sample_size_treatment: int
        power_achieved: float
        n_bootstrap: int = 0

    @_dc
    class _FakeBayesResult:
        experiment_id: str
        metric_name: str
        control_metric: float
        treatment_metric: float
        probability_treatment_better: float
        expected_loss_treatment: float
        expected_loss_control: float
        posterior_mean_lift: float
        posterior_std_lift: float
        credible_interval_95: tuple
        rope: tuple
        probability_in_rope: float
        decision: str
        method: str
        sample_size_control: int
        sample_size_treatment: int
        n_samples: int
        bayes_factor: float

    engine = RecommendationEngine()

    print("\n" + "=" * 60)
    print("DEMO 1 — Frequentist SHIP_TREATMENT")
    print("=" * 60)
    freq_result = _FakeFreqResult(
        experiment_id="EXP-2024-001",
        metric_name="open_rate",
        control_metric=0.221,
        treatment_metric=0.248,
        absolute_lift=0.027,
        relative_lift=0.122,
        confidence_interval_absolute=(0.013, 0.041),
        confidence_interval_relative=(0.059, 0.186),
        p_value=0.001,
        is_significant=True,
        significance_level=0.05,
        test_statistic=3.21,
        test_type="two_proportion_z",
        effect_size=0.063,
        effect_size_type="cohens_h",
        sample_size_control=5_000,
        sample_size_treatment=5_000,
        power_achieved=0.91,
    )
    secondary = {
        "click_rate": {
            "absolute_lift": 0.008,
            "p_value": 0.031,
            "is_significant": True,
        },
        "unsubscribe_rate": {
            "absolute_lift": -0.001,
            "p_value": 0.612,
            "is_significant": False,
        },
    }
    guardrails = [
        {
            "metric_name": "unsubscribe_rate",
            "is_degraded": False,
            "relative_change": -0.045,
            "p_value": 0.61,
        }
    ]
    segments = [
        {"segment_name": "Enterprise (500+ employees)", "relative_lift": 0.21, "is_significant": True},
        {"segment_name": "Mid-Market (50-499 employees)", "relative_lift": 0.11, "is_significant": True},
        {"segment_name": "SMB (1-49 employees)", "relative_lift": 0.03, "is_significant": False},
    ]

    report1 = engine.generate_report(
        experiment_id="EXP-2024-001",
        experiment_name="Q4 Subject Line Personalisation",
        primary_metric="open_rate",
        primary_result=freq_result,
        secondary_results=secondary,
        guardrail_results=guardrails,
        segment_results=segments,
        test_duration_days=14,
    )
    print(engine.format_console_report(report1))

    print("\n" + "=" * 60)
    print("DEMO 2 — Bayesian EXTEND_TEST")
    print("=" * 60)
    bayes_result = _FakeBayesResult(
        experiment_id="EXP-2024-002",
        metric_name="click_rate",
        control_metric=0.045,
        treatment_metric=0.049,
        probability_treatment_better=0.86,
        expected_loss_treatment=0.0012,
        expected_loss_control=0.0038,
        posterior_mean_lift=0.004,
        posterior_std_lift=0.002,
        credible_interval_95=(-0.001, 0.009),
        rope=(-0.002, 0.002),
        probability_in_rope=0.18,
        decision="CONTINUE_TESTING",
        method="beta_binomial_analytical",
        sample_size_control=2_500,
        sample_size_treatment=2_500,
        n_samples=100_000,
        bayes_factor=4.2,
    )

    report2 = engine.generate_report(
        experiment_id="EXP-2024-002",
        experiment_name="CTA Button Colour Test",
        primary_metric="click_rate",
        primary_result=bayes_result,
        test_duration_days=21,
    )
    print(engine.format_console_report(report2))

    print("\n" + "=" * 60)
    print("DEMO 3 — Guardrail Block")
    print("=" * 60)
    freq_result_bad = _FakeFreqResult(
        experiment_id="EXP-2024-003",
        metric_name="conversion_rate",
        control_metric=0.032,
        treatment_metric=0.035,
        absolute_lift=0.003,
        relative_lift=0.094,
        confidence_interval_absolute=(-0.001, 0.007),
        confidence_interval_relative=(-0.031, 0.218),
        p_value=0.082,
        is_significant=False,
        significance_level=0.05,
        test_statistic=1.74,
        test_type="two_proportion_z",
        effect_size=0.017,
        effect_size_type="cohens_h",
        sample_size_control=4_000,
        sample_size_treatment=4_000,
        power_achieved=0.42,
    )
    bad_guardrails = [
        {
            "metric_name": "spam_complaint_rate",
            "is_degraded": True,
            "relative_change": 0.42,
            "p_value": 0.003,
        }
    ]

    report3 = engine.generate_report(
        experiment_id="EXP-2024-003",
        experiment_name="Aggressive Re-engagement Campaign",
        primary_metric="conversion_rate",
        primary_result=freq_result_bad,
        guardrail_results=bad_guardrails,
        test_duration_days=30,
    )
    print(engine.format_console_report(report3))

    print("\n" + "=" * 60)
    print("PORTFOLIO SUMMARY")
    print("=" * 60)
    summary_df = engine.generate_portfolio_summary([report1, report2, report3])
    print(summary_df.to_string(index=False))

    # Save report 1 to experiments/ directory.
    saved_path = engine.save_report(report1, output_dir="experiments")
    print(f"\nReport saved to: {saved_path}")
