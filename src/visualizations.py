"""
visualizations.py
=================
Matplotlib/Seaborn visualisation library for the Campaign Experimentation
& Lift Measurement Framework.

All plots are non-interactive (Agg backend) and are saved to disk.
Every public method returns the absolute path to the saved PNG.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import VISUALS_DIR, SIGNIFICANCE_LEVEL, CONFIDENCE_LEVEL

# ── global aesthetics ─────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="colorblind")

DEFAULT_FIGSIZE = (10, 6)
DEFAULT_DPI = 120


# ─────────────────────────────────────────────────────────────────────────────
# Visualiser
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentVisualizer:
    """
    Factory for all experiment diagnostic plots.

    Parameters
    ----------
    visuals_dir : str | Path
        Directory where PNG files are written.  Created if it does not exist.
    dpi : int
        Resolution of saved figures.
    figsize : tuple[int, int]
        Default (width, height) in inches.
    """

    def __init__(
        self,
        visuals_dir: str | Path = VISUALS_DIR,
        dpi: int = DEFAULT_DPI,
        figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    ) -> None:
        self.visuals_dir = Path(visuals_dir)
        self.visuals_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize
        # Consistent colour palette — index by variant order.
        self.colors = sns.color_palette("colorblind", 10)

    # ── internal helper ───────────────────────────────────────────────────────

    def _save_fig(self, fig: plt.Figure, filename: str) -> str:
        """
        Save *fig* to ``self.visuals_dir / filename``, close it, and return
        the full absolute path as a string.
        """
        out_path = self.visuals_dir / filename
        fig.savefig(out_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(out_path.resolve())

    # ── 1. Posterior distributions ────────────────────────────────────────────

    def plot_posterior_distributions(
        self,
        posterior_samples: dict[str, np.ndarray],
        metric_name: str,
        experiment_id: str,
        hdi_alpha: float = 0.05,
    ) -> str:
        """
        Overlapping KDE curves for each variant's posterior, plus a difference
        distribution subplot.

        Parameters
        ----------
        posterior_samples : dict[str, np.ndarray]
            Keys are variant names (e.g. "control", "treatment"); values are
            1-D arrays of posterior samples.
        metric_name : str
            Metric being plotted (used in labels and filename).
        experiment_id : str
            Experiment identifier used in the filename.
        hdi_alpha : float
            Alpha for the Highest Density Interval (default 0.05 → 95 % HDI).

        Returns
        -------
        str
            Path to the saved PNG.
        """
        variant_names = list(posterior_samples.keys())
        n_variants = len(variant_names)

        fig, axes = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.4, self.figsize[1]))

        # ── left panel: raw posteriors ────────────────────────────────────────
        ax_post = axes[0]
        for i, (variant, samples) in enumerate(posterior_samples.items()):
            colour = self.colors[i % len(self.colors)]
            kde = stats.gaussian_kde(samples)
            x_grid = np.linspace(samples.min() - samples.std(), samples.max() + samples.std(), 500)
            y_grid = kde(x_grid)

            ax_post.plot(x_grid, y_grid, color=colour, lw=2, label=variant)

            # 95 % HDI shading
            hdi_lo = np.percentile(samples, hdi_alpha / 2 * 100)
            hdi_hi = np.percentile(samples, (1 - hdi_alpha / 2) * 100)
            mask = (x_grid >= hdi_lo) & (x_grid <= hdi_hi)
            ax_post.fill_between(x_grid, 0, y_grid, where=mask, alpha=0.25, color=colour)

        ax_post.set_title(f"Posterior Distributions — {metric_name}", fontsize=11)
        ax_post.set_xlabel(metric_name.replace("_", " ").title())
        ax_post.set_ylabel("Density")
        ax_post.legend()

        # ── right panel: lift (treatment - control) ───────────────────────────
        ax_diff = axes[1]

        if n_variants >= 2:
            base_samples = posterior_samples[variant_names[0]]
            for i in range(1, n_variants):
                comp_samples = posterior_samples[variant_names[i]]
                min_len = min(len(base_samples), len(comp_samples))
                diff = comp_samples[:min_len] - base_samples[:min_len]

                colour = self.colors[i % len(self.colors)]
                kde_d = stats.gaussian_kde(diff)
                x_d = np.linspace(diff.min() - diff.std(), diff.max() + diff.std(), 500)
                y_d = kde_d(x_d)

                ax_diff.plot(x_d, y_d, color=colour, lw=2,
                             label=f"{variant_names[i]} − {variant_names[0]}")

                # HDI shading
                hdi_lo = np.percentile(diff, hdi_alpha / 2 * 100)
                hdi_hi = np.percentile(diff, (1 - hdi_alpha / 2) * 100)
                mask = (x_d >= hdi_lo) & (x_d <= hdi_hi)
                ax_diff.fill_between(x_d, 0, y_d, where=mask, alpha=0.25, color=colour)

        ax_diff.axvline(0, color="black", lw=1.2, linestyle="--", label="No lift (0)")
        ax_diff.set_title("Lift Distribution (Treatment − Control)", fontsize=11)
        ax_diff.set_xlabel(f"Δ {metric_name.replace('_', ' ').title()}")
        ax_diff.set_ylabel("Density")
        ax_diff.legend()

        fig.suptitle(f"Posterior Distributions — {metric_name}", fontsize=13, y=1.02)
        fig.tight_layout()

        filename = f"{experiment_id}_posterior_{metric_name}.png"
        return self._save_fig(fig, filename)

    # ── 2. Forest plot — lift with CI ─────────────────────────────────────────

    def plot_lift_with_ci(
        self,
        results: list[dict],
        metric_names: list[str],
        experiment_id: str,
    ) -> str:
        """
        Horizontal forest plot: each metric on y-axis with lift estimate and CI.

        Each element in *results* must have keys:
        ``metric_name``, ``lift``, ``ci_lower``, ``ci_upper``, ``is_significant``,
        ``direction`` ("positive" | "negative" | "neutral").

        Returns
        -------
        str
            Path to the saved PNG.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        y_positions = list(range(len(metric_names)))
        green = self.colors[2]   # colourblind-safe green
        red = self.colors[3]     # colourblind-safe red
        grey = (0.6, 0.6, 0.6)

        for i, (metric, result) in enumerate(zip(metric_names, results)):
            lift = result.get("lift", 0.0)
            ci_lo = result.get("ci_lower", 0.0)
            ci_hi = result.get("ci_upper", 0.0)
            significant = result.get("is_significant", False)
            direction = result.get("direction", "positive" if lift > 0 else "negative")

            if significant and direction == "positive":
                colour = green
            elif significant and direction == "negative":
                colour = red
            else:
                colour = grey

            xerr_lo = lift - ci_lo
            xerr_hi = ci_hi - lift

            ax.errorbar(
                lift,
                i,
                xerr=[[max(0, xerr_lo)], [max(0, xerr_hi)]],
                fmt="o",
                color=colour,
                ecolor=colour,
                elinewidth=2,
                capsize=5,
                markersize=8,
                zorder=3,
            )

        ax.axvline(0, color="black", lw=1.2, linestyle="--", zorder=2)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([m.replace("_", " ").title() for m in metric_names])
        ax.set_xlabel("Lift (absolute)")
        ax.set_title(f"Metric Lifts with 95% CI — {experiment_id}", fontsize=12)

        # Legend
        handles = [
            mpatches.Patch(color=green, label="Significant positive"),
            mpatches.Patch(color=red, label="Significant negative"),
            mpatches.Patch(color=grey, label="Not significant"),
        ]
        ax.legend(handles=handles, loc="lower right", fontsize=9)
        fig.tight_layout()

        filename = f"{experiment_id}_lift_forest.png"
        return self._save_fig(fig, filename)

    # ── 3. Power curve ────────────────────────────────────────────────────────

    def plot_power_curve(
        self,
        effect_sizes: np.ndarray,
        powers: np.ndarray,
        alpha: float,
        mde: float | None = None,
        current_n: int | None = None,
    ) -> str:
        """
        Plot statistical power as a function of effect size.

        Parameters
        ----------
        effect_sizes : np.ndarray
            Array of effect sizes (x-axis).
        powers : np.ndarray
            Corresponding power values (y-axis).
        alpha : float
            Significance level used (annotated on the plot).
        mde : float, optional
            Minimum detectable effect — shown as a vertical dashed line.
        current_n : int, optional
            Current sample size (annotated in the title if provided).

        Returns
        -------
        str
            Path to the saved PNG.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(effect_sizes, powers, color=self.colors[0], lw=2.5, label="Power curve")
        ax.axhline(0.80, color=self.colors[3], lw=1.5, linestyle="--", label="80% power target")
        ax.fill_between(effect_sizes, 0, powers, alpha=0.12, color=self.colors[0])

        if mde is not None:
            ax.axvline(mde, color=self.colors[2], lw=1.5, linestyle="--",
                       label=f"MDE = {mde:.4f}")

        if current_n is not None:
            ax.set_title(f"Power Curve  (α = {alpha}, n = {current_n:,} per arm)", fontsize=12)
        else:
            ax.set_title(f"Power Curve  (α = {alpha})", fontsize=12)

        ax.set_xlabel("Effect Size")
        ax.set_ylabel("Statistical Power")
        ax.set_ylim(0, 1.05)
        ax.legend()
        fig.tight_layout()

        return self._save_fig(fig, "power_curve.png")

    # ── 4. Sequential monitoring ──────────────────────────────────────────────

    def plot_sequential_monitoring(
        self,
        z_stats: list[float],
        boundaries: list,
        cumulative_ns: list[int],
        experiment_id: str,
    ) -> str:
        """
        Plot sequential z-statistics against O'Brien-Fleming stopping boundaries.

        Parameters
        ----------
        z_stats : list[float]
            Observed z-statistics at each interim look.
        boundaries : list of dict
            Each element should have keys:
            ``efficacy_upper``, ``efficacy_lower``, ``futility_upper``,
            ``futility_lower``.
        cumulative_ns : list[int]
            Total sample sizes at each look (x-axis).
        experiment_id : str

        Returns
        -------
        str
            Path to the saved PNG.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        x = cumulative_ns
        eff_upper = [b["efficacy_upper"] for b in boundaries]
        eff_lower = [b["efficacy_lower"] for b in boundaries]
        fut_upper = [b.get("futility_upper", b["efficacy_upper"] * 0.5) for b in boundaries]
        fut_lower = [b.get("futility_lower", b["efficacy_lower"] * 0.5) for b in boundaries]

        # Shaded stopping regions
        ax.fill_between(x, eff_upper, max(eff_upper) * 2,
                        alpha=0.10, color=self.colors[2], label="Stop for efficacy region")
        ax.fill_between(x, min(eff_lower) * 2, eff_lower,
                        alpha=0.10, color=self.colors[2])
        ax.fill_between(x, fut_lower, fut_upper,
                        alpha=0.10, color=self.colors[3], label="Stop for futility region")

        # Boundary lines
        ax.plot(x, eff_upper, color=self.colors[3], lw=1.8, linestyle="--",
                label="Efficacy boundary (O'Brien-Fleming)")
        ax.plot(x, eff_lower, color=self.colors[3], lw=1.8, linestyle="--")
        ax.plot(x, fut_upper, color=self.colors[4], lw=1.5, linestyle="--",
                label="Futility boundary")
        ax.plot(x, fut_lower, color=self.colors[4], lw=1.5, linestyle="--")

        # Observed z-statistics
        ax.plot(x, z_stats, color=self.colors[0], lw=2.5, marker="o",
                markersize=7, label="Observed z-statistic", zorder=5)
        ax.axhline(0, color="black", lw=1.0, linestyle="-")

        ax.set_xlabel("Cumulative Sample Size")
        ax.set_ylabel("Z-Statistic")
        ax.set_title(f"Sequential Monitoring — {experiment_id}", fontsize=12)
        ax.legend(fontsize=8)
        fig.tight_layout()

        filename = f"{experiment_id}_sequential.png"
        return self._save_fig(fig, filename)

    # ── 5. Holdout comparison ─────────────────────────────────────────────────

    def plot_holdout_comparison(
        self,
        period_means: pd.DataFrame,
        experiment_id: str,
        metric_name: str,
    ) -> str:
        """
        Grouped bar chart comparing exposed vs holdout across baseline/test periods.

        Parameters
        ----------
        period_means : pd.DataFrame
            Must have columns: ``period`` ("baseline" | "test"),
            ``variant`` ("exposed" | "holdout"), ``mean``, ``ci_lower``, ``ci_upper``.
        experiment_id : str
        metric_name : str

        Returns
        -------
        str
            Path to the saved PNG.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        periods = period_means["period"].unique()
        variants = period_means["variant"].unique()
        n_periods = len(periods)
        n_variants = len(variants)
        bar_width = 0.35
        x = np.arange(n_periods)

        for j, variant in enumerate(variants):
            sub = period_means[period_means["variant"] == variant]
            means = [
                sub[sub["period"] == p]["mean"].values[0] if len(sub[sub["period"] == p]) else 0
                for p in periods
            ]
            ci_lo = [
                sub[sub["period"] == p]["ci_lower"].values[0] if len(sub[sub["period"] == p]) else 0
                for p in periods
            ]
            ci_hi = [
                sub[sub["period"] == p]["ci_upper"].values[0] if len(sub[sub["period"] == p]) else 0
                for p in periods
            ]
            yerr_lo = [m - lo for m, lo in zip(means, ci_lo)]
            yerr_hi = [hi - m for m, hi in zip(means, ci_hi)]

            offset = (j - (n_variants - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset, means, bar_width,
                label=variant,
                color=self.colors[j % len(self.colors)],
                alpha=0.85,
                yerr=[yerr_lo, yerr_hi],
                capsize=4,
                error_kw={"elinewidth": 1.5},
            )

        # Annotate incremental lift (test period — baseline period) for the
        # exposed variant if both periods are present.
        try:
            exposed_data = period_means[period_means["variant"] == "exposed"]
            baseline_val = exposed_data[exposed_data["period"] == "baseline"]["mean"].values[0]
            test_val = exposed_data[exposed_data["period"] == "test"]["mean"].values[0]
            incr = test_val - baseline_val
            ax.annotate(
                f"Incremental lift:\n{incr:+.4f}",
                xy=(x[-1], test_val),
                xytext=(x[-1] + 0.5, test_val),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"),
                color="black",
            )
        except (IndexError, KeyError):
            pass

        ax.set_xticks(x)
        ax.set_xticklabels([p.title() for p in periods])
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_title(f"Holdout Comparison — {metric_name} ({experiment_id})", fontsize=12)
        ax.legend()
        fig.tight_layout()

        filename = f"{experiment_id}_holdout_{metric_name}.png"
        return self._save_fig(fig, filename)

    # ── 6. Probability of being best ─────────────────────────────────────────

    def plot_probability_best(
        self,
        prob_best: dict[str, float],
        experiment_id: str,
        metric_name: str,
    ) -> str:
        """
        Horizontal bar chart of P(being best) for each variant.

        Parameters
        ----------
        prob_best : dict[str, float]
            Mapping of variant name -> probability of being the best variant.
        experiment_id : str
        metric_name : str

        Returns
        -------
        str
            Path to the saved PNG.
        """
        fig, ax = plt.subplots(figsize=(self.figsize[0], max(3, len(prob_best) * 1.2)))

        variants = list(prob_best.keys())
        probs = [prob_best[v] for v in variants]
        n = len(variants)
        equal_prob = 1.0 / n if n > 0 else 0.5

        # Blue gradient — higher probability => darker bar.
        blues = plt.cm.Blues(np.linspace(0.35, 0.85, n))
        sorted_idx = np.argsort(probs)

        for rank, idx in enumerate(sorted_idx):
            ax.barh(variants[idx], probs[idx], color=blues[rank], edgecolor="white")
            ax.text(
                probs[idx] + 0.005,
                idx if isinstance(idx, int) else variants.index(variants[idx]),
                f"{probs[idx]:.1%}",
                va="center",
                fontsize=9,
            )

        ax.axvline(
            equal_prob, color=self.colors[3], lw=1.5, linestyle="--",
            label=f"Equal probability ({equal_prob:.1%})"
        )
        ax.set_xlim(0, 1.10)
        ax.set_xlabel("P(being best)")
        ax.set_title(
            f"Probability of Being Best — {metric_name} ({experiment_id})", fontsize=12
        )
        ax.legend(fontsize=9)
        fig.tight_layout()

        filename = f"{experiment_id}_prob_best_{metric_name}.png"
        return self._save_fig(fig, filename)

    # ── 7. Segment lift ───────────────────────────────────────────────────────

    def plot_segment_lift(
        self,
        segment_df: pd.DataFrame,
        segment_col: str,
        metric_name: str,
        experiment_id: str,
        overall_lift: float | None = None,
    ) -> str:
        """
        Horizontal bar chart of per-segment lift, sorted descending.

        Parameters
        ----------
        segment_df : pd.DataFrame
            Must contain columns: the *segment_col* column, ``lift``,
            ``ci_lower``, ``ci_upper``, ``recommendation``
            ("increase" | "maintain" | "reduce").
        segment_col : str
            Column name for segment labels.
        metric_name : str
        experiment_id : str
        overall_lift : float, optional
            If provided, drawn as a vertical reference line.

        Returns
        -------
        str
            Path to the saved PNG.
        """
        df = segment_df.sort_values("lift", ascending=False).reset_index(drop=True)

        fig, ax = plt.subplots(
            figsize=(self.figsize[0], max(4, len(df) * 0.55))
        )

        colour_map = {
            "increase": self.colors[2],   # green
            "maintain": self.colors[0],   # blue
            "reduce":   self.colors[3],   # red
        }
        default_colour = self.colors[1]

        for i, row in df.iterrows():
            lift = row["lift"]
            ci_lo = row.get("ci_lower", lift)
            ci_hi = row.get("ci_upper", lift)
            rec = str(row.get("recommendation", "maintain")).lower()
            colour = colour_map.get(rec, default_colour)

            ax.barh(
                row[segment_col],
                lift,
                xerr=[[lift - ci_lo], [ci_hi - lift]],
                color=colour,
                alpha=0.8,
                capsize=4,
                error_kw={"elinewidth": 1.5},
            )

        ax.axvline(0, color="black", lw=1.2, linestyle="-")

        if overall_lift is not None:
            ax.axvline(
                overall_lift, color="gray", lw=1.5, linestyle="--",
                label=f"Overall lift = {overall_lift:+.4f}"
            )
            ax.legend(fontsize=9)

        # Custom legend for recommendations
        handles = [
            mpatches.Patch(color=colour_map["increase"], label="Increase targeting"),
            mpatches.Patch(color=colour_map["maintain"], label="Maintain"),
            mpatches.Patch(color=colour_map["reduce"], label="Reduce / Exclude"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="lower right")

        ax.set_xlabel(f"Lift in {metric_name.replace('_', ' ').title()}")
        ax.set_title(
            f"Segment Lift by {segment_col.replace('_', ' ').title()} — {experiment_id}",
            fontsize=12,
        )
        fig.tight_layout()

        filename = f"{experiment_id}_segments_{segment_col}.png"
        return self._save_fig(fig, filename)

    # ── 8. Multiple-comparison correction comparison ──────────────────────────

    def plot_correction_comparison(
        self,
        correction_df: pd.DataFrame,
        experiment_id: str,
        metric_name: str,
    ) -> str:
        """
        Grouped bar chart showing p-values (or adjusted p-values) under multiple
        correction methods; significant cells are annotated with a star.

        Parameters
        ----------
        correction_df : pd.DataFrame
            Rows = metrics; columns = correction method names
            (e.g. "none", "bonferroni", "holm", "fdr_bh").
            Values are p-values (floats).
        experiment_id : str
        metric_name : str

        Returns
        -------
        str
            Path to the saved PNG.
        """
        corrections = [c for c in correction_df.columns if c != "metric"]
        metrics_list = (
            correction_df["metric"].tolist()
            if "metric" in correction_df.columns
            else correction_df.index.tolist()
        )
        n_corrections = len(corrections)
        n_metrics = len(metrics_list)

        fig, ax = plt.subplots(figsize=(self.figsize[0], max(4, n_metrics * 1.0)))

        bar_width = 0.8 / n_corrections
        x = np.arange(n_metrics)

        for j, correction in enumerate(corrections):
            pvals = (
                correction_df[correction].values
                if correction in correction_df.columns
                else np.ones(n_metrics)
            )
            offset = (j - (n_corrections - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset,
                pvals,
                bar_width,
                label=correction,
                color=self.colors[j % len(self.colors)],
                alpha=0.8,
            )
            # Annotate significant p-values with a star
            for bar, pval in zip(bars, pvals):
                if pval < SIGNIFICANCE_LEVEL:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        min(pval + 0.01, 0.99),
                        "*",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        fontweight="bold",
                        color="black",
                    )

        ax.axhline(SIGNIFICANCE_LEVEL, color=self.colors[3], lw=1.5,
                   linestyle="--", label=f"α = {SIGNIFICANCE_LEVEL}")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace("_", " ").title() for m in metrics_list], rotation=30, ha="right"
        )
        ax.set_ylabel("Adjusted p-value")
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"Multiple Comparison Corrections — {metric_name} ({experiment_id})",
            fontsize=12,
        )
        ax.legend(fontsize=9)
        fig.tight_layout()

        filename = f"{experiment_id}_corrections_{metric_name}.png"
        return self._save_fig(fig, filename)

    # ── 9. Cumulative conversions over time ───────────────────────────────────

    def plot_cumulative_conversions(
        self,
        time_series_data: pd.DataFrame,
        variant_col: str,
        date_col: str,
        metric_col: str,
        experiment_id: str,
    ) -> str:
        """
        Line plot of the cumulative metric per variant over time with shaded CI bands.

        Parameters
        ----------
        time_series_data : pd.DataFrame
            Must contain columns for *variant_col*, *date_col*, *metric_col*,
            and optionally ``ci_lower`` / ``ci_upper`` for confidence bands.
        variant_col : str
            Column identifying the variant (e.g. "variant").
        date_col : str
            Date/datetime column (x-axis).
        metric_col : str
            Column with the cumulative metric value (y-axis).
        experiment_id : str

        Returns
        -------
        str
            Path to the saved PNG.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        variants = time_series_data[variant_col].unique()
        for i, variant in enumerate(sorted(variants)):
            sub = time_series_data[time_series_data[variant_col] == variant].sort_values(date_col)
            colour = self.colors[i % len(self.colors)]

            ax.plot(
                sub[date_col],
                sub[metric_col],
                color=colour,
                lw=2.5,
                label=str(variant),
                marker="o",
                markersize=4,
            )

            if "ci_lower" in sub.columns and "ci_upper" in sub.columns:
                ax.fill_between(
                    sub[date_col],
                    sub["ci_lower"],
                    sub["ci_upper"],
                    alpha=0.15,
                    color=colour,
                )

        ax.set_xlabel("Date")
        ax.set_ylabel(f"Cumulative {metric_col.replace('_', ' ').title()}")
        ax.set_title(
            f"Cumulative {metric_col.replace('_', ' ').title()} Over Time — {experiment_id}",
            fontsize=12,
        )
        ax.legend()
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()

        filename = f"{experiment_id}_cumulative_{metric_col}.png"
        return self._save_fig(fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Demo / smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    viz = ExperimentVisualizer(visuals_dir=VISUALS_DIR)

    print("\n" + "=" * 60)
    print("VISUALIZATIONS DEMO")
    print("=" * 60)

    # ── 1. Posterior distributions ────────────────────────────────────────────
    print("\n[1/8] Plotting posterior distributions...")
    n_samples = 50_000
    control_samples = rng.beta(220, 780, n_samples)        # ~22% open rate
    treatment_samples = rng.beta(248, 752, n_samples)      # ~24.8% open rate
    path = viz.plot_posterior_distributions(
        posterior_samples={"control": control_samples, "treatment": treatment_samples},
        metric_name="open_rate",
        experiment_id="EXP-2024-001",
    )
    print(f"   Saved: {path}")

    # ── 2. Forest plot ────────────────────────────────────────────────────────
    print("\n[2/8] Plotting lift forest plot...")
    forest_results = [
        {"lift": 0.027, "ci_lower": 0.013, "ci_upper": 0.041,
         "is_significant": True, "direction": "positive"},
        {"lift": 0.008, "ci_lower": 0.001, "ci_upper": 0.015,
         "is_significant": True, "direction": "positive"},
        {"lift": -0.001, "ci_lower": -0.005, "ci_upper": 0.003,
         "is_significant": False, "direction": "negative"},
        {"lift": 0.002, "ci_lower": -0.003, "ci_upper": 0.007,
         "is_significant": False, "direction": "positive"},
    ]
    path = viz.plot_lift_with_ci(
        results=forest_results,
        metric_names=["open_rate", "click_rate", "unsubscribe_rate", "conversion_rate"],
        experiment_id="EXP-2024-001",
    )
    print(f"   Saved: {path}")

    # ── 3. Power curve ────────────────────────────────────────────────────────
    print("\n[3/8] Plotting power curve...")
    effect_sizes = np.linspace(0.001, 0.05, 100)
    from statsmodels.stats.power import NormalIndPower
    power_analysis = NormalIndPower()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        powers = np.array([
            power_analysis.solve_power(
                effect_size=e, nobs1=5000, alpha=0.05, alternative="two-sided"
            )
            for e in effect_sizes
        ])
    path = viz.plot_power_curve(
        effect_sizes=effect_sizes,
        powers=powers,
        alpha=0.05,
        mde=0.027,
        current_n=5000,
    )
    print(f"   Saved: {path}")

    # ── 4. Sequential monitoring ──────────────────────────────────────────────
    print("\n[4/8] Plotting sequential monitoring...")
    looks = 5
    cum_ns = [1000, 2000, 3500, 5000, 7000]
    # Simulate gradually rising z-stat
    z_stats = [1.2, 1.8, 2.1, 2.6, 3.1]
    # O'Brien-Fleming boundaries (approximate)
    info_fractions = [n / cum_ns[-1] for n in cum_ns]
    obf_boundaries = [
        {
            "efficacy_upper": stats.norm.ppf(1 - 0.025) / np.sqrt(t),
            "efficacy_lower": -stats.norm.ppf(1 - 0.025) / np.sqrt(t),
            "futility_upper": stats.norm.ppf(1 - 0.025) / np.sqrt(t) * 0.5,
            "futility_lower": -stats.norm.ppf(1 - 0.025) / np.sqrt(t) * 0.5,
        }
        for t in info_fractions
    ]
    path = viz.plot_sequential_monitoring(
        z_stats=z_stats,
        boundaries=obf_boundaries,
        cumulative_ns=cum_ns,
        experiment_id="EXP-2024-001",
    )
    print(f"   Saved: {path}")

    # ── 5. Holdout comparison ─────────────────────────────────────────────────
    print("\n[5/8] Plotting holdout comparison...")
    holdout_df = pd.DataFrame(
        [
            {"period": "baseline", "variant": "exposed", "mean": 0.038, "ci_lower": 0.034, "ci_upper": 0.042},
            {"period": "baseline", "variant": "holdout", "mean": 0.037, "ci_lower": 0.033, "ci_upper": 0.041},
            {"period": "test",     "variant": "exposed", "mean": 0.052, "ci_lower": 0.047, "ci_upper": 0.057},
            {"period": "test",     "variant": "holdout", "mean": 0.038, "ci_lower": 0.034, "ci_upper": 0.042},
        ]
    )
    path = viz.plot_holdout_comparison(
        period_means=holdout_df,
        experiment_id="EXP-2024-001",
        metric_name="conversion_rate",
    )
    print(f"   Saved: {path}")

    # ── 6. Probability of being best ──────────────────────────────────────────
    print("\n[6/8] Plotting probability of being best...")
    path = viz.plot_probability_best(
        prob_best={"control": 0.072, "treatment_A": 0.612, "treatment_B": 0.316},
        experiment_id="EXP-2024-001",
        metric_name="open_rate",
    )
    print(f"   Saved: {path}")

    # ── 7. Segment lift ───────────────────────────────────────────────────────
    print("\n[7/8] Plotting segment lift...")
    seg_df = pd.DataFrame(
        [
            {"industry": "Technology",     "lift": 0.041, "ci_lower": 0.028, "ci_upper": 0.054, "recommendation": "increase"},
            {"industry": "Financial Svcs", "lift": 0.029, "ci_lower": 0.015, "ci_upper": 0.043, "recommendation": "increase"},
            {"industry": "Healthcare",     "lift": 0.018, "ci_lower": 0.004, "ci_upper": 0.032, "recommendation": "maintain"},
            {"industry": "Manufacturing",  "lift": 0.005, "ci_lower": -0.009, "ci_upper": 0.019, "recommendation": "maintain"},
            {"industry": "Retail",         "lift": -0.012, "ci_lower": -0.027, "ci_upper": 0.003, "recommendation": "reduce"},
            {"industry": "Education",      "lift": -0.021, "ci_lower": -0.038, "ci_upper": -0.004, "recommendation": "reduce"},
        ]
    )
    path = viz.plot_segment_lift(
        segment_df=seg_df,
        segment_col="industry",
        metric_name="open_rate",
        experiment_id="EXP-2024-001",
        overall_lift=0.027,
    )
    print(f"   Saved: {path}")

    # ── 8. Multiple-comparison corrections ───────────────────────────────────
    print("\n[8/8] Plotting correction comparison...")
    corr_df = pd.DataFrame(
        {
            "metric":      ["click_rate", "conversion_rate", "form_submit_rate", "mql_conversion_rate"],
            "none":        [0.003, 0.041, 0.089, 0.210],
            "bonferroni":  [0.012, 0.164, 0.356, 0.840],
            "holm":        [0.009, 0.123, 0.267, 0.630],
            "fdr_bh":      [0.006, 0.082, 0.178, 0.420],
        }
    )
    path = viz.plot_correction_comparison(
        correction_df=corr_df,
        experiment_id="EXP-2024-001",
        metric_name="open_rate",
    )
    print(f"   Saved: {path}")

    # ── 9. Cumulative conversions ─────────────────────────────────────────────
    print("\n[9/9] Plotting cumulative conversions over time...")
    dates = pd.date_range("2024-10-01", periods=21, freq="D")
    ts_rows = []
    ctrl_cum = 0.0
    trt_cum = 0.0
    for d in dates:
        ctrl_daily = rng.binomial(500, 0.038) / 500
        trt_daily = rng.binomial(500, 0.052) / 500
        ctrl_cum += ctrl_daily
        trt_cum += trt_daily
        ts_rows.append({"date": d, "variant": "control",   "cumulative_conversions": ctrl_cum,
                        "ci_lower": ctrl_cum * 0.94, "ci_upper": ctrl_cum * 1.06})
        ts_rows.append({"date": d, "variant": "treatment", "cumulative_conversions": trt_cum,
                        "ci_lower": trt_cum * 0.94, "ci_upper": trt_cum * 1.06})
    ts_df = pd.DataFrame(ts_rows)

    path = viz.plot_cumulative_conversions(
        time_series_data=ts_df,
        variant_col="variant",
        date_col="date",
        metric_col="cumulative_conversions",
        experiment_id="EXP-2024-001",
    )
    print(f"   Saved: {path}")

    print("\n" + "=" * 60)
    print(f"All plots saved to: {viz.visuals_dir.resolve()}")
    print("=" * 60)
