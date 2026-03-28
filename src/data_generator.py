"""
Synthetic marketing experiment data generator for the Campaign Experimentation
& Lift Measurement Framework.

Generates five reproducible experiment datasets covering email subject-line
A/B tests, landing-page multivariate tests, holdout/incrementality panels,
channel-mix comparisons, and send-time optimisation.  All datasets are saved
as CSV files under the project ``data/`` directory together with a JSON
metadata manifest.

Usage
-----
    python src/data_generator.py

Or programmatically::

    from src.data_generator import ExperimentDataGenerator
    gen = ExperimentDataGenerator(seed=42)
    data = gen.generate_all()
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats  # noqa: F401  (available for callers / notebooks)

# ---------------------------------------------------------------------------
# Optional: pull constants from the project config when importable
# ---------------------------------------------------------------------------
try:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from config import RANDOM_SEED, DATA_DIR  # noqa: F401
except ImportError:
    RANDOM_SEED: int = 42
    DATA_DIR: str = "data"


# ---------------------------------------------------------------------------
# Stratification constants
# ---------------------------------------------------------------------------
_INDUSTRIES = ["Technology", "Finance", "Healthcare", "Retail", "Manufacturing"]
_INDUSTRY_WEIGHTS = [0.30, 0.22, 0.20, 0.18, 0.10]

_COMPANY_SIZES = ["SMB", "Mid-Market", "Enterprise"]
_COMPANY_SIZE_WEIGHTS = [0.40, 0.35, 0.25]

_REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
_REGION_WEIGHTS = [0.50, 0.25, 0.15, 0.10]

_ENGAGEMENT_TIERS = ["High", "Medium", "Low"]
_ENGAGEMENT_TIER_WEIGHTS = [0.20, 0.50, 0.30]


class ExperimentDataGenerator:
    """Generate synthetic marketing experiment datasets for five experiments.

    Each experiment is designed to reflect realistic data-quality issues,
    heterogeneous treatment effects, and guardrail metrics so that downstream
    statistical analysis methods have a well-characterised ground truth to
    validate against.

    Parameters
    ----------
    seed:
        Master random seed.  All internal RNG instances are derived from this
        value so the full dataset is reproducible from a single integer.
    data_dir:
        Path to the output directory for CSV and metadata files.  Relative
        paths are resolved relative to the *project root* (the parent of the
        ``src/`` package directory).
    """

    def __init__(self, seed: int = RANDOM_SEED, data_dir: str = DATA_DIR) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Resolve data_dir relative to the project root when it is not absolute
        src_dir = Path(__file__).resolve().parent
        project_root = src_dir.parent
        self.data_dir = (
            Path(data_dir)
            if Path(data_dir).is_absolute()
            else project_root / data_dir
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """Generate all five experiment datasets, persist them, and return them.

        Saves each experiment as a CSV under ``self.data_dir`` and writes a
        JSON metadata manifest as ``experiments_metadata.json``.

        Returns
        -------
        dict
            Keys are experiment IDs (e.g. ``"exp001_email_subject_line"``),
            values are the corresponding :class:`pandas.DataFrame` objects.
        """
        generators = [
            ("exp001_email_subject_line", self.generate_exp001_email_subject_line),
            ("exp002_landing_page_mvt", self.generate_exp002_landing_page_mvt),
            ("exp003_holdout", self.generate_exp003_holdout),
            ("exp004_channel_mix", self.generate_exp004_channel_mix),
            ("exp005_send_time", self.generate_exp005_send_time),
        ]

        results: Dict[str, pd.DataFrame] = {}
        metadata: Dict[str, dict] = {}

        for exp_id, func in generators:
            df = func()
            results[exp_id] = df

            # Persist CSV
            csv_path = self.data_dir / f"{exp_id}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Saved {csv_path.name}  ({len(df):,} rows x {df.shape[1]} cols)")

            # Collect metadata
            metadata[exp_id] = self._build_metadata(exp_id, df)

        # Persist metadata manifest
        meta_path = self.data_dir / "experiments_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, default=str)
        print(f"  Saved {meta_path.name}")

        return results

    # ------------------------------------------------------------------
    # Experiment 1 — Email Subject-Line A/B Test
    # ------------------------------------------------------------------

    def generate_exp001_email_subject_line(self) -> pd.DataFrame:
        """Generate a 50 000-contact email subject-line A/B experiment.

        Design
        ------
        * 50/50 control vs treatment split.
        * Binary outcomes drawn from per-contact Beta-Bernoulli model so that
          individual-level heterogeneity is realistic.
        * Guardrail metrics (unsubscribe, spam complaint) are stable across
          arms.
        * ~1.5 % duplicate contact_ids appended to simulate upstream dedup
          failures.
        * Stratification columns added for sub-group / HTE analysis.

        Returns
        -------
        pandas.DataFrame
        """
        rng = np.random.default_rng(self.seed + 1)
        n_total = 50_000
        n_per_variant = n_total // 2

        records = []
        for variant_idx, (variant, params) in enumerate(
            [
                (
                    "control",
                    dict(
                        open_rate=0.22,
                        click_rate=0.035,
                        conversion_rate=0.008,
                        unsubscribe_rate=0.005,
                        spam_complaint_rate=0.001,
                    ),
                ),
                (
                    "treatment",
                    dict(
                        open_rate=0.26,
                        click_rate=0.048,
                        conversion_rate=0.011,
                        unsubscribe_rate=0.005,
                        spam_complaint_rate=0.001,
                    ),
                ),
            ]
        ):
            # Per-contact rates drawn from Beta distribution (concentration=50)
            concentration = 50
            opens = rng.beta(
                params["open_rate"] * concentration,
                (1 - params["open_rate"]) * concentration,
                n_per_variant,
            )
            clicks = rng.beta(
                params["click_rate"] * concentration,
                (1 - params["click_rate"]) * concentration,
                n_per_variant,
            )
            conversions = rng.beta(
                params["conversion_rate"] * concentration,
                (1 - params["conversion_rate"]) * concentration,
                n_per_variant,
            )
            unsubs = rng.beta(
                params["unsubscribe_rate"] * concentration,
                (1 - params["unsubscribe_rate"]) * concentration,
                n_per_variant,
            )
            spams = rng.beta(
                params["spam_complaint_rate"] * concentration,
                (1 - params["spam_complaint_rate"]) * concentration,
                n_per_variant,
            )

            # Bernoulli draw using the per-contact rate as probability
            open_outcome = rng.binomial(1, opens)
            click_outcome = rng.binomial(1, clicks)
            conv_outcome = rng.binomial(1, conversions)
            unsub_outcome = rng.binomial(1, unsubs)
            spam_outcome = rng.binomial(1, spams)

            # Assignment dates uniformly within 30-day window
            start_date = date(2025, 9, 15)
            day_offsets = rng.integers(0, 30, n_per_variant)
            assignment_dates = [
                (start_date + timedelta(days=int(d))).isoformat()
                for d in day_offsets
            ]
            day_of_week = [
                (start_date + timedelta(days=int(d))).weekday()
                for d in day_offsets
            ]

            contact_ids = self._generate_ids(
                "C", n_per_variant, rng, variant_idx * n_per_variant
            )

            chunk = pd.DataFrame(
                {
                    "contact_id": contact_ids,
                    "variant": variant,
                    "open_rate": open_outcome,
                    "click_rate": click_outcome,
                    "conversion_rate": conv_outcome,
                    "unsubscribe_rate": unsub_outcome,
                    "spam_complaint_rate": spam_outcome,
                    "day_of_week": day_of_week,
                    "assignment_date": assignment_dates,
                }
            )
            records.append(chunk)

        df = pd.concat(records, ignore_index=True)
        df = self.add_stratification_columns(df, len(df), seed_offset=1)
        df = self.introduce_data_quality_issues(
            df, id_col="contact_id", dq_rate=0.015, seed_offset=101
        )
        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Experiment 2 — Landing Page Multivariate Test
    # ------------------------------------------------------------------

    def generate_exp002_landing_page_mvt(self) -> pd.DataFrame:
        """Generate an 80 000-contact four-variant landing-page MVT.

        Design
        ------
        * Four variants: control, variant_b (no significant lift), variant_c
          (clear winner on conversions), variant_d (mid-performance).
        * ``time_on_page`` is continuous (Normal) rather than binary.
        * Guardrail: no unsubscribe column — bounce_rate serves that role.

        Returns
        -------
        pandas.DataFrame
        """
        rng = np.random.default_rng(self.seed + 2)
        n_per_variant = 20_000

        variant_specs = {
            "control": dict(
                bounce_rate=0.45,
                form_submit_rate=0.08,
                demo_request_rate=0.02,
                time_on_page_mu=180,
                time_on_page_sigma=60,
            ),
            "variant_b": dict(
                bounce_rate=0.43,
                form_submit_rate=0.083,
                demo_request_rate=0.021,
                time_on_page_mu=180,
                time_on_page_sigma=60,
            ),
            "variant_c": dict(
                bounce_rate=0.38,
                form_submit_rate=0.11,
                demo_request_rate=0.024,
                time_on_page_mu=180,
                time_on_page_sigma=60,
            ),
            "variant_d": dict(
                bounce_rate=0.40,
                form_submit_rate=0.09,
                demo_request_rate=0.032,
                time_on_page_mu=200,  # +20 s
                time_on_page_sigma=60,
            ),
        }

        records = []
        for v_idx, (variant, p) in enumerate(variant_specs.items()):
            concentration = 50
            bounce = rng.binomial(
                1,
                rng.beta(
                    p["bounce_rate"] * concentration,
                    (1 - p["bounce_rate"]) * concentration,
                    n_per_variant,
                ),
            )
            form_submit = rng.binomial(
                1,
                rng.beta(
                    p["form_submit_rate"] * concentration,
                    (1 - p["form_submit_rate"]) * concentration,
                    n_per_variant,
                ),
            )
            demo_request = rng.binomial(
                1,
                rng.beta(
                    p["demo_request_rate"] * concentration,
                    (1 - p["demo_request_rate"]) * concentration,
                    n_per_variant,
                ),
            )
            time_on_page = np.clip(
                rng.normal(p["time_on_page_mu"], p["time_on_page_sigma"], n_per_variant),
                a_min=1,
                a_max=None,
            ).round(1)

            contact_ids = self._generate_ids(
                "P", n_per_variant, rng, v_idx * n_per_variant
            )

            chunk = pd.DataFrame(
                {
                    "contact_id": contact_ids,
                    "variant": variant,
                    "bounce_rate": bounce,
                    "form_submit_rate": form_submit,
                    "demo_request_rate": demo_request,
                    "time_on_page": time_on_page,
                }
            )
            records.append(chunk)

        df = pd.concat(records, ignore_index=True)
        df = self.add_stratification_columns(df, len(df), seed_offset=2)
        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Experiment 3 — Holdout / Incrementality Panel
    # ------------------------------------------------------------------

    def generate_exp003_holdout(self) -> pd.DataFrame:
        """Generate a 30 000-account, two-period holdout panel dataset.

        Design
        ------
        * 80 % exposed / 20 % holdout accounts, each observed in both
          baseline (30 days) and test (60 days) periods — genuine panel data.
        * Baseline outcome rates are identical across arms to satisfy the
          parallel-trends assumption.
        * Enterprise accounts receive a higher incremental lift (~20 %) to
          model heterogeneous treatment effects.
        * ``pipeline_value`` is zero when no opportunity is created, and
          LogNormal otherwise.
        * ``days_to_opportunity`` is Poisson-distributed when an opportunity
          exists, NaN otherwise.

        Returns
        -------
        pandas.DataFrame
        """
        rng = np.random.default_rng(self.seed + 3)
        n_accounts = 30_000
        n_exposed = 24_000
        n_holdout = 6_000

        account_ids_exposed = self._generate_ids("A", n_exposed, rng, 0)
        account_ids_holdout = self._generate_ids("A", n_holdout, rng, n_exposed)

        all_account_ids = account_ids_exposed + account_ids_holdout
        all_variants = ["exposed"] * n_exposed + ["holdout"] * n_holdout

        # Assign stratification once per account so it is consistent across
        # both periods.
        strat_df = pd.DataFrame({"account_id": all_account_ids, "variant": all_variants})
        strat_df = self.add_stratification_columns(strat_df, n_accounts, seed_offset=3)

        # Build period rows: each account appears twice (baseline + test)
        baseline_start = date(2025, 7, 1)
        baseline_end = date(2025, 7, 31)
        test_start = date(2025, 8, 1)
        test_end = date(2025, 9, 29)

        period_rows = []
        for period, p_start, p_end in [
            ("baseline", baseline_start, baseline_end),
            ("test", test_start, test_end),
        ]:
            delta_days = (p_end - p_start).days + 1
            day_offsets = rng.integers(0, delta_days, n_accounts)
            period_dates = [
                (p_start + timedelta(days=int(d))).isoformat() for d in day_offsets
            ]

            for i in range(n_accounts):
                variant = all_variants[i]
                company_size = strat_df.loc[i, "company_size"]

                if period == "baseline":
                    opp_rate = 0.11  # parallel trends — identical rates
                else:
                    if variant == "holdout":
                        opp_rate = 0.123
                    else:
                        # Heterogeneous lift: Enterprise +20%, SMB +8%, Mid-Market +15%
                        base_lift = 0.142
                        if company_size == "Enterprise":
                            opp_rate = 0.11 * 1.20
                        elif company_size == "SMB":
                            opp_rate = 0.11 * 1.08
                        else:
                            opp_rate = base_lift

                period_rows.append(
                    {
                        "account_id": all_account_ids[i],
                        "variant": variant,
                        "period": period,
                        "date": period_dates[i],
                        "_opp_rate": opp_rate,
                        "company_size_tmp": company_size,
                    }
                )

        panel_df = pd.DataFrame(period_rows)

        # Draw binary opportunity outcomes
        opp_rates = panel_df["_opp_rate"].values
        panel_df["opportunity_created"] = rng.binomial(1, opp_rates)

        # pipeline_value: LogNormal when opportunity=1, else 0
        n_rows = len(panel_df)
        lognorm_draws = np.exp(rng.normal(10.5, 0.8, n_rows))
        panel_df["pipeline_value"] = np.where(
            panel_df["opportunity_created"] == 1, lognorm_draws.round(2), 0.0
        )

        # days_to_opportunity: Poisson(25) when opportunity=1, else NaN
        poisson_draws = rng.poisson(25, n_rows).astype(float)
        panel_df["days_to_opportunity"] = np.where(
            panel_df["opportunity_created"] == 1, poisson_draws, np.nan
        )

        # Clean up helper columns
        panel_df = panel_df.drop(columns=["_opp_rate", "company_size_tmp"])

        # Merge back full stratification
        panel_df = panel_df.merge(
            strat_df[["account_id", "industry", "company_size", "region", "engagement_tier"]],
            on="account_id",
            how="left",
        )

        panel_df = panel_df.reset_index(drop=True)
        return panel_df

    # ------------------------------------------------------------------
    # Experiment 4 — Channel Mix (Email-Only vs Email + Social)
    # ------------------------------------------------------------------

    def generate_exp004_channel_mix(self) -> pd.DataFrame:
        """Generate a 40 000-account channel-mix experiment.

        Design
        ------
        * 50/50 split: ``email_only`` (control) vs ``email_plus_social``
          (treatment).
        * ``mql_conversion_rate`` is binary; ``engagement_score_delta`` and
          ``cost_per_mql`` are continuous.
        * Cost is modelled as LogNormal so it cannot go negative and has a
          realistic right-skewed distribution.
        * Guardrail: unsubscribe_rate stable at 0.4 % in both arms.

        Returns
        -------
        pandas.DataFrame
        """
        rng = np.random.default_rng(self.seed + 4)
        n_total = 40_000
        n_per_variant = n_total // 2

        records = []
        concentration = 50
        for v_idx, (variant, mql_rate, eng_mu, eng_sigma, cost_mu, cost_sigma) in enumerate(
            [
                ("email_only", 0.045, 0.0, 5.0, 120.0, 30.0),
                ("email_plus_social", 0.058, 2.1, 5.5, 185.0, 45.0),
            ]
        ):
            mql_p = rng.beta(
                mql_rate * concentration,
                (1 - mql_rate) * concentration,
                n_per_variant,
            )
            mql_outcomes = rng.binomial(1, mql_p)

            engagement_delta = rng.normal(eng_mu, eng_sigma, n_per_variant).round(2)

            # cost_per_mql: LogNormal parameterised so median ~ cost_mu
            log_mu = np.log(cost_mu)
            log_sigma = cost_sigma / cost_mu  # delta method approximation
            cost = np.exp(rng.normal(log_mu, log_sigma, n_per_variant)).round(2)

            unsub_rate = 0.004
            unsub_p = rng.beta(
                unsub_rate * concentration,
                (1 - unsub_rate) * concentration,
                n_per_variant,
            )
            unsub_outcomes = rng.binomial(1, unsub_p)

            account_ids = self._generate_ids(
                "CH", n_per_variant, rng, v_idx * n_per_variant
            )

            chunk = pd.DataFrame(
                {
                    "account_id": account_ids,
                    "variant": variant,
                    "mql_conversion_rate": mql_outcomes,
                    "engagement_score_delta": engagement_delta,
                    "cost_per_mql": cost,
                    "unsubscribe_rate": unsub_outcomes,
                }
            )
            records.append(chunk)

        df = pd.concat(records, ignore_index=True)
        df = self.add_stratification_columns(df, len(df), seed_offset=4)
        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Experiment 5 — Send-Time Optimisation
    # ------------------------------------------------------------------

    def generate_exp005_send_time(self) -> pd.DataFrame:
        """Generate a 60 000-contact three-group send-time experiment.

        Design
        ------
        * Three equal groups: ``morning`` (06:00–11:59), ``afternoon``
          (12:00–17:59), ``evening`` (18:00–23:59).
        * Afternoon has the highest open and click rates; evening is weakest.
        * Guardrail: unsubscribe_rate stable at 0.5 % across all groups.

        Returns
        -------
        pandas.DataFrame
        """
        rng = np.random.default_rng(self.seed + 5)
        n_per_group = 20_000

        group_specs = {
            "morning": dict(open_rate=0.21, click_rate=0.042, unsubscribe_rate=0.005),
            "afternoon": dict(open_rate=0.25, click_rate=0.038, unsubscribe_rate=0.005),
            "evening": dict(open_rate=0.20, click_rate=0.033, unsubscribe_rate=0.005),
        }

        records = []
        concentration = 50
        for g_idx, (group, p) in enumerate(group_specs.items()):
            open_outcomes = rng.binomial(
                1,
                rng.beta(
                    p["open_rate"] * concentration,
                    (1 - p["open_rate"]) * concentration,
                    n_per_group,
                ),
            )
            click_outcomes = rng.binomial(
                1,
                rng.beta(
                    p["click_rate"] * concentration,
                    (1 - p["click_rate"]) * concentration,
                    n_per_group,
                ),
            )
            unsub_outcomes = rng.binomial(
                1,
                rng.beta(
                    p["unsubscribe_rate"] * concentration,
                    (1 - p["unsubscribe_rate"]) * concentration,
                    n_per_group,
                ),
            )

            contact_ids = self._generate_ids(
                "ST", n_per_group, rng, g_idx * n_per_group
            )

            chunk = pd.DataFrame(
                {
                    "contact_id": contact_ids,
                    "send_time_group": group,
                    "open_rate": open_outcomes,
                    "click_rate": click_outcomes,
                    "unsubscribe_rate": unsub_outcomes,
                }
            )
            records.append(chunk)

        df = pd.concat(records, ignore_index=True)
        df = self.add_stratification_columns(df, len(df), seed_offset=5)
        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def add_stratification_columns(
        self,
        df: pd.DataFrame,
        n: int,
        seed_offset: int = 0,
    ) -> pd.DataFrame:
        """Append industry, company_size, region, and engagement_tier columns.

        Columns are drawn independently using the specified proportions so
        they are uncorrelated with variant assignment by default (mimicking a
        properly randomised experiment with measured covariates).

        Parameters
        ----------
        df:
            DataFrame to augment in-place (a copy is returned).
        n:
            Number of rows to generate labels for.
        seed_offset:
            Added to ``self.seed`` for this RNG instance, allowing each call
            site to produce independent but reproducible draws.

        Returns
        -------
        pandas.DataFrame
            Input DataFrame with four additional columns appended.
        """
        rng = np.random.default_rng(self.seed + seed_offset + 1_000)
        df = df.copy()
        df["industry"] = rng.choice(_INDUSTRIES, size=n, p=_INDUSTRY_WEIGHTS)
        df["company_size"] = rng.choice(
            _COMPANY_SIZES, size=n, p=_COMPANY_SIZE_WEIGHTS
        )
        df["region"] = rng.choice(_REGIONS, size=n, p=_REGION_WEIGHTS)
        df["engagement_tier"] = rng.choice(
            _ENGAGEMENT_TIERS, size=n, p=_ENGAGEMENT_TIER_WEIGHTS
        )
        return df

    def introduce_data_quality_issues(
        self,
        df: pd.DataFrame,
        id_col: str,
        dq_rate: float = 0.015,
        seed_offset: int = 99,
    ) -> pd.DataFrame:
        """Append a fraction of duplicate rows to simulate upstream DQ issues.

        Duplicates share the same ``id_col`` value and variant as their
        source row, reflecting a realistic CRM or CDP deduplication failure
        rather than identical full-row duplicates.

        Parameters
        ----------
        df:
            Original clean DataFrame.
        id_col:
            Column name holding the contact/account identifier.
        dq_rate:
            Fraction of rows to duplicate (default 1.5 %).
        seed_offset:
            Seed offset for reproducible row sampling.

        Returns
        -------
        pandas.DataFrame
            DataFrame with duplicate rows appended at the end.
        """
        rng = np.random.default_rng(self.seed + seed_offset)
        n_dupes = max(1, int(len(df) * dq_rate))
        dupe_indices = rng.choice(len(df), size=n_dupes, replace=False)
        dupes = df.iloc[dupe_indices].copy()
        return pd.concat([df, dupes], ignore_index=True)

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_ids(
        prefix: str,
        n: int,
        rng: np.random.Generator,
        offset: int = 0,
    ) -> list[str]:
        """Generate a list of zero-padded uuid-like string identifiers.

        Parameters
        ----------
        prefix:
            Short alphanumeric prefix (e.g. ``"C"`` for contact).
        n:
            Number of IDs to generate.
        rng:
            Seeded NumPy :class:`numpy.random.Generator` instance.
        offset:
            Integer offset applied to the sequential base to ensure
            uniqueness when called multiple times within the same experiment.

        Returns
        -------
        list[str]
        """
        # Use sequential integers + 6-hex random suffix for readability
        hex_suffix = rng.integers(0x100000, 0xFFFFFF, n)
        return [
            f"{prefix}{str(offset + i).zfill(8)}-{hex_suffix[i]:06x}"
            for i in range(n)
        ]

    @staticmethod
    def _build_metadata(exp_id: str, df: pd.DataFrame) -> dict:
        """Construct a metadata dict for a single experiment DataFrame.

        Parameters
        ----------
        exp_id:
            Experiment identifier string.
        df:
            The generated experiment DataFrame.

        Returns
        -------
        dict
            Contains ``shape``, ``columns``, ``dtypes``, and ``date_ranges``
            for any date-like columns.
        """
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # Detect date columns and summarise their range
        date_ranges: dict = {}
        for col in df.columns:
            if "date" in col.lower() and df[col].dtype == object:
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().sum() > 0:
                        date_ranges[col] = {
                            "min": parsed.min().isoformat(),
                            "max": parsed.max().isoformat(),
                        }
                except Exception:
                    pass

        return {
            "experiment_id": exp_id,
            "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            "columns": list(df.columns),
            "dtypes": dtypes,
            "date_ranges": date_ranges,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gen = ExperimentDataGenerator()
    data = gen.generate_all()
    print(f"Generated {len(data)} experiments")
    for name, df in data.items():
        print(f"  {name}: {len(df):,} rows x {df.shape[1]} cols")
