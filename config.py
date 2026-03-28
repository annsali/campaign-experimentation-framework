"""
Default experiment parameters and significance thresholds
for the Campaign Experimentation & Lift Measurement Framework.
"""

# ── Frequentist defaults ──────────────────────────────────────────────────────
SIGNIFICANCE_LEVEL = 0.05          # alpha (two-sided)
POWER = 0.80                       # 1 - beta
CONFIDENCE_LEVEL = 0.95            # for CIs
BOOTSTRAP_ITERATIONS = 10_000

# ── Bayesian defaults ─────────────────────────────────────────────────────────
BAYESIAN_MC_SAMPLES = 100_000      # Monte Carlo draws for Beta-Binomial
MCMC_DRAWS = 2_000
MCMC_CHAINS = 4
MCMC_TUNE = 1_000
MCMC_TARGET_ACCEPT = 0.9
RHAT_THRESHOLD = 1.01              # convergence: R-hat must be < this
MIN_ESS = 400                      # minimum effective sample size per chain

# ── Decision thresholds ───────────────────────────────────────────────────────
PROB_TREATMENT_BETTER_SHIP = 0.95  # P(B>A) to declare "ship treatment"
PROB_TREATMENT_BETTER_EXTEND = 0.80  # below this -> inconclusive
EXPECTED_LOSS_THRESHOLD = 0.005    # max acceptable expected loss to ship

# ── Guardrail thresholds ──────────────────────────────────────────────────────
GUARDRAIL_RELATIVE_DEGRADATION = 0.10   # 10% relative degradation blocks ship
GUARDRAIL_ALPHA = 0.10                  # one-sided alpha for non-inferiority tests

# ── Multiple comparison correction ───────────────────────────────────────────
DEFAULT_MCC = "holm"               # bonferroni | holm | fdr_bh | dunnett

# ── Sequential testing ────────────────────────────────────────────────────────
MAX_INTERIM_LOOKS = 5
FUTILITY_THRESHOLD = 0.20          # beta-spending for futility stop
EFFICACY_THRESHOLD = 0.05          # alpha-spending for efficacy stop

# ── Holdout defaults ──────────────────────────────────────────────────────────
DEFAULT_HOLDOUT_FRACTION = 0.20
PARALLEL_TRENDS_ALPHA = 0.10       # p-value threshold for trends check
CONTAMINATION_FLAG_THRESHOLD = 0.02  # >2% contamination triggers warning

# ── Segment analysis ──────────────────────────────────────────────────────────
MIN_SEGMENT_SIZE = 100             # minimum n per subgroup for HTE analysis
SEGMENT_ALPHA = 0.05               # after Bonferroni correction in HTE

# ── Data generation ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── ROPE (Region of Practical Equivalence) ───────────────────────────────────
ROPE_DEFAULTS = {
    "open_rate":          (-0.005, 0.005),
    "click_rate":         (-0.002, 0.002),
    "conversion_rate":    (-0.001, 0.001),
    "bounce_rate":        (-0.010, 0.010),
    "form_submit_rate":   (-0.005, 0.005),
    "mql_conversion_rate": (-0.005, 0.005),
    "opportunity_created": (-0.005, 0.005),
    "default":            (-0.005, 0.005),
}

# ── Metric types ──────────────────────────────────────────────────────────────
PROPORTION_METRICS = {
    "open_rate", "click_rate", "conversion_rate", "bounce_rate",
    "form_submit_rate", "demo_request_rate", "mql_conversion_rate",
    "opportunity_created", "unsubscribe_rate", "spam_complaint_rate",
}
CONTINUOUS_METRICS = {
    "time_on_page", "pipeline_value", "days_to_opportunity",
    "engagement_score_delta", "cost_per_mql",
}

# ── Experiment catalog ────────────────────────────────────────────────────────
EXPERIMENT_CATALOG_PATH = "experiments/experiment_catalog.json"
DATA_DIR = "data"
VISUALS_DIR = "visuals"
