"""
=============================================================================
02_experiment_design.py  —  PHASE 2: Power Analysis & Experiment Design
=============================================================================

WHAT THIS FILE DOES
-------------------
Before running any experiment, we must answer: "How many users do I need
to reliably detect a real effect?" This is called POWER ANALYSIS.

If you run an experiment with too few users, you might miss a real effect
(false negative). If you run too long, you waste engineering resources and
expose more users to a potentially worse experience.

This file computes the exact sample size required, produces three power
curve charts, and saves the experiment configuration.

KEY CONCEPTS
------------
Significance level (α = 0.05)
  If there is NO real effect, we will still see a "significant" result
  5% of the time just by chance. This is our false-positive tolerance.

Statistical power (1-β = 0.80)
  If there IS a real effect, we want an 80% chance of detecting it.
  The remaining 20% is the false-negative rate (type II error).

Minimum Detectable Effect (MDE)
  The smallest lift we actually care about. If the personalized offer
  produces less than +1.5pp improvement, it's not worth the engineering
  cost to ship. So we set MDE = 1.5pp and size the experiment accordingly.

Cohen's h
  A standardized effect size measure for proportions.
  Accounts for the fact that 5% → 6% is a bigger effect than 50% → 51%,
  even though both are +1pp absolute.

OUTPUTS
-------
data/experiment_config.json         (read by phases 3 and 4)
outputs/figures/phase2_power_analysis.png
=============================================================================
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# statsmodels provides the industry-standard power analysis functions
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

# ─── File paths ───────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "..", "data")
FIG_DIR  = os.path.join(ROOT, "..", "outputs", "figures")

# ─── Load calibration params from Phase 1 ────────────────────────────────────
with open(os.path.join(DATA_DIR, "calibration_params.json")) as f:
    cal = json.load(f)

# ─── Plot styling ─────────────────────────────────────────────────────────────
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


# =============================================================================
# STEP 1: Define experiment parameters
#
# These come directly from calibration_params.json (Phase 1 output).
# The baseline and target rates are population-weighted averages across
# all 9 user segments.
# =============================================================================

ALPHA          = 0.05   # significance level (false positive rate we'll tolerate)
POWER          = 0.80   # desired statistical power (probability of detecting a real effect)

# Baseline = control acceptance rate (generic prompt, from calibration)
BASELINE       = cal["acceptance_baseline_control"]    # ~7.5%
# Target = treatment acceptance rate (personalized offer, from calibration)
TARGET         = cal["acceptance_baseline_treatment"]  # ~9.1%

MDE_ABS        = TARGET - BASELINE   # absolute MDE  (~+1.5pp)
MDE_REL        = MDE_ABS / BASELINE  # relative MDE  (~+20%)

DAILY_ELIGIBLE = cal["daily_eligible_users"]    # 8,000 users/day see the offer page
PLATFORM_MAU   = cal["platform_mau"]            # 2.4M users on the platform


# =============================================================================
# STEP 2: Compute required sample size
#
# We use Cohen's h as the effect size measure for proportions.
# proportion_effectsize(p1, p2) = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
# This transformation stabilizes variance across different rate ranges.
#
# NormalIndPower().solve_power() then tells us how many users per variant
# we need to achieve the desired power at the given alpha and effect size.
# =============================================================================

effect_h  = proportion_effectsize(BASELINE, TARGET)
# Note: effect_h will be negative if p2 < p1, so we take abs() below

analysis  = NormalIndPower()
n_per_var = int(np.ceil(analysis.solve_power(
    effect_size = abs(effect_h),   # effect size (Cohen's h)
    alpha       = ALPHA,           # significance level
    power       = POWER,           # desired power
    ratio       = 1.0,             # equal split: 50% control / 50% treatment
    alternative = "two-sided"      # two-tailed test (detect increase OR decrease)
)))
# Ceiling: always round up — you want AT LEAST this many users, never fewer

runtime = int(np.ceil(n_per_var * 2 / DAILY_ELIGIBLE))
# Total users needed divided by how many arrive per day = days to run the test

# ─── Print results ────────────────────────────────────────────────────────────
print("=" * 58)
print("  EXPERIMENT DESIGN — CREDIT LIMIT OFFER A/B TEST")
print("=" * 58)
print(f"\n  Pre-registered Hypothesis")
print(f"  H₀ : acceptance(treatment) = acceptance(control)  [no effect]")
print(f"  H₁ : acceptance(treatment) ≠ acceptance(control)  [two-tailed]")
print(f"\n  Statistical Parameters")
print(f"  {'Baseline acceptance rate':<38} {BASELINE:.3%}")
print(f"  {'Target acceptance rate (MDE)':<38} {TARGET:.3%}")
print(f"  {'Absolute MDE':<38} {MDE_ABS:+.3%}")
print(f"  {'Relative MDE':<38} {MDE_REL:+.1%}")
print(f"  {'Cohen h (effect size)':<38} {effect_h:.4f}")
print(f"  {'Significance level α':<38} {ALPHA}")
print(f"  {'Power 1-β':<38} {POWER:.0%}")
print(f"\n  Sample Size")
print(f"  {'Required n per variant':<38} {n_per_var:,}")
print(f"  {'Required n total':<38} {n_per_var*2:,}")
print(f"  {'Daily eligible users':<38} {DAILY_ELIGIBLE:,}")
print(f"  {'Estimated runtime':<38} {runtime} days")
print(f"\n  Pre-registered Guardrails (failure = do not ship)")
print(f"  {'Default rate lift':<38} ≤ +0.50pp vs control")
print(f"  {'Fraud flag rate lift':<38} ≤ +0.10pp vs control")
print(f"  {'Avg monthly spend change':<38} ≥ -5.00% vs control")


# =============================================================================
# STEP 3: Save experiment configuration
#
# This JSON is read by 03_simulate.py and 04_analysis.py.
# Centralizing the config means changing alpha in one place updates everything.
# =============================================================================

config = {
    "alpha":          ALPHA,
    "power":          POWER,
    "baseline":       BASELINE,
    "target":         TARGET,
    "mde_abs":        round(MDE_ABS, 6),
    "mde_rel":        round(MDE_REL, 6),
    "effect_h":       round(effect_h, 6),
    "n_per_variant":  n_per_var,
    "n_total":        n_per_var * 2,
    "runtime_days":   14,         # fixed at 14 days for richer temporal analysis
    "daily_eligible": DAILY_ELIGIBLE,
    "platform_mau":   PLATFORM_MAU,
    "revenue_per_acceptance": cal["revenue_per_acceptance"],  # $75.60
    # Guardrail thresholds — pre-registered BEFORE looking at the data
    "guardrails": {
        "default_rate_max_lift_pp": 0.0050,   # +0.50 percentage points max
        "fraud_rate_max_lift_pp":   0.0010,   # +0.10 percentage points max
        "spend_min_pct_change":     -0.05,    # -5% maximum allowed drop
    }
}
with open(os.path.join(DATA_DIR, "experiment_config.json"), "w") as f:
    json.dump(config, f, indent=2)


# =============================================================================
# STEP 4: Build power analysis charts (3 panels)
# =============================================================================

# Data for Panel A: how sample size changes as MDE changes
# (The smaller the effect you want to detect, the more users you need)
mde_range = np.linspace(0.003, 0.06, 200)
n_range   = []
for d in mde_range:
    try:
        n = int(np.ceil(analysis.solve_power(
            effect_size = abs(proportion_effectsize(BASELINE, BASELINE + d)),
            alpha=ALPHA, power=POWER, ratio=1.0, alternative="two-sided"
        )))
        n_range.append(n)
    except:
        n_range.append(np.nan)

# Data for Panels B and C: how power changes as sample size grows
n_vals   = np.linspace(5_000, 60_000, 300)
pwr_vals = [analysis.solve_power(
    effect_size=abs(effect_h), alpha=ALPHA,
    nobs1=n, ratio=1.0, alternative="two-sided"
) for n in n_vals]

# Data for Panel C: same but across different alpha values
# Shows how being stricter (α=0.01) or more lenient (α=0.10) affects required n
alpha_vals   = [0.01, 0.05, 0.10]
alpha_labels = ["α=0.01 (strict)", "α=0.05 (standard)", "α=0.10 (lenient)"]
alpha_colors = ["#2E8B57", "#4A90D9", "#A8C8F0"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# Panel A: Sample size vs MDE
# READING THIS CHART: Moving left = smaller effect to detect = exponentially more users needed
axes[0].plot(mde_range * 100, np.array(n_range) / 1000, color="#4A90D9", lw=2.2)
axes[0].axvline(MDE_ABS * 100, color="#E07B39", ls="--", lw=1.6,
                label=f"Our MDE: {MDE_ABS*100:.2f}pp")
axes[0].axhline(n_per_var / 1000, color="#E07B39", ls=":", lw=1.3,
                label=f"Required n: {n_per_var/1000:.0f}K")
axes[0].set_xlabel("Minimum Detectable Effect (pp)")
axes[0].set_ylabel("n per variant (thousands)")
axes[0].set_title("Panel A: Sample Size vs MDE\n(Larger effect → fewer users needed)", fontweight="bold")
axes[0].legend(fontsize=9)

# Panel B: Power vs sample size at our MDE
# READING THIS CHART: Starts near 0% power, crosses 80% threshold at our required n
axes[1].plot(n_vals / 1000, pwr_vals, color="#4A90D9", lw=2.2)
axes[1].axhline(0.80, color="#E07B39", ls="--", lw=1.6, label="80% power target")
axes[1].axvline(n_per_var / 1000, color="#E07B39", ls=":", lw=1.3,
                label=f"n = {n_per_var/1000:.0f}K per variant")
axes[1].fill_between(n_vals / 1000, pwr_vals, 0.80,
                     where=[p < 0.80 for p in pwr_vals],
                     alpha=0.08, color="#E07B39", label="Underpowered zone")
axes[1].set_xlabel("n per variant (thousands)")
axes[1].set_ylabel("Statistical power (1-β)")
axes[1].set_title("Panel B: Power vs Sample Size\n(80% threshold marked in orange)", fontweight="bold")
axes[1].legend(fontsize=9)

# Panel C: Power vs n across different alpha levels
# READING THIS CHART: Stricter α (green) needs more users to reach 80% power
for av, lbl, col in zip(alpha_vals, alpha_labels, alpha_colors):
    pwr_a = [analysis.solve_power(
        effect_size=abs(effect_h), alpha=av,
        nobs1=n, ratio=1.0, alternative="two-sided"
    ) for n in n_vals]
    axes[2].plot(n_vals / 1000, pwr_a, color=col, lw=2.2, label=lbl)
axes[2].axhline(0.80, color="gray", ls="--", lw=1.2, label="80% target")
axes[2].axvline(n_per_var / 1000, color="#E07B39", ls=":", lw=1.3,
                label=f"Our n: {n_per_var/1000:.0f}K")
axes[2].set_xlabel("n per variant (thousands)")
axes[2].set_ylabel("Statistical power (1-β)")
axes[2].set_title("Panel C: Power vs n Across α Levels\n(α=0.01 needs more users than α=0.05)", fontweight="bold")
axes[2].legend(fontsize=9)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "phase2_power_analysis.png")
plt.savefig(fig_path, bbox_inches="tight")
print(f"\n  Chart saved → {fig_path}")
print("  Phase 2 complete ✓\n")
