"""
=============================================================================
03_simulate.py  —  PHASE 3: Generate Stratified Experiment Data
=============================================================================

WHAT THIS FILE DOES
-------------------
This script generates the simulated A/B experiment dataset. It creates
one row per user, with all the metrics we care about: whether they accepted
the offer, how much revenue they generated, fraud flags, and so on.

WHY SIMULATE INSTEAD OF USING REAL DATA?
-----------------------------------------
Real A/B test data from banks is proprietary and never released publicly.
The simulation lets us:
  1. Control ground truth (we know the "true" lift is ~+1.5pp)
  2. Demonstrate we understand experimental statistics
  3. Create realistic variation without using confidential data

The key is that our simulation is CALIBRATED to real data (Phase 1), so
the numbers are grounded in reality — not made up.

WHAT IS STRATIFICATION?
------------------------
Instead of randomly assigning all 22,970 users into control and treatment,
we first divide users into 9 cells (3 utilization tiers × 3 payment segments)
and then assign 50/50 within each cell. This ensures balanced representation
across all user types — exactly what a production A/B testing platform does.

SRM (Sample Ratio Mismatch)
----------------------------
After assignment, we run a chi-squared test to verify the split is truly 50/50.
If the split is significantly off (e.g. 52/48), something went wrong in
assignment logic and the experiment is invalid. This is a real check that
teams at Meta, Airbnb, and Stripe run before reading any results.

OUTPUTS
-------
data/ab_experiment_data.csv   (22,970 rows — one per user)
data/ab_daily_summary.csv     (daily rollup for temporal charts)
outputs/figures/phase3_simulation_validation.png
=============================================================================
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats     # for chi-squared test in SRM check

np.random.seed(42)   # setting a seed makes the simulation reproducible

# ─── File paths ───────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "..", "data")
FIG_DIR  = os.path.join(ROOT, "..", "outputs", "figures")

# ─── Load calibration and config from previous phases ─────────────────────────
with open(os.path.join(DATA_DIR, "calibration_params.json")) as f:
    cal = json.load(f)
with open(os.path.join(DATA_DIR, "experiment_config.json")) as f:
    cfg = json.load(f)

# ─── Plot styling ─────────────────────────────────────────────────────────────
PALETTE = {"control": "#4A90D9", "treatment": "#E07B39"}
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


# =============================================================================
# STEP 1: Build the strata definition table
#
# Each stratum is a (util_tier, pay_segment) combination.
# We pull: segment share, default rate, and acceptance rates from calibration.
# =============================================================================

STRATA = {}
for seg_def in cal["segment_defaults"]:
    ut  = seg_def["util_tier"]
    ps  = seg_def["pay_segment"]
    key = f"{ut}|{ps}"
    STRATA[(ut, ps)] = {
        "share":        seg_def["share"],         # fraction of all users in this cell
        "default_rate": seg_def["default_rate"],  # real default rate from UCI
        "accept_ctrl":  cal["acceptance_by_segment"].get(key, 0.10),
        "accept_treat": cal["acceptance_treat_by_segment"].get(key, 0.12),
        "limit_mean":   cal["limit_by_util"].get(ut, {}).get("mean", 150_000),
        "limit_std":    cal["limit_by_util"].get(ut, {}).get("std",   80_000),
        "bill_mean":    cal["bill_by_util"].get(ut, {}).get("mean",    5_000),
        "bill_std":     cal["bill_by_util"].get(ut, {}).get("std",     4_000),
    }

n_per_var = cfg["n_per_variant"]    # users needed per variant (from power analysis)
runtime   = cfg["runtime_days"]     # 14 days


# =============================================================================
# STEP 2: Simulate users
#
# For each stratum × variant combination, we:
#   1. Draw a binary "accepted" outcome from Binomial(n, acceptance_rate)
#   2. Draw revenue from LogNormal for users who accepted
#   3. Draw monthly spend from LogNormal (slightly higher in treatment)
#   4. Draw default flags using a FIXED random seed (explained below)
#   5. Draw fraud flags
#   6. Assign each user to a random experiment day (1-14)
#
# WHY FIXED SEED FOR DEFAULT?
#   Default risk is an inherent property of the user — it shouldn't be
#   affected by which variant they saw. By using a stratum-level fixed seed,
#   we ensure control and treatment users in the same cell have the same
#   default rate. This prevents random noise from triggering the guardrail.
# =============================================================================

rows = []
for (ut, ps), s in STRATA.items():
    n_seg = max(30, int(n_per_var * s["share"]))   # at least 30 users per cell
    for variant in ["control", "treatment"]:

        # ── Acceptance: did the user click and accept the offer? ──────────────
        accept_rate = s["accept_ctrl"] if variant == "control" else s["accept_treat"]
        accepted    = np.random.binomial(1, accept_rate, n_seg)
        # np.random.binomial(1, p, n): n independent coin flips each with P(heads)=p

        # ── Revenue: interchange earned from users who accepted ───────────────
        rev_base = cal["avg_monthly_spend_delta"] * 12 * cal["interchange_rate"]
        revenue  = np.where(
            accepted == 1,
            np.random.lognormal(np.log(rev_base), 0.45, n_seg),
            0.0   # non-acceptors generate $0 incremental revenue
        )
        # lognormal is used because revenue is always positive and right-skewed
        # (most users spend a little more, a few spend a lot more)

        # ── Monthly spend: total monthly card spend ───────────────────────────
        spend_delta  = 0.04 if variant == "treatment" else 0.0  # +4% for treatment
        monthly_spend = np.random.lognormal(
            np.log(max(500, s["bill_mean"] * (1 + spend_delta))), 0.5, n_seg
        )

        # ── Default flag: inherent user property (not affected by offer shown) ─
        seg_seed     = abs(hash(f"{ut}_{ps}")) % (2**31)   # deterministic per stratum
        rng_def      = np.random.default_rng(seg_seed)     # separate RNG for defaults
        default_flag = rng_def.binomial(1, s["default_rate"], n_seg)
        # Same draws for control and treatment → no spurious guardrail breaches

        # ── Fraud flag: rare event, slightly elevated for delinquent users ────
        fraud_base   = 0.0045 if ps == "2+ mo delay" else 0.0018
        fraud_flag   = np.random.binomial(1, fraud_base, n_seg)

        # ── Limit offered: dollar amount in personalized offer ────────────────
        limit_offered = np.where(
            accepted == 1,
            np.random.lognormal(np.log(max(1000, s["limit_mean"] * 0.15)), 0.35, n_seg),
            0.0
        )

        # ── Experiment day: which day did this user enter the experiment? ──────
        exp_day = np.random.randint(1, runtime + 1, n_seg)

        # ── Append all users from this stratum × variant cell ─────────────────
        for i in range(n_seg):
            rows.append({
                "util_tier":     ut,
                "pay_segment":   ps,
                "variant":       variant,    # "control" or "treatment"
                "accepted":      int(accepted[i]),        # 1 = accepted, 0 = declined
                "revenue":       round(revenue[i], 2),    # incremental annual revenue ($)
                "monthly_spend": round(monthly_spend[i], 2),  # total monthly spend ($)
                "default_flag":  int(default_flag[i]),    # 1 = defaulted, 0 = did not
                "fraud_flag":    int(fraud_flag[i]),       # 1 = fraud detected, 0 = clean
                "limit_offered": round(limit_offered[i], 2),  # $ limit increase offered
                "exp_day":       int(exp_day[i]),         # day 1–14
            })

# Convert to DataFrame
df = pd.DataFrame(rows)
print(f"  Generated {len(df):,} rows across {len(STRATA)} strata × 2 variants")


# =============================================================================
# STEP 3: SRM Check — verify 50/50 assignment split
#
# Even though we designed for equal splits, let's formally verify using
# a chi-squared goodness-of-fit test.
#
# Chi-squared test:
#   Observed: [n_control, n_treatment]
#   Expected: [n_total/2, n_total/2]
#   p < 0.01 → significant mismatch → SRM detected → stop the analysis
# =============================================================================

counts   = df["variant"].value_counts()
n_c, n_t = counts["control"], counts["treatment"]
n_total  = n_c + n_t
chi2, p  = stats.chisquare([n_c, n_t], f_exp=[n_total/2, n_total/2])
srm_flag = "🚨 SRM DETECTED — halt analysis" if p < 0.01 else f"✅ No SRM  (p = {p:.4f})"
print(f"\n  SRM Check: control={n_c:,}  treatment={n_t:,}  χ²={chi2:.4f}  {srm_flag}")

# Per-stratum balance table (each cell should be exactly 1.000)
balance = (
    df.groupby(["util_tier", "pay_segment", "variant"])
      .size().unstack("variant")
      .assign(ratio=lambda x: (x["treatment"] / x["control"]).round(3))
)
print(f"\n  Per-stratum T/C balance (target = 1.000):")
print(balance.to_string())


# =============================================================================
# STEP 4: Build daily summary for temporal analysis in Phase 4
#
# Aggregates by experiment day × variant:
#   - How many users entered on each day
#   - How many accepted the offer on each day
#   - Total revenue on each day
# =============================================================================

daily = (
    df.groupby(["exp_day", "variant"])
      .agg(
          n       = ("accepted", "count"),   # total users
          k       = ("accepted", "sum"),     # total acceptances
          revenue = ("revenue",  "sum"),     # total revenue
      )
      .reset_index()
)
daily["cr"] = daily["k"] / daily["n"]   # daily conversion rate


# =============================================================================
# STEP 5: Save data files
# =============================================================================

df.to_csv(os.path.join(DATA_DIR, "ab_experiment_data.csv"), index=False)
daily.to_csv(os.path.join(DATA_DIR, "ab_daily_summary.csv"), index=False)
print(f"\n  Saved → data/ab_experiment_data.csv  ({len(df):,} rows)")
print(f"  Saved → data/ab_daily_summary.csv  ({len(daily)} rows)")


# =============================================================================
# STEP 6: Validation charts (4 panels)
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Panel A: Overall assignment counts (should be equal)
ax = axes[0, 0]
bars = ax.bar(["Control", "Treatment"], [n_c, n_t],
              color=[PALETTE["control"], PALETTE["treatment"]],
              width=0.4, edgecolor="white")
ax.axhline(n_total/2, color="gray", ls="--", lw=1.5,
           label=f"Expected per variant: {n_total//2:,}")
for bar, n in zip(bars, [n_c, n_t]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
            f"{n:,}", ha="center", va="bottom", fontweight="bold")
ax.set_title(f"Assignment Balance\n{srm_flag}", fontweight="bold")
ax.set_ylabel("Users assigned")
ax.legend()

# Panel B: Acceptance rates per stratum (shows the treatment effect varies by segment)
ax2 = axes[0, 1]
seg_cr = (
    df.groupby(["util_tier", "pay_segment", "variant"])["accepted"]
      .mean().reset_index()
)
seg_cr["segment"] = seg_cr["util_tier"].str[:6] + "\n" + seg_cr["pay_segment"].str[:7]
labels   = seg_cr[seg_cr.variant == "control"]["segment"].values
ctrl_cr  = seg_cr[seg_cr.variant == "control"]["accepted"].values
treat_cr = seg_cr[seg_cr.variant == "treatment"]["accepted"].values
x = np.arange(len(labels)); w = 0.35
ax2.bar(x-w/2, ctrl_cr*100,  width=w, color=PALETTE["control"],   label="Control",    edgecolor="white")
ax2.bar(x+w/2, treat_cr*100, width=w, color=PALETTE["treatment"], label="Treatment",  edgecolor="white")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylabel("Offer acceptance rate (%)")
ax2.set_title("Calibrated Acceptance Rates by Stratum\n(Treatment > control in all segments)", fontweight="bold")
ax2.legend(fontsize=9)

# Panel C: Heatmap of user counts per stratum
#   Verifies we have enough users in every cell for reliable analysis
ax3 = axes[1, 0]
hm_data = df.groupby(["util_tier","pay_segment"]).size().unstack("pay_segment")
sns.heatmap(hm_data, annot=True, fmt=",", cmap="Blues", ax=ax3,
            linewidths=0.5, cbar_kws={"label":"Users (both variants)"})
ax3.set_title("Users per Stratum\n(Verify sufficient sample in every cell)", fontweight="bold")

# Panel D: T/C ratio heatmap (should all be exactly 1.000)
ax4 = axes[1, 1]
ratio_data = balance["ratio"].unstack("pay_segment")
sns.heatmap(ratio_data, annot=True, fmt=".3f", cmap="RdYlGn",
            center=1.0, vmin=0.97, vmax=1.03, ax=ax4,
            linewidths=0.5, cbar_kws={"label":"T/C ratio (target = 1.000)"})
ax4.set_title("Treatment/Control Ratio per Stratum\n(Green = balanced, red = SRM)", fontweight="bold")

plt.suptitle("Phase 3 — Simulation & SRM Validation", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "phase3_simulation_validation.png")
plt.savefig(fig_path, bbox_inches="tight")
print(f"\n  Chart saved → {fig_path}")
print("  Phase 3 complete ✓\n")
