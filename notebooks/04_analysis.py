"""
04_analysis.py
===============
Full statistical analysis of the A/B experiment.
Reads simulated data from Phase 3.

Outputs:
    outputs/figures/phase4a_primary_metrics.png
    outputs/figures/phase4b_subgroup_analysis.png
    outputs/figures/phase4c_guardrails.png
    outputs/figures/phase4d_novelty_trend.png
    outputs/figures/phase4e_business_impact.png
    outputs/phase4_results_summary.txt
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

np.random.seed(0)

ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "..", "data")
FIG_DIR  = os.path.join(ROOT, "..", "outputs", "figures")
OUT_DIR  = os.path.join(ROOT, "..", "outputs")
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, "experiment_config.json")) as f:
    cfg = json.load(f)

df    = pd.read_csv(os.path.join(DATA_DIR, "ab_experiment_data.csv"))
daily = pd.read_csv(os.path.join(DATA_DIR, "ab_daily_summary.csv"))

ALPHA   = cfg["alpha"]
PALETTE = {"control": "#4A90D9", "treatment": "#E07B39"}
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

ctrl  = df[df.variant == "control"]
treat = df[df.variant == "treatment"]
n_c, n_t = len(ctrl), len(treat)

results = {}

# ══════════════════════════════════════════════════════════════════════════════
# PRIMARY METRIC: Offer Acceptance Rate
# ══════════════════════════════════════════════════════════════════════════════
k_c, k_t   = ctrl.accepted.sum(), treat.accepted.sum()
cr_c, cr_t = k_c / n_c, k_t / n_t
lift_abs   = cr_t - cr_c
lift_rel   = lift_abs / cr_c
se         = np.sqrt(cr_c*(1-cr_c)/n_c + cr_t*(1-cr_t)/n_t)
z_crit     = stats.norm.ppf(1 - ALPHA/2)
ci         = (lift_abs - z_crit*se, lift_abs + z_crit*se)
z, p_cr    = proportions_ztest([k_t, k_c], [n_t, n_c], alternative="two-sided")

results["primary"] = dict(cr_c=cr_c, cr_t=cr_t, lift_abs=lift_abs,
                           lift_rel=lift_rel, ci=ci, z=z, p=p_cr,
                           sig=(p_cr < ALPHA))

print("=" * 60)
print("  PRIMARY METRIC: Offer Acceptance Rate")
print("=" * 60)
print(f"  Control      {cr_c:.4%}  ({k_c:,} / {n_c:,})")
print(f"  Treatment    {cr_t:.4%}  ({k_t:,} / {n_t:,})")
print(f"  Lift         {lift_abs:+.4%}  ({lift_rel:+.2%} relative)")
print(f"  95% CI       [{ci[0]:+.4%},  {ci[1]:+.4%}]")
print(f"  Z-stat       {z:.4f}   p = {p_cr:.6f}")
print(f"  Result       {'✅ SIGNIFICANT' if p_cr < ALPHA else '❌ Not significant'}")

# ══════════════════════════════════════════════════════════════════════════════
# SECONDARY METRIC: Revenue per User (bootstrap CI)
# ══════════════════════════════════════════════════════════════════════════════
rpu_c, rpu_t = ctrl.revenue.mean(), treat.revenue.mean()
rpu_lift     = rpu_t - rpu_c
u_stat, p_rpu = stats.mannwhitneyu(treat.revenue, ctrl.revenue, alternative="two-sided")
rng = np.random.default_rng(99)
boot = [rng.choice(treat.revenue, len(treat), replace=True).mean()
        - rng.choice(ctrl.revenue, len(ctrl), replace=True).mean()
        for _ in range(5_000)]
rpu_ci = tuple(np.percentile(boot, [2.5, 97.5]))

results["secondary"] = dict(rpu_c=rpu_c, rpu_t=rpu_t, lift=rpu_lift,
                              ci=rpu_ci, u=u_stat, p=p_rpu, sig=(p_rpu < ALPHA))

print(f"\n  SECONDARY: Revenue per User")
print(f"  Control  ${rpu_c:.4f}   Treatment  ${rpu_t:.4f}   Lift ${rpu_lift:+.4f}")
print(f"  95% Bootstrap CI  [${rpu_ci[0]:+.4f},  ${rpu_ci[1]:+.4f}]")
print(f"  Mann-Whitney p = {p_rpu:.6f}  {'✅' if p_rpu < ALPHA else '❌'}")

# ══════════════════════════════════════════════════════════════════════════════
# SUBGROUP ANALYSIS (util_tier × pay_segment)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  SUBGROUP ANALYSIS")
sg = (
    df.groupby(["util_tier", "pay_segment", "variant"])["accepted"]
      .agg(["sum","count"]).unstack("variant")
)
sg.columns = ["k_ctrl","k_treat","n_ctrl","n_treat"]
sg["cr_ctrl"]  = sg.k_ctrl  / sg.n_ctrl
sg["cr_treat"] = sg.k_treat / sg.n_treat
sg["lift_abs"] = sg.cr_treat - sg.cr_ctrl
sg["lift_rel"] = sg.lift_abs / sg.cr_ctrl
sg = sg.reset_index()

print(f"\n  {'Util tier':<20} {'Pay segment':<15} {'Ctrl CR':>9} {'Treat CR':>9} {'Δ abs':>9} {'Δ rel':>9}")
print("  " + "─"*73)
for _, row in sg.iterrows():
    print(f"  {row['util_tier']:<20} {row['pay_segment']:<15} "
          f"{row['cr_ctrl']:>9.3%} {row['cr_treat']:>9.3%} "
          f"{row['lift_abs']:>+9.3%} {row['lift_rel']:>+9.2%}")

results["subgroup"] = sg.to_dict(orient="records")

# ══════════════════════════════════════════════════════════════════════════════
# GUARDRAIL 1: Default Rate
# ══════════════════════════════════════════════════════════════════════════════
k_def_c, k_def_t = ctrl.default_flag.sum(), treat.default_flag.sum()
dr_c, dr_t       = k_def_c/n_c, k_def_t/n_t
dr_lift           = dr_t - dr_c
_, p_def          = proportions_ztest([k_def_t, k_def_c], [n_t, n_c], alternative="larger")
gr_default        = dr_lift <= cfg["guardrails"]["default_rate_max_lift_pp"]

results["gr_default"] = dict(rate_c=dr_c, rate_t=dr_t, lift=dr_lift,
                               threshold=cfg["guardrails"]["default_rate_max_lift_pp"],
                               p=p_def, passed=gr_default)

print(f"\n  GUARDRAIL 1: Default Rate")
print(f"  Control {dr_c:.4%}   Treatment {dr_t:.4%}   Δ={dr_lift:+.4%}   "
      f"threshold ≤+0.50pp   {'✅ PASS' if gr_default else '🚨 FAIL'}")

# ══════════════════════════════════════════════════════════════════════════════
# GUARDRAIL 2: Fraud Flag Rate
# ══════════════════════════════════════════════════════════════════════════════
k_fr_c, k_fr_t = ctrl.fraud_flag.sum(), treat.fraud_flag.sum()
fr_c, fr_t     = k_fr_c/n_c, k_fr_t/n_t
fr_lift         = fr_t - fr_c
_, p_fr         = proportions_ztest([k_fr_t, k_fr_c], [n_t, n_c], alternative="larger")
gr_fraud        = fr_lift <= cfg["guardrails"]["fraud_rate_max_lift_pp"]

results["gr_fraud"] = dict(rate_c=fr_c, rate_t=fr_t, lift=fr_lift,
                            threshold=cfg["guardrails"]["fraud_rate_max_lift_pp"],
                            p=p_fr, passed=gr_fraud)

print(f"  GUARDRAIL 2: Fraud Rate")
print(f"  Control {fr_c:.4%}   Treatment {fr_t:.4%}   Δ={fr_lift:+.4%}   "
      f"threshold ≤+0.10pp   {'✅ PASS' if gr_fraud else '🚨 FAIL'}")

# ══════════════════════════════════════════════════════════════════════════════
# GUARDRAIL 3: Avg Monthly Spend (must not decline)
# ══════════════════════════════════════════════════════════════════════════════
sp_c, sp_t = ctrl.monthly_spend.mean(), treat.monthly_spend.mean()
sp_pct     = (sp_t - sp_c) / sp_c
_, p_sp    = stats.ttest_ind(treat.monthly_spend, ctrl.monthly_spend, alternative="less")
gr_spend   = sp_pct >= cfg["guardrails"]["spend_min_pct_change"]

results["gr_spend"] = dict(mean_c=sp_c, mean_t=sp_t, pct_chg=sp_pct,
                            threshold=-0.05, p=p_sp, passed=gr_spend)

print(f"  GUARDRAIL 3: Monthly Spend")
print(f"  Control ${sp_c:,.0f}   Treatment ${sp_t:,.0f}   Δ={sp_pct:+.2%}   "
      f"threshold ≥-5.00%   {'✅ PASS' if gr_spend else '🚨 FAIL'}")

# ══════════════════════════════════════════════════════════════════════════════
# NOVELTY EFFECT
# ══════════════════════════════════════════════════════════════════════════════
pivot = daily.pivot(index="exp_day", columns="variant", values="cr").ffill()
pivot["lift"] = pivot["treatment"] - pivot["control"]
early_lift  = pivot[pivot.index <= 3]["lift"].mean()
steady_lift = pivot[pivot.index > 3]["lift"].mean()
nov_ratio   = early_lift / steady_lift if steady_lift != 0 else np.nan
nov_flag    = (nov_ratio > 1.20) if not np.isnan(nov_ratio) else False

results["novelty"] = dict(early=early_lift, steady=steady_lift,
                           ratio=nov_ratio, flag=nov_flag)

print(f"\n  NOVELTY EFFECT")
print(f"  Early lift (d1-3)  {early_lift:+.4%}   Steady (d4+)  {steady_lift:+.4%}"
      f"   Ratio {nov_ratio:.3f}   {'⚠ Inflation' if nov_flag else '✅ Stable'}")

# ══════════════════════════════════════════════════════════════════════════════
# BUSINESS IMPACT
# ══════════════════════════════════════════════════════════════════════════════
rev_per_acc      = cfg["revenue_per_acceptance"]
platform_mau     = cfg["platform_mau"]
daily_eligible   = cfg["daily_eligible"]

incr_daily_acc   = daily_eligible * lift_abs
daily_rev_impact = incr_daily_acc * rev_per_acc
annual_rev       = daily_rev_impact * 365
ci_rev_low       = daily_eligible * ci[0] * rev_per_acc
ci_rev_high      = daily_eligible * ci[1] * rev_per_acc

results["business"] = dict(
    daily_eligible=daily_eligible, lift_abs=lift_abs,
    incr_daily_acceptances=incr_daily_acc,
    rev_per_acceptance=rev_per_acc,
    daily_rev_impact=daily_rev_impact,
    annual_rev_impact=annual_rev,
    ci_daily_low=ci_rev_low, ci_daily_high=ci_rev_high,
)

print(f"\n  BUSINESS IMPACT")
print(f"  Daily eligible users        {daily_eligible:>10,}")
print(f"  Incremental acceptances/day {incr_daily_acc:>10.0f}")
print(f"  Revenue per acceptance      ${rev_per_acc:>9.2f}")
print(f"  Daily revenue impact        ${daily_rev_impact:>9,.0f}")
print(f"  95% CI (daily)              [${ci_rev_low:,.0f}  –  ${ci_rev_high:,.0f}]")
print(f"  Annual revenue impact       ${annual_rev:>9,.0f}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL DECISION
# ══════════════════════════════════════════════════════════════════════════════
all_gr  = gr_default and gr_fraud and gr_spend
primary = results["primary"]["sig"]

print(f"\n{'='*60}")
print("  LAUNCH DECISION SCORECARD")
print(f"{'='*60}")
checks = [
    ("Primary metric significant (p < 0.05)", primary,     f"p = {p_cr:.6f}"),
    ("Default rate guardrail",                gr_default,   f"Δ = {dr_lift:+.4%}  (≤+0.50pp)"),
    ("Fraud rate guardrail",                  gr_fraud,     f"Δ = {fr_lift:+.4%}  (≤+0.10pp)"),
    ("Monthly spend guardrail",               gr_spend,     f"Δ = {sp_pct:+.2%}  (≥-5.00%)"),
    ("No novelty inflation",                  not nov_flag, f"Ratio = {nov_ratio:.3f}  (<1.20)"),
]
for label, passed, detail in checks:
    print(f"  {'✅' if passed else '🚨'}  {label:<44} {detail}")
print(f"\n  Daily revenue uplift  : ${daily_rev_impact:,.0f}")
print(f"  95% CI               : [${ci_rev_low:,.0f}  –  ${ci_rev_high:,.0f}]")
print(f"  Annualised           : ${annual_rev:,.0f}")
print()
if primary and all_gr:
    print("  🚀  RECOMMENDATION: SHIP — full rollout approved.")
    print("      Post-launch: monitor default + fraud rates for 14 days.")
else:
    print("  🛑  RECOMMENDATION: DO NOT SHIP — review failures above.")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with open(os.path.join(OUT_DIR, "phase4_results_summary.txt"), "w") as f:
    f.write(f"Offer Acceptance Rate\n")
    f.write(f"  Control   : {cr_c:.4%}\n")
    f.write(f"  Treatment : {cr_t:.4%}\n")
    f.write(f"  Lift      : {lift_abs:+.4%}  (95% CI [{ci[0]:+.4%}, {ci[1]:+.4%}])\n")
    f.write(f"  p-value   : {p_cr:.6f}  {'SIGNIFICANT' if primary else 'NOT SIGNIFICANT'}\n\n")
    f.write(f"Daily Revenue Impact : ${daily_rev_impact:,.0f}\n")
    f.write(f"Annual Revenue Impact: ${annual_rev:,.0f}\n\n")
    f.write(f"Guardrails: default={'PASS' if gr_default else 'FAIL'}  "
            f"fraud={'PASS' if gr_fraud else 'FAIL'}  "
            f"spend={'PASS' if gr_spend else 'FAIL'}\n")
    f.write(f"Novelty: ratio={nov_ratio:.3f}  {'FLAG' if nov_flag else 'OK'}\n")

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

# ── 4a: Primary metrics ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
rates = [cr_c, cr_t]
se_vals = [np.sqrt(r*(1-r)/n) for r, n in [(cr_c, n_c), (cr_t, n_t)]]
bars = ax.bar(["Control\n(Generic prompt)", "Treatment\n(Personalized offer)"],
              [r*100 for r in rates],
              color=[PALETTE["control"], PALETTE["treatment"]],
              width=0.45, edgecolor="white")
ax.errorbar(["Control\n(Generic prompt)", "Treatment\n(Personalized offer)"],
            [r*100 for r in rates],
            yerr=[s*1.96*100 for s in se_vals],
            fmt="none", color="#333", capsize=6, lw=1.5, zorder=5)
for bar, r in zip(bars, [r*100 for r in rates]):
    ax.text(bar.get_x()+bar.get_width()/2, r+0.1,
            f"{r:.3f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
ymin = min(cr_c, cr_t)*100*0.92
ax.set_ylim(ymin, max(cr_c, cr_t)*100*1.12)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
ax.set_title("Offer Acceptance Rate ± 95% CI", fontweight="bold")
ax.set_ylabel("Acceptance rate (%)")
ax.annotate("", xy=(1, cr_t*100), xytext=(1, cr_c*100),
            arrowprops=dict(arrowstyle="<->", color="#E07B39", lw=2))
ax.text(1.17, (cr_c+cr_t)/2*100,
        f"{lift_abs*100:+.3f}pp\np={p_cr:.4f}",
        color="#E07B39", fontsize=10, va="center")

ax2 = axes[1]
sub = df[df.revenue > 0]
sns.violinplot(data=sub, x="variant", y="revenue", order=["control","treatment"],
               palette=PALETTE, inner="quartile", ax=ax2, linewidth=1.2)
ax2.set_title("Revenue per User (acceptors only)", fontweight="bold")
ax2.set_ylabel("Revenue ($)"); ax2.set_xlabel("")
ax2.set_xticklabels(["Control", "Treatment"])
for i, (v, m) in enumerate([("control",rpu_c),("treatment",rpu_t)]):
    sub_v = sub[sub.variant==v].revenue
    ax2.text(i, sub_v.max()*0.93, f"mean=${m:.2f}",
             ha="center", fontsize=10, fontweight="bold", color=PALETTE[v])

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "phase4a_primary_metrics.png"), bbox_inches="tight")
plt.close()
print(f"\n  Chart saved → phase4a_primary_metrics.png")

# ── 4b: Subgroup analysis ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

pivot_abs = sg.pivot(index="util_tier", columns="pay_segment", values="lift_abs")
sns.heatmap(pivot_abs*100, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, linewidths=0.6, ax=axes[0],
            cbar_kws={"label":"Absolute lift (pp)"})
axes[0].set_title("Absolute Lift (pp) — Util Tier × Payment Segment", fontweight="bold")

ut_order = ["Low (<30%)", "Medium (30–70%)", "High (>70%)"]
ps_order  = ["On-time", "1-mo delay", "2+ mo delay"]
x = np.arange(len(ut_order)); w = 0.25
for i, ps in enumerate(ps_order):
    lifts = []
    for ut in ut_order:
        row = sg[(sg.util_tier==ut) & (sg.pay_segment==ps)]
        lifts.append(row["lift_rel"].values[0]*100 if len(row) else 0)
    axes[1].bar(x + (i-1)*w, lifts, width=w, label=ps, edgecolor="white",
                color=["#5CBF7A","#4A90D9","#E07B39"][i])
axes[1].axhline(lift_rel*100, color="gray", ls="--", lw=1.4,
                label=f"Overall: {lift_rel*100:.1f}%")
axes[1].set_xticks(x)
axes[1].set_xticklabels(["Low util\n(<30%)", "Med util\n(30-70%)", "High util\n(>70%)"])
axes[1].set_ylabel("Relative lift (%)"); axes[1].legend(fontsize=9)
axes[1].set_title("Relative Lift by Segment\n(Strongest: Medium-util, On-time)", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "phase4b_subgroup_analysis.png"), bbox_inches="tight")
plt.close()
print(f"  Chart saved → phase4b_subgroup_analysis.png")

# ── 4c: Guardrails ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

def gr_bar(ax, ctrl_v, treat_v, threshold_lift, unit, title):
    bars = ax.bar(["Control", "Treatment"],
                  [ctrl_v*100, treat_v*100] if unit == "%" else [ctrl_v, treat_v],
                  color=[PALETTE["control"], PALETTE["treatment"]],
                  width=0.4, edgecolor="white")
    for bar, v in zip(bars, [ctrl_v, treat_v]):
        label = f"{v*100:.3f}%" if unit == "%" else f"${v:,.0f}"
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*(1.01),
                label, ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_title(title, fontweight="bold")

gr_bar(axes[0], dr_c, dr_t, cfg["guardrails"]["default_rate_max_lift_pp"], "%",
       f"Default Rate  |  {'✅ PASS' if gr_default else '🚨 FAIL'}\n"
       f"(Δ={dr_lift*100:+.3f}pp, thresh ≤+0.50pp)")
gr_bar(axes[1], fr_c, fr_t, cfg["guardrails"]["fraud_rate_max_lift_pp"], "%",
       f"Fraud Flag Rate  |  {'✅ PASS' if gr_fraud else '🚨 FAIL'}\n"
       f"(Δ={fr_lift*100:+.3f}pp, thresh ≤+0.10pp)")
gr_bar(axes[2], sp_c, sp_t, None, "$",
       f"Avg Monthly Spend  |  {'✅ PASS' if gr_spend else '🚨 FAIL'}\n"
       f"(Δ={sp_pct:+.2%}, thresh ≥-5.00%)")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "phase4c_guardrails.png"), bbox_inches="tight")
plt.close()
print(f"  Chart saved → phase4c_guardrails.png")

# ── 4d: Novelty + cumulative trend ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

for v, color, lbl in [("control",PALETTE["control"],"Control"),
                       ("treatment",PALETTE["treatment"],"Treatment")]:
    axes[0].plot(pivot.index, pivot[v]*100, marker="o", ms=4, lw=2, color=color, label=lbl)
axes[0].axvspan(0.5, 3.5, alpha=0.07, color="orange", label="Early window (d1-3)")
axes[0].set_xlabel("Experiment day"); axes[0].set_ylabel("Acceptance rate (%)")
axes[0].set_title("Daily Acceptance Rate by Variant", fontweight="bold"); axes[0].legend()

axes[1].bar(pivot.index, pivot["lift"]*100, color="#4A90D9", alpha=0.55, label="Daily lift")
roll = pivot["lift"].rolling(3, min_periods=1).mean()
axes[1].plot(pivot.index, roll*100, color="#E07B39", lw=2, marker="o", ms=4, label="3d rolling avg")
axes[1].axhline(early_lift*100, ls="--", lw=1.3, color="orange",
                label=f"Early avg: {early_lift*100:.3f}pp")
axes[1].axhline(steady_lift*100, ls="--", lw=1.3, color="#2E8B57",
                label=f"Steady avg: {steady_lift*100:.3f}pp")
axes[1].set_xlabel("Experiment day"); axes[1].set_ylabel("Lift (pp)")
axes[1].set_title(f"Novelty Effect  |  Ratio={nov_ratio:.3f}  "
                  f"{'⚠ Inflation' if nov_flag else '✅ Stable'}", fontweight="bold")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "phase4d_novelty_trend.png"), bbox_inches="tight")
plt.close()
print(f"  Chart saved → phase4d_novelty_trend.png")

# ── 4e: Business impact ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

baseline_daily   = daily_eligible * cr_c * rev_per_acc
components       = ["Baseline\ndaily rev", "+ Incr.\nacceptances", "= New\ndaily total"]
values           = [baseline_daily, daily_rev_impact, baseline_daily + daily_rev_impact]
colors_wf        = ["#4A90D9","#5CBF7A","#E07B39"]
bars = axes[0].bar(components, [v/1000 for v in values], color=colors_wf,
                   width=0.5, edgecolor="white")
for bar, v in zip(bars, values):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                 f"${v/1000:.1f}K", ha="center", va="bottom",
                 fontweight="bold", fontsize=10)
axes[0].set_ylabel("Daily revenue ($K)")
axes[0].set_title("Daily Revenue Impact Waterfall", fontweight="bold")

x_labels = ["Conservative\n(CI lower)", "Point\nEstimate", "Optimistic\n(CI upper)"]
rev_vals  = [ci_rev_low/1000, daily_rev_impact/1000, ci_rev_high/1000]
bars2 = axes[1].bar(x_labels, rev_vals,
                    color=["#A8C8F0","#4A90D9","#A8C8F0"], width=0.5, edgecolor="white")
for bar, v in zip(bars2, rev_vals):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f"${v:.1f}K/day", ha="center", va="bottom",
                 fontweight="bold", fontsize=10)
axes[1].set_ylabel("Daily revenue impact ($K)")
axes[1].set_title("Revenue 95% Confidence Interval", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "phase4e_business_impact.png"), bbox_inches="tight")
plt.close()
print(f"  Chart saved → phase4e_business_impact.png")
print("\n  Phase 4 complete ✓")
