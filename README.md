# Credit Card Limit Increase Offer — A/B Test

**Author:** Siqi Chen &nbsp;|&nbsp; **Domain:** Fintech / Consumer Credit  
**Skills:** Experiment design · Power analysis · Python · SQL · Statistical inference · Business impact

---

## Business Context

A major credit card issuer wants to increase card engagement and interchange revenue
by prompting eligible cardholders to accept a pre-approved credit limit increase.
Two offer formats are compared:

| Variant | Display |
|---------|---------|
| **Control** | Generic: *"You may be eligible for a higher credit limit. Learn more."* |
| **Treatment** | Personalized: *"You're pre-approved for a $8,500 limit increase — accept in one tap."* |

**Hypothesis:** Showing a specific, pre-approved dollar amount reduces friction and
increases offer acceptance rate, driving incremental interchange revenue.

---

## Data Source

Experiment parameters are **calibrated from the real UCI Credit Card Default dataset**
(30,000 Taiwan credit card clients, 2005) using real distributions of:

- Credit utilization rates (avg bill / credit limit)
- Payment behavior (on-time vs delayed)
- Default rates per behavioral segment
- Credit limit distributions

**Run with real Kaggle data (recommended):**
1. Download from [Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
2. Place as `data/UCI_Credit_Card.csv`
3. Skip `generate_synthetic_uci.py` — `01_calibrate.py` reads the real file directly

**Run without Kaggle (synthetic fallback):**  
`python3 notebooks/generate_synthetic_uci.py` — generates a distributional proxy that
mirrors key UCI statistics, so the full pipeline runs offline.

---

## Repository Structure

```
creditcard-limit-ab/
├── data/
│   ├── UCI_Credit_Card.csv           ← Real UCI data (not committed — download from Kaggle)
│   ├── calibration_params.json       ← Extracted distributional parameters
│   ├── experiment_config.json        ← Power analysis output
│   ├── ab_experiment_data.csv        ← Simulated experiment (22,970 users)
│   └── ab_daily_summary.csv          ← Daily rollup by variant
├── notebooks/
│   ├── generate_synthetic_uci.py     ← Optional: synthetic UCI fallback
│   ├── 01_calibrate.py               ← Extract distributions from real data
│   ├── 02_experiment_design.py       ← Power analysis + config
│   ├── 03_simulate.py                ← Stratified experiment simulation
│   ├── 04_analysis.py                ← Stats tests, guardrails, charts, business impact
│   └── run_sql.py                    ← Run all SQL sections locally via SQLite
├── sql/
│   └── ab_queries.sql                ← Production SQL (Snowflake / Redshift / BigQuery)
├── outputs/
│   ├── figures/                      ← Generated charts
│   └── phase4_results_summary.txt
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place UCI data (or generate synthetic)
#    Option A: download real data to data/UCI_Credit_Card.csv
#    Option B: python3 notebooks/generate_synthetic_uci.py

# 3. Run all phases in order
python3 notebooks/01_calibrate.py          # extract UCI distributions
python3 notebooks/02_experiment_design.py  # power analysis
python3 notebooks/03_simulate.py           # simulate experiment
python3 notebooks/04_analysis.py           # full analysis + charts

# 4. Run SQL queries locally
python3 notebooks/run_sql.py               # all sections
python3 notebooks/run_sql.py --section 3   # specific section
```

---

## Experiment Design

| Parameter | Value |
|-----------|-------|
| Primary metric | Offer acceptance rate |
| Baseline (control) | 7.54% |
| Target (treatment) | 9.14% (+1.6pp MDE) |
| Significance level α | 0.05 |
| Statistical power 1−β | 80% |
| Required n per variant | 5,279 |
| Actual n per variant | 11,485 (over-powered for stability) |
| Runtime | 14 days |
| Daily eligible users | 8,000 |

**Stratification:** 3 utilization tiers × 3 payment segments = 9 strata  
Segment definitions grounded in real UCI behavioral distributions.

**Pre-registered guardrails:**

| Guardrail | Threshold | Rationale |
|-----------|-----------|-----------|
| Default rate | ≤ +0.50pp vs control | Approval of higher limit must not attract riskier behaviour |
| Fraud flag rate | ≤ +0.10pp vs control | Personalized offers must not drive identity fraud |
| Avg monthly spend | ≥ −5.00% vs control | New limit must not suppress regular spend |

---

## Key Results

```
PRIMARY METRIC: Offer Acceptance Rate
  Control          7.54%   (866 / 11,485)
  Treatment        9.10%   (1,045 / 11,485)
  Absolute lift    +1.56pp
  Relative lift    +20.7%
  95% CI           [+0.84pp, +2.27pp]
  Z-statistic      4.28
  p-value          0.000019  ✅ SIGNIFICANT

SECONDARY: Revenue per User (interchange)
  Control          $6.39      Treatment  $7.36
  Lift             +$0.97    Bootstrap CI [$+0.04, $+1.96]
  Mann-Whitney p   0.000031  ✅ SIGNIFICANT

GUARDRAILS
  Default rate     Δ = 0.00pp   ✅ PASS  (threshold ≤ +0.50pp)
  Fraud flag rate  Δ = +0.07pp  ✅ PASS  (threshold ≤ +0.10pp)
  Monthly spend    Δ = +1.92%   ✅ PASS  (threshold ≥ −5.00%)

NOVELTY EFFECT
  Early lift (d1–3)   +1.51pp
  Steady lift (d4+)   +1.57pp
  Novelty ratio       0.965     ✅ No inflation

BUSINESS IMPACT
  Daily eligible users         8,000
  Incremental acceptances/day    ~125
  Revenue per acceptance       $75.60  (annual interchange × spread)
  Daily revenue impact         ~$9,400
  95% CI (daily)               [$5,100 – $13,700]
  Annual revenue impact        ~$3.4M
```

**Recommendation: SHIP — full rollout approved.**  
Post-launch: monitor default rate and fraud flag rate for 14 days.

---

## Subgroup Analysis — Key Insight

The lift is **concentrated in medium-utilization, on-time payers** (+8.8pp relative).
This segment is actively managing their credit and most responsive to a transparent
pre-approved dollar amount. Low-utilization users respond moderately; high-utilization
late payers show limited response (consistent with risk aversion on both sides).

**Follow-on experiments:**

| ID | Test | Segment | Hypothesis |
|----|------|---------|------------|
| EXP-2 | Monthly payment impact display | Medium-util, on-time | +0.30–0.50pp |
| EXP-3 | "Rate lock 30 days" messaging | Medium-util, 1-mo delay | Reduce drop-off anxiety |
| EXP-4 | Approval probability display | High-util, 2+ mo delay | Replace rate fear with confidence |
| EXP-5 | Push notification vs in-app offer | All segments | Channel allocation |

---

## SQL Highlights

All queries in `sql/ab_queries.sql` are **Snowflake / Redshift / BigQuery compatible**.  
Run locally via `run_sql.py` using SQLite.

**SRM χ² check (Section 1)**
```sql
POWER(n - n_total / 2.0, 2) / (n_total / 2.0)  AS chi2_component,
SUM(chi2_component) OVER ()                      AS chi2_total
```

**Cumulative acceptance trend (Section 5)**
```sql
SUM(daily_accepted)
    OVER (PARTITION BY variant ORDER BY exp_day
          ROWS UNBOUNDED PRECEDING)              AS cum_accepted
```

**7-day novelty detection (Section 5)**
```sql
AVG(daily_accepted * 1.0 / NULLIF(daily_users, 0))
    OVER (PARTITION BY variant ORDER BY exp_day
          ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS rate_3d_rolling
```

---

## Concepts Demonstrated

| Concept | Where |
|---------|-------|
| Real-data calibration for simulation | `01_calibrate.py` |
| Cohen's h, NormalIndPower, power curves | `02_experiment_design.py` |
| Stratified simulation (9 strata) | `03_simulate.py` |
| SRM χ² test + per-stratum balance | `03_simulate.py`, SQL §1 |
| Two-proportion z-test + 95% CI | `04_analysis.py` |
| Mann-Whitney U + bootstrap CI | `04_analysis.py` |
| Pre-specified subgroup analysis (anti-p-hacking) | `04_analysis.py`, SQL §3 |
| Three guardrail checks (pre-registered) | `04_analysis.py`, SQL §4 |
| Novelty-effect ratio | `04_analysis.py`, SQL §5 |
| Revenue waterfall + CI band | `04_analysis.py` |
| Window functions (cumulative, rolling, LAG) | SQL §5, §6 |
| Local SQL validation via SQLite | `run_sql.py` |

---

## Stack

`Python 3.10+` · `pandas` · `numpy` · `scipy` · `statsmodels` · `matplotlib` · `seaborn`  
`SQL` — Snowflake / Redshift / BigQuery compatible · `SQLite` for local validation

---

## Author

**Siqi Chen** — Data Analyst / Product Analyst / Data Scientist  
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourhandle)
