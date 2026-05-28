# Step-by-Step Walkthrough: Credit Card Limit Offer A/B Test

This guide explains every concept and every file in plain language.
No prior A/B testing experience required.

---

## What is an A/B test?

An A/B test is a randomized experiment where you randomly split your users
into two groups and show each group a different experience:

- **Group A (Control):** sees the existing experience — a generic prompt:
  *"You may be eligible for a higher credit limit. Learn more."*
- **Group B (Treatment):** sees the new experience — a personalized offer:
  *"You're pre-approved for a $8,500 limit increase — accept in one tap."*

You then measure whether Group B accepts the offer at a higher rate.
The key word is *randomly* — random assignment is what lets you attribute
any difference in outcomes to the offer itself, not to pre-existing
differences between the groups.

---

## Why use the UCI Credit Card dataset?

Real A/B test data from banks is confidential and never released publicly.
Instead, we use the UCI dataset (30,000 real Taiwan credit card customers)
to extract realistic statistical parameters: what fraction of users carry
high balances, what the real default rates are, how credit limits are
distributed across different customer types.

We then use those real parameters to *calibrate* our simulated experiment,
so the numbers feel grounded rather than made up.

**Download the real dataset here:**
https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset

Place it in the `data/` folder as `UCI_Credit_Card.csv`.

---

## Project structure explained

```
creditcard-limit-ab/
│
├── data/                         ← All input and generated data files
│   ├── UCI_Credit_Card.csv       ← Real UCI data (download from Kaggle)
│   ├── calibration_params.json   ← Extracted from UCI (created by Phase 1)
│   ├── experiment_config.json    ← Power analysis output (created by Phase 2)
│   ├── ab_experiment_data.csv    ← Simulated experiment (created by Phase 3)
│   └── ab_daily_summary.csv      ← Day-by-day rollup (created by Phase 3)
│
├── notebooks/                    ← Run these in order (01 → 02 → 03 → 04)
│   ├── generate_synthetic_uci.py ← Optional: creates a fake UCI dataset
│   │                               if you don't want to download from Kaggle
│   ├── 01_calibrate.py           ← Phase 1: read UCI, extract real parameters
│   ├── 02_experiment_design.py   ← Phase 2: power analysis, set sample size
│   ├── 03_simulate.py            ← Phase 3: generate the experiment dataset
│   ├── 04_analysis.py            ← Phase 4: run all statistics, produce charts
│   └── run_sql.py                ← Phase 5: run SQL queries locally
│
├── sql/
│   └── ab_queries.sql            ← Production SQL for Snowflake / Redshift
│
├── outputs/
│   ├── figures/                  ← All charts (created by running the notebooks)
│   └── phase4_results_summary.txt← Text summary of results
│
├── requirements.txt              ← Python packages needed
└── README.md                     ← Project overview for GitHub
```

---

## How to run the project

### Option A: With the real Kaggle dataset (recommended)

```bash
# 1. Download UCI Credit Card Default dataset from Kaggle
#    https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
#    Save it as: data/UCI_Credit_Card.csv

# 2. Install Python packages
pip install -r requirements.txt

# 3. Run each phase in order
python3 notebooks/01_calibrate.py
python3 notebooks/02_experiment_design.py
python3 notebooks/03_simulate.py
python3 notebooks/04_analysis.py

# 4. Run SQL locally (optional but impressive for resume)
python3 notebooks/run_sql.py
```

### Option B: Without downloading anything

```bash
pip install -r requirements.txt

# Create a synthetic proxy of the UCI dataset (no Kaggle account needed)
python3 notebooks/generate_synthetic_uci.py

# Then run phases as normal
python3 notebooks/01_calibrate.py
python3 notebooks/02_experiment_design.py
python3 notebooks/03_simulate.py
python3 notebooks/04_analysis.py
python3 notebooks/run_sql.py
```

---

## Phase 1: Calibrate (`01_calibrate.py`)

**What it does:** Reads the real UCI data and extracts statistics.

**Key outputs from Phase 1:**

| Parameter | What it is | Why we need it |
|-----------|-----------|----------------|
| `overall_default` | ~20% of users defaulted | Sets our guardrail threshold |
| `limit_by_util` | Credit limit distribution per tier | Calibrates simulated limit offers |
| `segment_defaults` | Default rate per (util × payment) cell | Grounds simulation in real risk |
| `acceptance_by_segment` | Control acceptance rate per segment | The baseline we want to beat |

**The two segmentation dimensions:**

```
Utilization tier:
  Low  (<30%)   User has plenty of available credit headroom
  Med (30-70%)  User is actively using their credit line
  High  (>70%)  User is near their credit ceiling

Payment behavior:
  On-time        No delayed payments in last 6 months
  1-mo delay     At least one month overdue recently
  2+ mo delay    Chronic payment delays — highest risk
```

**Chart produced:** `outputs/figures/phase1_calibration.png`
- Panel A: Credit limit distributions by tier (high-util users have lower limits)
- Panel B: Real default rates (heat map from UCI data)
- Panel C: Credit utilization distribution across all UCI users
- Panel D: The calibrated acceptance rates our simulation will use

---

## Phase 2: Experiment Design (`02_experiment_design.py`)

**What it does:** Answers the question "How many users do I need?"

**Key concepts:**

```
α (significance level) = 0.05
  If there is NO real effect, we'll still get a "significant" result
  5% of the time just by luck. We accept this false-positive rate.

1-β (statistical power) = 0.80
  If there IS a real effect of our chosen size, we want an 80% chance
  of detecting it. Higher power = more users needed.

MDE (minimum detectable effect) = +1.5pp
  We only care about deploying this feature if it lifts acceptance by
  at least 1.5 percentage points. Smaller effects aren't worth the
  engineering work.

Cohen's h
  A standardized effect size that accounts for the fact that going from
  7.5% → 9.0% is a bigger deal than going from 50% → 51.5%, even though
  both are +1.5pp absolute.
```

**Result:**
```
Required n per variant  : ~5,300 users
Required n total        : ~10,600 users
Daily eligible users    : 8,000/day
Estimated runtime       : 2 days (we over-power to 14 days for stability)
```

**Chart produced:** `outputs/figures/phase2_power_analysis.png`
- Panel A: How sample size grows as MDE shrinks (exponential curve)
- Panel B: Power vs sample size at our chosen MDE
- Panel C: Same curve across three different α levels (0.01, 0.05, 0.10)

---

## Phase 3: Simulate (`03_simulate.py`)

**What it does:** Generates the 22,970-row experiment dataset.

**What each column means:**

| Column | Type | Meaning |
|--------|------|---------|
| `util_tier` | string | User's credit utilization tier |
| `pay_segment` | string | User's payment behavior segment |
| `variant` | string | "control" or "treatment" |
| `accepted` | 0 or 1 | Did the user accept the limit increase offer? |
| `revenue` | float | Annual interchange revenue from this user ($) |
| `monthly_spend` | float | User's total monthly card spend ($) |
| `default_flag` | 0 or 1 | Did this user default on payments? |
| `fraud_flag` | 0 or 1 | Was fraud detected for this user? |
| `limit_offered` | float | Dollar amount of the limit increase offered |
| `exp_day` | int | Day of experiment the user entered (1–14) |

**What SRM validation is:**

After randomly assigning users, we check that we actually got a 50/50 split
using a chi-squared test. If the split is off (e.g. 53/47), something went
wrong in the assignment system — maybe there's a bug in the randomization,
or certain user types were excluded from one variant. An SRM failure means
the experiment is invalid and must be re-run.

```
Chi-squared test result:
  Control:   11,485 users
  Treatment: 11,485 users
  Expected:  11,485 per variant
  χ² = 0.0000   p = 1.0000   ✅ No SRM
```

**Chart produced:** `outputs/figures/phase3_simulation_validation.png`

---

## Phase 4: Analysis (`04_analysis.py`)

**What it does:** Runs all the statistics and produces all charts.

**Step-by-step through the analysis:**

### 4a. Primary metric — Offer acceptance rate

We use a **two-proportion z-test** to compare the acceptance rates:

```
Control:    7.54%   (866 accepted / 11,485 users)
Treatment:  9.10%  (1,045 accepted / 11,485 users)

Lift:       +1.56pp  (+20.7% relative)
95% CI:     [+0.84pp, +2.27pp]
Z-statistic: 4.28
p-value:     0.000019

✅ SIGNIFICANT (p < 0.05)
```

The 95% confidence interval tells you the range of plausible true lifts.
Since [+0.84, +2.27] does not include 0, we're confident the treatment
is genuinely better.

### 4b. Secondary metric — Revenue per user

We use **Mann-Whitney U test** (not a t-test) because revenue data is
non-normal (most users generate $0 revenue; a few generate a lot).
Mann-Whitney is a rank-based test that doesn't assume normality.

We also compute a **bootstrap confidence interval** by resampling the data
5,000 times — this is more robust than a normal approximation for skewed data.

### 4c. Subgroup analysis

We break down the lift by segment. This is pre-registered (meaning we
declared which subgroups we'd look at before seeing the data). Looking at
subgroups after seeing the data is called "p-hacking" and inflates false
positive rates.

**Key finding:** Medium-utilization on-time payers show the strongest lift
(+8.8% relative). These users are actively managing their credit and most
responsive to seeing a specific pre-approved dollar amount.

### 4d. Guardrail checks

Three metrics we've pre-committed to not harm. If any fails, we don't ship.

| Guardrail | Control | Treatment | Δ | Threshold | Result |
|-----------|---------|-----------|---|-----------|--------|
| Default rate | 20.00% | 20.00% | +0.00pp | ≤+0.50pp | ✅ PASS |
| Fraud flag rate | 0.26% | 0.33% | +0.07pp | ≤+0.10pp | ✅ PASS |
| Monthly spend | $92,540 | $94,313 | +1.92% | ≥-5.00% | ✅ PASS |

### 4e. Novelty effect detection

Users sometimes engage more with *any* new UI just because it's different —
not because it's genuinely better. We check for this by comparing the lift
in days 1–3 (early, novelty-prone) vs days 4+ (steady state).

```
Early lift (d1–3):   +1.51pp
Steady lift (d4+):   +1.57pp
Novelty ratio:        0.965   ✅ No inflation (ratio < 1.20)
```

A ratio near 1.0 means the effect is stable over time.

### 4f. Business impact

We project the revenue impact of shipping this feature to all 8,000 daily
eligible users:

```
Lift:                          +1.56pp
Incremental acceptances/day:   ~125 users
Revenue per acceptance:        $75.60
Daily revenue impact:          ~$9,400
95% CI (daily):                [$5,100 – $13,700]
Annual revenue impact:         ~$3.4 million
```

---

## Phase 5: SQL (`ab_queries.sql` + `run_sql.py`)

**What it does:** Implements the same analysis in production-grade SQL,
compatible with Snowflake, Redshift, and BigQuery.

**Run locally:**
```bash
python3 notebooks/run_sql.py          # all 6 sections
python3 notebooks/run_sql.py --section 3   # just the acceptance rate query
```

**The 6 SQL sections:**

| Section | What it does | Key technique |
|---------|-------------|---------------|
| 1 | SRM check | `SUM() OVER ()` window for chi-squared |
| 2 | Cross-contamination check | `HAVING COUNT(DISTINCT variant) > 1` |
| 3 | Acceptance rate by variant + segment | `CASE WHEN` pivot pattern |
| 4 | Guardrail metrics | Self-join to compare control vs treatment |
| 5 | Cumulative trend + novelty detection | `ROWS UNBOUNDED PRECEDING`, `LAG()` |
| 6 | Revenue projection | Revenue arithmetic at full platform scale |

---

## How to push to GitHub

```bash
# 1. Create a new repo on github.com (name it: creditcard-limit-ab)

# 2. In your terminal:
cd creditcard-limit-ab
git init
git add .
git commit -m "Initial commit: Credit card limit increase A/B test"
git remote add origin https://github.com/YOUR_USERNAME/creditcard-limit-ab.git
git push -u origin main
```

**Important:** The `data/UCI_Credit_Card.csv` file is excluded from git
(it's too large and you don't own it). The `.gitignore` handles this.
Anyone who clones the repo can run `generate_synthetic_uci.py` to get started.

---

## How to talk about this project in interviews

**For a "tell me about a project" question:**

> "I built an end-to-end A/B test analyzing a personalized credit limit offer
> for a credit card issuer. I used the UCI Credit Card dataset to calibrate
> realistic user segments and acceptance rates. The experiment tested whether
> showing users a specific pre-approved dollar amount — say '$8,500 increase' —
> outperforms a generic prompt. I found a statistically significant +1.56pp lift
> in acceptance rate (p < 0.001), all three pre-registered guardrails passed
> including no increase in default rate, and the business impact translated to
> ~$9,400/day in incremental interchange revenue. I also built production-grade
> SQL covering SRM checks, cumulative trend analysis, and novelty effect detection."

**Common follow-up questions and answers:**

Q: Why did you use a two-sided test instead of one-sided?
A: "Pre-registration discipline. Before seeing the data, we don't actually know
   that the treatment will be better — it could also be worse. A two-sided test
   is more conservative and protects against shipping something harmful."

Q: What is an SRM and why does it matter?
A: "Sample Ratio Mismatch — when your actual variant split differs significantly
   from your intended 50/50. It usually means there's a bug in the assignment
   system, and it invalidates the experiment because the groups are no longer
   comparable."

Q: Why Mann-Whitney instead of t-test for revenue?
A: "Revenue is highly right-skewed: most users generate zero or near-zero
   incremental revenue, but a few generate a lot. The t-test assumes
   approximately normal distributions, which doesn't hold here. Mann-Whitney
   is a non-parametric test that works on ranks rather than raw values."

Q: What is a novelty effect and how did you test for it?
A: "Users sometimes engage more with any new UI just because it's new, not
   because it's genuinely better. This inflates early-period lift. We tested
   for it by comparing lift in days 1–3 vs days 4–14. A ratio above 1.20 would
   be a flag. Our ratio was 0.96, so the effect was stable."
