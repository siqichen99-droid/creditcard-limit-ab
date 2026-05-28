"""
generate_synthetic_uci.py
==========================
Generates a synthetic dataset matching key distributional properties
of the UCI Credit Card Default dataset. Allows the project to run
end-to-end without downloading from Kaggle.

Column names mirror the real UCI dataset exactly so 01_calibrate.py
works with either the real or synthetic version.

Real dataset reference:
  https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
"""

import os
import numpy as np
import pandas as pd

np.random.seed(0)

ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

N = 30_000  # matches real dataset size

# ── demographics (calibrated from real UCI distributions) ──────────────────────
sex        = np.random.choice([1, 2], N, p=[0.396, 0.604])
education  = np.random.choice([1, 2, 3, 4], N, p=[0.352, 0.468, 0.164, 0.016])
marriage   = np.random.choice([1, 2, 3], N, p=[0.455, 0.534, 0.011])
age        = np.clip(np.random.normal(35.5, 9.2, N).astype(int), 21, 79)

# ── credit limit — log-normal, calibrated to real mean ~$167K NT ──────────────
limit_bal  = np.random.lognormal(np.log(167_500), 0.75, N).clip(10_000, 1_000_000)
limit_bal  = (np.round(limit_bal / 10_000) * 10_000).astype(int)

# ── payment status (PAY_0 … PAY_6): -2,-1=paid duly, 0=min, 1-9=months delayed
pay_probs  = [0.28, 0.31, 0.18, 0.12, 0.05, 0.03, 0.02, 0.01]
pay_vals   = [-2, -1, 0, 1, 2, 3, 4, 5]

PAY = {}
for m in range(7):
    PAY[f"PAY_{m}"] = np.random.choice(pay_vals, N, p=pay_probs)

# ── bill amounts: correlated log-normal, proportional to limit ────────────────
BILL = {}
util_factor = np.random.beta(2, 3, N)  # overall utilization draw
for i, m in enumerate(range(1, 7)):
    noise = np.random.lognormal(0, 0.25, N)
    BILL[f"BILL_AMT{m}"] = np.maximum(0,
        (limit_bal * util_factor * noise * np.exp(-i * 0.05)).astype(int))

# ── payment amounts ────────────────────────────────────────────────────────────
PAY_AMT = {}
for i, m in enumerate(range(1, 7)):
    pay_frac = np.random.beta(1.5, 4, N)  # fraction of bill paid
    PAY_AMT[f"PAY_AMT{m}"] = np.maximum(0,
        (BILL[f"BILL_AMT{m}"] * pay_frac).astype(int))

# ── default label: logistic function of utilization + payment delay ────────────
log_odds = (
    -1.80
    + 0.90  * util_factor
    + 0.55  * (PAY["PAY_0"] > 1).astype(float)
    + 0.35  * (PAY["PAY_1"] > 1).astype(float)
    - 0.30  * (limit_bal / 500_000)
    + np.random.normal(0, 0.5, N)
)
prob_default = 1 / (1 + np.exp(-log_odds))
default      = np.random.binomial(1, prob_default.clip(0.01, 0.99), N)

# ── assemble ───────────────────────────────────────────────────────────────────
df = pd.DataFrame({
    "ID":          range(1, N+1),
    "LIMIT_BAL":   limit_bal,
    "SEX":         sex,
    "EDUCATION":   education,
    "MARRIAGE":    marriage,
    "AGE":         age,
    **PAY,
    **BILL,
    **PAY_AMT,
    "DEFAULT.PAYMENT.NEXT.MONTH": default,
})

out_path = os.path.join(DATA_DIR, "UCI_Credit_Card.csv")
df.to_csv(out_path, index=False)

print(f"  Synthetic UCI dataset → {out_path}")
print(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Default rate : {df['DEFAULT.PAYMENT.NEXT.MONTH'].mean():.2%}  "
      f"(real ≈ 22.12%)")
print(f"  Avg LIMIT_BAL: ${df['LIMIT_BAL'].mean():,.0f} NT  "
      f"(real ≈ $167,484 NT)")
print(f"\n  Note: Replace with the real UCI dataset for fully calibrated results.")
print(f"  https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset")
