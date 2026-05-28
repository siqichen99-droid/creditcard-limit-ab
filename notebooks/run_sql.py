"""
run_sql.py
===========
Runs all SQL sections locally against the simulated experiment data
using SQLite (no warehouse needed). Outputs results to stdout.

Usage:
    python3 notebooks/run_sql.py              # all sections
    python3 notebooks/run_sql.py --section 3  # specific section
"""

import os, sys, argparse
import pandas as pd
import sqlite3

ROOT     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "..", "data")
DB_PATH  = os.path.join(DATA_DIR, "experiment.db")

# ── load CSVs into SQLite ──────────────────────────────────────────────────────
def build_db():
    df = pd.read_csv(os.path.join(DATA_DIR, "ab_experiment_data.csv"))
    # Build experiment_assignments and user_events from flat file
    import numpy as np
    np.random.seed(42)
    n = len(df)
    df["user_id"]       = [f"U{i:06d}" for i in range(n)]
    df["experiment_id"] = "credit_limit_offer_v1"
    df["assigned_at"]   = pd.to_datetime("2024-01-01") + pd.to_timedelta(
                              df["exp_day"] - 1, unit="D")
    # event_ts = assigned_at + random hours
    df["event_ts"]      = df["assigned_at"] + pd.to_timedelta(
                              np.random.randint(1, 24, n), unit="h")

    assignments = df[["user_id","variant","util_tier","pay_segment",
                       "experiment_id","assigned_at"]].copy()

    events_rows = []
    for _, row in df.iterrows():
        # accepted event
        if row["accepted"] == 1:
            events_rows.append({"user_id": row["user_id"],
                                 "event_type": "offer_accepted",
                                 "event_ts": row["event_ts"],
                                 "revenue": row["revenue"],
                                 "monthly_spend": row["monthly_spend"],
                                 "default_flag": row["default_flag"],
                                 "fraud_flag": row["fraud_flag"]})
        # always log monthly spend
        events_rows.append({"user_id": row["user_id"],
                             "event_type": "session",
                             "event_ts": row["event_ts"],
                             "revenue": 0,
                             "monthly_spend": row["monthly_spend"],
                             "default_flag": row["default_flag"],
                             "fraud_flag": row["fraud_flag"]})
        # fraud / default events
        if row["fraud_flag"] == 1:
            events_rows.append({"user_id": row["user_id"],
                                 "event_type": "fraud_flag",
                                 "event_ts": row["event_ts"],
                                 "revenue": 0,
                                 "monthly_spend": row["monthly_spend"],
                                 "default_flag": 0, "fraud_flag": 1})
        if row["default_flag"] == 1:
            events_rows.append({"user_id": row["user_id"],
                                 "event_type": "default",
                                 "event_ts": row["event_ts"],
                                 "revenue": 0,
                                 "monthly_spend": row["monthly_spend"],
                                 "default_flag": 1, "fraud_flag": 0})

    events = pd.DataFrame(events_rows)

    con = sqlite3.connect(DB_PATH)
    assignments.to_sql("experiment_assignments", con, if_exists="replace", index=False)
    events.to_sql("user_events", con, if_exists="replace", index=False)
    con.close()
    print(f"  SQLite DB built → {DB_PATH}")
    print(f"  experiment_assignments: {len(assignments):,} rows")
    print(f"  user_events           : {len(events):,} rows")
    return assignments, events

# ── SQL adapted for SQLite ──────────────────────────────────────────────────────
QUERIES = {

    1: ("SRM Check", """
        WITH counts AS (
            SELECT variant, COUNT(DISTINCT user_id) AS n
            FROM experiment_assignments
            WHERE experiment_id='credit_limit_offer_v1'
            GROUP BY variant
        ),
        total AS (SELECT SUM(n) AS n_total FROM counts),
        chi2 AS (
            SELECT c.variant, c.n, t.n_total,
                   ROUND(c.n * 100.0 / t.n_total, 2) AS pct,
                   ROUND(((c.n - t.n_total/2.0)*(c.n - t.n_total/2.0))
                         / (t.n_total/2.0), 4) AS chi2_component
            FROM counts c, total t
        )
        SELECT variant, n AS n_users, pct AS pct_of_total,
               chi2_component,
               SUM(chi2_component) OVER () AS chi2_total,
               CASE WHEN SUM(chi2_component) OVER () > 6.635
                    THEN 'SRM DETECTED' ELSE 'No SRM' END AS srm_verdict
        FROM chi2 ORDER BY variant
    """),

    2: ("Cross-contamination Check", """
        SELECT user_id, COUNT(DISTINCT variant) AS n_variants
        FROM experiment_assignments
        WHERE experiment_id='credit_limit_offer_v1'
        GROUP BY user_id
        HAVING COUNT(DISTINCT variant) > 1
        LIMIT 10
    """),

    3: ("Primary Metric — Acceptance Rate", """
        WITH ul AS (
            SELECT a.user_id, a.variant,
                   MAX(CASE WHEN u.event_type='offer_accepted' THEN 1 ELSE 0 END) AS accepted,
                   SUM(CASE WHEN u.event_type='offer_accepted'
                            THEN COALESCE(u.revenue,0) ELSE 0 END) AS revenue
            FROM experiment_assignments a
            LEFT JOIN user_events u ON a.user_id=u.user_id
            WHERE a.experiment_id='credit_limit_offer_v1'
            GROUP BY a.user_id, a.variant
        )
        SELECT variant,
               COUNT(*) AS n_users,
               SUM(accepted) AS n_accepted,
               ROUND(AVG(accepted)*100, 4) AS acceptance_rate_pct,
               ROUND(AVG(revenue), 4) AS revenue_per_user
        FROM ul GROUP BY variant ORDER BY variant
    """),

    4: ("Segment Analysis — Util × Pay", """
        WITH ul AS (
            SELECT a.user_id, a.variant, a.util_tier, a.pay_segment,
                   MAX(CASE WHEN u.event_type='offer_accepted' THEN 1 ELSE 0 END) AS accepted
            FROM experiment_assignments a
            LEFT JOIN user_events u ON a.user_id=u.user_id
            WHERE a.experiment_id='credit_limit_offer_v1'
            GROUP BY a.user_id, a.variant, a.util_tier, a.pay_segment
        )
        SELECT util_tier, pay_segment,
               ROUND(MAX(CASE WHEN variant='control'   THEN 100.0*SUM(accepted)/COUNT(*) END) OVER
                     (PARTITION BY util_tier, pay_segment), 4) AS ctrl_pct,
               ROUND(MAX(CASE WHEN variant='treatment' THEN 100.0*SUM(accepted)/COUNT(*) END) OVER
                     (PARTITION BY util_tier, pay_segment), 4) AS treat_pct
        FROM ul GROUP BY util_tier, pay_segment, variant
        HAVING COUNT(*) >= 20
        ORDER BY util_tier, pay_segment, variant
        LIMIT 30
    """),

    5: ("Guardrails", """
        WITH ul AS (
            SELECT a.user_id, a.variant,
                   MAX(CASE WHEN u.event_type='default'    THEN 1 ELSE 0 END) AS default_flag,
                   MAX(CASE WHEN u.event_type='fraud_flag' THEN 1 ELSE 0 END) AS fraud_flag,
                   AVG(u.monthly_spend) AS avg_spend
            FROM experiment_assignments a
            LEFT JOIN user_events u ON a.user_id=u.user_id
            WHERE a.experiment_id='credit_limit_offer_v1'
            GROUP BY a.user_id, a.variant
        )
        SELECT variant,
               ROUND(AVG(default_flag)*100, 4) AS default_rate_pct,
               ROUND(AVG(fraud_flag)*100, 4)   AS fraud_rate_pct,
               ROUND(AVG(avg_spend), 2)         AS avg_monthly_spend
        FROM ul GROUP BY variant ORDER BY variant
    """),

    6: ("Cumulative Trend", """
        WITH daily AS (
            SELECT a.variant,
                   CAST(julianday(u.event_ts) - julianday(a.assigned_at) AS INT) + 1 AS exp_day,
                   COUNT(DISTINCT a.user_id) AS daily_users,
                   COUNT(DISTINCT CASE WHEN u.event_type='offer_accepted'
                                       THEN a.user_id END) AS daily_accepted
            FROM experiment_assignments a
            LEFT JOIN user_events u ON a.user_id=u.user_id
            WHERE a.experiment_id='credit_limit_offer_v1'
              AND CAST(julianday(u.event_ts) - julianday(a.assigned_at) AS INT) BETWEEN 0 AND 29
            GROUP BY a.variant,
                     CAST(julianday(u.event_ts) - julianday(a.assigned_at) AS INT) + 1
        )
        SELECT variant, exp_day, daily_users, daily_accepted,
               ROUND(daily_accepted * 100.0 / MAX(daily_users, 1), 4) AS daily_rate_pct,
               SUM(daily_users)    OVER (PARTITION BY variant ORDER BY exp_day) AS cum_users,
               SUM(daily_accepted) OVER (PARTITION BY variant ORDER BY exp_day) AS cum_accepted
        FROM daily
        ORDER BY variant, exp_day
        LIMIT 40
    """),
}

# ── runner ─────────────────────────────────────────────────────────────────────
def run(section=None):
    if not os.path.exists(DB_PATH):
        build_db()

    con = sqlite3.connect(DB_PATH)
    sections = [section] if section else list(QUERIES.keys())

    for s in sections:
        title, sql = QUERIES[s]
        print(f"\n{'='*60}")
        print(f"  SECTION {s}: {title}")
        print(f"{'='*60}")
        try:
            result = pd.read_sql_query(sql, con)
            if result.empty:
                print("  (no rows returned)")
            else:
                print(result.to_string(index=False))
        except Exception as e:
            print(f"  Error: {e}")

    con.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", type=int, default=None,
                        help="Run a specific section (1-6)")
    args = parser.parse_args()
    run(args.section)
