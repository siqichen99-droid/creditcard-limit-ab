-- ================================================================
-- ab_queries.sql
-- Credit Card Limit Increase A/B Test — Production SQL Suite
-- Dialect : Snowflake / Redshift / BigQuery compatible
-- Author  : Siqi Chen
--
-- Tables assumed:
--   experiment_assignments  (user_id, variant, assigned_at,
--                            util_tier, pay_segment, experiment_id)
--   user_events             (user_id, event_type, event_ts,
--                            revenue, monthly_spend, default_flag,
--                            fraud_flag, limit_offered)
-- ================================================================


-- ────────────────────────────────────────────────────────────────
-- SECTION 1 — DATA QUALITY: SRM CHECK
-- Detects sample ratio mismatch before any metric read-out.
-- A χ² stat > 6.635 = p < 0.01 → halt analysis.
-- ────────────────────────────────────────────────────────────────

WITH counts AS (
    SELECT
        variant,
        COUNT(DISTINCT user_id)                                     AS n
    FROM experiment_assignments
    WHERE experiment_id = 'credit_limit_offer_v1'
    GROUP BY variant
),
total AS (SELECT SUM(n) AS n_total FROM counts),
chi2 AS (
    SELECT
        c.variant,
        c.n,
        t.n_total,
        ROUND(c.n * 100.0 / t.n_total, 2)                          AS pct,
        ROUND(POWER(c.n - t.n_total / 2.0, 2)
              / (t.n_total / 2.0), 4)                               AS chi2_component
    FROM counts c CROSS JOIN total t
)
SELECT
    variant,
    n                                                               AS n_users,
    pct                                                             AS pct_of_total,
    chi2_component,
    SUM(chi2_component) OVER ()                                     AS chi2_total,
    CASE
        WHEN SUM(chi2_component) OVER () > 6.635
        THEN '🚨 SRM DETECTED (p<0.01) — halt'
        WHEN SUM(chi2_component) OVER () > 3.841
        THEN '⚠  SRM SUSPECTED (p<0.05)'
        ELSE '✅ No SRM'
    END                                                             AS srm_verdict
FROM chi2
ORDER BY variant;


-- ────────────────────────────────────────────────────────────────
-- SECTION 2 — DATA QUALITY: CROSS-CONTAMINATION CHECK
-- Flags users who appeared in both variants (invalid exposure).
-- ────────────────────────────────────────────────────────────────

SELECT
    user_id,
    COUNT(DISTINCT variant)                                         AS n_variants,
    LISTAGG(DISTINCT variant, ', ')
        WITHIN GROUP (ORDER BY variant)                             AS variants_seen
FROM experiment_assignments
WHERE experiment_id = 'credit_limit_offer_v1'
GROUP BY user_id
HAVING COUNT(DISTINCT variant) > 1
ORDER BY user_id
LIMIT 100;


-- ────────────────────────────────────────────────────────────────
-- SECTION 3 — PRIMARY METRIC: OFFER ACCEPTANCE RATE
-- Overall and by utilization tier × payment segment.
-- ────────────────────────────────────────────────────────────────

-- 3a. Overall acceptance rate
WITH user_level AS (
    SELECT
        a.user_id,
        a.variant,
        a.util_tier,
        a.pay_segment,
        MAX(CASE WHEN u.event_type = 'offer_accepted'
                 AND u.event_ts BETWEEN a.assigned_at
                             AND DATEADD('day', 30, a.assigned_at)
                 THEN 1 ELSE 0 END)                                 AS accepted,
        SUM(CASE WHEN u.event_type = 'interchange_revenue'
                 THEN COALESCE(u.revenue, 0) ELSE 0 END)            AS revenue,
        AVG(u.monthly_spend)                                        AS avg_monthly_spend,
        MAX(CASE WHEN u.event_type = 'default'
                 THEN 1 ELSE 0 END)                                 AS default_flag,
        MAX(CASE WHEN u.event_type = 'fraud_flag'
                 THEN 1 ELSE 0 END)                                 AS fraud_flag
    FROM experiment_assignments a
    LEFT JOIN user_events u ON a.user_id = u.user_id
    WHERE a.experiment_id = 'credit_limit_offer_v1'
    GROUP BY a.user_id, a.variant, a.util_tier, a.pay_segment
)
SELECT
    variant,
    COUNT(*)                                                        AS n_users,
    SUM(accepted)                                                   AS n_accepted,
    ROUND(AVG(accepted) * 100, 4)                                   AS acceptance_rate_pct,
    ROUND(AVG(revenue), 4)                                          AS revenue_per_user,
    ROUND(AVG(avg_monthly_spend), 2)                                AS avg_monthly_spend,
    ROUND(AVG(default_flag) * 100, 4)                               AS default_rate_pct,
    ROUND(AVG(fraud_flag) * 100, 4)                                 AS fraud_rate_pct
FROM user_level
GROUP BY variant
ORDER BY variant;


-- 3b. Segment-level acceptance — util_tier × pay_segment
WITH user_level AS (
    SELECT
        a.user_id, a.variant, a.util_tier, a.pay_segment,
        MAX(CASE WHEN u.event_type = 'offer_accepted' THEN 1 ELSE 0 END) AS accepted
    FROM experiment_assignments a
    LEFT JOIN user_events u ON a.user_id = u.user_id
    WHERE a.experiment_id = 'credit_limit_offer_v1'
    GROUP BY a.user_id, a.variant, a.util_tier, a.pay_segment
),
agg AS (
    SELECT
        util_tier, pay_segment, variant,
        COUNT(*)          AS n,
        SUM(accepted)     AS k,
        AVG(accepted)     AS accept_rate
    FROM user_level
    GROUP BY util_tier, pay_segment, variant
    HAVING COUNT(*) >= 30   -- suppress underpowered cells
)
SELECT
    util_tier,
    pay_segment,
    MAX(CASE WHEN variant = 'control'
             THEN ROUND(accept_rate * 100, 4) END)                  AS ctrl_rate_pct,
    MAX(CASE WHEN variant = 'treatment'
             THEN ROUND(accept_rate * 100, 4) END)                  AS treat_rate_pct,
    ROUND(
        MAX(CASE WHEN variant='treatment' THEN accept_rate END)
        - MAX(CASE WHEN variant='control'  THEN accept_rate END)
    , 4) * 100                                                      AS lift_pp,
    ROUND(
      (MAX(CASE WHEN variant='treatment' THEN accept_rate END)
       - MAX(CASE WHEN variant='control'  THEN accept_rate END))
      / NULLIF(MAX(CASE WHEN variant='control' THEN accept_rate END), 0)
    * 100, 2)                                                       AS lift_rel_pct
FROM agg
GROUP BY util_tier, pay_segment
ORDER BY
    CASE util_tier
        WHEN 'High (>70%)'      THEN 1
        WHEN 'Medium (30–70%)'  THEN 2
        ELSE 3 END,
    CASE pay_segment
        WHEN 'On-time'     THEN 1
        WHEN '1-mo delay'  THEN 2
        ELSE 3 END;


-- ────────────────────────────────────────────────────────────────
-- SECTION 4 — GUARDRAILS: DEFAULT RATE, FRAUD RATE, SPEND
-- Pre-registered thresholds:
--   default rate lift  ≤ +0.50pp
--   fraud rate lift    ≤ +0.10pp
--   monthly spend      ≥ -5.00%
-- ────────────────────────────────────────────────────────────────

WITH user_level AS (
    SELECT
        a.user_id, a.variant,
        MAX(CASE WHEN u.event_type = 'default'    THEN 1 ELSE 0 END) AS default_flag,
        MAX(CASE WHEN u.event_type = 'fraud_flag' THEN 1 ELSE 0 END) AS fraud_flag,
        AVG(u.monthly_spend)                                           AS avg_monthly_spend
    FROM experiment_assignments a
    LEFT JOIN user_events u ON a.user_id = u.user_id
    WHERE a.experiment_id = 'credit_limit_offer_v1'
    GROUP BY a.user_id, a.variant
),
summary AS (
    SELECT
        variant,
        ROUND(AVG(default_flag) * 100, 4)                           AS default_rate_pct,
        ROUND(AVG(fraud_flag) * 100, 4)                             AS fraud_rate_pct,
        ROUND(AVG(avg_monthly_spend), 2)                            AS avg_spend
    FROM user_level
    GROUP BY variant
)
SELECT
    s.variant,
    s.default_rate_pct,
    s.fraud_rate_pct,
    s.avg_spend,
    -- Default rate guardrail
    ROUND(s.default_rate_pct
          - MAX(CASE WHEN s2.variant='control' THEN s2.default_rate_pct END) OVER (), 4) AS default_lift_pp,
    CASE WHEN (s.default_rate_pct
               - MAX(CASE WHEN s2.variant='control' THEN s2.default_rate_pct END) OVER ()) > 0.50
         THEN '🚨 BREACH' ELSE '✅ OK' END                         AS default_guardrail,
    -- Fraud guardrail
    ROUND(s.fraud_rate_pct
          - MAX(CASE WHEN s2.variant='control' THEN s2.fraud_rate_pct END) OVER (), 4) AS fraud_lift_pp,
    CASE WHEN (s.fraud_rate_pct
               - MAX(CASE WHEN s2.variant='control' THEN s2.fraud_rate_pct END) OVER ()) > 0.10
         THEN '🚨 BREACH' ELSE '✅ OK' END                         AS fraud_guardrail,
    -- Spend guardrail
    ROUND(
        (s.avg_spend
         - MAX(CASE WHEN s2.variant='control' THEN s2.avg_spend END) OVER ())
        / NULLIF(MAX(CASE WHEN s2.variant='control' THEN s2.avg_spend END) OVER (), 0) * 100
    , 2)                                                             AS spend_pct_chg,
    CASE WHEN (
        (s.avg_spend - MAX(CASE WHEN s2.variant='control' THEN s2.avg_spend END) OVER ())
        / NULLIF(MAX(CASE WHEN s2.variant='control' THEN s2.avg_spend END) OVER (), 0) * 100
    ) < -5.0 THEN '🚨 BREACH' ELSE '✅ OK' END                    AS spend_guardrail
FROM summary s
CROSS JOIN summary s2   -- self-join trick to pivot control values into every row
GROUP BY s.variant, s.default_rate_pct, s.fraud_rate_pct, s.avg_spend
ORDER BY s.variant;


-- ────────────────────────────────────────────────────────────────
-- SECTION 5 — CUMULATIVE TREND + NOVELTY EFFECT DETECTION
-- Running acceptance rate using window functions.
-- Novelty ratio > 1.20 → recommend extending runtime.
-- ────────────────────────────────────────────────────────────────

-- 5a. Day-over-day cumulative acceptance rate
WITH daily AS (
    SELECT
        a.variant,
        DATEDIFF('day', a.assigned_at, u.event_ts) + 1             AS exp_day,
        COUNT(DISTINCT a.user_id)                                   AS daily_users,
        COUNT(DISTINCT CASE WHEN u.event_type = 'offer_accepted'
                            THEN a.user_id END)                     AS daily_accepted
    FROM experiment_assignments a
    LEFT JOIN user_events u ON a.user_id = u.user_id
    WHERE a.experiment_id = 'credit_limit_offer_v1'
      AND DATEDIFF('day', a.assigned_at, u.event_ts) BETWEEN 0 AND 29
    GROUP BY a.variant, DATEDIFF('day', a.assigned_at, u.event_ts) + 1
),
cumulative AS (
    SELECT
        variant, exp_day,
        daily_users, daily_accepted,
        ROUND(daily_accepted * 100.0 / NULLIF(daily_users, 0), 4)  AS daily_rate_pct,
        SUM(daily_users)
            OVER (PARTITION BY variant ORDER BY exp_day
                  ROWS UNBOUNDED PRECEDING)                         AS cum_users,
        SUM(daily_accepted)
            OVER (PARTITION BY variant ORDER BY exp_day
                  ROWS UNBOUNDED PRECEDING)                         AS cum_accepted,
        daily_accepted
            - LAG(daily_accepted, 1)
                OVER (PARTITION BY variant ORDER BY exp_day)        AS accepted_dod_delta,
        AVG(daily_accepted * 1.0 / NULLIF(daily_users, 0))
            OVER (PARTITION BY variant ORDER BY exp_day
                  ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)         AS rate_3d_rolling
    FROM daily
)
SELECT
    variant, exp_day,
    daily_users, daily_accepted, daily_rate_pct,
    cum_users, cum_accepted,
    ROUND(cum_accepted * 100.0 / NULLIF(cum_users, 0), 4)          AS cum_rate_pct,
    accepted_dod_delta,
    ROUND(rate_3d_rolling * 100, 4)                                 AS rate_3d_rolling_pct
FROM cumulative
ORDER BY variant, exp_day;


-- 5b. Novelty effect: early (d1-3) vs steady-state (d4+)
WITH windowed AS (
    SELECT
        a.user_id, a.variant,
        CASE WHEN DATEDIFF('day', a.assigned_at, u.event_ts) < 3
             THEN 'early_d1_3'
             ELSE 'steady_d4plus'
        END                                                         AS time_window,
        MAX(CASE WHEN u.event_type = 'offer_accepted' THEN 1 ELSE 0 END) AS accepted
    FROM experiment_assignments a
    LEFT JOIN user_events u ON a.user_id = u.user_id
    WHERE a.experiment_id = 'credit_limit_offer_v1'
      AND DATEDIFF('day', a.assigned_at, u.event_ts) BETWEEN 0 AND 29
    GROUP BY a.user_id, a.variant,
             CASE WHEN DATEDIFF('day', a.assigned_at, u.event_ts) < 3
                  THEN 'early_d1_3' ELSE 'steady_d4plus' END
),
agg AS (
    SELECT
        time_window, variant,
        COUNT(*) AS n, SUM(accepted) AS k,
        ROUND(AVG(accepted) * 100, 4) AS rate_pct
    FROM windowed GROUP BY time_window, variant
),
pivot AS (
    SELECT
        time_window,
        MAX(CASE WHEN variant='control'   THEN rate_pct END)        AS ctrl_pct,
        MAX(CASE WHEN variant='treatment' THEN rate_pct END)        AS treat_pct
    FROM agg GROUP BY time_window
)
SELECT
    time_window,
    ctrl_pct, treat_pct,
    ROUND(treat_pct - ctrl_pct, 4)                                  AS lift_pp,
    ROUND(
        (treat_pct - ctrl_pct)
        / NULLIF(MAX(CASE WHEN time_window='steady_d4plus'
                          THEN treat_pct - ctrl_pct END) OVER (), 0)
    , 3)                                                            AS novelty_ratio,
    CASE
        WHEN time_window = 'steady_d4plus' THEN '📌 Benchmark'
        WHEN (treat_pct - ctrl_pct)
             / NULLIF(MAX(CASE WHEN time_window='steady_d4plus'
                               THEN treat_pct - ctrl_pct END) OVER (), 0) > 1.20
        THEN '⚠  Novelty inflation — extend runtime'
        ELSE '✅ Consistent with steady state'
    END                                                             AS novelty_verdict
FROM pivot
ORDER BY time_window DESC;


-- ────────────────────────────────────────────────────────────────
-- SECTION 6 — BUSINESS IMPACT: REVENUE PROJECTION
-- Incremental acceptances × revenue per acceptance, scaled to
-- full platform eligible users (8,000/day).
-- ────────────────────────────────────────────────────────────────

WITH rates AS (
    SELECT
        variant,
        COUNT(*) AS n,
        SUM(CASE WHEN event_type='offer_accepted' THEN 1 ELSE 0 END) AS k,
        AVG(CASE WHEN event_type='offer_accepted' THEN 1.0 ELSE 0 END) AS accept_rate,
        AVG(CASE WHEN event_type='interchange_revenue' THEN revenue END) AS avg_rev_per_user
    FROM experiment_assignments a
    LEFT JOIN user_events u ON a.user_id = u.user_id
    WHERE a.experiment_id = 'credit_limit_offer_v1'
    GROUP BY variant
)
SELECT
    ROUND(MAX(CASE WHEN variant='control'   THEN accept_rate END) * 100, 4) AS ctrl_rate_pct,
    ROUND(MAX(CASE WHEN variant='treatment' THEN accept_rate END) * 100, 4) AS treat_rate_pct,
    ROUND(
      (MAX(CASE WHEN variant='treatment' THEN accept_rate END)
       - MAX(CASE WHEN variant='control'  THEN accept_rate END)) * 100
    , 4)                                                            AS lift_pp,
    ROUND(
      (MAX(CASE WHEN variant='treatment' THEN accept_rate END)
       - MAX(CASE WHEN variant='control'  THEN accept_rate END))
      / NULLIF(MAX(CASE WHEN variant='control' THEN accept_rate END), 0) * 100
    , 2)                                                            AS lift_rel_pct,
    -- Revenue projection
    8000                                                            AS daily_eligible_users,
    ROUND(8000 * (
        MAX(CASE WHEN variant='treatment' THEN accept_rate END)
        - MAX(CASE WHEN variant='control'  THEN accept_rate END)
    ), 0)                                                           AS incr_daily_acceptances,
    ROUND(MAX(CASE WHEN variant='control' THEN avg_rev_per_user END), 2) AS rev_per_acceptance,
    ROUND(8000 * (
        MAX(CASE WHEN variant='treatment' THEN accept_rate END)
        - MAX(CASE WHEN variant='control'  THEN accept_rate END)
    ) * MAX(CASE WHEN variant='control' THEN avg_rev_per_user END), 0) AS daily_rev_impact_usd,
    ROUND(8000 * (
        MAX(CASE WHEN variant='treatment' THEN accept_rate END)
        - MAX(CASE WHEN variant='control'  THEN accept_rate END)
    ) * MAX(CASE WHEN variant='control' THEN avg_rev_per_user END) * 365, 0) AS annual_rev_impact_usd
FROM rates;
