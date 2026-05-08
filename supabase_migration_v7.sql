-- ============================================================
--  Stock Agent — v7 schema migration
--  Adds breach-detection persistence + housekeeping for v7.
--  Safe to re-run (uses IF NOT EXISTS / IF EXISTS guards).
-- ============================================================

-- 1. Persist intraday SL / target breaches on paper trades
--    detect_breach() in agent/market_data.py writes these.
ALTER TABLE paper_portfolio
    ADD COLUMN IF NOT EXISTS breach_flag  TEXT,    -- 'sl_hit' | 'target_hit' | NULL
    ADD COLUMN IF NOT EXISTS breach_price FLOAT,   -- price at which a real broker would have filled
    ADD COLUMN IF NOT EXISTS breach_date  DATE;    -- bar date on which the breach was detected

CREATE INDEX IF NOT EXISTS idx_port_breach
    ON paper_portfolio (breach_flag)
    WHERE breach_flag IS NOT NULL;


-- 2. Per-trade health metrics — track our live results vs the
--    walk-forward backsimulation predictions for each signal.
CREATE TABLE IF NOT EXISTS health_checks (
    id              BIGSERIAL PRIMARY KEY,
    run_date        DATE NOT NULL,
    run_time        TIMESTAMPTZ DEFAULT NOW(),
    check_name      TEXT NOT NULL,
    status          TEXT NOT NULL,                 -- 'pass' | 'warn' | 'fail'
    metric_value    FLOAT,
    threshold_value FLOAT,
    detail          TEXT
);
CREATE INDEX IF NOT EXISTS idx_health_run_date
    ON health_checks (run_date DESC);
ALTER TABLE health_checks DISABLE ROW LEVEL SECURITY;


-- 3. Holiday calendar — populated at deploy time. The morning agent
--    skips runs on these dates. NSE publishes the calendar yearly
--    and it rarely changes mid-year.
CREATE TABLE IF NOT EXISTS market_holidays (
    holiday_date DATE PRIMARY KEY,
    description  TEXT,
    market       TEXT NOT NULL DEFAULT 'NSE'
);
ALTER TABLE market_holidays DISABLE ROW LEVEL SECURITY;


-- 4. Live-vs-backtest reconciliation table — populated nightly.
--    Once a paper trade closes, compare its actual outcome to what
--    the backsimulator predicted at signal time. Drift here is the
--    single best signal that something has broken.
CREATE TABLE IF NOT EXISTS reconciliation (
    id                   BIGSERIAL PRIMARY KEY,
    paper_trade_id       BIGINT REFERENCES paper_portfolio (id) ON DELETE CASCADE,
    recommendation_id    BIGINT,
    ticker               TEXT,
    signal_date          DATE,
    paper_return_pct     FLOAT,
    backsim_return_pct   FLOAT,
    return_diff_pct      FLOAT,         -- paper - backsim. Should be near zero if logic agrees.
    paper_exit_reason    TEXT,
    backsim_exit_reason  TEXT,
    created_at           TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (paper_trade_id)
);
ALTER TABLE reconciliation DISABLE ROW LEVEL SECURITY;
CREATE INDEX IF NOT EXISTS idx_recon_signal_date
    ON reconciliation (signal_date DESC);


-- 5. Optional housekeeping — purge ticker_run_log rows older than 90 days
--    (free-tier Supabase has 500 MB. ~250 rows/day = manageable but
--    not infinite. Run once on migration, then via a weekly job if desired.)
DELETE FROM ticker_run_log
WHERE created_at < NOW() - INTERVAL '90 days';


-- 6. Sanity: ensure there is at most one champion at any time.
--    This is a safety net for the optimizer's promote logic; if a
--    failed run ever leaves two champions, this constraint surfaces it.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_champion
    ON agent_params (status)
    WHERE status = 'champion';
