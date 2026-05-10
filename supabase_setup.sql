-- ====================================================================
--  Stock Agent — Canonical Supabase Schema (v8)
--  Run once on a fresh project. Idempotent: every CREATE uses IF NOT EXISTS,
--  every ALTER is wrapped in DO blocks that no-op if the column already exists.
--
--  This is the SOLE source of truth for the schema. There are no separate
--  v2-install / v5-upgrade / v6-upgrade / v7-upgrade files — the historical
--  migrations have been folded in. If you're upgrading an existing
--  pre-v8 project, see supabase_migrations/ for a numbered upgrade trail.
-- ====================================================================

-- --------------------------------------------------------------------
-- 1. recommendations
--    One row per (date, ticker, action). Re-runs on the same date wipe
--    and re-insert; that's an idempotency property of analyze.py.
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS recommendations (
    id                              BIGSERIAL PRIMARY KEY,
    date                            DATE        NOT NULL,
    ticker                          TEXT        NOT NULL,
    action                          TEXT        NOT NULL CHECK (action IN ('BUY','EXIT','SELL')),

    -- Scoring
    raw_score                       INTEGER,
    weighted_score_val              NUMERIC,
    technical_score                 NUMERIC,
    composite_score                 NUMERIC,
    final_score_multiplier          NUMERIC,
    fundamental_multiplier          NUMERIC,
    score_label                     TEXT,
    score_breakdown                 JSONB,
    p_win                           NUMERIC,    -- Phase 2/3 calibrated probability

    -- Strategy detail
    signals                         JSONB,
    strategy_weights                JSONB,
    backtest                        JSONB,
    active_strategies               TEXT,
    low_sample_warning              BOOLEAN     DEFAULT FALSE,

    -- Aggregated backtest stats (over active strategies for BUYs)
    win_rate                        NUMERIC,
    avg_return                      NUMERIC,
    median_return                   NUMERIC,
    profit_factor                   NUMERIC,
    max_drawdown                    NUMERIC,
    avg_trades                      INTEGER,
    benchmark_return_pct            NUMERIC,
    relative_return_pct             NUMERIC,
    benchmark_outperformance_rate   NUMERIC,

    -- Context (regime + param version active when this row was generated)
    market_regime                   TEXT,
    param_version                   TEXT,

    -- Fundamentals snapshot
    company_name                    TEXT,
    pe_ratio                        NUMERIC,
    debt_equity                     NUMERIC,
    de_ratio                        NUMERIC,    -- legacy alias kept for back-compat
    revenue_growth                  NUMERIC,
    sector                          TEXT,
    market_cap_cr                   NUMERIC,
    roe                             NUMERIC,
    fundamental_score               NUMERIC,
    fundamental_flag                TEXT,
    fundamental_warnings            JSONB,

    -- News snapshot
    news_score                      NUMERIC,
    news_sentiment                  TEXT,
    news_headline                   TEXT,
    news_headlines                  JSONB,
    news_alert                      BOOLEAN     DEFAULT FALSE,
    news_label                      TEXT,
    news_multiplier                 NUMERIC,
    news_count                      INTEGER,

    -- Streak (consecutive days same action)
    signal_streak                   INTEGER     DEFAULT 1,

    -- VIX snapshot at signal time (Phase 3)
    vix_level                       NUMERIC,
    vix_change_5d                   NUMERIC,
    vix_zscore_60d                  NUMERIC,

    -- Market context (price + indicators)
    price                           NUMERIC,
    change_1d                       NUMERIC,
    change_5d                       NUMERIC,
    change_30d                      NUMERIC,
    rsi                             NUMERIC,
    macd_hist                       NUMERIC,
    ema_bullish                     BOOLEAN,
    support                         NUMERIC,
    resistance                      NUMERIC,
    stop_loss                       NUMERIC,
    target                          NUMERIC,
    risk_pct                        NUMERIC,
    reward_pct                      NUMERIC,
    rr_ratio                        NUMERIC,
    rr_floor_applied                BOOLEAN     DEFAULT FALSE,
    volume                          BIGINT,
    avg_volume                      BIGINT,
    atr_pct                         NUMERIC,
    pct_from_52w_high               NUMERIC,
    pct_from_52w_low                NUMERIC,

    -- v8: risk-aware position-sizing recommendation (BUYs only)
    suggested_qty                   INTEGER     DEFAULT 0,
    size_blocked                    BOOLEAN     DEFAULT FALSE,
    size_reasons                    JSONB,

    created_at                      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_recommendations_date          ON recommendations(date DESC);
CREATE INDEX IF NOT EXISTS idx_recommendations_ticker_date   ON recommendations(ticker, date DESC);
CREATE INDEX IF NOT EXISTS idx_recommendations_action_date   ON recommendations(action, date DESC);

-- --------------------------------------------------------------------
-- 2. paper_portfolio
--    User-tracked simulated trades. status flips OPEN→CLOSED on exit.
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS paper_portfolio (
    id                  BIGSERIAL PRIMARY KEY,
    recommendation_id   BIGINT,                 -- nullable; manual buys can have no source rec
    ticker              TEXT        NOT NULL,
    buy_date            DATE        NOT NULL,
    buy_price           NUMERIC     NOT NULL,
    quantity            INTEGER     NOT NULL CHECK (quantity > 0),
    entry_stop_loss     NUMERIC,
    entry_target        NUMERIC,
    status              TEXT        NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN','CLOSED')),

    -- Set on close
    sell_date           DATE,
    sell_price          NUMERIC,
    pnl_inr             NUMERIC,
    pnl_pct             NUMERIC,
    exit_reason         TEXT,

    -- Sticky breach flags written by check_alerts.py once SL or target is touched
    breach_flag         TEXT        CHECK (breach_flag IN ('sl_hit','target_hit') OR breach_flag IS NULL),
    breach_price        NUMERIC,
    breach_date         DATE,

    notes               TEXT,
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_paper_portfolio_status        ON paper_portfolio(status);
CREATE INDEX IF NOT EXISTS idx_paper_portfolio_ticker_status ON paper_portfolio(ticker, status);
CREATE INDEX IF NOT EXISTS idx_paper_portfolio_open_dates    ON paper_portfolio(buy_date) WHERE status = 'OPEN';

-- --------------------------------------------------------------------
-- 3. ticker_run_log
--    One row per (date, ticker) showing whether data fetch + analysis
--    succeeded. Used for health/observability.
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ticker_run_log (
    id          BIGSERIAL PRIMARY KEY,
    date        DATE        NOT NULL,
    ticker      TEXT        NOT NULL,
    status      TEXT        NOT NULL,            -- 'ok', 'insufficient_data', exception text, etc.
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ticker_run_log_date ON ticker_run_log(date DESC);

-- --------------------------------------------------------------------
-- 4. agent_meta
--    Singleton-ish row keyed by id=1. Updated each daily run.
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_meta (
    id                      INTEGER PRIMARY KEY,
    last_run                DATE,
    total_signals           INTEGER,
    tickers_scanned         INTEGER,
    failed                  INTEGER,
    market_regime           TEXT,
    active_param_version    TEXT,
    total_buys              INTEGER,
    total_sells             INTEGER,
    total_exits             INTEGER,
    breadth_ratio           NUMERIC,
    breadth_label           TEXT,
    breadth_buys            INTEGER,
    breadth_sells           INTEGER,
    breadth_exits           INTEGER,
    breadth_neutral         INTEGER,
    updated_at              TIMESTAMPTZ DEFAULT now()
);

-- --------------------------------------------------------------------
-- 5. agent_params
--    Optimizer output. status='champion' = currently active.
--    A partial unique index enforces "at most one champion at a time".
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_params (
    id                  BIGSERIAL PRIMARY KEY,
    version             INTEGER     NOT NULL UNIQUE,
    status              TEXT        NOT NULL CHECK (status IN ('candidate','challenger','champion','retired')),
    params_json         JSONB       NOT NULL,
    objective_score     NUMERIC,
    profit_factor       NUMERIC,
    win_rate            NUMERIC,
    avg_return          NUMERIC,
    max_drawdown        NUMERIC,
    total_trades        INTEGER,
    train_start         DATE,
    train_end           DATE,
    valid_start         DATE,
    valid_end           DATE,
    run_date            DATE,
    rank                INTEGER,
    notes               TEXT,
    promoted_at         DATE,
    created_at          TIMESTAMPTZ DEFAULT now()
);
-- At most one champion at any time.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_agent_params_one_champion
    ON agent_params(status) WHERE status = 'champion';
CREATE INDEX IF NOT EXISTS idx_agent_params_status_promoted
    ON agent_params(status, promoted_at DESC);

-- --------------------------------------------------------------------
-- 6. optimization_runs
--    Log of every weekly optimizer run.
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS optimization_runs (
    id                      BIGSERIAL PRIMARY KEY,
    run_date                DATE        NOT NULL,
    n_trials                INTEGER,
    n_valid_trials          INTEGER,
    best_score              NUMERIC,
    best_profit_factor      NUMERIC,
    best_win_rate           NUMERIC,
    best_avg_return         NUMERIC,
    champion_version        INTEGER,
    challenger_version      INTEGER,
    stocks_used             INTEGER,
    created_at              TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_optimization_runs_date ON optimization_runs(run_date DESC);

-- --------------------------------------------------------------------
-- 7. stock_fundamentals
--    Periodic fundamentals snapshot (filled by analyze.py + agent).
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stock_fundamentals (
    ticker          TEXT PRIMARY KEY,
    company_name    TEXT,
    pe_ratio        NUMERIC,
    debt_equity     NUMERIC,
    revenue_growth  NUMERIC,
    market_cap_cr   NUMERIC,
    roe             NUMERIC,
    sector          TEXT,
    fundamental_score INTEGER,
    fundamental_flag  TEXT,
    last_updated    DATE
);

-- --------------------------------------------------------------------
-- 8. backtest_simulations
--    Walk-forward back-test outcomes for each historical recommendation.
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtest_simulations (
    id                              BIGSERIAL PRIMARY KEY,
    recommendation_id               BIGINT      NOT NULL UNIQUE,
    ticker                          TEXT        NOT NULL,
    signal_date                     DATE        NOT NULL,
    action                          TEXT        NOT NULL,
    entry_price                     NUMERIC,
    exit_price                      NUMERIC,
    exit_date                       DATE,
    exit_reason                     TEXT,
    actual_return_pct               NUMERIC,
    benchmark_return_pct            NUMERIC,
    relative_return_pct             NUMERIC,
    benchmark_outperformance_rate   NUMERIC,
    rr_ratio                        NUMERIC,
    composite_score                 NUMERIC,
    technical_score                 NUMERIC,
    predicted_win_rate              NUMERIC,
    was_win                         BOOLEAN,
    days_held                       INTEGER,
    run_date                        DATE,
    created_at                      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_backtest_simulations_ticker ON backtest_simulations(ticker);
CREATE INDEX IF NOT EXISTS idx_backtest_simulations_date   ON backtest_simulations(signal_date DESC);

-- --------------------------------------------------------------------
-- 9. simulation_meta
--    Singleton-ish summary row from the weekly back-simulation job.
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS simulation_meta (
    id                              INTEGER PRIMARY KEY,
    last_run                        DATE,
    total_simulated                 INTEGER,
    actual_win_rate                 NUMERIC,
    actual_avg_return               NUMERIC,
    actual_avg_relative_return      NUMERIC,
    benchmark_outperformance_rate   NUMERIC,
    pending_unprocessed             INTEGER,
    updated_at                      TIMESTAMPTZ DEFAULT now()
);

-- --------------------------------------------------------------------
-- 10. health_checks
--     Daily observability snapshot from health_check.py.
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS health_checks (
    id              BIGSERIAL PRIMARY KEY,
    run_date        DATE        NOT NULL,
    overall_status  TEXT,                       -- 'healthy', 'degraded', 'failed'
    checks_json     JSONB       NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_health_checks_date ON health_checks(run_date DESC);

-- --------------------------------------------------------------------
-- 11. market_holidays
--     NSE trading holidays. Cron jobs read this to skip non-trading days.
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS market_holidays (
    holiday_date    DATE PRIMARY KEY,
    holiday_name    TEXT,
    notes           TEXT
);

-- --------------------------------------------------------------------
-- 12. reconciliation
--     Periodic reconciliation between live broker (Zerodha CSV import)
--     and paper portfolio. Optional; populated by a manual import.
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reconciliation (
    id              BIGSERIAL PRIMARY KEY,
    run_date        DATE        NOT NULL,
    ticker          TEXT,
    paper_qty       INTEGER,
    actual_qty      INTEGER,
    paper_avg_price NUMERIC,
    actual_avg_price NUMERIC,
    diff_qty        INTEGER,
    diff_pct        NUMERIC,
    notes           TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- --------------------------------------------------------------------
-- 13. score_models
--     Phase 2/3 model artifacts (calibrated LR or LGBM bytes + metrics).
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS score_models (
    id              BIGSERIAL PRIMARY KEY,
    fingerprint     TEXT        NOT NULL UNIQUE,
    trained_at      TIMESTAMPTZ NOT NULL,
    pickle_b64      TEXT        NOT NULL,
    metrics         JSONB,
    feature_list    JSONB,
    status          TEXT        NOT NULL DEFAULT 'champion'
                    CHECK (status IN ('champion','retired','candidate'))
);
CREATE UNIQUE INDEX IF NOT EXISTS uniq_score_models_one_champion
    ON score_models(status) WHERE status = 'champion';

-- --------------------------------------------------------------------
-- 14. synthetic_training_data
--     Rows used to train the score_model. Generated by build_training_data.py.
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS synthetic_training_data (
    id                  BIGSERIAL PRIMARY KEY,
    signal_date         DATE        NOT NULL,
    ticker              TEXT        NOT NULL,

    -- Numeric features (must match score_model.NUMERIC_FEATURES)
    rsi                 NUMERIC,
    macd_hist           NUMERIC,
    atr_pct             NUMERIC,
    change_1d           NUMERIC,
    change_5d           NUMERIC,
    change_30d          NUMERIC,
    pct_from_52w_high   NUMERIC,
    pct_from_52w_low    NUMERIC,
    vol_ratio           NUMERIC,
    n_firing            INTEGER,
    risk_pct            NUMERIC,
    reward_pct          NUMERIC,
    rr_ratio            NUMERIC,
    regime_score        NUMERIC,
    mom_vs_nifty_30d    NUMERIC,
    vix_level           NUMERIC,
    vix_change_5d       NUMERIC,
    vix_zscore_60d      NUMERIC,

    -- Binary features (must match score_model.BINARY_FEATURES)
    ema_bullish         BOOLEAN,
    has_donchian        BOOLEAN,
    has_ema             BOOLEAN,
    has_rsi_trend       BOOLEAN,
    has_bollinger       BOOLEAN,
    single_strat        BOOLEAN,
    multi_strat         BOOLEAN,

    -- Context columns (informational, not features)
    regime              TEXT,
    entry_price         NUMERIC,
    exit_price          NUMERIC,
    actual_return_pct   NUMERIC,
    days_held           INTEGER,
    exit_reason         TEXT,

    -- Label
    was_win             BOOLEAN     NOT NULL,

    created_at          TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_synth_training_date   ON synthetic_training_data(signal_date);
CREATE INDEX IF NOT EXISTS idx_synth_training_ticker ON synthetic_training_data(ticker);

-- --------------------------------------------------------------------
--  RLS — DISABLED on all tables.
--
--  Rationale: this is a single-user paper-trading project where the
--  Supabase URL+anon key are stored in workflow secrets and Streamlit
--  secrets. The dashboard's password gate provides the auth boundary.
--  Enabling RLS without policies would silently break every read.
-- --------------------------------------------------------------------
ALTER TABLE recommendations           DISABLE ROW LEVEL SECURITY;
ALTER TABLE paper_portfolio           DISABLE ROW LEVEL SECURITY;
ALTER TABLE ticker_run_log            DISABLE ROW LEVEL SECURITY;
ALTER TABLE agent_meta                DISABLE ROW LEVEL SECURITY;
ALTER TABLE agent_params              DISABLE ROW LEVEL SECURITY;
ALTER TABLE optimization_runs         DISABLE ROW LEVEL SECURITY;
ALTER TABLE stock_fundamentals        DISABLE ROW LEVEL SECURITY;
ALTER TABLE backtest_simulations      DISABLE ROW LEVEL SECURITY;
ALTER TABLE simulation_meta           DISABLE ROW LEVEL SECURITY;
ALTER TABLE health_checks             DISABLE ROW LEVEL SECURITY;
ALTER TABLE market_holidays           DISABLE ROW LEVEL SECURITY;
ALTER TABLE reconciliation            DISABLE ROW LEVEL SECURITY;
ALTER TABLE score_models              DISABLE ROW LEVEL SECURITY;
ALTER TABLE synthetic_training_data   DISABLE ROW LEVEL SECURITY;
