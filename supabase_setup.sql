-- ============================================================
--  Run this in Supabase SQL Editor
--  Fresh install: run everything
--  Upgrading from v2: uncomment and run the ALTER TABLE block
-- ============================================================

-- 1. RECOMMENDATIONS
CREATE TABLE IF NOT EXISTS recommendations (
    id                  BIGSERIAL PRIMARY KEY,
    date                DATE         NOT NULL,
    ticker              TEXT         NOT NULL,
    action              TEXT         NOT NULL,
    raw_score           INT,
    weighted_score_val  FLOAT,
    composite_score     FLOAT,
    technical_score     FLOAT,
    final_score_multiplier FLOAT,
    fundamental_multiplier FLOAT,
    score_label         TEXT,
    score_breakdown     JSONB,
    signals             JSONB,
    strategy_weights    JSONB,
    backtest            JSONB,
    active_strategies   TEXT,
    low_sample_warning  BOOLEAN DEFAULT FALSE,
    win_rate            FLOAT,
    avg_return          FLOAT,
    median_return       FLOAT,
    profit_factor       FLOAT,
    max_drawdown        FLOAT,
    avg_trades          INT,
    market_regime       TEXT,
    param_version       TEXT,
    price               FLOAT,
    change_1d           FLOAT,
    change_5d           FLOAT,
    rsi                 FLOAT,
    macd_hist           FLOAT,
    ema_bullish         BOOLEAN,
    supertrend_up       BOOLEAN,
    supertrend_line     FLOAT,
    support             FLOAT,
    resistance          FLOAT,
    stop_loss           FLOAT,
    target              FLOAT,
    risk_pct            FLOAT,
    reward_pct          FLOAT,
    rr_ratio            FLOAT,
    benchmark_return_pct FLOAT,
    relative_return_pct  FLOAT,
    benchmark_outperformance_rate FLOAT,
    volume              BIGINT,
    avg_volume          BIGINT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_recs_date   ON recommendations (date DESC);
CREATE INDEX IF NOT EXISTS idx_recs_ticker ON recommendations (ticker);
CREATE INDEX IF NOT EXISTS idx_recs_score  ON recommendations (composite_score DESC);

-- 2. PAPER PORTFOLIO
CREATE TABLE IF NOT EXISTS paper_portfolio (
    id                 BIGSERIAL PRIMARY KEY,
    ticker             TEXT        NOT NULL,
    buy_date           DATE        NOT NULL,
    buy_price          FLOAT       NOT NULL,
    quantity           INT         NOT NULL,
    entry_stop_loss    FLOAT,
    entry_target       FLOAT,
    sell_date          DATE,
    sell_price         FLOAT,
    status             TEXT        DEFAULT 'OPEN',
    pnl_pct            FLOAT,
    pnl_inr            FLOAT,
    exit_reason        TEXT,
    notes              TEXT,
    recommendation_id  BIGINT,
    created_at         TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_port_status ON paper_portfolio (status);

-- 3. TICKER RUN LOG
CREATE TABLE IF NOT EXISTS ticker_run_log (
    id         BIGSERIAL PRIMARY KEY,
    date       DATE  NOT NULL,
    ticker     TEXT  NOT NULL,
    status     TEXT  NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_runlog_date ON ticker_run_log (date DESC);

-- 4. AGENT META
CREATE TABLE IF NOT EXISTS agent_meta (
    id                   INT PRIMARY KEY DEFAULT 1,
    last_run             DATE,
    total_signals        INT DEFAULT 0,
    tickers_scanned      INT DEFAULT 0,
    failed               INT DEFAULT 0,
    market_regime        TEXT,
    active_param_version TEXT,
    updated_at           TIMESTAMPTZ DEFAULT NOW()
);
INSERT INTO agent_meta (id) VALUES (1) ON CONFLICT (id) DO NOTHING;

-- 5. AGENT PARAMS  (optimizer output — champion/challenger/candidates)
CREATE TABLE IF NOT EXISTS agent_params (
    id               BIGSERIAL PRIMARY KEY,
    version          INT          NOT NULL UNIQUE,
    status           TEXT         NOT NULL DEFAULT 'candidate',
    params_json      JSONB        NOT NULL,
    objective_score  FLOAT,
    profit_factor    FLOAT,
    win_rate         FLOAT,
    avg_return       FLOAT,
    max_drawdown     FLOAT,
    total_trades     INT,
    train_start      DATE,
    train_end        DATE,
    valid_start      DATE,
    valid_end        DATE,
    run_date         DATE,
    promoted_at      DATE,
    rank             INT,
    notes            TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_params_status  ON agent_params (status);
CREATE INDEX IF NOT EXISTS idx_params_version ON agent_params (version DESC);

-- 6. OPTIMIZATION RUN LOG
CREATE TABLE IF NOT EXISTS optimization_runs (
    id                  BIGSERIAL PRIMARY KEY,
    run_date            DATE NOT NULL,
    n_trials            INT,
    n_valid_trials      INT,
    best_score          FLOAT,
    best_profit_factor  FLOAT,
    best_win_rate       FLOAT,
    best_avg_return     FLOAT,
    champion_version    INT,
    challenger_version  INT,
    stocks_used         INT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- DISABLE RLS (personal project)
ALTER TABLE recommendations   DISABLE ROW LEVEL SECURITY;
ALTER TABLE paper_portfolio   DISABLE ROW LEVEL SECURITY;
ALTER TABLE ticker_run_log    DISABLE ROW LEVEL SECURITY;
ALTER TABLE agent_meta        DISABLE ROW LEVEL SECURITY;
ALTER TABLE agent_params      DISABLE ROW LEVEL SECURITY;
ALTER TABLE optimization_runs DISABLE ROW LEVEL SECURITY;

-- ── UPGRADING FROM v2? Run only this block ──────────────────

ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS param_version TEXT;
ALTER TABLE agent_meta      ADD COLUMN IF NOT EXISTS active_param_version TEXT;

CREATE TABLE IF NOT EXISTS agent_params (
    id BIGSERIAL PRIMARY KEY, version INT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'candidate', params_json JSONB NOT NULL,
    objective_score FLOAT, profit_factor FLOAT, win_rate FLOAT,
    avg_return FLOAT, max_drawdown FLOAT, total_trades INT,
    train_start DATE, train_end DATE, valid_start DATE, valid_end DATE,
    run_date DATE, promoted_at DATE, rank INT, notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE agent_params DISABLE ROW LEVEL SECURITY;

CREATE TABLE IF NOT EXISTS optimization_runs (
    id BIGSERIAL PRIMARY KEY, run_date DATE NOT NULL,
    n_trials INT, n_valid_trials INT, best_score FLOAT,
    best_profit_factor FLOAT, best_win_rate FLOAT, best_avg_return FLOAT,
    champion_version INT, challenger_version INT, stocks_used INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE optimization_runs DISABLE ROW LEVEL SECURITY;



-- v5 compatibility upgrades
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS company_name TEXT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS debt_equity FLOAT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS fundamental_flag TEXT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS news_sentiment TEXT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS news_headline TEXT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS news_alert BOOLEAN DEFAULT FALSE;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS signal_streak INT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS streak INT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS news_label TEXT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS news_count INT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS news_headlines JSONB;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS news_multiplier FLOAT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS fundamental_score FLOAT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS fundamental_warnings JSONB;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS technical_score FLOAT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS final_score_multiplier FLOAT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS fundamental_multiplier FLOAT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS rr_ratio FLOAT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS benchmark_return_pct FLOAT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS relative_return_pct FLOAT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS benchmark_outperformance_rate FLOAT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS market_cap_cr FLOAT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS de_ratio FLOAT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS roe FLOAT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS sector TEXT;

ALTER TABLE agent_meta ADD COLUMN IF NOT EXISTS total_buys INT DEFAULT 0;
ALTER TABLE agent_meta ADD COLUMN IF NOT EXISTS total_sells INT DEFAULT 0;
ALTER TABLE agent_meta ADD COLUMN IF NOT EXISTS total_exits INT DEFAULT 0;
ALTER TABLE agent_meta ADD COLUMN IF NOT EXISTS breadth_ratio FLOAT;
ALTER TABLE agent_meta ADD COLUMN IF NOT EXISTS breadth_label TEXT;
ALTER TABLE agent_meta ADD COLUMN IF NOT EXISTS breadth_buys INT DEFAULT 0;
ALTER TABLE agent_meta ADD COLUMN IF NOT EXISTS breadth_sells INT DEFAULT 0;
ALTER TABLE agent_meta ADD COLUMN IF NOT EXISTS breadth_exits INT DEFAULT 0;
ALTER TABLE agent_meta ADD COLUMN IF NOT EXISTS breadth_neutral INT DEFAULT 0;

CREATE TABLE IF NOT EXISTS stock_fundamentals (
    ticker                TEXT PRIMARY KEY,
    fetch_date            DATE NOT NULL,
    market_cap_cr         FLOAT,
    pe_ratio              FLOAT,
    pb_ratio              FLOAT,
    de_ratio              FLOAT,
    roe                   FLOAT,
    revenue_growth        FLOAT,
    profit_growth         FLOAT,
    sector                TEXT,
    dividend_yield        FLOAT,
    fundamental_score     FLOAT,
    fundamental_warnings  JSONB,
    created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS backtest_simulations (
    id                  BIGSERIAL PRIMARY KEY,
    recommendation_id   BIGINT UNIQUE,
    ticker              TEXT NOT NULL,
    signal_date         DATE NOT NULL,
    action              TEXT NOT NULL,
    entry_price         FLOAT,
    exit_price          FLOAT,
    exit_date           DATE,
    exit_reason         TEXT,
    actual_return_pct   FLOAT,
    benchmark_return_pct FLOAT,
    relative_return_pct  FLOAT,
    benchmark_outperformance_rate FLOAT,
    rr_ratio            FLOAT,
    composite_score     FLOAT,
    technical_score     FLOAT,
    predicted_win_rate  FLOAT,
    was_win             BOOLEAN,
    days_held           INT,
    run_date            DATE,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_backsim_signal_date ON backtest_simulations (signal_date DESC);

CREATE TABLE IF NOT EXISTS simulation_meta (
    id                  INT PRIMARY KEY DEFAULT 1,
    last_run            DATE,
    total_simulated     INT DEFAULT 0,
    actual_win_rate     FLOAT,
    actual_avg_return   FLOAT,
    actual_avg_relative_return FLOAT,
    benchmark_outperformance_rate FLOAT,
    pending_unprocessed INT DEFAULT 0,
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);
INSERT INTO simulation_meta (id) VALUES (1) ON CONFLICT (id) DO NOTHING;

ALTER TABLE stock_fundamentals DISABLE ROW LEVEL SECURITY;
ALTER TABLE backtest_simulations DISABLE ROW LEVEL SECURITY;
ALTER TABLE simulation_meta DISABLE ROW LEVEL SECURITY;


-- v6 quant-quality upgrades: next-bar execution metadata, EXIT semantics, dynamic R:R,
-- benchmark-relative metrics, and final score multipliers. Safe to re-run.
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS technical_score FLOAT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS final_score_multiplier FLOAT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS fundamental_multiplier FLOAT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS rr_ratio FLOAT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS benchmark_return_pct FLOAT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS relative_return_pct FLOAT;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS benchmark_outperformance_rate FLOAT;

ALTER TABLE backtest_simulations ADD COLUMN IF NOT EXISTS benchmark_return_pct FLOAT;
ALTER TABLE backtest_simulations ADD COLUMN IF NOT EXISTS relative_return_pct FLOAT;
ALTER TABLE backtest_simulations ADD COLUMN IF NOT EXISTS benchmark_outperformance_rate FLOAT;
ALTER TABLE backtest_simulations ADD COLUMN IF NOT EXISTS rr_ratio FLOAT;
ALTER TABLE backtest_simulations ADD COLUMN IF NOT EXISTS technical_score FLOAT;

ALTER TABLE simulation_meta ADD COLUMN IF NOT EXISTS actual_avg_relative_return FLOAT;
ALTER TABLE simulation_meta ADD COLUMN IF NOT EXISTS benchmark_outperformance_rate FLOAT;

ALTER TABLE agent_meta ADD COLUMN IF NOT EXISTS total_exits INT DEFAULT 0;
ALTER TABLE agent_meta ADD COLUMN IF NOT EXISTS breadth_exits INT DEFAULT 0;


-- 2026-04-26: add dynamic R:R guard params to existing champion/challenger JSON
-- These are JSON parameters, not table columns. They prevent BUY recommendations
-- from showing stop downside larger than target upside.
UPDATE agent_params
SET params_json = jsonb_set(COALESCE(params_json, '{}'::jsonb), '{MIN_RR_RATIO}', '1.5'::jsonb, true)
WHERE status IN ('champion', 'challenger')
  AND NOT (COALESCE(params_json, '{}'::jsonb) ? 'MIN_RR_RATIO');

UPDATE agent_params
SET params_json = jsonb_set(COALESCE(params_json, '{}'::jsonb), '{MAX_RISK_PCT}', '8.0'::jsonb, true)
WHERE status IN ('champion', 'challenger')
  AND NOT (COALESCE(params_json, '{}'::jsonb) ? 'MAX_RISK_PCT');
