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

