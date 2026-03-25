-- ============================================================
--  Run this SQL in your Supabase project's SQL Editor
--  (supabase.com → your project → SQL Editor → New Query)
-- ============================================================

-- 1. Daily recommendations written by the agent
CREATE TABLE IF NOT EXISTS recommendations (
    id                 BIGSERIAL PRIMARY KEY,
    date               DATE          NOT NULL,
    ticker             TEXT          NOT NULL,
    action             TEXT          NOT NULL,   -- 'BUY' or 'SELL'
    score              INT           NOT NULL,   -- 1-4
    signals            JSONB,                    -- per-strategy signal values
    backtest           JSONB,                    -- per-strategy backtest stats
    win_rate           FLOAT,
    avg_return         FLOAT,
    active_strategies  TEXT,
    -- Price context
    price              FLOAT,
    change_1d          FLOAT,
    change_5d          FLOAT,
    rsi                FLOAT,
    macd_hist          FLOAT,
    ema_bullish        BOOLEAN,
    supertrend_up      BOOLEAN,
    support            FLOAT,
    resistance         FLOAT,
    stop_loss          FLOAT,
    target             FLOAT,
    risk_pct           FLOAT,
    reward_pct         FLOAT,
    volume             BIGINT,
    avg_volume         BIGINT,
    created_at         TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast date-based queries
CREATE INDEX IF NOT EXISTS idx_recs_date   ON recommendations (date DESC);
CREATE INDEX IF NOT EXISTS idx_recs_ticker ON recommendations (ticker);

-- ──────────────────────────────────────────────────────────

-- 2. Paper portfolio (your virtual trades)
CREATE TABLE IF NOT EXISTS paper_portfolio (
    id          BIGSERIAL PRIMARY KEY,
    ticker      TEXT          NOT NULL,
    buy_date    DATE          NOT NULL,
    buy_price   FLOAT         NOT NULL,
    quantity    INT           NOT NULL,
    sell_date   DATE,
    sell_price  FLOAT,
    status      TEXT          DEFAULT 'OPEN',   -- 'OPEN' or 'CLOSED'
    pnl_pct     FLOAT,
    pnl_inr     FLOAT,
    notes       TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_port_status ON paper_portfolio (status);

-- ──────────────────────────────────────────────────────────

-- 3. Agent metadata (last run info shown in dashboard sidebar)
CREATE TABLE IF NOT EXISTS agent_meta (
    id               INT PRIMARY KEY DEFAULT 1,
    last_run         DATE,
    total_signals    INT DEFAULT 0,
    tickers_scanned  INT DEFAULT 0,
    failed           INT DEFAULT 0,
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

-- Insert initial row
INSERT INTO agent_meta (id) VALUES (1) ON CONFLICT (id) DO NOTHING;

-- ──────────────────────────────────────────────────────────
-- Done! All 3 tables created.
-- Now add your SUPABASE_URL and SUPABASE_KEY (anon key) to:
--   • GitHub Secrets  (for the analysis agent)
--   • Streamlit Secrets  (for the dashboard)
-- ──────────────────────────────────────────────────────────
