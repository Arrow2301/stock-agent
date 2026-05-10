-- ====================================================================
--  Migration 001: v7 → v8
--
--  What changes in v8:
--   1. recommendations gets three new columns for risk-aware position
--      sizing (suggested_qty, size_blocked, size_reasons).
--   2. agent_params gets a partial unique index enforcing
--      "at most one champion at a time".
--   3. score_models gets the same partial unique index.
--   4. synthetic_training_data is created if it doesn't already exist.
--      (Many v7 deployments created it ad-hoc; the canonical schema
--      now defines it.)
--   5. paper_portfolio's quantity check constraint is added (was implicit).
--
--  Idempotent: every change is wrapped so re-running the migration is safe.
-- ====================================================================

-- (1) recommendations: add v8 sizing columns
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS suggested_qty INTEGER DEFAULT 0;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS size_blocked  BOOLEAN DEFAULT FALSE;
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS size_reasons  JSONB;

-- (2) agent_params: enforce single-champion
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE indexname = 'uniq_agent_params_one_champion'
    ) THEN
        CREATE UNIQUE INDEX uniq_agent_params_one_champion
            ON agent_params(status) WHERE status = 'champion';
    END IF;
END$$;

-- (3) score_models: enforce single-champion
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE indexname = 'uniq_score_models_one_champion'
    ) THEN
        CREATE UNIQUE INDEX uniq_score_models_one_champion
            ON score_models(status) WHERE status = 'champion';
    END IF;
END$$;

-- (4) synthetic_training_data: create if missing.
--     Schema must match agent/score_model.py:NUMERIC_FEATURES + BINARY_FEATURES.
CREATE TABLE IF NOT EXISTS synthetic_training_data (
    id                  BIGSERIAL PRIMARY KEY,
    signal_date         DATE        NOT NULL,
    ticker              TEXT        NOT NULL,
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
    ema_bullish         BOOLEAN,
    has_donchian        BOOLEAN,
    has_ema             BOOLEAN,
    has_rsi_trend       BOOLEAN,
    has_bollinger       BOOLEAN,
    single_strat        BOOLEAN,
    multi_strat         BOOLEAN,
    regime              TEXT,
    entry_price         NUMERIC,
    exit_price          NUMERIC,
    actual_return_pct   NUMERIC,
    days_held           INTEGER,
    exit_reason         TEXT,
    was_win             BOOLEAN     NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_synth_training_date   ON synthetic_training_data(signal_date);
CREATE INDEX IF NOT EXISTS idx_synth_training_ticker ON synthetic_training_data(ticker);
ALTER TABLE synthetic_training_data DISABLE ROW LEVEL SECURITY;

-- (5) paper_portfolio: ensure quantity > 0 check exists.
--     NOT VALID = constraint applies to new rows only; existing rows aren't
--     re-checked. Safer than risking the migration aborting on a legacy row
--     with quantity 0. Run `ALTER TABLE … VALIDATE CONSTRAINT …` later
--     after cleanup if you want full validation.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.check_constraints
        WHERE constraint_name = 'paper_portfolio_quantity_check'
    ) THEN
        ALTER TABLE paper_portfolio
            ADD CONSTRAINT paper_portfolio_quantity_check
            CHECK (quantity > 0) NOT VALID;
    END IF;
END$$;
