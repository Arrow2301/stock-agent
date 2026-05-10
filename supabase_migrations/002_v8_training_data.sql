-- ====================================================================
--  Migration 002: synthetic_training_data v8.1 schema additions
--
--  The rebuilt agent/build_training_data.py records additional fields
--  needed for trailing-stop research, gap-on-entry diagnostics, and
--  data-quality filtering. This migration adds those columns
--  (idempotent — safe to re-run).
--
--  After running this, RE-RUN agent/build_training_data.py with
--  --purge to regenerate clean training data:
--
--    python agent/build_training_data.py --purge
-- ====================================================================

ALTER TABLE synthetic_training_data ADD COLUMN IF NOT EXISTS signal_close       NUMERIC;
ALTER TABLE synthetic_training_data ADD COLUMN IF NOT EXISTS entry_price        NUMERIC;
ALTER TABLE synthetic_training_data ADD COLUMN IF NOT EXISTS sl_price           NUMERIC;
ALTER TABLE synthetic_training_data ADD COLUMN IF NOT EXISTS target_price       NUMERIC;
ALTER TABLE synthetic_training_data ADD COLUMN IF NOT EXISTS gap_pct            NUMERIC;
ALTER TABLE synthetic_training_data ADD COLUMN IF NOT EXISTS mfe_bar            INTEGER;
ALTER TABLE synthetic_training_data ADD COLUMN IF NOT EXISTS mae_bar            INTEGER;
ALTER TABLE synthetic_training_data ADD COLUMN IF NOT EXISTS rr_floor_applied   BOOLEAN;
ALTER TABLE synthetic_training_data ADD COLUMN IF NOT EXISTS data_quality_flags TEXT;

-- The pre-v8 generator stored entry close in `price`. We keep the column
-- to avoid breaking any old rows but the new generator doesn't write it.
-- (No-op DDL; left as a comment so the intent is clear in code review.)

-- The pre-v8 dataset's `regime` was wrong (SMA50/200 instead of EMA20/50+RSI).
-- Rather than try to fix in place, the recommended path is:
--   1. Run this migration.
--   2. python agent/build_training_data.py --purge
-- This wipes and regenerates everything with corrected logic.
