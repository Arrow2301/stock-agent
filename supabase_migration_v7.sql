-- ============================================================
--  Phase 3 — Migration v9
--
--  Adds India VIX feature columns to both training and serving
--  tables. Idempotent; safe to run multiple times.
--
--  Run this BEFORE deploying the new code. Order:
--    1. This migration                 (adds columns, no data change)
--    2. Update requirements.txt        (lightgbm)
--    3. Replace the three agent .py    (build_training_data, score_model, analyze)
--    4. Trigger build_training_data    (--purge to rebuild with VIX)
--    5. Trigger retrain_score_model    (produces LGBM champion)
--    6. Verify next daily run prints "Score model loaded: <new fingerprint>"
-- ============================================================

ALTER TABLE synthetic_training_data
  ADD COLUMN IF NOT EXISTS vix_level      numeric,
  ADD COLUMN IF NOT EXISTS vix_change_5d  numeric,
  ADD COLUMN IF NOT EXISTS vix_zscore_60d numeric;

ALTER TABLE recommendations
  ADD COLUMN IF NOT EXISTS vix_level      numeric,
  ADD COLUMN IF NOT EXISTS vix_change_5d  numeric,
  ADD COLUMN IF NOT EXISTS vix_zscore_60d numeric;

-- Quick visual verification:
--   SELECT column_name, data_type
--   FROM information_schema.columns
--   WHERE table_name = 'synthetic_training_data'
--     AND column_name LIKE 'vix_%';
