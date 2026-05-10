# Changelog — v8 cleanup

## v8.1 — synthetic-data generator post-mortem fixes

A user-reported audit of `synthetic_training_data` revealed four bugs in
the pre-v8 generator (`build_training_data.py`) that silently invalidated
any model trained on that dataset. All four are fixed in v8.1, with a
self-check harness that makes them structurally impossible to reintroduce:

### Bugs fixed

1. **Regime-label mismatch.** The pre-v8 generator labelled `regime` using
   SMA50/SMA200 cross. The live agent uses EMA20/EMA50 + RSI. The model's
   `regime_score` weight was therefore learned on a feature that didn't
   exist at prediction time. **Fix:** extracted `regime_from_close_series()`
   into `analyze.py` as a pure function; both modules import it. The
   self-check verifies they're the same function object.

2. **Entry-price look-ahead.** The pre-v8 `simulate_forward()` used
   `signal_day Close` as the entry price. The live agent enters at
   `signal_day+1 Open`. NSE overnight gaps average ~0.4% (occasionally
   2%+), so labels were biased relative to production execution.
   **Fix:** `simulate_forward()` now enters at next-bar open. Records the
   `gap_pct` for diagnostic purposes. Handles the edge case where the gap
   already breaches the SL on the entry bar's open (`exit_reason = 'gap_stop_at_open'`).

3. **VIX dual implementation.** Two different code paths computed nominally
   the same VIX features. Easy place for silent drift over time.
   **Fix:** extracted `vix_features_from_series()` into `analyze.py`; both
   modules import it.

4. **Missing MFE bar number.** The pre-v8 generator stored `mfe` and `mae`
   as scalars but discarded the bar number where each was reached. This
   prevented research on trailing-stop rules like "trail SL once MFE > 5%
   was reached at bar X". **Fix:** records `mfe_bar` and `mae_bar`.

### New columns in `synthetic_training_data`

`signal_close`, `entry_price`, `sl_price`, `target_price`, `gap_pct`,
`mfe_bar`, `mae_bar`, `rr_floor_applied`, `data_quality_flags`.

### Data-quality flags

The new generator tags noisy rows with a CSV of flags:
- `borderline_timeout` — return within ±1%, label is essentially noise
- `gap_stop` — instant stop on gap-down at entry open
- `very_low_vol` — ATR < 1%, ATR-based SL is unreliable
- `large_gap` — |signal-to-open gap| > 2%

`load_training_data()` in `score_model.py` now excludes these by default
during training (configurable via the `exclude_quality_flags` argument).

### Module-level changes

- **`analyze.py`**: Supabase client init is now lazy. The module is
  importable without `SUPABASE_URL`/`SUPABASE_KEY` set, which means it can
  be imported by tests. `run()` fails fast with a clear error if the
  client wasn't created.
- **`build_training_data.py`**: full rewrite. New `--self-check` flag runs
  4 structural correctness tests (none of which require Yahoo or
  Supabase access).
- **`score_model.py`**: `load_training_data()` accepts an
  `exclude_quality_flags` parameter for filtering noisy rows.
- **`supabase_setup.sql`**: canonical schema includes all v8.1 columns.
- **`supabase_migrations/002_v8_training_data.sql`**: idempotent migration
  that adds the new columns to existing deployments.

### Migration path

For users who have a populated `synthetic_training_data` table from before
v8.1, the recommended path is:

1. Run `supabase_migrations/002_v8_training_data.sql` in Supabase SQL Editor.
2. `python agent/build_training_data.py --purge` to wipe and regenerate
   with corrected logic.
3. `python -m agent.score_model` to retrain on the clean dataset.

The `--purge` step is necessary because the pre-v8 rows have a different
`regime` definition and entry-price semantics than the new ones; mixing
them would produce a model trained on inconsistent feature distributions.

### What this means for production

Models trained on the pre-v8 dataset should be considered untrustworthy.
The old generator's biases produced inverted relationships in some
features (e.g. `rr_ratio` had Pearson correlation **-0.14** with `was_win`
in the old data — opposite of intuition — driven by the MIN_RR_RATIO
floor stretching targets on tight-SL setups, which were then dominated by
stop-outs). These inverted relationships are mathematical artifacts of
the generator, not market truths. They will go away with v8.1's data.

---

## v8.0 — initial cleanup

This release is a senior-engineer / senior-quant pass over the v7 +
Phase 3 codebase. The goal was: reduce surface area, fix real
bugs, and add the missing risk primitives — without changing the
core signal logic that already works (68 % WR, +3.85 % avg in
production backsim).

---

## Added

### `agent/risk.py` (new)
- `suggest_qty(capital, risk_pct, entry, sl)` — fixed-fractional
  sizing with a 10 % notional cap.
- `regime_size_mult(regime)` — 1.0 / 0.75 / 0.50 for
  BULLISH / NEUTRAL / BEARISH+UNKNOWN.
- `sector_exposure_pct(...)` & `sector_room_remaining_pct(...)`
  — sector concentration cap at 30 %.
- `drawdown_brake(recent_pnls)` — soft brake (×0.5) at -5 %,
  hard brake (×0.0) at -10 % cumulative across the last 20 trades.
- `recommend_position_size(...)` — combines all four into one call,
  returning a dict with the suggested qty, gates that fired, and a
  human-readable trail.

### `agent/portfolio.py` (new)
- Centralises every paper-portfolio query that was previously
  re-written inline in `analyze.py`, `app.py`, and `check_alerts.py`.
- `list_open_positions(sb)`, `open_tickers(sb)`,
  `recent_closed_pnl_pcts(sb, n=20)`,
  `open_positions_with_sector(sb)`, `stale_open_positions(sb)`.

### `recommendations.suggested_qty` / `size_blocked` / `size_reasons`
- Three new columns persist the v8 sizing recommendation per BUY
  row. The dashboard's paper-buy form pre-fills `qty` from
  `suggested_qty` so the user is anchored to a risk-aware default.
  The form still allows editing.

### EXIT-on-holding alert
- When today's run produces an `action='EXIT'` row for a ticker the
  user currently holds, it surfaces:
  * In Telegram as a top-of-message ⚠️ banner.
  * In the dashboard portfolio page as a red banner above
    "Open Positions".
- This is REVIEW.md A3 — closed without auto-closing the position.

### `supabase_migrations/`
- New numbered-migration directory.
- `001_v7_to_v8.sql` adds the three sizing columns,
  enforces single-champion via partial unique indices on both
  `agent_params` and `score_models`, and creates
  `synthetic_training_data` if it doesn't exist.

### `CHANGELOG.md` (new — this file)

---

## Changed

### `agent/analyze.py`
- **Banner:** "Stock Agent v8" (was "v6" — the print string had
  fallen out of sync with reality across two releases).
- **Strategy memoisation** (REVIEW G1, perf): each ticker's strategy
  series is computed once and reused for both the today-signal
  read and the per-strategy backtest. Was 2× per ticker; now 1×.
  ~50 % faster per-ticker.
- **`weighted_vote()` confidence floor** (REVIEW B5): strategies
  with <5 historical trades now get 0 weight (was: weight by
  win-rate alone, which let a 100 % WR / 1-trade strategy
  dominate). Confidence ramps linearly to 1.0 at ≥30 trades.
- **`context()` minimum bars** (REVIEW G1): now `max(60,
  EMA_LONG×2)` (was 20). Many indicators aren't warmed up before
  60 bars and were producing noise.
- **`backtest()` returns `expectancy`** (per-trade
  expected return = WR × avg_win - (1-WR) × avg_loss). Useful
  alongside `profit_factor` and now consumed by the optimizer.
- **`dynamic_trade_levels()` MIN_RR_RATIO floor** is now applied
  on both the ATR path AND the ATR-fallback path (was
  ATR-path only). Prevents fallback setups from being published
  with R:R < the configured floor.
- **Dropped writes for dead columns** (`supertrend_up`,
  `supertrend_line`, the legacy `score`/`streak` keys). These
  fields were never read anywhere.
- **`NIFTY50` renamed to `NIFTY200`** (the variable was
  badly-named — its contents were the Nifty 200, not the 50).
  A back-compat alias `NIFTY50 = NIFTY200` is kept so external
  imports (`build_training_data.py`) continue working.
- **Pre-loads portfolio context once per run** (`held_tickers`,
  `open_pos_w_sector`, `recent_pnls`) instead of querying inside
  the per-ticker loop.

### `agent/optimize.py`
- **Strategy roster aligned with live agent.** Removed:
  * `RSI + MACD` — fired 0 times in 932 production recommendations.
  * `Volume Breakout` — anti-predictive when paired with Donchian.
  * `Supertrend` — was never a live strategy, only a display field.
- **Search space cleaned:**
  * Removed `SUPERTREND_MULT` (unused).
  * Removed `BT_SL_PCT`, `BT_TARGET_PCT` from search — these are
    last-resort fallbacks only used when ATR is missing. The live
    agent always uses the ATR path. Searching them was searching
    dead code. Optimizer now focuses on `ATR_*_BUFFER` /
    `MAX_RISK_*` / `MIN_RR_RATIO`.
  * Pinned `VOLUME_MULT` (unused with the v7+ strategy roster).
  * Pinned `RSI_OVERSOLD` / `RSI_OVERBOUGHT` / `MACD_*` (the
    strategies that consumed them are gone).
- **Promotion gate raised from ×1.05 → ×1.10.** A challenger must
  beat the champion's objective_score by ≥10 % to be eligible for
  human-review promotion. Helps prevent regime-flip overfitting.
- **Cron remains DISABLED** in `weekly_optimize.yml`. See header
  comment in that file for re-enable conditions.

### `agent/score_model.py`
- **Honest imputation for low-coverage features.** A column whose
  coverage is <5 % is set to `0.0` for all rows rather than
  median-imputed. (Median of an all-NaN series is NaN, which we
  coerce to 0.0; the previous behaviour silently fed the model a
  constant-zero column, polluting feature importance.)
  Most relevant for VIX features in early-rollout deployments
  where `vix_level` is 100 % NaN — the model now politely ignores
  them instead of pretending they're informative.
- **Coverage diagnostic in `main()`.** Prints any feature with
  <50 % coverage before training, with a "DROPPED" tag for
  features <5 %.
- **Train + test + full-data refit all use the same coverage
  mask** so the served model's imputation policy matches what
  was validated on holdout.

### `dashboard/app.py`
- **Removed the hardcoded password fallback** (`"stockagent123"`)
  — REVIEW A5. If `DASHBOARD_PASSWORD` isn't set in Streamlit
  secrets, the dashboard refuses to load with a clear error.
- **Live-price fallback no longer lies.** When yfinance fails for
  an open position, P&L now displays "—" and the position
  expander shows a yellow "Live price could not be fetched"
  banner. Previously, `lp = row.buy_price` made the P&L appear
  as 0.00 % even if the position was actually deeply red — REVIEW
  A4 underlying issue.
- **Paper-buy form pre-fills `qty` from `suggested_qty`.** The
  default `value=10` is replaced with the agent's risk-aware
  recommendation. A "Sizing rationale" expander shows the
  reason trail (regime mult, drawdown brake, sector cap).
- **EXIT-on-holding banner** added to the portfolio page —
  mirrors the Telegram alert.

### `agent/telegram_alerts.py`
- Added a single `_esc()` HTML-escape helper and applied it to
  every user-facing string. Previously tickers like `M&M` and
  `M&MFIN` rendered raw, breaking Telegram's HTML mode.

### `supabase_setup.sql` (full rewrite)
- One canonical schema. No more concatenated v2-install +
  v2→v5 + v5→v6 + v6→v7 patches.
- Removed the running `UPDATE agent_params ...` block at the
  bottom of the old file — it was overwriting optimizer
  learnings every time the SQL was run.
- All schema-drift columns (added by-hand in production) now
  formally defined: `signal_type`, `composite_score_raw`,
  `benchmark_symbol` removed (never read); all v7+ columns
  declared.
- Partial unique index `uniq_agent_params_one_champion` enforces
  "at most one champion at a time" — defends against the
  multiple-champion bug class.
- Same partial unique index on `score_models`.
- `synthetic_training_data` is now formally part of the schema
  (was created by-hand for some users).
- All RLS explicitly disabled (was inconsistent across tables).

### `README.md`
- Rewritten for v8: file inventory now matches reality, env vars
  for capital + risk are documented, position sizing is explained,
  optimizer-disabled state is called out.

### `.github/workflows/weekly_optimize.yml`
- Updated header comment to reflect v8 status. Cron remains
  disabled until Phase 3 LightGBM is verified with holdout AUC
  > 0.55 across one walk-forward window.

---

## Removed

- `INSTRUCTIONS.md`, `REVIEW.md`, `NEXT_STEPS.md` — these were
  v6→v7 migration scratchpads. Their relevant content is now
  folded into `CHANGELOG.md` and `README.md`. Historical
  diagnostics (REVIEW.md) are preserved in git history.
- `supabase_migration_v7.sql` — superseded by
  `supabase_migrations/001_v7_to_v8.sql`.
- Dead columns in writes: `supertrend_up`, `supertrend_line`,
  legacy `score`/`streak` field aliases.
- `sig_volume_breakout()` and `sig_rsi_macd()` from `optimize.py`
  — strategies that haven't been part of the live roster since
  v7 and were polluting the search space.

---

## Migration Path

- **Fresh install:** run `supabase_setup.sql` once. Done.
- **Existing v7 deployment:** run
  `supabase_migrations/001_v7_to_v8.sql`. Idempotent — safe to
  re-run.
- **No breaking changes** to the dashboard's existing reads;
  three new columns are additive.
- **Score model:** existing Phase 2 LR models will continue to
  load and predict. The Phase 3 LGBM upgrade is independent of
  this cleanup; train when ready via `python -m agent.score_model`.

---

## Production Data Observations Driving v8

(For context on why specific changes were made.)

| Observation | Implication | v8 response |
|---|---|---|
| 935 recommendations, `vix_level` NULL in 100 % | Phase 3 not yet deployed | `score_model.py` now drops near-zero-coverage features cleanly |
| Champion AUC 0.553 (n=1 in Q4) | Calibration broke on tied predictions | qcut already had a duplicates='drop' fallback; coverage gate added |
| `composite_score` r = -0.262 with realised win | Optimizer drift was real | Promotion gate raised to ×1.10; cron stays disabled |
| `pnl_pct` NULL in 919 / 935 | p_win column rolled out very recently | No code change needed; future trades populate it |
| Schema drift: `signal_type`, `composite_score_raw`, `benchmark_symbol` | Hand-added, never read | Removed from canonical schema |
| 68.2 % WR / +3.85 % avg in 239 backsim trades | Core signal logic actually works | **Don't change strategies; tighten everything around them** |
