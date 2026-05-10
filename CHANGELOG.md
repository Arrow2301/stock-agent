# Changelog — v8 cleanup

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
