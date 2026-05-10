# 🇮🇳 Indian Stock Agent — v8

A daily NSE signal-generation pipeline + paper-trading dashboard.
100 % free hosting (GitHub Actions + Supabase + Streamlit Cloud).

> ⚠️ **Paper trading only.** Not financial advice. See disclaimer at the bottom.

---

## Architecture

```
┌────────────────────────┐
│ GitHub Actions cron    │  Mon-Fri 07:00 IST  →  agent/analyze.py
│   (also: weekly        │  Mon-Fri 12:00 IST  →  agent/check_alerts.py (midday SL/target check)
│    backsim, weekly     │  Mon-Fri 19:00 IST  →  agent/health_check.py
│    optimizer*, build   │  Sun   23:00 IST    →  agent/backsimulate.py
│    training data,      │  Sun   23:00 IST    →  agent/optimize.py     *cron disabled, manual only
│    retrain model)      │  Sat   02:00 IST    →  agent/build_training_data.py
└──────────┬─────────────┘  Sat   04:00 IST    →  agent/score_model.py
           │
           ▼ writes signals + meta + breach flags
┌────────────────────────┐
│ Supabase (PostgreSQL)  │
│ recommendations         │
│ paper_portfolio         │
│ agent_meta / params     │
│ score_models / synth_*  │
│ backtest_simulations    │
└──────────┬─────────────┘
           │
           ▼ reads + paper trades
┌────────────────────────┐
│ Streamlit dashboard    │  password-gated
│ dashboard/app.py        │
└────────────────────────┘
```

---

## Quick Start

1. **Create a Supabase project**, open SQL Editor, paste **`supabase_setup.sql`**, run it.
2. **Push the repo to GitHub.** Set repository secrets:
   - `SUPABASE_URL`, `SUPABASE_KEY`
   - `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (optional, for daily alerts)
   - `HF_TOKEN` (optional, enables FinBERT for news sentiment instead of VADER)
   - `ACCOUNT_INR` (optional, default `100000`) — base capital for the
     position-sizing recommendation
   - `RISK_PCT_PER_TRADE` (optional, default `1.0`) — % of capital risked per trade
3. **Run "Daily Stock Analysis" workflow once manually** to verify it writes to Supabase.
4. **Deploy the dashboard** on Streamlit Cloud pointing at `dashboard/app.py`.
   In Streamlit secrets set `SUPABASE_URL`, `SUPABASE_KEY`, and a strong
   **`DASHBOARD_PASSWORD`** (this is now mandatory — no fallback).

---

## What the Agent Does Each Morning

For every Nifty 200 ticker:

1. **Fetch 430 days of OHLCV** from Yahoo Finance (`<TICKER>.NS`).
2. **Compute four strategies** (Donchian breakout, EMA crossover, RSI trend
   shift, Bollinger). Strategy series are computed *once* and reused for
   today's signal + the per-strategy historical backtest.
3. **Hard gates:** reject BUYs that fire on >1 strategy (multi-strat is
   anti-predictive in production data) and BUYs in a BEARISH NIFTY regime.
4. **Confidence-weighted vote:** weight each strategy by `win_rate ×
   confidence`, where confidence ramps from 0 at <5 historical trades to
   1.0 at ≥30 trades.
5. **Composite score** (0-100) blends strategy vote, RSI, volume, R:R and
   regime, then is multiplied by fundamental and news multipliers.
6. **Phase 3 P(win)** from the calibrated LightGBM scorer (if a champion
   model is deployed in `score_models`).
7. **Position-sizing recommendation** (new in v8) using fixed-fractional
   risk + regime size mult + drawdown brake + sector cap. Stored as
   `suggested_qty` on each recommendation row, surfaced in both Telegram
   and the dashboard's paper-buy form.
8. **EXIT-on-holding alert** (new in v8): if today's EXIT signal hits
   a stock you currently hold, both Telegram and the dashboard portfolio
   page surface a banner. *No auto-close.*

---

## Position Sizing (new in v8)

`agent/risk.py` exposes:

| Function | Purpose |
|---|---|
| `suggest_qty(capital, risk_pct, entry, sl)` | Fixed-fractional sizing. Caps single-position notional at 10 % of capital. |
| `regime_size_mult(regime)` | 1.0 / 0.75 / 0.50 for BULLISH / NEUTRAL / BEAR/UNKNOWN |
| `drawdown_brake(recent_pnls)` | Reduces sizing to 0.5× at -5 % cumulative, 0× at -10 %. |
| `sector_room_remaining_pct(...)` | Cap at 30 % of capital per sector. |
| `recommend_position_size(...)` | Combines all four into one call. |

Defaults: ₹100,000 account, 1 % risk per trade. Override via `ACCOUNT_INR`
and `RISK_PCT_PER_TRADE` env vars (in workflow + Streamlit secrets).

---

## Customising the Watchlist

In `agent/analyze.py`:

```python
EXTRA_WATCHLIST = [
    "TATAPOWER",   # adds beyond the Nifty 200
]
```

`NIFTY200` itself is a literal list — edit it if NSE re-indexes. There's a
back-compat alias `NIFTY50 = NIFTY200` for old imports; the contents are
the Nifty 200 despite the name.

---

## Adjusting Signal Sensitivity

| Var (in `DEFAULT_PARAMS`) | Default | Effect |
|---|---|---|
| `MIN_WEIGHTED_SCORE` | 0.08 | Higher → fewer, higher-conviction signals |
| `RSI_OVERSOLD` / `RSI_OVERBOUGHT` | 48 / 58 | Buy/sell trigger thresholds |
| `MIN_RR_RATIO` | 1.50 | Reject setups with reward < 1.5× risk |
| `BT_MAX_HOLD` | 15 | Backtest max bars; affects historical win rate |

When the optimizer is enabled and finds a champion, those values come from
`agent_params` table and override `DEFAULT_PARAMS`. The optimizer cron is
**disabled by default** in `weekly_optimize.yml` until Phase 3 is verified
live (see file's header comment).

---

## File Structure

```
stock-agent/
├── .github/workflows/        Cron-driven jobs
│   ├── daily_analysis.yml    Mon-Fri 07:00 IST
│   ├── midday_alert.yml      Mon-Fri 12:00 IST
│   ├── daily_health_check.yml Mon-Fri 19:00 IST
│   ├── weekly_backsimulate.yml Sun 23:00 IST
│   ├── weekly_optimize.yml   manual only — see header
│   ├── build_training_data.yml Sat 02:00 IST
│   └── retrain_score_model.yml Sat 04:00 IST
├── agent/
│   ├── analyze.py           Daily run (the main file)
│   ├── check_alerts.py      Midday SL/target breach scanner
│   ├── health_check.py      Daily observability snapshot
│   ├── backsimulate.py      Walk-forward simulator for closed recs
│   ├── optimize.py          Optuna walk-forward param search (4-strategy roster)
│   ├── score_model.py       Calibrated LGBM (Phase 3) train + serve
│   ├── build_training_data.py Generate synthetic_training_data rows
│   ├── market_data.py       Shared OHLC + breach detection
│   ├── news_v2.py           Junk filter + financial keyword overrides
│   ├── holiday_calendar.py  NSE holidays (update yearly)
│   ├── telegram_alerts.py   Helper for Telegram messages
│   ├── risk.py              ★ v8: position sizing primitives
│   └── portfolio.py         ★ v8: shared paper-portfolio queries
├── dashboard/app.py         Streamlit dashboard
├── supabase_setup.sql       Canonical schema (single install)
├── supabase_migrations/     Numbered upgrade migrations
│   └── 001_v7_to_v8.sql
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

---

## Troubleshooting

| Symptom | Likely Cause |
|---|---|
| GH Action: "ticker returned no data" | Yahoo rate-limited a few tickers — the rest still complete. Safe to ignore. |
| Dashboard: "No signals today" | Agent may not have run. Trigger manually via Actions. |
| Dashboard: "Live unavailable" on a position | yfinance failed; refresh after a minute. P&L shown as "—" rather than misleading the user with buy_price. |
| Phase 3 model "AUC 0.55X" | Likely too few rows or VIX coverage <5 %. `score_model.py` now drops near-empty features automatically and reports coverage in the training log. |
| Champion params look stale | Optimizer is intentionally disabled; manually trigger `Weekly Parameter Optimization` after deploying Phase 3 LGBM. |
| `synthetic_training_data` table missing | Run `supabase_migrations/001_v7_to_v8.sql` (creates the table if missing). |

---

## ⚠️ Disclaimer

Educational and paper-trading use only. **Not financial advice.** This
agent backtests well in some market regimes and badly in others; past
performance is not predictive. Never invest more than you can afford to
lose. Always cross-check signals with your own research and start with
paper trading for at least 4–6 weeks before using real capital.
