# Stock Agent — Consolidated Next Steps

This document supersedes the earlier `REVIEW.md` for *what to do*, while
that document remains the reference for *what was found*. Every item below
is anchored to evidence from the actual data tables you provided
(932 recommendations, 239 backsimulated outcomes, 50 parameter versions,
10 optimisation runs).

There are four phases. The first phase is "stop the bleeding" — small
deletions and gates that, based on your historical data, should immediately
raise win rate from ~68% to ~71-72% with roughly half the trade count.
The remaining phases are how you actually rebuild the scoring logic on a
sound footing.

---

## Phase 1 — Stop the bleeding (this week, ~1 hour of work)

### 1.1  Disable broken news scoring
**Why:** 43 of 62 "profit" headlines are mislabelled POSITIVE; news_score
correlation with outcome is +0.082, statistically indistinguishable from
zero; the multiplier currently amplifies false positives.
**How:** In `agent/analyze.py`, in `apply_score_multipliers()`, hardcode
`news_mult = 1.0`. Keep displaying headlines on the dashboard for context
but do not let them affect the score. Or use the new `agent/news_v2.py`
included alongside this document, which keeps headline display while
returning a neutral multiplier and pre-filtering junk headlines (option
chains, top-gainers lists, holiday notices).

### 1.2  Delete the dead and harmful strategies
**Why:** Of your six strategies, three are doing nothing or active harm.
Evidence from 932 recommendations and 239 backsim outcomes:

| Strategy | Fired (BUY) | Backsim WR | Verdict |
|---|---:|---:|---|
| Donchian | 626 | 67.7% (n=189) | **Keep — primary** |
| EMA Crossover | 110 | 66.0% (n=53) | Keep |
| RSI Trend Shift | 104 | 63.3% (n=49) | Marginal — keep for now |
| Volume Breakout | 112 | **58.6%** (n=29) | **Delete: hurts** |
| Bollinger | 23 | 66.7% (n=9) | Sample too small — keep but monitor |
| RSI + MACD | **0** | n/a | **Delete: never fires** |
| Supertrend | (not in strategy list) | n/a | **Delete display + computation** |

The two key proofs:

- *Donchian alone:* 69.4% WR over 160 trades.
- *Donchian + Volume Breakout:* 58.6% WR over 29 trades.
- The "confirmation" indicator removes 11pp of edge.

**How:** In `agent/analyze.py:get_strategies()`, return only Donchian, EMA
Crossover, RSI Trend Shift, Bollinger. In `analyze.py:context()`, delete
the Supertrend computation and the `supertrend_up` / `supertrend_line`
fields. In `analyze.py`, delete `sig_rsi_macd` and `sig_volume_breakout`
function definitions if you want a clean cut, or just remove them from
`get_strategies()` and let the dead code rot until the v8 cleanup.

### 1.3  Add a hard gate: only single-strategy signals
**Why:** Your data shows multi-strategy "confirmation" is anti-predictive.

| Strategies firing | Trades | Win rate | Avg return |
|---|---:|---:|---:|
| 1 | 170 | **70.6%** | +4.20% |
| 2 | 49 | 65.3% | +3.01% |
| 3 | 19 | 57.9% | +3.31% |
| 4 | 1 | 0% | -5.00% |

**How:** In `analyze.py:run()`, after computing `today_sigs`, add:

```python
buy_count = sum(1 for v in today_sigs.values() if v == 1)
if action == "BUY" and buy_count > 1:
    continue   # multi-strategy = anti-predictive in your data
```

This change alone, applied to your historical 727 BUY signals, would have
cut you to ~513 single-strategy signals with a WR jump from 68% to ~71%.

### 1.4  Add a hard regime veto
**Why:** Of the failing top-quartile signals (42% WR), 44% were in BEARISH
regime. The current ±10 point regime nudge is too soft — bearish-regime
signals ride other-component points to the top of the leaderboard, then
fail.

**How:** In `analyze.py:run()`, after computing regime:

```python
if action == "BUY" and regime_label == "BEARISH":
    continue   # hard veto on new longs in bearish regime
```

### 1.5  Pause the weekly optimiser
**Why:** Optuna is currently maximising the same composite score that we've
shown is anti-predictive (Pearson r = -0.262 vs win, p < 0.0001). Every
weekly run trains a worse model. The most recent run's claimed WR was
51.0% vs the historical actual of 68.2% — the optimiser is also drifting
downward.

**How:** Either disable the workflow file or comment out the cron line:

```yaml
# .github/workflows/weekly_optimize.yml
on:
  # schedule:
  #   - cron: "30 17 * * 0"
  workflow_dispatch:    # keep manual trigger for testing
```

Re-enable in Phase 2 once the scoring objective is fixed.

### 1.6  (already done) — Apply OHLC patch + holiday calendar + health check
These were the deliverables from the previous turn. Make sure they're
deployed:
- `agent/market_data.py`, the rewritten `agent/check_alerts.py`
- `dashboard/PATCHES.md` (3 small surgical edits to the dashboard)
- `supabase_migration_v7.sql`
- `agent/holiday_calendar.py`, `agent/health_check.py`
- `.github/workflows/daily_health_check.yml`

---

## Phase 2 — Replace the scoring (next 1–2 weeks)

This is where the real upgrade happens. Phase 1 is a damage-control patch;
Phase 2 builds a scoring system that genuinely predicts outcomes.

### 2.1  Build a calibrated probability model
**What:** Replace the hand-weighted `composite_score` with a logistic
regression (or gradient-boosted trees) trained on the
`backtest_simulations` table. Features are already in your data:
`n_strategies_firing`, `vol_ratio`, `regime_bearish` (binary),
`change_5d`, `risk_pct`, `reward_pct`, `news_score` (once Phase 3 fixes
news), `signal_streak`, `sector` one-hot, `market_cap_log`, etc.

**Why:** You have 239 labelled outcomes — enough to fit a 6–8 feature model
without overfitting. The model outputs a calibrated `P(win)` between 0
and 1. You then rank signals by `P(win) × expected_return`, which is what
expected value actually is. This replaces the current weighted-sum
heuristic with something that *learns* from outcomes.

**Implementation sketch:**

```python
# agent/score_model.py
import json
from datetime import date, timedelta
from supabase import create_client
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

def train_score_model(sb):
    sims = pd.DataFrame(sb.table("backtest_simulations").select("*").execute().data)
    recs = pd.DataFrame(sb.table("recommendations")
                          .select("id, signals, rsi, change_5d, risk_pct, reward_pct, "
                                  "market_regime, volume, avg_volume, signal_streak, news_score")
                          .execute().data)
    df = sims.merge(recs.rename(columns={"id": "recommendation_id"}),
                    on="recommendation_id")
    df["n_firing"] = df.signals.apply(_count_firing)
    df["vol_ratio"] = df.volume / df.avg_volume.replace(0, np.nan)
    df["regime_bearish"] = (df.market_regime == "BEARISH").astype(int)
    df = df.dropna(subset=["was_win", "vol_ratio", "rsi", "change_5d",
                            "risk_pct", "reward_pct", "n_firing"])
    feats = ["n_firing", "vol_ratio", "rsi", "change_5d", "risk_pct",
             "reward_pct", "regime_bearish", "signal_streak", "news_score"]
    X = df[feats].fillna(0)
    y = df.was_win.astype(int)
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf",   CalibratedClassifierCV(LogisticRegression(max_iter=1000), cv=5)),
    ])
    pipe.fit(X, y)
    return pipe, feats

def predict_p_win(pipe, feats, signal_features):
    X = pd.DataFrame([signal_features])[feats].fillna(0)
    return float(pipe.predict_proba(X)[0, 1])
```

Run weekly via a new GitHub Action that retrains on the latest
`backtest_simulations` rows and writes the model to a Supabase row (as
pickled bytes or to GitHub artifact storage). The live `analyze.py`
loads it at startup. Each new closed paper trade adds a fresh training
point.

**Validation:** before promoting v2 of the scoring, hold out the most
recent 3 weeks of trades, fit the model on the older trades only, and
verify that the LR model's quartile WR is monotonic and >65% in Q4 (where
the current composite is at 42%).

### 2.2  Switch from win-rate-weighted vote to expectancy-weighted
**Why:** A 30% WR strategy with 4R wins beats a 70% WR strategy with 0.5R
wins. `weighted_vote()` currently uses win rate. `optimize.py` already
computes `expectancy` per strategy — port that calculation into
`analyze.py.backtest()` and use it in the vote.

**How:** In `analyze.py:backtest()`, return `expectancy` alongside
`win_rate`. In `weighted_vote()`:

```python
weights[name] = round(max(bt.get("expectancy", 0), 0) * conf, 3)
```

This change is moot once 2.1 is done (the LR model implicitly learns
expectancy), but for the interim it makes the legacy scoring less wrong.

### 2.3  Fix the "BT_SL_PCT / BT_TARGET_PCT" optimiser noise
**Why:** Optuna spends search budget on these two parameters, but they're
only used in the *fallback* path of `dynamic_trade_levels`, which fires
rarely (your data shows mean risk_pct ~5.4% with std 1.16, so the
ATR-based path dominates). The optimiser is fitting noise.

**How:** Remove `BT_SL_PCT` and `BT_TARGET_PCT` from the Optuna search
space in `optimize.py`. Hard-code as 5.0 and 10.0 (which is what the
v21 champion converged on anyway). Now optimisation focuses only on
the parameters that actually affect outcomes.

### 2.4  Re-tune optimiser objective for actual edge
**Why:** Once 2.1 is in, the objective should be the LR model's holdout
log-loss (or AUC), not a hand-crafted composite of profit factor + win
rate + drawdown. Switching the optimiser objective to "AUC of trained
model on holdout" makes Optuna optimise for *predictive power* rather
than for *historical performance numbers*.

---

## Phase 3 — Add features that actually predict (3–4 weeks out)

### 3.1  Momentum vs Nifty (relative strength)
**Why:** Most-cited equity factor in academic literature; your system
doesn't have it.
**How:** Compute `stock_return_30d - nifty_return_30d`. Store in
`recommendations`. Use as a hard filter (only take signals with > 0)
in Phase 1.5 mode, or as a feature in the LR model in 2.1.

### 3.2  Trend-quality filter (ADX)
**Why:** Donchian breakouts in non-trending markets are whipsaws. ADX > 20
is the textbook quality filter; in your context-bearish backsims, lots
of failures are likely choppy-market false breakouts.
**How:** Add `def adx(...)` next to the other indicator helpers in
`analyze.py`. Return as part of `context()`. Use as a feature.

### 3.3  Multi-timeframe trend confirmation
**Why:** A daily-bar Donchian breakout is more reliable when the
weekly trend is also up. Cheap to add.
**How:** Resample `df` to weekly, compute `weekly_ema_50` and
`weekly_ema_200`, take long signals only when `weekly_ema_50 >
weekly_ema_200`. As a feature in 2.1's LR.

### 3.4  Real news scoring (FinBERT or domain LR)
**Why:** Phase 1 disabled news; this re-enables it correctly.
**How:** Two paths.
- *Easy:* set `HF_TOKEN` secret in GitHub Actions; FinBERT path will
  activate. Combine with the junk-headline filter (in
  `agent/news_v2.py` deliverable).
- *Better:* train a TF-IDF + LR classifier on the 789 headlines you've
  already collected. Manually label 200 of them (couple hours of work),
  fit, validate, deploy. Will beat both VADER and FinBERT for Indian
  market terminology.

### 3.5  India VIX as a regime input
**Why:** When INDIAVIX > 25, breakout strategies under-perform; mean
reversion strategies outperform. INDIAVIX is on yfinance as `^INDIAVIX`.
Two lines to fetch, one feature to add.

---

## Phase 4 — Validation and infrastructure (4–8 weeks out)

### 4.1  Position sizing with fixed-fractional risk
The single largest improvement to risk-adjusted returns. Detailed in
REVIEW.md §C1.

### 4.2  Realistic Indian costs in *all* backtest engines
Currently `optimize.py` deducts 0.15%, others deduct 0%. Real cost is
~0.35–0.45% delivery. Detailed in REVIEW.md §B2.

### 4.3  Walk-forward optimiser with frozen holdout
Currently the "out-of-sample" data is reused across windows. Detailed in
REVIEW.md §B3 + §B4.

### 4.4  Survivorship-bias-aware optimisation universe
Use the 2020 Nifty 100 composition for optimisation, not today's. Detailed
in REVIEW.md §B1.

### 4.5  Reconciliation table population
Compare every closed paper trade against its backsim prediction; surface
divergence in the dashboard. Schema is ready (v7 migration). Code in
REVIEW.md §E5.

---

## Sequencing guide — what to do in what order

If you have **2 hours this weekend**, do all of Phase 1. The data-driven
gates and deletions take ~30 minutes to apply, ~30 minutes to test, and
will improve win rate immediately.

If you have **a weekend or two**, add Phase 2.1 (the LR scoring model).
This is where the system becomes data-driven. Total ~150 lines of code
including the training workflow.

If you have **a month**, work through Phase 3 features in any order.
ADX and multi-timeframe are the cheapest; momentum-vs-Nifty has the
strongest academic support; news is the most user-visible.

Phase 4 is for when you're considering moving to real money. Don't
attempt it until Phases 1–3 have produced 3+ months of clean live results
that match backsim within ±5pp WR.

---

## What this plan explicitly does *not* do

- **Add more strategies.** You already have too many overlapping ones.
  The fix is consolidation, not expansion.
- **Move to ML for everything.** A calibrated LR is all you need for
  signal scoring at this scale. GBM / XGBoost give marginal gains;
  deep learning is unnecessary for 239 trades.
- **Add intraday data.** Daily OHLC is sufficient for swing trades held
  10–15 days. Adding intraday only helps for same-bar SL/target
  ambiguity, which is a marginal effect.
- **Add options strategies.** Out of scope for an equity-cash agent;
  fundamentally different modelling.
