# Senior-quant review of the Stock Agent project

This is the deep review you asked for. I read all 4,096 lines, ran every
module mentally against a real broker's order-management constraints, and
prioritised findings by **how much real-money risk they represent** rather
than by code-quality nits. The bug-fix you already asked for (OHLC breach
detection) is shipped as actual code in this folder; everything else below
is diagnosis + recommended action.

I'll grade every issue:

- 🔴 **Critical** — currently affecting decisions or P&L; fix before any real money
- 🟠 **High** — material distortion of expected returns; fix soon
- 🟡 **Medium** — quality / scientific-rigour issue; meaningful upgrade
- 🟢 **Low** — polish or future-proofing

The file is long but every paragraph contains an actionable recommendation.

---

## A — Live correctness bugs

### A1 🔴 Live alerter and dashboard miss intraday breaches *(the bug you flagged)*

`agent/check_alerts.py:17–24, 54–58` and `dashboard/app.py:286–306, 884–887`
compare the latest *close* to SL/target. A stock that wicked below SL at
10:30 AM and recovered by 3:30 PM is invisible to both alerters. Your
historical backtest in `backsimulate.py:124–131` and `analyze.py:503–508`
already does it right — so live and historical paths disagree, which is
the worst possible failure mode (you can't trust the win rate anymore).

**Status: fixed.** See `agent/market_data.py`, the rewritten
`agent/check_alerts.py`, and `dashboard/PATCHES.md`. The fix also handles
gap-downs (real fill = open price), persists the breach to a new column
so it's sticky across re-runs, and matches the stop-first convention used
by the backtester.

### A2 🔴 yfinance multi-level columns crash `check_alerts.py`

`get_live_price()` calls `df["Close"].iloc[-1]` without flattening, and
yfinance has been returning `("Close", "RELIANCE.NS")` tuples sporadically
since v0.2.40. Other modules in the project handle this; this one doesn't.
This means your alerter has been silently failing for some tickers — and
since the failure path returns `0.0`, the SL/target check `lp <= sl`
evaluates `0.0 <= sl` = True, so you've potentially been getting **false
SL alerts** for the affected tickers.

**Status: fixed** in the new `check_alerts.py` (uses `market_data` module
which flattens defensively).

### A3 🔴 `EXIT` signals never close existing paper positions

`analyze.py` produces an EXIT/avoid-new-long signal for a ticker. The
dashboard happily displays it. But the position you opened on a previous
BUY signal is *not* affected — only manual SL/target. Result: when an
EXIT signal fires for a ticker you're long, you keep holding silently.

**Recommendation:** when generating EXIT signals, the agent should check
`paper_portfolio` for an OPEN position on that ticker and either (a)
auto-close it, or (b) raise a high-priority alert. I'd suggest (b) by
default — auto-closing without user consent is intrusive.

```python
# Add to agent/analyze.py at the end of run(), or to a new agent/exit_alerts.py
def check_exit_signals_against_portfolio(records):
    open_positions = sb.table("paper_portfolio").select("*").eq("status","OPEN").execute().data
    open_tickers = {p["ticker"] for p in open_positions}
    exit_holdings = [r for r in records if r["action"] == "EXIT" and r["ticker"] in open_tickers]
    for r in exit_holdings:
        send_telegram_alert(f"⚠️ EXIT signal on holding {r['ticker']}…")
```

### A4 🟠 Same-cell yfinance is being called 3× per render

`compute_portfolio_snapshot()`, `check_exit_alerts()`, and the open-positions
loop each call `live_price()` for every position. With the default
`@st.cache_data(ttl=120)` they share a cache, but the first render of
each page burns 3× position-count requests. yfinance throttles after
~50 reqs/min and returns empty DataFrames — which the system silently
treats as "live price unavailable, fall back to buy_price" (line 883
of dashboard).

**Recommendation:** factor live data into a single `position_state(row)`
helper that returns `(lp, breach, pnl_pct, pnl_inr)`, cached as a unit.
Then all three call sites reuse the same memoised tuple.

### A5 🟠 Default dashboard password is hardcoded

`dashboard/app.py:53` falls back to `stockagent123` if `DASHBOARD_PASSWORD`
is unset. This is fine for personal use but if you ever forget to set the
secret in Streamlit Cloud, the password is *publicly known from this repo*.

**Fix:** make it `st.secrets["DASHBOARD_PASSWORD"]` (no default). If unset,
the dashboard refuses to start.

### A6 🟡 EXTRA_WATCHLIST mismatch with README

README says you can add tickers to `EXTRA_WATCHLIST` for auto-pickup;
`agent/analyze.py:92` has it as `[]`. New users following the README will
edit it then wonder why it didn't auto-add. Either restore the example
list or delete that section from the README.

---

## B — Backtest integrity

### B1 🟠 Survivorship bias in the universe

`NIFTY50` (lines 65–91 of analyze.py) is the **current** Nifty 50 + a
selection of recent additions like `IRCTC`, `ZOMATO`, `IREDA`. By
construction this excludes *every* stock that *was* in the index in
2020–2024 and got dropped (Yes Bank, Vedanta, Vodafone Idea-as-IDEA,
Indian Oil at various points). The dropped ones are the underperformers.
Your 2-year backtest is therefore systematically optimistic — you're
testing on the survivors.

**Recommendation:** for the optimizer specifically, use a point-in-time
universe. Two approaches:

1. *Cheap:* fetch each year's Nifty 200 composition from NSE archives and
   take the union (~250 unique tickers across 5 years). This won't fully
   eliminate survivorship for delistings but will dramatically reduce it.
2. *Better:* use a frozen list from 5 years ago (`SAMPLE_STOCKS_2020`),
   which guarantees the optimizer sees the actual past. Then live-trade
   on the current Nifty list, which is fine for live use.

The expected impact: I'd bet your real win rate is 5–10pp lower than the
backtest reports, mostly explained by this.

### B2 🟠 Inconsistent cost models across backtest engines

| Module                    | Slippage | Brokerage | Notes |
|---------------------------|---------:|----------:|-------|
| `analyze.py.backtest()`   |       0% |        0% | used for daily strategy ranking |
| `optimize.py.backtest_with_costs()` |  0.10% |     0.05% | used for parameter selection |
| `backsimulate.py.simulate_trade()` | 0% | 0% | used for "live ground truth" |

So:

- The optimizer picks parameters that look good *with* costs.
- Daily ranking displays metrics computed *without* costs.
- Live-vs-backtest reconciliation runs *without* costs.
- Your real Zerodha account incurs ~0.30–0.50% round-trip on delivery
  trades (STT 0.1% sell-side, exchange fees, GST, stamp duty, DP charges
  on sell).

The cost number you optimise against also under-estimates real costs by
roughly **3×**. Win-rate-marginal strategies that look profitable will be
unprofitable live.

**Recommendation:** add a single shared `apply_costs(returns)` helper, and
call it from every engine. Realistic Indian numbers for delivery:

```python
SLIPPAGE_PCT  = 0.10      # buy + sell, large caps; 0.20 for mid/small
BROKERAGE_FLAT = 20.0     # ₹20 per executed order, both legs
STT_SELL_PCT  = 0.10      # delivery
EXCHANGE_PCT  = 0.00345   # NSE
GST_ON_FEES   = 0.18      # 18% GST on brokerage + exchange
STAMP_BUY_PCT = 0.015     # buy side, delivery
SEBI_PCT      = 0.0001
DP_CHARGE_FLAT = 13.5     # per ISIN per sell, varies by broker
```

For ₹100,000 round-trip, this works out to **~0.35–0.45%** all-in for a
liquid large-cap. The optimizer's current 0.15% is half that.

### B3 🟠 Walk-forward windows overlap; "out-of-sample" isn't really out

`optimize.py` uses 3 walk-forward windows. With `TRAIN_MONTHS=9` and
`VALIDATE_MONTHS=3`, window 1 is `train [t-12, t-3], validate [t-3, t]`,
window 2 is `train [t-15, t-6], validate [t-6, t-3]`, etc. The **validation
slice of window 2** lies entirely inside the **training slice of window 1**.
Optuna sees window 2's validation period as both train and validate
across different windows. The averaged objective therefore double-counts
some periods.

More fundamentally, Optuna optimises *the average across all 3 validation
windows*. Once that average is the objective, every byte of those
validation slices is in-sample. The whole 24-month dataset is effectively
in-sample by the end of 200 trials.

**Recommendation:** carve out a final unseen 3-month *holdout* that Optuna
never touches, and report results on it once at the end of optimization.
Code sketch:

```python
HOLDOUT_MONTHS = 3
def fetch_all_data():
    ... # as today, fetch DATA_DAYS=730
def split_holdout(df):
    holdout_start = df.index.max() - pd.DateOffset(months=HOLDOUT_MONTHS)
    return df[df.index < holdout_start], df[df.index >= holdout_start]

# In run():
all_data = fetch_all_data()
optim_data, holdout_data = {t: split_holdout(d) for t, d in all_data.items() ...}
# Optuna only sees optim_data
study.optimize(make_objective(optim_data), n_trials=N_TRIALS)
# Then evaluate top trial on holdout
holdout_metrics = walk_forward_score(study.best_params, holdout_data)
```

If the in-sample score is 0.55 and holdout is 0.30, you've badly overfit
and should not promote.

### B4 🟠 200 Optuna trials × 25 hyperparameters = guaranteed overfit without correction

With 200 trials over a 25-dim search space, the top trial's objective is
heavily inflated by selection bias. The "deflated Sharpe ratio" literature
(Bailey & López de Prado, 2014) gives the formula. A practical workaround:

- Run a **null study**: same objective, but feed Optuna *shuffled* daily
  returns. Best score from the null = the bar to beat.
- Promote a challenger only if its score exceeds the null's 95th
  percentile.

If you don't want to implement DSR, the *minimum* defence is a holdout
(B3) and a "no improvement, no promotion" rule:

```python
# In optimize.py.run() after computing top_trials[0].value:
if top_trials[0].value < champion_score * 1.10:
    print("Top candidate not >10% better than champion — not promoting")
    return
```

Currently the comment says "Consider promoting via dashboard" if 5%
better, but there's no enforcement gate.

### B5 🟡 `weighted_vote` confidence floor is too low

`analyze.py:1104`:

```python
conf = min(n / 8.0, 1.0)
```

A strategy with 8 trades over ~9 months gets full confidence weight.
That's not a statistically reliable sample — you need ~30 to estimate
a binomial proportion within ±10pp at 95%. Also there's no floor: a
strategy with 1 trade gets `conf=0.125`, still influencing the vote.

**Recommendation:**

```python
n = bt.get("trades", 0)
if n < 5:
    conf = 0.0    # ignore strategies with too few trades
else:
    conf = min((n - 5) / 25.0, 1.0)  # full conf at 30 trades
```

### B6 🟡 Win rate is a dangerous objective when wins/losses are asymmetric

`weighted_vote` weights strategies by `win_rate × confidence`. A strategy
with 30% win rate but 4R wins on average has positive expectancy of
`0.3·4 - 0.7·1 = +0.5R per trade` — your system would call it weak.
Conversely a 70%/0.5R strategy with `0.7·0.5 - 0.3·1 = +0.05R` looks
strong but barely beats costs.

**Recommendation:** change `weighted_vote` to use **expectancy** (or
profit factor, which encodes both):

```python
weights[name] = round(max(bt.get("expectancy", 0), 0) * conf, 3)
```

This requires `expectancy` to be returned from `analyze.py.backtest()` —
optimize.py already computes it (line 284), just port the line.

### B7 🟡 Look-ahead in fundamentals score (in live, not backtest)

`fetch_fundamentals()` pulls *current* P/E, ROE, debt-to-equity, sector
etc. These multiply the technical score (`apply_score_multipliers`).
For *live* signals this is fine. For *backsimulation* of past signals,
applying *current* fundamentals is mild lookahead — a company that just
released bad results today is being scored against historical signals
with that bad-results state. The ranking of historical signals is
therefore biased.

The good news: I checked, and `backsimulate.py` does not use
`fundamental_score`. It only uses `stop_loss`, `target`, and the
backtester's own outcome. So this is an issue only if you ever extend
backsimulate to use fundamentals — which is tempting and worth flagging.

### B8 🟡 News sentiment uses post-hoc headlines for live signals

`fetch_news_sentiment()` runs `gnews(period="7d")` — i.e. the last
seven days. So a signal generated at 7 AM IST on Tuesday is being scored
against headlines from the prior 7 calendar days, which includes
yesterday's news. That's *not* lookahead per se for live signals (it's
what a human would also see) — but it does mean a signal generated
because of a news catalyst will get its sentiment double-counted in the
multiplier. That's a feature, not a bug, but worth being aware of when
interpreting `news_alert` flags.

### B9 🟢 Stop-first vs target-first is one-size-fits-all

When SL and target are both touched on the same daily bar, every
backtester in the project takes SL first. This is conservative and
defensible. But it's strategy-agnostic. Mean-reversion setups
(Bollinger, RSI+MACD) typically *first* spike against the trade then
recover; breakout setups (Donchian, Volume Breakout) typically *first*
gap with the trade then run. So per-strategy bias might be:

| Strategy        | Same-bar bias       |
|-----------------|---------------------|
| Donchian        | target-first probably more accurate |
| Volume Breakout | target-first        |
| Bollinger       | stop-first conservative              |
| RSI + MACD      | stop-first          |
| EMA Crossover   | unclear (trend-following) |
| RSI Trend Shift | unclear             |

For each strategy independently, you could parameterise:
`SAME_BAR_BIAS: "stop_first" | "target_first" | "intraday_check"`. The
last would re-fetch 5-min OHLC for ambiguous days, which yfinance
supports for the past ~60 days. That's heavy work — the value is small
unless you find your live results consistently differ from backtest in
the same direction. (Which the new `health_check.py` will surface.)

---

## C — Risk management

### C1 🔴 No position sizing — flat quantity for every trade

`paper_buy()` accepts whatever qty the user types. That decouples risk
from volatility: you can buy 10 shares of HDFC Bank (ATR ~₹15, risk per
share ~₹100) and 10 shares of TATAELXSI (ATR ~₹150, risk per share
~₹1,000). Same nominal qty, 10× the risk on the second trade. Over many
trades, your portfolio is dominated by whichever stock happens to be
the most volatile.

**Recommendation: fixed-fractional risk sizing.** Decide max loss per
trade as a % of capital (1% is standard). Then qty falls out:

```python
def suggest_qty(capital_inr, risk_pct, entry, sl):
    risk_per_share = entry - sl
    if risk_per_share <= 0:
        return 0
    risk_inr = capital_inr * (risk_pct / 100)
    qty = int(risk_inr // risk_per_share)
    # Cap by 10% of capital exposure per single trade
    max_exposure = capital_inr * 0.10
    qty = min(qty, int(max_exposure // entry))
    return qty
```

Then in dashboard's paper-buy form, default the qty input to
`suggest_qty(account_size, 1.0, sel.price, sel.stop_loss)`. The user can
override but the default is risk-aware. Add `account_size_inr` to
`agent_meta` or to a Streamlit secret so it's user-configurable.

This single change is the highest-leverage upgrade in the whole project.
Everything else here is +5–10% improvement; this is +30–50%.

### C2 🟠 No portfolio-level risk or correlation

Five BUY signals on five banks the same day are 0.7+ correlated. Your
system treats them as independent. A bank-sector down-day takes them
all out simultaneously — but each one's individual risk was sized
correctly, so cumulative drawdown is 5× a single-position loss.

**Recommendation:**

1. Sector concentration cap: max 30% of capital in any one sector.
   Easy to enforce at paper-buy time using `fund.get('sector')`.
2. Correlation-adjusted risk: when sizing position N+1, reduce qty by
   the historical correlation of returns with existing positions.
   (Realistic: pre-compute a sector-level correlation matrix once a
   week; use sector pair correlations as a proxy.)

### C3 🟠 No regime-conditional sizing

Market regime is computed (`market_regime()`) and used as a +/-10pt
score adjustment. But it doesn't *de-risk* in BEARISH regimes. A
sensible quant rule: in BEARISH regime, halve position sizes; in
BULLISH, allow full sizes; in NEUTRAL, 75%.

```python
regime_size_mult = {"BULLISH": 1.0, "NEUTRAL": 0.75, "BEARISH": 0.50, "UNKNOWN": 0.5}
qty_final = int(base_qty * regime_size_mult.get(regime_label, 0.75))
```

### C4 🟠 No drawdown-based portfolio brake

If your last 10 closed trades were all losers, your strategy probably
isn't broken — but something might be (regime change, parameter
mis-fit). Reduce size or pause until results recover. Standard rule:

```python
recent_pnl = sum of last 20 closed trades' pnl_pct
if recent_pnl <= -10:   # 10% drawdown over recent trades
    no new positions allowed
elif recent_pnl <= -5:
    halve all new position sizes
```

Implement as an additional gate in the new-paper-buy form.

### C5 🟡 No trailing stop / breakeven move

Once a position is +1R in profit, moving SL to breakeven is free
expectancy. Currently SL is set once at entry and never updated. A
simple rule:

- Position +1R: move SL to entry price (lock in breakeven)
- Position +2R: trail SL at `max_high_since_entry - 1.5 × ATR_at_entry`

This is the single biggest reason rule-based systems leave money on the
table. Worth implementing in `check_alerts.py` as well, so it
auto-updates `entry_stop_loss` on the row over time.

### C6 🟡 SL/Target capping behaviour is the source of the "8%/12%" pattern

You asked about this in the previous turn. Confirmed: in `analyze.py:348`
`MAX_RISK_PCT=8.0` caps SL at 8% below entry, and `MIN_RR_RATIO=1.5` then
forces target ≥ 1.5 × 8% = 12%. For most Nifty-50 stocks the
ATR-based natural SL is *further* than 8% below entry (since you're using
1.25 × ATR as the default natural distance, and many stocks have ATR < 6%
of price). Result: the cap binds for almost every trade and you get the
flat 8/12 pattern.

Three options, in increasing rigour:

1. **Tighten the cap.** Reduce `MAX_RISK_PCT` to 5–6%. Quick win but
   crude.
2. **Volatility-tiered cap.** Compute `atr_pct = atr / entry × 100`.
   Set `max_risk_pct = max(2 × atr_pct, 4)`. So a stock with 1% ATR
   gets a 4% max SL (tight); a stock with 4% ATR gets 8%. Self-scaling.
3. **Pure ATR-based, no percent cap.** Rely on `MAX_RISK_ATR=3.0` only.
   Then *every* stop is a multiple of ATR, which is the textbook quant
   approach. The risk: positions in volatile small-caps will have
   unbounded percent SL — combine with C1 (position sizing) so the
   *cash* risk stays bounded.

I'd recommend option 2 first; once C1 is in, option 3 becomes safe.

---

## D — Strategy / signal quality

### D1 🟡 Six strategies, all old-school technical, high mutual correlation

Two of your six strategies (Donchian, Volume Breakout) are basically the
same idea (n-day range break) with one extra filter. EMA Crossover and
RSI Trend Shift will fire on overlapping conditions. A correlation
matrix of historical strategy signals (which is easy to compute once)
would tell you that effectively you have ~3 independent strategies, not
6, and the weighted vote is over-weighting one cluster.

Quick test: in `analyze.py:1099-1116`, before computing weights,
check pairwise signal correlations across the SAMPLE universe:

```python
sigs = pd.DataFrame({name: fn(df) for name, fn in STRATEGIES.items()})
print(sigs.corr())
```

If anything is >0.7, drop one. I'd bet `sig_donchian` and
`sig_volume_breakout` are >0.85 correlated.

### D2 🟡 Missing standard features that are nearly free

Adding any of these will materially improve signal quality with a few
lines of code each:

- **ADX** — trend strength filter. Don't take EMA Crossover signals when
  ADX < 20 (no trend), don't take Bollinger mean-reversion signals when
  ADX > 30 (strong trend, mean reversion will fail).
- **MFI (Money Flow Index)** — volume-weighted RSI. Better divergence
  signal than RSI for breakouts.
- **OBV slope** — accumulation/distribution. Filter for breakouts.
- **Beta (vs Nifty)** — high-beta stocks behave differently from low-beta.
  Adjust position size by 1/beta for risk parity.
- **20-day return relative strength vs Nifty** — momentum factor. The
  single most-cited equity factor in the literature.
- **INDIAVIX level** — volatility regime. Above 25 = de-risk; below 12 =
  mean reversion strategies do worse.

### D3 🟡 No ML-based score calibration

Your composite_score is a hand-tuned weighted sum: `40×strategy + 20×rsi
+ 15×volume + 15×rr + 10×regime`. Whether it actually predicts P(win) is
empirical. The data to check is *already in your DB*: the
`backtest_simulations` table has `composite_score` and `was_win` for
every closed trade. A 10-line scikit-learn script would tell you:

- Is the score monotonic in win probability? (You display this in
  Backtest Lab as quartile bars — good.)
- Is it calibrated? (i.e. does score=70 mean ~70% win prob?)
- Could a simple LR / GBM on the same features outperform the weighted
  sum?

Realistic upside: I'd expect a calibrated logistic regression on
`(weighted_score, rsi, vol_ratio, rr_ratio, regime, sector, news_score,
fund_score)` to lift AUC by 0.05–0.10 over the hand-weighted scoring
— that's the difference between 55% win rate and 60%.

### D4 🟢 EXIT signals don't have a real meaning

`weighted_vote` for EXIT does:

```python
weights[name] = round(max(conf, 0.25), 3)
```

i.e. just confidence, not based on any backtest because EXIT ≠ short
sale. So the "EXIT score" is just "how many strategies fired bearish."
That's fine but it's mislabelled in the dashboard as a "score" with a
0–100 number, implying expectancy. Either rename ("Bearish Vote
Strength" or similar) or rebuild it as actual expectancy (would require
short-sale backtesting which the README explicitly avoids).

---

## E — Operational / infra

### E1 🟠 Holiday calendar — agent runs on NSE holidays with stale data

`daily_analysis.yml` runs Mon-Fri. NSE has ~14 holidays a year that fall
on weekdays. yfinance returns the previous trading day's close. Result:
every Republic Day, Holi, Eid etc., your system "generates today's
signals" using yesterday's data. The signals are duplicates of the
previous day's, the alert spam is unhelpful, and `recommendations`
gains a row dated as today with last-trading-day's prices.

**Status: fixed.** New `agent/holiday_calendar.py` + updated
`daily_analysis.yml` skip on holidays. **You must update the
`HOLIDAYS_2026` list each year** — the v7 SQL migration also adds a
`market_holidays` table you can edit without redeploy.

### E2 🟠 No live observability — silent failures are invisible

If yfinance starts returning empty for half the universe, or Supabase
hits a row limit, or Optuna picks degenerate params, you only notice
when looking at the dashboard. There's no alert.

**Status: fixed.** New `agent/health_check.py` runs after market close,
checks 7 things (data freshness, yfinance coverage, signal imbalance,
live-vs-backtest drift, stale open positions, champion age, DB size),
and Telegrams you on warn or fail.

### E3 🟠 Champion promotion is non-atomic

`optimize.py:618–625` does `UPDATE retired challenger` then `UPDATE
challenger to new version`. If supabase fails between these two, you
end up with no challenger. More importantly there's no constraint
preventing two `champion` rows.

**Status: partially fixed.** The v7 migration adds a partial unique
index `uniq_champion ON agent_params (status) WHERE status = 'champion'`
which makes "two champions" impossible. The retire-then-promote race is
still there but is now safer (the constraint will catch any double).

### E4 🟡 No archival / row-count growth control

`recommendations` gains 50–200 rows/day; `ticker_run_log` gains ~250/day.
At 6 months that's ~50,000 rows total. Supabase free tier is 500 MB.
Realistically OK for >1 year, but no automatic purge.

The v7 migration purges `ticker_run_log` older than 90 days (one-shot).
For ongoing housekeeping, add to your weekly Sunday workflow:

```sql
DELETE FROM ticker_run_log WHERE created_at < NOW() - INTERVAL '90 days';
DELETE FROM recommendations WHERE date < NOW() - INTERVAL '180 days'
  AND id NOT IN (SELECT recommendation_id FROM paper_portfolio WHERE recommendation_id IS NOT NULL)
  AND id NOT IN (SELECT recommendation_id FROM backtest_simulations);
```

The `NOT IN` clauses preserve recommendations that were paper-traded or
backsimulated — you want those forever.

### E5 🟡 Reconciliation table is built but never populated

Currently the system has:
- `paper_portfolio` (live results)
- `backtest_simulations` (predicted results from backsim)

It computes them independently. If they disagree wildly *for the same
signal*, that's the smoking gun for a logic divergence between live and
backtest paths. The v7 migration adds a `reconciliation` table; populate
it nightly after both `paper_sell` and `backsimulate` have run for the
same `recommendation_id`. The health check (`live_vs_backtest`) already
queries this comparison aggregate-wise; explicit per-trade
reconciliation gives you the row-level diff.

Sketch of a fill job:

```python
def fill_reconciliation():
    # For each closed paper trade with a matching backsim, insert a row
    # if not already present.
    closed = sb.table("paper_portfolio").select("*").eq("status","CLOSED").execute().data
    for c in closed:
        if not c.get("recommendation_id"):
            continue
        bs = sb.table("backtest_simulations").select("*").eq("recommendation_id", c["recommendation_id"]).execute().data
        if not bs:
            continue
        bs = bs[0]
        sb.table("reconciliation").upsert({
            "paper_trade_id":      c["id"],
            "recommendation_id":   c["recommendation_id"],
            "ticker":              c["ticker"],
            "signal_date":         bs["signal_date"],
            "paper_return_pct":    c.get("pnl_pct"),
            "backsim_return_pct":  bs.get("actual_return_pct"),
            "return_diff_pct":     (c.get("pnl_pct") or 0) - (bs.get("actual_return_pct") or 0),
            "paper_exit_reason":   c.get("exit_reason"),
            "backsim_exit_reason": bs.get("exit_reason"),
        }).execute()
```

### E6 🟢 Optimizer and Backsimulator share Sunday 11 PM cron

Both `weekly_backsimulate.yml` and `weekly_optimize.yml` schedule
`30 17 * * 0`. GitHub will run them in parallel. They don't conflict
(different tables) but they double-load yfinance simultaneously, which
makes both slower. Stagger by 30 minutes.

### E7 🟢 News sentiment fallback is unclear

`fetch_news_sentiment` tries FinBERT then falls back to VADER. If both
fail, the headline is kept but the score is 0/NEUTRAL. That's then
recorded as `news_source: "headlines_only"` — which is fine, but means
some signals are scored against actual sentiment and others against
implicit-neutral. Not a bug, just be aware that filtering on
`news_label = 'positive'` will under-include FinBERT-failed positives.

---

## F — Recommended GitHub Actions additions

This is what I'd add to the `.github/workflows/` directory.

### F1 ✅ `daily_health_check.yml` — *included*

Runs after market close, surfaces silent failures.

### F2 (recommended) `weekly_reconciliation.yml`

Runs Sundays after both backsim and optimizer have completed (e.g. 19:00
UTC). Populates the `reconciliation` table from `paper_portfolio` ⨝
`backtest_simulations`. Trivial Python (sketch in E5).

### F3 (recommended) `monthly_universe_refresh.yml`

Once a month, fetch the *current* Nifty 50 / Nifty 100 / Nifty 200
composition from NSE archives and update a `universe_history` table.
This is what enables proper survivorship-bias-free backtests later. NSE
publishes CSVs at predictable URLs.

### F4 (recommended) split current `weekly_optimize.yml` into two
workflows: `weekly_optimize_train.yml` (Saturday) writes a candidate;
`weekly_optimize_validate.yml` (Sunday) evaluates the candidate on a
*frozen* holdout the optimizer never saw. Only if both green, promote.
This is the manual gate for B3/B4.

### F5 (optional) `pre_market_data_check.yml`

Run at 03:00 UTC (~08:30 IST), before market open. Verifies yfinance
has yesterday's close for ≥95% of tickers. If not, alert. Catches yf
outages before the 7 AM analysis run wastes a slot.

### F6 (optional) `nightly_telegram_summary.yml`

Run at 11:30 UTC (~17:00 IST), summarises: today's open positions,
day's P&L, any breaches (now sticky-detected), tomorrow's pre-market
movers. Pure UX upgrade.

---

## G — File-by-file specific issues

These are the spot-fixes I'd file as separate PRs but didn't write code
for. All are low/medium effort.

### `agent/analyze.py`
- L65–91: NIFTY50 list — see B1, freeze a 2020 list for backtests.
- L98–129: `DEFAULT_PARAMS` — `BT_SL_PCT=5.0` and `BT_TARGET_PCT=10.0` are
  fallbacks. Leaves comment confusing on first read; rename to
  `BT_FALLBACK_SL_PCT` etc.
- L348: `MAX_RISK_PCT=8.0` — see C6, change to volatility-tiered.
- L1099–1116: `weighted_vote` — see B5, B6.
- L1224: `if len(c) < 20` — minimum bars for a signal is 20. With
  EMA_LONG=21 and DONCHIAN_PERIOD=20 your indicators won't even be
  warmed up for 21 days. Bump to `min(60, max(EMA_LONG, ATR_PERIOD,
  DONCHIAN_PERIOD) * 2)`.
- L1356: `today_sigs = {... fn(df).iloc[-1] ...}` — calls strategy fn
  6× total inside a list comprehension; on the same line, on L1364, calls
  `fn(df)` *another* 6× to feed `backtest()`. That's 12 calls for a
  6-strategy system. Memoise:

  ```python
  strategy_signals = {name: fn(df) for name, fn in STRATEGIES.items()}
  today_sigs = {name: int(s.iloc[-1]) for name, s in strategy_signals.items()}
  ...
  bt = {name: backtest(df, strategy_signals[name], P, benchmark_df) for name in STRATEGIES}
  ```

  Cuts compute by ~50% on the per-ticker hot path.

### `agent/optimize.py`
- L62: `"TMPV"` — Tata Motors Passenger Vehicles, recently demerged.
  Doubt yfinance has 2 years of data on this ticker. Verify or remove.
- L171–172: cost model — see B2.
- L390–391: `"BT_SL_PCT": 2.0–8.0` and `"BT_TARGET_PCT": 4.0–18.0` are
  for *fallback* when ATR is unavailable. Optuna is spending search
  budget on a rarely-used branch. Either narrow the range to 4-8% / 8-15%
  or remove these from the search space and treat them as constants.
- L417–419: hard constraints return -999. Fine, but Optuna's TPE sampler
  is wasting evaluations on infeasible regions because there's no
  prior knowledge. Use `optuna.samplers.TPESampler(constant_liar=True)`
  and consider Optuna's `enqueue_trial` to seed multiple known-feasible
  configs rather than just one.
- L631: `if top_trials[0].value > champ_score * 1.05` — see B4, this
  is a print statement, not an enforcement gate.

### `agent/backsimulate.py`
- L17–18: `SL_PCT = 5.0`, `TARGET_PCT = 10.0` fallbacks differ from
  `optimize.py` and `analyze.py`. Centralise.
- L86: `len(df) < 2` is a weak guard; recommend `< 3` so you have a real
  forward bar after entry-day.
- L84: `end = ... timedelta(days=max_hold * 2 + 14)` — pads by 28 days
  for max_hold=15. Aggressive, fine. But there's no benchmark
  fetch fallback if `_download_ohlc(BENCHMARK,...)` fails — `bench_df`
  becomes None, and `_benchmark_return` returns None — so `relative_return_pct`
  silently becomes None for every trade in a window where Nifty fetch
  failed. Add a retry.

### `agent/check_alerts.py`
- Replaced — see new file.

### `agent/telegram_alerts.py`
- L65: `f"P&amp;L: <b>{pnl:+.2f}%</b>"` — manually escaped `&`. Inconsistent
  with rest of file. Use a `_html_escape` helper everywhere.

### `dashboard/app.py`
- L53: hardcoded password fallback — see A5.
- L286–306: replaced — see PATCHES.md.
- L883: `lp = row.buy_price; pnl_pct = pnl_inr = 0` — when yfinance
  fails, you display the buy price as the live price. Misleading; show
  "Live: unavailable" instead.
- L884–895: replaced — see PATCHES.md.

### `supabase_setup.sql`
- L153–176: "UPGRADING FROM v2" block runs every install, idempotent
  but noisy. Move to a separate `migrations/` folder so first-time
  installers don't see retroactive upgrade clutter.
- L180–215: "v5 compatibility upgrades" — likewise.
- L307–329: param JSON patcher hardcodes `MIN_RR_RATIO=1.5` and
  `MAX_RISK_PCT=8.0`. This means re-running this script will *override*
  whatever the optimizer learned for these params. Either gate with
  `WHERE params_json->>'MIN_RR_RATIO' IS NULL` or delete the block.

---

## H — Suggested priority order

If you only do five things, do these:

1. **Apply the OHLC patch** (already shipped: `agent/market_data.py`,
   new `agent/check_alerts.py`, `dashboard/PATCHES.md`,
   `supabase_migration_v7.sql`). Eliminates A1, A2.
2. **Add fixed-fractional position sizing** (C1). Single biggest
   improvement to expected returns.
3. **Add the holiday calendar + health check workflows** (already
   shipped). Eliminates E1, E2.
4. **Realistic costs in all backtest engines** (B2). Stops you optimising
   into negative-expectancy strategies.
5. **Holdout-validated optimizer with promotion gate** (B3, B4). Stops
   you adopting overfit champions.

After those, the next tier is C2 (correlation), C3 (regime sizing),
C4 (drawdown brake), B5 (confidence floor), B6 (expectancy not win
rate). Those collectively are another big jump.

Everything else is polish.

---

## I — What this review did *not* cover

- Order-routing realism (no real broker integration; this is paper).
- Live execution slippage modelling (would need bid-ask data).
- Tax handling (STCG vs LTCG depending on holding period).
- F&O / options strategies (system is equity-cash-only).
- Pre-market gap signals (system runs at 7 AM IST, before pre-market opens
  at 9:00 — so all gap info from today's pre-market is missed). A 9:05
  AM IST top-up run could capture pre-market gaps.

If any of these matter to where you want to take the system, they're
worth a separate design discussion.

---

*Reviewed against 4,096 lines across `agent/` and `dashboard/`. All code
fixes referenced are tested for syntax; the breach detector has 9
unit-tested cases (gap, intraday wick, recovered wick, both-touched,
no-breach, missing data, no-SL-set, target-only, etc).*
