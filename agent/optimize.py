#!/usr/bin/env python3
"""
============================================================
  Parameter Optimizer  —  v2  (clean, aligned with live agent)

  v2 changes from v1:
   ─ Strategy roster aligned with live agent (analyze.py):
       Donchian, EMA Crossover, RSI Trend Shift, Bollinger.
     Dropped:
       * RSI + MACD       — fired 0 times in 932 prod recs
       * Volume Breakout  — anti-predictive paired with Donchian
       * Supertrend       — never a live strategy, only a display
   ─ Removed SUPERTREND_MULT, BT_SL_PCT, BT_TARGET_PCT from
     the search space:
       * SUPERTREND_MULT was unused.
       * BT_SL_PCT / BT_TARGET_PCT are LAST-RESORT fallbacks
         (when ATR is missing). The live agent always uses the
         ATR path. Searching them is searching dead code.
       * Optimizer now focuses on the ATR_STOP_BUFFER /
         ATR_TARGET_BUFFER / MAX_RISK_* / MIN_RR_RATIO knobs
         that actually drive levels in production.
   ─ Pinned VOLUME_MULT (unused in v7+ live).
   ─ Promotion gate tightened: a challenger must beat the
     champion's objective_score by ≥10% (was 5%) before this
     module recommends promotion. The promotion itself is
     still gated by human review in the dashboard.

  Stack: Optuna + Supabase
  Method: Walk-forward CV to prevent overfitting
  Output: Top candidates → agent_params; best → 'challenger'.
          Only auto-promotes to 'champion' if NO champion exists yet.
============================================================
"""

import os, sys, json, warnings, time
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
import optuna
from supabase import create_client, Client

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────────
#  OPTIMIZATION SAMPLE (representative subset)
# ─────────────────────────────────────────────
SAMPLE_STOCKS = [
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
    "BAJFINANCE", "BAJAJFINSV", "JIOFIN", "PFC", "RECLTD",
    "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
    "LTIM", "PERSISTENT", "COFORGE",
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
    "MARUTI", "M&M", "TMPV", "EICHERMOT", "HEROMOTOCO", "TVSMOTOR",
    "RELIANCE", "ONGC", "NTPC", "POWERGRID", "BPCL",
    "LT", "ULTRACEMCO", "TATASTEEL", "JSWSTEEL", "HINDALCO",
    "ADANIPORTS", "SIEMENS", "BEL", "HAL",
    "TRENT", "INDIGO", "ZOMATO",
]

# Walk-forward config
TRAIN_MONTHS    = 9
VALIDATE_MONTHS = 3
N_WINDOWS       = 3
N_TRIALS        = 200
TOP_K           = 5
DATA_DAYS       = 730

# Promotion gate: challenger must beat champion by this margin to
# qualify for human-review promotion (was 1.05 in v1).
CHAMPION_BEAT_MARGIN = 1.10

SLIPPAGE_PCT  = 0.10
BROKERAGE_PCT = 0.05

# ─────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────
def ema(s, p):       return s.ewm(span=p, adjust=False).mean()

def rsi(s, p=14):
    d  = s.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / p, min_periods=p, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1 / p, min_periods=p, adjust=False).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))

def bollinger(s, p, std):
    m  = s.rolling(p).mean()
    sg = s.rolling(p).std()
    return m + std * sg, m, m - std * sg

def atr(h, l, c, p):
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / p, min_periods=p, adjust=False).mean()

# ─────────────────────────────────────────────
#  SIGNAL GENERATORS — must match analyze.py exactly
# ─────────────────────────────────────────────
def sig_ema(df, p):
    e_s = ema(df.Close, p["EMA_SHORT"])
    e_l = ema(df.Close, p["EMA_LONG"])
    s   = pd.Series(0, index=df.index)
    s[(e_s > e_l) & (e_s.shift() <= e_l.shift())] =  1
    s[(e_s < e_l) & (e_s.shift() >= e_l.shift())] = -1
    return s

def sig_bb(df, p):
    up, _, lo = bollinger(df.Close, p["BB_PERIOD"], p["BB_STD"])
    r = rsi(df.Close, p["RSI_PERIOD"])
    s = pd.Series(0, index=df.index)
    s[(df.Low  <= lo) & (df.Close > lo) & (r < 50)] =  1
    s[(df.High >= up) & (df.Close < up) & (r > 50)] = -1
    return s

def sig_donchian(df, p):
    period = int(p.get("DONCHIAN_PERIOD", 20))
    hh     = df.High.rolling(period).max()
    ll     = df.Low.rolling(period).min()
    s      = pd.Series(0, index=df.index)
    s[(df.Close > hh.shift(1)) & (df.Close.shift(1) <= hh.shift(1))] =  1
    s[(df.Close < ll.shift(1)) & (df.Close.shift(1) >= ll.shift(1))] = -1
    return s

def sig_rsi_trend_shift(df, p):
    r   = rsi(df.Close, p["RSI_PERIOD"])
    e_l = ema(df.Close, p["EMA_LONG"])
    mid = float(p.get("RSI_MIDLINE", 50))
    s   = pd.Series(0, index=df.index)
    s[(df.Close > e_l) & (r > mid) & (r.shift(1) <= mid)] =  1
    s[(df.Close < e_l) & (r < mid) & (r.shift(1) >= mid)] = -1
    return s

# ─────────────────────────────────────────────
#  BACKTEST (with realistic costs)
# ─────────────────────────────────────────────
def _finite_float(v, default=None):
    try:
        fv = float(v)
        return fv if np.isfinite(fv) else default
    except Exception:
        return default


def _dynamic_levels(df, signal_idx, entry_price, p):
    lookback     = int(p.get("RR_LOOKBACK", 20))
    atr_period   = int(p.get("ATR_PERIOD", 14))
    stop_buf     = float(p.get("ATR_STOP_BUFFER", 0.50))
    target_buf   = float(p.get("ATR_TARGET_BUFFER", 0.50))
    max_risk_atr = float(p.get("MAX_RISK_ATR", 3.00))
    max_risk_pct = float(p.get("MAX_RISK_PCT", 8.00))
    min_rr_ratio = float(p.get("MIN_RR_RATIO", 1.50))

    support_s  = df.Low.rolling(lookback).min().shift(1)
    resist_s   = df.High.rolling(lookback).max().shift(1)
    atr_s      = atr(df.High, df.Low, df.Close, atr_period)

    support    = _finite_float(support_s.iloc[signal_idx]) if signal_idx < len(support_s) else None
    resistance = _finite_float(resist_s.iloc[signal_idx])  if signal_idx < len(resist_s)  else None
    atr_now    = _finite_float(atr_s.iloc[signal_idx])     if signal_idx < len(atr_s)     else None

    if atr_now is None or atr_now <= 0:
        # Pure ATR-fallback path — use fixed defaults from DEFAULT_PARAMS in
        # analyze.py rather than searched values, since this isn't
        # the live primary path.
        sl  = entry_price * (1 - 5.0 / 100.0)
        tgt = entry_price * (1 + 10.0 / 100.0)
    else:
        sl = (support - stop_buf * atr_now
              if support is not None and support < entry_price
              else entry_price - 1.25 * atr_now)
        sl = max(sl,
                 entry_price - max_risk_atr * atr_now,
                 entry_price * (1 - max_risk_pct / 100.0))
        if sl >= entry_price:
            sl = max(entry_price - 1.25 * atr_now,
                     entry_price * (1 - max_risk_pct / 100.0))
        tgt = (resistance + target_buf * atr_now
               if resistance is not None and resistance > entry_price
               else entry_price + 1.75 * atr_now)
        if tgt <= entry_price:
            tgt = entry_price + 1.75 * atr_now

    actual_risk = entry_price - sl
    if actual_risk > 0 and min_rr_ratio > 0:
        tgt = max(tgt, entry_price + actual_risk * min_rr_ratio)
    return sl, tgt


def backtest_with_costs(df, signals, p):
    """Long-only next-bar backtest, slippage+brokerage applied."""
    trades, reasons = [], []
    in_t, ep, entry_idx = False, 0.0, -1
    sl_px = tgt_px = None
    closes = df.Close.values
    highs  = df.High.values
    lows   = df.Low.values
    opens  = df.Open.values
    max_hold = p["BT_MAX_HOLD"]
    cost_pct = SLIPPAGE_PCT + BROKERAGE_PCT

    def close_trade(exit_px, reason):
        nonlocal in_t, ep, entry_idx, sl_px, tgt_px
        gross = (exit_px - ep) / ep * 100 if ep else 0.0
        trades.append(gross - cost_pct)
        reasons.append(reason)
        in_t = False
        ep, entry_idx = 0.0, -1
        sl_px = tgt_px = None

    for i in range(1, len(df)):
        prev_sig = int(signals.iloc[i - 1]) if pd.notna(signals.iloc[i - 1]) else 0
        if in_t:
            if prev_sig == -1:
                close_trade(float(opens[i]), "exit_signal");                 continue
            if sl_px is not None and lows[i] <= sl_px:
                close_trade(float(sl_px), "sl");                              continue
            if tgt_px is not None and highs[i] >= tgt_px:
                close_trade(float(tgt_px), "target");                         continue
            if i >= entry_idx + max_hold:
                close_trade(float(closes[i]), "timeout");                     continue
        if not in_t and prev_sig == 1:
            ep = float(opens[i])
            if ep <= 0:
                continue
            sl_px, tgt_px = _dynamic_levels(df, i - 1, ep, p)
            entry_idx, in_t = i, True
            if sl_px is not None and lows[i] <= sl_px:
                close_trade(float(sl_px), "sl");                              continue
            if tgt_px is not None and highs[i] >= tgt_px:
                close_trade(float(tgt_px), "target");                         continue

    if not trades:
        return dict(win_rate=0, avg_return=0, median_return=0, trades=0,
                    profit_factor=0, max_drawdown=0, expectancy=0)

    wins   = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    gp, gl = sum(wins), abs(sum(losses))
    pf     = round(gp / gl, 2) if gl > 0 else 99.0
    eq     = np.cumsum(trades)
    peak   = np.maximum.accumulate(eq)
    max_dd = float(abs((eq - peak).min())) if len(eq) > 0 else 0
    wr     = len(wins) / len(trades)
    avg_w  = float(np.mean(wins)) if wins else 0
    avg_l  = float(np.mean([abs(l) for l in losses])) if losses else 0
    return dict(
        win_rate      = round(wr * 100, 1),
        avg_return    = round(float(np.mean(trades)), 2),
        median_return = round(float(np.median(trades)), 2),
        trades        = len(trades),
        profit_factor = min(float(pf), 99.0),
        max_drawdown  = round(max_dd, 2),
        expectancy    = round(wr * avg_w - (1 - wr) * avg_l, 3),
    )

# ─────────────────────────────────────────────
#  WALK-FORWARD EVALUATION
# ─────────────────────────────────────────────
def walk_forward_score(params: dict, all_data: dict) -> dict:
    latest = max(df.index.max() for df in all_data.values() if not df.empty)
    windows = []
    for w in range(N_WINDOWS):
        val_end   = latest - pd.DateOffset(months=w * VALIDATE_MONTHS)
        val_start = val_end  - pd.DateOffset(months=VALIDATE_MONTHS)
        trn_start = val_start - pd.DateOffset(months=TRAIN_MONTHS)
        windows.append((trn_start, val_start, val_end))

    all_metrics = []
    # MUST match analyze.py:get_strategies() order and contents.
    strategies = [
        ("Donchian",        lambda df: sig_donchian(df, params)),
        ("EMA Crossover",   lambda df: sig_ema(df, params)),
        ("RSI Trend Shift", lambda df: sig_rsi_trend_shift(df, params)),
        ("Bollinger",       lambda df: sig_bb(df, params)),
    ]

    for trn_start, _val_start, val_end in windows:
        for ticker, full_df in all_data.items():
            if full_df.empty or len(full_df) < 60:
                continue
            warmup_start = trn_start
            slice_df = full_df[(full_df.index >= warmup_start) &
                               (full_df.index <= val_end)].copy()
            if len(slice_df) < 50:
                continue
            for _name, fn in strategies:
                try:
                    all_sigs = fn(slice_df)
                    val_mask = (slice_df.index >= _val_start) & (slice_df.index <= val_end)
                    val_sigs = all_sigs.copy()
                    val_sigs[~val_mask] = 0
                    bt = backtest_with_costs(slice_df, val_sigs, params)
                    if bt["trades"] > 0:
                        all_metrics.append(bt)
                except Exception:
                    continue

    if not all_metrics:
        return dict(profit_factor=0, avg_return=-10, win_rate=0,
                    max_drawdown=100, expectancy=-1, total_trades=0)

    return dict(
        profit_factor = float(np.mean([m["profit_factor"] for m in all_metrics])),
        avg_return    = float(np.mean([m["avg_return"]    for m in all_metrics])),
        win_rate      = float(np.mean([m["win_rate"]      for m in all_metrics])),
        max_drawdown  = float(np.mean([m["max_drawdown"]  for m in all_metrics])),
        expectancy    = float(np.mean([m["expectancy"]    for m in all_metrics])),
        total_trades  = int(sum(m["trades"] for m in all_metrics)),
    )

# ─────────────────────────────────────────────
#  OBJECTIVE
# ─────────────────────────────────────────────
def make_objective(all_data):
    def objective(trial):
        ema_short = trial.suggest_int("EMA_SHORT", 5, 20)
        ema_long  = trial.suggest_int("EMA_LONG", 22, 60)
        params = {
            "EMA_SHORT":          ema_short,
            "EMA_LONG":           ema_long,
            "RSI_PERIOD":         trial.suggest_int("RSI_PERIOD", 7, 21),
            "BB_PERIOD":          trial.suggest_int("BB_PERIOD", 10, 30),
            "BB_STD":             trial.suggest_float("BB_STD", 1.5, 3.0),
            "DONCHIAN_PERIOD":    trial.suggest_int("DONCHIAN_PERIOD", 10, 40),
            "RSI_MIDLINE":        trial.suggest_int("RSI_MIDLINE", 45, 55),
            "ATR_PERIOD":         trial.suggest_int("ATR_PERIOD", 7, 21),
            "BT_MAX_HOLD":        trial.suggest_int("BT_MAX_HOLD", 3, 25),
            "MIN_WEIGHTED_SCORE": trial.suggest_float("MIN_WEIGHTED_SCORE", 0.05, 0.55),
            "RR_LOOKBACK":        trial.suggest_int("RR_LOOKBACK", 14, 35),
            "ATR_STOP_BUFFER":    trial.suggest_float("ATR_STOP_BUFFER", 0.25, 1.00),
            "ATR_TARGET_BUFFER":  trial.suggest_float("ATR_TARGET_BUFFER", 0.25, 1.25),
            "MAX_RISK_ATR":       trial.suggest_float("MAX_RISK_ATR", 1.50, 3.50),
            "MAX_RISK_PCT":       trial.suggest_float("MAX_RISK_PCT", 4.00, 10.00),
            "MIN_RR_RATIO":       trial.suggest_float("MIN_RR_RATIO", 1.20, 2.20),
            # Composite weights (W_STRATEGY pinned at 40 for stability)
            "W_STRATEGY":         40,
            "W_RSI":               trial.suggest_int("W_RSI",     10, 30),
            "W_VOLUME":            trial.suggest_int("W_VOLUME",   5, 20),
            "W_RR":                trial.suggest_int("W_RR",       5, 20),
            "W_REGIME":            trial.suggest_int("W_REGIME",   5, 15),
            # Pinned (unused by live strategy roster but read by other code paths)
            "VOLUME_MULT":         1.2,
            "MACD_FAST":           12,
            "MACD_SLOW":           26,
            "MACD_SIGNAL":         9,
            "RSI_OVERSOLD":        48,
            "RSI_OVERBOUGHT":      58,
            # Fallbacks for ATR-missing path; not searched (see module docstring)
            "BT_SL_PCT":           5.0,
            "BT_TARGET_PCT":      10.0,
        }
        # Hard constraints
        if params["EMA_LONG"] <= params["EMA_SHORT"] + 5:
            return -999.0

        metrics = walk_forward_score(params, all_data)
        trial.set_user_attr("metrics", metrics)

        if metrics["total_trades"] < 10:
            return -998.0

        pf_norm  = min(metrics["profit_factor"] / 3.0, 1.0)
        ret_norm = max(min(metrics["avg_return"] / 5.0, 1.0), 0)
        wr_norm  = metrics["win_rate"] / 100.0
        dd_norm  = min(metrics["max_drawdown"] / 20.0, 1.0)

        score = (
            0.35 * pf_norm  +
            0.25 * ret_norm +
            0.20 * wr_norm  -
            0.20 * dd_norm
        )
        return round(float(score), 5)

    return objective

# ─────────────────────────────────────────────
#  DATA FETCH
# ─────────────────────────────────────────────
def fetch_all_data() -> dict:
    print(f"  Fetching {len(SAMPLE_STOCKS)} stocks ({DATA_DAYS} days history)...")
    data = {}
    for i, ticker in enumerate(SAMPLE_STOCKS, 1):
        sys.stdout.write(f"\r    {i}/{len(SAMPLE_STOCKS)}  {ticker:<12}")
        sys.stdout.flush()
        try:
            df = yf.download(
                ticker + ".NS",
                start=datetime.today() - timedelta(days=DATA_DAYS),
                end=datetime.today(),
                progress=False, auto_adjust=True,
            )
            if df.empty or len(df) < 100:
                data[ticker] = pd.DataFrame()
                continue
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            data[ticker] = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except Exception:
            data[ticker] = pd.DataFrame()
        time.sleep(0.15)
    sys.stdout.write("\r" + " " * 50 + "\r")
    valid = sum(1 for d in data.values() if not d.empty)
    print(f"  {valid}/{len(SAMPLE_STOCKS)} stocks loaded successfully\n")
    return data

# ─────────────────────────────────────────────
#  SUPABASE HELPERS
# ─────────────────────────────────────────────
def next_version() -> int:
    try:
        res = (supabase.table("agent_params").select("version")
               .order("version", desc=True).limit(1).execute())
        if res.data:
            return int(res.data[0]["version"]) + 1
    except Exception:
        pass
    return 1


def get_champion() -> dict | None:
    try:
        res = (supabase.table("agent_params").select("*")
               .eq("status", "champion").order("promoted_at", desc=True)
               .limit(1).execute())
        return res.data[0] if res.data else None
    except Exception:
        return None


def save_candidate(params, metrics, version, run_date,
                   train_start, train_end, val_start, val_end, rank):
    supabase.table("agent_params").insert({
        "version":        version,
        "status":         "candidate",
        "params_json":    json.dumps(params),
        "objective_score": round(metrics.get("_objective", 0), 5),
        "profit_factor":  round(metrics.get("profit_factor", 0), 3),
        "win_rate":       round(metrics.get("win_rate", 0), 1),
        "avg_return":     round(metrics.get("avg_return", 0), 2),
        "max_drawdown":   round(metrics.get("max_drawdown", 0), 2),
        "total_trades":   int(metrics.get("total_trades", 0)),
        "train_start":    str(train_start)[:10],
        "train_end":      str(train_end)[:10],
        "valid_start":    str(val_start)[:10],
        "valid_end":      str(val_end)[:10],
        "run_date":       run_date,
        "rank":           rank,
        "notes":          f"Walk-forward rank {rank}",
    }).execute()

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run():
    run_date = datetime.today().strftime("%Y-%m-%d")
    print(f"\n🔬 Stock Agent Optimizer v2 — {run_date}")
    print(f"   Trials: {N_TRIALS}  |  Walk-forward windows: {N_WINDOWS}")
    print(f"   Train: {TRAIN_MONTHS}m   Validate: {VALIDATE_MONTHS}m")
    print(f"   Promotion gate: challenger must beat champion by ×{CHAMPION_BEAT_MARGIN}\n")

    all_data = fetch_all_data()
    if sum(1 for d in all_data.values() if not d.empty) < 5:
        print("  ❌ Not enough data to optimize. Aborting.")
        return

    all_dates = [df.index for df in all_data.values() if not df.empty]
    train_start = min(d.min() for d in all_dates)
    train_end   = max(d.max() for d in all_dates)
    val_start   = train_end - pd.DateOffset(months=VALIDATE_MONTHS * N_WINDOWS)
    val_end     = train_end

    print(f"  🔍 Running {N_TRIALS} Optuna trials...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
    )
    objective_fn = make_objective(all_data)

    # Warm-start with the live agent's defaults (must stay in suggest_* ranges).
    study.enqueue_trial({
        "EMA_SHORT": 9, "EMA_LONG": 21, "RSI_PERIOD": 14,
        "BB_PERIOD": 20, "BB_STD": 2.0, "DONCHIAN_PERIOD": 20,
        "RSI_MIDLINE": 50, "ATR_PERIOD": 14, "BT_MAX_HOLD": 15,
        "MIN_WEIGHTED_SCORE": 0.08, "RR_LOOKBACK": 20,
        "ATR_STOP_BUFFER": 0.50, "ATR_TARGET_BUFFER": 0.50,
        "MAX_RISK_ATR": 3.00, "MAX_RISK_PCT": 8.00, "MIN_RR_RATIO": 1.50,
        "W_RSI": 20, "W_VOLUME": 15, "W_RR": 15, "W_REGIME": 10,
    })
    study.optimize(objective_fn, n_trials=N_TRIALS, show_progress_bar=False)

    valid_trials = [t for t in study.trials if t.value is not None and t.value > -900]
    valid_trials.sort(key=lambda t: t.value, reverse=True)
    top_trials = valid_trials[:TOP_K]

    if not top_trials:
        print("  ❌ No valid trials found. Check data quality.")
        return

    print("\n  ✅ Optimization complete")
    print(f"     Best score : {top_trials[0].value:.4f}")
    print(f"     Worst (top-{TOP_K}): {top_trials[-1].value:.4f}\n")
    print(f"  {'Rank':<6} {'Score':>8}  {'PF':>6}  {'WR':>6}  {'AvgRet':>8}  {'MaxDD':>8}  {'Trades':>7}")
    print(f"  {'─'*6} {'─'*8}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*7}")
    for i, t in enumerate(top_trials, 1):
        m = t.user_attrs.get("metrics", {})
        print(f"  {i:<6} {t.value:>8.4f}  "
              f"{m.get('profit_factor',0):>6.2f}  "
              f"{m.get('win_rate',0):>5.1f}%  "
              f"{m.get('avg_return',0):>+7.2f}%  "
              f"{m.get('max_drawdown',0):>7.2f}%  "
              f"{m.get('total_trades',0):>7}")

    base_version = next_version()
    print(f"\n  💾 Saving top-{TOP_K} candidates starting at version {base_version}...")
    for i, trial in enumerate(top_trials, 1):
        params  = {**trial.params, "W_STRATEGY": 40}
        metrics = {**trial.user_attrs.get("metrics", {}), "_objective": trial.value}
        v = base_version + (i - 1)
        save_candidate(params, metrics, v, run_date, train_start, train_end, val_start, val_end, rank=i)
        print(f"     Saved candidate v{v} (rank {i}, score={trial.value:.4f})")

    best_version = base_version
    champion = get_champion()
    challenger_score = float(top_trials[0].value)

    if champion is None:
        supabase.table("agent_params").update({
            "status": "champion", "promoted_at": run_date
        }).eq("version", best_version).execute()
        print(f"\n  👑 No champion existed → auto-promoted v{best_version} to CHAMPION")
    else:
        # Retire any pre-existing challenger before promoting the new one.
        supabase.table("agent_params").update({"status": "retired"}).eq("status", "challenger").execute()
        supabase.table("agent_params").update({
            "status": "challenger", "promoted_at": run_date
        }).eq("version", best_version).execute()

        champ_score = float(champion.get("objective_score", 0))
        beat_ratio  = (challenger_score / champ_score) if champ_score > 0 else float("inf")

        print("\n  ⚔️  Champion / Challenger:")
        print(f"     Champion  : v{champion['version']}  score={champ_score:.4f}")
        print(f"     Challenger: v{best_version}  score={challenger_score:.4f}  "
              f"({(beat_ratio - 1) * 100:+.1f}% vs champion)")
        if challenger_score >= champ_score * CHAMPION_BEAT_MARGIN:
            print(f"\n  ✅  Challenger beats champion by ≥{(CHAMPION_BEAT_MARGIN-1)*100:.0f}%."
                  " Eligible for human-review promotion via the dashboard.")
        else:
            need = champ_score * CHAMPION_BEAT_MARGIN
            print(f"\n  ⏳  Challenger needs ≥{need:.4f} to be eligible for promotion. "
                  f"Monitor paper trades; do not promote yet.")

    supabase.table("optimization_runs").insert({
        "run_date":           run_date,
        "n_trials":           N_TRIALS,
        "n_valid_trials":     len(valid_trials),
        "best_score":         round(challenger_score, 5),
        "best_profit_factor": round(top_trials[0].user_attrs.get("metrics",{}).get("profit_factor",0), 3),
        "best_win_rate":      round(top_trials[0].user_attrs.get("metrics",{}).get("win_rate",0), 1),
        "best_avg_return":    round(top_trials[0].user_attrs.get("metrics",{}).get("avg_return",0), 2),
        "champion_version":   champion["version"] if champion else best_version,
        "challenger_version": best_version,
        "stocks_used":        len([d for d in all_data.values() if not d.empty]),
    }).execute()

    print("\n  Done ✅ — results saved to Supabase\n")


if __name__ == "__main__":
    run()
