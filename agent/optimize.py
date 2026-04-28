#!/usr/bin/env python3
"""
============================================================
  Parameter Optimizer  —  v1
  Runs weekly via GitHub Actions (Sunday 11 PM IST)

  Stack: Optuna + Supabase
  Method: Walk-forward cross-validation to prevent overfitting
  Output: Writes top candidates to agent_params table
          Promotes best candidate as challenger

  Flow:
  1. Fetch 2 years of NSE data for a sample of 20 stocks
  2. Split into rolling 3-month train / 1-month validate windows
  3. Optuna searches 200 parameter combinations
  4. Each trial is scored on walk-forward validation metrics
  5. Top 5 candidates saved to Supabase as 'candidate'
  6. Best candidate promoted to 'challenger'
  7. If no champion exists yet, challenger auto-promoted to champion
============================================================
"""

import os, sys, json, warnings
from datetime import datetime, timedelta, date

import yfinance as yf
import pandas as pd
import numpy as np
import optuna
from supabase import create_client, Client

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  SUPABASE
# ─────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────────
#  OPTIMIZATION SAMPLE STOCKS
#  (representative subset — enough variety, not too slow)
# ─────────────────────────────────────────────
SAMPLE_STOCKS = [
    # Banks / Financials
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
    "BAJFINANCE", "BAJAJFINSV", "JIOFIN", "PFC", "RECLTD",

    # IT / Tech
    "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
    "LTIM", "PERSISTENT", "COFORGE",

    # FMCG / Consumption
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",

    # Pharma / Healthcare
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",

    # Auto / Auto Ancillaries
    "MARUTI", "M&M", "TMPV", "EICHERMOT", "HEROMOTOCO",
    "TVSMOTOR",

    # Energy / Utilities
    "RELIANCE", "ONGC", "NTPC", "POWERGRID", "BPCL",

    # Metals / Industrials / Infra
    "LT", "ULTRACEMCO", "TATASTEEL", "JSWSTEEL", "HINDALCO",
    "ADANIPORTS", "SIEMENS", "BEL", "HAL",

    # Retail / Travel / Misc high-beta
    "TRENT", "INDIGO", "ZOMATO"
]

# Walk-forward config
TRAIN_MONTHS    = 9      # months of training data per window
VALIDATE_MONTHS = 3      # months of validation data per window
N_WINDOWS       = 3      # number of walk-forward windows
N_TRIALS        = 200    # Optuna trials
TOP_K           = 5      # top candidates to save
DATA_DAYS       = 730    # how many days of history to fetch (2 years)

# ─────────────────────────────────────────────
#  INDICATORS  (same as analyze.py — duplicated for standalone use)
# ─────────────────────────────────────────────

def ema(s, p):       return s.ewm(span=p, adjust=False).mean()
def rsi(s, p=14):
    d=s.diff()
    ag=d.clip(lower=0).ewm(alpha=1/p,min_periods=p,adjust=False).mean()
    al=(-d.clip(upper=0)).ewm(alpha=1/p,min_periods=p,adjust=False).mean()
    return 100-(100/(1+ag/al.replace(0,np.nan)))
def macd_ind(s, fast, slow, sig):
    ml=ema(s,fast)-ema(s,slow); sl=ema(ml,sig); return ml,sl,ml-sl
def bollinger(s, p, std):
    m=s.rolling(p).mean(); sg=s.rolling(p).std(); return m+std*sg,m,m-std*sg
def atr(h,l,c,p):
    tr=pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/p,min_periods=p,adjust=False).mean()
def supertrend(h,l,c,p,mult):
    atr_v=atr(h,l,c,p); hl2=(h+l)/2
    up=hl2+mult*atr_v; dn=hl2-mult*atr_v
    trend=pd.Series(1,index=c.index); fu,fl=up.copy(),dn.copy()
    for i in range(1,len(c)):
        fu.iloc[i]=up.iloc[i] if(up.iloc[i]<fu.iloc[i-1] or c.iloc[i-1]>fu.iloc[i-1]) else fu.iloc[i-1]
        fl.iloc[i]=dn.iloc[i] if(dn.iloc[i]>fl.iloc[i-1] or c.iloc[i-1]<fl.iloc[i-1]) else fl.iloc[i-1]
        if   trend.iloc[i-1]==-1 and c.iloc[i]>fu.iloc[i]: trend.iloc[i]= 1
        elif trend.iloc[i-1]== 1 and c.iloc[i]<fl.iloc[i]: trend.iloc[i]=-1
        else:                                                 trend.iloc[i]=trend.iloc[i-1]
    return trend, pd.Series(np.where(trend==1,fl,fu),index=c.index)

# ─────────────────────────────────────────────
#  SIGNAL GENERATORS  (param-aware)
# ─────────────────────────────────────────────

def sig_ema(df, p):
    e_s=ema(df.Close,p["EMA_SHORT"]); e_l=ema(df.Close,p["EMA_LONG"])
    s=pd.Series(0,index=df.index)
    s[(e_s>e_l)&(e_s.shift()<=e_l.shift())]= 1
    s[(e_s<e_l)&(e_s.shift()>=e_l.shift())]=-1
    return s

def sig_rsi_macd(df, p):
    r=rsi(df.Close,p["RSI_PERIOD"])
    _,_,hist=macd_ind(df.Close,p["MACD_FAST"],p["MACD_SLOW"],p["MACD_SIGNAL"])
    s=pd.Series(0,index=df.index)
    s[(r<p["RSI_OVERSOLD"]) &(hist>0)&(hist.shift()<=0)]= 1
    s[(r>p["RSI_OVERBOUGHT"])&(hist<0)&(hist.shift()>=0)]=-1
    return s

def sig_bb(df, p):
    up,_,lo=bollinger(df.Close,p["BB_PERIOD"],p["BB_STD"])
    r=rsi(df.Close,p["RSI_PERIOD"]); s=pd.Series(0,index=df.index)
    s[(df.Low<=lo) &(df.Close>lo)&(r<50)]= 1
    s[(df.High>=up)&(df.Close<up)&(r>50)]=-1
    return s

def sig_donchian(df, p):
    period = int(p.get("DONCHIAN_PERIOD", 20))
    hh = df.High.rolling(period).max()
    ll = df.Low.rolling(period).min()
    s = pd.Series(0, index=df.index)
    s[(df.Close > hh.shift(1)) & (df.Close.shift(1) <= hh.shift(1))] = 1
    s[(df.Close < ll.shift(1)) & (df.Close.shift(1) >= ll.shift(1))] = -1
    return s

def sig_volume_breakout(df, p):
    period   = int(p.get("DONCHIAN_PERIOD", 20))
    vol_mult = float(p.get("VOLUME_MULT", 1.5))
    hh = df.High.rolling(period).max()
    ll = df.Low.rolling(period).min()
    avg_vol = df.Volume.rolling(20).mean()
    s = pd.Series(0, index=df.index)
    s[(df.Close > hh.shift(1)) & (df.Close.shift(1) <= hh.shift(1)) & (df.Volume > avg_vol * vol_mult)] = 1
    s[(df.Close < ll.shift(1)) & (df.Close.shift(1) >= ll.shift(1)) & (df.Volume > avg_vol * vol_mult)] = -1
    return s

def sig_rsi_trend_shift(df, p):
    r   = rsi(df.Close, p["RSI_PERIOD"])
    e_l = ema(df.Close, p["EMA_LONG"])
    mid = float(p.get("RSI_MIDLINE", 50))
    s   = pd.Series(0, index=df.index)
    s[(df.Close > e_l) & (r > mid) & (r.shift(1) <= mid)] = 1
    s[(df.Close < e_l) & (r < mid) & (r.shift(1) >= mid)] = -1
    return s

# ─────────────────────────────────────────────
#  BACKTEST  (with realistic slippage + brokerage)
# ─────────────────────────────────────────────
SLIPPAGE_PCT  = 0.10   # 0.1% slippage per trade (entry + exit combined)
BROKERAGE_PCT = 0.05   # Zerodha-style flat brokerage approximation

def _finite_float(v, default=None):
    try:
        fv = float(v)
        return fv if np.isfinite(fv) else default
    except Exception:
        return default


def _dynamic_levels(df, signal_idx, entry_price, p):
    lookback = int(p.get("RR_LOOKBACK", 20))
    atr_period = int(p.get("ATR_PERIOD", 14))
    support_s = df.Low.rolling(lookback).min().shift(1)
    resist_s = df.High.rolling(lookback).max().shift(1)
    atr_s = atr(df.High, df.Low, df.Close, atr_period)
    support = _finite_float(support_s.iloc[signal_idx] if signal_idx < len(support_s) else None)
    resistance = _finite_float(resist_s.iloc[signal_idx] if signal_idx < len(resist_s) else None)
    atr_now = _finite_float(atr_s.iloc[signal_idx] if signal_idx < len(atr_s) else None)
    stop_buf = float(p.get("ATR_STOP_BUFFER", 0.50))
    target_buf = float(p.get("ATR_TARGET_BUFFER", 0.50))
    max_risk_atr = float(p.get("MAX_RISK_ATR", 3.00))
    max_risk_pct = float(p.get("MAX_RISK_PCT", 8.00))
    min_rr_ratio = float(p.get("MIN_RR_RATIO", 1.50))

    if atr_now is None or atr_now <= 0:
        # ATR fallback: same risk envelope and R:R floor as the live agent.
        sl_pct_p  = min(float(p["BT_SL_PCT"]), max_risk_pct)
        tgt_pct_p = float(p["BT_TARGET_PCT"])
        sl  = entry_price * (1 - sl_pct_p  / 100.0)
        tgt = entry_price * (1 + tgt_pct_p / 100.0)
    else:
        sl = (support - stop_buf * atr_now) if support is not None and support < entry_price else entry_price - 1.25 * atr_now
        sl = max(sl, entry_price - max_risk_atr * atr_now, entry_price * (1 - max_risk_pct / 100.0))
        if sl >= entry_price:
            sl = max(entry_price - 1.25 * atr_now, entry_price * (1 - max_risk_pct / 100.0))
        tgt = (resistance + target_buf * atr_now) if resistance is not None and resistance > entry_price else entry_price + 1.75 * atr_now
        if tgt <= entry_price:
            tgt = entry_price + 1.75 * atr_now

    # Universal R:R floor — applies regardless of which path produced sl/tgt.
    actual_risk = entry_price - sl
    if actual_risk > 0 and min_rr_ratio > 0:
        tgt = max(tgt, entry_price + actual_risk * min_rr_ratio)
    return sl, tgt


def backtest_with_costs(df, signals, p):
    """
    Long-only next-bar backtest with:
    - BUY signal on bar i enters at bar i+1 open
    - EXIT signal on bar i closes an open long at bar i+1 open
    - Dynamic stop/target from prior support/resistance plus ATR
    - Slippage + brokerage deducted from each trade
    """
    trades, reasons = [], []
    in_t, ep, entry_idx = False, 0.0, -1
    sl_px = tgt_px = None
    closes=df.Close.values; highs=df.High.values
    lows=df.Low.values; opens=df.Open.values
    max_hold=p["BT_MAX_HOLD"]
    cost_pct = SLIPPAGE_PCT + BROKERAGE_PCT   # deducted from each trade's gross return

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
                close_trade(float(opens[i]), "exit_signal")
                continue
            if sl_px is not None and lows[i] <= sl_px:
                close_trade(float(sl_px), "sl")
                continue
            if tgt_px is not None and highs[i] >= tgt_px:
                close_trade(float(tgt_px), "target")
                continue
            if i >= entry_idx + max_hold:
                close_trade(float(closes[i]), "timeout")
                continue
        if not in_t and prev_sig == 1:
            ep = float(opens[i])
            if ep <= 0:
                continue
            sl_px, tgt_px = _dynamic_levels(df, i - 1, ep, p)
            entry_idx, in_t = i, True
            if sl_px is not None and lows[i] <= sl_px:
                close_trade(float(sl_px), "sl")
                continue
            if tgt_px is not None and highs[i] >= tgt_px:
                close_trade(float(tgt_px), "target")
                continue

    if not trades:
        return dict(win_rate=0, avg_return=0, median_return=0, trades=0,
                    profit_factor=0, max_drawdown=0, expectancy=0)

    wins=[t for t in trades if t>0]; losses=[t for t in trades if t<=0]
    gp,gl=sum(wins),abs(sum(losses))
    pf=round(gp/gl,2) if gl>0 else 99.0
    eq=np.cumsum(trades); peak=np.maximum.accumulate(eq)
    max_dd=float(abs((eq-peak).min())) if len(eq)>0 else 0
    wr=len(wins)/len(trades)
    avg_win =float(np.mean(wins)) if wins else 0
    avg_loss=float(np.mean([abs(l) for l in losses])) if losses else 0
    expectancy = wr*avg_win - (1-wr)*avg_loss

    return dict(
        win_rate=round(wr*100,1),
        avg_return=round(float(np.mean(trades)),2),
        median_return=round(float(np.median(trades)),2),
        trades=len(trades),
        profit_factor=min(float(pf),99.0),
        max_drawdown=round(max_dd,2),
        expectancy=round(expectancy,3),
    )

# ─────────────────────────────────────────────
#  WALK-FORWARD EVALUATION
# ─────────────────────────────────────────────

def walk_forward_score(params: dict, all_data: dict) -> dict:
    """
    Run walk-forward backtesting over N_WINDOWS.
    For each window:
      - Train period: first TRAIN_MONTHS months of the slice
      - Validate period: next VALIDATE_MONTHS months
    We only evaluate signals on the VALIDATE slice to prevent lookahead.

    Returns aggregated metrics across all windows and stocks.
    """
    # Build time windows
    latest = max(df.index.max() for df in all_data.values() if not df.empty)
    windows = []
    for w in range(N_WINDOWS):
        # End of validate is 'w * validate_months' before now
        val_end   = latest - pd.DateOffset(months=w * VALIDATE_MONTHS)
        val_start = val_end  - pd.DateOffset(months=VALIDATE_MONTHS)
        trn_start = val_start - pd.DateOffset(months=TRAIN_MONTHS)
        windows.append((trn_start, val_start, val_end))

    all_metrics = []
    strategies  = [
        ("EMA Crossover",   lambda df: sig_ema(df, params)),
        ("RSI + MACD",      lambda df: sig_rsi_macd(df, params)),
        ("Bollinger",       lambda df: sig_bb(df, params)),
        ("Donchian",        lambda df: sig_donchian(df, params)),
        ("Volume Breakout", lambda df: sig_volume_breakout(df, params)),
        ("RSI Trend Shift", lambda df: sig_rsi_trend_shift(df, params)),
    ]

    for trn_start, val_start, val_end in windows:
        for ticker, full_df in all_data.items():
            if full_df.empty or len(full_df) < 60:
                continue
            # Slice validate window (with enough warmup data from train period)
            warmup_start = trn_start
            slice_df = full_df[(full_df.index >= warmup_start) & (full_df.index <= val_end)].copy()
            if len(slice_df) < 50:
                continue

            # Generate signals on full slice (warmup included) to prime indicators
            for name, fn in strategies:
                try:
                    all_sigs = fn(slice_df)
                    # Only trade in the validation window
                    val_mask = (slice_df.index >= val_start) & (slice_df.index <= val_end)
                    val_sigs = all_sigs.copy()
                    val_sigs[~val_mask] = 0   # zero out signals outside validate window

                    bt = backtest_with_costs(slice_df, val_sigs, params)
                    if bt["trades"] > 0:
                        all_metrics.append(bt)
                except Exception:
                    continue

    if not all_metrics:
        return dict(profit_factor=0, avg_return=-10, win_rate=0,
                    max_drawdown=100, expectancy=-1, total_trades=0)

    return dict(
        profit_factor = float(np.mean([m["profit_factor"]  for m in all_metrics])),
        avg_return    = float(np.mean([m["avg_return"]      for m in all_metrics])),
        win_rate      = float(np.mean([m["win_rate"]        for m in all_metrics])),
        max_drawdown  = float(np.mean([m["max_drawdown"]    for m in all_metrics])),
        expectancy    = float(np.mean([m["expectancy"]      for m in all_metrics])),
        total_trades  = int(sum(m["trades"]                 for m in all_metrics)),
    )

# ─────────────────────────────────────────────
#  OBJECTIVE FUNCTION
# ─────────────────────────────────────────────

def make_objective(all_data):
    def objective(trial):
        # --- Parameter search space ---
        ema_short = trial.suggest_int("EMA_SHORT",   5,  20)
        ema_long  = trial.suggest_int("EMA_LONG",   22,  60)   # constrained > EMA_SHORT
        params = {
            "EMA_SHORT":          ema_short,
            "EMA_LONG":           ema_long,
            "RSI_PERIOD":         trial.suggest_int(  "RSI_PERIOD",    7,  21),
            "RSI_OVERSOLD":       trial.suggest_int(  "RSI_OVERSOLD", 25,  50),
            "RSI_OVERBOUGHT":     trial.suggest_int(  "RSI_OVERBOUGHT",55, 78),
            "MACD_FAST":          trial.suggest_int(  "MACD_FAST",     8,  16),
            "MACD_SLOW":          trial.suggest_int(  "MACD_SLOW",    20,  35),
            "MACD_SIGNAL":        trial.suggest_int(  "MACD_SIGNAL",   5,  12),
            "BB_PERIOD":          trial.suggest_int(  "BB_PERIOD",    10,  30),
            "BB_STD":             trial.suggest_float("BB_STD",       1.5,  3.0),
            "ATR_PERIOD":         trial.suggest_int(  "ATR_PERIOD",    7,  21),
            "SUPERTREND_MULT":    trial.suggest_float("SUPERTREND_MULT",1.5, 5.0),
            "BT_SL_PCT":          trial.suggest_float("BT_SL_PCT",    2.0,  8.0),
            "BT_TARGET_PCT":      trial.suggest_float("BT_TARGET_PCT", 4.0, 18.0),
            "BT_MAX_HOLD":        trial.suggest_int(  "BT_MAX_HOLD",   3,  25),
            "MIN_WEIGHTED_SCORE": trial.suggest_float("MIN_WEIGHTED_SCORE",0.05,0.55),
            "RR_LOOKBACK":        trial.suggest_int(  "RR_LOOKBACK", 14, 35),
            "ATR_STOP_BUFFER":    trial.suggest_float("ATR_STOP_BUFFER", 0.25, 1.00),
            "ATR_TARGET_BUFFER":  trial.suggest_float("ATR_TARGET_BUFFER", 0.25, 1.25),
            "MAX_RISK_ATR":       trial.suggest_float("MAX_RISK_ATR", 1.50, 3.50),
            "MAX_RISK_PCT":       trial.suggest_float("MAX_RISK_PCT", 4.00, 10.00),
            "MIN_RR_RATIO":       trial.suggest_float("MIN_RR_RATIO", 1.20, 2.20),
            # New strategy params
            "DONCHIAN_PERIOD":    trial.suggest_int(  "DONCHIAN_PERIOD",  10, 40),
            "VOLUME_MULT":        trial.suggest_float("VOLUME_MULT",      1.2,  3.0),
            "RSI_MIDLINE":        trial.suggest_int(  "RSI_MIDLINE",      45,  55),
            # Composite score weights (sampled independently, normalised below)
            "W_STRATEGY":         40,  # kept fixed — strategy quality is the backbone
            "W_RSI":              trial.suggest_int("W_RSI",    10, 30),
            "W_VOLUME":           trial.suggest_int("W_VOLUME",  5, 20),
            "W_RR":               trial.suggest_int("W_RR",      5, 20),
            "W_REGIME":           trial.suggest_int("W_REGIME",  5, 15),
        }
        # Hard constraint: EMA_LONG > EMA_SHORT + 5
        if params["EMA_LONG"] <= params["EMA_SHORT"] + 5:
            return -999.0
        # Hard constraint: MACD_SLOW > MACD_FAST + 4
        if params["MACD_SLOW"] <= params["MACD_FAST"] + 4:
            return -999.0
        # Hard constraint: target > 1.5× SL (minimum R:R)
        if params["BT_TARGET_PCT"] < params["BT_SL_PCT"] * 1.5:
            return -999.0

        metrics = walk_forward_score(params, all_data)
        trial.set_user_attr("metrics", metrics)

        if metrics["total_trades"] < 10:
            return -998.0   # too few trades — params too restrictive

        # Composite objective: profit factor + avg return + win rate - drawdown
        pf_norm  = min(metrics["profit_factor"] / 3.0, 1.0)     # 3.0 = excellent
        ret_norm = max(min(metrics["avg_return"] / 5.0, 1.0), 0) # 5% avg = excellent
        wr_norm  = metrics["win_rate"] / 100.0
        dd_norm  = min(metrics["max_drawdown"] / 20.0, 1.0)      # 20% dd = terrible

        score = (
            0.35 * pf_norm  +
            0.25 * ret_norm +
            0.20 * wr_norm  -
            0.20 * dd_norm
        )
        return round(float(score), 5)

    return objective

# ─────────────────────────────────────────────
#  DATA FETCH (for optimizer — cached dict)
# ─────────────────────────────────────────────

def fetch_all_data() -> dict:
    print(f"  Fetching {len(SAMPLE_STOCKS)} stocks ({DATA_DAYS} days history)...")
    data = {}
    for i, ticker in enumerate(SAMPLE_STOCKS, 1):
        sys.stdout.write(f"\r    {i}/{len(SAMPLE_STOCKS)}  {ticker:<12}")
        sys.stdout.flush()
        try:
            df = yf.download(
                ticker+".NS",
                start=datetime.today()-timedelta(days=DATA_DAYS),
                end=datetime.today(),
                progress=False, auto_adjust=True
            )
            if df.empty or len(df) < 100:
                data[ticker] = pd.DataFrame()
                continue
            df.columns = [c[0] if isinstance(c,tuple) else c for c in df.columns]
            data[ticker] = df[["Open","High","Low","Close","Volume"]].dropna()
        except Exception:
            data[ticker] = pd.DataFrame()
        import time; time.sleep(0.15)
    sys.stdout.write("\r" + " "*50 + "\r")
    valid = sum(1 for d in data.values() if not d.empty)
    print(f"  {valid}/{len(SAMPLE_STOCKS)} stocks loaded successfully\n")
    return data

# ─────────────────────────────────────────────
#  SUPABASE HELPERS
# ─────────────────────────────────────────────

def next_version() -> int:
    """Get next version number for agent_params."""
    try:
        res = supabase.table("agent_params").select("version").order("version",desc=True).limit(1).execute()
        if res.data:
            return int(res.data[0]["version"]) + 1
    except Exception:
        pass
    return 1

def get_champion() -> dict | None:
    try:
        res = supabase.table("agent_params").select("*").eq("status","champion")\
              .order("promoted_at",desc=True).limit(1).execute()
        return res.data[0] if res.data else None
    except Exception:
        return None

def save_candidate(params, metrics, version, run_date, train_start, train_end, val_start, val_end, rank):
    supabase.table("agent_params").insert({
        "version":        version,
        "status":         "candidate",
        "params_json":    json.dumps(params),
        "objective_score":round(metrics.get("_objective", 0), 5),
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
    print(f"\n🔬 Stock Agent Optimizer — {run_date}")
    print(f"   Trials: {N_TRIALS}  |  Walk-forward windows: {N_WINDOWS}")
    print(f"   Train: {TRAIN_MONTHS}m  Validate: {VALIDATE_MONTHS}m\n")

    # 1. Fetch historical data
    all_data = fetch_all_data()
    if sum(1 for d in all_data.values() if not d.empty) < 5:
        print("  ❌ Not enough data to optimize. Aborting.")
        return

    # Compute time range for logging
    all_dates = [df.index for df in all_data.values() if not df.empty]
    train_start = min(d.min() for d in all_dates)
    train_end   = max(d.max() for d in all_dates)
    val_start   = train_end - pd.DateOffset(months=VALIDATE_MONTHS * N_WINDOWS)
    val_end     = train_end

    # 2. Run Optuna
    print(f"  🔍 Running {N_TRIALS} Optuna trials...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
    )

    objective_fn = make_objective(all_data)

    # Warm-start the search with the live agent's DEFAULT_PARAMS so the first
    # trial is the configuration users would see if they ran today without an
    # active champion. Values are kept inside the suggest_* ranges above.
    default_params_for_optuna = {
        "EMA_SHORT": 9, "EMA_LONG": 21,
        "RSI_PERIOD": 14, "RSI_OVERSOLD": 48, "RSI_OVERBOUGHT": 58,
        "MACD_FAST": 12, "MACD_SLOW": 26, "MACD_SIGNAL": 9,
        "BB_PERIOD": 20, "BB_STD": 2.0,
        "ATR_PERIOD": 14, "SUPERTREND_MULT": 3.0,
        "BT_SL_PCT": 5.0, "BT_TARGET_PCT": 10.0, "BT_MAX_HOLD": 15,
        "MIN_WEIGHTED_SCORE": 0.08,
        "RR_LOOKBACK": 20, "ATR_STOP_BUFFER": 0.50, "ATR_TARGET_BUFFER": 0.50,
        "MAX_RISK_ATR": 3.00, "MAX_RISK_PCT": 8.00, "MIN_RR_RATIO": 1.50,
        "DONCHIAN_PERIOD": 20, "VOLUME_MULT": 1.2, "RSI_MIDLINE": 50,
        "W_RSI": 20, "W_VOLUME": 15, "W_RR": 15, "W_REGIME": 10,
    }
    study.enqueue_trial(default_params_for_optuna)
    study.optimize(objective_fn, n_trials=N_TRIALS, show_progress_bar=False)

    # 3. Get top-K valid trials
    valid_trials = [
        t for t in study.trials
        if t.value is not None and t.value > -900
    ]
    valid_trials.sort(key=lambda t: t.value, reverse=True)
    top_trials = valid_trials[:TOP_K]

    if not top_trials:
        print("  ❌ No valid trials found. Check data quality.")
        return

    print(f"\n  ✅ Optimization complete")
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

    # 4. Save top-K candidates to Supabase
    base_version = next_version()
    print(f"\n  💾 Saving top-{TOP_K} candidates starting at version {base_version}...")
    saved_ids = []
    for i, trial in enumerate(top_trials, 1):
        params = {**trial.params, "W_STRATEGY": 40}
        metrics = {**trial.user_attrs.get("metrics", {}), "_objective": trial.value}
        v = base_version + (i - 1)
        save_candidate(params, metrics, v, run_date, train_start, train_end, val_start, val_end, rank=i)
        saved_ids.append(v)
        print(f"     Saved candidate v{v} (rank {i}, score={trial.value:.4f})")

    best_version = base_version
    champion = get_champion()

    if champion is None:
        # No champion yet → auto-promote to champion
        supabase.table("agent_params")\
            .update({"status": "champion", "promoted_at": run_date})\
            .eq("version", best_version)\
            .execute()
        print(f"\n  👑 No champion existed → auto-promoted v{best_version} to CHAMPION")
    else:
        # Champion exists → promote best candidate as challenger
        # First retire any existing challenger
        supabase.table("agent_params")\
            .update({"status": "retired"})\
            .eq("status", "challenger")\
            .execute()
        supabase.table("agent_params")\
            .update({"status": "challenger", "promoted_at": run_date})\
            .eq("version", best_version)\
            .execute()

        champ_score = float(champion.get("objective_score", 0))
        print(f"\n  ⚔️  Champion/Challenger mode:")
        print(f"     Champion  : v{champion['version']}  score={champ_score:.4f}")
        print(f"     Challenger: v{best_version}  score={top_trials[0].value:.4f}")
        if top_trials[0].value > champ_score * 1.05:
            print(f"\n  ℹ️  Challenger scores ≥5% better. Consider promoting via dashboard.")
        else:
            print(f"\n  ℹ️  Challenger not yet ≥5% better than champion. Monitor paper trades.")

    # 6. Update optimization run log
    supabase.table("optimization_runs").insert({
        "run_date":           run_date,
        "n_trials":           N_TRIALS,
        "n_valid_trials":     len(valid_trials),
        "best_score":         round(top_trials[0].value, 5),
        "best_profit_factor": round(top_trials[0].user_attrs.get("metrics",{}).get("profit_factor",0), 3),
        "best_win_rate":      round(top_trials[0].user_attrs.get("metrics",{}).get("win_rate",0), 1),
        "best_avg_return":    round(top_trials[0].user_attrs.get("metrics",{}).get("avg_return",0), 2),
        "champion_version":   champion["version"] if champion else best_version,
        "challenger_version": best_version,
        "stocks_used":        len([d for d in all_data.values() if not d.empty]),
    }).execute()

    print(f"\n  Done ✅ — results saved to Supabase\n")


if __name__ == "__main__":
    run()
