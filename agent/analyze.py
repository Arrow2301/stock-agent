#!/usr/bin/env python3
"""
============================================================
  Indian Stock Market Analysis Agent  —  v8

  v8 (this version) cleanup over v7+Phase 3:
   ─ Strategy series are computed ONCE per ticker (was 2×).
   ─ EXIT signals on a stock you currently hold raise an
     alert (REVIEW.md A3 fix). No auto-close — alert only.
   ─ A risk-aware "suggested_qty" is computed for every BUY
     using agent/risk.py:recommend_position_size and stored
     on the recommendation row.
   ─ Dead supertrend_up / supertrend_line / score columns
     are no longer written.
   ─ Banner now correctly says "Stock Agent v8".

  Phase 3 (carried forward):
   ─ India VIX features attached to every BUY's feature
     vector (vix_level, vix_change_5d, vix_zscore_60d).
   ─ Calibrated LightGBM scorer via agent/score_model.py.

  v7 (carried forward):
   ─ 4-strategy roster (Donchian, EMA, RSI Trend Shift, Bollinger).
   ─ Hard gates: BUY rejected if multi-strat or BEARISH regime.
   ─ news_v2: junk filter + financial keyword overrides.
   ─ Holiday-aware workflow.
============================================================
"""

import os
import sys
import time
import json
import math
import warnings
from datetime import datetime, timedelta

import requests
import yfinance as yf
import pandas as pd
import numpy as np
from supabase import create_client, Client

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from news_v2 import fetch_news_sentiment           # noqa: E402
from risk import recommend_position_size           # noqa: E402
from portfolio import open_tickers, open_positions_with_sector, recent_closed_pnl_pcts  # noqa: E402

try:
    from score_model import load_model, predict_p_win
    _SCORE_MODEL = load_model()           # (pipeline, features, label_kind) or None
except Exception as _score_err:
    print(f"  ⚠️  score_model unavailable: {_score_err}")
    _SCORE_MODEL = None

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  SUPABASE
#
#  We tolerate a missing or invalid client at IMPORT TIME so this
#  module can be imported by the self-check harness, by build_training_data
#  in --self-check mode, and by unit tests. run() will fail fast if the
#  client wasn't successfully created.
# ─────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as _e:
        # Print but don't crash — the caller may be importing for testing.
        print(f"  ⚠️  Supabase client init deferred: {_e}")

# Capital base for the suggested-qty computation. Defaults to ₹100,000.
DEFAULT_ACCOUNT_INR = float(os.environ.get("ACCOUNT_INR", "100000"))
DEFAULT_RISK_PCT    = float(os.environ.get("RISK_PCT_PER_TRADE", "1.0"))

# ─────────────────────────────────────────────
#  WATCHLIST  (Nifty 200, despite the historical name)
# ─────────────────────────────────────────────
NIFTY200 = [
    "360ONE", "ABB", "ACC", "APLAPOLLO", "AUBANK", "ADANIENSOL", "ADANIENT", "ADANIGREEN",
    "ADANIPORTS", "ADANIPOWER", "ATGL", "ABCAPITAL", "ALKEM", "AMBUJACEM", "APOLLOHOSP", "ASHOKLEY",
    "ASIANPAINT", "ASTRAL", "AUROPHARMA", "DMART", "AXISBANK", "BSE", "BAJAJ-AUTO", "BAJFINANCE",
    "BAJAJFINSV", "BAJAJHLDNG", "BAJAJHFL", "BANKBARODA", "BANKINDIA", "BDL", "BEL", "BHARATFORG",
    "BHEL", "BPCL", "BHARTIARTL", "BHARTIHEXA", "BIOCON", "BLUESTARCO", "BOSCHLTD", "BRITANNIA",
    "CGPOWER", "CANBK", "CHOLAFIN", "CIPLA", "COALINDIA", "COCHINSHIP", "COFORGE", "COLPAL",
    "CONCOR", "COROMANDEL", "CUMMINSIND", "DLF", "DABUR", "DIVISLAB", "DIXON", "DRREDDY",
    "EICHERMOT", "ETERNAL", "EXIDEIND", "NYKAA", "FEDERALBNK", "FORTIS", "GAIL", "GMRAIRPORT",
    "GLENMARK", "GODFRYPHLP", "GODREJCP", "GODREJPROP", "GRASIM", "HCLTECH", "HDFCAMC", "HDFCBANK",
    "HDFCLIFE", "HAVELLS", "HEROMOTOCO", "HINDALCO", "HAL", "HINDPETRO", "HINDUNILVR", "HINDZINC",
    "POWERINDIA", "HUDCO", "HYUNDAI", "ICICIBANK", "ICICIGI", "IDFCFIRSTB", "IRB", "ITCHOTELS",
    "ITC", "INDIANB", "INDHOTEL", "IOC", "IRCTC", "IRFC", "IREDA", "IGL",
    "INDUSTOWER", "INDUSINDBK", "NAUKRI", "INFY", "INDIGO", "JSWENERGY", "JSWSTEEL", "JINDALSTEL",
    "JIOFIN", "JUBLFOOD", "KEI", "KPITTECH", "KALYANKJIL", "KOTAKBANK", "LTF", "LICHSGFIN",
    "LTIM", "LT", "LICI", "LODHA", "LUPIN", "MRF", "M&MFIN", "M&M",
    "MANKIND", "MARICO", "MARUTI", "MFSL", "MAXHEALTH", "MAZDOCK", "MOTILALOFS", "MPHASIS",
    "MUTHOOTFIN", "NHPC", "NMDC", "NTPCGREEN", "NTPC", "NATIONALUM", "NESTLEIND", "OBEROIRLTY",
    "ONGC", "OIL", "PAYTM", "OFSS", "POLICYBZR", "PIIND", "PAGEIND", "PATANJALI",
    "PERSISTENT", "PHOENIXLTD", "PIDILITIND", "POLYCAB", "PFC", "POWERGRID", "PREMIERENE", "PRESTIGE",
    "PNB", "RECLTD", "RVNL", "RELIANCE", "SBICARD", "SBILIFE", "SRF", "MOTHERSON",
    "SHREECEM", "SHRIRAMFIN", "ENRIN", "SIEMENS", "SOLARINDS", "SONACOMS", "SBIN", "SAIL",
    "SUNPHARMA", "SUPREMEIND", "SUZLON", "SWIGGY", "TVSMOTOR", "TATACOMM", "TCS", "TATACONSUM",
    "TATAELXSI", "TMPV", "TATAPOWER", "TATASTEEL", "TATATECH", "TECHM", "TITAN", "TORNTPHARM",
    "TORNTPOWER", "TRENT", "TIINDIA", "UPL", "ULTRACEMCO", "UNIONBANK", "UNITDSPR", "VBL",
    "VEDL", "VMM", "IDEA", "VOLTAS", "WAAREEENER", "WIPRO", "YESBANK", "ZYDUSLIFE",
]
EXTRA_WATCHLIST: list[str] = []
ALL_TICKERS = NIFTY200 + EXTRA_WATCHLIST

# Backward-compat alias for build_training_data.py and any external code.
NIFTY50 = NIFTY200

# ─────────────────────────────────────────────
#  DEFAULT PARAMETERS
# ─────────────────────────────────────────────
DEFAULT_PARAMS = {
    "EMA_SHORT":          9,
    "EMA_LONG":           21,
    "RSI_PERIOD":         14,
    "RSI_OVERSOLD":       48,
    "RSI_OVERBOUGHT":     58,
    "RSI_MIDLINE":        50,
    "MACD_FAST":          12,
    "MACD_SLOW":          26,
    "MACD_SIGNAL":        9,
    "BB_PERIOD":          20,
    "BB_STD":             2.0,
    "DONCHIAN_PERIOD":    20,
    "VOLUME_MULT":        1.2,
    "ATR_PERIOD":         14,
    # BT_*_PCT are LAST-RESORT fallbacks for dynamic_trade_levels()
    # when ATR is unavailable.
    "BT_SL_PCT":          5.0,
    "BT_TARGET_PCT":      10.0,
    "BT_MAX_HOLD":        15,
    "RR_LOOKBACK":        20,
    "ATR_STOP_BUFFER":    0.50,
    "ATR_TARGET_BUFFER":  0.50,
    "MAX_RISK_ATR":       3.00,
    "MAX_RISK_PCT":       8.00,
    "MIN_RR_RATIO":       1.50,
    "MIN_WEIGHTED_SCORE": 0.08,
    "W_STRATEGY":         40,
    "W_RSI":              20,
    "W_VOLUME":           15,
    "W_RR":               15,
    "W_REGIME":           10,
}

FUND_HIGH_PE_THRESHOLD   = 100
FUND_LOW_REV_THRESHOLD   = -0.25
FUND_HIGH_DE_THRESHOLD   = 400

HF_FINBERT_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

NSE_COMPANY_NAMES: dict[str, str] = {
    "RELIANCE":   "Reliance Industries",
    "TCS":        "Tata Consultancy Services",
    "HDFCBANK":   "HDFC Bank",
    "INFY":       "Infosys",
    "ICICIBANK":  "ICICI Bank",
    "ITC":        "ITC Limited",
    "SBIN":       "State Bank of India",
    "BHARTIARTL": "Bharti Airtel",
    "KOTAKBANK":  "Kotak Mahindra Bank",
    "LT":         "Larsen Toubro",
    "AXISBANK":   "Axis Bank",
    "ASIANPAINT": "Asian Paints",
    "MARUTI":     "Maruti Suzuki",
    "TITAN":      "Titan Company",
    "SUNPHARMA":  "Sun Pharmaceutical",
    "WIPRO":      "Wipro",
    "ONGC":       "ONGC India",
    "NTPC":       "NTPC Limited",
    "TATAMOTORS": "Tata Motors",
    "TATASTEEL":  "Tata Steel",
    "BAJFINANCE": "Bajaj Finance",
    "HCLTECH":    "HCL Technologies",
    "M&M":        "Mahindra Mahindra",
    "NESTLEIND":  "Nestle India",
    "CIPLA":      "Cipla",
    "DRREDDY":    "Dr Reddy Laboratories",
    "HEROMOTOCO": "Hero MotoCorp",
    "BRITANNIA":  "Britannia Industries",
    "APOLLOHOSP": "Apollo Hospitals",
    "ADANIPORTS": "Adani Ports",
    "INDUSINDBK": "IndusInd Bank",
    "IRCTC":      "Indian Railway Catering Tourism",
    "IRFC":       "Indian Railway Finance Corporation",
    "ZOMATO":     "Zomato",
    "LTIM":       "LTIMindtree",
    "BAJAJ-AUTO": "Bajaj Auto",
    "BEL":        "Bharat Electronics",
    "HAL":        "Hindustan Aeronautics",
    "TRENT":      "Trent Westside",
    "INDIGO":     "IndiGo Airlines",
    "BPCL":       "Bharat Petroleum",
    "DLF":        "DLF Limited",
}

# ─────────────────────────────────────────────
#  ACTIVE PARAMS
# ─────────────────────────────────────────────
def load_active_params() -> tuple[dict, str]:
    try:
        res = (
            supabase.table("agent_params")
            .select("*")
            .eq("status", "champion")
            .order("promoted_at", desc=True)
            .limit(1)
            .execute()
        )
        if res.data:
            row    = res.data[0]
            params = (json.loads(row["params_json"])
                      if isinstance(row["params_json"], str)
                      else row["params_json"])
            merged = {**DEFAULT_PARAMS, **params}
            return merged, f"v{row['version']} (score={row['objective_score']:.3f})"
    except Exception as e:
        print(f"  ⚠️  Could not load champion params: {e}")
    return DEFAULT_PARAMS.copy(), "defaults"

# ─────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────
def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def rsi(s, p=14):
    d  = s.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / p, min_periods=p, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1 / p, min_periods=p, adjust=False).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))

def macd(s, fast, slow, sig):
    ml = ema(s, fast) - ema(s, slow)
    sl = ema(ml, sig)
    return ml, sl, ml - sl

def bollinger(s, p, std):
    m  = s.rolling(p).mean()
    sg = s.rolling(p).std()
    return m + std * sg, m, m - std * sg

def atr(h, l, c, p):
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / p, min_periods=p, adjust=False).mean()

# ─────────────────────────────────────────────
#  SIGNAL GENERATORS — v7 four-strategy roster
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

def get_strategies(p: dict) -> dict:
    """v7 strategy roster — 4 strategies, evidence-based."""
    return {
        "Donchian":        lambda df: sig_donchian(df, p),
        "EMA Crossover":   lambda df: sig_ema(df, p),
        "RSI Trend Shift": lambda df: sig_rsi_trend_shift(df, p),
        "Bollinger":       lambda df: sig_bb(df, p),
    }

# ─────────────────────────────────────────────
#  DYNAMIC LEVELS + BACKTEST
# ─────────────────────────────────────────────
def _finite_float(v, default=None):
    try:
        fv = float(v)
        return fv if math.isfinite(fv) else default
    except Exception:
        return default


def dynamic_trade_levels(df: pd.DataFrame, signal_idx: int, entry_price: float, p: dict) -> dict:
    """Build SL/target from data available at signal_idx."""
    if df is None or df.empty or signal_idx < 1 or entry_price is None or entry_price <= 0:
        return {"stop_loss": None, "target": None,
                "risk_pct": None, "reward_pct": None, "rr_ratio": None}

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

    rr_floor_applied = False

    if atr_now is None or atr_now <= 0:
        sl_pct_p  = min(float(p.get("BT_SL_PCT", 5.0)), max_risk_pct)
        tgt_pct_p = float(p.get("BT_TARGET_PCT", 10.0))
        sl  = entry_price * (1 - sl_pct_p  / 100.0)
        tgt = entry_price * (1 + tgt_pct_p / 100.0)
    else:
        fallback_sl  = entry_price - 1.25 * atr_now
        fallback_tgt = entry_price + 1.75 * atr_now
        sl = (support - stop_buf * atr_now
              if support is not None and support < entry_price
              else fallback_sl)
        sl = max(sl,
                 entry_price - max_risk_atr * atr_now,
                 entry_price * (1 - max_risk_pct / 100.0))
        if sl >= entry_price:
            sl = max(fallback_sl, entry_price * (1 - max_risk_pct / 100.0))
        tgt = (resistance + target_buf * atr_now
               if resistance is not None and resistance > entry_price
               else fallback_tgt)
        if tgt <= entry_price:
            tgt = fallback_tgt

    actual_risk = entry_price - sl
    if actual_risk > 0 and min_rr_ratio > 0:
        min_tgt = entry_price + actual_risk * min_rr_ratio
        if tgt < min_tgt:
            tgt = min_tgt
            rr_floor_applied = True

    if not (sl is not None and tgt is not None and sl < entry_price < tgt):
        return sanitize_for_json({
            "stop_loss": None, "target": None,
            "risk_pct": None, "reward_pct": None, "rr_ratio": None,
            "support":    round(support, 2)    if support    is not None else None,
            "resistance": round(resistance, 2) if resistance is not None else None,
            "atr":        round(atr_now, 2)    if atr_now    is not None else None,
            "rr_floor_applied": False,
        })

    risk_pct   = (entry_price - sl) / entry_price * 100
    reward_pct = (tgt - entry_price) / entry_price * 100
    rr_ratio   = reward_pct / risk_pct if risk_pct > 0 else None

    return sanitize_for_json({
        "stop_loss":  round(sl, 2),
        "target":     round(tgt, 2),
        "risk_pct":   round(risk_pct, 2),
        "reward_pct": round(reward_pct, 2),
        "rr_ratio":   round(rr_ratio, 2) if rr_ratio is not None else None,
        "support":    round(support, 2)    if support    is not None else None,
        "resistance": round(resistance, 2) if resistance is not None else None,
        "atr":        round(atr_now, 2)    if atr_now    is not None else None,
        "rr_floor_applied": rr_floor_applied,
    })


def _benchmark_return(benchmark_df, entry_date, exit_date) -> float | None:
    if benchmark_df is None or benchmark_df.empty:
        return None
    try:
        b = benchmark_df.copy().sort_index()
        b.columns = [c[0] if isinstance(c, tuple) else c for c in b.columns]
        if "Open" not in b.columns or "Close" not in b.columns:
            return None
        e_slice = b[b.index >= entry_date]
        x_slice = b[b.index <= exit_date]
        if e_slice.empty or x_slice.empty:
            return None
        b_entry = float(e_slice.iloc[0]["Open"])
        b_exit  = float(x_slice.iloc[-1]["Close"])
        if b_entry <= 0:
            return None
        return round((b_exit - b_entry) / b_entry * 100, 2)
    except Exception:
        return None


def backtest(df: pd.DataFrame, signals: pd.Series, p: dict,
             benchmark_df: pd.DataFrame | None = None) -> dict:
    """Long-only backtest with next-bar execution.
       Stop-first when both touched same bar.
    """
    trades, reasons, bench_returns, rel_returns = [], [], [], []
    in_t, ep, entry_idx = False, 0.0, -1
    stop_px = target_px = None
    opens, closes = df.Open.values, df.Close.values
    highs, lows   = df.High.values, df.Low.values
    max_hold = int(p["BT_MAX_HOLD"])

    def close_trade(exit_px: float, exit_idx: int, reason: str):
        nonlocal in_t, ep, entry_idx, stop_px, target_px
        ret = (exit_px - ep) / ep * 100 if ep else 0.0
        trades.append(ret)
        reasons.append(reason)
        b_ret = _benchmark_return(benchmark_df, df.index[entry_idx], df.index[exit_idx])
        if b_ret is not None:
            bench_returns.append(b_ret)
            rel_returns.append(ret - b_ret)
        in_t = False
        ep, entry_idx = 0.0, -1
        stop_px = target_px = None

    for i in range(1, len(df)):
        prev_sig = int(signals.iloc[i - 1]) if pd.notna(signals.iloc[i - 1]) else 0

        if in_t:
            if prev_sig == -1:
                close_trade(float(opens[i]), i, "exit_signal")
                continue
            if stop_px is not None and lows[i] <= stop_px:
                close_trade(float(stop_px), i, "sl")
                continue
            if target_px is not None and highs[i] >= target_px:
                close_trade(float(target_px), i, "target")
                continue
            if i >= entry_idx + max_hold:
                close_trade(float(closes[i]), i, "timeout")
                continue

        if not in_t and prev_sig == 1:
            ep = float(opens[i])
            if ep <= 0:
                continue
            lv = dynamic_trade_levels(df, i - 1, ep, p)
            stop_px   = _finite_float(lv.get("stop_loss"))
            target_px = _finite_float(lv.get("target"))
            entry_idx, in_t = i, True
            if stop_px is not None and lows[i] <= stop_px:
                close_trade(float(stop_px), i, "sl")
                continue
            if target_px is not None and highs[i] >= target_px:
                close_trade(float(target_px), i, "target")
                continue

    if not trades:
        return dict(
            win_rate=0, avg_return=0, median_return=0, trades=0,
            profit_factor=0, max_drawdown=0,
            sl_exits=0, target_exits=0, timeout_exits=0, exit_signal_exits=0,
            trade_returns=[], benchmark_return_pct=None, relative_return_pct=None,
            benchmark_outperformance_rate=None, expectancy=0.0,
        )

    wins   = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    gp, gl = sum(wins), abs(sum(losses))
    pf     = round(gp / gl, 2) if gl > 0 else 99.0
    eq     = np.cumsum(trades)
    peak   = np.maximum.accumulate(eq)
    max_dd = round(float(abs((eq - peak).min())), 2)
    outperf = [1 for r in rel_returns if r > 0]

    wr      = len(wins) / len(trades)
    avg_w   = float(np.mean(wins))   if wins   else 0.0
    avg_l   = float(np.mean([abs(x) for x in losses])) if losses else 0.0
    expect  = round(wr * avg_w - (1 - wr) * avg_l, 3)

    return dict(
        win_rate      = round(wr * 100, 1),
        avg_return    = round(float(np.mean(trades)), 2),
        median_return = round(float(np.median(trades)), 2),
        trades        = len(trades),
        profit_factor = min(float(pf), 99.0),
        max_drawdown  = max_dd,
        sl_exits          = reasons.count("sl"),
        target_exits     = reasons.count("target"),
        timeout_exits     = reasons.count("timeout"),
        exit_signal_exits = reasons.count("exit_signal"),
        trade_returns = [round(t, 3) for t in trades],
        benchmark_return_pct = round(float(np.mean(bench_returns)), 2) if bench_returns else None,
        relative_return_pct  = round(float(np.mean(rel_returns)),    2) if rel_returns   else None,
        benchmark_outperformance_rate = (round(len(outperf) / len(rel_returns) * 100, 1)
                                         if rel_returns else None),
        expectancy = expect,
    )

# ─────────────────────────────────────────────
#  JSON SANITISER
# ─────────────────────────────────────────────
def sanitize_for_json(obj):
    if isinstance(obj, dict):   return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):  return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.generic): obj = obj.item()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj): return None
        return obj
    return obj

# ─────────────────────────────────────────────
#  FUNDAMENTALS  (informational; not a hard filter)
# ─────────────────────────────────────────────
def fetch_fundamentals(ticker: str) -> dict:
    _default = {
        "company_name":      NSE_COMPANY_NAMES.get(ticker, ticker),
        "pe_ratio":          None,
        "debt_equity":       None,
        "revenue_growth":    None,
        "sector":            None,
        "market_cap_cr":     None,
        "roe":               None,
        "fundamental_score": 50,
        "fundamental_flag":  "DATA_UNAVAILABLE",
    }
    try:
        info = yf.Ticker(ticker + ".NS").info
        if not info or len(info) < 5:
            return _default

        pe   = info.get("trailingPE") or info.get("forwardPE")
        de   = info.get("debtToEquity")
        rg   = info.get("revenueGrowth")
        name = (info.get("shortName") or info.get("longName")
                or NSE_COMPANY_NAMES.get(ticker, ticker))

        flags = []
        if pe is not None and math.isfinite(float(pe)) and float(pe) > FUND_HIGH_PE_THRESHOLD:
            flags.append("HIGH_PE")
        if rg is not None and math.isfinite(float(rg)) and float(rg) < FUND_LOW_REV_THRESHOLD:
            flags.append("LOW_REVENUE")
        if de is not None and math.isfinite(float(de)) and float(de) > FUND_HIGH_DE_THRESHOLD:
            flags.append("HIGH_DEBT")

        def _safe(v):
            if v is None: return None
            try:
                fv = float(v)
                return None if not math.isfinite(fv) else fv
            except Exception:
                return None

        mc_raw = info.get("marketCap") or 0
        mc_cr  = round(mc_raw / 1e7, 0) if mc_raw > 0 else None

        roe_raw = info.get("returnOnEquity")
        roe_val = (round(float(roe_raw) * 100, 1)
                   if roe_raw is not None and math.isfinite(float(roe_raw))
                   else None)

        sector_val = info.get("sector") or None

        f_score = 50
        pe_v = _safe(pe)
        if pe_v is not None:
            if 5 < pe_v < 30:   f_score += 10
            elif pe_v > 80:     f_score -= 10
            elif pe_v < 0:      f_score -= 15
        if roe_val is not None:
            if roe_val > 15:    f_score += 10
            elif roe_val < 0:   f_score -= 10
        rg_v = _safe(rg)
        if rg_v is not None:
            if rg_v > 0.10:     f_score += 10
            elif rg_v < -0.10:  f_score -= 15
        de_v = _safe(de)
        if de_v is not None:
            de_adj = de_v / 100 if de_v > 20 else de_v
            if de_adj < 0.5:    f_score += 10
            elif de_adj > 3:    f_score -= 10
        if mc_cr and mc_cr > 20000:  f_score += 5
        elif mc_cr and mc_cr < 500:  f_score -= 10
        f_score = max(0, min(100, f_score))

        return {
            "company_name":      name,
            "pe_ratio":          round(pe_v, 1)       if pe_v  is not None else None,
            "debt_equity":       round(de_v, 1)       if de_v  is not None else None,
            "revenue_growth":    round(rg_v * 100, 1) if rg_v  is not None else None,
            "sector":            sector_val,
            "market_cap_cr":     mc_cr,
            "roe":               roe_val,
            "fundamental_score": f_score,
            "fundamental_flag":  ", ".join(flags) if flags else "OK",
        }
    except Exception:
        return _default

# ─────────────────────────────────────────────
#  SCORE MULTIPLIERS
# ─────────────────────────────────────────────
def _fundamental_multiplier_from_score(score) -> float:
    fs = _finite_float(score, 50.0)
    if fs >= 80: return 1.06
    if fs >= 65: return 1.03
    if fs <= 25: return 0.94
    if fs <= 40: return 0.97
    return 1.00


def apply_score_multipliers(technical_score, fundamental_score,
                            news_multiplier) -> tuple[float, float, float]:
    fund_mult  = _fundamental_multiplier_from_score(fundamental_score)
    news_mult  = _finite_float(news_multiplier, 1.0)
    final_mult = max(0.88, min(1.12, fund_mult * news_mult))
    final      = max(0.0, min(100.0, float(technical_score or 0) * final_mult))
    return round(final, 1), round(final_mult, 3), round(fund_mult, 3)


def get_signal_streaks(today: str) -> dict[str, tuple[str, int]]:
    streaks: dict[str, tuple[str, int]] = {}
    try:
        start = (datetime.today() - timedelta(days=14)).strftime("%Y-%m-%d")
        res = (
            supabase.table("recommendations")
            .select("date, ticker, action")
            .gte("date", start)
            .lt("date", today)
            .order("date", desc=True)
            .execute()
        )
        if not res.data:
            return streaks
        df = (
            pd.DataFrame(res.data)
            .drop_duplicates(subset=["date", "ticker"])
            .sort_values(["ticker", "date"], ascending=[True, False])
        )
        for ticker, grp in df.groupby("ticker"):
            if grp.empty:
                continue
            last_action = "EXIT" if grp.iloc[0]["action"] == "SELL" else grp.iloc[0]["action"]
            streak = 0
            for _, row in grp.iterrows():
                row_action = "EXIT" if row["action"] == "SELL" else row["action"]
                if row_action == last_action:
                    streak += 1
                else:
                    break
            streaks[ticker] = (last_action, streak)
    except Exception as e:
        print(f"\n  ⚠️  Signal streak fetch failed: {e}")
    return streaks

# ─────────────────────────────────────────────
#  MARKET BREADTH
# ─────────────────────────────────────────────
def compute_market_breadth(records: list[dict]) -> dict:
    buy_ct  = sum(1 for r in records if r.get("action") == "BUY")
    exit_ct = sum(1 for r in records if r.get("action") in ("EXIT", "SELL"))
    total   = buy_ct + exit_ct
    if total == 0:
        return {"buy_count": 0, "exit_count": 0, "sell_count": 0,
                "breadth_ratio": 0.5, "breadth_label": "NEUTRAL"}
    ratio = buy_ct / total
    if   ratio >= 0.70: label = "VERY BULLISH"
    elif ratio >= 0.55: label = "BULLISH"
    elif ratio >= 0.45: label = "NEUTRAL"
    elif ratio >= 0.30: label = "BEARISH"
    else:               label = "VERY BEARISH"
    return {
        "buy_count":     buy_ct,
        "exit_count":    exit_ct,
        "sell_count":    exit_ct,
        "breadth_ratio": round(ratio, 3),
        "breadth_label": label,
    }

# ─────────────────────────────────────────────
#  TELEGRAM
# ─────────────────────────────────────────────
def _esc_html(s) -> str:
    return (str(s or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def _build_telegram_message(records, regime, breadth, today,
                             exit_holdings_alerts=None) -> str:
    regime_e = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(regime, "⬜")
    news_e   = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "🟡"}

    lines = [
        f"🇮🇳 <b>Indian Stock Agent — {today}</b>",
        f"Market Regime  : {regime_e} {regime}",
        f"Market Breadth : {breadth.get('breadth_label','?')} "
        f"({breadth.get('buy_count',0)} buys / {breadth.get('exit_count', breadth.get('sell_count',0))} exits)",
        "",
    ]

    # v8: EXIT signals on stocks the user already holds (REVIEW A3).
    if exit_holdings_alerts:
        lines.append("⚠️  <b>EXIT SIGNAL ON HOLDINGS</b>")
        for r in exit_holdings_alerts:
            lines.append(
                f"• <b>{_esc_html(r['ticker'])}</b> — "
                f"composite {r.get('composite_score', 0):.0f}/100"
            )
        lines.append("Review these positions on the dashboard.")
        lines.append("")

    buys  = sorted([r for r in records if r.get("action") == "BUY"],
                   key=lambda x: x.get("composite_score", 0), reverse=True)[:5]
    exits = sorted([r for r in records if r.get("action") in ("EXIT", "SELL")],
                   key=lambda x: x.get("composite_score", 0), reverse=True)[:3]

    if buys:
        lines.append("🟢 <b>BUY SIGNALS</b>")
        for idx, r in enumerate(buys, 1):
            streak = r.get("signal_streak", 1) or 1
            streak_str = f" 🔥{streak}d" if streak >= 2 else ""
            ns   = r.get("news_sentiment")
            ne   = news_e.get(ns, "") if ns else ""
            flag = " ⚠️ Bad news!" if r.get("news_alert") else ""

            price = float(r.get("price") or 0)
            sl    = float(r.get("stop_loss") or 0)
            tgt   = float(r.get("target") or 0)
            sl_pct  = ((sl - price) / price * 100) if price > 0 and sl > 0 else 0.0
            tgt_pct = ((tgt - price) / price * 100) if price > 0 and tgt > 0 else 0.0
            active_str = r.get("active_strategies") or ""
            sug_qty = r.get("suggested_qty")

            lines.append(
                f"\n{idx}. <b>{_esc_html(r['ticker'])}</b>{streak_str} — "
                f"{r.get('composite_score', 0):.0f}/100 ({_esc_html(r.get('score_label',''))})"
            )
            lines.append(
                f"   ₹{price:,.2f} | "
                f"SL ₹{sl:,.2f} ({sl_pct:+.2f}%) | "
                f"Target ₹{tgt:,.2f} ({tgt_pct:+.2f}%)"
            )
            if sug_qty and sug_qty > 0:
                lines.append(f"   Suggested qty (₹{int(DEFAULT_ACCOUNT_INR):,} acct, "
                             f"{DEFAULT_RISK_PCT:.1f}% risk): <b>{sug_qty}</b>")
            if active_str:
                lines.append(f"   {_esc_html(active_str)}")
            if ne:
                headline = r.get("news_headline") or ""
                short_hl = (headline[:75] + "…") if len(headline) > 75 else headline
                lines.append(f"   News: {ne} {_esc_html(ns)}{flag}")
                if short_hl:
                    lines.append(f"   📰 {_esc_html(short_hl)}")
            fund_flag = r.get("fundamental_flag") or ""
            if fund_flag and fund_flag not in ("OK", "DATA_UNAVAILABLE", ""):
                lines.append(f"   ⚠️ Fundamentals: {_esc_html(fund_flag)}")

    if exits:
        lines.append("\n🔴 <b>EXIT / AVOID-NEW-LONG SIGNALS</b>")
        for idx, r in enumerate(exits, 1):
            lines.append(
                f"\n{idx}. <b>{_esc_html(r['ticker'])}</b> — "
                f"{r.get('composite_score', 0):.0f}/100 | ₹{r.get('price', 0) or 0:,.2f}"
            )
            active_str = r.get("active_strategies") or ""
            if active_str:
                lines.append(f"   {_esc_html(active_str)}")
            lines.append("   Not a short-sell recommendation.")

    lines.append(f"\n📊 {len(records)} total signals | {today}")
    lines.append("⚠️ Paper trading only. Not financial advice.")
    return "\n".join(lines)


def send_telegram_alert(bot_token: str, chat_id: str, message: str) -> bool:
    if not bot_token or not chat_id:
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=15,
        )
        if resp.status_code == 200:
            print("  📱 Telegram alert sent ✅")
            return True
        print(f"  ⚠️  Telegram error {resp.status_code}: {resp.text[:100]}")
        return False
    except Exception as e:
        print(f"  ⚠️  Telegram failed: {e}")
        return False

# ─────────────────────────────────────────────
#  WEIGHTED VOTE + COMPOSITE SCORE
# ─────────────────────────────────────────────
def weighted_vote(today_sigs: dict, bt_results: dict, action: str) -> tuple[float, dict]:
    """REVIEW B5 fix: confidence floor at 5 trades, full conf at 30."""
    weights: dict[str, float] = {}
    is_buy = action == "BUY"
    for name, bt in bt_results.items():
        n = int(bt.get("trades", 0) or 0)
        if n < 5:
            conf = 0.0
        else:
            conf = min((n - 5) / 25.0, 1.0)
        if is_buy:
            wr = float(bt.get("win_rate", 50)) / 100.0
            weights[name] = round(wr * conf, 3)
        else:
            weights[name] = round(max(conf, 0.25), 3)

    total_w  = sum(weights.values()) or 1e-9
    target_v = 1 if is_buy else -1
    signal_w = sum(weights.get(name, 0) for name, v in today_sigs.items() if v == target_v)
    return round(signal_w / total_w, 3), weights


def composite_score(today_sigs, bt_results, ctx, regime_sc, action, p) -> tuple[float, dict]:
    w_ratio, _ = weighted_vote(today_sigs, bt_results, action)
    strat_pts  = round(w_ratio * p["W_STRATEGY"], 2)

    r = ctx.get("rsi")
    if r is None:
        rsi_pts = 0.0
    elif action == "BUY":
        rsi_pts = max(0.0, min((60 - r) / 40 * p["W_RSI"], p["W_RSI"]))
    else:
        rsi_pts = max(0.0, min((r - 40) / 40 * p["W_RSI"], p["W_RSI"]))

    avg_vol   = ctx.get("avg_volume") or 0
    cur_vol   = ctx.get("volume") or 0
    vol_ratio = (cur_vol / avg_vol) if avg_vol > 0 else 1.0
    vol_pts   = min(vol_ratio / 2.0 * p["W_VOLUME"], p["W_VOLUME"])

    rr     = _finite_float(ctx.get("rr_ratio"), 0.0) or 0.0
    min_rr = float(p.get("MIN_RR_RATIO", 1.5))
    rr_good     = max(min_rr * 2.0, 3.0)
    rr_exit_cap = max(min_rr * 1.33, 2.0)
    if action == "BUY":
        rr_pts = max(0.0, min(rr / rr_good * p["W_RR"], p["W_RR"]))
    else:
        rr_pts = max(0.0, min((rr_exit_cap - rr) / rr_exit_cap * p["W_RR"], p["W_RR"]))

    reg_pts = (regime_sc * p["W_REGIME"]
               if action == "BUY"
               else (1 - regime_sc) * p["W_REGIME"])

    total = round(strat_pts + rsi_pts + vol_pts + rr_pts + reg_pts, 1)
    breakdown = dict(
        strategy = round(strat_pts, 1),
        rsi      = round(rsi_pts, 1),
        volume   = round(vol_pts, 1),
        rr       = round(rr_pts, 1),
        regime   = round(reg_pts, 1),
        rr_ratio = round(rr, 2),
    )
    return min(total, 100.0), breakdown


def score_label(s: float) -> str:
    if s >= 80: return "Very Strong"
    if s >= 65: return "Strong"
    if s >= 50: return "Good"
    if s >= 35: return "Moderate"
    return "Weak"

# ─────────────────────────────────────────────
#  DATA FETCH
# ─────────────────────────────────────────────
def fetch(ticker: str, days: int = 430) -> tuple:
    try:
        df = yf.download(
            ticker + ".NS",
            start=datetime.today() - timedelta(days=days),
            end=datetime.today(),
            progress=False, auto_adjust=True,
        )
        if df.empty or len(df) < 80:
            return None, "insufficient_data"
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna(), "ok"
    except Exception as e:
        return None, str(e)[:120]


def fetch_benchmark(days: int = 430) -> pd.DataFrame:
    try:
        df = yf.download(
            "^NSEI",
            start=datetime.today() - timedelta(days=days),
            end=datetime.today(),
            progress=False, auto_adjust=True,
        )
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df[["Open", "High", "Low", "Close"]].dropna()
    except Exception:
        return pd.DataFrame()

# ─────────────────────────────────────────────
#  PRICE CONTEXT
# ─────────────────────────────────────────────
def context(df: pd.DataFrame, p: dict) -> dict:
    c = df.Close
    if len(c) < max(60, int(p.get("EMA_LONG", 21)) * 2):
        return sanitize_for_json({
            "price": None, "change_1d": None, "change_5d": None, "change_30d": None,
            "rsi": None, "macd_hist": None, "ema_bullish": None,
            "support": None, "resistance": None,
            "stop_loss": None, "target": None,
            "risk_pct": None, "reward_pct": None, "rr_ratio": None,
            "volume": 0, "avg_volume": 0,
            "atr_pct": None, "pct_from_52w_high": None, "pct_from_52w_low": None,
        })

    r        = float(rsi(c, p["RSI_PERIOD"]).iloc[-1])
    _, _, h  = macd(c, p["MACD_FAST"], p["MACD_SLOW"], p["MACD_SIGNAL"])
    e_s      = float(ema(c, p["EMA_SHORT"]).iloc[-1])
    e_l      = float(ema(c, p["EMA_LONG"]).iloc[-1])

    price   = float(c.iloc[-1])
    atr_now = float(atr(df.High, df.Low, c, p["ATR_PERIOD"]).iloc[-1])

    prev_1  = float(c.iloc[-2])  if len(c) >= 2  and pd.notna(c.iloc[-2])  else None
    prev_5  = float(c.iloc[-6])  if len(c) >= 6  and pd.notna(c.iloc[-6])  else None
    prev_30 = float(c.iloc[-31]) if len(c) >= 31 and pd.notna(c.iloc[-31]) else None

    lookback         = min(252, len(c))
    high_52w         = float(c.iloc[-lookback:].max())
    low_52w          = float(c.iloc[-lookback:].min())
    pct_from_52w_high = (round((price - high_52w) / high_52w * 100, 2)
                         if high_52w > 0 and math.isfinite(high_52w) else None)
    pct_from_52w_low  = (round((price - low_52w)  / low_52w  * 100, 2)
                         if low_52w > 0 and math.isfinite(low_52w)  else None)
    atr_pct           = (round(atr_now / price * 100, 3)
                         if price > 0 and math.isfinite(atr_now) else None)

    levels  = dynamic_trade_levels(df, len(df) - 1, price, p)
    sl, tgt = levels.get("stop_loss"), levels.get("target")

    low20  = levels.get("support")
    high20 = levels.get("resistance")
    vol20  = df.Volume.rolling(20).mean().iloc[-1]

    out = dict(
        price      = round(price, 2) if math.isfinite(price) else None,
        change_1d  = round((price - prev_1)  / prev_1  * 100, 2)
                     if prev_1  not in (None, 0) and math.isfinite(prev_1)  else None,
        change_5d  = round((price - prev_5)  / prev_5  * 100, 2)
                     if prev_5  not in (None, 0) and math.isfinite(prev_5)  else None,
        change_30d = round((price - prev_30) / prev_30 * 100, 2)
                     if prev_30 not in (None, 0) and math.isfinite(prev_30) else None,
        rsi        = round(r, 1) if math.isfinite(r) else None,
        macd_hist  = (round(float(h.iloc[-1]), 3)
                      if pd.notna(h.iloc[-1]) and math.isfinite(float(h.iloc[-1])) else None),
        ema_bullish = bool(e_s > e_l) if math.isfinite(e_s) and math.isfinite(e_l) else None,
        support    = round(float(low20),  2) if low20  is not None and math.isfinite(float(low20))  else None,
        resistance = round(float(high20), 2) if high20 is not None and math.isfinite(float(high20)) else None,
        stop_loss  = sl,
        target     = tgt,
        risk_pct   = levels.get("risk_pct"),
        reward_pct = levels.get("reward_pct"),
        rr_ratio   = levels.get("rr_ratio"),
        rr_floor_applied = bool(levels.get("rr_floor_applied", False)),
        volume     = int(df.Volume.iloc[-1]) if pd.notna(df.Volume.iloc[-1]) else 0,
        avg_volume = int(vol20) if pd.notna(vol20) and math.isfinite(float(vol20)) else 0,
        atr_pct           = atr_pct,
        pct_from_52w_high = pct_from_52w_high,
        pct_from_52w_low  = pct_from_52w_low,
    )
    return sanitize_for_json(out)

# ─────────────────────────────────────────────
#  MODEL FEATURE PACKING (Phase 3)
# ─────────────────────────────────────────────
_REGIME_SCORE_MAP = {"BULLISH": 1.0, "NEUTRAL": 0.5, "BEARISH": 0.0}


def momentum_vs_nifty_30d(df, benchmark_df, days: int = 30):
    if benchmark_df is None or benchmark_df.empty or len(df) < days + 1:
        return None
    try:
        stock_ret = (df.Close.iloc[-1] / df.Close.iloc[-(days + 1)] - 1) * 100
        bench = benchmark_df[benchmark_df.index <= df.index[-1]]
        if len(bench) < days + 1:
            return None
        bench_ret = (bench.Close.iloc[-1] / bench.Close.iloc[-(days + 1)] - 1) * 100
        return round(float(stock_ret - bench_ret), 2)
    except Exception:
        return None


_VIX_NONE = {"vix_level": None, "vix_change_5d": None, "vix_zscore_60d": None}


def vix_features_from_series(close_series: pd.Series) -> dict:
    """Pure VIX feature computation. Used by both the live agent
    (passing the last 6 months of ^INDIAVIX) and the synthetic
    generator (passing all VIX history sliced to the as-of date).
    Single source of truth — eliminates the v7 risk that the two
    code paths silently diverge.
    """
    if close_series is None or len(close_series) == 0:
        return dict(_VIX_NONE)
    series = close_series.dropna()
    if series.empty:
        return dict(_VIX_NONE)
    level = float(series.iloc[-1])
    if not math.isfinite(level):
        return dict(_VIX_NONE)
    change_5d = None
    if len(series) >= 6:
        prev = float(series.iloc[-6])
        if math.isfinite(prev):
            change_5d = round(level - prev, 3)
    zscore_60d = None
    if len(series) >= 60:
        window = series.iloc[-60:]
        mu = float(window.mean())
        sd = float(window.std(ddof=0))
        if sd > 1e-9 and math.isfinite(mu) and math.isfinite(sd):
            zscore_60d = round((level - mu) / sd, 3)
    return {"vix_level":      round(level, 3),
            "vix_change_5d":  change_5d,
            "vix_zscore_60d": zscore_60d}


def fetch_vix_features_today() -> dict:
    """Live-agent wrapper: fetch ^INDIAVIX last 6 months and compute features."""
    try:
        df = yf.download("^INDIAVIX", period="6mo",
                         progress=False, auto_adjust=True, threads=False)
        if df is None or df.empty:
            return dict(_VIX_NONE)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        if "Close" not in df.columns:
            return dict(_VIX_NONE)
        return vix_features_from_series(df["Close"].dropna())
    except Exception as e:
        print(f"  ⚠️  VIX fetch failed: {e}")
        return dict(_VIX_NONE)


def signal_features_for_model(ctx, today_sigs, regime_label,
                              mom_vs_nifty_30d, vix_features=None) -> dict:
    has_donchian   = int(today_sigs.get("Donchian", 0) ==  1)
    has_ema        = int(today_sigs.get("EMA Crossover", 0) ==  1)
    has_rsi_trend  = int(today_sigs.get("RSI Trend Shift", 0) ==  1)
    has_bollinger  = int(today_sigs.get("Bollinger", 0) ==  1)
    n_firing       = has_donchian + has_ema + has_rsi_trend + has_bollinger

    vol_now = ctx.get("volume", 0) or 0
    vol_avg = ctx.get("avg_volume", 0) or 0
    vol_ratio = round(vol_now / vol_avg, 3) if vol_avg > 0 else 1.0

    vix = vix_features or {}
    return {
        "rsi":               ctx.get("rsi") if ctx.get("rsi") is not None else 50,
        "macd_hist":         ctx.get("macd_hist") if ctx.get("macd_hist") is not None else 0,
        "atr_pct":           ctx.get("atr_pct") if ctx.get("atr_pct") is not None else 0,
        "change_1d":         ctx.get("change_1d") if ctx.get("change_1d") is not None else 0,
        "change_5d":         ctx.get("change_5d") if ctx.get("change_5d") is not None else 0,
        "change_30d":        ctx.get("change_30d") if ctx.get("change_30d") is not None else 0,
        "pct_from_52w_high": ctx.get("pct_from_52w_high") if ctx.get("pct_from_52w_high") is not None else 0,
        "pct_from_52w_low":  ctx.get("pct_from_52w_low")  if ctx.get("pct_from_52w_low")  is not None else 0,
        "vol_ratio":         vol_ratio,
        "n_firing":          n_firing,
        "risk_pct":          ctx.get("risk_pct") if ctx.get("risk_pct") is not None else 0,
        "reward_pct":        ctx.get("reward_pct") if ctx.get("reward_pct") is not None else 0,
        "rr_ratio":          ctx.get("rr_ratio") if ctx.get("rr_ratio") is not None else 0,
        "regime_score":      _REGIME_SCORE_MAP.get(regime_label, 0.5),
        "mom_vs_nifty_30d":  mom_vs_nifty_30d if mom_vs_nifty_30d is not None else 0,
        "vix_level":         vix.get("vix_level")      if vix.get("vix_level")      is not None else 0,
        "vix_change_5d":     vix.get("vix_change_5d")  if vix.get("vix_change_5d")  is not None else 0,
        "vix_zscore_60d":    vix.get("vix_zscore_60d") if vix.get("vix_zscore_60d") is not None else 0,
        "ema_bullish":       int(bool(ctx.get("ema_bullish"))),
        "has_donchian":      has_donchian,
        "has_ema":           has_ema,
        "has_rsi_trend":     has_rsi_trend,
        "has_bollinger":     has_bollinger,
        "single_strat":      int(n_firing == 1),
        "multi_strat":       int(n_firing >= 2),
    }

# ─────────────────────────────────────────────
#  MARKET REGIME
#
#  This is the SINGLE SOURCE OF TRUTH for regime classification.
#  Both the live agent and the synthetic-data generator
#  (agent/build_training_data.py) must call regime_from_close_series()
#  so labels match exactly. Splitting fetch from logic prevents the
#  silent feature mismatch that v7 had — production used EMA20/50+RSI,
#  but the synthetic generator was using SMA50/200, so the model's
#  `regime_score` weight was learned on a feature it would never see.
# ─────────────────────────────────────────────
def regime_from_close_series(close: pd.Series, p: dict | None = None
                             ) -> tuple[str, float]:
    """Classify regime from a Nifty (or other index) close-price series.

    Pure function. No I/O. Uses ONLY data ≤ the last index of `close` —
    safe to call point-in-time during synthetic-data generation by passing
    the slice up to (and including) the as-of date.

    Returns (label, score) where score is in {0.0, 0.5, 1.0}.
    """
    if p is None:
        p = DEFAULT_PARAMS
    if close is None or len(close) < 55:
        return "UNKNOWN", 0.5
    try:
        c = close.dropna()
        if len(c) < 55:
            return "UNKNOWN", 0.5
        e50   = float(ema(c, 50).iloc[-1])
        e20   = float(ema(c, 20).iloc[-1])
        r     = float(rsi(c, int(p.get("RSI_PERIOD", 14))).iloc[-1])
        price = float(c.iloc[-1])
        if not all(math.isfinite(x) for x in (e50, e20, r, price)):
            return "UNKNOWN", 0.5
        if price > e20 > e50 and r > 50: return "BULLISH", 1.0
        if price < e20 < e50 and r < 50: return "BEARISH", 0.0
        return "NEUTRAL", 0.5
    except Exception:
        return "UNKNOWN", 0.5


def market_regime(p: dict) -> tuple[str, float]:
    """Live-agent wrapper: fetch Nifty's last 120 days and classify."""
    try:
        df = yf.download("^NSEI", period="120d",
                         progress=False, auto_adjust=True)
        if df.empty:
            return "UNKNOWN", 0.5
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        if "Close" not in df.columns:
            return "UNKNOWN", 0.5
        return regime_from_close_series(df["Close"].squeeze(), p)
    except Exception:
        return "UNKNOWN", 0.5

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run():
    if supabase is None:
        raise RuntimeError(
            "Supabase client not initialized. SUPABASE_URL and SUPABASE_KEY "
            "must be set to valid values before running the agent."
        )
    today = datetime.today().strftime("%Y-%m-%d")
    print(f"\n🇮🇳 Indian Stock Agent v8 — {today}")

    P, param_version = load_active_params()
    hf_token  = os.environ.get("HF_TOKEN", "")
    tg_token  = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat   = os.environ.get("TELEGRAM_CHAT_ID", "")

    if hf_token:
        print("  ℹ️  HF_TOKEN set — using FinBERT for news sentiment")
    else:
        print("  ℹ️  HF_TOKEN not set — falling back to VADER + financial overrides")
    if not tg_token:
        print("  ℹ️  TELEGRAM_BOT_TOKEN not set — Telegram alerts disabled")

    print(f"   Params       : {param_version}")
    print(f"   Stocks       : {len(ALL_TICKERS)} NSE tickers")
    print(f"   Account base : ₹{int(DEFAULT_ACCOUNT_INR):,}  |  Risk/trade: {DEFAULT_RISK_PCT:.1f}%\n")

    STRATEGIES = get_strategies(P)

    print("   Checking NIFTY market regime...")
    regime_label, regime_sc = market_regime(P)
    print(f"   Regime : {regime_label}  (score={regime_sc})")

    print("   Loading NIFTY benchmark (^NSEI)...")
    benchmark_df = fetch_benchmark()
    print(f"   Benchmark bars: {len(benchmark_df)}")

    print("   Fetching India VIX (^INDIAVIX)...")
    vix_features = fetch_vix_features_today()
    if vix_features.get("vix_level") is not None:
        print(f"   VIX level: {vix_features['vix_level']}  "
              f"5d Δ: {vix_features.get('vix_change_5d')}  "
              f"60d z: {vix_features.get('vix_zscore_60d')}")
    else:
        print("   ⚠️  VIX data unavailable — features will fall back to 0")

    print("   Loading signal streak history...")
    streaks = get_signal_streaks(today)
    print(f"   {len(streaks)} tickers have recent signal history")

    held_tickers      = open_tickers(supabase)
    open_pos_w_sector = open_positions_with_sector(supabase)
    recent_pnls       = recent_closed_pnl_pcts(supabase, n=20)
    print(f"   Open positions: {len(held_tickers)}  |  Recent closed: {len(recent_pnls)}\n")

    # Idempotent re-run safety
    supabase.table("recommendations").delete().eq("date", today).execute()
    supabase.table("ticker_run_log").delete().eq("date", today).execute()

    records, run_logs = [], []
    gate_counts = {"fetched": 0, "any_signal": 0, "passed_weight": 0,
                   "rejected_multi_strat": 0, "rejected_bearish_regime": 0}
    exit_holdings_alerts: list[dict] = []

    for i, ticker in enumerate(ALL_TICKERS, 1):
        sys.stdout.write(f"\r  {i}/{len(ALL_TICKERS)}  {ticker:<15}")
        sys.stdout.flush()

        df, status = fetch(ticker)
        run_logs.append(sanitize_for_json({"date": today, "ticker": ticker, "status": status}))
        if df is None:
            continue
        gate_counts["fetched"] += 1

        try:
            # REVIEW G1 fix: compute strategy series ONCE; reuse for both
            # today_sigs and the per-strategy backtest.
            strategy_series = {name: fn(df) for name, fn in STRATEGIES.items()}
            today_sigs = {name: int(s.iloc[-1]) for name, s in strategy_series.items()}
            buy_count  = sum(1 for v in today_sigs.values() if v ==  1)
            sell_count = sum(1 for v in today_sigs.values() if v == -1)

            if buy_count == 0 and sell_count == 0:
                continue
            gate_counts["any_signal"] += 1

            action = "BUY" if buy_count >= sell_count else "EXIT"

            # ─── v7 hard gates ───
            if action == "BUY" and buy_count > 1:
                gate_counts["rejected_multi_strat"] += 1
                continue
            if action == "BUY" and regime_label == "BEARISH":
                gate_counts["rejected_bearish_regime"] += 1
                continue

            bt = {name: backtest(df, s, P, benchmark_df)
                  for name, s in strategy_series.items()}
            w_ratio, weights = weighted_vote(today_sigs, bt, action)
            if w_ratio < P["MIN_WEIGHTED_SCORE"]:
                continue
            gate_counts["passed_weight"] += 1

            ctx = context(df, P)
            technical_score, c_breakdown = composite_score(
                today_sigs, bt, ctx, regime_sc, action, P
            )

            fund = fetch_fundamentals(ticker)
            company_name = fund.get("company_name") or NSE_COMPANY_NAMES.get(ticker, ticker)

            if technical_score >= 25:
                news = fetch_news_sentiment(ticker, company_name, hf_token)
                news["news_alert"] = (
                    (action == "BUY"  and news.get("news_sentiment") == "NEGATIVE") or
                    (action == "EXIT" and news.get("news_sentiment") == "POSITIVE")
                )
            else:
                news = {
                    "news_score": 0.0, "news_sentiment": "NEUTRAL",
                    "news_headline": None, "news_headlines": [],
                    "news_count": 0, "news_multiplier": 1.0,
                    "news_source": "disabled", "news_alert": False,
                }

            final_score, final_multiplier, fundamental_multiplier = apply_score_multipliers(
                technical_score, fund.get("fundamental_score", 50),
                news.get("news_multiplier", 1.0)
            )
            c_breakdown.update({
                "technical_score":        round(float(technical_score or 0), 1),
                "final_score":            final_score,
                "news_multiplier":        float(news.get("news_multiplier", 1.0)),
                "fundamental_multiplier": fundamental_multiplier,
                "final_score_multiplier": final_multiplier,
            })

            # Phase 3 P(win)
            p_win_score = None
            if _SCORE_MODEL is not None and action == "BUY":
                pipeline, feats, _label_kind = _SCORE_MODEL
                mom = momentum_vs_nifty_30d(df, benchmark_df)
                sig_feats = signal_features_for_model(
                    ctx, today_sigs, regime_label, mom, vix_features
                )
                try:
                    p_win_score = round(
                        predict_p_win(pipeline, feats, sig_feats, label_kind=_label_kind), 4
                    )
                except Exception as e:
                    print(f"\n  ⚠️  predict_p_win failed for {ticker}: {e}")
            c_breakdown["p_win"] = p_win_score

            prev_action, prev_streak = streaks.get(ticker, (None, 0))
            streak_today = (prev_streak + 1) if prev_action == action else 1

            active = [
                n for n, v in today_sigs.items()
                if (v == 1 and action == "BUY") or (v == -1 and action == "EXIT")
            ]
            def agg(k):
                vals = [bt[n].get(k) for n in active if bt[n].get(k) is not None]
                return round(float(np.mean(vals)), 2) if vals and action == "BUY" else None
            trade_vals = [bt[n].get("trades", 0) for n in active]
            avg_trade_count = int(round(np.mean(trade_vals))) if trade_vals and action == "BUY" else 0
            low_smp = (avg_trade_count < 5) if action == "BUY" else False

            # v8 — risk-aware suggested qty for BUY
            sized = {"qty": 0, "blocked": False, "reasons": []}
            if action == "BUY" and ctx.get("price") and ctx.get("stop_loss"):
                sized = recommend_position_size(
                    capital_inr=DEFAULT_ACCOUNT_INR,
                    risk_pct=DEFAULT_RISK_PCT,
                    entry_price=float(ctx["price"]),
                    stop_loss=float(ctx["stop_loss"]),
                    regime_label=regime_label,
                    recent_pnl_pcts=recent_pnls,
                    open_positions=open_pos_w_sector,
                    sector=fund.get("sector"),
                )

            # v8 — EXIT-vs-portfolio guard (REVIEW A3)
            if action == "EXIT" and ticker in held_tickers:
                exit_holdings_alerts.append({
                    "ticker": ticker, "composite_score": float(final_score or 0),
                })

            record = dict(
                date              = today,
                ticker            = ticker,
                action            = action,
                raw_score         = int(buy_count if action == "BUY" else sell_count),
                weighted_score_val= float(w_ratio),
                technical_score   = float(technical_score) if technical_score is not None else 0.0,
                composite_score   = float(final_score)     if final_score     is not None else 0.0,
                final_score_multiplier = float(final_multiplier),
                fundamental_multiplier = float(fundamental_multiplier),
                score_label       = score_label(final_score if final_score is not None else 0),
                score_breakdown   = json.dumps(sanitize_for_json(c_breakdown)),
                signals           = json.dumps(sanitize_for_json(today_sigs)),
                strategy_weights  = json.dumps(sanitize_for_json(weights)),
                backtest          = json.dumps(sanitize_for_json(bt)),
                active_strategies = ", ".join(active),
                low_sample_warning= bool(low_smp),
                win_rate          = agg("win_rate"),
                avg_return        = agg("avg_return"),
                median_return     = agg("median_return"),
                profit_factor     = agg("profit_factor"),
                max_drawdown      = agg("max_drawdown"),
                avg_trades        = avg_trade_count,
                benchmark_return_pct = agg("benchmark_return_pct"),
                relative_return_pct  = agg("relative_return_pct"),
                benchmark_outperformance_rate = agg("benchmark_outperformance_rate"),
                market_regime     = regime_label,
                param_version     = param_version,
                company_name      = fund.get("company_name"),
                pe_ratio          = fund.get("pe_ratio"),
                debt_equity       = fund.get("debt_equity"),
                revenue_growth    = fund.get("revenue_growth"),
                fundamental_flag  = fund.get("fundamental_flag"),
                de_ratio          = fund.get("debt_equity"),
                sector            = fund.get("sector"),
                market_cap_cr     = fund.get("market_cap_cr"),
                roe               = fund.get("roe"),
                fundamental_score = fund.get("fundamental_score", 50),
                fundamental_warnings = json.dumps(
                    [] if fund.get("fundamental_flag") in (None, "", "OK", "DATA_UNAVAILABLE")
                    else [x.strip() for x in str(fund.get("fundamental_flag", "")).split(',') if x.strip()]
                ),
                news_score        = news.get("news_score"),
                news_sentiment    = news.get("news_sentiment"),
                news_headline     = news.get("news_headline"),
                news_alert        = bool(news.get("news_alert", False)),
                news_label        = (news.get("news_sentiment") or "NEUTRAL").lower(),
                news_headlines    = json.dumps(sanitize_for_json(
                    news.get("news_headlines") or
                    ([news.get("news_headline")] if news.get("news_headline") else [])
                )),
                news_multiplier   = float(news.get("news_multiplier", 1.0)),
                news_count        = int(news.get("news_count",
                                        1 if news.get("news_headline") else 0)),
                signal_streak     = streak_today,
                p_win             = p_win_score,
                vix_level         = vix_features.get("vix_level"),
                vix_change_5d     = vix_features.get("vix_change_5d"),
                vix_zscore_60d    = vix_features.get("vix_zscore_60d"),
                # v8 fields
                suggested_qty     = sized.get("qty", 0),
                size_blocked      = bool(sized.get("blocked", False)),
                size_reasons      = json.dumps(sanitize_for_json(sized.get("reasons", []))),
                **ctx,
            )

            safe_record = sanitize_for_json(record)
            json.dumps(safe_record)
            records.append(safe_record)

        except Exception as e:
            print(f"\n  ⚠️  Skipping {ticker}: {e}")

        time.sleep(0.1)

    sys.stdout.write("\r" + " " * 60 + "\r")

    failed_count = sum(1 for l in run_logs if l["status"] != "ok")
    print("  Pipeline summary:")
    print(f"    Tickers scanned         : {len(ALL_TICKERS)}")
    print(f"    Data fetched OK         : {gate_counts['fetched']}")
    print(f"    Any signal fired        : {gate_counts['any_signal']}")
    print(f"    Rejected: multi-strat   : {gate_counts['rejected_multi_strat']}")
    print(f"    Rejected: bearish regime: {gate_counts['rejected_bearish_regime']}")
    print(f"    Passed weight           : {gate_counts['passed_weight']}  "
          f"(threshold={P['MIN_WEIGHTED_SCORE']})")
    print(f"    Final signals           : {len(records)}")
    print(f"    Failed fetches          : {failed_count}")
    if exit_holdings_alerts:
        print(f"    EXIT signals on holdings: {len(exit_holdings_alerts)} ⚠️")
    print()

    breadth = compute_market_breadth(records)
    print(f"  Market Breadth : {breadth['breadth_label']} "
          f"({breadth['buy_count']} buy / {breadth['exit_count']} exit)\n")

    if records:
        print(f"  Saving {len(records)} recommendations to Supabase...")
        saved = 0
        for i in range(0, len(records), 20):
            batch = records[i:i + 20]
            try:
                supabase.table("recommendations").insert(batch).execute()
                saved += len(batch)
            except Exception as e:
                print(f"  INSERT ERROR batch {i // 20 + 1}: {e}")
                print(f"  First record keys: {list(batch[0].keys())}")
        print(f"  Saved {saved}/{len(records)} recommendations")
    else:
        print(f"  No signals met threshold (MIN_WEIGHTED_SCORE={P['MIN_WEIGHTED_SCORE']})")
        print(f"  Scanned {len(ALL_TICKERS)} tickers, 0 passed filter")

    try:
        for i in range(0, len(run_logs), 50):
            supabase.table("ticker_run_log").insert(run_logs[i:i + 50]).execute()
    except Exception as e:
        print(f"  ⚠️  Run log insert failed: {e}")

    try:
        supabase.table("agent_meta").upsert(sanitize_for_json({
            "id":                    1,
            "last_run":              today,
            "total_signals":         len(records),
            "tickers_scanned":       len(ALL_TICKERS),
            "failed":                failed_count,
            "market_regime":         regime_label,
            "active_param_version":  param_version,
            "total_buys":            breadth["buy_count"],
            "total_sells":           breadth["sell_count"],
            "total_exits":           breadth["exit_count"],
            "breadth_ratio":         breadth["breadth_ratio"],
            "breadth_label":         breadth["breadth_label"],
            "breadth_buys":          breadth["buy_count"],
            "breadth_sells":         breadth["sell_count"],
            "breadth_exits":         breadth["exit_count"],
            "breadth_neutral":       max(0, len(ALL_TICKERS) - breadth["buy_count"] - breadth["exit_count"]),
        })).execute()
    except Exception as e:
        print(f"  ⚠️  Meta upsert failed: {e}")

    if tg_token and tg_chat and (records or exit_holdings_alerts):
        msg = _build_telegram_message(records, regime_label, breadth, today,
                                       exit_holdings_alerts=exit_holdings_alerts)
        send_telegram_alert(tg_token, tg_chat, msg)
    elif tg_token and not records:
        send_telegram_alert(
            tg_token, tg_chat,
            f"🇮🇳 <b>Indian Stock Agent — {today}</b>\n"
            f"No signals today. Market: {regime_label}.",
        )

    print("  Done ✅\n")


if __name__ == "__main__":
    run()
