#!/usr/bin/env python3
"""
============================================================
  Indian Stock Market Analysis Agent  —  v5
  Runs daily via GitHub Actions at 7:00 AM IST

  New in v5:
  ─ Fundamental screening data (P/E, D/E, revenue growth)
    fetched via yfinance for every signal that passes the
    weighted-vote filter. Stored for display; not a hard filter.
  ─ News sentiment via gnews + FinBERT (HuggingFace Inference API).
    Fetched only for signals with composite_score >= 25.
    Requires HF_TOKEN GitHub Secret (free tier, optional).
    Adds: news_score, news_sentiment, news_headline, news_alert.
  ─ Signal streak: consecutive days the same ticker fired the
    same action. Loaded in one DB query before the main loop.
  ─ Market breadth: buy/sell signal ratio, saved to agent_meta.
  ─ Telegram morning alert via Bot API after each run.
    Requires TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID Secrets (optional).
  ─ All 6 strategies active in get_strategies().
  ─ trade_returns list stored in each strategy's backtest JSON
    so the dashboard histogram works.
============================================================
"""

import os
import sys
import time
import json
import math
import re
import warnings
from datetime import datetime, timedelta

import requests
import yfinance as yf
import pandas as pd
import numpy as np
from supabase import create_client, Client

try:
    from gnews import GNews
    _GNEWS_OK = True
except ImportError:
    _GNEWS_OK = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_OK = True
except ImportError:
    _VADER_OK = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  SUPABASE
# ─────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────────
#  WATCHLIST
# ─────────────────────────────────────────────
NIFTY50 = [
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
EXTRA_WATCHLIST = []
ALL_TICKERS = NIFTY50 + EXTRA_WATCHLIST

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
    "SUPERTREND_MULT":    3.0,
    "BT_SL_PCT":          5.0,   # fallback only if ATR/S-R levels are unavailable
    "BT_TARGET_PCT":      10.0,  # fallback only if ATR/S-R levels are unavailable
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

# ─────────────────────────────────────────────
#  FUNDAMENTAL SCREEN CONSTANTS
# ─────────────────────────────────────────────
# Thresholds used to set the fundamental_flag (display only — not a hard filter)
FUND_HIGH_PE_THRESHOLD   = 100   # flag if P/E > 100
FUND_LOW_REV_THRESHOLD   = -0.25 # flag if revenue growth < -25%
FUND_HIGH_DE_THRESHOLD   = 400   # flag if D/E % > 400 (non-financial)

# FinBERT via HuggingFace Inference API (free tier)
HF_FINBERT_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

# Known company name overrides for better news search results
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
#  LOAD ACTIVE PARAMS FROM SUPABASE
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
            params = json.loads(row["params_json"]) if isinstance(row["params_json"], str) else row["params_json"]
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

def supertrend(h, l, c, p, mult):
    atr_v = atr(h, l, c, p)
    hl2   = (h + l) / 2
    up    = hl2 + mult * atr_v
    dn    = hl2 - mult * atr_v
    trend = pd.Series(1, index=c.index)
    fu, fl = up.copy(), dn.copy()
    for i in range(1, len(c)):
        fu.iloc[i] = up.iloc[i] if (up.iloc[i] < fu.iloc[i - 1] or c.iloc[i - 1] > fu.iloc[i - 1]) else fu.iloc[i - 1]
        fl.iloc[i] = dn.iloc[i] if (dn.iloc[i] > fl.iloc[i - 1] or c.iloc[i - 1] < fl.iloc[i - 1]) else fl.iloc[i - 1]
        if   trend.iloc[i - 1] == -1 and c.iloc[i] > fu.iloc[i]: trend.iloc[i] =  1
        elif trend.iloc[i - 1] ==  1 and c.iloc[i] < fl.iloc[i]: trend.iloc[i] = -1
        else:                                                       trend.iloc[i] = trend.iloc[i - 1]
    return trend, pd.Series(np.where(trend == 1, fl, fu), index=c.index)

# ─────────────────────────────────────────────
#  SIGNAL GENERATORS
# ─────────────────────────────────────────────

def sig_ema(df, p):
    e_s = ema(df.Close, p["EMA_SHORT"])
    e_l = ema(df.Close, p["EMA_LONG"])
    s   = pd.Series(0, index=df.index)
    s[(e_s > e_l) & (e_s.shift() <= e_l.shift())] =  1
    s[(e_s < e_l) & (e_s.shift() >= e_l.shift())] = -1
    return s

def sig_rsi_macd(df, p):
    r       = rsi(df.Close, p["RSI_PERIOD"])
    _, _, h = macd(df.Close, p["MACD_FAST"], p["MACD_SLOW"], p["MACD_SIGNAL"])
    s       = pd.Series(0, index=df.index)
    s[(r < p["RSI_OVERSOLD"])   & (h > 0) & (h.shift() <= 0)] =  1
    s[(r > p["RSI_OVERBOUGHT"]) & (h < 0) & (h.shift() >= 0)] = -1
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

def sig_volume_breakout(df, p):
    period   = int(p.get("DONCHIAN_PERIOD", 20))
    vol_mult = float(p.get("VOLUME_MULT", 1.5))
    hh       = df.High.rolling(period).max()
    ll       = df.Low.rolling(period).min()
    avg_vol  = df.Volume.rolling(20).mean()
    s        = pd.Series(0, index=df.index)
    s[(df.Close > hh.shift(1)) & (df.Close.shift(1) <= hh.shift(1)) & (df.Volume > avg_vol * vol_mult)] =  1
    s[(df.Close < ll.shift(1)) & (df.Close.shift(1) >= ll.shift(1)) & (df.Volume > avg_vol * vol_mult)] = -1
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
    return {
        "EMA Crossover":   lambda df: sig_ema(df, p),
        "RSI + MACD":      lambda df: sig_rsi_macd(df, p),
        "Bollinger":       lambda df: sig_bb(df, p),
        "Donchian":        lambda df: sig_donchian(df, p),
        "Volume Breakout": lambda df: sig_volume_breakout(df, p),
        "RSI Trend Shift": lambda df: sig_rsi_trend_shift(df, p),
    }

# ─────────────────────────────────────────────
#  DYNAMIC LEVELS + BACKTEST
#  Long-only BUY trades use next-bar execution. Bearish signals
#  are EXIT / avoid-new-long signals, not short entries.
# ─────────────────────────────────────────────

def _finite_float(v, default=None):
    try:
        fv = float(v)
        return fv if math.isfinite(fv) else default
    except Exception:
        return default


def dynamic_trade_levels(df: pd.DataFrame, signal_idx: int, entry_price: float, p: dict) -> dict:
    """
    Build stop/target from information available at signal_idx.
    The trade is assumed to execute on the next bar open, so levels are
    based on prior support/resistance plus ATR, not a fixed 2:1 template.
    """
    if df is None or df.empty or signal_idx < 1 or entry_price is None or entry_price <= 0:
        return {"stop_loss": None, "target": None, "risk_pct": None, "reward_pct": None, "rr_ratio": None}

    lookback = int(p.get("RR_LOOKBACK", 20))
    atr_period = int(p.get("ATR_PERIOD", 14))
    stop_buf = float(p.get("ATR_STOP_BUFFER", 0.50))
    target_buf = float(p.get("ATR_TARGET_BUFFER", 0.50))
    max_risk_atr = float(p.get("MAX_RISK_ATR", 3.00))
    max_risk_pct = float(p.get("MAX_RISK_PCT", 8.00))
    min_rr_ratio = float(p.get("MIN_RR_RATIO", 1.50))

    # Prior support/resistance exclude the signal bar itself. Indicators may
    # use the signal close; the order still waits until the next open.
    support_s = df.Low.rolling(lookback).min().shift(1)
    resist_s  = df.High.rolling(lookback).max().shift(1)
    atr_s     = atr(df.High, df.Low, df.Close, atr_period)

    support = _finite_float(support_s.iloc[signal_idx] if signal_idx < len(support_s) else None)
    resistance = _finite_float(resist_s.iloc[signal_idx] if signal_idx < len(resist_s) else None)
    atr_now = _finite_float(atr_s.iloc[signal_idx] if signal_idx < len(atr_s) else None)

    rr_floor_applied = False

    if atr_now is None or atr_now <= 0:
        # Last-resort fallback when ATR is unavailable. We still apply the
        # MIN_RR_RATIO floor so the published BUY plan can never have its
        # target sitting closer to entry than its stop — that would publish a
        # negative-expectancy setup regardless of the technical score.
        sl_pct_p = float(p.get("BT_SL_PCT", 5.0))
        tgt_pct_p = float(p.get("BT_TARGET_PCT", 10.0))
        # Cap risk by configured max_risk_pct as well — same envelope as main path.
        sl_pct_p = min(sl_pct_p, max_risk_pct)
        sl  = entry_price * (1 - sl_pct_p  / 100.0)
        tgt = entry_price * (1 + tgt_pct_p / 100.0)
    else:
        fallback_sl = entry_price - 1.25 * atr_now
        fallback_tgt = entry_price + 1.75 * atr_now

        if support is not None and support < entry_price:
            sl = support - stop_buf * atr_now
        else:
            sl = fallback_sl

        # Avoid pathological stale support making risk unbounded. Also cap
        # cash risk as a percentage of entry so a far-away support level does
        # not produce a stop whose downside dwarfs the target.
        sl = max(
            sl,
            entry_price - max_risk_atr * atr_now,
            entry_price * (1 - max_risk_pct / 100.0),
        )
        if sl >= entry_price:
            sl = max(fallback_sl, entry_price * (1 - max_risk_pct / 100.0))

        if resistance is not None and resistance > entry_price:
            tgt = resistance + target_buf * atr_now
        else:
            tgt = fallback_tgt

        if tgt <= entry_price:
            tgt = fallback_tgt

    # ── Universal MIN_RR_RATIO floor ─────────────────────────────────────
    # Applies in BOTH paths above. If natural target < entry + risk × min_rr,
    # raise target so reward ≥ min_rr × risk. This is a floor, not a fixed
    # template — stronger resistance/ATR targets keep their natural R:R.
    actual_risk = entry_price - sl
    if actual_risk > 0 and min_rr_ratio > 0:
        min_tgt = entry_price + actual_risk * min_rr_ratio
        if tgt < min_tgt:
            tgt = min_tgt
            rr_floor_applied = True

    # Final sanity guard: target must strictly exceed entry, which must
    # strictly exceed stop. If not, signal is invalid.
    if not (sl is not None and tgt is not None and sl < entry_price < tgt):
        return sanitize_for_json({
            "stop_loss": None, "target": None,
            "risk_pct": None, "reward_pct": None, "rr_ratio": None,
            "support": round(support, 2) if support is not None else None,
            "resistance": round(resistance, 2) if resistance is not None else None,
            "atr": round(atr_now, 2) if atr_now is not None else None,
            "rr_floor_applied": False,
        })

    risk_pct = (entry_price - sl) / entry_price * 100
    reward_pct = (tgt - entry_price) / entry_price * 100
    rr_ratio = reward_pct / risk_pct if risk_pct > 0 else None

    return sanitize_for_json({
        "stop_loss": round(sl, 2),
        "target": round(tgt, 2),
        "risk_pct": round(risk_pct, 2),
        "reward_pct": round(reward_pct, 2),
        "rr_ratio": round(rr_ratio, 2) if rr_ratio is not None else None,
        "support": round(support, 2) if support is not None else None,
        "resistance": round(resistance, 2) if resistance is not None else None,
        "atr": round(atr_now, 2) if atr_now is not None else None,
        "rr_floor_applied": rr_floor_applied,
    })


def _benchmark_return(benchmark_df: pd.DataFrame | None, entry_date, exit_date) -> float | None:
    if benchmark_df is None or benchmark_df.empty:
        return None
    try:
        b = benchmark_df.copy().sort_index()
        b.columns = [c[0] if isinstance(c, tuple) else c for c in b.columns]
        if "Open" not in b.columns or "Close" not in b.columns:
            return None
        entry_slice = b[b.index >= entry_date]
        exit_slice = b[b.index <= exit_date]
        if entry_slice.empty or exit_slice.empty:
            return None
        b_entry = float(entry_slice.iloc[0]["Open"])
        b_exit = float(exit_slice.iloc[-1]["Close"])
        if b_entry <= 0:
            return None
        return round((b_exit - b_entry) / b_entry * 100, 2)
    except Exception:
        return None


def backtest(df: pd.DataFrame, signals: pd.Series, p: dict, benchmark_df: pd.DataFrame | None = None) -> dict:
    """
    Long-only backtest with next-bar execution.

    Signal semantics:
      • BUY generated on bar i enters long at bar i+1 open.
      • EXIT generated on bar i closes an open long at bar i+1 open.
      • EXIT is never treated as a short-sale entry.
    """
    trades, reasons, bench_returns, rel_returns = [], [], [], []
    in_t, ep, entry_idx = False, 0.0, -1
    stop_px = target_px = None
    opens  = df.Open.values
    closes = df.Close.values
    highs  = df.High.values
    lows   = df.Low.values
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
            # A bearish signal exits the long next open; it does not open a short.
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
            stop_px = _finite_float(lv.get("stop_loss"))
            target_px = _finite_float(lv.get("target"))
            entry_idx, in_t = i, True

            # After entering at the open, same-bar high/low may legitimately
            # trigger the precomputed stop/target. Stop-first is conservative.
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
            benchmark_outperformance_rate=None,
        )

    wins   = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    gp, gl = sum(wins), abs(sum(losses))
    pf     = round(gp / gl, 2) if gl > 0 else 99.0
    eq     = np.cumsum(trades)
    peak   = np.maximum.accumulate(eq)
    max_dd = round(float(abs((eq - peak).min())), 2)
    outperf = [1 for r in rel_returns if r > 0]

    return dict(
        win_rate      = round(len(wins) / len(trades) * 100, 1),
        avg_return    = round(float(np.mean(trades)), 2),
        median_return = round(float(np.median(trades)), 2),
        trades        = len(trades),
        profit_factor = min(float(pf), 99.0),
        max_drawdown  = max_dd,
        sl_exits      = reasons.count("sl"),
        target_exits  = reasons.count("target"),
        timeout_exits = reasons.count("timeout"),
        exit_signal_exits = reasons.count("exit_signal"),
        trade_returns = [round(t, 3) for t in trades],
        benchmark_return_pct = round(float(np.mean(bench_returns)), 2) if bench_returns else None,
        relative_return_pct = round(float(np.mean(rel_returns)), 2) if rel_returns else None,
        benchmark_outperformance_rate = round(len(outperf) / len(rel_returns) * 100, 1) if rel_returns else None,
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
#  FUNDAMENTALS
#  Fetches P/E, D/E, revenue growth, company name
#  from yfinance for stocks that pass the vote filter.
#  Stored as informational data — not a hard gate.
# ─────────────────────────────────────────────

def fetch_fundamentals(ticker: str) -> dict:
    """
    Returns dict with company_name, pe_ratio, debt_equity,
    revenue_growth, fundamental_flag.
    All numeric fields default to None on failure (permissive).
    """
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
        de   = info.get("debtToEquity")    # percentage, e.g. 50 means 50%
        rg   = info.get("revenueGrowth")   # decimal, e.g. 0.12 = +12%
        name = (info.get("shortName") or info.get("longName")
                or NSE_COMPANY_NAMES.get(ticker, ticker))

        # Build flags for display
        flags = []
        if pe  is not None and math.isfinite(float(pe))  and float(pe)  > FUND_HIGH_PE_THRESHOLD:
            flags.append("HIGH_PE")
        if rg  is not None and math.isfinite(float(rg))  and float(rg)  < FUND_LOW_REV_THRESHOLD:
            flags.append("LOW_REVENUE")
        if de  is not None and math.isfinite(float(de))  and float(de)  > FUND_HIGH_DE_THRESHOLD:
            flags.append("HIGH_DEBT")

        def _safe(v):
            if v is None: return None
            try:
                fv = float(v)
                return None if not math.isfinite(fv) else fv
            except Exception:
                return None

        # Market cap in crores
        mc_raw = info.get("marketCap") or 0
        mc_cr  = round(mc_raw / 1e7, 0) if mc_raw > 0 else None

        # ROE
        roe_raw = info.get("returnOnEquity")
        roe_val = round(float(roe_raw) * 100, 1) if roe_raw is not None and math.isfinite(float(roe_raw)) else None

        # Sector
        sector_val = info.get("sector") or None

        # Compute a real fundamental score (0-100)
        f_score = 50
        if _safe(pe) is not None:
            if 5 < _safe(pe) < 30:   f_score += 10
            elif _safe(pe) > 80:     f_score -= 10
            elif _safe(pe) < 0:      f_score -= 15
        if roe_val is not None:
            if roe_val > 15:         f_score += 10
            elif roe_val < 0:        f_score -= 10
        if _safe(rg) is not None:
            if _safe(rg) > 0.10:     f_score += 10
            elif _safe(rg) < -0.10:  f_score -= 15
        if _safe(de) is not None:
            de_adj = _safe(de) / 100 if _safe(de) > 20 else _safe(de)
            if de_adj < 0.5:         f_score += 10
            elif de_adj > 3:         f_score -= 10
        if mc_cr and mc_cr > 20000:  f_score += 5
        elif mc_cr and mc_cr < 500:  f_score -= 10
        f_score = max(0, min(100, f_score))

        return {
            "company_name":      name,
            "pe_ratio":          round(_safe(pe), 1)       if _safe(pe)  is not None else None,
            "debt_equity":       round(_safe(de), 1)       if _safe(de)  is not None else None,
            "revenue_growth":    round(_safe(rg) * 100, 1) if _safe(rg)  is not None else None,
            "sector":            sector_val,
            "market_cap_cr":     mc_cr,
            "roe":               roe_val,
            "fundamental_score": f_score,
            "fundamental_flag":  ", ".join(flags) if flags else "OK",
        }
    except Exception:
        return _default

# ─────────────────────────────────────────────
#  NEWS SENTIMENT
#  gnews  →  headlines  →  FinBERT (HF API)
#  Only runs when HF_TOKEN env var is set.
# ─────────────────────────────────────────────

def _normalize_news_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9&+ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _company_aliases(ticker: str, company_name: str) -> list[str]:
    raw = [ticker or "", company_name or ""]
    cleaned = company_name or ""
    for suffix in [" Limited", " Ltd", " Ltd.", " Corporation", " Corp", " Company"]:
        if cleaned.endswith(suffix):
            raw.append(cleaned[: -len(suffix)])

    aliases: list[str] = []
    seen: set[str] = set()
    for item in raw:
        norm = _normalize_news_text(item.replace("NSE:", "").replace("BSE:", ""))
        if norm and len(norm) >= 3 and norm not in seen:
            aliases.append(norm)
            seen.add(norm)
        compact = norm.replace(" ", "")
        if compact and len(compact) >= 4 and compact not in seen:
            aliases.append(compact)
            seen.add(compact)
    return aliases


def _headline_is_relevant(headline: str, ticker: str, company_name: str) -> bool:
    norm = _normalize_news_text(headline)
    if not norm:
        return False
    aliases = _company_aliases(ticker, company_name)
    if not aliases:
        return False
    compact_norm = norm.replace(" ", "")
    return any(alias in norm or alias.replace(" ", "") in compact_norm for alias in aliases)


def _fetch_news_headlines(ticker: str, company_name: str, n: int = 5) -> list[str]:
    """Fetch recent headlines via GNews and keep only clearly relevant ones."""
    if not _GNEWS_OK:
        return []

    queries: list[str] = []
    if company_name:
        queries.extend([
            f'"{company_name}" stock',
            f'"{company_name}" share',
            f'"{company_name}" news',
        ])
    queries.extend([
        f'"{ticker}" NSE',
        f'"{ticker}" stock',
        f'"{ticker}" share',
    ])

    seen: set[str] = set()
    kept: list[str] = []
    try:
        gn = GNews(language="en", country="IN", period="7d", max_results=max(8, n * 2))
        for query in queries:
            for row in gn.get_news(query) or []:
                headline = (row.get("title") or "").strip()
                if len(headline) <= 10:
                    continue
                key = headline.lower()
                if key in seen:
                    continue
                seen.add(key)
                if _headline_is_relevant(headline, ticker, company_name):
                    kept.append(headline)
                    if len(kept) >= n:
                        return kept
        return kept[:n]
    except Exception:
        return []


def _label_from_score(score: float) -> str:
    if score >= 0.12:
        return "POSITIVE"
    if score <= -0.12:
        return "NEGATIVE"
    return "NEUTRAL"


def _news_multiplier_from_score(score: float) -> float:
    # Controlled multiplier: news can nudge, not dominate, the technical model.
    if score >= 0.35:
        return 1.06
    if score >= 0.12:
        return 1.03
    if score <= -0.35:
        return 0.94
    if score <= -0.12:
        return 0.97
    return 1.00


def _fundamental_multiplier_from_score(score: float | None) -> float:
    # Controlled multiplier: fundamentals remain secondary to technicals.
    fs = _finite_float(score, 50.0)
    if fs >= 80:
        return 1.06
    if fs >= 65:
        return 1.03
    if fs <= 25:
        return 0.94
    if fs <= 40:
        return 0.97
    return 1.00


def apply_score_multipliers(technical_score: float, fundamental_score: float | None, news_multiplier: float | None) -> tuple[float, float, float]:
    fund_mult = _fundamental_multiplier_from_score(fundamental_score)
    news_mult = _finite_float(news_multiplier, 1.0)
    final_mult = max(0.88, min(1.12, fund_mult * news_mult))
    final_score = max(0.0, min(100.0, float(technical_score or 0) * final_mult))
    return round(final_score, 1), round(final_mult, 3), round(fund_mult, 3)


def _score_headline_with_finbert(headline: str, hf_token: str):
    if not headline or not hf_token:
        return None
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    for attempt in range(3):
        try:
            resp = requests.post(
                HF_FINBERT_URL,
                headers=headers,
                json={"inputs": headline[:512]},
                timeout=25,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    scores_raw = data[0] if isinstance(data[0], list) else data
                    sd = {
                        str(item.get("label", "")).lower(): float(item.get("score", 0.0))
                        for item in scores_raw
                        if isinstance(item, dict)
                    }
                    pos = sd.get("positive", 0.0)
                    neg = sd.get("negative", 0.0)
                    net = pos - neg
                    return round(net, 3), _label_from_score(net)
            elif resp.status_code == 503 and attempt < 2:
                try:
                    wait = min(float(resp.json().get("estimated_time", 20)), 30)
                except Exception:
                    wait = 10
                time.sleep(wait)
                continue
            else:
                break
        except requests.Timeout:
            if attempt < 2:
                time.sleep(5)
                continue
            break
        except Exception:
            break
    return None


def _score_headline_with_vader(headline: str):
    if not headline or not _VADER_OK:
        return None
    try:
        analyzer = SentimentIntensityAnalyzer()
        compound = float(analyzer.polarity_scores(headline).get("compound", 0.0))
        return round(compound, 3), _label_from_score(compound)
    except Exception:
        return None


def fetch_news_sentiment(ticker: str, company_name: str, hf_token: str) -> dict:
    """Fetch relevant headlines, score them one by one, and return display-ready fields."""
    empty = {
        "news_score": 0.0,
        "news_sentiment": "NEUTRAL",
        "news_headline": None,
        "news_headlines": [],
        "news_count": 0,
        "news_multiplier": 1.0,
        "news_alert": False,
    }

    headlines = _fetch_news_headlines(ticker, company_name)
    if not headlines:
        return empty

    scored = []
    finbert_used = False
    for headline in headlines:
        result = _score_headline_with_finbert(headline, hf_token) if hf_token else None
        if result is not None:
            finbert_used = True
        else:
            result = _score_headline_with_vader(headline)
        if result is not None:
            scored.append((headline, result[0], result[1]))

    if not scored:
        return {**empty, "news_headlines": headlines, "news_count": len(headlines), "news_headline": headlines[0]}

    scored.sort(key=lambda item: abs(item[1]), reverse=True)
    ordered_headlines = [h for h, _, _ in scored]
    avg_score = round(float(sum(item[1] for item in scored) / len(scored)), 3)
    sentiment = _label_from_score(avg_score)
    multiplier = _news_multiplier_from_score(avg_score)

    return {
        "news_score": avg_score,
        "news_sentiment": sentiment,
        "news_headline": ordered_headlines[0],
        "news_headlines": ordered_headlines,
        "news_count": len(ordered_headlines),
        "news_multiplier": multiplier,
        "news_source": "finbert" if finbert_used else ("vader" if _VADER_OK else "headlines_only"),
        "news_alert": False,
    }


def get_signal_streaks(today: str) -> dict[str, tuple[str, int]]:
    """
    Query the last 14 days of recommendations (excluding today).
    Returns {ticker: (last_action, consecutive_day_count_before_today)}.
    Returning the action lets the caller correctly flip the streak when today's
    action differs from yesterday's (a BUY today after a 3-day EXIT streak is
    a fresh 1-day BUY, not a 4-day streak).
    """
    streaks: dict[str, tuple[str, int]] = {}
    try:
        start = (datetime.today() - timedelta(days=14)).strftime("%Y-%m-%d")
        res   = (
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
    """
    Summarises BUY vs EXIT / avoid-new-long ratio across today's signals.
    Label ranges: ≥70% buy = VERY BULLISH … ≤30% buy = VERY BEARISH.
    """
    buy_ct  = sum(1 for r in records if r.get("action") == "BUY")
    exit_ct = sum(1 for r in records if r.get("action") in ("EXIT", "SELL"))
    total   = buy_ct + exit_ct

    if total == 0:
        return {"buy_count": 0, "exit_count": 0, "sell_count": 0, "breadth_ratio": 0.5, "breadth_label": "NEUTRAL"}

    ratio = buy_ct / total
    if   ratio >= 0.70: label = "VERY BULLISH"
    elif ratio >= 0.55: label = "BULLISH"
    elif ratio >= 0.45: label = "NEUTRAL"
    elif ratio >= 0.30: label = "BEARISH"
    else:               label = "VERY BEARISH"

    return {
        "buy_count":     buy_ct,
        "exit_count":    exit_ct,
        "sell_count":    exit_ct,  # backward-compatible agent_meta field
        "breadth_ratio": round(ratio, 3),
        "breadth_label": label,
    }

# ─────────────────────────────────────────────
#  TELEGRAM
#  Morning alert with top signals.
#  Requires TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID env vars.
# ─────────────────────────────────────────────

def _build_telegram_message(records: list[dict], regime: str,
                             breadth: dict, today: str) -> str:
    regime_e = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(regime, "⬜")
    news_e   = {"POSITIVE": "🟢", "NEGATIVE": "🔴", "NEUTRAL": "🟡"}

    def esc(s: str) -> str:
        # Telegram's HTML parse_mode rejects bare `&`. Tickers like M&M would
        # otherwise break the entire message. Order matters — replace & first.
        return (str(s or "")
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

    lines = [
        f"🇮🇳 <b>Indian Stock Agent — {today}</b>",
        f"Market Regime  : {regime_e} {regime}",
        f"Market Breadth : {breadth.get('breadth_label','?')} "
        f"({breadth.get('buy_count',0)} buys / {breadth.get('exit_count', breadth.get('sell_count',0))} exits)",
        "",
    ]

    buys  = sorted([r for r in records if r.get("action") == "BUY"],
                   key=lambda x: x.get("composite_score", 0), reverse=True)[:5]
    exits = sorted([r for r in records if r.get("action") in ("EXIT", "SELL")],
                   key=lambda x: x.get("composite_score", 0), reverse=True)[:3]

    if buys:
        lines.append("🟢 <b>BUY SIGNALS</b>")
        for idx, r in enumerate(buys, 1):
            streak = r.get("signal_streak", 1)
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
            lines.append(
                f"\n{idx}. <b>{esc(r['ticker'])}</b>{streak_str} — "
                f"{r.get('composite_score', 0):.0f}/100 ({esc(r.get('score_label',''))})"
            )
            lines.append(
                f"   ₹{price:,.2f} | "
                f"SL ₹{sl:,.2f} ({sl_pct:+.2f}%) | "
                f"Target ₹{tgt:,.2f} ({tgt_pct:+.2f}%)"
            )

            if active_str:
                lines.append(f"   {esc(active_str)}")
            if ne:
                headline = r.get("news_headline") or ""
                short_hl = (headline[:75] + "…") if len(headline) > 75 else headline
                lines.append(f"   News: {ne} {esc(ns)}{flag}")
                if short_hl:
                    lines.append(f"   📰 {esc(short_hl)}")
            fund_flag = r.get("fundamental_flag") or ""
            if fund_flag and fund_flag not in ("OK", "DATA_UNAVAILABLE", ""):
                lines.append(f"   ⚠️ Fundamentals: {esc(fund_flag)}")

    if exits:
        lines.append("\n🔴 <b>EXIT / AVOID-NEW-LONG SIGNALS</b>")
        for idx, r in enumerate(exits, 1):
            lines.append(
                f"\n{idx}. <b>{esc(r['ticker'])}</b> — "
                f"{r.get('composite_score', 0):.0f}/100 | ₹{r.get('price', 0) or 0:,.2f}"
            )
            active_str = r.get("active_strategies") or ""
            if active_str:
                lines.append(f"   {esc(active_str)}")
            lines.append("   Not a short-sell recommendation.")

    lines.append(f"\n📊 {len(records)} total signals | {today}")
    lines.append("⚠️ Paper trading only. Not financial advice.")
    return "\n".join(lines)


def send_telegram_alert(bot_token: str, chat_id: str, message: str) -> bool:
    """POST message to Telegram Bot API. Returns True on success."""
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
    weights: dict[str, float] = {}
    is_buy = action == "BUY"
    for name, bt in bt_results.items():
        n = bt.get("trades", 0)
        conf = min(n / 8.0, 1.0)
        if is_buy:
            wr = bt.get("win_rate", 50) / 100
            weights[name] = round(wr * conf, 3)
        else:
            # EXIT is a regime/risk signal, not a short sale. Do not use long-trade
            # win-rate/profit-factor to imply bearish-trade expectancy.
            weights[name] = round(max(conf, 0.25), 3)

    total_w  = sum(weights.values()) or 1e-9
    target_v = 1 if is_buy else -1
    signal_w = sum(weights.get(name, 0) for name, v in today_sigs.items() if v == target_v)
    return round(signal_w / total_w, 3), weights


def composite_score(today_sigs: dict, bt_results: dict, ctx: dict,
                    regime_sc: float, action: str, p: dict) -> tuple[float, dict]:
    w_ratio, _ = weighted_vote(today_sigs, bt_results, action)
    strat_pts  = round(w_ratio * p["W_STRATEGY"], 2)

    r = ctx.get("rsi")
    if r is None:
        rsi_pts = 0.0
    elif action == "BUY":
        rsi_pts = max(0.0, min((60 - r) / 40 * p["W_RSI"],   p["W_RSI"]))
    else:
        rsi_pts = max(0.0, min((r - 40) / 40 * p["W_RSI"],   p["W_RSI"]))

    avg_vol   = ctx.get("avg_volume") or 0
    cur_vol   = ctx.get("volume")     or 0
    vol_ratio = (cur_vol / avg_vol) if avg_vol > 0 else 1.0
    vol_pts   = min(vol_ratio / 2.0 * p["W_VOLUME"], p["W_VOLUME"])

    rr = ctx.get("rr_ratio")
    rr = _finite_float(rr, 0.0) or 0.0
    # The "good R:R" anchor for BUY scoring (3.0) and the EXIT penalty cutoff
    # are derived from MIN_RR_RATIO so re-tuning the floor stays consistent.
    min_rr = float(p.get("MIN_RR_RATIO", 1.5))
    rr_good = max(min_rr * 2.0, 3.0)            # full BUY R:R points at this R:R
    rr_exit_cap = max(min_rr * 1.33, 2.0)        # above this, EXIT R:R penalty is 0
    if action == "BUY":
        rr_pts = max(0.0, min(rr / rr_good * p["W_RR"], p["W_RR"]))
    else:
        # For EXIT, a poor long-side R:R strengthens the avoid-new-long signal.
        rr_pts = max(0.0, min((rr_exit_cap - rr) / rr_exit_cap * p["W_RR"], p["W_RR"]))

    reg_pts = (regime_sc * p["W_REGIME"]
               if action == "BUY"
               else (1 - regime_sc) * p["W_REGIME"])

    total = round(strat_pts + rsi_pts + vol_pts + rr_pts + reg_pts, 1)
    breakdown = dict(
        strategy = round(strat_pts, 1),
        rsi      = round(rsi_pts,   1),
        volume   = round(vol_pts,   1),
        rr       = round(rr_pts,    1),
        regime   = round(reg_pts,   1),
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
            progress=False,
            auto_adjust=True,
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
            progress=False,
            auto_adjust=True,
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
    if len(c) < 20:
        return sanitize_for_json({
            "price": None, "change_1d": None, "change_5d": None,
            "rsi": None, "macd_hist": None, "ema_bullish": None,
            "supertrend_up": None, "supertrend_line": None,
            "support": None, "resistance": None,
            "stop_loss": None, "target": None,
            "risk_pct": None, "reward_pct": None, "rr_ratio": None,
            "volume": 0, "avg_volume": 0,
        })

    r        = float(rsi(c, p["RSI_PERIOD"]).iloc[-1])
    _, _, h  = macd(c, p["MACD_FAST"], p["MACD_SLOW"], p["MACD_SIGNAL"])
    e_s      = float(ema(c, p["EMA_SHORT"]).iloc[-1])
    e_l      = float(ema(c, p["EMA_LONG"]).iloc[-1])
    trend, st_line = supertrend(df.High, df.Low, c, p["ATR_PERIOD"], p["SUPERTREND_MULT"])

    price   = float(c.iloc[-1])
    atr_now = float(atr(df.High, df.Low, c, p["ATR_PERIOD"]).iloc[-1])

    prev_1 = float(c.iloc[-2]) if len(c) >= 2 and pd.notna(c.iloc[-2]) else None
    prev_5 = float(c.iloc[-6]) if len(c) >= 6 and pd.notna(c.iloc[-6]) else None

    levels = dynamic_trade_levels(df, len(df) - 1, price, p)
    sl  = levels.get("stop_loss")
    tgt = levels.get("target")

    low20  = levels.get("support")
    high20 = levels.get("resistance")
    vol20  = df.Volume.rolling(20).mean().iloc[-1]

    out = dict(
        price    = round(price, 2) if math.isfinite(price) else None,
        change_1d = round((price - prev_1) / prev_1 * 100, 2)
                    if prev_1 not in (None, 0) and math.isfinite(prev_1) else None,
        change_5d = round((price - prev_5) / prev_5 * 100, 2)
                    if prev_5 not in (None, 0) and math.isfinite(prev_5) else None,
        rsi         = round(r, 1) if math.isfinite(r) else None,
        macd_hist   = round(float(h.iloc[-1]), 3)
                      if pd.notna(h.iloc[-1]) and math.isfinite(float(h.iloc[-1])) else None,
        ema_bullish     = bool(e_s > e_l) if math.isfinite(e_s) and math.isfinite(e_l) else None,
        supertrend_up   = bool(trend.iloc[-1] == 1) if pd.notna(trend.iloc[-1]) else None,
        supertrend_line = round(float(st_line.iloc[-1]), 2)
                          if pd.notna(st_line.iloc[-1]) and math.isfinite(float(st_line.iloc[-1])) else None,
        support    = round(float(low20),  2) if low20 is not None and math.isfinite(float(low20))  else None,
        resistance = round(float(high20), 2) if high20 is not None and math.isfinite(float(high20)) else None,
        stop_loss  = sl,
        target     = tgt,
        risk_pct   = levels.get("risk_pct"),
        reward_pct = levels.get("reward_pct"),
        rr_ratio   = levels.get("rr_ratio"),
        rr_floor_applied = bool(levels.get("rr_floor_applied", False)),
        volume     = int(df.Volume.iloc[-1]) if pd.notna(df.Volume.iloc[-1]) else 0,
        avg_volume = int(vol20) if pd.notna(vol20) and math.isfinite(float(vol20)) else 0,
    )
    return sanitize_for_json(out)

# ─────────────────────────────────────────────
#  MARKET REGIME  (NIFTY 50 EMA trend)
# ─────────────────────────────────────────────

def market_regime(p: dict) -> tuple[str, float]:
    try:
        df = yf.download("^NSEI", period="120d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 55:
            return "UNKNOWN", 0.5
        # yfinance can return MultiIndex columns; flatten defensively.
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        if "Close" not in df.columns:
            return "UNKNOWN", 0.5
        close = df["Close"].squeeze()
        e50   = float(ema(close, 50).iloc[-1])
        e20   = float(ema(close, 20).iloc[-1])
        r     = float(rsi(close, p["RSI_PERIOD"]).iloc[-1])
        price = float(close.iloc[-1])
        if price > e20 > e50 and r > 50: return "BULLISH", 1.0
        if price < e20 < e50 and r < 50: return "BEARISH", 0.0
        return "NEUTRAL", 0.5
    except Exception:
        return "UNKNOWN", 0.5

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run():
    today = datetime.today().strftime("%Y-%m-%d")
    print(f"\n🇮🇳 Indian Stock Agent v6 — {today}")

    # ── Credentials
    P, param_version = load_active_params()
    hf_token  = os.environ.get("HF_TOKEN", "")
    tg_token  = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat   = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not hf_token:
        if _VADER_OK:
            print("  ℹ️  HF_TOKEN not set — news sentiment will use VADER fallback")
        else:
            print("  ℹ️  HF_TOKEN not set and vaderSentiment unavailable — news disabled")
    if not tg_token:
        print("  ℹ️  TELEGRAM_BOT_TOKEN not set — Telegram alerts disabled")
    if not _GNEWS_OK:
        print("  ℹ️  gnews not installed — news features disabled")

    print(f"   Params : {param_version}")
    print(f"   Stocks : {len(ALL_TICKERS)} NSE tickers\n")

    STRATEGIES = get_strategies(P)

    print("   Checking NIFTY market regime...")
    regime_label, regime_sc = market_regime(P)
    print(f"   Regime : {regime_label}  (score={regime_sc})")

    print("   Loading NIFTY benchmark (^NSEI) for relative metrics...")
    benchmark_df = fetch_benchmark()
    print(f"   Benchmark bars: {len(benchmark_df)}")

    # ── Pre-fetch signal streaks in one query
    print("   Loading signal streak history...")
    streaks = get_signal_streaks(today)
    print(f"   {len(streaks)} tickers have recent signal history\n")

    # ── Wipe today's stale rows (idempotent re-runs)
    supabase.table("recommendations").delete().eq("date", today).execute()
    supabase.table("ticker_run_log").delete().eq("date", today).execute()

    records, run_logs = [], []
    gate_counts = {"fetched": 0, "any_signal": 0, "passed_weight": 0}

    for i, ticker in enumerate(ALL_TICKERS, 1):
        sys.stdout.write(f"\r  {i}/{len(ALL_TICKERS)}  {ticker:<15}")
        sys.stdout.flush()

        df, status = fetch(ticker)
        run_logs.append(sanitize_for_json({"date": today, "ticker": ticker, "status": status}))
        if df is None:
            continue
        gate_counts["fetched"] += 1

        try:
            today_sigs = {name: int(fn(df).iloc[-1]) for name, fn in STRATEGIES.items()}
            buy_count  = sum(1 for v in today_sigs.values() if v ==  1)
            sell_count = sum(1 for v in today_sigs.values() if v == -1)

            if buy_count == 0 and sell_count == 0:
                continue
            gate_counts["any_signal"] += 1

            bt     = {name: backtest(df, fn(df), P, benchmark_df) for name, fn in STRATEGIES.items()}
            action = "BUY" if buy_count >= sell_count else "EXIT"
            w_ratio, weights = weighted_vote(today_sigs, bt, action)

            if w_ratio < P["MIN_WEIGHTED_SCORE"]:
                continue
            gate_counts["passed_weight"] += 1

            # ── Context & raw technical score (computed before fund/news for the threshold check)
            ctx     = context(df, P)
            technical_score, c_breakdown = composite_score(today_sigs, bt, ctx, regime_sc, action, P)

            # ── Fundamentals (all passing stocks)
            fund = fetch_fundamentals(ticker)
            company_name = fund.get("company_name") or NSE_COMPANY_NAMES.get(ticker, ticker)

            # ── News sentiment (gate on score + at least one available backend)
            # Previously only ran when HF_TOKEN was set, which silently disabled
            # VADER even when it was installed. Now: run if EITHER FinBERT
            # (HF token) OR VADER is available, and gnews can fetch headlines.
            news_backend_ok = _GNEWS_OK and (bool(hf_token) or _VADER_OK)
            if technical_score >= 25 and news_backend_ok:
                news = fetch_news_sentiment(ticker, company_name, hf_token)
                # news_alert: technical signal contradicts news sentiment
                news["news_alert"] = (
                    (action == "BUY"  and news.get("news_sentiment") == "NEGATIVE") or
                    (action == "EXIT" and news.get("news_sentiment") == "POSITIVE")
                )
            else:
                news = {
                    "news_score": 0.0,
                    "news_sentiment": "NEUTRAL",
                    "news_headline": None,
                    "news_headlines": [],
                    "news_count": 0,
                    "news_multiplier": 1.0,
                    "news_source": "disabled",
                    "news_alert": False,
                }

            final_score, final_multiplier, fundamental_multiplier = apply_score_multipliers(
                technical_score, fund.get("fundamental_score", 50), news.get("news_multiplier", 1.0)
            )
            c_breakdown.update({
                "technical_score": round(float(technical_score or 0), 1),
                "final_score": final_score,
                "news_multiplier": float(news.get("news_multiplier", 1.0)),
                "fundamental_multiplier": fundamental_multiplier,
                "final_score_multiplier": final_multiplier,
            })

            # ── Signal streak (previous days + today = total)
            # Reset to 1 if today's action flips relative to the previous streak.
            prev_action, prev_streak = streaks.get(ticker, (None, 0))
            streak_today = (prev_streak + 1) if prev_action == action else 1

            # ── Aggregate backtest for active BUY strategies only. EXIT signals are
            # not short trades, so win-rate/profit-factor are intentionally blank.
            active  = [
                n for n, v in today_sigs.items()
                if (v == 1 and action == "BUY") or (v == -1 and action == "EXIT")
            ]
            def agg(k):
                vals = [bt[n].get(k) for n in active if bt[n].get(k) is not None]
                return round(float(np.mean(vals)), 2) if vals and action == "BUY" else None
            trade_vals = [bt[n].get("trades", 0) for n in active]
            avg_trade_count = int(round(np.mean(trade_vals))) if trade_vals and action == "BUY" else 0
            low_smp = (avg_trade_count < 5) if action == "BUY" else False

            record = dict(
                # Core
                date              = today,
                ticker            = ticker,
                action            = action,
                score             = int(round(float(final_score))) if final_score is not None else 0,
                raw_score         = int(buy_count if action == "BUY" else sell_count),
                weighted_score_val= float(w_ratio),
                technical_score   = float(technical_score) if technical_score is not None else 0.0,
                composite_score   = float(final_score) if final_score is not None else 0.0,
                final_score_multiplier = float(final_multiplier),
                fundamental_multiplier = float(fundamental_multiplier),
                score_label       = score_label(final_score if final_score is not None else 0),
                score_breakdown   = json.dumps(sanitize_for_json(c_breakdown)),
                # Strategies
                signals           = json.dumps(sanitize_for_json(today_sigs)),
                strategy_weights  = json.dumps(sanitize_for_json(weights)),
                backtest          = json.dumps(sanitize_for_json(bt)),
                active_strategies = ", ".join(active),
                low_sample_warning= bool(low_smp),
                # Backtest aggregates
                win_rate          = agg("win_rate"),
                avg_return        = agg("avg_return"),
                median_return     = agg("median_return"),
                profit_factor     = agg("profit_factor"),
                max_drawdown      = agg("max_drawdown"),
                avg_trades        = avg_trade_count,
                benchmark_return_pct = agg("benchmark_return_pct"),
                relative_return_pct  = agg("relative_return_pct"),
                benchmark_outperformance_rate = agg("benchmark_outperformance_rate"),
                # Context
                market_regime     = regime_label,
                param_version     = param_version,
                # Fundamentals
                company_name      = fund.get("company_name"),
                pe_ratio          = fund.get("pe_ratio"),
                debt_equity       = fund.get("debt_equity"),
                revenue_growth    = fund.get("revenue_growth"),
                fundamental_flag  = fund.get("fundamental_flag"),
                # Additional fundamental fields from yfinance
                de_ratio          = fund.get("debt_equity"),
                sector            = fund.get("sector"),
                market_cap_cr     = fund.get("market_cap_cr"),
                roe               = fund.get("roe"),
                fundamental_score = fund.get("fundamental_score", 50),
                fundamental_warnings = json.dumps([] if fund.get("fundamental_flag") in (None, "", "OK", "DATA_UNAVAILABLE") else [x.strip() for x in str(fund.get("fundamental_flag", "")).split(',') if x.strip()]),
                # News
                news_score        = news.get("news_score"),
                news_sentiment    = news.get("news_sentiment"),
                news_headline     = news.get("news_headline"),
                news_alert        = bool(news.get("news_alert", False)),
                # Dashboard/schema aliases for compatibility
                news_label        = (news.get("news_sentiment") or "NEUTRAL").lower(),
                news_headlines    = json.dumps(sanitize_for_json(news.get("news_headlines") or ([news.get("news_headline")] if news.get("news_headline") else []))),
                news_multiplier   = float(news.get("news_multiplier", 1.0)),
                news_count        = int(news.get("news_count", 1 if news.get("news_headline") else 0)),
                # Streak
                signal_streak     = streak_today,
                streak            = streak_today,
                **ctx,
            )

            safe_record = sanitize_for_json(record)
            json.dumps(safe_record)   # validate serializability
            records.append(safe_record)

        except Exception as e:
            print(f"\n  ⚠️  Skipping {ticker}: {e}")

        time.sleep(0.1)

    sys.stdout.write("\r" + " " * 60 + "\r")

    # ── Pipeline summary
    failed_count = sum(1 for l in run_logs if l["status"] != "ok")
    print(f"  Pipeline summary:")
    print(f"    Tickers scanned  : {len(ALL_TICKERS)}")
    print(f"    Data fetched OK  : {gate_counts['fetched']}")
    print(f"    Any signal fired : {gate_counts['any_signal']}")
    print(f"    Passed weight    : {gate_counts['passed_weight']}  (threshold={P['MIN_WEIGHTED_SCORE']})")
    print(f"    Final signals    : {len(records)}")
    print(f"    Failed fetches   : {failed_count}\n")

    # ── Market breadth
    breadth = compute_market_breadth(records)
    print(f"  Market Breadth : {breadth['breadth_label']} "
          f"({breadth['buy_count']} buy / {breadth['exit_count']} exit)\n")

    # ── Save recommendations
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

    # ── Save run log
    try:
        for i in range(0, len(run_logs), 50):
            supabase.table("ticker_run_log").insert(run_logs[i:i + 50]).execute()
    except Exception as e:
        print(f"  ⚠️  Run log insert failed: {e}")

    # ── Update agent_meta (includes breadth)
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

    # ── Telegram morning alert
    if tg_token and tg_chat and records:
        msg = _build_telegram_message(records, regime_label, breadth, today)
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
