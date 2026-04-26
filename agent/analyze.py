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
    "BT_SL_PCT":          5.0,
    "BT_TARGET_PCT":      10.0,
    "BT_MAX_HOLD":        15,
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
#  BACKTEST  (SL + Target + Max Hold)
#  trade_returns list is stored so dashboard
#  histograms work without extra DB columns.
# ─────────────────────────────────────────────

def backtest(df: pd.DataFrame, signals: pd.Series, p: dict, benchmark_close: pd.Series | None = None) -> dict:
    """
    Long-entry backtest using next-bar execution.

    A signal that appears on day i is entered at day i+1 open, so the
    strategy does not use the same candle's close/high/low to enter at an
    already-known open. Only BUY signals are traded here; bearish signals are
    treated elsewhere as EXIT signals, not short entries.
    """
    trades, reasons, bench_returns, rel_returns = [], [], [], []
    in_t, ep, entry_idx = False, 0.0, -1
    closes = df.Close.values
    highs  = df.High.values
    lows   = df.Low.values
    opens  = df.Open.values
    sl_pct   = p["BT_SL_PCT"]
    tgt_pct  = p["BT_TARGET_PCT"]
    max_hold = p["BT_MAX_HOLD"]

    aligned_benchmark = None
    if benchmark_close is not None and not benchmark_close.empty:
        try:
            aligned_benchmark = benchmark_close.reindex(df.index).ffill()
        except Exception:
            aligned_benchmark = None

    def close_trade(exit_idx: int, gross_return: float, reason: str):
        trades.append(gross_return)
        reasons.append(reason)
        if aligned_benchmark is not None and entry_idx >= 0:
            try:
                b_entry = float(aligned_benchmark.iloc[entry_idx])
                b_exit  = float(aligned_benchmark.iloc[exit_idx])
                if b_entry > 0 and math.isfinite(b_entry) and math.isfinite(b_exit):
                    b_ret = (b_exit - b_entry) / b_entry * 100
                    bench_returns.append(b_ret)
                    rel_returns.append(gross_return - b_ret)
            except Exception:
                pass

    for i in range(1, len(df)):
        # Enter at today's open only if yesterday produced a BUY signal.
        if (not in_t) and int(signals.iloc[i - 1]) == 1 and opens[i] > 0:
            ep, entry_idx, in_t = opens[i], i, True

        if in_t:
            sl_px  = ep * (1 - sl_pct  / 100)
            tgt_px = ep * (1 + tgt_pct / 100)
            if lows[i] <= sl_px:
                close_trade(i, (sl_px  - ep) / ep * 100, "sl");      in_t = False
            elif highs[i] >= tgt_px:
                close_trade(i, (tgt_px - ep) / ep * 100, "target");  in_t = False
            elif i >= entry_idx + max_hold:
                close_trade(i, (closes[i] - ep) / ep * 100, "timeout"); in_t = False

    if not trades:
        return dict(
            win_rate=0, avg_return=0, median_return=0, trades=0,
            profit_factor=0, max_drawdown=0,
            sl_exits=0, target_exits=0, timeout_exits=0,
            trade_returns=[], benchmark_avg_return=0,
            avg_relative_return=0, benchmark_outperformance_rate=0,
            execution="next_open",
        )

    wins   = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    gp, gl = sum(wins), abs(sum(losses))
    pf     = round(gp / gl, 2) if gl > 0 else 99.0
    eq     = np.cumsum(trades)
    peak   = np.maximum.accumulate(eq)
    max_dd = round(float(abs((eq - peak).min())), 2)

    outperformance_rate = (
        round(sum(1 for r in rel_returns if r > 0) / len(rel_returns) * 100, 1)
        if rel_returns else 0
    )

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
        trade_returns = [round(t, 3) for t in trades],
        benchmark_avg_return = round(float(np.mean(bench_returns)), 2) if bench_returns else 0,
        avg_relative_return  = round(float(np.mean(rel_returns)), 2) if rel_returns else 0,
        benchmark_outperformance_rate = outperformance_rate,
        execution="next_open",
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
            "fundamental_flag":  "OK" if not flags else ",".join(flags),
        }

    except Exception as e:
        print(f"    ⚠️ fundamentals failed for {ticker}: {e}")
        return _default

def fundamental_multiplier(fundamental_score: float | None, warnings: str | None = None) -> float:
    """
    Converts fundamental_score into a controlled score multiplier.
    This intentionally has a small range so technicals remain primary.
    """
    if fundamental_score is None:
        return 1.00
    try:
        fs = float(fundamental_score)
    except Exception:
        return 1.00

    if fs >= 80:
        mult = 1.10
    elif fs >= 65:
        mult = 1.05
    elif fs >= 45:
        mult = 1.00
    elif fs >= 30:
        mult = 0.94
    else:
        mult = 0.88

    if warnings and warnings not in ("OK", "DATA_UNAVAILABLE"):
        mult -= 0.03

    return round(max(0.85, min(1.12, mult)), 3)

# ─────────────────────────────────────────────
#  NEWS SENTIMENT
# ─────────────────────────────────────────────

def clean_headline(h: str) -> str:
    return re.sub(r"\s+", " ", h or "").strip()[:240]

def fetch_news_headlines(company_name: str, ticker: str, max_articles: int = 5) -> list[str]:
    if not _GNEWS_OK:
        return []
    try:
        google_news = GNews(language="en", country="IN", period="7d", max_results=max_articles)
        query       = f'"{company_name}" stock OR shares'
        articles    = google_news.get_news(query)
        headlines   = []
        for a in articles[:max_articles]:
            title = clean_headline(a.get("title", ""))
            if title:
                headlines.append(title)
        return headlines
    except Exception as e:
        print(f"    ⚠️ news fetch failed for {ticker}: {e}")
        return []

def hf_finbert_score(texts: list[str]) -> tuple[float, str, str]:
    """
    Returns score in [-1, +1], label, main headline.
    Requires HF_TOKEN. Falls back to neutral if unavailable.
    """
        if not texts:
        return 0.0, "neutral", ""

    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        return fallback_news_sentiment(texts)

    try:
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"inputs": texts[:5]}
        r = requests.post(HF_FINBERT_URL, headers=headers, json=payload, timeout=20)

        if r.status_code != 200:
            print(f"    ⚠️ FinBERT HTTP {r.status_code}: {r.text[:120]}")
            return fallback_news_sentiment(texts)

        out = r.json()

        scores = []
        labels = []
        for item in out:
            if isinstance(item, list):
                probs = {x.get("label", "").lower(): x.get("score", 0) for x in item}
            elif isinstance(item, dict):
                probs = {item.get("label", "").lower(): item.get("score", 0)}
            else:
                continue

            pos = probs.get("positive", 0)
            neg = probs.get("negative", 0)
            neu = probs.get("neutral", 0)
            sc = pos - neg
            scores.append(sc)

            if pos >= max(neg, neu):
                labels.append("positive")
            elif neg >= max(pos, neu):
                labels.append("negative")
            else:
                labels.append("neutral")

        if not scores:
            return fallback_news_sentiment(texts)

        avg = float(np.mean(scores))
        label = (
            "positive" if avg > 0.15
            else "negative" if avg < -0.15
            else "neutral"
        )
        return round(avg, 3), label, texts[0]

    except Exception as e:
        print(f"    ⚠️ FinBERT failed: {e}")
        return fallback_news_sentiment(texts)

def fallback_news_sentiment(texts: list[str]) -> tuple[float, str, str]:
    """
    Lightweight fallback when HF_TOKEN is missing/unavailable.
    Uses VADER if installed, else a simple financial keyword heuristic.
    """
    joined = " ".join(texts[:5])
    if _VADER_OK:
        try:
            analyzer = SentimentIntensityAnalyzer()
            compound = analyzer.polarity_scores(joined).get("compound", 0.0)
            label = "positive" if compound > 0.15 else "negative" if compound < -0.15 else "neutral"
            return round(float(compound), 3), label, texts[0] if texts else ""
        except Exception:
            pass

    pos_words = [
        "beats", "beat", "profit rises", "surges", "rally", "upgrade", "wins order",
        "record high", "strong", "growth", "expands", "raises guidance", "dividend",
    ]
    neg_words = [
        "misses", "loss", "falls", "plunges", "downgrade", "probe", "fraud",
        "weak", "declines", "default", "resigns", "layoff", "tax notice",
    ]

    text = joined.lower()
    pos = sum(1 for w in pos_words if w in text)
    neg = sum(1 for w in neg_words if w in text)

    if pos == neg == 0:
        return 0.0, "neutral", texts[0] if texts else ""

    score = (pos - neg) / max(1, pos + neg)
    label = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
    return round(float(score), 3), label, texts[0] if texts else ""

def news_multiplier(news_score: float | None, news_label: str | None = None) -> float:
    """
    Converts news_score into a controlled score multiplier.
    Positive news can boost the final score slightly; negative news
    can reduce it meaningfully.
    """
    if news_score is None:
        return 1.00
    try:
        ns = float(news_score)
    except Exception:
        return 1.00

    if ns >= 0.45:
        mult = 1.10
    elif ns >= 0.15:
        mult = 1.05
    elif ns <= -0.45:
        mult = 0.82
    elif ns <= -0.15:
        mult = 0.92
    else:
        mult = 1.00

    if news_label == "negative":
        mult = min(mult, 0.92)
    elif news_label == "positive":
        mult = max(mult, 1.03)

    return round(max(0.80, min(1.12, mult)), 3)

def fetch_news_signal(company_name: str, ticker: str) -> dict:
    headlines = fetch_news_headlines(company_name, ticker, max_articles=5)
    score, label, main = hf_finbert_score(headlines)
    alert = bool(label == "negative" and score <= -0.25)
    return {
        "news_score":     score,
        "news_sentiment": label,
        "news_headline":  main,
        "news_alert":     alert,
        "news_headlines": headlines,
        "news_count":     len(headlines),
        "news_label":     label,
    }

# ─────────────────────────────────────────────
#  SIGNAL STREAK
# ─────────────────────────────────────────────

def load_previous_streaks() -> dict:
    """
    Returns {(ticker, action): streak}
    Looks back up to 10 recent recommendation rows.
    """
    try:
        res = (
            supabase.table("recommendations")
            .select("ticker,action,signal_streak,date")
            .order("date", desc=True)
            .limit(500)
            .execute()
        )
        out = {}
        for row in res.data or []:
            key = (row.get("ticker"), row.get("action"))
            if key not in out:
                out[key] = int(row.get("signal_streak") or row.get("streak") or 0)
        return out
    except Exception as e:
        print(f"⚠️ could not load previous streaks: {e}")
        return {}

# ─────────────────────────────────────────────
#  MARKET REGIME
# ─────────────────────────────────────────────

def get_market_regime(p):
    try:
        df = yf.download("^NSEI", period="9mo", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return "neutral"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        c = df["Close"].dropna()
        e50, e200 = ema(c, 50).iloc[-1], ema(c, 200).iloc[-1]
        if c.iloc[-1] > e50 > e200:
            return "bull"
        if c.iloc[-1] < e50 < e200:
            return "bear"
        return "neutral"
    except Exception:
        return "neutral"

def load_benchmark_returns(period="3y"):
    """
    Loads NIFTY 50 daily closes for benchmark-relative backtests.
    """
    try:
        bdf = yf.download("^NSEI", period=period, interval="1d", progress=False, auto_adjust=True)
        if bdf.empty:
            return pd.Series(dtype=float)
        if isinstance(bdf.columns, pd.MultiIndex):
            bdf.columns = bdf.columns.get_level_values(0)
        return bdf["Close"].dropna()
    except Exception as e:
        print(f"⚠️ benchmark load failed: {e}")
        return pd.Series(dtype=float)

# ─────────────────────────────────────────────
#  SCORE HELPERS
# ─────────────────────────────────────────────

def calculate_support_resistance(df: pd.DataFrame, lookback: int = 20) -> tuple[float, float]:
    """
    Prior support/resistance. Uses only completed prior candles
    to avoid using today's high/low in same-day decision.
    """
    hist = df.iloc[:-1].tail(lookback)
    if hist.empty:
        return None, None
    return float(hist.Low.min()), float(hist.High.max())

def calculate_adaptive_stop_target(df: pd.DataFrame, price: float, p: dict) -> tuple[float, float, float, float, float]:
    """
    Stop/target use prior support/resistance plus ATR. This avoids a fixed
    mechanical 2:1 R:R for every stock.
    """
    support, resistance = calculate_support_resistance(df, lookback=20)
    atr_val = float(atr(df.High, df.Low, df.Close, p["ATR_PERIOD"]).iloc[-1])
    if not math.isfinite(atr_val) or atr_val <= 0:
        atr_val = price * 0.02

    # Stop: below support or ATR cushion, but capped to a reasonable range.
    candidates_stop = [
        price - 1.5 * atr_val,
    ]
    if support and support < price:
        candidates_stop.append(support - 0.25 * atr_val)

    stop = max(candidates_stop)
    max_risk_price = price * 0.92
    min_risk_price = price * 0.985
    stop = max(stop, max_risk_price)
    stop = min(stop, min_risk_price)

    # Target: prior resistance if useful, otherwise ATR extension.
    candidates_target = [
        price + 2.5 * atr_val,
    ]
    if resistance and resistance > price:
        candidates_target.append(resistance)

    target = max(candidates_target)
    target = max(target, price * 1.025)

    risk_pct = (price - stop) / price * 100 if price > 0 else 0
    reward_pct = (target - price) / price * 100 if price > 0 else 0
    rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0

    return (
        round(stop, 2),
        round(target, 2),
        round(risk_pct, 2),
        round(reward_pct, 2),
        round(rr_ratio, 2),
    )

def score_components(
    sigs: dict,
    strategy_weights: dict,
    action: str,
    df: pd.DataFrame,
    p: dict,
    regime: str,
    risk_pct: float,
    reward_pct: float,
    rr_ratio: float,
):
    close = float(df.Close.iloc[-1])
    r_val = float(rsi(df.Close, p["RSI_PERIOD"]).iloc[-1])
    vol   = float(df.Volume.iloc[-1])
    avgv  = float(df.Volume.rolling(20).mean().iloc[-1])
    active = [name for name, val in sigs.items() if val != 0]
    dir_val = 1 if action == "BUY" else -1

    signed_w = sum(strategy_weights.get(k, 0) * dir_val for k, v in sigs.items() if v == dir_val)
    all_w    = sum(abs(strategy_weights.get(k, 0)) for k in sigs.keys()) or 1
    strategy_score = max(0, min(100, (signed_w / all_w) * 100))

    if action == "BUY":
        if r_val < 30:
            rsi_score = 100
        elif r_val < 40:
            rsi_score = 80
        elif r_val < 55:
            rsi_score = 65
        elif r_val < 70:
            rsi_score = 45
        else:
            rsi_score = 20
    else:
        if r_val > 70:
            rsi_score = 100
        elif r_val > 60:
            rsi_score = 80
                elif r_val > 45:
            rsi_score = 55
        else:
            rsi_score = 30

    if avgv > 0:
        vol_ratio = vol / avgv
    else:
        vol_ratio = 1
    volume_score = max(0, min(100, vol_ratio * 50))

    # Dynamic R:R score. Avoid giving full credit to a fixed 2:1 setup.
    if rr_ratio >= 3:
        rr_score = 100
    elif rr_ratio >= 2:
        rr_score = 80
    elif rr_ratio >= 1.5:
        rr_score = 65
    elif rr_ratio >= 1:
        rr_score = 45
    else:
        rr_score = 20

    if action == "BUY":
        regime_score = 100 if regime == "bull" else 50 if regime == "neutral" else 20
    else:
        regime_score = 100 if regime == "bear" else 50 if regime == "neutral" else 20

    raw = (
        strategy_score * p["W_STRATEGY"] / 100
        + rsi_score * p["W_RSI"] / 100
        + volume_score * p["W_VOLUME"] / 100
        + rr_score * p["W_RR"] / 100
        + regime_score * p["W_REGIME"] / 100
    )

    return {
        "strategy_score": round(strategy_score, 1),
        "rsi_score":      round(rsi_score, 1),
        "volume_score":   round(volume_score, 1),
        "rr_score":       round(rr_score, 1),
        "regime_score":   round(regime_score, 1),
        "raw_score":      round(raw, 1),
        "rsi":            round(r_val, 2),
        "volume_ratio":   round(vol_ratio, 2),
        "active":         active,
    }

def label_score(score):
    if score >= 75: return "Strong"
    if score >= 60: return "Medium"
    if score >= 45: return "Weak"
    return "Low"

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    print("🚀 Starting Indian Stock Agent v5")
    today = datetime.now().date().isoformat()

    params, param_version = load_active_params()
    print(f"⚙️  Active params: {param_version}")
    print("   " + json.dumps(params, indent=2)[:500].replace("\n", " "))

    regime = get_market_regime(params)
    print(f"📈 Market regime: {regime}")

    benchmark_close = load_benchmark_returns(period="3y")
    prev_streaks    = load_previous_streaks()

    all_recs, run_log = [], []
    buys = sells = neutral = 0

    # Strategy functions
    strategies = get_strategies(params)

    for ticker in ALL_TICKERS:
        print(f"\n📊 {ticker}")
        try:
            df = yf.download(
                ticker + ".NS",
                period="3y",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if df.empty or len(df) < 252:
                print("  ⚠️ insufficient data")
                run_log.append({"date": today, "ticker": ticker, "status": "insufficient"})
                neutral += 1
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            if df.empty:
                print("  ⚠️ empty after dropna")
                run_log.append({"date": today, "ticker": ticker, "status": "empty"})
                neutral += 1
                continue

            # Last technical values
            close = float(df.Close.iloc[-1])
            change_1d = float((df.Close.iloc[-1] / df.Close.iloc[-2] - 1) * 100) if len(df) > 2 else 0
            change_5d = float((df.Close.iloc[-1] / df.Close.iloc[-6] - 1) * 100) if len(df) > 6 else 0

            r_series = rsi(df.Close, params["RSI_PERIOD"])
            r_val = float(r_series.iloc[-1]) if not pd.isna(r_series.iloc[-1]) else None
            _, _, macd_hist = macd(df.Close, params["MACD_FAST"], params["MACD_SLOW"], params["MACD_SIGNAL"])
            mh_val = float(macd_hist.iloc[-1]) if not pd.isna(macd_hist.iloc[-1]) else None
            ema_bullish = bool(ema(df.Close, params["EMA_SHORT"]).iloc[-1] > ema(df.Close, params["EMA_LONG"]).iloc[-1])
            st_trend, st_line = supertrend(df.High, df.Low, df.Close, params["ATR_PERIOD"], params["SUPERTREND_MULT"])
            supertrend_up = bool(st_trend.iloc[-1] == 1)
            supertrend_line = float(st_line.iloc[-1]) if not pd.isna(st_line.iloc[-1]) else None

            support, resistance = calculate_support_resistance(df, lookback=20)
            stop_loss, target, risk_pct, reward_pct, rr_ratio = calculate_adaptive_stop_target(df, close, params)

            # Per strategy signals and historical performance
            sigs = {}
            bts  = {}
            weights = {}
            for name, fn in strategies.items():
                sig_series = fn(df)
                latest_sig = int(sig_series.iloc[-1])
                sigs[name] = latest_sig

                bt = backtest(df, sig_series, params, benchmark_close=benchmark_close)
                bts[name] = bt

                # Weight: profit factor and sample size. Penalize very tiny samples.
                sample_mult = min(1.0, bt["trades"] / 10) if bt["trades"] else 0
                pf_score = min(bt["profit_factor"], 3) / 3
                wr_score = bt["win_rate"] / 100
                rel_score = 0.55 if bt.get("avg_relative_return", 0) > 0 else 0.45
                weights[name] = round((0.45 * wr_score + 0.35 * pf_score + 0.20 * rel_score) * sample_mult, 3)

            buy_w  = sum(weights[k] for k, v in sigs.items() if v == 1)
            sell_w = sum(weights[k] for k, v in sigs.items() if v == -1)

            if buy_w > sell_w and buy_w >= params["MIN_WEIGHTED_SCORE"]:
                action = "BUY"
                buys += 1
            elif sell_w > buy_w and sell_w >= params["MIN_WEIGHTED_SCORE"]:
                # SELL is not modelled as short-selling. It means exit/avoid long.
                action = "EXIT"
                sells += 1
            else:
                print("  → no weighted signal")
                run_log.append({"date": today, "ticker": ticker, "status": "no_signal"})
                neutral += 1
                continue

            signal_type = "LONG_ENTRY" if action == "BUY" else "EXIT_LONG_OR_AVOID"
            print(f"  ✅ {action} | buy_w={buy_w:.2f} sell_w={sell_w:.2f}")

            # Score components
            components = score_components(
                sigs=sigs,
                strategy_weights=weights,
                action="BUY" if action == "BUY" else "SELL",
                df=df,
                p=params,
                regime=regime,
                risk_pct=risk_pct,
                reward_pct=reward_pct,
                rr_ratio=rr_ratio,
            )
            technical_score = components["raw_score"]

            # Low sample warning
            active_names = components["active"]
            active_bts = [bts[n] for n in active_names] if active_names else []
            avg_trades = int(np.mean([bt["trades"] for bt in active_bts])) if active_bts else 0
            low_sample_warning = bool(avg_trades < 5)

            # Aggregate backtest stats for active strategies
            if active_bts:
                win_rate   = round(float(np.mean([bt["win_rate"] for bt in active_bts])), 1)
                avg_return = round(float(np.mean([bt["avg_return"] for bt in active_bts])), 2)
                med_return = round(float(np.mean([bt["median_return"] for bt in active_bts])), 2)
                pf         = round(float(np.mean([bt["profit_factor"] for bt in active_bts])), 2)
                max_dd     = round(float(np.mean([bt["max_drawdown"] for bt in active_bts])), 2)
                benchmark_avg_return = round(float(np.mean([bt.get("benchmark_avg_return", 0) for bt in active_bts])), 2)
                relative_avg_return  = round(float(np.mean([bt.get("avg_relative_return", 0) for bt in active_bts])), 2)
                benchmark_outperformance_rate = round(
                    float(np.mean([bt.get("benchmark_outperformance_rate", 0) for bt in active_bts])),
                    1,
                )
            else:
                win_rate = avg_return = med_return = pf = max_dd = 0
                benchmark_avg_return = relative_avg_return = benchmark_outperformance_rate = 0

            # Fundamentals for all weighted signals
            fund = fetch_fundamentals(ticker)

            # News only for decent raw technical signals to limit API calls
            if technical_score >= 25:
                news = fetch_news_signal(fund["company_name"], ticker)
            else:
                news = {
                    "news_score": 0.0,
                    "news_sentiment": "neutral",
                    "news_headline": "",
                    "news_alert": False,
                    "news_headlines": [],
                    "news_count": 0,
                    "news_label": "neutral",
                }

            f_mult = fundamental_multiplier(fund.get("fundamental_score"), fund.get("fundamental_flag"))
            n_mult = news_multiplier(news.get("news_score"), news.get("news_sentiment"))

            final_multiplier = round(f_mult * n_mult, 3)
            composite_score = round(max(0, min(100, technical_score * final_multiplier)), 1)

            score_label = label_score(composite_score)

            # Streak
            prev_streak = prev_streaks.get((ticker, action), 0)
            signal_streak = prev_streak + 1

            rec = {
                "date": today,
                "ticker": ticker,
                "action": action,
                "signal_type": signal_type,

                "score": int(round(composite_score)),
                "raw_score": int(round(technical_score)),
                "weighted_score_val": float(buy_w if action == "BUY" else sell_w),
                "technical_score": technical_score,
                "composite_score": composite_score,
                "final_score_multiplier": final_multiplier,
                "score_label": score_label,

                "signals": sanitize_for_json(sigs),
                "strategy_weights": sanitize_for_json(weights),
                "backtest": sanitize_for_json(bts),
                "score_breakdown": sanitize_for_json({
                    **components,
                    "technical_score": technical_score,
                    "fundamental_multiplier": f_mult,
                    "news_multiplier": n_mult,
                    "final_multiplier": final_multiplier,
                    "benchmark_avg_return": benchmark_avg_return,
                    "relative_avg_return": relative_avg_return,
                    "benchmark_outperformance_rate": benchmark_outperformance_rate,
                }),
                              "benchmark_symbol": "^NSEI",
                    "benchmark_return_pct": benchmark_avg_return,
                    "relative_return_pct": relative_avg_return,
                    "benchmark_outperformance_rate": benchmark_outperformance_rate,
                }),

                "active_strategies": ", ".join(active_names),
                "low_sample_warning": low_sample_warning,

                "win_rate": win_rate,
                "avg_return": avg_return,
                "median_return": med_return,
                "profit_factor": pf,
                "max_drawdown": max_dd,
                "avg_trades": avg_trades,

                "benchmark_symbol": "^NSEI",
                "benchmark_return_pct": benchmark_avg_return,
                "relative_return_pct": relative_avg_return,
                "benchmark_outperformance_rate": benchmark_outperformance_rate,

                "market_regime": regime,
                "param_version": param_version,

                "price": round(close, 2),
                "change_1d": round(change_1d, 2),
                "change_5d": round(change_5d, 2),
                "rsi": round(r_val, 2) if r_val is not None else None,
                "macd_hist": round(mh_val, 4) if mh_val is not None else None,
                "ema_bullish": ema_bullish,
                "supertrend_up": supertrend_up,
                "supertrend_line": round(supertrend_line, 2) if supertrend_line is not None else None,

                "support": round(support, 2) if support is not None else None,
                "resistance": round(resistance, 2) if resistance is not None else None,
                "stop_loss": stop_loss,
                "target": target,
                "risk_pct": risk_pct,
                "reward_pct": reward_pct,
                "rr_ratio": rr_ratio,

                "volume": int(df.Volume.iloc[-1]),
                "avg_volume": int(df.Volume.rolling(20).mean().iloc[-1]),

                "company_name": fund.get("company_name"),
                "pe_ratio": fund.get("pe_ratio"),
                "revenue_growth": fund.get("revenue_growth"),
                "debt_equity": fund.get("debt_equity"),
                "de_ratio": fund.get("debt_equity"),
                "fundamental_score": fund.get("fundamental_score"),
                "fundamental_flag": fund.get("fundamental_flag"),
                "fundamental_warnings": sanitize_for_json(
                    [] if fund.get("fundamental_flag") in ("OK", "DATA_UNAVAILABLE", None)
                    else str(fund.get("fundamental_flag")).split(",")
                ),
                "market_cap_cr": fund.get("market_cap_cr"),
                "roe": fund.get("roe"),
                "sector": fund.get("sector"),

                "news_score": news.get("news_score"),
                "news_sentiment": news.get("news_sentiment"),
                "news_headline": news.get("news_headline"),
                "news_alert": news.get("news_alert"),
                "news_label": news.get("news_label") or news.get("news_sentiment"),
                "news_count": news.get("news_count"),
                "news_headlines": sanitize_for_json(news.get("news_headlines", [])),
                "news_multiplier": n_mult,

                "signal_streak": signal_streak,
                "streak": signal_streak,
            }

            all_recs.append(sanitize_for_json(rec))
            run_log.append({"date": today, "ticker": ticker, "status": action})

            # Small pause to reduce yfinance/news throttling risk
            time.sleep(0.15)

        except Exception as e:
            print(f"  ❌ failed: {e}")
            run_log.append({"date": today, "ticker": ticker, "status": "error"})
            neutral += 1
            continue

    # ─────────────────────────────────────────
    # Save recommendations
    # ─────────────────────────────────────────
    print(f"\n💾 Saving {len(all_recs)} recommendations...")

    if all_recs:
        try:
            batch_size = 100
            for i in range(0, len(all_recs), batch_size):
                batch = all_recs[i:i + batch_size]
                supabase.table("recommendations").insert(batch).execute()
            print("✅ Recommendations saved")
        except Exception as e:
            print(f"❌ Recommendation insert failed: {e}")

    # ─────────────────────────────────────────
    # Save run log
    # ─────────────────────────────────────────
    if run_log:
        try:
            batch_size = 200
            for i in range(0, len(run_log), batch_size):
                batch = run_log[i:i + batch_size]
                supabase.table("ticker_run_log").insert(batch).execute()
            print("✅ Run log saved")
        except Exception as e:
            print(f"⚠️ Run log insert failed: {e}")

    # ─────────────────────────────────────────
    # Market breadth + meta update
    # ─────────────────────────────────────────
    signal_total = len(all_recs)
    breadth_buys = sum(1 for r in all_recs if r.get("action") == "BUY")
    breadth_sells = sum(1 for r in all_recs if r.get("action") in ("EXIT", "SELL"))
    breadth_total = breadth_buys + breadth_sells

    if breadth_total > 0:
        breadth_ratio = breadth_buys / breadth_total
    else:
        breadth_ratio = 0.5

    if breadth_ratio >= 0.70:
        breadth_label = "VERY_BULLISH"
    elif breadth_ratio >= 0.55:
        breadth_label = "BULLISH"
    elif breadth_ratio >= 0.45:
        breadth_label = "NEUTRAL"
    elif breadth_ratio >= 0.30:
        breadth_label = "BEARISH"
    else:
        breadth_label = "VERY_BEARISH"

    meta = {
        "id": 1,
        "last_run": today,
        "total_signals": signal_total,
        "tickers_scanned": len(ALL_TICKERS),
        "failed": sum(1 for r in run_log if r.get("status") == "error"),
        "market_regime": regime,
        "active_param_version": param_version,
        "total_buys": breadth_buys,
        "total_sells": breadth_sells,
        "breadth_ratio": round(breadth_ratio, 3),
        "breadth_label": breadth_label,
        "breadth_buys": breadth_buys,
        "breadth_sells": breadth_sells,
        "breadth_neutral": neutral,
        "updated_at": datetime.now().isoformat(),
    }

    try:
        supabase.table("agent_meta").upsert(meta).execute()
        print("✅ Agent meta updated")
    except Exception as e:
        print(f"⚠️ Agent meta update failed: {e}")

    # ─────────────────────────────────────────
    # Optional Telegram alert
    # ─────────────────────────────────────────
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if bot_token and chat_id and all_recs:
        try:
            top_buys = sorted(
                [r for r in all_recs if r.get("action") == "BUY"],
                key=lambda x: x.get("composite_score", 0),
                reverse=True,
            )[:5]

            top_exits = sorted(
                [r for r in all_recs if r.get("action") in ("EXIT", "SELL")],
                key=lambda x: x.get("composite_score", 0),
                reverse=True,
            )[:3]

            lines = [
                f"🇮🇳 <b>Stock Agent — {today}</b>",
                f"Market regime: <b>{regime}</b>",
                f"Breadth: <b>{breadth_label}</b> "
                f"({breadth_buys} BUY / {breadth_sells} EXIT)",
                "",
            ]

            if top_buys:
                lines.append("🟢 <b>Top BUY signals</b>")
                for idx, r in enumerate(top_buys, 1):
                    lines.append(
                        f"{idx}. <b>{r['ticker']}</b> — "
                        f"{r.get('composite_score', 0):.0f}/100 "
                        f"₹{r.get('price', 0):,.2f}"
                    )
                    lines.append(
                        f"   SL ₹{r.get('stop_loss', 0):,.2f} | "
                        f"Target ₹{r.get('target', 0):,.2f} | "
                        f"R:R {r.get('rr_ratio', 0):.2f}"
                    )
                    if r.get("active_strategies"):
                        lines.append(f"   {r.get('active_strategies')}")
                    if r.get("news_headline"):
                        headline = str(r.get("news_headline"))[:90]
                        lines.append(f"   📰 {headline}")

            if top_exits:
                lines.append("")
                lines.append("🔴 <b>Top EXIT / avoid-long signals</b>")
                for idx, r in enumerate(top_exits, 1):
                    lines.append(
                        f"{idx}. <b>{r['ticker']}</b> — "
                        f"{r.get('composite_score', 0):.0f}/100 "
                        f"₹{r.get('price', 0):,.2f}"
                    )
                    if r.get("active_strategies"):
                        lines.append(f"   {r.get('active_strategies')}")

            lines.append("")
            lines.append("Paper-trading signal only. Not financial advice.")

            msg = "\n".join(lines)

            resp = requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": msg,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=20,
            )

            if resp.status_code == 200:
                print("✅ Telegram alert sent")
            else:
                print(f"⚠️ Telegram failed: {resp.status_code} {resp.text[:200]}")

        except Exception as e:
            print(f"⚠️ Telegram alert failed: {e}")

    print("\n✅ Daily analysis complete")
    print(f"Signals: {signal_total} | BUY: {breadth_buys} | EXIT: {breadth_sells} | Neutral: {neutral}")


if __name__ == "__main__":
    main()
