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

def backtest(df: pd.DataFrame, signals: pd.Series, p: dict) -> dict:
    trades, reasons = [], []
    in_t, ep, entry_idx = False, 0.0, -1
    closes = df.Close.values
    highs  = df.High.values
    lows   = df.Low.values
    opens  = df.Open.values
    sl_pct   = p["BT_SL_PCT"]
    tgt_pct  = p["BT_TARGET_PCT"]
    max_hold = p["BT_MAX_HOLD"]

    for i in range(1, len(df)):
        if in_t:
            sl_px  = ep * (1 - sl_pct  / 100)
            tgt_px = ep * (1 + tgt_pct / 100)
            if lows[i] <= sl_px:
                trades.append((sl_px  - ep) / ep * 100); reasons.append("sl");      in_t = False
            elif highs[i] >= tgt_px:
                trades.append((tgt_px - ep) / ep * 100); reasons.append("target");  in_t = False
            elif i >= entry_idx + max_hold:
                trades.append((closes[i] - ep) / ep * 100); reasons.append("timeout"); in_t = False
        if signals.iloc[i] == 1 and not in_t:
            ep, entry_idx, in_t = opens[i], i, True

    if not trades:
        return dict(
            win_rate=0, avg_return=0, median_return=0, trades=0,
            profit_factor=0, max_drawdown=0,
            sl_exits=0, target_exits=0, timeout_exits=0,
            trade_returns=[],
        )

    wins   = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    gp, gl = sum(wins), abs(sum(losses))
    pf     = round(gp / gl, 2) if gl > 0 else 99.0
    eq     = np.cumsum(trades)
    peak   = np.maximum.accumulate(eq)
    max_dd = round(float(abs((eq - peak).min())), 2)

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
        "company_name":     NSE_COMPANY_NAMES.get(ticker, ticker),
        "pe_ratio":         None,
        "debt_equity":      None,
        "revenue_growth":   None,
        "fundamental_flag": "DATA_UNAVAILABLE",
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

        return {
            "company_name":     name,
            "pe_ratio":         round(_safe(pe), 1)         if _safe(pe)  is not None else None,
            "debt_equity":      round(_safe(de), 1)         if _safe(de)  is not None else None,
            "revenue_growth":   round(_safe(rg) * 100, 1)   if _safe(rg)  is not None else None,
            "fundamental_flag": ", ".join(flags) if flags else "OK",
        }
    except Exception:
        return _default

# ─────────────────────────────────────────────
#  NEWS SENTIMENT
#  gnews  →  headlines  →  FinBERT (HF API)
#  Only runs when HF_TOKEN env var is set.
# ─────────────────────────────────────────────

def _fetch_news_headlines(ticker: str, company_name: str, n: int = 5) -> list[str]:
    """
    Fetch recent headlines via gnews.
    Returns list of headline strings, empty list on failure.
    """
    if not _GNEWS_OK:
        return []
    try:
        gn      = GNews(language="en", country="IN", period="3d", max_results=n)
        results = gn.get_news(f"{company_name} NSE India stock")
        if not results:
            results = gn.get_news(f"{ticker} NSE share price India")
        headlines = [
            r.get("title", "").strip()
            for r in (results or [])
            if r.get("title") and len(r["title"].strip()) > 10
        ]
        return headlines[:n]
    except Exception:
        return []


def _score_with_finbert(headlines: list[str], hf_token: str) -> tuple[float, str]:
    """
    Score concatenated headlines with FinBERT via HuggingFace Inference API.
    Returns (net_score ∈ [-1,1], sentiment ∈ {POSITIVE, NEUTRAL, NEGATIVE}).
    """
    if not headlines or not hf_token:
        return 0.0, "NEUTRAL"

    text = ". ".join(headlines)[:512]
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type":  "application/json",
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                HF_FINBERT_URL,
                headers=headers,
                json={"inputs": text},
                timeout=25,
            )
            if resp.status_code == 200:
                data = resp.json()
                # Format: [[{label, score}, {label, score}, {label, score}]]
                if isinstance(data, list) and data:
                    scores_raw = data[0] if isinstance(data[0], list) else data
                    sd = {item["label"].lower(): item["score"] for item in scores_raw}
                    pos = sd.get("positive", 0.0)
                    neg = sd.get("negative", 0.0)
                    neu = sd.get("neutral",  0.0)
                    net = pos - neg
                    if   pos > neg and pos > neu: sent = "POSITIVE"
                    elif neg > pos and neg > neu: sent = "NEGATIVE"
                    else:                          sent = "NEUTRAL"
                    return round(net, 3), sent

            elif resp.status_code == 503:
                # Model cold-starting; wait and retry
                wait = min(float(resp.json().get("estimated_time", 20)), 30)
                if attempt < 2:
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

    return 0.0, "NEUTRAL"


def fetch_news_sentiment(ticker: str, company_name: str, hf_token: str) -> dict:
    """
    Full pipeline: gnews headlines → FinBERT scoring.
    Returns dict with news_score, news_sentiment, news_headline.
    news_alert is set later (requires knowing action).
    """
    _empty = {
        "news_score":     None,
        "news_sentiment": None,
        "news_headline":  None,
        "news_alert":     False,
    }
    if not hf_token:
        return _empty

    headlines = _fetch_news_headlines(ticker, company_name)
    if not headlines:
        return {**_empty, "news_sentiment": "NEUTRAL", "news_score": 0.0}

    net_score, sentiment = _score_with_finbert(headlines, hf_token)
    return {
        "news_score":     net_score,
        "news_sentiment": sentiment,
        "news_headline":  headlines[0],   # most recent headline
        "news_alert":     False,          # caller sets this based on action
    }

# ─────────────────────────────────────────────
#  SIGNAL STREAK
#  One DB query before the main loop; O(1) per ticker.
# ─────────────────────────────────────────────

def get_signal_streaks(today: str) -> dict[str, int]:
    """
    Query the last 12 days of recommendations (excluding today).
    Returns {ticker: consecutive_day_count_before_today}.
    A ticker that had a BUY yesterday and the day before returns 2.
    """
    streaks: dict[str, int] = {}
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
            last_action = grp.iloc[0]["action"]
            streak = 0
            for _, row in grp.iterrows():
                if row["action"] == last_action:
                    streak += 1
                else:
                    break
            streaks[ticker] = streak

    except Exception as e:
        print(f"\n  ⚠️  Signal streak fetch failed: {e}")

    return streaks

# ─────────────────────────────────────────────
#  MARKET BREADTH
# ─────────────────────────────────────────────

def compute_market_breadth(records: list[dict]) -> dict:
    """
    Summarises buy/sell ratio across today's signals.
    Label ranges: ≥70% buy = VERY BULLISH … ≤30% buy = VERY BEARISH.
    """
    buy_ct  = sum(1 for r in records if r.get("action") == "BUY")
    sell_ct = sum(1 for r in records if r.get("action") == "SELL")
    total   = buy_ct + sell_ct

    if total == 0:
        return {"buy_count": 0, "sell_count": 0, "breadth_ratio": 0.5, "breadth_label": "NEUTRAL"}

    ratio = buy_ct / total
    if   ratio >= 0.70: label = "VERY BULLISH"
    elif ratio >= 0.55: label = "BULLISH"
    elif ratio >= 0.45: label = "NEUTRAL"
    elif ratio >= 0.30: label = "BEARISH"
    else:               label = "VERY BEARISH"

    return {
        "buy_count":     buy_ct,
        "sell_count":    sell_ct,
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

    lines = [
        f"🇮🇳 <b>Indian Stock Agent — {today}</b>",
        f"Market Regime  : {regime_e} {regime}",
        f"Market Breadth : {breadth.get('breadth_label','?')} "
        f"({breadth.get('buy_count',0)} buys / {breadth.get('sell_count',0)} sells)",
        "",
    ]

    buys  = sorted([r for r in records if r.get("action") == "BUY"],
                   key=lambda x: x.get("composite_score", 0), reverse=True)[:5]
    sells = sorted([r for r in records if r.get("action") == "SELL"],
                   key=lambda x: x.get("composite_score", 0), reverse=True)[:3]

    if buys:
        lines.append("🟢 <b>BUY SIGNALS</b>")
        for idx, r in enumerate(buys, 1):
            streak = r.get("signal_streak", 1)
            streak_str = f" 🔥{streak}d" if streak >= 2 else ""
            ns   = r.get("news_sentiment")
            ne   = news_e.get(ns, "") if ns else ""
            flag = " ⚠️ Bad news!" if r.get("news_alert") else ""

            price = r.get("price")   or 0
            sl    = r.get("stop_loss") or 0
            tgt   = r.get("target")   or 0
            lines.append(
                f"\n{idx}. <b>{r['ticker']}</b>{streak_str} — "
                f"{r.get('composite_score', 0):.0f}/100 ({r.get('score_label','')})"
            )
            lines.append(f"   ₹{price:,.2f} | SL ₹{sl:,.2f} | Target ₹{tgt:,.2f}")
            lines.append(f"   {r.get('active_strategies', '')}")
            if ne:
                headline = r.get("news_headline") or ""
                short_hl = (headline[:75] + "…") if len(headline) > 75 else headline
                lines.append(f"   News: {ne} {ns}{flag}")
                if short_hl:
                    lines.append(f"   📰 {short_hl}")
            fund_flag = r.get("fundamental_flag") or ""
            if fund_flag and fund_flag not in ("OK", "DATA_UNAVAILABLE", ""):
                lines.append(f"   ⚠️ Fundamentals: {fund_flag}")

    if sells:
        lines.append("\n🔴 <b>SELL / EXIT SIGNALS</b>")
        for idx, r in enumerate(sells, 1):
            lines.append(
                f"\n{idx}. <b>{r['ticker']}</b> — "
                f"{r.get('composite_score', 0):.0f}/100 | ₹{r.get('price', 0):,.2f}"
            )
            lines.append(f"   {r.get('active_strategies', '')}")

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
    for name, bt in bt_results.items():
        wr   = bt.get("win_rate", 50) / 100
        n    = bt.get("trades", 0)
        conf = min(n / 8.0, 1.0)
        weights[name] = round(wr * conf, 3)

    total_w  = sum(weights.values()) or 1e-9
    target_v = 1 if action == "BUY" else -1
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

    reward_pct = ctx.get("reward_pct") or 0
    risk_pct   = ctx.get("risk_pct")   or 0
    rr         = (reward_pct / risk_pct) if risk_pct > 0 else 0
    rr_pts     = min(rr / 3.0 * p["W_RR"], p["W_RR"])

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
            "risk_pct": None, "reward_pct": None,
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

    sl  = round(price - 1.5 * atr_now, 2) if math.isfinite(price) and math.isfinite(atr_now) else None
    tgt = round(price + 3.0 * atr_now, 2) if math.isfinite(price) and math.isfinite(atr_now) else None

    low20  = df.Low.rolling(20).min().iloc[-1]
    high20 = df.High.rolling(20).max().iloc[-1]
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
        support    = round(float(low20),  2) if pd.notna(low20)  and math.isfinite(float(low20))  else None,
        resistance = round(float(high20), 2) if pd.notna(high20) and math.isfinite(float(high20)) else None,
        stop_loss  = sl,
        target     = tgt,
        risk_pct   = round((price - sl)  / price * 100, 2) if sl  and price else None,
        reward_pct = round((tgt - price) / price * 100, 2) if tgt and price else None,
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
    print(f"\n🇮🇳 Indian Stock Agent v5 — {today}")

    # ── Credentials
    P, param_version = load_active_params()
    hf_token  = os.environ.get("HF_TOKEN", "")
    tg_token  = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat   = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not hf_token:
        print("  ℹ️  HF_TOKEN not set — news sentiment will be skipped")
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

            bt     = {name: backtest(df, fn(df), P) for name, fn in STRATEGIES.items()}
            action = "BUY" if buy_count >= sell_count else "SELL"
            w_ratio, weights = weighted_vote(today_sigs, bt, action)

            if w_ratio < P["MIN_WEIGHTED_SCORE"]:
                continue
            gate_counts["passed_weight"] += 1

            # ── Context & composite score (computed before fund/news for the threshold check)
            ctx     = context(df, P)
            c_score, c_breakdown = composite_score(today_sigs, bt, ctx, regime_sc, action, P)

            # ── Fundamentals (all passing stocks)
            fund = fetch_fundamentals(ticker)
            company_name = fund.get("company_name") or NSE_COMPANY_NAMES.get(ticker, ticker)

            # ── News sentiment (only if score is interesting enough and HF token present)
            if c_score >= 25 and hf_token:
                news = fetch_news_sentiment(ticker, company_name, hf_token)
                # news_alert: technical signal contradicts news sentiment
                news["news_alert"] = (
                    (action == "BUY"  and news.get("news_sentiment") == "NEGATIVE") or
                    (action == "SELL" and news.get("news_sentiment") == "POSITIVE")
                )
            else:
                news = {
                    "news_score":     None,
                    "news_sentiment": None,
                    "news_headline":  None,
                    "news_alert":     False,
                }

            # ── Signal streak (previous days + today = total)
            prev_streak  = streaks.get(ticker, 0)
            streak_today = prev_streak + 1

            # ── Aggregate backtest for active strategies
            active  = [
                n for n, v in today_sigs.items()
                if (v == 1 and action == "BUY") or (v == -1 and action == "SELL")
            ]
            agg     = lambda k: round(float(np.mean([bt[n][k] for n in active])), 2) if active else 0
            low_smp = (int(np.mean([bt[n]["trades"] for n in active])) < 5) if active else True

            record = dict(
                # Core
                date              = today,
                ticker            = ticker,
                action            = action,
                score             = int(round(float(c_score))) if c_score is not None else 0,
                raw_score         = int(buy_count if action == "BUY" else sell_count),
                weighted_score_val= float(w_ratio),
                composite_score   = float(c_score) if c_score is not None else 0.0,
                score_label       = score_label(c_score if c_score is not None else 0),
                score_breakdown   = json.dumps(sanitize_for_json(c_breakdown)),
                # Strategies
                signals           = json.dumps(sanitize_for_json(today_sigs)),
                strategy_weights  = json.dumps(sanitize_for_json(weights)),
                backtest          = json.dumps(sanitize_for_json(bt)),
                active_strategies = ", ".join(active),
                low_sample_warning= bool(low_smp),
                # Backtest aggregates
                win_rate          = float(agg("win_rate")),
                avg_return        = float(agg("avg_return")),
                median_return     = float(agg("median_return")),
                profit_factor     = float(agg("profit_factor")),
                max_drawdown      = float(agg("max_drawdown")),
                avg_trades        = int(round(np.mean([bt[n]["trades"] for n in active]))) if active else 0,
                # Context
                market_regime     = regime_label,
                param_version     = param_version,
                # Fundamentals
                company_name      = fund.get("company_name"),
                pe_ratio          = fund.get("pe_ratio"),
                debt_equity       = fund.get("debt_equity"),
                revenue_growth    = fund.get("revenue_growth"),
                fundamental_flag  = fund.get("fundamental_flag"),
                # Dashboard/schema aliases for compatibility
                de_ratio          = fund.get("debt_equity"),
                sector            = None,
                market_cap_cr     = None,
                roe               = None,
                fundamental_score = 50 if fund.get("fundamental_flag") in (None, "", "DATA_UNAVAILABLE") else (85 if fund.get("fundamental_flag") == "OK" else 60),
                fundamental_warnings = json.dumps([] if fund.get("fundamental_flag") in (None, "", "OK", "DATA_UNAVAILABLE") else [x.strip() for x in str(fund.get("fundamental_flag", "")).split(',') if x.strip()]),
                # News
                news_score        = news.get("news_score"),
                news_sentiment    = news.get("news_sentiment"),
                news_headline     = news.get("news_headline"),
                news_alert        = bool(news.get("news_alert", False)),
                # Dashboard/schema aliases for compatibility
                news_label        = (news.get("news_sentiment") or "NEUTRAL").lower() if news.get("news_sentiment") else "neutral",
                news_headlines    = json.dumps([news.get("news_headline")] if news.get("news_headline") else []),
                news_multiplier   = 1.0,
                news_count        = 1 if news.get("news_headline") else 0,
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
          f"({breadth['buy_count']} buy / {breadth['sell_count']} sell)\n")

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
            "breadth_ratio":         breadth["breadth_ratio"],
            "breadth_label":         breadth["breadth_label"],
            "breadth_buys":          breadth["buy_count"],
            "breadth_sells":         breadth["sell_count"],
            "breadth_neutral":       max(0, len(records) and (len(ALL_TICKERS) - breadth["buy_count"] - breadth["sell_count"]) or 0),
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
