#!/usr/bin/env python3
"""
============================================================
  Indian Stock Market Analysis Agent  —  v4
  Runs daily via GitHub Actions at 7:00 AM IST

  v4 live setup:
  ─ Uses only 3 live strategies:
      • EMA Crossover
      • RSI + MACD
      • Bollinger
  ─ Lowers MIN_WEIGHTED_SCORE so single-strategy signals
    like Bollinger can survive more often
  ─ Keeps Supertrend only for context display
============================================================
"""

import os
import sys
import time
import json
import warnings
import math
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
from supabase import create_client, Client

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
#  LOAD ACTIVE PARAMS
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
            row = res.data[0]
            params = json.loads(row["params_json"]) if isinstance(row["params_json"], str) else row["params_json"]
            merged = {**DEFAULT_PARAMS, **params}
            return merged, f"v{row['version']} (score={row['objective_score']:.3f})"
    except Exception as e:
        print(f"  ⚠️ Could not load champion params: {e}")
    return DEFAULT_PARAMS.copy(), "defaults"

# ─────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────
def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def rsi(s, p=14):
    d = s.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / p, min_periods=p, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1 / p, min_periods=p, adjust=False).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))

def macd(s, fast, slow, sig):
    ml = ema(s, fast) - ema(s, slow)
    sl = ema(ml, sig)
    return ml, sl, ml - sl

def bollinger(s, p, std):
    m = s.rolling(p).mean()
    sg = s.rolling(p).std()
    return m + std * sg, m, m - std * sg

def atr(h, l, c, p):
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / p, min_periods=p, adjust=False).mean()

def supertrend(h, l, c, p, mult):
    atr_v = atr(h, l, c, p)
    hl2 = (h + l) / 2
    up = hl2 + mult * atr_v
    dn = hl2 - mult * atr_v
    trend = pd.Series(1, index=c.index)
    fu, fl = up.copy(), dn.copy()

    for i in range(1, len(c)):
        fu.iloc[i] = up.iloc[i] if (up.iloc[i] < fu.iloc[i - 1] or c.iloc[i - 1] > fu.iloc[i - 1]) else fu.iloc[i - 1]
        fl.iloc[i] = dn.iloc[i] if (dn.iloc[i] > fl.iloc[i - 1] or c.iloc[i - 1] < fl.iloc[i - 1]) else fl.iloc[i - 1]
        if trend.iloc[i - 1] == -1 and c.iloc[i] > fu.iloc[i]:
            trend.iloc[i] = 1
        elif trend.iloc[i - 1] == 1 and c.iloc[i] < fl.iloc[i]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]

    return trend, pd.Series(np.where(trend == 1, fl, fu), index=c.index)

# ─────────────────────────────────────────────
#  SIGNAL GENERATORS
# ─────────────────────────────────────────────
def sig_ema(df, p):
    e_s = ema(df.Close, p["EMA_SHORT"])
    e_l = ema(df.Close, p["EMA_LONG"])
    s = pd.Series(0, index=df.index)
    s[(e_s > e_l) & (e_s.shift() <= e_l.shift())] = 1
    s[(e_s < e_l) & (e_s.shift() >= e_l.shift())] = -1
    return s

def sig_rsi_macd(df, p):
    r = rsi(df.Close, p["RSI_PERIOD"])
    _, _, hist = macd(df.Close, p["MACD_FAST"], p["MACD_SLOW"], p["MACD_SIGNAL"])
    s = pd.Series(0, index=df.index)
    s[(r < p["RSI_OVERSOLD"]) & (hist > 0) & (hist.shift() <= 0)] = 1
    s[(r > p["RSI_OVERBOUGHT"]) & (hist < 0) & (hist.shift() >= 0)] = -1
    return s

def sig_bb(df, p):
    up, _, lo = bollinger(df.Close, p["BB_PERIOD"], p["BB_STD"])
    r = rsi(df.Close, p["RSI_PERIOD"])
    s = pd.Series(0, index=df.index)
    s[(df.Low <= lo) & (df.Close > lo) & (r < 50)] = 1
    s[(df.High >= up) & (df.Close < up) & (r > 50)] = -1
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
    period = int(p.get("DONCHIAN_PERIOD", 20))
    vol_mult = float(p.get("VOLUME_MULT", 1.5))
    hh = df.High.rolling(period).max()
    ll = df.Low.rolling(period).min()
    avg_vol = df.Volume.rolling(20).mean()
    s = pd.Series(0, index=df.index)

    buy_cond = (
        (df.Close > hh.shift(1)) &
        (df.Close.shift(1) <= hh.shift(1)) &
        (df.Volume > avg_vol * vol_mult)
    )
    sell_cond = (
        (df.Close < ll.shift(1)) &
        (df.Close.shift(1) >= ll.shift(1)) &
        (df.Volume > avg_vol * vol_mult)
    )

    s[buy_cond] = 1
    s[sell_cond] = -1
    return s

def sig_rsi_trend_shift(df, p):
    r = rsi(df.Close, p["RSI_PERIOD"])
    e_l = ema(df.Close, p["EMA_LONG"])
    mid = float(p.get("RSI_MIDLINE", 50))
    s = pd.Series(0, index=df.index)

    buy_cond = (
        (df.Close > e_l) &
        (r > mid) &
        (r.shift(1) <= mid)
    )
    sell_cond = (
        (df.Close < e_l) &
        (r < mid) &
        (r.shift(1) >= mid)
    )

    s[buy_cond] = 1
    s[sell_cond] = -1
    return s

def get_strategies(p):
    return {
        "EMA Crossover":    lambda df: sig_ema(df, p),
        "RSI + MACD":       lambda df: sig_rsi_macd(df, p),
        "Bollinger":        lambda df: sig_bb(df, p),
        "Donchian":         lambda df: sig_donchian(df, p),
        "Volume Breakout":  lambda df: sig_volume_breakout(df, p),
        "RSI Trend Shift":  lambda df: sig_rsi_trend_shift(df, p),
    }

# ─────────────────────────────────────────────
#  BACKTEST
# ─────────────────────────────────────────────
def backtest(df, signals, p):
    trades, reasons = [], []
    in_t, ep, entry_idx = False, 0.0, -1
    closes, highs, lows, opens = df.Close.values, df.High.values, df.Low.values, df.Open.values
    sl_pct = p["BT_SL_PCT"]
    tgt_pct = p["BT_TARGET_PCT"]
    max_hold = p["BT_MAX_HOLD"]

    for i in range(1, len(df)):
        if in_t:
            sl_px = ep * (1 - sl_pct / 100)
            tgt_px = ep * (1 + tgt_pct / 100)
            if lows[i] <= sl_px:
                trades.append((sl_px - ep) / ep * 100)
                reasons.append("sl")
                in_t = False
            elif highs[i] >= tgt_px:
                trades.append((tgt_px - ep) / ep * 100)
                reasons.append("target")
                in_t = False
            elif i >= entry_idx + max_hold:
                trades.append((closes[i] - ep) / ep * 100)
                reasons.append("timeout")
                in_t = False

        if signals.iloc[i] == 1 and not in_t:
            ep, entry_idx, in_t = opens[i], i, True

    if not trades:
        return dict(
            win_rate=0, avg_return=0, median_return=0, trades=0,
            profit_factor=0, max_drawdown=0, sl_exits=0, target_exits=0, timeout_exits=0
        )

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    gp, gl = sum(wins), abs(sum(losses))
    pf = round(gp / gl, 2) if gl > 0 else 99.0
    eq = np.cumsum(trades)
    peak = np.maximum.accumulate(eq)
    max_dd = round(float(abs((eq - peak).min())), 2)

    return dict(
        win_rate=round(len(wins) / len(trades) * 100, 1),
        avg_return=round(float(np.mean(trades)), 2),
        median_return=round(float(np.median(trades)), 2),
        trades=len(trades),
        profit_factor=min(float(pf), 99.0),
        max_drawdown=max_dd,
        sl_exits=reasons.count("sl"),
        target_exits=reasons.count("target"),
        timeout_exits=reasons.count("timeout"),
    )

# ─────────────────────────────────────────────
#  JSON SANITIZER
# ─────────────────────────────────────────────
def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj

# ─────────────────────────────────────────────
#  WEIGHTED VOTE + COMPOSITE SCORE
# ─────────────────────────────────────────────
def weighted_vote(today_sigs, bt_results, action):
    weights = {}
    for name, bt in bt_results.items():
        wr = bt.get("win_rate", 50) / 100
        n = bt.get("trades", 0)
        weights[name] = round(wr * min(n / 8.0, 1.0), 3)

    total_w = sum(weights.values()) or 1e-9
    target_v = 1 if action == "BUY" else -1
    signal_w = sum(weights.get(n, 0) for n, v in today_sigs.items() if v == target_v)
    return round(signal_w / total_w, 3), weights

def composite_score(today_sigs, bt_results, ctx, regime_sc, action, p):
    w_ratio, _ = weighted_vote(today_sigs, bt_results, action)
    strat_pts = round(w_ratio * p["W_STRATEGY"], 2)

    r = ctx.get("rsi")
    if r is None:
        rsi_pts = 0
    elif action == "BUY":
        rsi_pts = max(0, min((60 - r) / 40 * p["W_RSI"], p["W_RSI"]))
    else:
        rsi_pts = max(0, min((r - 40) / 40 * p["W_RSI"], p["W_RSI"]))

    avg_volume = ctx.get("avg_volume") or 0
    volume = ctx.get("volume") or 0
    vol_ratio = (volume / avg_volume) if avg_volume > 0 else 1.0
    vol_pts = min(vol_ratio / 2.0 * p["W_VOLUME"], p["W_VOLUME"])

    reward_pct = ctx.get("reward_pct") or 0
    risk_pct = ctx.get("risk_pct") or 0
    rr = (reward_pct / risk_pct) if risk_pct > 0 else 0
    rr_pts = min(rr / 3.0 * p["W_RR"], p["W_RR"])

    reg_pts = regime_sc * p["W_REGIME"] if action == "BUY" else (1 - regime_sc) * p["W_REGIME"]

    total = round(strat_pts + rsi_pts + vol_pts + rr_pts + reg_pts, 1)
    breakdown = dict(
        strategy=round(strat_pts, 1),
        rsi=round(rsi_pts, 1),
        volume=round(vol_pts, 1),
        rr=round(rr_pts, 1),
        regime=round(reg_pts, 1),
    )
    return min(total, 100.0), breakdown

def score_label(s):
    if s >= 80:
        return "Very Strong"
    if s >= 65:
        return "Strong"
    if s >= 50:
        return "Good"
    if s >= 35:
        return "Moderate"
    return "Weak"

# ─────────────────────────────────────────────
#  DATA & CONTEXT
# ─────────────────────────────────────────────
def fetch(ticker, days=430):
    try:
        df = yf.download(
            ticker + ".NS",
            start=datetime.today() - timedelta(days=days),
            end=datetime.today(),
            progress=False,
            auto_adjust=True
        )
        if df.empty or len(df) < 80:
            return None, "insufficient_data"
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna(), "ok"
    except Exception as e:
        return None, str(e)[:120]

def context(df, p):
    c = df.Close
    if len(c) < 20:
        return sanitize_for_json({
            "price": None,
            "change_1d": None,
            "change_5d": None,
            "rsi": None,
            "macd_hist": None,
            "ema_bullish": None,
            "supertrend_up": None,
            "supertrend_line": None,
            "support": None,
            "resistance": None,
            "stop_loss": None,
            "target": None,
            "risk_pct": None,
            "reward_pct": None,
            "volume": 0,
            "avg_volume": 0,
        })

    r = float(rsi(c, p["RSI_PERIOD"]).iloc[-1])
    _, _, hist = macd(c, p["MACD_FAST"], p["MACD_SLOW"], p["MACD_SIGNAL"])
    e_s = float(ema(c, p["EMA_SHORT"]).iloc[-1])
    e_l = float(ema(c, p["EMA_LONG"]).iloc[-1])
    trend, st_line = supertrend(df.High, df.Low, c, p["ATR_PERIOD"], p["SUPERTREND_MULT"])
    price = float(c.iloc[-1])
    atr_now = float(atr(df.High, df.Low, c, p["ATR_PERIOD"]).iloc[-1])

    prev_1 = float(c.iloc[-2]) if len(c) >= 2 and pd.notna(c.iloc[-2]) else None
    prev_5 = float(c.iloc[-6]) if len(c) >= 6 and pd.notna(c.iloc[-6]) else None

    sl = round(price - 1.5 * atr_now, 2) if math.isfinite(price) and math.isfinite(atr_now) else None
    tgt = round(price + 3.0 * atr_now, 2) if math.isfinite(price) and math.isfinite(atr_now) else None

    low20 = df.Low.rolling(20).min().iloc[-1]
    high20 = df.High.rolling(20).max().iloc[-1]
    vol20 = df.Volume.rolling(20).mean().iloc[-1]

    out = dict(
        price=round(price, 2) if math.isfinite(price) else None,
        change_1d=round((price - prev_1) / prev_1 * 100, 2) if prev_1 not in [None, 0] and math.isfinite(prev_1) else None,
        change_5d=round((price - prev_5) / prev_5 * 100, 2) if prev_5 not in [None, 0] and math.isfinite(prev_5) else None,
        rsi=round(r, 1) if math.isfinite(r) else None,
        macd_hist=round(float(hist.iloc[-1]), 3) if pd.notna(hist.iloc[-1]) and math.isfinite(float(hist.iloc[-1])) else None,
        ema_bullish=bool(e_s > e_l) if math.isfinite(e_s) and math.isfinite(e_l) else None,
        supertrend_up=bool(trend.iloc[-1] == 1) if pd.notna(trend.iloc[-1]) else None,
        supertrend_line=round(float(st_line.iloc[-1]), 2) if pd.notna(st_line.iloc[-1]) and math.isfinite(float(st_line.iloc[-1])) else None,
        support=round(float(low20), 2) if pd.notna(low20) and math.isfinite(float(low20)) else None,
        resistance=round(float(high20), 2) if pd.notna(high20) and math.isfinite(float(high20)) else None,
        stop_loss=sl,
        target=tgt,
        risk_pct=round((price - sl) / price * 100, 2) if sl is not None and price not in [None, 0] and math.isfinite(price) else None,
        reward_pct=round((tgt - price) / price * 100, 2) if tgt is not None and price not in [None, 0] and math.isfinite(price) else None,
        volume=int(df.Volume.iloc[-1]) if pd.notna(df.Volume.iloc[-1]) else 0,
        avg_volume=int(vol20) if pd.notna(vol20) and math.isfinite(float(vol20)) else 0,
    )
    return sanitize_for_json(out)

def market_regime(p):
    try:
        df = yf.download("^NSEI", period="120d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 55:
            return "UNKNOWN", 0.5
        close = df["Close"].squeeze()
        e50 = float(ema(close, 50).iloc[-1])
        e20 = float(ema(close, 20).iloc[-1])
        r = float(rsi(close, p["RSI_PERIOD"]).iloc[-1])
        price = float(close.iloc[-1])

        if price > e20 > e50 and r > 50:
            return "BULLISH", 1.0
        if price < e20 < e50 and r < 50:
            return "BEARISH", 0.0
        return "NEUTRAL", 0.5
    except Exception:
        return "UNKNOWN", 0.5

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run():
    today = datetime.today().strftime("%Y-%m-%d")
    print(f"\n🇮🇳 Indian Stock Agent v4 — {today}")

    P, param_version = load_active_params()
    print(f"   Params: {param_version}")
    print(f"   Scanning {len(ALL_TICKERS)} NSE stocks...\n")

    STRATEGIES = get_strategies(P)

    print("   Checking NIFTY market regime...")
    regime_label, regime_sc = market_regime(P)
    print(f"   Regime: {regime_label}  (score={regime_sc})\n")

    supabase.table("recommendations").delete().eq("date", today).execute()
    supabase.table("ticker_run_log").delete().eq("date", today).execute()

    records, run_logs = [], []
    gate_counts = {"fetched": 0, "any_signal": 0, "passed_weight": 0}

    for i, ticker in enumerate(ALL_TICKERS, 1):
        sys.stdout.write(f"\r  {i}/{len(ALL_TICKERS)}  {ticker:<15}")
        sys.stdout.flush()

        df, status = fetch(ticker)
        run_logs.append(sanitize_for_json(dict(date=today, ticker=ticker, status=status)))
        if df is None:
            continue
        gate_counts["fetched"] += 1

        try:
            today_sigs = {name: int(fn(df).iloc[-1]) for name, fn in STRATEGIES.items()}
            buy_count = sum(1 for v in today_sigs.values() if v == 1)
            sell_count = sum(1 for v in today_sigs.values() if v == -1)

            if buy_count == 0 and sell_count == 0:
                continue
            gate_counts["any_signal"] += 1

            bt = {name: backtest(df, fn(df), P) for name, fn in STRATEGIES.items()}
            action = "BUY" if buy_count >= sell_count else "SELL"
            w_ratio, weights = weighted_vote(today_sigs, bt, action)

            if w_ratio < P["MIN_WEIGHTED_SCORE"]:
                continue
            gate_counts["passed_weight"] += 1

            ctx = context(df, P)
            c_score, c_breakdown = composite_score(today_sigs, bt, ctx, regime_sc, action, P)

            active = [
                n for n, v in today_sigs.items()
                if (v == 1 and action == "BUY") or (v == -1 and action == "SELL")
            ]

            agg = lambda k: round(float(np.mean([bt[n][k] for n in active])), 2) if active else 0
            low_smp = int(np.mean([bt[n]["trades"] for n in active])) < 5 if active else True

            record = dict(
                date=today,
                ticker=ticker,
                action=action,
                raw_score=int(buy_count if action == "BUY" else sell_count),
                weighted_score_val=float(w_ratio) if w_ratio is not None else 0.0,
                composite_score=float(c_score) if c_score is not None else 0.0,
                score_label=score_label(c_score if c_score is not None else 0),
                score_breakdown=json.dumps(sanitize_for_json(c_breakdown)),
                signals=json.dumps(sanitize_for_json(today_sigs)),
                strategy_weights=json.dumps(sanitize_for_json(weights)),
                backtest=json.dumps(sanitize_for_json(bt)),
                active_strategies=", ".join(active),
                low_sample_warning=bool(low_smp),
                win_rate=float(agg("win_rate")) if active else 0.0,
                avg_return=float(agg("avg_return")) if active else 0.0,
                median_return=float(agg("median_return")) if active else 0.0,
                profit_factor=float(agg("profit_factor")) if active else 0.0,
                max_drawdown=float(agg("max_drawdown")) if active else 0.0,
                avg_trades=int(round(np.mean([bt[n]["trades"] for n in active]))) if active else 0,
                market_regime=regime_label,
                param_version=param_version,
                **ctx,
            )

            safe_record = sanitize_for_json(record)
            json.dumps(safe_record)
            records.append(safe_record)

        except Exception as e:
            print(f"\n  Skipping {ticker} due to processing error: {e}")

        time.sleep(0.1)

    sys.stdout.write("\r" + " " * 60 + "\r")
    failed_count = sum(1 for l in run_logs if l['status'] != 'ok')
    print(f"  Pipeline summary:")
    print(f"    Tickers scanned  : {len(ALL_TICKERS)}")
    print(f"    Data fetched OK  : {gate_counts['fetched']}")
    print(f"    Any signal fired : {gate_counts['any_signal']}")
    print(f"    Passed weight    : {gate_counts['passed_weight']}  (MIN_WEIGHTED_SCORE={P['MIN_WEIGHTED_SCORE']})")
    print(f"    Final signals    : {len(records)}")
    print(f"    Failed fetches   : {failed_count}\n")

    if records:
        print(f"  Saving {len(records)} recommendations to Supabase...")
        saved = 0
        for i in range(0, len(records), 20):
            batch = records[i:i + 20]
            try:
                supabase.table("recommendations").insert(batch).execute()
                saved += len(batch)
            except Exception as e:
                print(f"  INSERT ERROR batch {i//20 + 1}: {e}")
                print(f"  First record keys: {list(batch[0].keys())}")
        print(f"  Saved {saved}/{len(records)} recommendations")
    else:
        print(f"  No signals met threshold (MIN_WEIGHTED_SCORE={P['MIN_WEIGHTED_SCORE']})")
        print(f"  Scanned {len(ALL_TICKERS)} tickers, 0 passed filter")

    try:
        for i in range(0, len(run_logs), 50):
            supabase.table("ticker_run_log").insert(run_logs[i:i + 50]).execute()
    except Exception as e:
        print(f"  Run log insert failed: {e}")

    try:
        supabase.table("agent_meta").upsert(sanitize_for_json({
            "id": 1,
            "last_run": today,
            "total_signals": len(records),
            "tickers_scanned": len(ALL_TICKERS),
            "failed": sum(1 for l in run_logs if l["status"] != "ok"),
            "market_regime": regime_label,
            "active_param_version": param_version,
        })).execute()
    except Exception as e:
        print(f"  Meta upsert failed: {e}")

    print("  Done ✅\n")

if __name__ == "__main__":
    run()#!/usr/bin/env python3
"""
============================================================
  Indian Stock Market Analysis Agent  —  v4
  Runs daily via GitHub Actions at 7:00 AM IST

  v4 live setup:
  ─ Uses only 3 live strategies:
      • EMA Crossover
      • RSI + MACD
      • Bollinger
  ─ Lowers MIN_WEIGHTED_SCORE so single-strategy signals
    like Bollinger can survive more often
  ─ Keeps Supertrend only for context display
============================================================
"""

import os
import sys
import time
import json
import warnings
import math
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
from supabase import create_client, Client

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
    "RSI_OVERSOLD":       42,
    "RSI_OVERBOUGHT":     62,
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
    "MIN_WEIGHTED_SCORE": 0.15,
    "W_STRATEGY":         40,
    "W_RSI":              20,
    "W_VOLUME":           15,
    "W_RR":               15,
    "W_REGIME":           10,
}

# ─────────────────────────────────────────────
#  LOAD ACTIVE PARAMS
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
            row = res.data[0]
            params = json.loads(row["params_json"]) if isinstance(row["params_json"], str) else row["params_json"]
            merged = {**DEFAULT_PARAMS, **params}
            return merged, f"v{row['version']} (score={row['objective_score']:.3f})"
    except Exception as e:
        print(f"  ⚠️ Could not load champion params: {e}")
    return DEFAULT_PARAMS.copy(), "defaults"

# ─────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────
def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def rsi(s, p=14):
    d = s.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / p, min_periods=p, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1 / p, min_periods=p, adjust=False).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))

def macd(s, fast, slow, sig):
    ml = ema(s, fast) - ema(s, slow)
    sl = ema(ml, sig)
    return ml, sl, ml - sl

def bollinger(s, p, std):
    m = s.rolling(p).mean()
    sg = s.rolling(p).std()
    return m + std * sg, m, m - std * sg

def atr(h, l, c, p):
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / p, min_periods=p, adjust=False).mean()

def supertrend(h, l, c, p, mult):
    atr_v = atr(h, l, c, p)
    hl2 = (h + l) / 2
    up = hl2 + mult * atr_v
    dn = hl2 - mult * atr_v
    trend = pd.Series(1, index=c.index)
    fu, fl = up.copy(), dn.copy()

    for i in range(1, len(c)):
        fu.iloc[i] = up.iloc[i] if (up.iloc[i] < fu.iloc[i - 1] or c.iloc[i - 1] > fu.iloc[i - 1]) else fu.iloc[i - 1]
        fl.iloc[i] = dn.iloc[i] if (dn.iloc[i] > fl.iloc[i - 1] or c.iloc[i - 1] < fl.iloc[i - 1]) else fl.iloc[i - 1]
        if trend.iloc[i - 1] == -1 and c.iloc[i] > fu.iloc[i]:
            trend.iloc[i] = 1
        elif trend.iloc[i - 1] == 1 and c.iloc[i] < fl.iloc[i]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]

    return trend, pd.Series(np.where(trend == 1, fl, fu), index=c.index)

# ─────────────────────────────────────────────
#  SIGNAL GENERATORS
# ─────────────────────────────────────────────
def sig_ema(df, p):
    e_s = ema(df.Close, p["EMA_SHORT"])
    e_l = ema(df.Close, p["EMA_LONG"])
    s = pd.Series(0, index=df.index)
    s[(e_s > e_l) & (e_s.shift() <= e_l.shift())] = 1
    s[(e_s < e_l) & (e_s.shift() >= e_l.shift())] = -1
    return s

def sig_rsi_macd(df, p):
    r = rsi(df.Close, p["RSI_PERIOD"])
    _, _, hist = macd(df.Close, p["MACD_FAST"], p["MACD_SLOW"], p["MACD_SIGNAL"])
    s = pd.Series(0, index=df.index)
    s[(r < p["RSI_OVERSOLD"]) & (hist > 0) & (hist.shift() <= 0)] = 1
    s[(r > p["RSI_OVERBOUGHT"]) & (hist < 0) & (hist.shift() >= 0)] = -1
    return s

def sig_bb(df, p):
    up, _, lo = bollinger(df.Close, p["BB_PERIOD"], p["BB_STD"])
    r = rsi(df.Close, p["RSI_PERIOD"])
    s = pd.Series(0, index=df.index)
    s[(df.Low <= lo) & (df.Close > lo) & (r < 50)] = 1
    s[(df.High >= up) & (df.Close < up) & (r > 50)] = -1
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
    period = int(p.get("DONCHIAN_PERIOD", 20))
    vol_mult = float(p.get("VOLUME_MULT", 1.5))
    hh = df.High.rolling(period).max()
    ll = df.Low.rolling(period).min()
    avg_vol = df.Volume.rolling(20).mean()
    s = pd.Series(0, index=df.index)

    buy_cond = (
        (df.Close > hh.shift(1)) &
        (df.Close.shift(1) <= hh.shift(1)) &
        (df.Volume > avg_vol * vol_mult)
    )
    sell_cond = (
        (df.Close < ll.shift(1)) &
        (df.Close.shift(1) >= ll.shift(1)) &
        (df.Volume > avg_vol * vol_mult)
    )

    s[buy_cond] = 1
    s[sell_cond] = -1
    return s

def sig_rsi_trend_shift(df, p):
    r = rsi(df.Close, p["RSI_PERIOD"])
    e_l = ema(df.Close, p["EMA_LONG"])
    mid = float(p.get("RSI_MIDLINE", 50))
    s = pd.Series(0, index=df.index)

    buy_cond = (
        (df.Close > e_l) &
        (r > mid) &
        (r.shift(1) <= mid)
    )
    sell_cond = (
        (df.Close < e_l) &
        (r < mid) &
        (r.shift(1) >= mid)
    )

    s[buy_cond] = 1
    s[sell_cond] = -1
    return s

def get_strategies(p):
    return {
        "EMA Crossover":    lambda df: sig_ema(df, p),
        "RSI + MACD":       lambda df: sig_rsi_macd(df, p),
        "Bollinger":        lambda df: sig_bb(df, p),
        "Donchian":         lambda df: sig_donchian(df, p),
        "Volume Breakout":  lambda df: sig_volume_breakout(df, p),
        "RSI Trend Shift":  lambda df: sig_rsi_trend_shift(df, p),
    }

# ─────────────────────────────────────────────
#  BACKTEST
# ─────────────────────────────────────────────
def backtest(df, signals, p):
    trades, reasons = [], []
    in_t, ep, entry_idx = False, 0.0, -1
    closes, highs, lows, opens = df.Close.values, df.High.values, df.Low.values, df.Open.values
    sl_pct = p["BT_SL_PCT"]
    tgt_pct = p["BT_TARGET_PCT"]
    max_hold = p["BT_MAX_HOLD"]

    for i in range(1, len(df)):
        if in_t:
            sl_px = ep * (1 - sl_pct / 100)
            tgt_px = ep * (1 + tgt_pct / 100)
            if lows[i] <= sl_px:
                trades.append((sl_px - ep) / ep * 100)
                reasons.append("sl")
                in_t = False
            elif highs[i] >= tgt_px:
                trades.append((tgt_px - ep) / ep * 100)
                reasons.append("target")
                in_t = False
            elif i >= entry_idx + max_hold:
                trades.append((closes[i] - ep) / ep * 100)
                reasons.append("timeout")
                in_t = False

        if signals.iloc[i] == 1 and not in_t:
            ep, entry_idx, in_t = opens[i], i, True

    if not trades:
        return dict(
            win_rate=0, avg_return=0, median_return=0, trades=0,
            profit_factor=0, max_drawdown=0, sl_exits=0, target_exits=0, timeout_exits=0
        )

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    gp, gl = sum(wins), abs(sum(losses))
    pf = round(gp / gl, 2) if gl > 0 else 99.0
    eq = np.cumsum(trades)
    peak = np.maximum.accumulate(eq)
    max_dd = round(float(abs((eq - peak).min())), 2)

    return dict(
        win_rate=round(len(wins) / len(trades) * 100, 1),
        avg_return=round(float(np.mean(trades)), 2),
        median_return=round(float(np.median(trades)), 2),
        trades=len(trades),
        profit_factor=min(float(pf), 99.0),
        max_drawdown=max_dd,
        sl_exits=reasons.count("sl"),
        target_exits=reasons.count("target"),
        timeout_exits=reasons.count("timeout"),
    )

# ─────────────────────────────────────────────
#  JSON SANITIZER
# ─────────────────────────────────────────────
def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj

# ─────────────────────────────────────────────
#  WEIGHTED VOTE + COMPOSITE SCORE
# ─────────────────────────────────────────────
def weighted_vote(today_sigs, bt_results, action):
    weights = {}
    for name, bt in bt_results.items():
        wr = bt.get("win_rate", 50) / 100
        n = bt.get("trades", 0)
        weights[name] = round(wr * min(n / 8.0, 1.0), 3)

    total_w = sum(weights.values()) or 1e-9
    target_v = 1 if action == "BUY" else -1
    signal_w = sum(weights.get(n, 0) for n, v in today_sigs.items() if v == target_v)
    return round(signal_w / total_w, 3), weights

def composite_score(today_sigs, bt_results, ctx, regime_sc, action, p):
    w_ratio, _ = weighted_vote(today_sigs, bt_results, action)
    strat_pts = round(w_ratio * p["W_STRATEGY"], 2)

    r = ctx.get("rsi")
    if r is None:
        rsi_pts = 0
    elif action == "BUY":
        rsi_pts = max(0, min((60 - r) / 40 * p["W_RSI"], p["W_RSI"]))
    else:
        rsi_pts = max(0, min((r - 40) / 40 * p["W_RSI"], p["W_RSI"]))

    avg_volume = ctx.get("avg_volume") or 0
    volume = ctx.get("volume") or 0
    vol_ratio = (volume / avg_volume) if avg_volume > 0 else 1.0
    vol_pts = min(vol_ratio / 2.0 * p["W_VOLUME"], p["W_VOLUME"])

    reward_pct = ctx.get("reward_pct") or 0
    risk_pct = ctx.get("risk_pct") or 0
    rr = (reward_pct / risk_pct) if risk_pct > 0 else 0
    rr_pts = min(rr / 3.0 * p["W_RR"], p["W_RR"])

    reg_pts = regime_sc * p["W_REGIME"] if action == "BUY" else (1 - regime_sc) * p["W_REGIME"]

    total = round(strat_pts + rsi_pts + vol_pts + rr_pts + reg_pts, 1)
    breakdown = dict(
        strategy=round(strat_pts, 1),
        rsi=round(rsi_pts, 1),
        volume=round(vol_pts, 1),
        rr=round(rr_pts, 1),
        regime=round(reg_pts, 1),
    )
    return min(total, 100.0), breakdown

def score_label(s):
    if s >= 80:
        return "Very Strong"
    if s >= 65:
        return "Strong"
    if s >= 50:
        return "Good"
    if s >= 35:
        return "Moderate"
    return "Weak"

# ─────────────────────────────────────────────
#  DATA & CONTEXT
# ─────────────────────────────────────────────
def fetch(ticker, days=430):
    try:
        df = yf.download(
            ticker + ".NS",
            start=datetime.today() - timedelta(days=days),
            end=datetime.today(),
            progress=False,
            auto_adjust=True
        )
        if df.empty or len(df) < 80:
            return None, "insufficient_data"
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna(), "ok"
    except Exception as e:
        return None, str(e)[:120]

def context(df, p):
    c = df.Close
    if len(c) < 20:
        return sanitize_for_json({
            "price": None,
            "change_1d": None,
            "change_5d": None,
            "rsi": None,
            "macd_hist": None,
            "ema_bullish": None,
            "supertrend_up": None,
            "supertrend_line": None,
            "support": None,
            "resistance": None,
            "stop_loss": None,
            "target": None,
            "risk_pct": None,
            "reward_pct": None,
            "volume": 0,
            "avg_volume": 0,
        })

    r = float(rsi(c, p["RSI_PERIOD"]).iloc[-1])
    _, _, hist = macd(c, p["MACD_FAST"], p["MACD_SLOW"], p["MACD_SIGNAL"])
    e_s = float(ema(c, p["EMA_SHORT"]).iloc[-1])
    e_l = float(ema(c, p["EMA_LONG"]).iloc[-1])
    trend, st_line = supertrend(df.High, df.Low, c, p["ATR_PERIOD"], p["SUPERTREND_MULT"])
    price = float(c.iloc[-1])
    atr_now = float(atr(df.High, df.Low, c, p["ATR_PERIOD"]).iloc[-1])

    prev_1 = float(c.iloc[-2]) if len(c) >= 2 and pd.notna(c.iloc[-2]) else None
    prev_5 = float(c.iloc[-6]) if len(c) >= 6 and pd.notna(c.iloc[-6]) else None

    sl = round(price - 1.5 * atr_now, 2) if math.isfinite(price) and math.isfinite(atr_now) else None
    tgt = round(price + 3.0 * atr_now, 2) if math.isfinite(price) and math.isfinite(atr_now) else None

    low20 = df.Low.rolling(20).min().iloc[-1]
    high20 = df.High.rolling(20).max().iloc[-1]
    vol20 = df.Volume.rolling(20).mean().iloc[-1]

    out = dict(
        price=round(price, 2) if math.isfinite(price) else None,
        change_1d=round((price - prev_1) / prev_1 * 100, 2) if prev_1 not in [None, 0] and math.isfinite(prev_1) else None,
        change_5d=round((price - prev_5) / prev_5 * 100, 2) if prev_5 not in [None, 0] and math.isfinite(prev_5) else None,
        rsi=round(r, 1) if math.isfinite(r) else None,
        macd_hist=round(float(hist.iloc[-1]), 3) if pd.notna(hist.iloc[-1]) and math.isfinite(float(hist.iloc[-1])) else None,
        ema_bullish=bool(e_s > e_l) if math.isfinite(e_s) and math.isfinite(e_l) else None,
        supertrend_up=bool(trend.iloc[-1] == 1) if pd.notna(trend.iloc[-1]) else None,
        supertrend_line=round(float(st_line.iloc[-1]), 2) if pd.notna(st_line.iloc[-1]) and math.isfinite(float(st_line.iloc[-1])) else None,
        support=round(float(low20), 2) if pd.notna(low20) and math.isfinite(float(low20)) else None,
        resistance=round(float(high20), 2) if pd.notna(high20) and math.isfinite(float(high20)) else None,
        stop_loss=sl,
        target=tgt,
        risk_pct=round((price - sl) / price * 100, 2) if sl is not None and price not in [None, 0] and math.isfinite(price) else None,
        reward_pct=round((tgt - price) / price * 100, 2) if tgt is not None and price not in [None, 0] and math.isfinite(price) else None,
        volume=int(df.Volume.iloc[-1]) if pd.notna(df.Volume.iloc[-1]) else 0,
        avg_volume=int(vol20) if pd.notna(vol20) and math.isfinite(float(vol20)) else 0,
    )
    return sanitize_for_json(out)

def market_regime(p):
    try:
        df = yf.download("^NSEI", period="120d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 55:
            return "UNKNOWN", 0.5
        close = df["Close"].squeeze()
        e50 = float(ema(close, 50).iloc[-1])
        e20 = float(ema(close, 20).iloc[-1])
        r = float(rsi(close, p["RSI_PERIOD"]).iloc[-1])
        price = float(close.iloc[-1])

        if price > e20 > e50 and r > 50:
            return "BULLISH", 1.0
        if price < e20 < e50 and r < 50:
            return "BEARISH", 0.0
        return "NEUTRAL", 0.5
    except Exception:
        return "UNKNOWN", 0.5

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run():
    today = datetime.today().strftime("%Y-%m-%d")
    print(f"\n🇮🇳 Indian Stock Agent v4 — {today}")

    P, param_version = load_active_params()
    print(f"   Params: {param_version}")
    print(f"   Scanning {len(ALL_TICKERS)} NSE stocks...\n")

    STRATEGIES = get_strategies(P)

    print("   Checking NIFTY market regime...")
    regime_label, regime_sc = market_regime(P)
    print(f"   Regime: {regime_label}  (score={regime_sc})\n")

    supabase.table("recommendations").delete().eq("date", today).execute()
    supabase.table("ticker_run_log").delete().eq("date", today).execute()

    records, run_logs = [], []

    for i, ticker in enumerate(ALL_TICKERS, 1):
        sys.stdout.write(f"\r  {i}/{len(ALL_TICKERS)}  {ticker:<15}")
        sys.stdout.flush()

        df, status = fetch(ticker)
        run_logs.append(sanitize_for_json(dict(date=today, ticker=ticker, status=status)))
        if df is None:
            continue

        try:
            today_sigs = {name: int(fn(df).iloc[-1]) for name, fn in STRATEGIES.items()}
            buy_count = sum(1 for v in today_sigs.values() if v == 1)
            sell_count = sum(1 for v in today_sigs.values() if v == -1)

            if buy_count == 0 and sell_count == 0:
                continue

            bt = {name: backtest(df, fn(df), P) for name, fn in STRATEGIES.items()}
            action = "BUY" if buy_count >= sell_count else "SELL"
            w_ratio, weights = weighted_vote(today_sigs, bt, action)

            if w_ratio < P["MIN_WEIGHTED_SCORE"]:
                continue

            ctx = context(df, P)
            c_score, c_breakdown = composite_score(today_sigs, bt, ctx, regime_sc, action, P)

            active = [
                n for n, v in today_sigs.items()
                if (v == 1 and action == "BUY") or (v == -1 and action == "SELL")
            ]

            agg = lambda k: round(float(np.mean([bt[n][k] for n in active])), 2) if active else 0
            low_smp = int(np.mean([bt[n]["trades"] for n in active])) < 5 if active else True

            record = dict(
                date=today,
                ticker=ticker,
                action=action,
                raw_score=int(buy_count if action == "BUY" else sell_count),
                weighted_score_val=float(w_ratio) if w_ratio is not None else 0.0,
                composite_score=float(c_score) if c_score is not None else 0.0,
                score_label=score_label(c_score if c_score is not None else 0),
                score_breakdown=json.dumps(sanitize_for_json(c_breakdown)),
                signals=json.dumps(sanitize_for_json(today_sigs)),
                strategy_weights=json.dumps(sanitize_for_json(weights)),
                backtest=json.dumps(sanitize_for_json(bt)),
                active_strategies=", ".join(active),
                low_sample_warning=bool(low_smp),
                win_rate=float(agg("win_rate")) if active else 0.0,
                avg_return=float(agg("avg_return")) if active else 0.0,
                median_return=float(agg("median_return")) if active else 0.0,
                profit_factor=float(agg("profit_factor")) if active else 0.0,
                max_drawdown=float(agg("max_drawdown")) if active else 0.0,
                avg_trades=int(round(np.mean([bt[n]["trades"] for n in active]))) if active else 0,
                market_regime=regime_label,
                param_version=param_version,
                **ctx,
            )

            safe_record = sanitize_for_json(record)
            json.dumps(safe_record)
            records.append(safe_record)

        except Exception as e:
            print(f"\n  Skipping {ticker} due to processing error: {e}")

        time.sleep(0.1)

    sys.stdout.write("\r" + " " * 60 + "\r")
    print(f"  ✅ {len(records)} signals  |  ❌ {sum(1 for l in run_logs if l['status'] != 'ok')} failed\n")

    if records:
        print(f"  Saving {len(records)} recommendations to Supabase...")
        saved = 0
        for i in range(0, len(records), 20):
            batch = records[i:i + 20]
            try:
                supabase.table("recommendations").insert(batch).execute()
                saved += len(batch)
            except Exception as e:
                print(f"  INSERT ERROR batch {i//20 + 1}: {e}")
                print(f"  First record keys: {list(batch[0].keys())}")
        print(f"  Saved {saved}/{len(records)} recommendations")
    else:
        print(f"  No signals met threshold (MIN_WEIGHTED_SCORE={P['MIN_WEIGHTED_SCORE']})")
        print(f"  Scanned {len(ALL_TICKERS)} tickers, 0 passed filter")

    try:
        for i in range(0, len(run_logs), 50):
            supabase.table("ticker_run_log").insert(run_logs[i:i + 50]).execute()
    except Exception as e:
        print(f"  Run log insert failed: {e}")

    try:
        supabase.table("agent_meta").upsert(sanitize_for_json({
            "id": 1,
            "last_run": today,
            "total_signals": len(records),
            "tickers_scanned": len(ALL_TICKERS),
            "failed": sum(1 for l in run_logs if l["status"] != "ok"),
            "market_regime": regime_label,
            "active_param_version": param_version,
        })).execute()
    except Exception as e:
        print(f"  Meta upsert failed: {e}")

    print("  Done ✅\n")

if __name__ == "__main__":
    run()
