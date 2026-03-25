#!/usr/bin/env python3
"""
============================================================
  Indian Stock Market Analysis Agent
  Runs daily via GitHub Actions at 7:00 AM IST
  Writes recommendations → Supabase
  Exchange: NSE
============================================================
"""

import os
import sys
import time
import warnings
import json
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
from supabase import create_client, Client

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  SUPABASE  (credentials via GitHub Secrets)
# ─────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────────
#  WATCHLIST  (Nifty 50 + extras)
# ─────────────────────────────────────────────
NIFTY50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
    "SUNPHARMA", "ULTRACEMCO", "WIPRO", "ONGC", "NTPC",
    "POWERGRID", "COALINDIA", "BAJFINANCE", "HCLTECH", "TECHM",
    "M&M", "NESTLEIND", "TCMV", "TATASTEEL",
    "JSWSTEEL", "HINDALCO", "BAJAJFINSV", "GRASIM",
    "CIPLA", "DRREDDY", "EICHERMOT", "HEROMOTOCO", "BRITANNIA",
    "APOLLOHOSP", "TATACONSUM", "ADANIPORTS", "BPCL",
    "INDUSINDBK", "SBILIFE", "HDFCLIFE",

    "360ONE", "ABB", "ACC", "APLAPOLLO", "AUBANK",
    "ADANIENSOL", "ADANIENT", "ADANIGREEN", "ADANIPOWER", "ATGL",
    "ABCAPITAL", "ALKEM", "AMBUJACEM", "ASHOKLEY", "ASTRAL",
    "AUROPHARMA", "DMART", "BSE", "BAJAJ-AUTO", "BAJAJHLDNG",
    "BAJAJHFL", "BANKBARODA", "BANKINDIA", "BDL", "BEL",
    "BHARATFORG", "BHEL", "BHARTIHEXA", "BIOCON", "BLUESTARCO",
    "BOSCHLTD", "CGPOWER", "CANBK", "CHOLAFIN", "COCHINSHIP",
    "COFORGE", "COLPAL", "CONCOR", "COROMANDEL", "CUMMINSIND",
    "DLF", "DABUR", "DIVISLAB", "DIXON", "ETERNAL",
    "EXIDEIND", "NYKAA", "FEDERALBNK", "FORTIS", "GAIL",
    "GMRAIRPORT", "GLENMARK", "GODFRYPHLP", "GODREJCP", "GODREJPROP",
    "HDFCAMC", "HAVELLS", "HAL", "HINDPETRO", "HINDZINC",
    "POWERINDIA", "HUDCO", "HYUNDAI", "ICICIGI", "IDFCFIRSTB",
    "IRB", "ITCHOTELS", "INDIANB", "INDHOTEL", "IOC",
    "IRCTC", "IRFC", "IREDA", "IGL", "INDUSTOWER",
    "NAUKRI", "INDIGO", "JSWENERGY", "JINDALSTEL", "JIOFIN",
    "JUBLFOOD", "KEI", "KPITTECH", "KALYANKJIL", "LTF",
    "LICHSGFIN", "LTM", "LICI", "LODHA", "LUPIN",
    "MRF", "M&MFIN", "MANKIND", "MARICO", "MFSL",
    "MAXHEALTH", "MAZDOCK", "MOTILALOFS", "MPHASIS", "MUTHOOTFIN",
    "NHPC", "NMDC", "NTPCGREEN", "NATIONALUM", "OBEROIRLTY",
    "OIL", "PAYTM", "OFSS", "POLICYBZR", "PIIND",
    "PAGEIND", "PATANJALI", "PERSISTENT", "PHOENIXLTD", "PIDILITIND",
    "POLYCAB", "PFC", "PREMIERENE", "PRESTIGE", "PNB",
    "RECLTD", "RVNL", "SBICARD", "SRF", "MOTHERSON",
    "SHREECEM", "SHRIRAMFIN", "ENRIN", "SIEMENS", "SOLARINDS",
    "SONACOMS", "SAIL", "SUPREMEIND", "SUZLON", "SWIGGY",
    "TVSMOTOR", "TATACOMM", "TATAELXSI", "TMPV", "TATAPOWER",
    "TATATECH", "TORNTPHARM", "TORNTPOWER", "TRENT", "TIINDIA",
    "UPL", "UNIONBANK", "UNITDSPR", "VBL", "VEDL",
    "VMM", "IDEA", "VOLTAS", "WAAREEENER", "YESBANK", "ZYDUSLIFE"
]

# Add your own tickers here (no .NS needed):
EXTRA_WATCHLIST = [
    # "IRCTC", "ZOMATO", "IRFC", "DELHIVERY"
]

ALL_TICKERS = NIFTY50 + EXTRA_WATCHLIST

# ─────────────────────────────────────────────
#  STRATEGY PARAMETERS
# ─────────────────────────────────────────────
EMA_SHORT       = 9
EMA_LONG        = 21
RSI_PERIOD      = 14
RSI_OVERSOLD    = 40
RSI_OVERBOUGHT  = 65
MACD_FAST       = 12
MACD_SLOW       = 26
MACD_SIGNAL     = 9
BB_PERIOD       = 20
BB_STD          = 2.0
ATR_PERIOD      = 14
SUPERTREND_MULT = 3.0
HOLD_DAYS       = 10          # backtest holding period
MIN_BUY_SCORE   = 1           # minimum strategies agreeing
MIN_SELL_SCORE  = 1

# ─────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────

def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0)
    l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    al = l.ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(s, fast=12, slow=26, signal=9):
    ml = ema(s, fast) - ema(s, slow)
    sl = ema(ml, signal)
    return ml, sl, ml - sl

def bollinger(s, p=20, std=2.0):
    m = s.rolling(p).mean()
    sg = s.rolling(p).std()
    return m + std*sg, m, m - std*sg

def atr(h, l, c, p=14):
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, min_periods=p, adjust=False).mean()

def supertrend(h, l, c, p=14, mult=3.0):
    atr_v = atr(h, l, c, p)
    hl2   = (h + l) / 2
    up    = hl2 + mult * atr_v
    dn    = hl2 - mult * atr_v
    trend = pd.Series(1, index=c.index)
    fu, fl = up.copy(), dn.copy()
    for i in range(1, len(c)):
        fu.iloc[i] = up.iloc[i] if (up.iloc[i] < fu.iloc[i-1] or c.iloc[i-1] > fu.iloc[i-1]) else fu.iloc[i-1]
        fl.iloc[i] = dn.iloc[i] if (dn.iloc[i] > fl.iloc[i-1] or c.iloc[i-1] < fl.iloc[i-1]) else fl.iloc[i-1]
        if   trend.iloc[i-1] == -1 and c.iloc[i] > fu.iloc[i]:  trend.iloc[i] =  1
        elif trend.iloc[i-1] ==  1 and c.iloc[i] < fl.iloc[i]:  trend.iloc[i] = -1
        else:                                                      trend.iloc[i] = trend.iloc[i-1]
    return trend, pd.Series(np.where(trend==1, fl, fu), index=c.index)

# ─────────────────────────────────────────────
#  SIGNAL GENERATORS
# ─────────────────────────────────────────────

def sig_ema(df):
    e9, e21 = ema(df.Close, EMA_SHORT), ema(df.Close, EMA_LONG)
    s = pd.Series(0, index=df.index)
    s[(e9 > e21) & (e9.shift() <= e21.shift())]  =  1
    s[(e9 < e21) & (e9.shift() >= e21.shift())]  = -1
    return s

def sig_rsi_macd(df):
    r = rsi(df.Close, RSI_PERIOD)
    _, _, hist = macd(df.Close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    s = pd.Series(0, index=df.index)
    s[(r < RSI_OVERSOLD)   & (hist > 0) & (hist.shift() <= 0)] =  1
    s[(r > RSI_OVERBOUGHT) & (hist < 0) & (hist.shift() >= 0)] = -1
    return s

def sig_bb(df):
    up, mid, lo = bollinger(df.Close, BB_PERIOD, BB_STD)
    r = rsi(df.Close, RSI_PERIOD)
    s = pd.Series(0, index=df.index)
    s[(df.Low <= lo) & (df.Close > lo) & (r < 50)] =  1
    s[(df.High >= up) & (df.Close < up) & (r > 50)] = -1
    return s

def sig_supertrend(df):
    trend, _ = supertrend(df.High, df.Low, df.Close, ATR_PERIOD, SUPERTREND_MULT)
    s = pd.Series(0, index=df.index)
    s[(trend ==  1) & (trend.shift() == -1)] =  1
    s[(trend == -1) & (trend.shift() ==  1)] = -1
    return s

STRATEGIES = {
    "EMA Crossover": sig_ema,
    "RSI + MACD":    sig_rsi_macd,
    "Bollinger":     sig_bb,
    "Supertrend":    sig_supertrend,
}

# ─────────────────────────────────────────────
#  BACKTESTER
# ─────────────────────────────────────────────

def backtest(df, signals, hold=HOLD_DAYS):
    trades, in_t, ep, ex = [], False, 0.0, -1
    closes, opens = df.Close.values, df.Open.values
    for i in range(1, len(df)):
        if in_t and i >= ex:
            trades.append((closes[i] - ep) / ep * 100)
            in_t = False
        if signals.iloc[i] == 1 and not in_t:
            ep, ex, in_t = opens[i], i + hold, True
    if not trades:
        return dict(win_rate=0, avg_return=0, trades=0)
    wins = [t for t in trades if t > 0]
    return dict(
        win_rate=round(len(wins)/len(trades)*100, 1),
        avg_return=round(float(np.mean(trades)), 2),
        trades=len(trades),
    )

# ─────────────────────────────────────────────
#  DATA FETCH
# ─────────────────────────────────────────────

def fetch(ticker, days=430):
    try:
        df = yf.download(
            ticker + ".NS",
            start=datetime.today() - timedelta(days=days),
            end=datetime.today(),
            progress=False, auto_adjust=True
        )
        if df.empty or len(df) < 60:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except Exception:
        return None

# ─────────────────────────────────────────────
#  CONTEXT
# ─────────────────────────────────────────────

def context(df):
    c = df.Close
    r = rsi(c, RSI_PERIOD).iloc[-1]
    _, _, hist = macd(c, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    e9  = ema(c, EMA_SHORT).iloc[-1]
    e21 = ema(c, EMA_LONG).iloc[-1]
    trend, st_line = supertrend(df.High, df.Low, c, ATR_PERIOD, SUPERTREND_MULT)
    up, _, lo = bollinger(c, BB_PERIOD, BB_STD)

    # Support / Resistance (20-day low/high)
    support    = round(float(df.Low.rolling(20).min().iloc[-1]), 2)
    resistance = round(float(df.High.rolling(20).max().iloc[-1]), 2)
    price      = float(c.iloc[-1])

    # Suggested SL and Target for swing trade
    atr_now = float(atr(df.High, df.Low, c, ATR_PERIOD).iloc[-1])
    stop_loss = round(price - 1.5 * atr_now, 2)
    target    = round(price + 3.0 * atr_now, 2)     # 2:1 reward/risk
    risk_pct  = round((price - stop_loss) / price * 100, 2)
    reward_pct= round((target - price)    / price * 100, 2)

    return dict(
        price       = round(price, 2),
        change_1d   = round((c.iloc[-1]-c.iloc[-2])/c.iloc[-2]*100, 2),
        change_5d   = round((c.iloc[-1]-c.iloc[-6])/c.iloc[-6]*100, 2),
        rsi         = round(float(r), 1),
        macd_hist   = round(float(hist.iloc[-1]), 3),
        ema_bullish = bool(e9 > e21),
        supertrend_up = bool(trend.iloc[-1] == 1),
        support     = support,
        resistance  = resistance,
        stop_loss   = stop_loss,
        target      = target,
        risk_pct    = risk_pct,
        reward_pct  = reward_pct,
        volume      = int(df.Volume.iloc[-1]),
        avg_volume  = int(df.Volume.rolling(20).mean().iloc[-1]),
    )

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run():
    today = datetime.today().strftime("%Y-%m-%d")
    print(f"\n🇮🇳 Indian Stock Agent — {today}")
    print(f"   Scanning {len(ALL_TICKERS)} NSE stocks...\n")

    # Delete today's old recommendations (idempotent re-runs)
    supabase.table("recommendations").delete().eq("date", today).execute()

    records = []
    failed  = []

    for i, ticker in enumerate(ALL_TICKERS, 1):
        sys.stdout.write(f"\r  {i}/{len(ALL_TICKERS)}  {ticker:<15}")
        sys.stdout.flush()

        df = fetch(ticker)
        if df is None:
            failed.append(ticker)
            continue

        # Today's signals
        today_sigs = {name: int(fn(df).iloc[-1]) for name, fn in STRATEGIES.items()}
        buy_score  = sum(1 for v in today_sigs.values() if v ==  1)
        sell_score = sum(1 for v in today_sigs.values() if v == -1)

        if buy_score < MIN_BUY_SCORE and sell_score < MIN_SELL_SCORE:
            continue

        # Backtest all strategies
        bt = {name: backtest(df, fn(df)) for name, fn in STRATEGIES.items()}

        # Aggregate backtest (weighted by strategies that fired)
        active_buy  = [name for name, v in today_sigs.items() if v ==  1]
        active_sell = [name for name, v in today_sigs.items() if v == -1]
        active = active_buy if buy_score >= sell_score else active_sell
        agg_wr  = round(float(np.mean([bt[n]["win_rate"]   for n in active])), 1) if active else 0
        agg_ret = round(float(np.mean([bt[n]["avg_return"] for n in active])), 2) if active else 0

        ctx = context(df)
        action = "BUY" if buy_score >= MIN_BUY_SCORE and buy_score >= sell_score else "SELL"

        rec = dict(
            date         = today,
            ticker       = ticker,
            action       = action,
            score        = buy_score if action == "BUY" else sell_score,
            signals      = json.dumps(today_sigs),
            backtest     = json.dumps(bt),
            win_rate     = agg_wr,
            avg_return   = agg_ret,
            active_strategies = ", ".join(active),
            **ctx,
        )
        records.append(rec)
        time.sleep(0.1)

    print(f"\n\n  ✅ {len(records)} signals found  |  ❌ {len(failed)} failed")

    if records:
        # Insert in batches of 20
        for i in range(0, len(records), 20):
            supabase.table("recommendations").insert(records[i:i+20]).execute()
        print(f"  💾 Saved to Supabase")
    else:
        print("  ℹ️  No signals above threshold today")

    if failed:
        print(f"  ⚠️  Failed: {', '.join(failed)}")

    # Update last_run table
    supabase.table("agent_meta").upsert({
        "id": 1,
        "last_run": today,
        "total_signals": len(records),
        "tickers_scanned": len(ALL_TICKERS),
        "failed": len(failed),
    }).execute()

    print("\n  Done ✅\n")


if __name__ == "__main__":
    run()
