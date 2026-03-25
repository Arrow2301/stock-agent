#!/usr/bin/env python3
"""
============================================================
  Indian Stock Market Analysis Agent  —  v2
  Runs daily via GitHub Actions at 7:00 AM IST
  Exchange: NSE

  Improvements over v1:
  ─ Backtest exits on stop-loss / target / max-hold (was fixed hold only)
  ─ Backtest tracks profit factor, median return, max drawdown, exit reasons
  ─ Weighted strategy scoring (win_rate × sample_confidence)
  ─ Composite score 0–100 (strategy + RSI + volume + R:R + regime)
  ─ Market regime filter via NIFTY 50-day EMA
  ─ Per-ticker run log saved to Supabase for debugging
  ─ Fixed M&M ticker (was "MM")
============================================================
"""

import os, sys, time, json, warnings
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
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
    "SUNPHARMA", "ULTRACEMCO", "WIPRO", "ONGC", "NTPC",
    "POWERGRID", "COALINDIA", "BAJFINANCE", "HCLTECH", "TECHM",
    "M&M", "NESTLEIND", "TATAMOTORS", "TATASTEEL",
    "JSWSTEEL", "HINDALCO", "BAJAJFINSV", "GRASIM",
    "CIPLA", "DRREDDY", "EICHERMOT", "HEROMOTOCO", "BRITANNIA",
    "APOLLOHOSP", "TATACONSUM", "ADANIPORTS", "BPCL",
    "INDUSINDBK", "SBILIFE", "HDFCLIFE",
]
EXTRA_WATCHLIST = []   # Add your own: "IRCTC", "ZOMATO", etc.
ALL_TICKERS = NIFTY50 + EXTRA_WATCHLIST

# ─────────────────────────────────────────────
#  PARAMETERS
# ─────────────────────────────────────────────
EMA_SHORT, EMA_LONG = 9, 21
RSI_PERIOD          = 14
RSI_OVERSOLD        = 42
RSI_OVERBOUGHT      = 62
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_PERIOD, BB_STD   = 20, 2.0
ATR_PERIOD          = 14
SUPERTREND_MULT     = 3.0
BT_SL_PCT           = 5.0    # backtest stop-loss %
BT_TARGET_PCT       = 10.0   # backtest take-profit %
BT_MAX_HOLD         = 15     # fallback max hold (trading days)
MIN_WEIGHTED_SCORE  = 0.28   # min weighted vote ratio to emit a signal

# ─────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────

def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def rsi(s, p=14):
    d = s.diff()
    ag = d.clip(lower=0).ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1/p, min_periods=p, adjust=False).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))

def macd(s, fast=12, slow=26, sig=9):
    ml = ema(s, fast) - ema(s, slow)
    sl = ema(ml, sig)
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
        fu.iloc[i] = up.iloc[i] if (up.iloc[i]<fu.iloc[i-1] or c.iloc[i-1]>fu.iloc[i-1]) else fu.iloc[i-1]
        fl.iloc[i] = dn.iloc[i] if (dn.iloc[i]>fl.iloc[i-1] or c.iloc[i-1]<fl.iloc[i-1]) else fl.iloc[i-1]
        if   trend.iloc[i-1]==-1 and c.iloc[i]>fu.iloc[i]: trend.iloc[i]= 1
        elif trend.iloc[i-1]== 1 and c.iloc[i]<fl.iloc[i]: trend.iloc[i]=-1
        else:                                                 trend.iloc[i]=trend.iloc[i-1]
    return trend, pd.Series(np.where(trend==1, fl, fu), index=c.index)

# ─────────────────────────────────────────────
#  SIGNAL GENERATORS
# ─────────────────────────────────────────────

def sig_ema(df):
    e9, e21 = ema(df.Close, EMA_SHORT), ema(df.Close, EMA_LONG)
    s = pd.Series(0, index=df.index)
    s[(e9>e21)&(e9.shift()<=e21.shift())]  =  1
    s[(e9<e21)&(e9.shift()>=e21.shift())]  = -1
    return s

def sig_rsi_macd(df):
    r = rsi(df.Close, RSI_PERIOD)
    _, _, hist = macd(df.Close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    s = pd.Series(0, index=df.index)
    s[(r<RSI_OVERSOLD)   & (hist>0) & (hist.shift()<=0)] =  1
    s[(r>RSI_OVERBOUGHT) & (hist<0) & (hist.shift()>=0)] = -1
    return s

def sig_bb(df):
    up, _, lo = bollinger(df.Close, BB_PERIOD, BB_STD)
    r = rsi(df.Close, RSI_PERIOD)
    s = pd.Series(0, index=df.index)
    s[(df.Low<=lo)  & (df.Close>lo) & (r<50)] =  1
    s[(df.High>=up) & (df.Close<up) & (r>50)] = -1
    return s

def sig_supertrend(df):
    trend, _ = supertrend(df.High, df.Low, df.Close, ATR_PERIOD, SUPERTREND_MULT)
    s = pd.Series(0, index=df.index)
    s[(trend== 1)&(trend.shift()==-1)] =  1
    s[(trend==-1)&(trend.shift()== 1)] = -1
    return s

STRATEGIES = {
    "EMA Crossover": sig_ema,
    "RSI + MACD":    sig_rsi_macd,
    "Bollinger":     sig_bb,
    "Supertrend":    sig_supertrend,
}

# ─────────────────────────────────────────────
#  IMPROVED BACKTEST (SL + Target + Max Hold)
# ─────────────────────────────────────────────

def backtest(df, signals, sl_pct=BT_SL_PCT, tgt_pct=BT_TARGET_PCT, max_hold=BT_MAX_HOLD):
    trades, reasons = [], []
    in_t, ep, entry_idx = False, 0.0, -1
    closes = df.Close.values
    highs  = df.High.values
    lows   = df.Low.values
    opens  = df.Open.values

    for i in range(1, len(df)):
        if in_t:
            sl_px  = ep * (1 - sl_pct  / 100)
            tgt_px = ep * (1 + tgt_pct / 100)
            if lows[i] <= sl_px:
                trades.append((sl_px - ep) / ep * 100); reasons.append("sl");      in_t=False
            elif highs[i] >= tgt_px:
                trades.append((tgt_px - ep) / ep * 100); reasons.append("target"); in_t=False
            elif i >= entry_idx + max_hold:
                trades.append((closes[i]-ep)/ep*100);     reasons.append("timeout"); in_t=False

        if signals.iloc[i] == 1 and not in_t:
            ep, entry_idx, in_t = opens[i], i, True

    if not trades:
        return dict(win_rate=0, avg_return=0, median_return=0, trades=0,
                    profit_factor=0, max_drawdown=0,
                    sl_exits=0, target_exits=0, timeout_exits=0)

    wins   = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    gp, gl = sum(wins), abs(sum(losses))
    pf     = round(gp/gl, 2) if gl > 0 else 99.0

    eq    = np.cumsum(trades)
    peak  = np.maximum.accumulate(eq)
    max_dd = round(float(abs((eq - peak).min())), 2)

    return dict(
        win_rate      = round(len(wins)/len(trades)*100, 1),
        avg_return    = round(float(np.mean(trades)), 2),
        median_return = round(float(np.median(trades)), 2),
        trades        = len(trades),
        profit_factor = min(float(pf), 99.0),
        max_drawdown  = max_dd,
        sl_exits      = reasons.count("sl"),
        target_exits  = reasons.count("target"),
        timeout_exits = reasons.count("timeout"),
    )

# ─────────────────────────────────────────────
#  WEIGHTED STRATEGY SCORING
# ─────────────────────────────────────────────

def weighted_vote(today_sigs, bt_results, action):
    weights = {}
    for name, bt in bt_results.items():
        wr   = bt.get("win_rate", 50) / 100
        n    = bt.get("trades", 0)
        conf = min(n / 8.0, 1.0)        # full confidence at ≥8 trades
        weights[name] = round(wr * conf, 3)

    total_w  = sum(weights.values()) or 1e-9
    signal_w = 0.0
    target_v = 1 if action == "BUY" else -1
    for name, sig in today_sigs.items():
        if sig == target_v:
            signal_w += weights.get(name, 0)

    return round(signal_w / total_w, 3), weights

# ─────────────────────────────────────────────
#  MARKET REGIME
# ─────────────────────────────────────────────

def market_regime():
    try:
        df = yf.download("^NSEI", period="120d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 55:
            return "UNKNOWN", 0.5
        close = df["Close"].squeeze()
        e50   = float(ema(close, 50).iloc[-1])
        e20   = float(ema(close, 20).iloc[-1])
        r     = float(rsi(close, 14).iloc[-1])
        price = float(close.iloc[-1])
        if price > e20 > e50 and r > 50:  return "BULLISH", 1.0
        if price < e20 < e50 and r < 50:  return "BEARISH", 0.0
        return "NEUTRAL", 0.5
    except Exception:
        return "UNKNOWN", 0.5

# ─────────────────────────────────────────────
#  COMPOSITE SCORE  0–100
# ─────────────────────────────────────────────

def composite_score(today_sigs, bt_results, ctx, regime_sc, action):
    w_ratio, _ = weighted_vote(today_sigs, bt_results, action)
    strat_pts  = round(w_ratio * 40, 2)

    r = ctx["rsi"]
    if action == "BUY":
        rsi_pts = max(0, min((60 - r) / 40 * 20, 20))
    else:
        rsi_pts = max(0, min((r - 40) / 40 * 20, 20))

    avg_vol   = ctx["avg_volume"]
    vol_ratio = ctx["volume"] / avg_vol if avg_vol > 0 else 1.0
    vol_pts   = min(vol_ratio / 2.0 * 15, 15)

    rr      = ctx["reward_pct"] / ctx["risk_pct"] if ctx.get("risk_pct", 0) > 0 else 0
    rr_pts  = min(rr / 3.0 * 15, 15)

    reg_pts = regime_sc * 10 if action == "BUY" else (1 - regime_sc) * 10

    total = round(strat_pts + rsi_pts + vol_pts + rr_pts + reg_pts, 1)
    breakdown = dict(
        strategy = round(strat_pts, 1),
        rsi      = round(rsi_pts,  1),
        volume   = round(vol_pts,  1),
        rr       = round(rr_pts,   1),
        regime   = round(reg_pts,  1),
    )
    return min(total, 100.0), breakdown

def score_label(s):
    if s >= 80: return "Very Strong"
    if s >= 65: return "Strong"
    if s >= 50: return "Good"
    if s >= 35: return "Moderate"
    return "Weak"

# ─────────────────────────────────────────────
#  DATA FETCH
# ─────────────────────────────────────────────

def fetch(ticker, days=430):
    try:
        df = yf.download(
            ticker+".NS",
            start=datetime.today()-timedelta(days=days),
            end=datetime.today(),
            progress=False, auto_adjust=True
        )
        if df.empty or len(df) < 80:
            return None, "insufficient_data"
        df.columns = [c[0] if isinstance(c,tuple) else c for c in df.columns]
        return df[["Open","High","Low","Close","Volume"]].dropna(), "ok"
    except Exception as e:
        return None, str(e)[:120]

# ─────────────────────────────────────────────
#  PRICE CONTEXT
# ─────────────────────────────────────────────

def context(df):
    c = df.Close
    r = float(rsi(c, RSI_PERIOD).iloc[-1])
    _, _, hist = macd(c, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    e9  = float(ema(c, EMA_SHORT).iloc[-1])
    e21 = float(ema(c, EMA_LONG).iloc[-1])
    trend, st_line = supertrend(df.High, df.Low, c, ATR_PERIOD, SUPERTREND_MULT)

    price    = float(c.iloc[-1])
    atr_now  = float(atr(df.High, df.Low, c, ATR_PERIOD).iloc[-1])
    sl       = round(price - 1.5 * atr_now, 2)
    tgt      = round(price + 3.0 * atr_now, 2)
    risk_pct = round((price - sl) / price * 100, 2)
    rwd_pct  = round((tgt - price) / price * 100, 2)

    return dict(
        price           = round(price, 2),
        change_1d       = round((c.iloc[-1]-c.iloc[-2])/c.iloc[-2]*100, 2),
        change_5d       = round((c.iloc[-1]-c.iloc[-6])/c.iloc[-6]*100, 2),
        rsi             = round(r, 1),
        macd_hist       = round(float(hist.iloc[-1]), 3),
        ema_bullish     = bool(e9 > e21),
        supertrend_up   = bool(trend.iloc[-1] == 1),
        supertrend_line = round(float(st_line.iloc[-1]), 2),
        support         = round(float(df.Low.rolling(20).min().iloc[-1]), 2),
        resistance      = round(float(df.High.rolling(20).max().iloc[-1]), 2),
        stop_loss       = sl,
        target          = tgt,
        risk_pct        = risk_pct,
        reward_pct      = rwd_pct,
        volume          = int(df.Volume.iloc[-1]),
        avg_volume      = int(df.Volume.rolling(20).mean().iloc[-1]),
    )

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run():
    today = datetime.today().strftime("%Y-%m-%d")
    print(f"\n🇮🇳 Indian Stock Agent v2 — {today}")
    print(f"   Scanning {len(ALL_TICKERS)} NSE stocks...\n")

    print("   Checking NIFTY market regime...")
    regime_label, regime_sc = market_regime()
    print(f"   Regime: {regime_label}  (score={regime_sc})\n")

    # Wipe today's stale rows (idempotent re-runs)
    supabase.table("recommendations").delete().eq("date", today).execute()
    supabase.table("ticker_run_log").delete().eq("date", today).execute()

    records, run_logs = [], []

    for i, ticker in enumerate(ALL_TICKERS, 1):
        sys.stdout.write(f"\r  {i}/{len(ALL_TICKERS)}  {ticker:<15}")
        sys.stdout.flush()

        df, status = fetch(ticker)
        run_logs.append(dict(date=today, ticker=ticker, status=status))
        if df is None:
            continue

        today_sigs = {name: int(fn(df).iloc[-1]) for name, fn in STRATEGIES.items()}
        buy_count  = sum(1 for v in today_sigs.values() if v ==  1)
        sell_count = sum(1 for v in today_sigs.values() if v == -1)
        if buy_count == 0 and sell_count == 0:
            continue

        bt     = {name: backtest(df, fn(df)) for name, fn in STRATEGIES.items()}
        action = "BUY" if buy_count >= sell_count else "SELL"
        w_ratio, weights = weighted_vote(today_sigs, bt, action)
        if w_ratio < MIN_WEIGHTED_SCORE:
            continue

        ctx = context(df)
        c_score, c_breakdown = composite_score(today_sigs, bt, ctx, regime_sc, action)

        active   = [n for n,v in today_sigs.items()
                    if (v==1 and action=="BUY") or (v==-1 and action=="SELL")]
        agg      = lambda key: round(float(np.mean([bt[n][key] for n in active])), 2) if active else 0
        low_samp = int(np.mean([bt[n]["trades"] for n in active])) < 5 if active else True

        records.append(dict(
            date               = today,
            ticker             = ticker,
            action             = action,
            raw_score          = buy_count if action=="BUY" else sell_count,
            weighted_score_val = w_ratio,
            composite_score    = c_score,
            score_label        = score_label(c_score),
            score_breakdown    = json.dumps(c_breakdown),
            signals            = json.dumps(today_sigs),
            strategy_weights   = json.dumps(weights),
            backtest           = json.dumps(bt),
            active_strategies  = ", ".join(active),
            low_sample_warning = low_samp,
            win_rate           = agg("win_rate"),
            avg_return         = agg("avg_return"),
            median_return      = agg("median_return"),
            profit_factor      = agg("profit_factor"),
            max_drawdown       = agg("max_drawdown"),
            avg_trades         = int(np.mean([bt[n]["trades"] for n in active])) if active else 0,
            market_regime      = regime_label,
            **ctx,
        ))
        time.sleep(0.1)

    sys.stdout.write("\r" + " "*60 + "\r")
    print(f"  ✅ {len(records)} signals  |  ❌ {sum(1 for l in run_logs if l['status']!='ok')} failed\n")

    if records:
        for i in range(0, len(records), 20):
            supabase.table("recommendations").insert(records[i:i+20]).execute()

    for i in range(0, len(run_logs), 50):
        supabase.table("ticker_run_log").insert(run_logs[i:i+50]).execute()

    supabase.table("agent_meta").upsert({
        "id": 1, "last_run": today,
        "total_signals": len(records),
        "tickers_scanned": len(ALL_TICKERS),
        "failed": sum(1 for l in run_logs if l["status"]!="ok"),
        "market_regime": regime_label,
    }).execute()

    print("  Done ✅\n")


if __name__ == "__main__":
    run()
