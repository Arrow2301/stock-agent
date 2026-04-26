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
    "MARUTI", "M&M", "TATAMOTORS", "EICHERMOT", "HEROMOTOCO",
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

def backtest_with_costs(df, signals, p):
    """
    Backtest with:
    - Stop-loss, target, or max-hold exit
    - Slippage + brokerage deducted from each trade
    """
    trades, reasons = [], []
    in_t, ep, entry_idx = False, 0.0, -1
    closes=df.Close.values; highs=df.High.values
    lows=df.Low.values; opens=df.Open.values
    sl_pct=p["BT_SL_PCT"]; tgt_pct=p["BT_TARGET_PCT"]; max_hold=p["BT_MAX_HOLD"]
    cost_pct = SLIPPAGE_PCT + BROKERAGE_PCT   # deducted from each trade's gross return

    for i in range(1, len(df)):
        # Next-bar execution: a signal on day i-1 is entered at day i open.
        if (not in_t) and signals.iloc[i-1] == 1 and opens[i] > 0:
            ep, entry_idx, in_t = opens[i], i, True

        if in_t:
            sl_px=ep*(1-sl_pct/100); tgt_px=ep*(1+tgt_pct/100)
            gross = None
            if   lows[i]  <= sl_px:          gross=(sl_px-ep)/ep*100;   reasons.append("sl")
            elif highs[i] >= tgt_px:         gross=(tgt_px-ep)/ep*100;  reasons.append("target")
            elif i >= entry_idx + max_hold:  gross=(closes[i]-ep)/ep*100; reasons.append("timeout")
            if gross is not None:
                trades.append(gross - cost_pct)
                in_t = False

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
    expectancy = wr*avg_win - (1-wr)*avg_loss   # per-trade expected return

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
            "RSI_OVERSOLD":       trial.suggest_int(  "RSI_OVERSOLD", 25,  45),
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
            "MIN_WEIGHTED_SCORE": trial.suggest_float("MIN_WEIGHTED_SCORE",0.10,0.55),
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

def get_next_version() -> int:
    try:
        res = (
            supabase.table("agent_params")
            .select("version")
            .order("version", desc=True)
            .limit(1)
            .execute()
        )
        if res.data:
            return int(res.data[0]["version"]) + 1
    except Exception:
        pass
    return 1

def demote_old_challengers():
    try:
        supabase.table("agent_params").update({"status": "candidate"}).eq("status", "challenger").execute()
    except Exception as e:
        print(f"⚠️  Could not demote old challengers: {e}")

def champion_exists() -> bool:
    try:
        res = supabase.table("agent_params").select("id").eq("status", "champion").limit(1).execute()
        return bool(res.data)
    except Exception:
        return False

def promote_to_champion(version: int):
    try:
        # Demote existing champion
        supabase.table("agent_params").update({"status": "retired"}).eq("status", "champion").execute()
        supabase.table("agent_params").update({
            "status": "champion",
            "promoted_at": date.today().isoformat(),
            "notes": "Auto-promoted because no previous champion existed"
        }).eq("version", version).execute()
        print(f"🏆 Version {version} promoted to CHAMPION")
    except Exception as e:
        print(f"⚠️  Champion promotion failed: {e}")

def save_optimization_run(n_trials, n_valid, best, champion_version, challenger_version, stocks_used):
    try:
        supabase.table("optimization_runs").insert({
            "run_date": date.today().isoformat(),
            "n_trials": n_trials,
            "n_valid_trials": n_valid,
            "best_score": best["objective_score"] if best else None,
            "best_profit_factor": best["profit_factor"] if best else None,
            "best_win_rate": best["win_rate"] if best else None,
            "best_avg_return": best["avg_return"] if best else None,
            "champion_version": champion_version,
            "challenger_version": challenger_version,
            "stocks_used": stocks_used,
        }).execute()
    except Exception as e:
        print(f"⚠️  Could not save optimization run: {e}")

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    global N_TRIALS

    env_trials = os.getenv("N_TRIALS") or os.getenv("n_trials")
    if env_trials:
        try:
            N_TRIALS = int(env_trials)
            print(f"Using N_TRIALS from environment: {N_TRIALS}")
        except Exception:
            print(f"⚠️ Invalid N_TRIALS={env_trials!r}; using default {N_TRIALS}")

    print("🧪 Starting Parameter Optimizer")
    print(f"   Trials: {N_TRIALS} | Windows: {N_WINDOWS} | Stocks: {len(SAMPLE_STOCKS)}")

    all_data = fetch_all_data()
    valid_count = sum(1 for d in all_data.values() if not d.empty)
    if valid_count < 5:
        print("❌ Not enough data to optimize")
        sys.exit(1)

    # Run Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(make_objective(all_data), n_trials=N_TRIALS, show_progress_bar=False)

    # Valid trials only
    completed = [t for t in study.trials if t.value is not None and t.value > -900]
    completed = sorted(completed, key=lambda t: t.value, reverse=True)

    if not completed:
        print("❌ No valid trials produced")
        save_optimization_run(N_TRIALS, 0, None, None, None, valid_count)
        sys.exit(1)

    print(f"\n✅ Completed {len(completed)} valid trials")
    print("Top 5 objective scores:", [round(t.value, 4) for t in completed[:5]])

    # Save top K candidates
    base_version = get_next_version()
    demote_old_challengers()

    saved = []
    for rank, trial in enumerate(completed[:TOP_K], 1):
        version = base_version + rank - 1
        params  = trial.params.copy()

        # Re-add fixed fields
        params["W_STRATEGY"] = 40

        metrics = trial.user_attrs.get("metrics", {})

        status = "challenger" if rank == 1 else "candidate"

        row = {
            "version": version,
            "status": status,
            "params_json": params,
            "objective_score": float(trial.value),
            "profit_factor": metrics.get("profit_factor"),
            "win_rate": metrics.get("win_rate"),
            "avg_return": metrics.get("avg_return"),
            "max_drawdown": metrics.get("max_drawdown"),
            "total_trades": metrics.get("total_trades"),
            "train_start": (date.today() - timedelta(days=DATA_DAYS)).isoformat(),
            "train_end": date.today().isoformat(),
            "valid_start": None,
            "valid_end": None,
            "run_date": date.today().isoformat(),
            "rank": rank,
            "notes": (
                f"Optuna WF score={trial.value:.4f}; "
                f"trades={metrics.get('total_trades')}; "
                f"next-open execution; SELL treated as exit/avoid-long"
            ),
        }
        try:
            supabase.table("agent_params").insert(row).execute()
            saved.append(row)
            print(
                f"  Saved v{version} [{status}] "
                f"score={trial.value:.4f} "
                f"PF={metrics.get('profit_factor'):.2f} "
                f"WR={metrics.get('win_rate'):.1f}% "
                f"TR={metrics.get('total_trades')}"
            )
        except Exception as e:
            print(f"⚠️  Could not save version {version}: {e}")

    challenger_version = saved[0]["version"] if saved else None

    # If no champion exists, auto-promote challenger
    champion_version = None
    if challenger_version and not champion_exists():
        promote_to_champion(challenger_version)
        champion_version = challenger_version
    else:
        try:
            res = supabase.table("agent_params").select("version").eq("status", "champion").limit(1).execute()
            if res.data:
                champion_version = res.data[0]["version"]
        except Exception:
            pass

    # Save run log
    best = saved[0] if saved else None
    save_optimization_run(
        n_trials=N_TRIALS,
        n_valid=len(completed),
        best=best,
        champion_version=champion_version,
        challenger_version=challenger_version,
        stocks_used=valid_count,
    )

    print("\n🏁 Optimization complete")
    if challenger_version:
        print(f"   Challenger: v{challenger_version}")
    if champion_version:
        print(f"   Champion:   v{champion_version}")


if __name__ == "__main__":
    main()
