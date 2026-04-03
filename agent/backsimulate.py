#!/usr/bin/env python3
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
from supabase import create_client, Client

warnings.filterwarnings("ignore")
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
SL_PCT = 5.0
TARGET_PCT = 10.0
MAX_HOLD = 15
MIN_AGE_DAYS = MAX_HOLD + 7
BATCH_SIZE = 50


def simulate_trade(ticker: str, signal_date: str, action: str, sl_pct: float = SL_PCT, tgt_pct: float = TARGET_PCT, max_hold: int = MAX_HOLD):
    try:
        sig_dt = datetime.strptime(signal_date, "%Y-%m-%d")
        start = sig_dt + timedelta(days=1)
        end = min(sig_dt + timedelta(days=max_hold * 2 + 14), datetime.today())
        df = yf.download(ticker + '.NS', start=start, end=end, progress=False, auto_adjust=True)
        if df.empty or len(df) < 2:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open", "High", "Low", "Close"]].dropna()
        if df.empty:
            return None
        entry_price = float(df["Open"].iloc[0])
        if entry_price <= 0:
            return None
        sl_price = entry_price * (1 - sl_pct / 100)
        tgt_price = entry_price * (1 + tgt_pct / 100)
        for i in range(1, min(max_hold + 1, len(df))):
            row = df.iloc[i]
            high, low = float(row["High"]), float(row["Low"])
            exit_date = str(df.index[i].date())
            if action == "BUY":
                if low <= sl_price:
                    return {"entry_price": round(entry_price, 2), "exit_price": round(sl_price, 2), "exit_date": exit_date, "exit_reason": "sl_hit", "actual_return_pct": round((sl_price - entry_price) / entry_price * 100, 2), "days_held": i, "was_win": False}
                if high >= tgt_price:
                    return {"entry_price": round(entry_price, 2), "exit_price": round(tgt_price, 2), "exit_date": exit_date, "exit_reason": "target_hit", "actual_return_pct": round((tgt_price - entry_price) / entry_price * 100, 2), "days_held": i, "was_win": True}
        exit_idx = min(max_hold, len(df) - 1)
        final_price = float(df["Close"].iloc[exit_idx])
        exit_date = str(df.index[exit_idx].date())
        ret = round((final_price - entry_price) / entry_price * 100, 2)
        return {"entry_price": round(entry_price, 2), "exit_price": round(final_price, 2), "exit_date": exit_date, "exit_reason": "timeout", "actual_return_pct": ret, "days_held": exit_idx, "was_win": ret > 0}
    except Exception:
        return None


def get_pending_recs():
    cutoff = (datetime.today() - timedelta(days=MIN_AGE_DAYS)).strftime('%Y-%m-%d')
    try:
        recs_res = supabase.table('recommendations').select('id, ticker, date, action, composite_score, win_rate, stop_loss, target').eq('action', 'BUY').lte('date', cutoff).order('date', desc=False).execute()
        all_recs = recs_res.data or []
        if not all_recs:
            return []
        sim_res = supabase.table('backtest_simulations').select('recommendation_id').execute()
        simulated_ids = {r['recommendation_id'] for r in (sim_res.data or [])}
        return [r for r in all_recs if r['id'] not in simulated_ids][:BATCH_SIZE]
    except Exception as e:
        print(f"  ❌ Failed to fetch pending recs: {e}")
        return []


def run():
    run_date = datetime.today().strftime('%Y-%m-%d')
    print(f"\n🔬 Walk-Forward Backsimulation — {run_date}")
    pending = get_pending_recs()
    if not pending:
        print("  ✅ No pending recommendations to simulate\n")
        return
    results = []
    wins = losses = errors = 0
    for i, rec in enumerate(pending, 1):
        ticker, signal_date, rec_id = rec['ticker'], str(rec['date']), rec['id']
        sys.stdout.write(f"\r  {i}/{len(pending)}  {ticker:<12}  {signal_date}")
        sys.stdout.flush()
        sim = simulate_trade(ticker, signal_date, 'BUY', SL_PCT, TARGET_PCT, MAX_HOLD)
        if sim is None:
            errors += 1
            continue
        wins += int(sim['was_win'])
        losses += int(not sim['was_win'])
        results.append({
            'recommendation_id': rec_id, 'ticker': ticker, 'signal_date': signal_date, 'action': 'BUY',
            'entry_price': sim['entry_price'], 'exit_price': sim['exit_price'], 'exit_date': sim['exit_date'],
            'exit_reason': sim['exit_reason'], 'actual_return_pct': sim['actual_return_pct'],
            'composite_score': float(rec.get('composite_score') or 0), 'predicted_win_rate': float(rec.get('win_rate') or 0),
            'was_win': bool(sim['was_win']), 'days_held': int(sim['days_held']), 'run_date': run_date,
        })
        time.sleep(0.15)
    sys.stdout.write("\r" + " " * 60 + "\r")
    total = wins + losses
    actual_wr = round(wins / total * 100, 1) if total else 0
    avg_ret = round(float(np.mean([r['actual_return_pct'] for r in results])), 2) if results else 0
    if results:
        for i in range(0, len(results), 20):
            try:
                supabase.table('backtest_simulations').insert(results[i:i + 20]).execute()
            except Exception as e:
                print(f"\n  ❌ Insert error batch {i // 20 + 1}: {e}")
        try:
            supabase.table('simulation_meta').upsert({'id': 1, 'last_run': run_date, 'total_simulated': total, 'actual_win_rate': actual_wr, 'actual_avg_return': avg_ret, 'pending_unprocessed': len(pending) - total}).execute()
        except Exception as e:
            print(f"  ⚠️ Meta update failed: {e}")
    print("  Done ✅\n")


if __name__ == '__main__':
    run()
