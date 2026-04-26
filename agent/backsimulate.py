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

SL_PCT = 5.0              # fallback only when stored dynamic SL is unavailable
TARGET_PCT = 10.0         # fallback only when stored dynamic target is unavailable
MAX_HOLD = 15
MIN_AGE_DAYS = MAX_HOLD + 7
BATCH_SIZE = 50
BENCHMARK = "^NSEI"


def _flatten(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def _safe_float(v, default=None):
    try:
        if v is None:
            return default
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _download_ohlc(symbol: str, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    df = _flatten(df)
    if df is None or df.empty:
        return None
    return df[["Open", "High", "Low", "Close"]].dropna()


def _benchmark_return(bench_df, entry_date, exit_date):
    if bench_df is None or bench_df.empty:
        return None
    try:
        b = bench_df.sort_index()
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


def simulate_trade(
    ticker: str,
    signal_date: str,
    stop_loss: float | None = None,
    target: float | None = None,
    max_hold: int = MAX_HOLD,
):
    """
    Walk-forward BUY simulation with next-bar execution.
    A signal generated on day i enters at the next available trading day open.
    EXIT recommendations are intentionally not simulated as shorts.
    """
    try:
        sig_dt = datetime.strptime(signal_date, "%Y-%m-%d")
        start = sig_dt + timedelta(days=1)
        end = min(sig_dt + timedelta(days=max_hold * 2 + 14), datetime.today())
        df = _download_ohlc(ticker + ".NS", start=start, end=end)
        if df is None or len(df) < 2:
            return None
        bench_df = _download_ohlc(BENCHMARK, start=start, end=end)

        entry_idx = 0
        entry_price = float(df["Open"].iloc[entry_idx])
        if entry_price <= 0:
            return None

        sl_price = _safe_float(stop_loss)
        tgt_price = _safe_float(target)
        if not sl_price or sl_price >= entry_price:
            sl_price = entry_price * (1 - SL_PCT / 100)
        if not tgt_price or tgt_price <= entry_price:
            tgt_price = entry_price * (1 + TARGET_PCT / 100)

        def _finalize(exit_price, exit_idx, exit_reason):
            exit_date = str(df.index[exit_idx].date())
            actual_return = round((exit_price - entry_price) / entry_price * 100, 2)
            benchmark_return = _benchmark_return(bench_df, df.index[entry_idx], df.index[exit_idx])
            relative_return = round(actual_return - benchmark_return, 2) if benchmark_return is not None else None
            risk_pct = (entry_price - sl_price) / entry_price * 100 if sl_price < entry_price else None
            reward_pct = (tgt_price - entry_price) / entry_price * 100 if tgt_price > entry_price else None
            rr_ratio = round(reward_pct / risk_pct, 2) if risk_pct and reward_pct is not None and risk_pct > 0 else None
            return {
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "exit_date": exit_date,
                "exit_reason": exit_reason,
                "actual_return_pct": actual_return,
                "benchmark_return_pct": benchmark_return,
                "relative_return_pct": relative_return,
                "benchmark_outperformance_rate": 100.0 if relative_return is not None and relative_return > 0 else (0.0 if relative_return is not None else None),
                "days_held": exit_idx,
                "was_win": actual_return > 0,
                "rr_ratio": rr_ratio,
            }

        for i in range(0, min(max_hold + 1, len(df))):
            row = df.iloc[i]
            high, low = float(row["High"]), float(row["Low"])
            # Stop-first is conservative if both levels are touched intraday.
            if low <= sl_price:
                return _finalize(sl_price, i, "sl_hit")
            if high >= tgt_price:
                return _finalize(tgt_price, i, "target_hit")

        exit_idx = min(max_hold, len(df) - 1)
        final_price = float(df["Close"].iloc[exit_idx])
        return _finalize(final_price, exit_idx, "timeout")
    except Exception:
        return None


def get_pending_recs():
    cutoff = (datetime.today() - timedelta(days=MIN_AGE_DAYS)).strftime('%Y-%m-%d')
    try:
        recs_res = (
            supabase.table('recommendations')
            .select('id, ticker, date, action, composite_score, technical_score, win_rate, stop_loss, target, rr_ratio')
            .eq('action', 'BUY')
            .lte('date', cutoff)
            .order('date', desc=False)
            .execute()
        )
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
        print("  ✅ No pending BUY recommendations to simulate\n")
        return
    results = []
    wins = losses = errors = 0
    for i, rec in enumerate(pending, 1):
        ticker, signal_date, rec_id = rec['ticker'], str(rec['date']), rec['id']
        sys.stdout.write(f"\r  {i}/{len(pending)}  {ticker:<12}  {signal_date}")
        sys.stdout.flush()
        sim = simulate_trade(
            ticker,
            signal_date,
            stop_loss=rec.get('stop_loss'),
            target=rec.get('target'),
            max_hold=MAX_HOLD,
        )
        if sim is None:
            errors += 1
            continue
        wins += int(sim['was_win'])
        losses += int(not sim['was_win'])
        results.append({
            'recommendation_id': rec_id, 'ticker': ticker, 'signal_date': signal_date, 'action': 'BUY',
            'entry_price': sim['entry_price'], 'exit_price': sim['exit_price'], 'exit_date': sim['exit_date'],
            'exit_reason': sim['exit_reason'], 'actual_return_pct': sim['actual_return_pct'],
            'benchmark_return_pct': sim.get('benchmark_return_pct'),
            'relative_return_pct': sim.get('relative_return_pct'),
            'benchmark_outperformance_rate': sim.get('benchmark_outperformance_rate'),
            'rr_ratio': sim.get('rr_ratio') if sim.get('rr_ratio') is not None else rec.get('rr_ratio'),
            'composite_score': float(rec.get('composite_score') or 0),
            'technical_score': float(rec.get('technical_score') or 0),
            'predicted_win_rate': float(rec.get('win_rate') or 0),
            'was_win': bool(sim['was_win']), 'days_held': int(sim['days_held']), 'run_date': run_date,
        })
        time.sleep(0.15)
    sys.stdout.write("\r" + " " * 60 + "\r")
    total = wins + losses
    actual_wr = round(wins / total * 100, 1) if total else 0
    avg_ret = round(float(np.mean([r['actual_return_pct'] for r in results])), 2) if results else 0
    avg_rel = round(float(np.mean([r['relative_return_pct'] for r in results if r.get('relative_return_pct') is not None])), 2) if any(r.get('relative_return_pct') is not None for r in results) else None
    outperf_vals = [r['relative_return_pct'] for r in results if r.get('relative_return_pct') is not None]
    outperf_rate = round(sum(1 for r in outperf_vals if r > 0) / len(outperf_vals) * 100, 1) if outperf_vals else None
    if results:
        for i in range(0, len(results), 20):
            try:
                supabase.table('backtest_simulations').insert(results[i:i + 20]).execute()
            except Exception as e:
                print(f"\n  ❌ Insert error batch {i // 20 + 1}: {e}")
        try:
            supabase.table('simulation_meta').upsert({
                'id': 1,
                'last_run': run_date,
                'total_simulated': total,
                'actual_win_rate': actual_wr,
                'actual_avg_return': avg_ret,
                'actual_avg_relative_return': avg_rel,
                'benchmark_outperformance_rate': outperf_rate,
                'pending_unprocessed': len(pending) - total,
            }).execute()
        except Exception as e:
            print(f"  ⚠️ Meta update failed: {e}")
    print("  Done ✅\n")


if __name__ == '__main__':
    run()
