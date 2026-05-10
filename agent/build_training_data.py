#!/usr/bin/env python3
"""
============================================================
  Synthetic Training Data Generator — v8

  Generates labelled (signal, outcome) rows by replaying the v8
  strategy roster across historical NSE data. The output table
  feeds agent/score_model.py.

  WHY THIS WAS REBUILT
  ────────────────────
  The pre-v8 generator had four silent bugs that invalidated
  everything trained on it. They are documented here both as a
  post-mortem and as test cases the test suite must enforce.

   Bug 1 — Regime mismatch
     The generator used SMA50/SMA200 for the `regime` label;
     the live agent uses EMA20/EMA50 + RSI. The model was
     trained on a feature it would never see in production.
     FIX: both call analyze.regime_from_close_series() now.

   Bug 2 — Entry-price look-ahead
     simulate_forward used signal_day Close as the entry price;
     the live agent enters at signal_day+1 Open. NSE overnight
     gaps average ~0.4%, occasionally ~2%, so a setup could
     "win" in synthetic but fail or vice versa in production.
     FIX: simulate_forward now enters at df.Open.iloc[i+1].

   Bug 3 — Two VIX code paths
     analyze.fetch_vix_features_today() and the generator each
     reimplemented the same maths. Easy to drift over time.
     FIX: both call analyze.vix_features_from_series().

   Bug 4 — MFE/MAE recorded but bar-of-MFE was not
     Trailing-stop research needs to know WHEN the MFE happened
     (was it bar 2 or bar 13?) to simulate "trail SL once
     MFE > 5%" rules. The previous version stored only the
     scalar MFE, throwing away the temporal info.
     FIX: also record mfe_bar (and mae_bar) — the bar-index
     within the holding window where each was reached.

  WHAT THIS GENERATOR DOES
  ────────────────────────
  For every (ticker, day) where any v8 strategy fires BUY:
    1. Compute features with df.iloc[: i + 1] only — strictly
       point-in-time (no look-ahead).
    2. Use the next-bar OPEN at i+1 as entry price (matches
       live execution).
    3. Walk forward up to HOLDING_PERIOD_DAYS bars on full OHLC.
       SL takes priority on same-bar SL+target touches.
    4. Record full features + MFE/MAE + their bar indices +
       gap-on-entry diagnostic + the gating flag (now using
       v8 logic) so downstream code can study what to gate.

  USAGE
  ─────
    # Full run (~30-60 min on GH Actions):
    python agent/build_training_data.py

    # Quick local check:
    python agent/build_training_data.py \\
        --tickers RELIANCE,TCS,HDFCBANK,INFY,SBIN \\
        --start 2024-01-01 --end 2024-12-31 \\
        --csv /tmp/sample.csv

    # Self-check feature parity vs live agent:
    python agent/build_training_data.py --self-check

  KNOWN LIMITATIONS
  ─────────────────
   * Survivorship bias. The watchlist is the present-day
     Nifty 200; delisted tickers (Yes Bank pre-2020 mess,
     DHFL, etc.) are absent. Backtest results are biased
     UPWARD by survival.
   * No corporate-action handling beyond yfinance's auto_adjust.
     Splits and bonuses are smooth; demergers (Reliance Jio
     spin-off, etc.) may produce anomalous bars on the
     event date.
   * Costs not modelled. Slippage and brokerage are applied
     at backsim time, not here. The label is gross.
============================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import math
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import (
    ema, rsi, macd, bollinger, atr,
    sig_donchian, sig_ema, sig_rsi_trend_shift, sig_bb,
    dynamic_trade_levels,
    regime_from_close_series,            # v8: shared with live agent
    vix_features_from_series,            # v8: shared with live agent
    DEFAULT_PARAMS,
    NIFTY200,
)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
HOLDING_PERIOD_DAYS = 15           # forward simulation window
WARMUP_DAYS         = 100          # min bars before first signal (indicator warmup)
BATCH_INSERT_SIZE   = 500
DEFAULT_START       = "2020-04-01"
DEFAULT_END         = "2026-04-30"
DEFAULT_UNIVERSE    = NIFTY200

REGIME_SCORE = {"BULLISH": 1.0, "NEUTRAL": 0.5, "BEARISH": 0.0, "UNKNOWN": 0.5}

# VIX prefetch buffer — need enough history before the first signal
# date to compute the 60-day z-score on day 1 of the window.
VIX_LOOKBACK_BUFFER_DAYS = 90


# ─────────────────────────────────────────────
#  DATA FETCH
# ─────────────────────────────────────────────
def fetch_ohlc(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            ticker + ".NS",
            start=start, end=end,
            progress=False, auto_adjust=True, threads=False,
        )
        if df is None or df.empty:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        # Need warmup + holding period + 1 (the next-bar open)
        return df if len(df) >= WARMUP_DAYS + HOLDING_PERIOD_DAYS + 1 else None
    except Exception:
        return None


def fetch_nifty(start: str, end: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            "^NSEI",
            start=start, end=end,
            progress=False, auto_adjust=True, threads=False,
        )
        if df is None or df.empty:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df.dropna(subset=["Close"])
    except Exception:
        return None


def fetch_vix_history(start: str, end: str) -> pd.DataFrame | None:
    try:
        pad_start = (pd.Timestamp(start) - pd.Timedelta(days=VIX_LOOKBACK_BUFFER_DAYS)
                    ).strftime("%Y-%m-%d")
        df = yf.download(
            "^INDIAVIX",
            start=pad_start, end=end,
            progress=False, auto_adjust=True, threads=False,
        )
        if df is None or df.empty:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.dropna(subset=["Close"])
        return df[["Close"]] if len(df) else None
    except Exception:
        return None


# ─────────────────────────────────────────────
#  REGIME (v8: shared logic with live agent)
# ─────────────────────────────────────────────
def compute_regime_series(nifty_df: pd.DataFrame, p: dict) -> pd.Series:
    """Classify regime for every day in the Nifty series, point-in-time.

    Calls analyze.regime_from_close_series() with the CLOSE PRICES UP TO
    AND INCLUDING each day. This is the ONLY place regime gets computed
    for the synthetic data, and it uses the same function the live agent
    uses on real-time fetches.
    """
    close = nifty_df["Close"].dropna()
    if len(close) < 55:
        return pd.Series("UNKNOWN", index=nifty_df.index, dtype=object)

    out: dict = {}
    for i in range(len(close)):
        d = close.index[i]
        # Slice strictly point-in-time. The live agent fetches "last 120 days"
        # and computes the regime — equivalent to taking the most recent 120
        # bars at any historical point.
        slice_pit = close.iloc[max(0, i - 120):i + 1]
        label, _ = regime_from_close_series(slice_pit, p)
        out[d] = label
    return pd.Series(out, index=nifty_df.index)


# ─────────────────────────────────────────────
#  VIX (v8: shared logic with live agent)
# ─────────────────────────────────────────────
def vix_features_at(vix_df: pd.DataFrame | None, asof_ts: pd.Timestamp) -> dict:
    """Compute the 3 VIX features as of `asof_ts`, point-in-time.
    Slices the precomputed VIX history then delegates to the shared
    function from analyze.py.
    """
    if vix_df is None or vix_df.empty:
        return {"vix_level": None, "vix_change_5d": None, "vix_zscore_60d": None}
    pit = vix_df[vix_df.index <= asof_ts]
    if pit.empty:
        return {"vix_level": None, "vix_change_5d": None, "vix_zscore_60d": None}
    return vix_features_from_series(pit["Close"])


# ─────────────────────────────────────────────
#  FEATURES (point-in-time, no look-ahead)
# ─────────────────────────────────────────────
def compute_features(
    df_pit: pd.DataFrame,
    nifty_pit: pd.DataFrame | None,
    regime_label: str,
    today_sigs: dict,
    p: dict,
) -> dict:
    if len(df_pit) < 30:
        return {}
    c     = df_pit.Close
    price = float(c.iloc[-1])
    if not math.isfinite(price) or price <= 0:
        return {}

    rsi_v       = float(rsi(c, p["RSI_PERIOD"]).iloc[-1])
    _, _, h_v   = macd(c, p["MACD_FAST"], p["MACD_SLOW"], p["MACD_SIGNAL"])
    macd_h      = float(h_v.iloc[-1]) if pd.notna(h_v.iloc[-1]) else 0.0
    e_s         = float(ema(c, p["EMA_SHORT"]).iloc[-1])
    e_l         = float(ema(c, p["EMA_LONG"]).iloc[-1])
    ema_bull    = bool(e_s > e_l)
    atr_v       = float(atr(df_pit.High, df_pit.Low, c, p["ATR_PERIOD"]).iloc[-1])
    atr_pct     = round((atr_v / price) * 100, 3) if math.isfinite(atr_v) else None

    def _pct_change(n):
        if len(c) < n + 1: return None
        prev = float(c.iloc[-(n + 1)])
        if prev <= 0: return None
        return round((price - prev) / prev * 100, 3)
    change_1d  = _pct_change(1)
    change_5d  = _pct_change(5)
    change_30d = _pct_change(30)

    lookback = min(252, len(c))
    high_52w = float(c.iloc[-lookback:].max())
    low_52w  = float(c.iloc[-lookback:].min())
    pct_from_52w_high = round((price - high_52w) / high_52w * 100, 2) if high_52w > 0 else None
    pct_from_52w_low  = round((price - low_52w)  / low_52w  * 100, 2) if low_52w  > 0 else None

    vol_20  = df_pit.Volume.rolling(20).mean().iloc[-1]
    vol_now = df_pit.Volume.iloc[-1]
    vol_ratio = round(float(vol_now) / float(vol_20), 3) if vol_20 and vol_20 > 0 else None

    has_donchian   = int(today_sigs.get("Donchian",        0) ==  1)
    has_ema        = int(today_sigs.get("EMA Crossover",   0) ==  1)
    has_rsi_trend  = int(today_sigs.get("RSI Trend Shift", 0) ==  1)
    has_bollinger  = int(today_sigs.get("Bollinger",       0) ==  1)
    n_firing       = has_donchian + has_ema + has_rsi_trend + has_bollinger
    single_strat   = int(n_firing == 1)
    multi_strat    = int(n_firing >= 2)

    # Trade levels — IMPORTANT: these are computed off the signal-day close
    # because the levels feature describes the SETUP at decision time. The
    # actual fill happens at next-bar open, but the SL/target for the trade
    # are set against the close (and stay fixed). This matches what live
    # analyze.py:context() does.
    levels = dynamic_trade_levels(df_pit, len(df_pit) - 1, price, p)
    risk_pct   = levels.get("risk_pct")
    reward_pct = levels.get("reward_pct")
    rr_ratio   = levels.get("rr_ratio")
    rr_floor_applied = bool(levels.get("rr_floor_applied", False))
    sl_price   = levels.get("stop_loss")
    tgt_price  = levels.get("target")

    mom_vs_nifty_30d = None
    if nifty_pit is not None and len(nifty_pit) >= 31 and change_30d is not None:
        n_close = float(nifty_pit.Close.iloc[-1])
        n_prev  = float(nifty_pit.Close.iloc[-31])
        if n_prev > 0:
            n_ret_30d = (n_close - n_prev) / n_prev * 100
            mom_vs_nifty_30d = round(change_30d - n_ret_30d, 3)

    return dict(
        # Setup snapshot at signal-day close
        signal_close      = round(price, 2),
        rsi               = round(rsi_v, 2)  if math.isfinite(rsi_v) else None,
        macd_hist         = round(macd_h, 4),
        ema_bullish       = ema_bull,
        atr_pct           = atr_pct,
        change_1d         = change_1d,
        change_5d         = change_5d,
        change_30d        = change_30d,
        pct_from_52w_high = pct_from_52w_high,
        pct_from_52w_low  = pct_from_52w_low,
        vol_ratio         = vol_ratio,
        has_donchian      = has_donchian,
        has_ema           = has_ema,
        has_rsi_trend     = has_rsi_trend,
        has_bollinger     = has_bollinger,
        n_firing          = n_firing,
        single_strat      = single_strat,
        multi_strat       = multi_strat,
        risk_pct          = risk_pct,
        reward_pct        = reward_pct,
        rr_ratio          = rr_ratio,
        rr_floor_applied  = rr_floor_applied,
        sl_price          = sl_price,
        target_price      = tgt_price,
        regime            = regime_label,
        regime_score      = REGIME_SCORE.get(regime_label, 0.5),
        mom_vs_nifty_30d  = mom_vs_nifty_30d,
    )


# ─────────────────────────────────────────────
#  FORWARD SIMULATION  (v8: enters at NEXT-BAR OPEN)
# ─────────────────────────────────────────────
def simulate_forward(
    df: pd.DataFrame,
    signal_idx: int,
    sl_price: float | None,
    target_price: float | None,
    holding_period: int = HOLDING_PERIOD_DAYS,
) -> dict | None:
    """Walk forward from the bar AFTER signal_idx.

    v8 critical change vs pre-v8: entry price is df.Open.iloc[signal_idx + 1],
    not df.Close.iloc[signal_idx]. This matches the live agent's execution
    semantics and `agent/backsimulate.py`.

    Returns a dict including:
        was_win, actual_return_pct, exit_reason, days_held,
        entry_price, exit_price,
        mfe, mae, mfe_bar, mae_bar,        # for trailing-stop research
        gap_pct,                            # signal-close → entry-open gap
    or None if there isn't enough forward data.
    """
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None
    end_idx = min(entry_idx + holding_period - 1, len(df) - 1)
    if end_idx < entry_idx:
        return None

    try:
        entry_price = float(df.Open.iloc[entry_idx])
        signal_close = float(df.Close.iloc[signal_idx])
    except Exception:
        return None
    if not (math.isfinite(entry_price) and entry_price > 0):
        return None
    if not (math.isfinite(signal_close) and signal_close > 0):
        return None

    gap_pct = round((entry_price - signal_close) / signal_close * 100, 3)

    sl  = float(sl_price)     if sl_price     is not None and math.isfinite(float(sl_price))     else None
    tgt = float(target_price) if target_price is not None and math.isfinite(float(target_price)) else None

    # If the gap blew through SL on the entry bar's open, it's an immediate
    # stop. We model this honestly — the position fills at the open, then
    # the open IS already past the SL, so we exit at open with the loss.
    if sl is not None and entry_price <= sl:
        return dict(
            entry_price        = round(entry_price, 2),
            exit_price         = round(entry_price, 2),
            actual_return_pct  = 0.0,
            was_win            = False,
            exit_reason        = "gap_stop_at_open",
            days_held          = 0,
            mfe                = 0.0,
            mae                = 0.0,
            mfe_bar            = 0,
            mae_bar            = 0,
            gap_pct            = gap_pct,
        )

    mfe_pct = 0.0
    mae_pct = 0.0
    mfe_bar = 0
    mae_bar = 0

    for j in range(entry_idx, end_idx + 1):
        try:
            high  = float(df.High.iloc[j])
            low   = float(df.Low.iloc[j])
            close = float(df.Close.iloc[j])
            popen = float(df.Open.iloc[j])
        except Exception:
            continue
        if not all(math.isfinite(v) for v in (high, low, close, popen)):
            continue

        bar_offset = j - entry_idx
        excursion_high = (high - entry_price) / entry_price * 100
        excursion_low  = (low  - entry_price) / entry_price * 100
        if excursion_high > mfe_pct:
            mfe_pct = excursion_high
            mfe_bar = bar_offset
        if excursion_low < mae_pct:
            mae_pct = excursion_low
            mae_bar = bar_offset

        if j == entry_idx:
            # First bar = entry bar. We've already filled at popen above
            # for the entry. Don't re-check open. But intra-bar SL/target
            # checks below DO apply — we got filled at open and the bar's
            # extremes can hit SL/target the same day.
            pass

        # SL takes priority (worst-case for a long).
        if sl is not None and low <= sl:
            # Same-bar gap-through? Use the worse of bar open vs SL.
            fill = popen if (j == entry_idx and popen <= sl) else sl
            return dict(
                entry_price        = round(entry_price, 2),
                exit_price         = round(fill, 2),
                actual_return_pct  = round((fill - entry_price) / entry_price * 100, 3),
                was_win            = False,
                exit_reason        = "sl_hit",
                days_held          = bar_offset,
                mfe                = round(mfe_pct, 3),
                mae                = round(mae_pct, 3),
                mfe_bar            = mfe_bar,
                mae_bar            = mae_bar,
                gap_pct            = gap_pct,
            )
        if tgt is not None and high >= tgt:
            fill = popen if (j == entry_idx and popen >= tgt) else tgt
            return dict(
                entry_price        = round(entry_price, 2),
                exit_price         = round(fill, 2),
                actual_return_pct  = round((fill - entry_price) / entry_price * 100, 3),
                was_win            = True,
                exit_reason        = "target_hit",
                days_held          = bar_offset,
                mfe                = round(mfe_pct, 3),
                mae                = round(mae_pct, 3),
                mfe_bar            = mfe_bar,
                mae_bar            = mae_bar,
                gap_pct            = gap_pct,
            )

    # Timeout — close at last bar's close
    last_close = float(df.Close.iloc[end_idx])
    if not math.isfinite(last_close):
        return None
    ret = (last_close - entry_price) / entry_price * 100
    return dict(
        entry_price        = round(entry_price, 2),
        exit_price         = round(last_close, 2),
        actual_return_pct  = round(ret, 3),
        was_win            = ret > 0,
        exit_reason        = "timeout",
        days_held          = end_idx - entry_idx,
        mfe                = round(mfe_pct, 3),
        mae                = round(mae_pct, 3),
        mfe_bar            = mfe_bar,
        mae_bar            = mae_bar,
        gap_pct            = gap_pct,
    )


# ─────────────────────────────────────────────
#  TICKER RUNNER
# ─────────────────────────────────────────────
def process_ticker(
    ticker: str,
    df: pd.DataFrame,
    nifty_df: pd.DataFrame,
    regime_series: pd.Series,
    vix_df: pd.DataFrame | None,
    p: dict,
) -> list[dict]:
    rows: list[dict] = []
    if len(df) < WARMUP_DAYS + HOLDING_PERIOD_DAYS + 1:
        return rows

    # Strategy series — backward-looking, safe to compute on full df.
    sig_series = {
        "Donchian":        sig_donchian(df, p),
        "EMA Crossover":   sig_ema(df, p),
        "RSI Trend Shift": sig_rsi_trend_shift(df, p),
        "Bollinger":       sig_bb(df, p),
    }

    # Last index where we have entry+holding ahead. We need
    # signal_idx + 1 + (holding-1) <= len(df)-1, so:
    last_safe_idx = len(df) - HOLDING_PERIOD_DAYS - 1
    for i in range(WARMUP_DAYS, last_safe_idx + 1):
        today_sigs = {name: int(s.iloc[i]) for name, s in sig_series.items()}
        buy_count  = sum(1 for v in today_sigs.values() if v == 1)
        if buy_count == 0:
            continue

        signal_date_iso = df.index[i].date().isoformat()

        # Regime point-in-time. Generated by the SAME function the live
        # agent uses, so labels match.
        try:
            regime_label = regime_series.loc[df.index[i]] if df.index[i] in regime_series.index else None
        except Exception:
            regime_label = None
        if not isinstance(regime_label, str):
            prior = regime_series[regime_series.index <= df.index[i]]
            regime_label = prior.iloc[-1] if len(prior) else "UNKNOWN"

        df_pit    = df.iloc[: i + 1]
        nifty_pit = nifty_df[nifty_df.index <= df.index[i]] if nifty_df is not None else None

        feats = compute_features(df_pit, nifty_pit, regime_label, today_sigs, p)
        if not feats:
            continue

        vix_feats = vix_features_at(vix_df, df.index[i])

        # Forward simulation: enters at next-bar OPEN, uses signal-day SL/target.
        outcome = simulate_forward(
            df, i,
            sl_price=feats.get("sl_price"),
            target_price=feats.get("target_price"),
        )
        if outcome is None:
            continue

        # v8 gating reflects the LIVE agent: BUY rejected if multi_strat OR
        # BEARISH regime. Stored as a flag, not used to filter rows — the
        # model can still learn from gated-out rows for diagnostic purposes.
        gated_out = bool(feats["multi_strat"] == 1 or feats["regime"] == "BEARISH")

        # Data-quality flag. Used by score_model.py to optionally exclude
        # noisy rows during training.
        dq_flags = []
        if abs(outcome["gap_pct"]) > 2.0:
            dq_flags.append("large_gap")
        if feats.get("atr_pct") is not None and feats["atr_pct"] < 1.0:
            dq_flags.append("very_low_vol")
        if outcome["exit_reason"] == "timeout" and abs(outcome["actual_return_pct"]) < 1.0:
            dq_flags.append("borderline_timeout")
        if outcome["exit_reason"] == "gap_stop_at_open":
            dq_flags.append("gap_stop")

        rows.append({
            "ticker":       ticker,
            "signal_date":  signal_date_iso,
            "gated_out":    gated_out,
            **feats,
            **vix_feats,
            **outcome,
            "data_quality_flags": ",".join(dq_flags) if dq_flags else None,
        })

    return rows


# ─────────────────────────────────────────────
#  PERSISTENCE
# ─────────────────────────────────────────────
def get_supabase_client():
    if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY")):
        return None
    from supabase import create_client
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def write_rows(rows: list[dict], sb=None, csv_path: str | None = None) -> None:
    if not rows:
        return
    if csv_path:
        df_out = pd.DataFrame(rows)
        write_header = not os.path.exists(csv_path)
        df_out.to_csv(csv_path, mode="a", index=False, header=write_header)
    if sb is not None:
        for k in range(0, len(rows), BATCH_INSERT_SIZE):
            batch = rows[k : k + BATCH_INSERT_SIZE]
            try:
                sb.table("synthetic_training_data").insert(batch).execute()
            except Exception as e:
                print(f"  ⚠️  Supabase insert failed for batch of {len(batch)}: {e}")


# ─────────────────────────────────────────────
#  SELF-CHECK  (run with --self-check)
# ─────────────────────────────────────────────
def self_check() -> int:
    """End-to-end test that the generator's outputs match what the live agent
    produces for the same inputs. Returns exit code (0 = OK)."""
    print("\n🔬 Self-check — generator vs live agent parity\n")

    # Test 1: regime function is the SAME object as analyze.py uses
    print("[1/4] Regime function source-of-truth check...")
    from analyze import regime_from_close_series as live_regime_fn
    if regime_from_close_series is not live_regime_fn:
        print("   ❌ regime functions differ (import drift)"); return 1
    # Smoke test: noisy uptrend → BULLISH. We can't use a perfectly monotonic
    # series because RSI's denominator becomes 0 (no down-moves) → NaN, which
    # correctly maps to UNKNOWN. Real markets always have small pullbacks.
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    drift = np.linspace(0, 0.4, n)
    noise = rng.normal(0, 0.005, n).cumsum()
    uptrend = pd.Series(100 * (1 + drift) * (1 + noise), index=dates)
    label, score = regime_from_close_series(uptrend)
    assert label == "BULLISH", f"Expected BULLISH on noisy uptrend, got {label}"
    print(f"   ✅ regime fn shared, BULLISH on uptrend → score={score}")

    # Test 2: VIX function shared
    print("[2/4] VIX function source-of-truth check...")
    from analyze import vix_features_from_series as live_vix_fn
    if vix_features_from_series is not live_vix_fn:
        print("   ❌ VIX functions differ (import drift)"); return 1
    fake_vix = pd.Series([15, 16, 17, 16, 15, 14, 13] * 10,
                         index=pd.date_range("2024-01-01", periods=70, freq="B"))
    vf = vix_features_from_series(fake_vix)
    assert vf["vix_level"] is not None, "vix_level should not be None"
    assert vf["vix_change_5d"] is not None, "vix_change_5d should not be None"
    assert vf["vix_zscore_60d"] is not None, "vix_zscore_60d should not be None"
    print(f"   ✅ VIX fn shared, sample features = {vf}")

    # Test 3: simulate_forward enters at NEXT-bar open, not signal close
    print("[3/4] Entry-price look-ahead check...")
    n = 30
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    df_test = pd.DataFrame({
        "Open":  [100.0] * n,
        "High":  [101.0] * n,
        "Low":   [99.0]  * n,
        "Close": [100.5] * n,
        "Volume": [1_000_000] * n,
    }, index=dates)
    # Simulate from signal_idx=10. Manipulate the next bar's open to verify.
    df_test.loc[df_test.index[11], "Open"] = 105.0   # 4.5% gap up vs signal close
    df_test.loc[df_test.index[11], "High"] = 105.5
    df_test.loc[df_test.index[11], "Low"]  = 104.5
    out = simulate_forward(df_test, signal_idx=10, sl_price=98.0, target_price=110.0)
    assert out is not None, "simulate_forward returned None unexpectedly"
    assert abs(out["entry_price"] - 105.0) < 0.01, \
        f"Expected entry=105.0 (next-bar open), got {out['entry_price']}"
    assert abs(out["gap_pct"] - 4.478) < 0.01, \
        f"Expected gap_pct ≈ 4.478, got {out['gap_pct']}"
    print(f"   ✅ entry_price={out['entry_price']} (next-bar open), gap_pct={out['gap_pct']}%")

    # Test 4: gap_stop_at_open is handled
    print("[4/4] Gap-stop-at-open handling...")
    df_test2 = df_test.copy()
    df_test2.loc[df_test2.index[11], "Open"] = 95.0   # gap below SL=98
    df_test2.loc[df_test2.index[11], "High"] = 96.0
    df_test2.loc[df_test2.index[11], "Low"]  = 94.0
    out2 = simulate_forward(df_test2, signal_idx=10, sl_price=98.0, target_price=110.0)
    assert out2["exit_reason"] == "gap_stop_at_open", \
        f"Expected gap_stop_at_open, got {out2['exit_reason']}"
    assert out2["actual_return_pct"] == 0.0, \
        f"Expected 0% (filled at gap-down open, exited same), got {out2['actual_return_pct']}"
    print(f"   ✅ gap_stop_at_open handled correctly")

    print("\n✅ All self-checks passed.\n")
    return 0


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=None)
    ap.add_argument("--start",   default=DEFAULT_START)
    ap.add_argument("--end",     default=DEFAULT_END)
    ap.add_argument("--csv",     default=None)
    ap.add_argument("--purge",   action="store_true")
    ap.add_argument("--self-check", action="store_true",
                    help="Run end-to-end correctness checks and exit")
    args = ap.parse_args()

    if args.self_check:
        sys.exit(self_check())

    tickers = args.tickers.split(",") if args.tickers else DEFAULT_UNIVERSE
    print(f"\n📊 Synthetic training data generator (v8)")
    print(f"   Tickers : {len(tickers)}")
    print(f"   Window  : {args.start}  →  {args.end}")
    print(f"   Forward : {HOLDING_PERIOD_DAYS} bars")
    print(f"   Output  : {('csv:' + args.csv) if args.csv else ''} "
          f"{'+supabase' if os.environ.get('SUPABASE_URL') else ''}\n")

    sb = get_supabase_client()
    if sb is not None and args.purge:
        try:
            sb.table("synthetic_training_data").delete().neq("ticker", "__never__").execute()
            print("   Purged synthetic_training_data\n")
        except Exception as e:
            print(f"   ⚠️  Purge failed: {e}\n")
    if args.csv and os.path.exists(args.csv):
        os.remove(args.csv)

    print("   Fetching Nifty 50 benchmark...")
    nifty_df = fetch_nifty(args.start, args.end)
    if nifty_df is None:
        print("   ❌ Nifty fetch failed; aborting.")
        return
    print(f"   Nifty bars: {len(nifty_df)}")

    print("   Computing point-in-time regime series...")
    regime_series = compute_regime_series(nifty_df, DEFAULT_PARAMS)
    regime_counts = regime_series.value_counts(dropna=False).to_dict()
    print(f"   Regime distribution: {regime_counts}")

    print("   Fetching India VIX (^INDIAVIX)...")
    vix_df = fetch_vix_history(args.start, args.end)
    if vix_df is None or vix_df.empty:
        print("   ⚠️  VIX fetch failed; rows will have null VIX features.\n")
    else:
        print(f"   VIX bars: {len(vix_df)}\n")

    P = DEFAULT_PARAMS
    grand_total = 0
    skipped     = 0
    t0 = time.time()
    for n, ticker in enumerate(tickers, 1):
        df = fetch_ohlc(ticker, args.start, args.end)
        if df is None:
            skipped += 1
            print(f"   {n:>3}/{len(tickers)}  {ticker:<14}  ⚠️  insufficient data")
            continue
        rows = process_ticker(ticker, df, nifty_df, regime_series, vix_df, P)
        write_rows(rows, sb=sb, csv_path=args.csv)
        grand_total += len(rows)
        print(f"   {n:>3}/{len(tickers)}  {ticker:<14}  bars={len(df):4}  "
              f"signals={len(rows):4}  total={grand_total:6}")
        if n % 25 == 0:
            time.sleep(2)

    dt = time.time() - t0
    print(f"\n   ✅ Done in {dt/60:.1f}m")
    print(f"      Tickers processed : {len(tickers) - skipped}")
    print(f"      Tickers skipped   : {skipped} (insufficient history)")
    print(f"      Training rows     : {grand_total}")


if __name__ == "__main__":
    main()
