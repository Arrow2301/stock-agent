#!/usr/bin/env python3
"""
============================================================
  Market Data + Breach Detection — shared utility module
  Used by both check_alerts.py and dashboard/app.py.

  Why this module exists
  ──────────────────────
  Live SL/target hit detection used to compare *latest close* to the
  level. That misses intraday wicks: a stock that prints a low below
  the stop-loss at 10:30 AM and recovers above it by 3:30 PM would
  never trigger an alert with close-only logic, yet a real broker
  with a stop-loss order would have filled at the SL.

  This module fetches the full OHLC history since entry day and
  detects breaches over the entire holding period, matching the
  semantics of the historical backtester (`agent/backsimulate.py`)
  so live and backtest results stay consistent.

  Conventions (MUST match backsimulate.py for parity)
  ────────────────────────────────────────────────────
  • Long-only.
  • Stop-first when both SL and target are touched in the same bar.
    This is the conservative assumption since daily OHLC does not
    tell us the intraday sequence.
  • Gap handling on entry day: if the entry-day open is already
    below SL or above target (rare, but possible), we use the open
    price as the fill, not the SL / target level.
  • All breaches are reported with the bar index (days_held) and
    breach price so persistence layers can store them durably.
============================================================
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf


# ─────────────────────────────────────────────
#  OHLC FETCH
# ─────────────────────────────────────────────
def fetch_ohlc_since(
    ticker: str,
    start_date: str | date | datetime,
    end_date: Optional[str | date | datetime] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of [Open, High, Low, Close, Volume] from start_date
    through end_date (inclusive). Tolerates yfinance's occasional
    multi-level column responses. Returns an empty DataFrame on error.

    Indian NSE convention: ticker is the bare symbol (e.g. "RELIANCE"),
    we append ".NS" before calling yfinance.
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date[:10], "%Y-%m-%d").date()
    if isinstance(start_date, datetime):
        start_date = start_date.date()

    if end_date is None:
        end_date = date.today() + timedelta(days=1)
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date[:10], "%Y-%m-%d").date()
    elif isinstance(end_date, datetime):
        end_date = end_date.date()

    try:
        df = yf.download(
            ticker + ".NS",
            start=start_date,
            end=end_date + timedelta(days=1),  # yf end is exclusive
            progress=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        # yfinance occasionally returns MultiIndex columns like
        # ("Close", "RELIANCE.NS"). Flatten defensively.
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        if not keep:
            return pd.DataFrame()
        return df[keep].dropna(how="all").sort_index()
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
#  BREACH DETECTION
# ─────────────────────────────────────────────
def detect_breach(
    ohlc: pd.DataFrame,
    entry_price: float,
    stop_loss: Optional[float],
    target: Optional[float],
) -> Optional[dict]:
    """
    Walk forward through `ohlc` (one row per trading day) and return the
    first SL or target breach for a long position, or None if neither
    has been touched yet.

    Returns a dict like::

        {
            "type": "sl_hit" | "target_hit",
            "level": <SL or target price>,    # the published level
            "fill_price": <price at which a stop/target would have filled>,
            "breach_date": "YYYY-MM-DD",
            "day_high": <intraday high>,
            "day_low":  <intraday low>,
            "days_held": <int, 0-indexed: 0 = entry day>,
            "gapped":   <bool — True if the bar opened past the level>,
        }

    Semantics
    ─────────
    - For each bar i (from entry day onward), we check:
        a) Gap: did the bar open past the level? Then fill = open.
        b) Otherwise, did Low <= SL during the day? Stop hit at SL.
        c) Otherwise, did High >= target during the day? Target at target.
    - Stop-first when both touched on the same bar (conservative).
    """
    if ohlc is None or ohlc.empty:
        return None
    if entry_price is None or entry_price <= 0:
        return None

    sl = float(stop_loss) if stop_loss not in (None, 0, 0.0) and stop_loss > 0 else None
    tgt = float(target) if target not in (None, 0, 0.0) and target > 0 else None
    if sl is None and tgt is None:
        return None

    for i, (idx, row) in enumerate(ohlc.iterrows()):
        try:
            o = float(row["Open"])
            h = float(row["High"])
            l = float(row["Low"])
        except (KeyError, ValueError, TypeError):
            continue

        bar_date = idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]

        # Stop check (priority 1 — conservative)
        if sl is not None:
            if o <= sl:
                # Bar gapped down through the stop. Real fill is the open.
                return {
                    "type": "sl_hit",
                    "level": sl,
                    "fill_price": o,
                    "breach_date": bar_date,
                    "day_high": h,
                    "day_low": l,
                    "days_held": i,
                    "gapped": True,
                }
            if l <= sl:
                return {
                    "type": "sl_hit",
                    "level": sl,
                    "fill_price": sl,
                    "breach_date": bar_date,
                    "day_high": h,
                    "day_low": l,
                    "days_held": i,
                    "gapped": False,
                }

        # Target check
        if tgt is not None:
            if o >= tgt:
                return {
                    "type": "target_hit",
                    "level": tgt,
                    "fill_price": o,
                    "breach_date": bar_date,
                    "day_high": h,
                    "day_low": l,
                    "days_held": i,
                    "gapped": True,
                }
            if h >= tgt:
                return {
                    "type": "target_hit",
                    "level": tgt,
                    "fill_price": tgt,
                    "breach_date": bar_date,
                    "day_high": h,
                    "day_low": l,
                    "days_held": i,
                    "gapped": False,
                }
    return None


# ─────────────────────────────────────────────
#  CONVENIENCE: live snapshot
# ─────────────────────────────────────────────
def latest_snapshot(ticker: str) -> Optional[dict]:
    """
    Return the most recent OHLC bar for a ticker as a dict, or None
    if data is unavailable. During market hours yfinance returns a
    provisional bar with the running session's High/Low/Close so far.
    """
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=7)  # short lookback, last completed bar
    df = fetch_ohlc_since(ticker, start, end)
    if df.empty:
        return None
    last = df.iloc[-1]
    return {
        "date": df.index[-1].date().isoformat() if hasattr(df.index[-1], "date") else str(df.index[-1])[:10],
        "open":   float(last.get("Open", 0)),
        "high":   float(last.get("High", 0)),
        "low":    float(last.get("Low", 0)),
        "close":  float(last.get("Close", 0)),
        "volume": int(last.get("Volume", 0)) if "Volume" in last else 0,
    }
