#!/usr/bin/env python3
"""
============================================================
  Portfolio Queries (new in v8)

  Single place to read paper_portfolio. Used by:
    * agent/analyze.py     — EXIT-vs-portfolio guard (REVIEW A3)
                             + sector exposure + drawdown brake
    * dashboard/app.py     — same calls, identical results
    * agent/check_alerts.py — already reads paper_portfolio
                             directly; that's fine, single-purpose.

  All functions return plain Python dicts/lists (no DataFrames)
  so they're cheap and trivial to mock in tests.
============================================================
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable


def list_open_positions(sb) -> list[dict]:
    """All currently-OPEN paper trades."""
    try:
        res = (
            sb.table("paper_portfolio")
            .select("*")
            .eq("status", "OPEN")
            .execute()
        )
        return res.data or []
    except Exception:
        return []


def open_tickers(sb) -> set[str]:
    """Set of tickers with ≥1 open position. Used by the EXIT guard."""
    return {p["ticker"] for p in list_open_positions(sb) if p.get("ticker")}


def recent_closed_pnl_pcts(sb, n: int = 20) -> list[float]:
    """
    pnl_pct of the most recent `n` CLOSED paper trades, newest first.
    Used by the drawdown brake. Empty list if none yet.
    """
    try:
        res = (
            sb.table("paper_portfolio")
            .select("pnl_pct, sell_date")
            .eq("status", "CLOSED")
            .order("sell_date", desc=True)
            .limit(n)
            .execute()
        )
        out: list[float] = []
        for r in res.data or []:
            v = r.get("pnl_pct")
            if v is None:
                continue
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                continue
        return out
    except Exception:
        return []


def open_positions_with_sector(sb) -> list[dict]:
    """
    Open positions, augmented with `sector` looked up from
    stock_fundamentals. Falls back to `recommendations.sector`
    keyed by recommendation_id, then to None. Used by the sector
    cap. Returns dicts with at least: ticker, buy_price, quantity, sector.
    """
    positions = list_open_positions(sb)
    if not positions:
        return []

    # Pull sectors from fundamentals (one query)
    try:
        f_res = (
            sb.table("stock_fundamentals")
            .select("ticker, sector")
            .execute()
        )
        sec_by_ticker: dict[str, str] = {
            r["ticker"]: r.get("sector") for r in (f_res.data or [])
            if r.get("ticker")
        }
    except Exception:
        sec_by_ticker = {}

    # For positions whose ticker isn't in fundamentals yet, try the
    # recommendation row. Cheaper than re-fetching from yfinance.
    needs_lookup = [p["recommendation_id"] for p in positions
                    if p.get("ticker") and not sec_by_ticker.get(p["ticker"])
                    and p.get("recommendation_id")]
    rec_sector: dict[int, str] = {}
    if needs_lookup:
        try:
            r_res = (
                sb.table("recommendations")
                .select("id, sector")
                .in_("id", needs_lookup)
                .execute()
            )
            rec_sector = {r["id"]: r.get("sector") for r in (r_res.data or [])}
        except Exception:
            rec_sector = {}

    out: list[dict] = []
    for p in positions:
        ticker = p.get("ticker")
        sector = sec_by_ticker.get(ticker)
        if not sector and p.get("recommendation_id"):
            sector = rec_sector.get(p["recommendation_id"])
        out.append({
            "ticker":     ticker,
            "buy_price":  p.get("buy_price"),
            "quantity":   p.get("quantity"),
            "sector":     sector,
            "buy_date":   p.get("buy_date"),
            "id":         p.get("id"),
        })
    return out


def stale_open_positions(sb, max_hold_days: int = 15) -> list[dict]:
    """
    Open positions with buy_date older than `max_hold_days`.
    Used by health_check — also useful for a dashboard banner.
    """
    cutoff = (date.today() - timedelta(days=max_hold_days)).isoformat()
    try:
        res = (
            sb.table("paper_portfolio")
            .select("id, ticker, buy_date")
            .eq("status", "OPEN")
            .lt("buy_date", cutoff)
            .execute()
        )
        return res.data or []
    except Exception:
        return []


__all__ = [
    "list_open_positions",
    "open_tickers",
    "recent_closed_pnl_pcts",
    "open_positions_with_sector",
    "stale_open_positions",
]
