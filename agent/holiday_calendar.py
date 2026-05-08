#!/usr/bin/env python3
"""
============================================================
  NSE Holiday Calendar
  ────────────────────
  The daily analysis workflow runs Mon–Fri but does not currently
  check whether NSE was actually open. On Indian market holidays
  (Republic Day, Holi, Diwali, etc.) yfinance silently returns the
  previous trading day's close, and the agent generates "today's"
  signals from stale data.

  This module exposes:
    is_market_open(today=None) -> bool
    next_market_day(d)         -> date

  Two lookup paths:
    1. A static, hand-curated list (this file). Reliable, but you
       must update it once a year.
    2. Optional Supabase fallback to the `market_holidays` table
       populated by the v7 migration. Lets you correct mistakes
       without a redeploy.

  The static list ships with the 2026 NSE holiday calendar. Update
  HOLIDAYS once a year — NSE publishes the next-year calendar in
  early November of the prior year.
============================================================
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Iterable

# ─────────────────────────────────────────────
#  STATIC NSE HOLIDAY LIST (UPDATE ANNUALLY)
#  Source: https://www.nseindia.com/resources/exchange-communication-holidays
#  Trading holidays only (settlement-only and Muhurat sessions excluded).
# ─────────────────────────────────────────────
HOLIDAYS_2026: tuple[str, ...] = (
    # NOTE: These are illustrative. Verify against the NSE 2026 calendar
    # before relying on them. Update this tuple each year.
    "2026-01-26",  # Republic Day (Mon)
    "2026-03-04",  # Holi (Wed)
    "2026-03-19",  # Eid-ul-Fitr (Thu)
    "2026-04-03",  # Good Friday (Fri)
    "2026-04-14",  # Ambedkar Jayanti (Tue)
    "2026-05-01",  # Maharashtra Day (Fri)
    "2026-08-15",  # Independence Day (Sat — already weekend)
    "2026-08-26",  # Ganesh Chaturthi (Wed)
    "2026-10-02",  # Gandhi Jayanti (Fri)
    "2026-10-21",  # Dussehra (Wed)
    "2026-11-09",  # Diwali Laxmi Pujan (Mon)  Muhurat trading typically held this evening
    "2026-11-10",  # Diwali Balipratipada (Tue)
    "2026-11-25",  # Guru Nanak Jayanti (Wed)
    "2026-12-25",  # Christmas (Fri)
)


def _parse(holidays: Iterable[str]) -> set[date]:
    out: set[date] = set()
    for h in holidays:
        try:
            out.add(datetime.strptime(h, "%Y-%m-%d").date())
        except ValueError:
            continue
    return out


_STATIC_HOLIDAYS: set[date] = _parse(HOLIDAYS_2026)


# ─────────────────────────────────────────────
#  SUPABASE FALLBACK (optional)
# ─────────────────────────────────────────────
def _fetch_holidays_from_supabase() -> set[date]:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        return set()
    try:
        from supabase import create_client
        sb = create_client(url, key)
        res = sb.table("market_holidays").select("holiday_date").eq("market", "NSE").execute()
        return _parse(str(r["holiday_date"])[:10] for r in (res.data or []))
    except Exception:
        return set()


# ─────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────
def is_market_open(today: date | None = None, *, allow_supabase: bool = True) -> bool:
    """
    Return True if NSE is open for normal trading on `today`.

    Closed on weekends and any date present in either the static list
    or the optional Supabase market_holidays table.
    """
    today = today or date.today()
    if today.weekday() >= 5:
        return False
    if today in _STATIC_HOLIDAYS:
        return False
    if allow_supabase and today in _fetch_holidays_from_supabase():
        return False
    return True


def next_market_day(d: date | None = None) -> date:
    """First trading day on or after `d`."""
    cur = d or date.today()
    for _ in range(15):
        if is_market_open(cur, allow_supabase=False):
            return cur
        cur += timedelta(days=1)
    return cur


if __name__ == "__main__":
    today = date.today()
    print(f"Today is {today} ({today.strftime('%A')})")
    print(f"  Market open?     {is_market_open(today)}")
    print(f"  Next market day: {next_market_day(today)}")
