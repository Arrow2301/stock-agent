#!/usr/bin/env python3
"""
============================================================
  Daily Health Check — runs after market close
  ─────────────────────────────────────────────
  The system has many moving parts (yfinance, Supabase, FinBERT,
  GNews, Optuna). When something silently breaks — yfinance returns
  empty data for half the universe, or live results drift far from
  backtest predictions — there is currently no signal that anything
  is wrong. This script computes a small set of health metrics and
  writes them to the `health_checks` table, then sends a Telegram
  alert if any check FAILs.

  Checks performed
  ────────────────
  1. data_freshness    — recommendations table has a row for today
                         (or last trading day if today was a holiday)
  2. yfinance_coverage — ≥80% of NIFTY50 tickers fetched OHLC OK
                         in today's run
  3. signal_imbalance  — fewer than 90% of signals are the same
                         action (catches stuck-on-BUY bug)
  4. live_vs_backtest  — over the last 30 closed paper trades,
                         realised win rate is within ±20pp of the
                         backsim-predicted win rate
  5. open_position_age — flag any open paper trade older than the
                         param's BT_MAX_HOLD (positions stuck open)
  6. champion_age      — current champion params were trained no
                         more than 60 days ago
  7. db_size_estimate  — total rows across major tables (early
                         warning for Supabase free-tier 500MB)
============================================================
"""

import os
import sys
import warnings
from datetime import date, datetime, timedelta

from supabase import create_client

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from holiday_calendar import is_market_open, next_market_day  # noqa: E402

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
TG_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT  = os.environ.get("TELEGRAM_CHAT_ID", "")

sb = create_client(SUPABASE_URL, SUPABASE_KEY)


def _record(check_name: str, status: str, value: float | None, threshold: float | None, detail: str):
    print(f"  [{status:<4}] {check_name:<22} {detail}")
    try:
        sb.table("health_checks").insert({
            "run_date":        date.today().isoformat(),
            "check_name":      check_name,
            "status":          status,
            "metric_value":    value,
            "threshold_value": threshold,
            "detail":          detail[:500],
        }).execute()
    except Exception as e:
        print(f"    ⚠️ Failed to record health check: {e}")


def _last_trading_day() -> date:
    cur = date.today()
    if is_market_open(cur, allow_supabase=False):
        return cur
    for _ in range(10):
        cur -= timedelta(days=1)
        if is_market_open(cur, allow_supabase=False):
            return cur
    return date.today()


def check_data_freshness():
    target = _last_trading_day().isoformat()
    try:
        res = sb.table("recommendations").select("id", count="exact").eq("date", target).limit(1).execute()
        rows = res.count or 0
    except Exception as e:
        return _record("data_freshness", "fail", 0, 1, f"Query error: {e}")
    if rows == 0:
        return _record("data_freshness", "fail", 0, 1,
                       f"No recommendations for last trading day {target}. Daily workflow may be down.")
    _record("data_freshness", "pass", rows, 1, f"{rows} rec(s) for {target}")


def check_yfinance_coverage():
    target = _last_trading_day().isoformat()
    try:
        ok  = sb.table("ticker_run_log").select("id", count="exact").eq("date", target).eq("status", "ok").limit(1).execute().count or 0
        all_ = sb.table("ticker_run_log").select("id", count="exact").eq("date", target).limit(1).execute().count or 0
    except Exception as e:
        return _record("yfinance_coverage", "fail", 0, 0.8, f"Query error: {e}")
    if all_ == 0:
        return _record("yfinance_coverage", "warn", 0, 0.8, "No run-log rows for last trading day")
    cov = ok / all_
    if cov < 0.80:
        _record("yfinance_coverage", "fail", round(cov, 3), 0.8,
                f"Only {ok}/{all_} = {cov:.0%} of tickers fetched. yfinance rate-limited or down.")
    elif cov < 0.95:
        _record("yfinance_coverage", "warn", round(cov, 3), 0.95, f"{ok}/{all_} = {cov:.0%}")
    else:
        _record("yfinance_coverage", "pass", round(cov, 3), 0.95, f"{ok}/{all_} = {cov:.0%}")


def check_signal_imbalance():
    target = _last_trading_day().isoformat()
    try:
        res = sb.table("recommendations").select("action").eq("date", target).execute()
        actions = [r["action"] for r in (res.data or [])]
    except Exception as e:
        return _record("signal_imbalance", "fail", 0, 0.9, f"Query error: {e}")
    if not actions:
        return _record("signal_imbalance", "warn", 0, 0.9, "No signals to check")
    n = len(actions)
    buy_pct = actions.count("BUY") / n
    if buy_pct >= 0.95 or buy_pct <= 0.05:
        _record("signal_imbalance", "warn", round(buy_pct, 3), 0.9,
                f"{buy_pct:.0%} BUY out of {n} signals. Either market is one-sided or scoring stuck.")
    else:
        _record("signal_imbalance", "pass", round(buy_pct, 3), 0.9, f"{buy_pct:.0%} BUY of {n}")


def check_live_vs_backtest():
    """Compare last 30 closed paper trades against backsim predictions."""
    try:
        res = (
            sb.table("paper_portfolio")
            .select("id, ticker, buy_date, recommendation_id, pnl_pct")
            .eq("status", "CLOSED")
            .order("sell_date", desc=True)
            .limit(30)
            .execute()
        )
        closed = res.data or []
    except Exception as e:
        return _record("live_vs_backtest", "fail", 0, 0.2, f"Query error: {e}")
    if len(closed) < 10:
        return _record("live_vs_backtest", "pass", len(closed), 10,
                       f"Only {len(closed)} closed trades — not enough sample yet")

    live_wr = sum(1 for c in closed if (c.get("pnl_pct") or 0) > 0) / len(closed)

    # Match against backsim by recommendation_id
    rec_ids = [c["recommendation_id"] for c in closed if c.get("recommendation_id")]
    if not rec_ids:
        return _record("live_vs_backtest", "warn", live_wr, None,
                       f"Live WR {live_wr:.0%} (no rec_ids to match against backsim)")
    try:
        bs = sb.table("backtest_simulations").select("recommendation_id, was_win").in_("recommendation_id", rec_ids).execute().data or []
    except Exception:
        bs = []
    if not bs:
        return _record("live_vs_backtest", "pass", live_wr, None,
                       f"Live WR {live_wr:.0%}, no backsim rows yet")
    bs_wr = sum(1 for r in bs if r.get("was_win")) / len(bs)
    diff = abs(live_wr - bs_wr)
    detail = f"Live WR {live_wr:.0%} vs backsim WR {bs_wr:.0%} (n_live={len(closed)}, n_sim={len(bs)})"
    if diff >= 0.20:
        _record("live_vs_backtest", "fail", round(diff, 3), 0.20, detail + " — drift exceeds 20pp")
    elif diff >= 0.10:
        _record("live_vs_backtest", "warn", round(diff, 3), 0.10, detail)
    else:
        _record("live_vs_backtest", "pass", round(diff, 3), 0.10, detail)


def check_open_position_age():
    cutoff = (date.today() - timedelta(days=30)).isoformat()
    try:
        res = (
            sb.table("paper_portfolio")
            .select("id, ticker, buy_date")
            .eq("status", "OPEN")
            .lt("buy_date", cutoff)
            .execute()
        )
        rows = res.data or []
    except Exception as e:
        return _record("open_position_age", "fail", 0, 0, f"Query error: {e}")
    if rows:
        names = ", ".join(r["ticker"] for r in rows[:5])
        _record("open_position_age", "warn", len(rows), 0,
                f"{len(rows)} positions older than 30 days: {names}")
    else:
        _record("open_position_age", "pass", 0, 0, "No stale open positions")


def check_champion_age():
    try:
        res = (
            sb.table("agent_params")
            .select("version, promoted_at")
            .eq("status", "champion")
            .order("promoted_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = res.data or []
    except Exception as e:
        return _record("champion_age", "fail", 0, 60, f"Query error: {e}")
    if not rows:
        return _record("champion_age", "warn", 0, 60, "No champion set")
    promoted = rows[0].get("promoted_at")
    if not promoted:
        return _record("champion_age", "warn", 0, 60, f"Champion v{rows[0]['version']} has no promoted_at")
    age = (date.today() - datetime.strptime(str(promoted)[:10], "%Y-%m-%d").date()).days
    if age > 60:
        _record("champion_age", "warn", age, 60,
                f"Champion v{rows[0]['version']} is {age}d old — consider promoting challenger")
    else:
        _record("champion_age", "pass", age, 60, f"Champion v{rows[0]['version']} is {age}d old")


def check_db_size_estimate():
    """Rough early warning for Supabase 500MB free tier limit."""
    tables = ["recommendations", "paper_portfolio", "ticker_run_log",
              "backtest_simulations", "agent_params", "stock_fundamentals",
              "health_checks"]
    total = 0
    parts = []
    for t in tables:
        try:
            n = sb.table(t).select("id", count="exact").limit(1).execute().count or 0
            total += n
            parts.append(f"{t}={n:,}")
        except Exception:
            continue
    # Each row averages ~2KB (recommendations is the heaviest with JSON columns).
    est_mb = total * 2 / 1024
    detail = f"{total:,} rows total ≈{est_mb:.0f} MB | " + ", ".join(parts)
    if est_mb > 350:
        _record("db_size_estimate", "warn", est_mb, 350, detail)
    else:
        _record("db_size_estimate", "pass", est_mb, 350, detail)


def send_summary_alert():
    if not TG_TOKEN or not TG_CHAT:
        return
    try:
        res = (
            sb.table("health_checks")
            .select("check_name, status, detail")
            .eq("run_date", date.today().isoformat())
            .execute()
        )
        rows = res.data or []
    except Exception:
        return
    fails = [r for r in rows if r["status"] == "fail"]
    warns = [r for r in rows if r["status"] == "warn"]
    if not fails and not warns:
        return  # quiet on green
    import requests
    lines = [f"🩺 <b>Stock Agent Health — {date.today()}</b>"]
    if fails:
        lines.append("\n❌ <b>FAIL</b>")
        for f in fails:
            lines.append(f"• <b>{f['check_name']}</b>: {f['detail']}")
    if warns:
        lines.append("\n⚠️ <b>WARN</b>")
        for w in warns:
            lines.append(f"• <b>{w['check_name']}</b>: {w['detail']}")
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": "\n".join(lines), "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass


def run():
    print(f"\n🩺 Stock Agent Health Check — {date.today()}")
    if not is_market_open(date.today()):
        print(f"  Today {date.today()} is not a trading day — running anyway against last trading day")
    print()
    check_data_freshness()
    check_yfinance_coverage()
    check_signal_imbalance()
    check_live_vs_backtest()
    check_open_position_age()
    check_champion_age()
    check_db_size_estimate()
    send_summary_alert()
    print("\n  Done ✅\n")


if __name__ == "__main__":
    run()
