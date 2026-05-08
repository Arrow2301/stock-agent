#!/usr/bin/env python3
"""
============================================================
  Live Alert Checker — v2 (OHLC + persistence)
  Runs midday via GitHub Actions and on-demand from the dashboard.

  What's new in v2
  ────────────────
  • Uses agent.market_data.fetch_ohlc_since + detect_breach to find SL
    or target hits using the *full intraday range over every trading
    day since entry*, not just the latest close. This matches the
    semantics of agent/backsimulate.py and means live alerts will no
    longer miss intraday wicks that recover by close.
  • Persists the breach to the paper_portfolio row (breach_flag,
    breach_price, breach_date) so subsequent runs do not silently
    "un-alert" when price recovers.
  • Optional auto-close: if AUTO_CLOSE_ON_BREACH is set in env, the
    paper trade is closed automatically at the breach fill price.
    Default is False — alert only.
  • Handles gap-down / gap-up properly (uses bar open as fill price
    when the bar gapped past the level).
============================================================
"""

import os
import sys
import warnings
from datetime import datetime

from supabase import create_client, Client

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  ENV + DEPS
# ─────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
AUTO_CLOSE_ON_BREACH = os.environ.get("AUTO_CLOSE_ON_BREACH", "false").lower() == "true"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
sys.path.insert(0, os.path.dirname(__file__))

from market_data import fetch_ohlc_since, detect_breach  # noqa: E402
from telegram_alerts import send_exit_alert              # noqa: E402


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def _safe_float(v, default=0.0):
    try:
        if v is None:
            return default
        f = float(v)
        return f if f == f else default  # NaN check
    except Exception:
        return default


def _persist_breach(row_id: int, breach: dict, sell_now: bool, qty: int, buy_price: float) -> None:
    """
    Mark the paper_portfolio row as breached. Idempotent — re-running
    on an already-breached row is a no-op.
    """
    update = {
        "breach_flag":  breach["type"],            # "sl_hit" | "target_hit"
        "breach_price": breach["fill_price"],
        "breach_date":  breach["breach_date"],
    }
    if sell_now:
        sell_price = float(breach["fill_price"])
        pnl_pct = round((sell_price - buy_price) / buy_price * 100, 2) if buy_price else 0.0
        pnl_inr = round((sell_price - buy_price) * qty, 2)
        update.update({
            "status":      "CLOSED",
            "sell_date":   breach["breach_date"],
            "sell_price":  sell_price,
            "pnl_pct":     pnl_pct,
            "pnl_inr":     pnl_inr,
            "exit_reason": breach["type"],
        })
    try:
        supabase.table("paper_portfolio").update(update).eq("id", row_id).execute()
    except Exception as e:
        print(f"  ⚠️ Failed to persist breach for row {row_id}: {e}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run() -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n📱 Alert Checker v2 — {now}")
    print(f"   Auto-close on breach: {AUTO_CLOSE_ON_BREACH}\n")

    try:
        res = (
            supabase.table("paper_portfolio")
            .select("*")
            .eq("status", "OPEN")
            .execute()
        )
        positions = res.data or []
    except Exception as e:
        print(f"  ❌ Failed to fetch portfolio: {e}")
        return

    if not positions:
        print("  ℹ️  No open positions to check\n")
        return

    alerts_sent = 0
    skipped_already_flagged = 0
    no_data = 0

    for row in positions:
        ticker = row.get("ticker")
        if not ticker:
            continue

        # If a previous run already detected a breach and we're in alert-only
        # mode, do not re-alert. (Auto-close mode would have closed it.)
        if row.get("breach_flag"):
            skipped_already_flagged += 1
            continue

        sl  = _safe_float(row.get("entry_stop_loss"))
        tgt = _safe_float(row.get("entry_target"))
        buy = _safe_float(row.get("buy_price"))
        qty = int(row.get("quantity") or 0)
        buy_date = str(row.get("buy_date") or "")[:10]

        if buy <= 0 or not buy_date:
            continue
        if sl <= 0 and tgt <= 0:
            continue  # nothing to monitor

        ohlc = fetch_ohlc_since(ticker, buy_date)
        if ohlc.empty:
            no_data += 1
            print(f"  ⚠️  No OHLC for {ticker} since {buy_date}")
            continue

        breach = detect_breach(
            ohlc=ohlc,
            entry_price=buy,
            stop_loss=sl if sl > 0 else None,
            target=tgt if tgt > 0 else None,
        )

        if breach is None:
            last_close = float(ohlc.iloc[-1]["Close"])
            pnl = round((last_close - buy) / buy * 100, 2) if buy else 0.0
            print(f"  ✅ {ticker:<12} ₹{last_close:,.2f} P&L {pnl:+.1f}% — no alert")
            continue

        # We have a real breach. Send alert + persist.
        fill = float(breach["fill_price"])
        pnl  = round((fill - buy) / buy * 100, 2) if buy else 0.0
        gap_note = " (gap)" if breach.get("gapped") else ""
        print(
            f"  🚨 {ticker:<12} {breach['type']:<11} on {breach['breach_date']} "
            f"fill ₹{fill:,.2f}{gap_note}  P&L {pnl:+.1f}%"
        )

        send_exit_alert({
            "ticker":    ticker,
            "type":      "SL_HIT" if breach["type"] == "sl_hit" else "TARGET_HIT",
            "lp":        fill,
            "level":     breach["level"],
            "pnl":       pnl,
            "id":        row.get("id"),
            "buy_price": buy,
            "qty":       qty,
        })
        alerts_sent += 1

        _persist_breach(
            row_id=int(row["id"]),
            breach=breach,
            sell_now=AUTO_CLOSE_ON_BREACH,
            qty=qty,
            buy_price=buy,
        )

    print(
        f"\n  Summary: {alerts_sent} alert(s) sent | "
        f"{skipped_already_flagged} already flagged | {no_data} missing data"
    )
    print("  Done ✅\n")


if __name__ == "__main__":
    run()
