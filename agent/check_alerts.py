#!/usr/bin/env python3
import os
import sys
import warnings
from datetime import datetime
import yfinance as yf
from supabase import create_client, Client

warnings.filterwarnings("ignore")
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
sys.path.insert(0, os.path.dirname(__file__))
from telegram_alerts import send_exit_alert


def get_live_price(ticker: str) -> float:
    try:
        df = yf.download(ticker + ".NS", period="1d", progress=False, auto_adjust=True)
        if df.empty:
            return 0.0
        return float(df["Close"].iloc[-1])
    except Exception:
        return 0.0


def run():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n📱 Midday Alert Checker — {now}\n")
    try:
        res = supabase.table("paper_portfolio").select("*").eq("status", "OPEN").execute()
        positions = res.data or []
    except Exception as e:
        print(f" ❌ Failed to fetch portfolio: {e}")
        return
    if not positions:
        print(" ℹ️ No open positions to check")
        return
    alerts_sent = 0
    for row in positions:
        ticker = row.get("ticker")
        if not ticker:
            continue
        lp = get_live_price(ticker)
        if lp <= 0:
            continue
        sl = float(row.get("entry_stop_loss") or 0)
        tgt = float(row.get("entry_target") or 0)
        buy = float(row.get("buy_price") or 0)
        qty = int(row.get("quantity") or 0)
        if buy <= 0:
            continue
        pnl = round((lp - buy) / buy * 100, 2)
        if sl > 0 and lp <= sl:
            send_exit_alert({"ticker": ticker, "type": "SL_HIT", "lp": lp, "level": sl, "pnl": pnl, "id": row.get("id"), "buy_price": buy, "qty": qty})
            alerts_sent += 1
        elif tgt > 0 and lp >= tgt:
            send_exit_alert({"ticker": ticker, "type": "TARGET_HIT", "lp": lp, "level": tgt, "pnl": pnl, "id": row.get("id"), "buy_price": buy, "qty": qty})
            alerts_sent += 1
        else:
            print(f" ✅ {ticker:<12} ₹{lp:,.2f} P&L {pnl:+.1f}% — no alert")
    if alerts_sent == 0:
        print(f"\n No SL/target breaches found across {len(positions)} positions")
    else:
        print(f"\n 📱 {alerts_sent} alert(s) sent")
    print(" Done ✅\n")


if __name__ == "__main__":
    run()
