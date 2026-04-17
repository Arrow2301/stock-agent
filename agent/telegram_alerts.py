#!/usr/bin/env python3
import os
import requests
from datetime import datetime

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
TELEGRAM_API_BASE = "https://api.telegram.org/bot"


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("  ⚠️ Telegram not configured — skipping alert")
        return False
    try:
        url = f"{TELEGRAM_API_BASE}{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"  ⚠️ Telegram exception: {e}")
        return False


def send_morning_summary(buy_recs: list[dict], sell_recs: list[dict], meta: dict, breadth: dict):
    today = datetime.today().strftime("%A, %d %B %Y")
    regime = meta.get("market_regime", "UNKNOWN")
    regime_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(regime, "⬜")
    lines = [
        f"🇮🇳 <b>Stock Agent — {today}</b>",
        f"📊 Market: {regime_emoji} <b>{regime}</b>",
        f"📈 Breadth: {breadth.get('buys',0)} BUY / {breadth.get('sells',0)} SELL / {breadth.get('neutral',0)} Neutral",
        f"🎯 Total signals: <b>{len(buy_recs) + len(sell_recs)}</b>",
        "",
    ]
    if buy_recs:
        lines.append("🟢 <b>TOP BUYS</b>")
        for i, r in enumerate(buy_recs[:3], 1):
            price = float(r.get("price", 0) or 0)
            stop_loss = float(r.get("stop_loss", 0) or 0)
            target = float(r.get("target", 0) or 0)
            sl_pct = ((stop_loss - price) / price * 100) if price else 0
            target_pct = ((target - price) / price * 100) if price else 0

            lines.append(
                f"{i}. <b>{r['ticker']}</b> — Score <b>{r.get('composite_score',0):.0f}</b>/100 | "
                f"₹{price:,.0f} | "
                f"SL ₹{stop_loss:,.0f} ({sl_pct:+.2f}%) | "
                f"Target ₹{target:,.0f} ({target_pct:+.2f}%)"
            )
    send_message("\n".join(lines))


def send_exit_alert(alert: dict):
    ticker = alert["ticker"]
    lp = alert["lp"]
    level = float(alert["level"])
    pnl = alert["pnl"]
    kind = alert["type"]
    emoji = "🚨" if kind == "SL_HIT" else "🎯"
    title = "STOP-LOSS HIT" if kind == "SL_HIT" else "TARGET HIT"
    action = "Consider closing position immediately" if kind == "SL_HIT" else "Consider booking profits"
    send_message(
        f"{emoji} <b>{title} — {ticker}</b>\n\n"
        f"Live Price: ₹{lp:,.2f}\n"
        f"{'Stop Loss' if kind == 'SL_HIT' else 'Target'}: ₹{level:,.2f}\n"
        f"P&amp;L: <b>{pnl:+.2f}%</b>\n\n💡 {action}"
    )


def send_optimizer_summary(champion: dict, challenger: dict, best_score: float):
    if not challenger:
        return
    delta = best_score - float(champion.get("objective_score", 0) if champion else 0)
    verdict = "🟢 Challenger is BETTER" if delta > 0 else "🔴 Champion still ahead"
    send_message(
        f"🔬 <b>Weekly Optimizer Complete</b>\n\n"
        f"👑 Champion: v{champion.get('version','?') if champion else '?'}\n"
        f"⚔️ Challenger: v{challenger.get('version','?')} (score={best_score:.4f})\n"
        f"{verdict}"
    )
