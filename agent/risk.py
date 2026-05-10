#!/usr/bin/env python3
"""
============================================================
  Position Sizing + Portfolio Risk Gates  (new in v8)

  This module exists because the previous versions decoupled
  risk from volatility — every trade was a flat user-typed
  quantity, so a high-ATR stock and a low-ATR stock with the
  same nominal qty had wildly different cash risk.

  Implements four standard quant primitives:

    1. Fixed-fractional position sizing      → suggest_qty()
    2. Regime-conditional sizing multiplier  → regime_size_mult()
    3. Sector concentration cap              → sector_exposure_pct()
    4. Recent-drawdown brake                 → drawdown_brake()

  Every function is pure (no I/O) so they can be called from
  both the agent (analyze.py) and the dashboard (app.py)
  without coupling either to Supabase or yfinance.
============================================================
"""
from __future__ import annotations

from typing import Iterable


# ─────────────────────────────────────────────
#  Fixed-fractional sizing
# ─────────────────────────────────────────────
def suggest_qty(
    capital_inr: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    *,
    max_exposure_pct: float = 10.0,
) -> int:
    """
    Recommended quantity for a long entry.

    capital_inr      : total account size in INR (₹)
    risk_pct         : max % of capital you're willing to lose if SL hits.
                       1.0 = 1%. Industry standard is 0.5%-2.0%.
    entry_price      : intended entry price
    stop_loss        : stop-loss price (must be < entry_price)
    max_exposure_pct : cap on a single position's notional, as % of capital.
                       Default 10% prevents one-trade concentration when
                       a tight SL would otherwise allow a huge qty.

    Returns int qty. Returns 0 if SL is invalid or non-protective.
    """
    if capital_inr <= 0 or entry_price <= 0:
        return 0
    risk_per_share = entry_price - stop_loss
    if risk_per_share <= 0:
        # SL above or at entry — invalid for a long.
        return 0
    risk_inr = capital_inr * (risk_pct / 100.0)
    qty_by_risk = int(risk_inr // risk_per_share)
    qty_by_exposure = int((capital_inr * (max_exposure_pct / 100.0)) // entry_price)
    return max(0, min(qty_by_risk, qty_by_exposure))


# ─────────────────────────────────────────────
#  Regime-conditional sizing multiplier
# ─────────────────────────────────────────────
_REGIME_SIZE: dict[str, float] = {
    "BULLISH": 1.00,
    "NEUTRAL": 0.75,
    "BEARISH": 0.50,
    "UNKNOWN": 0.50,
}


def regime_size_mult(regime_label: str | None) -> float:
    """Return a [0, 1] multiplier on base position size, conditioned on regime."""
    return _REGIME_SIZE.get((regime_label or "UNKNOWN").upper(), 0.75)


# ─────────────────────────────────────────────
#  Sector concentration cap
# ─────────────────────────────────────────────
def sector_exposure_pct(
    open_positions: Iterable[dict],
    sector: str | None,
    capital_inr: float,
) -> float:
    """
    Current % of capital deployed in a given sector.

    open_positions is an iterable of dicts with at least
    'buy_price', 'quantity', 'sector'. capital_inr is total
    account size — keep > 0 or this returns 0.0.
    """
    if not sector or capital_inr <= 0:
        return 0.0
    sector_l = sector.strip().lower()
    deployed = 0.0
    for p in open_positions:
        ps = (p.get("sector") or "").strip().lower()
        if ps != sector_l:
            continue
        try:
            deployed += float(p.get("buy_price") or 0) * float(p.get("quantity") or 0)
        except (TypeError, ValueError):
            continue
    return round(deployed / capital_inr * 100.0, 2)


def sector_room_remaining_pct(
    open_positions: Iterable[dict],
    sector: str | None,
    capital_inr: float,
    *,
    sector_cap_pct: float = 30.0,
) -> float:
    """How many percent of capital are still available in this sector before the cap."""
    used = sector_exposure_pct(open_positions, sector, capital_inr)
    return max(0.0, sector_cap_pct - used)


# ─────────────────────────────────────────────
#  Recent-drawdown brake
# ─────────────────────────────────────────────
def drawdown_brake(recent_pnl_pcts: list[float]) -> tuple[float, str]:
    """
    Reduce or pause sizing based on the recent closed-trade P&L stream.

    Pass the pnl_pct of the most recent N closed trades (suggested N=20).
    Returns (size_multiplier, reason). The multiplier is one of:
       1.00 — no brake
       0.50 — soft brake (≥ 5% drawdown across the window)
       0.00 — hard brake / pause new entries (≥ 10% drawdown)
    """
    if not recent_pnl_pcts:
        return 1.00, "no recent trades"
    cum = sum(recent_pnl_pcts)
    if cum <= -10.0:
        return 0.00, f"hard brake ({cum:+.1f}% over last {len(recent_pnl_pcts)})"
    if cum <= -5.0:
        return 0.50, f"soft brake ({cum:+.1f}% over last {len(recent_pnl_pcts)})"
    return 1.00, f"normal sizing ({cum:+.1f}% over last {len(recent_pnl_pcts)})"


# ─────────────────────────────────────────────
#  Combined sizing recommendation
# ─────────────────────────────────────────────
def recommend_position_size(
    capital_inr: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    *,
    regime_label: str | None = None,
    recent_pnl_pcts: list[float] | None = None,
    open_positions: Iterable[dict] | None = None,
    sector: str | None = None,
    max_exposure_pct: float = 10.0,
    sector_cap_pct: float = 30.0,
) -> dict:
    """
    Combine all four primitives into a single recommendation.

    Returns a dict::

        {
          "qty": int,              # recommended quantity (post-multipliers)
          "base_qty": int,         # qty from pure fixed-fractional sizing
          "regime_mult": float,    # regime size multiplier in [0, 1]
          "drawdown_mult": float,  # drawdown brake multiplier in [0, 1]
          "sector_room_pct": float | None,  # % capital still free in this sector
          "blocked": bool,         # True if any gate forced qty to 0
          "reasons": list[str],    # human-readable trail
        }
    """
    reasons: list[str] = []
    base = suggest_qty(
        capital_inr, risk_pct, entry_price, stop_loss,
        max_exposure_pct=max_exposure_pct,
    )
    if base == 0:
        reasons.append("base sizing returned 0 (invalid SL or zero capital)")
        return {
            "qty": 0, "base_qty": 0,
            "regime_mult": 1.0, "drawdown_mult": 1.0,
            "sector_room_pct": None, "blocked": True, "reasons": reasons,
        }

    rmult = regime_size_mult(regime_label)
    if rmult < 1.0:
        reasons.append(f"regime {regime_label}: ×{rmult:.2f}")

    dmult, dwhy = drawdown_brake(list(recent_pnl_pcts or []))
    if dmult < 1.0:
        reasons.append(f"drawdown brake: ×{dmult:.2f} ({dwhy})")

    qty = int(base * rmult * dmult)

    # Sector cap is a hard cap — it may force qty further down or to zero.
    sector_room: float | None = None
    if open_positions is not None and sector and capital_inr > 0:
        sector_room = sector_room_remaining_pct(
            open_positions, sector, capital_inr, sector_cap_pct=sector_cap_pct
        )
        max_qty_by_sector = int((capital_inr * (sector_room / 100.0)) // entry_price)
        if qty > max_qty_by_sector:
            reasons.append(
                f"sector '{sector}' cap: {sector_room:.1f}% room → max qty {max_qty_by_sector}"
            )
            qty = max(0, max_qty_by_sector)

    blocked = qty == 0
    if blocked and base > 0:
        reasons.append("all gates combined → qty 0 (blocked)")

    return {
        "qty": qty,
        "base_qty": base,
        "regime_mult": round(rmult, 3),
        "drawdown_mult": round(dmult, 3),
        "sector_room_pct": sector_room,
        "blocked": blocked,
        "reasons": reasons,
    }


__all__ = [
    "suggest_qty",
    "regime_size_mult",
    "sector_exposure_pct",
    "sector_room_remaining_pct",
    "drawdown_brake",
    "recommend_position_size",
]
