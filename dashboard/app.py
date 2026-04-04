"""
============================================================
  Indian Stock Agent — Dashboard v5
  New in v5:
  ─ News sentiment display per stock (headlines + scores)
  ─ Fundamentals panel per stock (P/E, D/E, market cap, ROE)
  ─ Signal streak badges (🔥 N-day streak)
  ─ Market breadth gauge in sidebar
  ─ Strategy Stats: per-strategy return % histograms
  ─ 🔬 Backtest Lab page: walk-forward simulation results
============================================================
"""

import json
from datetime import datetime, date, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from supabase import create_client, Client

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🇮🇳 Stock Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  SUPABASE
# ─────────────────────────────────────────────
@st.cache_resource
def get_supabase() -> Client:
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

sb = get_supabase()

# ─────────────────────────────────────────────
#  PASSWORD GATE
# ─────────────────────────────────────────────
def check_password():
    if st.session_state.get("authenticated"):
        return True
    st.title("🇮🇳 Indian Stock Agent")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if pwd == st.secrets.get("DASHBOARD_PASSWORD", "stockagent123"):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False

if not check_password():
    st.stop()

# ─────────────────────────────────────────────
#  FORMAT HELPERS
# ─────────────────────────────────────────────
def safe_float(v, default=0.0):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default

def safe_int(v, default=0):
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default

def fmt_inr(v):
    x = safe_float(v, None)
    return "—" if x is None else f"₹{x:,.2f}"

def fmt_pct(v, digits=2, signed=True):
    x = safe_float(v, None)
    if x is None:
        return "—"
    return f"{x:+.{digits}f}%" if signed else f"{x:.{digits}f}%"

def fmt_num(v, digits=2):
    x = safe_float(v, None)
    return "—" if x is None else f"{x:.{digits}f}"

def score_color(s):
    s = safe_float(s)
    if s >= 80: return "#00c853"
    if s >= 65: return "#69f0ae"
    if s >= 50: return "#ffeb3b"
    if s >= 35: return "#ffa726"
    return "#ef5350"

def news_color(label):
    return {"positive": "#26a69a", "negative": "#ef5350", "neutral": "#9e9e9e"}.get(
        str(label).lower(), "#9e9e9e"
    )

def status_badge(label, kind="neutral"):
    palette = {
        "buy": "#26a69a", "sell": "#ef5350", "champion": "#ffd54f",
        "challenger": "#90caf9", "candidate": "#c5e1a5", "retired": "#e0e0e0",
        "bullish": "#a5d6a7", "bearish": "#ef9a9a", "neutral": "#eeeeee",
    }
    bg = palette.get(kind, "#eeeeee")
    return (f'<span style="background:{bg};color:#111;padding:3px 10px;'
            f'border-radius:12px;font-weight:700;font-size:0.95em">{label}</span>')

def score_badge(s, label):
    col = score_color(s)
    return (f'<span style="background:{col};color:#111;padding:3px 10px;'
            f'border-radius:12px;font-weight:700;font-size:1.05em">'
            f'{safe_float(s):.0f} — {label}</span>')

def streak_badge(streak):
    if streak < 2:
        return ""
    flame = "🔥" * min(streak, 5)
    return (f'<span style="background:#ff6f00;color:#fff;padding:2px 8px;'
            f'border-radius:10px;font-weight:700;font-size:0.85em">'
            f'{flame} {streak}-day streak</span>')

def fund_warning_badge(warnings):
    if not warnings:
        return ""
    badges = []
    for w in warnings[:3]:
        badges.append(f'<span style="background:#b71c1c;color:#fff;padding:2px 6px;'
                      f'border-radius:8px;font-size:0.78em">{w}</span>')
    return " ".join(badges)

# ─────────────────────────────────────────────
#  DATA HELPERS
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_recs(target_date=None, days_back=None):
    q = sb.table("recommendations").select("*").order("composite_score", desc=True)
    if target_date:
        q = q.eq("date", target_date)
    elif days_back:
        q = q.gte("date", (date.today() - timedelta(days=days_back)).isoformat())
    res = q.execute()
    if not res.data:
        return pd.DataFrame()
    df = pd.DataFrame(res.data)
    for col in ["signals", "backtest", "score_breakdown", "strategy_weights",
                "news_headlines", "fundamental_warnings"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else (x or []))
    numeric_cols = [
        "price", "change_1d", "change_5d", "rsi", "stop_loss", "target", "risk_pct",
        "reward_pct", "win_rate", "avg_return", "median_return", "profit_factor",
        "max_drawdown", "composite_score", "weighted_score_val", "raw_score",
        "avg_trades", "news_score", "news_multiplier", "fundamental_score",
        "market_cap_cr", "pe_ratio", "de_ratio", "revenue_growth", "roe", "streak",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compatibility with older analyze.py field names
    if "streak" not in df.columns and "signal_streak" in df.columns:
        df["streak"] = pd.to_numeric(df["signal_streak"], errors="coerce")
    if "news_label" not in df.columns and "news_sentiment" in df.columns:
        df["news_label"] = df["news_sentiment"].fillna("NEUTRAL").astype(str).str.lower()
    if "news_headlines" not in df.columns and "news_headline" in df.columns:
        df["news_headlines"] = df["news_headline"].apply(lambda x: [x] if isinstance(x, str) and x.strip() else [])
    if "news_multiplier" not in df.columns:
        df["news_multiplier"] = 1.0
    if "fundamental_score" not in df.columns and "fundamental_flag" in df.columns:
        def _fs(v):
            if v in [None, "", "DATA_UNAVAILABLE"]: return 50
            return 85 if str(v) == "OK" else 60
        df["fundamental_score"] = df["fundamental_flag"].apply(_fs)
    if "fundamental_warnings" not in df.columns and "fundamental_flag" in df.columns:
        df["fundamental_warnings"] = df["fundamental_flag"].apply(lambda x: [] if x in [None, "", "OK", "DATA_UNAVAILABLE"] else [s.strip() for s in str(x).split(',') if s.strip()])
    if "de_ratio" not in df.columns and "debt_equity" in df.columns:
        df["de_ratio"] = pd.to_numeric(df["debt_equity"], errors="coerce")
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    return df

@st.cache_data(ttl=60)
def load_portfolio():
    res = sb.table("paper_portfolio").select("*").order("buy_date", desc=True).execute()
    return pd.DataFrame(res.data) if res.data else pd.DataFrame()

@st.cache_data(ttl=90)
def live_price(ticker):
    try:
        df = yf.download(ticker + ".NS", period="3d", progress=False, auto_adjust=True)
        return float(df["Close"].iloc[-1]) if not df.empty else 0.0
    except Exception:
        return 0.0

@st.cache_data(ttl=600)
def price_history(ticker, days=90):
    try:
        df = yf.download(ticker + ".NS",
                         start=date.today() - timedelta(days=days),
                         end=date.today(),
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=120)
def get_meta():
    res = sb.table("agent_meta").select("*").eq("id", 1).execute()
    return res.data[0] if res.data else {}

@st.cache_data(ttl=120)
def load_agent_params(status=None):
    q = sb.table("agent_params").select("*").order("version", desc=True)
    if status:
        q = q.eq("status", status)
    return pd.DataFrame(q.execute().data or [])

@st.cache_data(ttl=120)
def load_opt_runs():
    res = sb.table("optimization_runs").select("*").order("run_date", desc=True).execute()
    return pd.DataFrame(res.data or [])

@st.cache_data(ttl=300)
def load_simulations(days_back=90):
    cutoff = (date.today() - timedelta(days=days_back)).isoformat()
    res = sb.table("backtest_simulations").select("*").gte("signal_date", cutoff)\
          .order("signal_date", desc=True).execute()
    if not res.data:
        return pd.DataFrame()
    df = pd.DataFrame(res.data)
    for col in ["actual_return_pct", "composite_score", "predicted_win_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data(ttl=120)
def load_sim_meta():
    try:
        res = sb.table("simulation_meta").select("*").eq("id", 1).execute()
        return res.data[0] if res.data else {}
    except Exception:
        return {}

# ─────────────────────────────────────────────
#  PORTFOLIO / ALERTS
# ─────────────────────────────────────────────
def paper_buy(ticker, price, qty, sl, tgt, notes, rec_id=None):
    sb.table("paper_portfolio").insert({
        "ticker": ticker, "buy_date": date.today().isoformat(),
        "buy_price": price, "quantity": qty,
        "entry_stop_loss": sl, "entry_target": tgt,
        "status": "OPEN", "notes": notes, "recommendation_id": rec_id,
    }).execute()
    st.cache_data.clear()

def paper_sell(trade_id, sell_price, buy_price, qty, exit_reason="manual"):
    pnl_pct = round((sell_price - buy_price) / buy_price * 100, 2) if buy_price else 0
    pnl_inr = round((sell_price - buy_price) * qty, 2)
    sb.table("paper_portfolio").update({
        "sell_date": date.today().isoformat(), "sell_price": sell_price,
        "status": "CLOSED", "pnl_pct": pnl_pct, "pnl_inr": pnl_inr,
        "exit_reason": exit_reason,
    }).eq("id", trade_id).execute()
    st.cache_data.clear()

def promote_param(version, new_status):
    if new_status == "champion":
        sb.table("agent_params").update({"status": "retired"}).eq("status", "champion").execute()
        sb.table("agent_params").update({"status": "retired"}).eq("status", "challenger").execute()
    elif new_status == "challenger":
        sb.table("agent_params").update({"status": "retired"}).eq("status", "challenger").execute()
    sb.table("agent_params").update({
        "status": new_status, "promoted_at": date.today().isoformat(),
    }).eq("version", version).execute()
    st.cache_data.clear()

def check_exit_alerts(port):
    alerts = []
    if port.empty:
        return alerts
    for _, row in port[port.status == "OPEN"].iterrows():
        lp = live_price(row.ticker)
        if lp <= 0:
            continue
        sl  = safe_float(row.get("entry_stop_loss"), 0)
        tgt = safe_float(row.get("entry_target"), 0)
        buy = safe_float(row.get("buy_price"), 0)
        pnl = round((lp - buy) / buy * 100, 2) if buy > 0 else 0
        if sl > 0 and lp <= sl:
            alerts.append({"ticker": row.ticker, "type": "SL_HIT", "lp": lp,
                            "level": sl, "pnl": pnl, "id": row.id,
                            "buy_price": buy, "qty": row.quantity})
        elif tgt > 0 and lp >= tgt:
            alerts.append({"ticker": row.ticker, "type": "TARGET_HIT", "lp": lp,
                            "level": tgt, "pnl": pnl, "id": row.id,
                            "buy_price": buy, "qty": row.quantity})
    return alerts

def compute_portfolio_snapshot(port):
    empty = {"open_pos": pd.DataFrame(), "closed_pos": pd.DataFrame(),
             "total_inv": 0.0, "total_cur": 0.0, "open_pnl": 0.0,
             "closed_pnl": 0.0, "win_ct": 0, "win_rate": 0.0}
    if port.empty:
        return empty
    open_pos   = port[port.status == "OPEN"].copy()
    closed_pos = port[port.status == "CLOSED"].copy()
    total_inv  = total_cur = 0.0
    for _, row in open_pos.iterrows():
        lp = live_price(row.ticker)
        total_inv += safe_float(row.buy_price) * safe_float(row.quantity)
        total_cur += (lp if lp > 0 else safe_float(row.buy_price)) * safe_float(row.quantity)
    closed_pnl = float(closed_pos.pnl_inr.sum()) if (not closed_pos.empty and "pnl_inr" in closed_pos) else 0.0
    win_ct     = len(closed_pos[closed_pos.pnl_pct > 0]) if (not closed_pos.empty and "pnl_pct" in closed_pos) else 0
    win_rate   = win_ct / len(closed_pos) * 100 if not closed_pos.empty else 0.0
    return {"open_pos": open_pos, "closed_pos": closed_pos,
            "total_inv": total_inv, "total_cur": total_cur,
            "open_pnl": total_cur - total_inv, "closed_pnl": closed_pnl,
            "win_ct": win_ct, "win_rate": win_rate}

# ─────────────────────────────────────────────
#  CHART HELPERS
# ─────────────────────────────────────────────
def candlestick(df, ticker, buy_price=None, sl=None, tgt=None):
    df = df.copy()
    df["ema9"]  = df.Close.ewm(span=9,  adjust=False).mean()
    df["ema21"] = df.Close.ewm(span=21, adjust=False).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close,
        name=ticker, increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
    fig.add_trace(go.Scatter(x=df.index, y=df.ema9,  name="EMA 9",
                              line=dict(color="#ffb300", width=1.2)))
    fig.add_trace(go.Scatter(x=df.index, y=df.ema21, name="EMA 21",
                              line=dict(color="#ab47bc", width=1.2)))
    if buy_price:
        fig.add_hline(y=buy_price, line_color="#00e676", line_dash="dash",
                      annotation_text=f"Buy ₹{buy_price:.2f}")
    if sl:
        fig.add_hline(y=sl,  line_color="#ef5350", line_dash="dot",
                      annotation_text=f"SL ₹{sl:.2f}")
    if tgt:
        fig.add_hline(y=tgt, line_color="#69f0ae", line_dash="dot",
                      annotation_text=f"Tgt ₹{tgt:.2f}")
    fig.update_layout(height=360, xaxis_rangeslider_visible=False,
                      margin=dict(t=10, b=10, l=0, r=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.02))
    return fig

def breadth_gauge(buys, sells, neutral):
    total = buys + sells + neutral or 1
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(buys / (buys + sells) * 100) if (buys + sells) > 0 else 50,
        title={"text": "Market Breadth"},
        gauge={
            "axis":  {"range": [0, 100]},
            "bar":   {"color": "#26a69a"},
            "steps": [
                {"range": [0,  40], "color": "#ef9a9a"},
                {"range": [40, 60], "color": "#fff9c4"},
                {"range": [60, 100], "color": "#a5d6a7"},
            ],
            "threshold": {"line": {"color": "black", "width": 2}, "value": 50},
        },
    ))
    fig.update_layout(height=200, margin=dict(t=30, b=10, l=10, r=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig

def strategy_histogram(returns_list, strategy_name, bins=20):
    """Return % histogram for a single strategy."""
    arr = pd.Series(
        [float(r) for r in returns_list if r is not None and pd.notna(r)
         and isinstance(r, (int, float)) and np.isfinite(float(r))],
        dtype=float,
    )
    if arr.empty:
        return None
    wins  = (arr > 0).sum()
    total = len(arr)
    wr    = wins / total * 100
    avg   = arr.mean()
    med   = arr.median()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=arr, nbinsx=bins,
        marker_color=["#26a69a" if v > 0 else "#ef5350" for v in arr],
        marker_line_width=0.5, marker_line_color="white",
        name="Returns",
    ))
    fig.add_vline(x=0,   line_color="#ef5350", line_dash="dash", line_width=2,
                  annotation_text="0%", annotation_position="top left")
    fig.add_vline(x=avg, line_color="#ffb300", line_dash="dot",  line_width=1.5,
                  annotation_text=f"Avg {avg:+.1f}%", annotation_position="top right")
    fig.update_layout(
        title=f"{strategy_name}  —  {wr:.0f}% positive  |  avg {avg:+.2f}%  |  n={total}",
        height=250,
        margin=dict(t=40, b=20, l=0, r=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis_title="Return %",
    )
    return fig

# ─────────────────────────────────────────────
#  COMPONENT: NEWS PANEL
# ─────────────────────────────────────────────
def render_news_panel(row):
    """Render news headlines with sentiment for a recommendation row."""
    headlines = row.get("news_headlines", []) or []
    ns        = safe_float(row.get("news_score"), None)
    label     = str(row.get("news_label", "neutral")).lower()
    mult      = safe_float(row.get("news_multiplier"), 1.0)

    if not headlines:
        st.caption("📰 No recent news found for this stock.")
        return

    col = news_color(label)
    st.markdown(
        f'📰 **News Sentiment:** '
        f'<span style="background:{col};color:#fff;padding:2px 8px;'
        f'border-radius:8px;font-weight:700">{label.upper()}</span> '
        f'&nbsp; score: {ns:+.3f} &nbsp; multiplier: ×{mult:.2f}',
        unsafe_allow_html=True,
    )
    for h in headlines:
        st.markdown(f"- {h}")

# ─────────────────────────────────────────────
#  COMPONENT: FUNDAMENTALS PANEL
# ─────────────────────────────────────────────
def render_fundamentals_panel(row):
    """Render key fundamental metrics for a recommendation row."""
    fs   = safe_float(row.get("fundamental_score"), 50)
    warn = row.get("fundamental_warnings", []) or []
    mc   = row.get("market_cap_cr")
    pe   = row.get("pe_ratio")
    de   = row.get("de_ratio")
    roe  = row.get("roe")
    rg   = row.get("revenue_growth")
    sect = row.get("sector", "Unknown")

    st.markdown(
        f'📊 **Fundamentals** — Score: {fs:.0f}/100 &nbsp; Sector: *{sect}* &nbsp; '
        + (fund_warning_badge(warn) if warn else ""),
        unsafe_allow_html=True,
    )
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Market Cap", f"₹{mc:,.0f}Cr" if mc else "—")
    f2.metric("P/E Ratio",  f"{pe:.1f}"   if pe  else "—")
    f3.metric("Debt/Equity",f"{de:.2f}"   if de  else "—")
    f4.metric("ROE",        f"{roe:.1f}%" if roe  else "—")
    f5.metric("Rev. Growth",f"{rg:+.1f}%" if rg  else "—")

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🇮🇳 Stock Agent")
    meta = get_meta()
    if meta:
        st.success(f"Last run: **{meta.get('last_run','N/A')}**")
        regime = meta.get("market_regime", "?")
        emoji  = {"BULLISH":"🟢","BEARISH":"🔴","NEUTRAL":"🟡"}.get(regime,"⬜")
        st.caption(f"Market: {emoji} **{regime}**  |  Signals: {meta.get('total_signals',0)}")
        st.caption(f"Params: {meta.get('active_param_version','defaults')}")

        # Breadth gauge
        b_buy = safe_int(meta.get("breadth_buys", 0))
        b_sell= safe_int(meta.get("breadth_sells", 0))
        b_neu = safe_int(meta.get("breadth_neutral", 0))
        if b_buy + b_sell > 0:
            st.plotly_chart(breadth_gauge(b_buy, b_sell, b_neu),
                            use_container_width=True, key="sidebar_breadth")
            st.caption(f"Breadth: {b_buy} BUY / {b_sell} SELL / {b_neu} Neutral")
    else:
        st.warning("Agent hasn't run yet")

    st.divider()
    page = st.radio("Navigate", [
        "📊 Today's Signals",
        "💼 My Paper Portfolio",
        "📅 Signal History",
        "📈 Strategy Stats",
        "🔬 Backtest Lab",
        "🤖 Optimizer",
    ])
    st.divider()
    st.caption("NSE: 9:15 AM – 3:30 PM IST")
    st.caption("Agent: 7:00 AM IST, Mon–Fri")
    st.caption("Optimizer: Sunday 11 PM IST")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

# ─────────────────────────────────────────────
#  GLOBAL: PORTFOLIO + ALERTS
# ─────────────────────────────────────────────
port_all           = load_portfolio()
portfolio_snapshot = compute_portfolio_snapshot(port_all)
alerts             = check_exit_alerts(port_all)

for a in alerts:
    if a["type"] == "SL_HIT":
        st.error(f"🚨 **STOP-LOSS HIT — {a['ticker']}** | Live: ₹{a['lp']:,.2f} ≤ "
                 f"SL: ₹{float(a['level']):,.2f} | P&L: {a['pnl']:+.2f}%")
    else:
        st.success(f"🎯 **TARGET HIT — {a['ticker']}** | Live: ₹{a['lp']:,.2f} ≥ "
                   f"Target: ₹{float(a['level']):,.2f} | P&L: {a['pnl']:+.2f}%")

# ═══════════════════════════════════════════════
#  PAGE: TODAY'S SIGNALS
# ═══════════════════════════════════════════════
if page == "📊 Today's Signals":
    today_str = date.today().isoformat()
    st.title("📊 Today's Signals")
    st.caption(f"{datetime.today().strftime('%A, %d %B %Y')}  |  Sorted by Composite Score ↓")

    recs = load_recs(target_date=today_str)
    if recs.empty:
        st.info("No signals today. Agent runs at 7:00 AM IST. Trigger manually from GitHub Actions if needed.")
        st.stop()

    buys  = recs[recs.action == "BUY"].copy()
    sells = recs[recs.action == "SELL"].copy()

    # ── Header metrics
    h1, h2, h3, h4, h5, h6 = st.columns(6)
    h1.metric("Signals",   len(recs))
    h2.metric("Buys",      len(buys))
    h3.metric("Sells",     len(sells))
    h4.metric("Top Score", f"{safe_float(recs.composite_score.max()):.0f}/100")
    h5.metric("Open P&L",  f"₹{portfolio_snapshot['open_pnl']:+,.0f}")
    h6.metric("Market",    meta.get("market_regime", "?"))

    # ── Filters
    with st.expander("🔧 Filters & Ranking", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        min_cs          = f1.slider("Min Composite Score",    0, 100, 30)
        min_wr          = f2.slider("Min Positive Return %",  0, 100, 40)
        min_pf          = f3.slider("Min Profit Factor",      0.0, 5.0, 0.8, 0.1)
        action_filter   = f4.selectbox("Action", ["All","BUY","SELL"])

        f5, f6, f7, f8 = st.columns(4)
        ticker_q        = f5.text_input("Ticker search", "")
        news_filter     = f6.selectbox("News Sentiment",
                                       ["All","positive","neutral","negative"])
        regime_filter   = f7.selectbox("Regime filter",
                                       ["All","BULLISH","NEUTRAL","BEARISH","UNKNOWN"])
        sort_col        = f8.selectbox("Sort by",
                                       ["composite_score","win_rate","profit_factor",
                                        "avg_return","streak","fundamental_score",
                                        "news_score","rsi","reward_pct"], index=0)

        filtered = recs.copy()
        filtered = filtered[
            (filtered.composite_score.fillna(0) >= min_cs) &
            (filtered.win_rate.fillna(0)         >= min_wr) &
            (filtered.profit_factor.fillna(0)    >= min_pf)
        ]
        if action_filter != "All":
            filtered = filtered[filtered.action == action_filter]
        if ticker_q.strip():
            filtered = filtered[
                filtered.ticker.astype(str).str.contains(ticker_q.strip(), case=False, na=False)]
        if news_filter != "All" and "news_label" in filtered.columns:
            filtered = filtered[filtered.news_label == news_filter]
        if regime_filter != "All" and "market_regime" in filtered.columns:
            filtered = filtered[filtered.market_regime == regime_filter]

        ascending = sort_col in ["risk_pct", "rsi"]
        if sort_col in filtered.columns:
            filtered = filtered.sort_values(sort_col, ascending=ascending, na_position="last")

    if filtered.empty:
        st.warning("No signals match the selected filters.")
        st.stop()

    # ── Leaderboard table
    st.subheader("📋 Signal Leaderboard")
    board = filtered.copy()
    board_cols = ["ticker", "action", "composite_score", "score_label",
                  "streak", "win_rate", "profit_factor", "avg_return",
                  "news_label", "fundamental_score", "sector",
                  "rsi", "price", "stop_loss", "target",
                  "reward_pct", "risk_pct", "market_regime"]
    board_cols = [c for c in board_cols if c in board.columns]
    board_show = board[board_cols].rename(columns={
        "ticker": "Ticker", "action": "Action",
        "composite_score": "Score", "score_label": "Label",
        "streak": "Streak", "win_rate": "Win %",
        "profit_factor": "PF", "avg_return": "Avg Ret %",
        "news_label": "News", "fundamental_score": "Fund Score",
        "sector": "Sector", "rsi": "RSI",
        "price": "Price ₹", "stop_loss": "SL ₹", "target": "Target ₹",
        "reward_pct": "Reward %", "risk_pct": "Risk %",
        "market_regime": "Regime",
    })
    st.dataframe(board_show, use_container_width=True, hide_index=True)

    # ── Detail view
    chosen = st.selectbox("📂 Open details for ticker", filtered["ticker"].tolist())
    sel    = filtered[filtered["ticker"] == chosen].iloc[0]

    st.divider()
    st.subheader(f"🔎 Signal Details — {sel.ticker}")

    # Score + streak + news warning badge
    badges_html = (score_badge(sel.composite_score, sel.score_label) + " &nbsp; " +
                   streak_badge(safe_int(sel.get("streak"), 1)) + " &nbsp; " +
                   status_badge(sel.action, "buy" if sel.action == "BUY" else "sell"))
    st.markdown(badges_html, unsafe_allow_html=True)
    st.markdown("")

    if bool(sel.get("low_sample_warning", False)):
        st.warning("⚠️ Fewer than 5 backtest trades — interpret win rate cautiously.")

    detail_left, detail_right = st.columns([1.2, 1.0])

    with detail_left:
        # Score breakdown
        bd = sel.get("score_breakdown", {}) or {}
        if bd:
            bd_df = pd.DataFrame([
                {"Component": "Strategy",  "Score": bd.get("strategy", 0), "Max": 40},
                {"Component": "RSI",       "Score": bd.get("rsi",      0), "Max": 20},
                {"Component": "Volume",    "Score": bd.get("volume",   0), "Max": 15},
                {"Component": "R:R",       "Score": bd.get("rr",       0), "Max": 15},
                {"Component": "Regime",    "Score": bd.get("regime",   0), "Max": 10},
            ])
            bd_df["%"] = (bd_df.Score / bd_df.Max * 100).clip(0, 100)
            fig_bd = px.bar(bd_df, x="Score", y="Component", orientation="h",
                            text="Score", color="%",
                            color_continuous_scale="RdYlGn", range_color=[0, 100])
            fig_bd.update_layout(height=220, margin=dict(t=0,b=0,l=0,r=0),
                                  coloraxis_showscale=False,
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            fig_bd.update_traces(texttemplate="%{text:.1f}", textposition="inside")
            st.plotly_chart(fig_bd, use_container_width=True)

        # Key metrics grid
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Price",          fmt_inr(sel.get("price")))
        m1.metric("1D Change",      fmt_pct(sel.get("change_1d")))
        m1.metric("5D Change",      fmt_pct(sel.get("change_5d")))
        m2.metric("RSI",            fmt_num(sel.get("rsi"), 1))
        m2.metric("Win Rate",       fmt_pct(sel.get("win_rate"),    1, signed=False))
        m2.metric("Profit Factor",  fmt_num(sel.get("profit_factor"), 2))
        m3.metric("Stop Loss",      fmt_inr(sel.get("stop_loss")))
        m3.metric("Target",         fmt_inr(sel.get("target")))
        m3.metric("Reward",         fmt_pct(sel.get("reward_pct")))
        m4.metric("Risk",           fmt_pct(sel.get("risk_pct")))
        m4.metric("Avg Return",     fmt_pct(sel.get("avg_return")))
        m4.metric("Median Return",  fmt_pct(sel.get("median_return")))

        st.caption(f"Strategies: {sel.get('active_strategies','—')}")
        st.caption(f"Regime: {sel.get('market_regime','—')} | Params: {sel.get('param_version','—')}")

        # News panel
        st.divider()
        render_news_panel(sel)

        # Fundamentals panel
        st.divider()
        render_fundamentals_panel(sel)

    with detail_right:
        hist_df = price_history(sel.ticker, 90)
        if not hist_df.empty:
            st.plotly_chart(
                candlestick(hist_df, sel.ticker,
                            sl=safe_float(sel.get("stop_loss"), None) or None,
                            tgt=safe_float(sel.get("target"), None) or None),
                use_container_width=True)
        else:
            st.info("Price chart unavailable.")

    # ── Strategy breakdown table
    st.subheader("🧠 Strategy Breakdown")
    sigs = sel.get("signals", {}) or {}
    bts  = sel.get("backtest", {}) or {}
    wts  = sel.get("strategy_weights", {}) or {}
    strat_rows = []
    for name, val in sigs.items():
        b = bts.get(name, {})
        strat_rows.append({
            "Strategy":       name,
            "Signal":         "BUY" if val == 1 else ("SELL" if val == -1 else "None"),
            "Weight":         safe_float(wts.get(name, 0)),
            "Win Rate %":     safe_float(b.get("win_rate", 0)),
            "Avg Return %":   safe_float(b.get("avg_return", 0)),
            "Median Ret %":   safe_float(b.get("median_return", 0)),
            "Profit Factor":  safe_float(b.get("profit_factor", 0)),
            "Trades":         safe_int(b.get("trades", 0)),
            "SL Exits":       safe_int(b.get("sl_exits", 0)),
            "Target Exits":   safe_int(b.get("target_exits", 0)),
        })
    st.dataframe(pd.DataFrame(strat_rows), use_container_width=True, hide_index=True)

    # ── Paper trade form
    st.subheader("📝 Paper Trade")
    pb1, pb2, pb3, pb4 = st.columns(4)
    qty   = pb1.number_input("Qty", min_value=1, value=10, key=f"qty_{sel.ticker}")
    price = pb2.number_input("Price (₹)", value=safe_float(sel.get("price"), 0.0),
                              key=f"px_{sel.ticker}")
    sl_in = pb3.number_input("Stop Loss (₹)", value=safe_float(sel.get("stop_loss"), 0.0),
                              key=f"sl_{sel.ticker}")
    tg_in = pb4.number_input("Target (₹)", value=safe_float(sel.get("target"), 0.0),
                              key=f"tg_{sel.ticker}")
    notes = st.text_input("Notes", value=str(sel.get("active_strategies", "")),
                           key=f"nt_{sel.ticker}")
    if st.button(f"🟢 Paper Buy {sel.ticker}", key=f"buy_{sel.ticker}"):
        paper_buy(sel.ticker, price, qty, sl_in, tg_in, notes, str(sel.get("id", "")))
        st.success(f"✅ Paper bought {qty} × {sel.ticker} @ ₹{price:.2f}")
        st.balloons()

# ═══════════════════════════════════════════════
#  PAGE: PAPER PORTFOLIO
# ═══════════════════════════════════════════════
elif page == "💼 My Paper Portfolio":
    st.title("💼 My Paper Portfolio")
    snap       = portfolio_snapshot
    open_pos   = snap["open_pos"]
    closed_pos = snap["closed_pos"]
    total_inv  = snap["total_inv"]

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Open Positions", len(open_pos))
    c2.metric("Open P&L",  f"₹{snap['open_pnl']:+,.0f}",
               delta=f"{snap['open_pnl']/total_inv*100:+.1f}%" if total_inv else "–")
    c3.metric("Realised P&L",  f"₹{snap['closed_pnl']:+,.0f}")
    c4.metric("Total P&L",     f"₹{snap['open_pnl']+snap['closed_pnl']:+,.0f}")
    c5.metric("Win Rate",      f"{snap['win_rate']:.0f}%",
               delta=f"{snap['win_ct']}/{len(closed_pos)}" if not closed_pos.empty else "0/0")

    st.divider()
    st.subheader("📂 Open Positions")
    if alerts:
        st.warning(f"⚠️ {len(alerts)} position(s) need attention — see alerts above!")
    if open_pos.empty:
        st.info("No open positions. Go to 'Today's Signals' to paper buy.")
    else:
        for _, row in open_pos.iterrows():
            lp = live_price(row.ticker)
            if lp > 0:
                pnl_pct = (lp - row.buy_price) / row.buy_price * 100 if row.buy_price else 0
                pnl_inr = (lp - row.buy_price) * row.quantity
            else:
                lp = row.buy_price; pnl_pct = pnl_inr = 0
            sl  = safe_float(row.get("entry_stop_loss"), 0)
            tgt = safe_float(row.get("entry_target"), 0)
            sl_hit  = sl > 0 and lp <= sl
            tgt_hit = tgt > 0 and lp >= tgt
            icon    = "🚨" if sl_hit else ("🎯" if tgt_hit else ("🟢" if pnl_pct >= 0 else "🔴"))
            with st.expander(
                f"{icon} **{row.ticker}** | {safe_int(row.quantity)} shares | "
                f"Buy ₹{safe_float(row.buy_price):,.2f} → Live ₹{lp:,.2f} | "
                f"P&L {pnl_pct:+.2f}%", expanded=sl_hit or tgt_hit
            ):
                if sl_hit:  st.error(f"🚨 Stop-loss breached! ₹{lp:,.2f} ≤ SL ₹{sl:,.2f}")
                if tgt_hit: st.success(f"🎯 Target hit! ₹{lp:,.2f} ≥ Target ₹{tgt:,.2f}")
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric("Buy Price",  fmt_inr(row.buy_price))
                pc1.metric("Buy Date",   str(row.buy_date))
                pc2.metric("Live Price", fmt_inr(lp))
                pc2.metric("Stop Loss",  fmt_inr(sl) if sl else "Not set")
                pc2.metric("Target",     fmt_inr(tgt) if tgt else "Not set")
                pc3.metric("P&L %",  fmt_pct(pnl_pct))
                pc3.metric("P&L ₹",  f"₹{pnl_inr:+,.0f}")
                if sl > 0:
                    sl_dist = (lp - sl) / lp * 100
                    pc3.metric("Dist to SL", f"{sl_dist:.1f}%",
                                delta_color="inverse" if sl_dist < 2 else "normal")
                hist_df = price_history(row.ticker, 60)
                if not hist_df.empty:
                    st.plotly_chart(candlestick(hist_df, row.ticker,
                                                buy_price=row.buy_price,
                                                sl=sl or None, tgt=tgt or None),
                                    use_container_width=True)
                s1, s2, s3 = st.columns(3)
                sell_px = s1.number_input("Sell Price (₹)", value=float(lp),
                                           key=f"spx_{row.id}")
                er = s2.selectbox("Exit Reason",
                                   ["manual","sl_hit","target_hit","other"],
                                   key=f"er_{row.id}")
                if s3.button(f"🔴 Sell {row.ticker}", key=f"sell_{row.id}"):
                    paper_sell(row.id, sell_px, row.buy_price, row.quantity, er)
                    st.success(f"✅ Closed {row.ticker} @ ₹{sell_px:.2f}")
                    st.rerun()

    st.divider()
    st.subheader("✅ Closed Trades")
    if closed_pos.empty:
        st.info("No closed trades yet.")
    else:
        cs = closed_pos.sort_values("sell_date").copy()
        cs["cum_pnl"] = cs.pnl_inr.fillna(0).cumsum()
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=cs.sell_date, y=cs.cum_pnl,
                                     fill="tozeroy", line=dict(color="#26a69a")))
        fig_eq.add_hline(y=0, line_color="#ef5350", line_dash="dash")
        fig_eq.update_layout(height=260, title="Cumulative P&L (₹)",
                               margin=dict(t=30,b=10,l=0,r=0),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_eq, use_container_width=True)
        disp_cols = ["ticker","buy_date","sell_date","buy_price","sell_price",
                     "quantity","pnl_pct","pnl_inr","exit_reason"]
        disp_cols = [c for c in disp_cols if c in closed_pos.columns]
        st.dataframe(closed_pos[disp_cols].fillna("—"), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════
#  PAGE: SIGNAL HISTORY
# ═══════════════════════════════════════════════
elif page == "📅 Signal History":
    st.title("📅 Signal History")
    days_back = st.slider("Past N days", 7, 60, 14)
    hist = load_recs(days_back=days_back)
    if hist.empty:
        st.info("No history yet."); st.stop()

    act_f = st.selectbox("Action filter", ["All","BUY","SELL"])
    if act_f != "All":
        hist = hist[hist.action == act_f]

    daily = hist.groupby(["date","action"]).size().reset_index(name="count")
    fig = px.bar(daily, x="date", y="count", color="action",
                  color_discrete_map={"BUY":"#26a69a","SELL":"#ef5350"},
                  title="Daily signal count")
    fig.update_layout(height=250, margin=dict(t=30,b=10,l=0,r=0),
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    show_cols = ["date","ticker","action","composite_score","score_label","streak",
                 "win_rate","avg_return","profit_factor","news_label","fundamental_score",
                 "sector","rsi","active_strategies","market_regime","param_version"]
    show_cols = [c for c in show_cols if c in hist.columns]
    st.dataframe(hist[show_cols], use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════
#  PAGE: STRATEGY STATS
# ═══════════════════════════════════════════════
elif page == "📈 Strategy Stats":
    st.title("📈 Strategy Performance Stats")

    res = sb.table("recommendations").select(
        "ticker,backtest,action,win_rate,avg_return,composite_score,"
        "profit_factor,max_drawdown,median_return,date,param_version"
    ).execute()
    all_recs = pd.DataFrame(res.data or [])

    if all_recs.empty:
        st.info("No data yet. Run the agent first."); st.stop()

    LIVE_STRATEGIES = [
        "EMA Crossover", "RSI + MACD", "Bollinger",
        "Donchian", "Volume Breakout", "RSI Trend Shift",
    ]

    f1, f2 = st.columns(2)
    days_back  = f1.slider("History window (days)", 7, 180, 60)
    strat_mode = f2.selectbox("Strategy view",
                               ["All strategies", "Current live only"], index=0)

    if "date" in all_recs.columns:
        cutoff = (date.today() - timedelta(days=days_back)).isoformat()
        all_recs = all_recs[all_recs["date"] >= cutoff].copy()

    # ── Unpack backtest JSON into rows
    agg_rows  = []
    # trade_returns per strategy (for histograms)
    strat_returns: dict[str, list] = {s: [] for s in LIVE_STRATEGIES}

    for _, r in all_recs.iterrows():
        try:
            bt = json.loads(r.backtest) if isinstance(r.backtest, str) else r.backtest
            if not isinstance(bt, dict):
                continue
            for name, stats in bt.items():
                if strat_mode == "Current live only" and name not in LIVE_STRATEGIES:
                    continue
                if not isinstance(stats, dict):
                    continue
                agg_rows.append({
                    "Strategy":       name,
                    "Win Rate":       safe_float(stats.get("win_rate", 0)),
                    "Avg Return":     safe_float(stats.get("avg_return", 0)),
                    "Median Return":  safe_float(stats.get("median_return", 0)),
                    "Profit Factor":  safe_float(stats.get("profit_factor", 0)),
                    "Max Drawdown":   safe_float(stats.get("max_drawdown", 0)),
                    "SL Exits":       safe_int(stats.get("sl_exits", 0)),
                    "Target Exits":   safe_int(stats.get("target_exits", 0)),
                    "Timeout Exits":  safe_int(stats.get("timeout_exits", 0)),
                    "Trades":         safe_int(stats.get("trades", 0)),
                })
                # Collect trade_returns for histogram
                tr = stats.get("trade_returns", [])
                if isinstance(tr, list) and tr:
                    strat_returns.setdefault(name, []).extend(tr)
        except Exception:
            continue

    if not agg_rows:
        st.info("No backtest data in selected window."); st.stop()

    bt_df = pd.DataFrame(agg_rows)
    agg   = bt_df.groupby("Strategy").agg(
        Win_Rate      = ("Win Rate",     "mean"),
        Avg_Return    = ("Avg Return",   "mean"),
        Median_Return = ("Median Return","mean"),
        Profit_Factor = ("Profit Factor","mean"),
        Max_Drawdown  = ("Max Drawdown", "mean"),
        Total_Trades  = ("Trades",       "sum"),
        SL_Exits      = ("SL Exits",     "sum"),
        Target_Exits  = ("Target Exits", "sum"),
        Timeout_Exits = ("Timeout Exits","sum"),
    ).reset_index()

    # Low-sample warnings
    for _, row in agg.iterrows():
        if 0 < row.Total_Trades < 20:
            st.warning(f"⚠️ **{row.Strategy}**: only {int(row.Total_Trades)} trades — interpret cautiously.")

    # ── Bar charts
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(agg, x="Strategy", y="Win_Rate",
                      color="Win_Rate", color_continuous_scale="teal",
                      text="Win_Rate", title="Win Rate %")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=300, showlegend=False,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.bar(agg, x="Strategy", y="Profit_Factor",
                       color="Profit_Factor", color_continuous_scale="RdYlGn",
                       text="Profit_Factor", title="Profit Factor (>1 = profitable)")
        fig2.add_hline(y=1.0, line_dash="dash", line_color="#ef5350")
        fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig2.update_layout(height=300, showlegend=False,
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    # Exit breakdown
    exit_df = []
    for _, row in agg.iterrows():
        tot = int(row.Total_Trades)
        if tot > 0:
            exit_df.append({"Strategy": row.Strategy, "Type": "SL",     "Count": int(row.SL_Exits)})
            exit_df.append({"Strategy": row.Strategy, "Type": "Target",  "Count": int(row.Target_Exits)})
            exit_df.append({"Strategy": row.Strategy, "Type": "Timeout",
                             "Count": max(0, tot - int(row.SL_Exits) - int(row.Target_Exits))})
    if exit_df:
        fig3 = px.bar(pd.DataFrame(exit_df), x="Strategy", y="Count", color="Type",
                       color_discrete_map={"SL":"#ef5350","Target":"#26a69a","Timeout":"#ffa726"},
                       barmode="stack", title="Exit Reason Breakdown")
        fig3.update_layout(height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(agg.style.format({
        "Win_Rate":"{:.1f}%","Avg_Return":"{:+.2f}%","Median_Return":"{:+.2f}%",
        "Profit_Factor":"{:.2f}","Max_Drawdown":"{:.1f}%",
    }), use_container_width=True, hide_index=True)

    # ── Per-strategy return histograms
    st.divider()
    st.subheader("📊 Return % Histograms by Strategy")

    strategies_with_data = [s for s, v in strat_returns.items() if len(v) >= 5]
    if not strategies_with_data:
        st.info("Return histograms require `trade_returns` in the backtest JSON. "
                "These appear after the agent has run with v5 of analyze.py.")
    else:
        bins = st.slider("Histogram bins", 10, 50, 20, key="hist_bins")
        # Display in 2-column grid
        for i in range(0, len(strategies_with_data), 2):
            cols = st.columns(2)
            for j, strat in enumerate(strategies_with_data[i:i+2]):
                rets = strat_returns[strat]
                fig_h = strategy_histogram(rets, strat, bins)
                if fig_h:
                    cols[j].plotly_chart(fig_h, use_container_width=True)
                    arr   = pd.Series([float(r) for r in rets if r is not None], dtype=float)
                    wins  = (arr > 0).sum()
                    cols[j].caption(
                        f"n={len(arr)} | Wins: {wins} | "
                        f"P10: {arr.quantile(0.1):+.2f}% | P90: {arr.quantile(0.9):+.2f}%"
                    )

    # ── Composite score vs realised P&L
    closed_port = load_portfolio()
    if not closed_port.empty:
        closed_port = closed_port[(closed_port.status == "CLOSED") &
                                   closed_port.recommendation_id.notna()].copy()
    if not closed_port.empty:
        st.divider()
        st.subheader("🔗 Composite Score vs Realised P&L")
        rec_ids = closed_port.recommendation_id.astype(str).tolist()
        rc = sb.table("recommendations")\
               .select("id,composite_score,score_label").in_("id", rec_ids).execute()
        if rc.data:
            rc_df = pd.DataFrame(rc.data)
            rc_df["id"] = rc_df["id"].astype(str)
            closed_port["recommendation_id"] = closed_port["recommendation_id"].astype(str)
            merged = closed_port.merge(rc_df, left_on="recommendation_id", right_on="id", how="inner")
            if not merged.empty:
                fig_sc = px.scatter(merged, x="composite_score", y="pnl_pct",
                                     color="pnl_pct", color_continuous_scale="RdYlGn",
                                     hover_data=["ticker","score_label","exit_reason"],
                                     title="Composite Score vs Realised P&L %")
                fig_sc.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_sc.update_layout(height=350,
                                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_sc, use_container_width=True)

# ═══════════════════════════════════════════════
#  PAGE: BACKTEST LAB
# ═══════════════════════════════════════════════
elif page == "🔬 Backtest Lab":
    st.title("🔬 Backtest Lab")
    st.caption("Walk-forward simulation: did our signals actually work in live markets?")

    sim_meta = load_sim_meta()
    if sim_meta:
        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Last Run",         str(sim_meta.get("last_run", "Never")))
        sm2.metric("Total Simulated",  safe_int(sim_meta.get("total_simulated", 0)))
        sm3.metric("Actual Win Rate",  f"{safe_float(sim_meta.get('actual_win_rate',0)):.1f}%")
        sm4.metric("Actual Avg Return",f"{safe_float(sim_meta.get('actual_avg_return',0)):+.2f}%")
    else:
        st.info("No simulations yet. The backsimulator runs weekly (Sunday) or trigger from GitHub Actions.")
        st.stop()

    days_back = st.slider("Look back N days", 30, 365, 90, key="sim_days")
    sims = load_simulations(days_back)

    if sims.empty:
        st.info(f"No simulations found in the past {days_back} days.")
        st.stop()

    wins   = sims[sims.was_win == True]
    losses = sims[sims.was_win == False]
    total  = len(sims)
    wr     = len(wins) / total * 100 if total > 0 else 0
    avg_r  = sims.actual_return_pct.mean()

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Total Trades",    total)
    s2.metric("Wins",            len(wins))
    s3.metric("Losses",          len(losses))
    s4.metric("Actual Win Rate", f"{wr:.1f}%")
    s5.metric("Actual Avg Return", f"{avg_r:+.2f}%")

    st.divider()

    # ── Return distribution
    c1, c2 = st.columns(2)
    with c1:
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Histogram(
            x=sims.actual_return_pct.dropna(), nbinsx=25,
            marker_color="#26a69a", name="Returns",
            marker_line_width=0.5, marker_line_color="white",
        ))
        fig_ret.add_vline(x=0,      line_color="#ef5350", line_dash="dash",
                           annotation_text="0%")
        fig_ret.add_vline(x=avg_r,  line_color="#ffb300", line_dash="dot",
                           annotation_text=f"Avg {avg_r:+.1f}%")
        fig_ret.update_layout(title="Actual Return Distribution",
                               height=300, xaxis_title="Return %",
                               margin=dict(t=40,b=10,l=0,r=0),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ret, use_container_width=True)

    with c2:
        exit_counts = sims.exit_reason.value_counts().reset_index()
        exit_counts.columns = ["Exit Reason", "Count"]
        fig_exit = px.pie(exit_counts, names="Exit Reason", values="Count",
                           color="Exit Reason",
                           color_discrete_map={"sl_hit":"#ef5350",
                                               "target_hit":"#26a69a",
                                               "timeout":"#ffa726"},
                           title="Exit Reason Breakdown")
        fig_exit.update_layout(height=300, margin=dict(t=40,b=10,l=0,r=0))
        st.plotly_chart(fig_exit, use_container_width=True)

    # ── Composite score vs actual return
    st.subheader("📈 Composite Score vs Actual Return %")
    st.caption("Does a higher signal score actually predict better outcomes?")
    if "composite_score" in sims.columns and sims.composite_score.notna().any():
        fig_scatter = px.scatter(
            sims.dropna(subset=["composite_score","actual_return_pct"]),
            x="composite_score", y="actual_return_pct",
            color="actual_return_pct", color_continuous_scale="RdYlGn",
            hover_data=["ticker","signal_date","exit_reason","days_held"],
            trendline="ols",
            title="Signal Composite Score vs Realised Return",
        )
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_scatter.update_layout(height=380,
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Bucket analysis: score quartiles vs win rate
        sims_clean = sims.dropna(subset=["composite_score","actual_return_pct"]).copy()
        if len(sims_clean) >= 10:
            sims_clean["score_bucket"] = pd.qcut(
                sims_clean.composite_score, q=4,
                labels=["Q1 Low","Q2","Q3","Q4 High"], duplicates="drop")
            bucket_stats = sims_clean.groupby("score_bucket", observed=True).agg(
                Win_Rate  = ("was_win", lambda x: x.mean() * 100),
                Avg_Return= ("actual_return_pct", "mean"),
                Count     = ("actual_return_pct", "count"),
            ).reset_index()
            bucket_stats.columns = ["Score Bucket","Win Rate %","Avg Return %","Trades"]
            st.subheader("📊 Win Rate by Score Quartile")
            fig_bucket = px.bar(bucket_stats, x="Score Bucket", y="Win Rate %",
                                 color="Win Rate %", color_continuous_scale="RdYlGn",
                                 text="Win Rate %", title="Do higher-scored signals win more?")
            fig_bucket.add_hline(y=50, line_dash="dash", line_color="gray",
                                  annotation_text="50% line")
            fig_bucket.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_bucket.update_layout(height=300, plot_bgcolor="rgba(0,0,0,0)",
                                      paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bucket, use_container_width=True)
            st.dataframe(bucket_stats, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📋 Recent Simulated Trades")
    disp_cols = ["ticker","signal_date","exit_date","entry_price","exit_price",
                 "actual_return_pct","exit_reason","days_held",
                 "composite_score","predicted_win_rate","was_win"]
    disp_cols = [c for c in disp_cols if c in sims.columns]
    st.dataframe(
        sims[disp_cols].head(50).style.format({
            "actual_return_pct":  "{:+.2f}%",
            "composite_score":    "{:.1f}",
            "predicted_win_rate": "{:.1f}%",
            "entry_price":        "₹{:.2f}",
            "exit_price":         "₹{:.2f}",
        }),
        use_container_width=True, hide_index=True
    )

# ═══════════════════════════════════════════════
#  PAGE: OPTIMIZER
# ═══════════════════════════════════════════════
elif page == "🤖 Optimizer":
    st.title("🤖 Parameter Optimizer")
    st.caption("Champion vs challenger, full version history, promotion controls, optimization history.")

    params_df = load_agent_params()
    opt_runs  = load_opt_runs()

    if params_df.empty:
        st.info("No optimization runs yet. The optimizer runs every Sunday at 11 PM IST.")
        st.stop()

    champion   = params_df[params_df.status == "champion"].head(1)
    challenger = params_df[params_df.status == "challenger"].head(1)
    candidates = params_df[params_df.status == "candidate"].copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 👑 Champion")
        if not champion.empty:
            r = champion.iloc[0]
            st.success(f"**v{r['version']}** | Score: {safe_float(r['objective_score']):.4f}")
            st.caption(f"PF: {safe_float(r.get('profit_factor',0)):.2f} | "
                       f"WR: {safe_float(r.get('win_rate',0)):.1f}% | "
                       f"Avg: {safe_float(r.get('avg_return',0)):+.2f}%")
            st.caption(f"Promoted: {r.get('promoted_at','?')}")
        else:
            st.warning("No champion yet")

    with col2:
        st.markdown("### ⚔️ Challenger")
        if not challenger.empty:
            r = challenger.iloc[0]
            champ_s = safe_float(champion.iloc[0]["objective_score"]) if not champion.empty else 0
            delta   = safe_float(r["objective_score"]) - champ_s
            marker  = "🟢" if delta > 0 else "🔴"
            st.info(f"**v{r['version']}** | Score: {safe_float(r['objective_score']):.4f} {marker} {delta:+.4f}")
            st.caption(f"PF: {safe_float(r.get('profit_factor',0)):.2f} | "
                       f"WR: {safe_float(r.get('win_rate',0)):.1f}% | "
                       f"Avg: {safe_float(r.get('avg_return',0)):+.2f}%")
        else:
            st.info("No challenger yet")

    with col3:
        st.markdown("### 📋 Candidates")
        st.metric("Available", len(candidates))
        if not opt_runs.empty:
            st.caption(f"Last run: {opt_runs.iloc[0]['run_date']}")

    st.divider()

    if not champion.empty and not challenger.empty:
        st.subheader("📊 Parameter Diff — Champion vs Challenger")
        champ_p = json.loads(champion.iloc[0]["params_json"]) if isinstance(champion.iloc[0]["params_json"],str) else champion.iloc[0]["params_json"]
        chall_p = json.loads(challenger.iloc[0]["params_json"]) if isinstance(challenger.iloc[0]["params_json"],str) else challenger.iloc[0]["params_json"]
        all_keys = sorted(set(list(champ_p.keys()) + list(chall_p.keys())))
        diff_rows = []
        for k in all_keys:
            cv  = champ_p.get(k, "–")
            chv = chall_p.get(k, "–")
            diff_rows.append({"Parameter": k,
                               f"Champion v{champion.iloc[0]['version']}": cv,
                               f"Challenger v{challenger.iloc[0]['version']}": chv,
                               "Changed": "✅" if cv != chv else ""})
        diff_df = pd.DataFrame(diff_rows)
        changed_only = st.checkbox("Show changed params only", value=True)
        if changed_only:
            diff_df = diff_df[diff_df.Changed == "✅"]
        st.dataframe(diff_df, use_container_width=True, hide_index=True)

        st.subheader("📈 Walk-Forward Metrics")
        metrics = [("Objective Score","objective_score"),("Profit Factor","profit_factor"),
                   ("Win Rate %","win_rate"),("Avg Return %","avg_return"),("Max Drawdown","max_drawdown")]
        mc_cols = st.columns(5)
        for col, (label, field) in zip(mc_cols, metrics):
            cv  = safe_float(champion.iloc[0].get(field, 0))
            chv = safe_float(challenger.iloc[0].get(field, 0))
            d   = chv - cv
            better = (d > 0 and field != "max_drawdown") or (d < 0 and field == "max_drawdown")
            col.metric(label, f"{chv:.3f}", delta=f"{d:+.3f}",
                        delta_color="normal" if better else "inverse")

        st.divider()
        st.warning("⚠️ Only promote after monitoring paper trades for 2–4 weeks.")
        pr1, pr2 = st.columns(2)
        with pr1:
            if st.button(f"👑 Promote Challenger v{challenger.iloc[0]['version']} → Champion",
                          type="primary"):
                promote_param(int(challenger.iloc[0]["version"]), "champion")
                st.success("✅ Promoted!"); st.rerun()
        with pr2:
            if st.button(f"🗑️ Retire Challenger v{challenger.iloc[0]['version']}"):
                promote_param(int(challenger.iloc[0]["version"]), "retired")
                st.info("Retired."); st.rerun()

    st.divider()
    st.subheader("📋 Candidates")
    if not candidates.empty:
        cand_cols = ["version","objective_score","profit_factor","win_rate",
                     "avg_return","max_drawdown","total_trades","run_date","rank"]
        cand_cols = [c for c in cand_cols if c in candidates.columns]
        st.dataframe(
            candidates[cand_cols].style.format({
                "objective_score":"{:.4f}","profit_factor":"{:.2f}",
                "win_rate":"{:.1f}%","avg_return":"{:+.2f}%","max_drawdown":"{:.1f}%",
            }), use_container_width=True, hide_index=True)
        sel_v = st.selectbox("Promote a candidate", candidates["version"].tolist())
        mc1, mc2 = st.columns(2)
        if mc1.button("⚔️ Set as Challenger"):
            promote_param(int(sel_v), "challenger"); st.rerun()
        if mc2.button("👑 Promote to Champion"):
            promote_param(int(sel_v), "champion");   st.rerun()

    st.divider()
    st.subheader("📜 Optimization History")
    if not opt_runs.empty:
        fig_h = px.line(opt_runs.sort_values("run_date"), x="run_date", y="best_score",
                         markers=True, title="Best Objective Score per Run")
        fig_h.update_layout(height=260, margin=dict(t=30,b=10,l=0,r=0),
                             plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_h, use_container_width=True)
        show_opt = ["run_date","n_trials","n_valid_trials","best_score",
                    "best_profit_factor","best_win_rate","best_avg_return",
                    "champion_version","challenger_version"]
        show_opt = [c for c in show_opt if c in opt_runs.columns]
        st.dataframe(opt_runs[show_opt].style.format({
            "best_score":"{:.4f}","best_profit_factor":"{:.2f}",
            "best_win_rate":"{:.1f}%","best_avg_return":"{:+.2f}%",
        }), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("⚙️ Active Parameters")
    if not champion.empty:
        ap = json.loads(champion.iloc[0]["params_json"]) if isinstance(champion.iloc[0]["params_json"],str) else champion.iloc[0]["params_json"]
        st.dataframe(pd.DataFrame([{"Parameter":k,"Value":v} for k,v in sorted(ap.items())]),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Using default parameters.")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ Paper trading & educational purposes only. Not financial advice. "
    "Past backtest results do not guarantee future performance."
)
