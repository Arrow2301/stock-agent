"""
============================================================
  Indian Stock Agent — Streamlit Dashboard
  Hosted on Streamlit Community Cloud (free)
  Features:
    • Today's buy/sell recommendations with backtest stats
    • One-click paper trading (buy / sell)
    • Live portfolio P&L tracker
    • Historical recommendation log
============================================================
"""

import os
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
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = get_supabase()

# ─────────────────────────────────────────────
#  PASSWORD GATE  (simple, keeps it personal)
# ─────────────────────────────────────────────
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True
    with st.container():
        st.title("🇮🇳 Indian Stock Agent")
        st.subheader("🔐 Login")
        pwd = st.text_input("Password", type="password", key="pwd_input")
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
#  DATA HELPERS
# ─────────────────────────────────────────────

@st.cache_data(ttl=300)   # refresh every 5 minutes
def load_recommendations(target_date: str = None):
    q = supabase.table("recommendations").select("*").order("score", desc=True)
    if target_date:
        q = q.eq("date", target_date)
    res = q.execute()
    if not res.data:
        return pd.DataFrame()
    df = pd.DataFrame(res.data)
    df["signals"]  = df["signals"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    df["backtest"] = df["backtest"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    return df

@st.cache_data(ttl=60)
def load_portfolio():
    res = supabase.table("paper_portfolio").select("*").order("buy_date", desc=True).execute()
    return pd.DataFrame(res.data) if res.data else pd.DataFrame()

@st.cache_data(ttl=60)
def get_live_price(ticker: str) -> float:
    try:
        data = yf.download(ticker + ".NS", period="2d", progress=False, auto_adjust=True)
        if data.empty:
            return 0.0
        col = data.columns[0][0] if isinstance(data.columns[0], tuple) else data.columns[0]
        return float(data["Close"].iloc[-1])
    except Exception:
        return 0.0

@st.cache_data(ttl=300)
def get_meta():
    res = supabase.table("agent_meta").select("*").eq("id", 1).execute()
    return res.data[0] if res.data else {}

@st.cache_data(ttl=600)
def get_price_history(ticker: str, days: int = 90):
    try:
        df = yf.download(
            ticker + ".NS",
            start=date.today() - timedelta(days=days),
            end=date.today(),
            progress=False, auto_adjust=True
        )
        if df.empty:
            return pd.DataFrame()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception:
        return pd.DataFrame()

# ─────────────────────────────────────────────
#  PORTFOLIO ACTIONS
# ─────────────────────────────────────────────

def paper_buy(ticker, price, qty, notes=""):
    supabase.table("paper_portfolio").insert({
        "ticker":    ticker,
        "buy_date":  date.today().isoformat(),
        "buy_price": price,
        "quantity":  qty,
        "status":    "OPEN",
        "notes":     notes,
    }).execute()
    st.cache_data.clear()

def paper_sell(trade_id, ticker, sell_price, buy_price, qty):
    pnl_pct = round((sell_price - buy_price) / buy_price * 100, 2)
    pnl_inr = round((sell_price - buy_price) * qty, 2)
    supabase.table("paper_portfolio").update({
        "sell_date":  date.today().isoformat(),
        "sell_price": sell_price,
        "status":     "CLOSED",
        "pnl_pct":    pnl_pct,
        "pnl_inr":    pnl_inr,
    }).eq("id", trade_id).execute()
    st.cache_data.clear()

# ─────────────────────────────────────────────
#  CHART HELPERS
# ─────────────────────────────────────────────

def candlestick_chart(df: pd.DataFrame, ticker: str, buy_price: float = None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close,
        name=ticker, increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ))
    # 9 & 21 EMA
    df["ema9"]  = df.Close.ewm(span=9,  adjust=False).mean()
    df["ema21"] = df.Close.ewm(span=21, adjust=False).mean()
    fig.add_trace(go.Scatter(x=df.index, y=df.ema9,  name="EMA 9",  line=dict(color="#ffb300", width=1.2)))
    fig.add_trace(go.Scatter(x=df.index, y=df.ema21, name="EMA 21", line=dict(color="#ab47bc", width=1.2)))
    # Buy price line
    if buy_price:
        fig.add_hline(y=buy_price, line_color="#00e676", line_dash="dash",
                      annotation_text=f"Buy ₹{buy_price:.2f}", annotation_position="top left")
    fig.update_layout(
        height=380, margin=dict(t=20, b=20, l=0, r=0),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def pnl_chart(closed_trades: pd.DataFrame):
    if closed_trades.empty:
        return None
    df = closed_trades.sort_values("sell_date").copy()
    df["cumulative_pnl"] = df["pnl_inr"].cumsum()
    fig = px.area(df, x="sell_date", y="cumulative_pnl",
                  labels={"cumulative_pnl": "Cumulative P&L (₹)", "sell_date": "Date"},
                  color_discrete_sequence=["#26a69a"])
    fig.add_hline(y=0, line_color="#ef5350", line_dash="dash")
    fig.update_layout(height=300, margin=dict(t=10, b=20, l=0, r=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("🇮🇳 Stock Agent")

    meta = get_meta()
    if meta:
        st.success(f"Last run: **{meta.get('last_run', 'N/A')}**")
        st.caption(f"Scanned: {meta.get('tickers_scanned', 0)} stocks  |  Signals: {meta.get('total_signals', 0)}")
    else:
        st.warning("Agent hasn't run yet today")

    st.divider()
    page = st.radio("Navigate", [
        "📊 Today's Signals",
        "💼 My Paper Portfolio",
        "📅 Signal History",
        "📈 Strategy Stats",
    ])

    st.divider()
    st.caption("NSE market hours: 9:15 AM – 3:30 PM IST")
    st.caption("Agent runs: 7:00 AM IST daily (GitHub Actions)")
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

# ─────────────────────────────────────────────
#  PAGE: TODAY'S SIGNALS
# ─────────────────────────────────────────────

if page == "📊 Today's Signals":
    today_str = date.today().isoformat()
    st.title("📊 Today's Signals")
    st.caption(f"Date: {datetime.today().strftime('%A, %d %B %Y')}")

    recs = load_recommendations(today_str)

    if recs.empty:
        st.info("No signals found for today. The agent may not have run yet, or no stocks crossed the threshold today.")
        st.stop()

    buys  = recs[recs.action == "BUY"].copy()
    sells = recs[recs.action == "SELL"].copy()

    # ── Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 Buy Signals",   len(buys))
    c2.metric("🔴 Sell Signals",  len(sells))
    c3.metric("⭐ Top Score",     recs.score.max() if not recs.empty else 0, help="Max = 4")
    c4.metric("📦 Stocks Scanned", meta.get("tickers_scanned", "–"))

    st.divider()

    # ── BUY signals
    if not buys.empty:
        st.subheader("🟢 Buy Recommendations")
        for _, row in buys.iterrows():
            score_stars = "⭐" * int(row.score)
            with st.expander(
                f"{score_stars}  **{row.ticker}**  —  ₹{row.price:,.2f}  |  "
                f"Score {row.score}/4  |  Win Rate {row.win_rate:.0f}%  |  "
                f"Avg Return {row.avg_return:+.2f}%",
                expanded=(row.score >= 3)
            ):
                col1, col2, col3 = st.columns(3)
                col1.metric("Price",    f"₹{row.price:,.2f}")
                col1.metric("1D Change", f"{row.change_1d:+.2f}%",
                             delta_color="normal" if row.change_1d >= 0 else "inverse")
                col1.metric("5D Change", f"{row.change_5d:+.2f}%",
                             delta_color="normal" if row.change_5d >= 0 else "inverse")
                col2.metric("RSI",         f"{row.rsi:.1f}")
                col2.metric("Stop Loss",   f"₹{row.stop_loss:,.2f}  ({row.risk_pct:.1f}% risk)")
                col2.metric("Target",      f"₹{row.target:,.2f}  ({row.reward_pct:.1f}% upside)")
                col3.metric("Strategies",  row.active_strategies)
                col3.metric("Win Rate",    f"{row.win_rate:.0f}%")
                col3.metric("Avg Return",  f"{row.avg_return:+.2f}%")

                # Strategy signal table
                sigs = row.signals or {}
                bts  = row.backtest or {}
                sig_rows = []
                for name, val in sigs.items():
                    fired = "🟢 BUY" if val == 1 else ("🔴 SELL" if val == -1 else "⬜ None")
                    b = bts.get(name, {})
                    sig_rows.append({
                        "Strategy": name,
                        "Signal":   fired,
                        "Win Rate": f"{b.get('win_rate', 0):.0f}%",
                        "Avg Return": f"{b.get('avg_return', 0):+.2f}%",
                        "Trades (1yr)": b.get("trades", 0),
                    })
                st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)

                # Chart
                hist_df = get_price_history(row.ticker, 90)
                if not hist_df.empty:
                    st.plotly_chart(candlestick_chart(hist_df, row.ticker), use_container_width=True)

                # Paper buy form
                st.markdown("**📝 Paper Trade**")
                pb_col1, pb_col2, pb_col3 = st.columns(3)
                qty   = pb_col1.number_input("Quantity", min_value=1, value=10, key=f"qty_{row.ticker}")
                price = pb_col2.number_input("Buy Price (₹)", value=float(row.price), key=f"px_{row.ticker}")
                notes = pb_col3.text_input("Notes", value=row.active_strategies, key=f"note_{row.ticker}")
                if st.button(f"🟢 Paper Buy {row.ticker}", key=f"buy_{row.ticker}"):
                    paper_buy(row.ticker, price, qty, notes)
                    st.success(f"✅ Paper bought {qty} × {row.ticker} @ ₹{price:.2f}")
                    st.balloons()

    st.divider()

    # ── SELL signals
    if not sells.empty:
        st.subheader("🔴 Sell / Exit Recommendations")
        for _, row in sells.iterrows():
            score_stars = "🔴" * int(row.score)
            with st.expander(
                f"{score_stars}  **{row.ticker}**  —  ₹{row.price:,.2f}  |  Score {row.score}/4",
                expanded=(row.score >= 3)
            ):
                col1, col2 = st.columns(2)
                col1.metric("Price",       f"₹{row.price:,.2f}")
                col1.metric("RSI",         f"{row.rsi:.1f}", help="RSI > 65 = overbought")
                col2.metric("Strategies",  row.active_strategies)
                col2.metric("Win Rate",    f"{row.win_rate:.0f}%")
                hist_df = get_price_history(row.ticker, 90)
                if not hist_df.empty:
                    st.plotly_chart(candlestick_chart(hist_df, row.ticker), use_container_width=True)

# ─────────────────────────────────────────────
#  PAGE: PAPER PORTFOLIO
# ─────────────────────────────────────────────

elif page == "💼 My Paper Portfolio":
    st.title("💼 My Paper Portfolio")

    port = load_portfolio()

    if not port.empty:
        open_pos   = port[port.status == "OPEN"].copy()
        closed_pos = port[port.status == "CLOSED"].copy()
    else:
        open_pos   = pd.DataFrame()
        closed_pos = pd.DataFrame()

    # ── Summary metrics
    total_invested = 0
    total_current  = 0
    if not open_pos.empty:
        for _, row in open_pos.iterrows():
            lp = get_live_price(row.ticker)
            total_invested += row.buy_price * row.quantity
            total_current  += (lp if lp > 0 else row.buy_price) * row.quantity

    total_pnl_closed = float(closed_pos.pnl_inr.sum()) if not closed_pos.empty else 0
    open_pnl         = total_current - total_invested
    win_trades = len(closed_pos[closed_pos.pnl_pct > 0]) if not closed_pos.empty else 0
    win_rate   = win_trades / len(closed_pos) * 100 if not closed_pos.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Open Positions",   len(open_pos))
    c2.metric("Open P&L",         f"₹{open_pnl:+,.0f}",
               delta=f"{open_pnl/total_invested*100:+.1f}%" if total_invested else "–")
    c3.metric("Realised P&L",     f"₹{total_pnl_closed:+,.0f}")
    c4.metric("Win Rate (closed)", f"{win_rate:.0f}%", delta=f"{win_trades}/{len(closed_pos)} trades")

    st.divider()

    # ── Open positions
    st.subheader("📂 Open Positions")
    if open_pos.empty:
        st.info("No open positions. Go to 'Today's Signals' to paper buy a stock.")
    else:
        for _, row in open_pos.iterrows():
            live_price = get_live_price(row.ticker)
            if live_price > 0:
                pnl_pct = (live_price - row.buy_price) / row.buy_price * 100
                pnl_inr = (live_price - row.buy_price) * row.quantity
            else:
                live_price = row.buy_price
                pnl_pct = pnl_inr = 0

            color = "🟢" if pnl_pct >= 0 else "🔴"
            with st.expander(
                f"{color} **{row.ticker}**  |  {row.quantity} shares  |  "
                f"Buy ₹{row.buy_price:,.2f}  →  Live ₹{live_price:,.2f}  |  "
                f"P&L {pnl_pct:+.2f}%  (₹{pnl_inr:+,.0f})",
                expanded=False
            ):
                col1, col2, col3 = st.columns(3)
                col1.metric("Buy Price",  f"₹{row.buy_price:,.2f}")
                col1.metric("Buy Date",   str(row.buy_date))
                col2.metric("Live Price", f"₹{live_price:,.2f}")
                col2.metric("Invested",   f"₹{row.buy_price * row.quantity:,.0f}")
                col3.metric("P&L %",  f"{pnl_pct:+.2f}%",
                             delta_color="normal" if pnl_pct >= 0 else "inverse")
                col3.metric("P&L ₹",  f"₹{pnl_inr:+,.0f}",
                             delta_color="normal" if pnl_inr >= 0 else "inverse")
                if row.notes:
                    st.caption(f"Signal: {row.notes}")

                hist_df = get_price_history(row.ticker, 60)
                if not hist_df.empty:
                    st.plotly_chart(
                        candlestick_chart(hist_df, row.ticker, buy_price=row.buy_price),
                        use_container_width=True
                    )

                # Sell form
                st.markdown("**Paper Sell**")
                s1, s2 = st.columns(2)
                sell_px = s1.number_input("Sell Price (₹)", value=float(live_price),
                                           key=f"sell_px_{row.id}")
                if s2.button(f"🔴 Paper Sell {row.ticker}", key=f"sell_{row.id}"):
                    paper_sell(row.id, row.ticker, sell_px, row.buy_price, row.quantity)
                    st.success(f"✅ Closed {row.ticker}: ₹{sell_px:.2f}")
                    st.rerun()

    st.divider()

    # ── Closed positions
    st.subheader("✅ Closed Trades")
    if closed_pos.empty:
        st.info("No closed trades yet.")
    else:
        pnl_fig = pnl_chart(closed_pos)
        if pnl_fig:
            st.plotly_chart(pnl_fig, use_container_width=True)

        display_cols = ["ticker", "buy_date", "sell_date", "buy_price",
                        "sell_price", "quantity", "pnl_pct", "pnl_inr"]
        display_cols = [c for c in display_cols if c in closed_pos.columns]
        closed_display = closed_pos[display_cols].copy()
        closed_display.columns = [c.replace("_", " ").title() for c in display_cols]

        def color_pnl(val):
            try:
                v = float(val)
                return "color: #26a69a" if v > 0 else "color: #ef5350"
            except Exception:
                return ""

        st.dataframe(
            closed_display.style
                .applymap(color_pnl, subset=["Pnl Pct", "Pnl Inr"])
                .format({"Pnl Pct": "{:+.2f}%", "Pnl Inr": "₹{:+,.0f}",
                          "Buy Price": "₹{:.2f}", "Sell Price": "₹{:.2f}"}),
            use_container_width=True, hide_index=True
        )

# ─────────────────────────────────────────────
#  PAGE: SIGNAL HISTORY
# ─────────────────────────────────────────────

elif page == "📅 Signal History":
    st.title("📅 Signal History")
    st.caption("Browse all past recommendations generated by the agent")

    days_back = st.slider("Show past N days", 7, 60, 14)
    start_date = (date.today() - timedelta(days=days_back)).isoformat()

    res = supabase.table("recommendations").select("*")\
          .gte("date", start_date).order("date", desc=True).execute()
    hist = pd.DataFrame(res.data) if res.data else pd.DataFrame()

    if hist.empty:
        st.info("No history yet.")
    else:
        # Summary chart: signals per day
        daily = hist.groupby(["date","action"]).size().reset_index(name="count")
        fig = px.bar(daily, x="date", y="count", color="action",
                     color_discrete_map={"BUY": "#26a69a", "SELL": "#ef5350"},
                     title="Daily signal count")
        fig.update_layout(height=280, margin=dict(t=40,b=20,l=0,r=0),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # Filterable table
        action_filter = st.selectbox("Filter by action", ["All","BUY","SELL"])
        if action_filter != "All":
            hist = hist[hist.action == action_filter]

        display = hist[["date","ticker","action","score","price","rsi",
                         "win_rate","avg_return","active_strategies"]].copy()
        display.columns = ["Date","Ticker","Action","Score","Price (₹)","RSI",
                            "Win Rate %","Avg Return %","Strategies"]
        st.dataframe(display, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
#  PAGE: STRATEGY STATS
# ─────────────────────────────────────────────

elif page == "📈 Strategy Stats":
    st.title("📈 Strategy Performance Stats")
    st.caption("Aggregated backtest stats across all recommendations generated by the agent")

    res = supabase.table("recommendations").select("ticker, backtest, action, win_rate, avg_return").execute()
    all_recs = pd.DataFrame(res.data) if res.data else pd.DataFrame()

    if all_recs.empty:
        st.info("No data yet. Run the agent first.")
    else:
        # Unpack backtest JSON and aggregate by strategy
        rows = []
        for _, r in all_recs.iterrows():
            try:
                bt = json.loads(r.backtest) if isinstance(r.backtest, str) else r.backtest
                for name, stats in bt.items():
                    rows.append({
                        "Strategy":   name,
                        "Win Rate":   stats.get("win_rate", 0),
                        "Avg Return": stats.get("avg_return", 0),
                        "Trades":     stats.get("trades", 0),
                    })
            except Exception:
                continue

        if not rows:
            st.info("No backtest data available yet.")
        else:
            bt_df = pd.DataFrame(rows)
            agg = bt_df.groupby("Strategy").agg(
                Win_Rate   = ("Win Rate",   "mean"),
                Avg_Return = ("Avg Return", "mean"),
                Total_Trades = ("Trades",   "sum"),
            ).reset_index()
            agg.columns = ["Strategy", "Avg Win Rate %", "Avg Return %", "Total Trades"]

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(agg, x="Strategy", y="Avg Win Rate %",
                              color="Avg Win Rate %", color_continuous_scale="teal",
                              title="Win Rate by Strategy")
                fig.update_layout(height=300, showlegend=False,
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig2 = px.bar(agg, x="Strategy", y="Avg Return %",
                               color="Avg Return %", color_continuous_scale="RdYlGn",
                               title="Avg Return % by Strategy")
                fig2.update_layout(height=300, showlegend=False,
                                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(agg.style.format(
                {"Avg Win Rate %": "{:.1f}%", "Avg Return %": "{:+.2f}%"}
            ), use_container_width=True, hide_index=True)

        # Overall portfolio stats (from paper trades)
        closed = load_portfolio()
        if not closed.empty:
            closed = closed[closed.status == "CLOSED"]
        if not closed.empty and "pnl_pct" in closed.columns:
            st.divider()
            st.subheader("📊 Your Paper Trading Stats")
            total = len(closed)
            wins  = len(closed[closed.pnl_pct > 0])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Closed Trades", total)
            c2.metric("Win Rate",  f"{wins/total*100:.0f}%")
            c3.metric("Avg P&L",   f"{closed.pnl_pct.mean():+.2f}%")
            c4.metric("Best Trade",  f"{closed.pnl_pct.max():+.2f}%" if not closed.empty else "–")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ This dashboard is for **paper trading and educational purposes only**. "
    "Not financial advice. Always do your own research before trading real money."
)
