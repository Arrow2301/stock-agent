"""
============================================================
  Indian Stock Agent — Dashboard v2
  Hosted on Streamlit Community Cloud

  New in v2:
  ─ Exit alert banner (stop-loss / target hit warnings)
  ─ Composite score 0–100 with breakdown
  ─ Backtest shows SL exits, target exits, profit factor, median return
  ─ Filters: min composite score, action, min win rate
  ─ Portfolio stores SL / target at entry; tracks breach in real time
  ─ Paper buys linked to recommendation ID for later analysis
  ─ Strategy Stats with sample-size warnings
  ─ Score vs realised P&L scatter (after enough trades)
============================================================
"""

import os, json
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
#  HELPERS
# ─────────────────────────────────────────────

def score_color(s):
    if s >= 80: return "#00c853"
    if s >= 65: return "#69f0ae"
    if s >= 50: return "#ffeb3b"
    if s >= 35: return "#ffa726"
    return "#ef5350"

def score_badge(s, label):
    col = score_color(s)
    return f"""<span style="background:{col};color:#111;padding:3px 10px;
               border-radius:12px;font-weight:700;font-size:1.1em">{s:.0f} — {label}</span>"""

@st.cache_data(ttl=300)
def load_recs(target_date=None, days_back=None):
    q = sb.table("recommendations").select("*").order("composite_score", desc=True)
    if target_date:
        q = q.eq("date", target_date)
    elif days_back:
        start = (date.today()-timedelta(days=days_back)).isoformat()
        q = q.gte("date", start)
    res = q.execute()
    if not res.data: return pd.DataFrame()
    df = pd.DataFrame(res.data)
    for col in ["signals","backtest","score_breakdown","strategy_weights"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x,str) else (x or {}))
    return df

@st.cache_data(ttl=60)
def load_portfolio():
    res = sb.table("paper_portfolio").select("*").order("buy_date",desc=True).execute()
    return pd.DataFrame(res.data) if res.data else pd.DataFrame()

@st.cache_data(ttl=90)
def live_price(ticker):
    try:
        df = yf.download(ticker+".NS", period="3d", progress=False, auto_adjust=True)
        if df.empty: return 0.0
        return float(df["Close"].iloc[-1])
    except: return 0.0

@st.cache_data(ttl=600)
def price_history(ticker, days=90):
    try:
        df = yf.download(
            ticker+".NS",
            start=date.today()-timedelta(days=days),
            end=date.today(),
            progress=False, auto_adjust=True
        )
        if df.empty: return pd.DataFrame()
        df.columns = [c[0] if isinstance(c,tuple) else c for c in df.columns]
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except: return pd.DataFrame()

@st.cache_data(ttl=120)
def get_meta():
    res = sb.table("agent_meta").select("*").eq("id",1).execute()
    return res.data[0] if res.data else {}

def paper_buy(ticker, price, qty, sl, tgt, notes, rec_id=None):
    sb.table("paper_portfolio").insert({
        "ticker":            ticker,
        "buy_date":          date.today().isoformat(),
        "buy_price":         price,
        "quantity":          qty,
        "entry_stop_loss":   sl,
        "entry_target":      tgt,
        "status":            "OPEN",
        "notes":             notes,
        "recommendation_id": rec_id,
    }).execute()
    st.cache_data.clear()

def paper_sell(trade_id, sell_price, buy_price, qty, exit_reason="manual"):
    pnl_pct = round((sell_price-buy_price)/buy_price*100, 2)
    pnl_inr = round((sell_price-buy_price)*qty, 2)
    sb.table("paper_portfolio").update({
        "sell_date":   date.today().isoformat(),
        "sell_price":  sell_price,
        "status":      "CLOSED",
        "pnl_pct":     pnl_pct,
        "pnl_inr":     pnl_inr,
        "exit_reason": exit_reason,
    }).eq("id", trade_id).execute()
    st.cache_data.clear()

def candlestick(df, ticker, buy_price=None, sl=None, tgt=None):
    df = df.copy()
    df["ema9"]  = df.Close.ewm(span=9,  adjust=False).mean()
    df["ema21"] = df.Close.ewm(span=21, adjust=False).mean()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close,
        name=ticker, increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df.ema9,  name="EMA 9",  line=dict(color="#ffb300",width=1.2)))
    fig.add_trace(go.Scatter(x=df.index, y=df.ema21, name="EMA 21", line=dict(color="#ab47bc",width=1.2)))
    if buy_price:
        fig.add_hline(y=buy_price, line_color="#00e676", line_dash="dash",
                      annotation_text=f"Buy ₹{buy_price:.2f}")
    if sl:
        fig.add_hline(y=sl,  line_color="#ef5350", line_dash="dot",
                      annotation_text=f"SL ₹{sl:.2f}")
    if tgt:
        fig.add_hline(y=tgt, line_color="#69f0ae", line_dash="dot",
                      annotation_text=f"Target ₹{tgt:.2f}")
    fig.update_layout(height=360, xaxis_rangeslider_visible=False,
                      margin=dict(t=10,b=10,l=0,r=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.02))
    return fig

# ─────────────────────────────────────────────
#  EXIT ALERT CHECKER
# ─────────────────────────────────────────────

def check_exit_alerts(port: pd.DataFrame):
    """Returns list of dicts for positions that hit SL or target."""
    alerts = []
    if port.empty: return alerts
    open_pos = port[port.status=="OPEN"]
    for _, row in open_pos.iterrows():
        lp = live_price(row.ticker)
        if lp <= 0: continue
        sl  = row.get("entry_stop_loss")
        tgt = row.get("entry_target")
        if sl and lp <= float(sl):
            pnl = round((lp - row.buy_price)/row.buy_price*100, 2)
            alerts.append({"ticker": row.ticker, "type": "SL_HIT", "lp": lp,
                            "level": sl, "pnl": pnl, "id": row.id,
                            "buy_price": row.buy_price, "qty": row.quantity})
        elif tgt and lp >= float(tgt):
            pnl = round((lp - row.buy_price)/row.buy_price*100, 2)
            alerts.append({"ticker": row.ticker, "type": "TARGET_HIT", "lp": lp,
                            "level": tgt, "pnl": pnl, "id": row.id,
                            "buy_price": row.buy_price, "qty": row.quantity})
    return alerts

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("🇮🇳 Stock Agent")
    meta = get_meta()
    if meta:
        st.success(f"Last run: **{meta.get('last_run','N/A')}**")
        regime = meta.get("market_regime", "?")
        regime_emoji = {"BULLISH":"🟢","BEARISH":"🔴","NEUTRAL":"🟡"}.get(regime,"⬜")
        st.caption(f"Market: {regime_emoji} **{regime}**  |  Signals: {meta.get('total_signals',0)}")
    else:
        st.warning("Agent hasn't run yet")

    st.divider()
    page = st.radio("Navigate", [
        "📊 Today's Signals",
        "💼 My Paper Portfolio",
        "📅 Signal History",
        "📈 Strategy Stats",
    ])

    st.divider()
    st.caption("NSE hours: 9:15 AM – 3:30 PM IST")
    st.caption("Agent runs: 7:00 AM IST, Mon–Fri")
    if st.button("🔄 Refresh"):
        st.cache_data.clear(); st.rerun()

# ─────────────────────────────────────────────
#  EXIT ALERT BANNER (shown on all pages)
# ─────────────────────────────────────────────

port_all = load_portfolio()
alerts   = check_exit_alerts(port_all)

if alerts:
    for a in alerts:
        if a["type"] == "SL_HIT":
            st.error(
                f"🚨 **STOP-LOSS HIT — {a['ticker']}** | "
                f"Live: ₹{a['lp']:,.2f}  ≤  SL: ₹{float(a['level']):,.2f} | "
                f"P&L: {a['pnl']:+.2f}% | Consider closing this position now"
            )
        else:
            st.success(
                f"🎯 **TARGET HIT — {a['ticker']}** | "
                f"Live: ₹{a['lp']:,.2f}  ≥  Target: ₹{float(a['level']):,.2f} | "
                f"P&L: {a['pnl']:+.2f}% | Consider booking profits"
            )

# ─────────────────────────────────────────────
#  PAGE: TODAY'S SIGNALS
# ─────────────────────────────────────────────

if page == "📊 Today's Signals":
    today_str = date.today().isoformat()
    st.title("📊 Today's Signals")
    st.caption(f"{datetime.today().strftime('%A, %d %B %Y')}  |  Sorted by Composite Score ↓")

    recs = load_recs(target_date=today_str)
    if recs.empty:
        st.info("No signals today yet. The agent runs at 7:00 AM IST — check back then, or trigger a manual run from GitHub Actions.")
        st.stop()

    buys  = recs[recs.action=="BUY"].copy()
    sells = recs[recs.action=="SELL"].copy()

    # ── Filters
    with st.expander("🔧 Filters", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        min_cs  = fc1.slider("Min Composite Score", 0, 100, 30)
        min_wr  = fc2.slider("Min Win Rate %", 0, 100, 40)
        min_pf  = fc3.slider("Min Profit Factor", 0.0, 5.0, 0.8, 0.1)

        buys  = buys[ (buys.composite_score>=min_cs) & (buys.win_rate>=min_wr)  & (buys.profit_factor>=min_pf)]
        sells = sells[(sells.composite_score>=min_cs) & (sells.win_rate>=min_wr) & (sells.profit_factor>=min_pf)]

    # ── Summary metrics
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🟢 Buy Signals",  len(buys))
    c2.metric("🔴 Sell Signals", len(sells))
    top = recs.composite_score.max() if not recs.empty else 0
    c3.metric("🏆 Top Score",    f"{top:.0f}/100")
    c4.metric("Market",         meta.get("market_regime","?"))
    c5.metric("Scanned",        meta.get("tickers_scanned","–"))

    st.divider()

    def render_signals(df_recs, action):
        if df_recs.empty:
            st.info(f"No {action} signals match current filters.")
            return
        for _, row in df_recs.iterrows():
            cs    = row.composite_score
            label = row.score_label
            lsw   = row.get("low_sample_warning", False)
            warn  = "  ⚠️ Low backtest sample" if lsw else ""

            with st.expander(
                f"**{row.ticker}**  |  "
                f"Composite: {cs:.0f}/100 ({label}){warn}  |  "
                f"Win Rate: {row.win_rate:.0f}%  |  "
                f"Profit Factor: {row.profit_factor:.2f}  |  "
                f"Avg Return: {row.avg_return:+.2f}%",
                expanded=(cs >= 65)
            ):
                # Composite score badge
                st.markdown(score_badge(cs, label), unsafe_allow_html=True)
                st.markdown("")

                # Score breakdown bar
                bd = row.score_breakdown or {}
                if bd:
                    bd_df = pd.DataFrame([
                        {"Component": "Strategy (40)", "Score": bd.get("strategy",0), "Max": 40},
                        {"Component": "RSI (20)",      "Score": bd.get("rsi",0),      "Max": 20},
                        {"Component": "Volume (15)",   "Score": bd.get("volume",0),   "Max": 15},
                        {"Component": "Risk:Reward (15)", "Score": bd.get("rr",0),    "Max": 15},
                        {"Component": "Regime (10)",   "Score": bd.get("regime",0),   "Max": 10},
                    ])
                    bd_df["%"] = (bd_df.Score / bd_df.Max * 100).clip(0,100)
                    fig_bd = px.bar(bd_df, x="Score", y="Component", orientation="h",
                                    color="%", color_continuous_scale="RdYlGn", range_color=[0,100],
                                    text="Score")
                    fig_bd.update_layout(height=200, margin=dict(t=0,b=0,l=0,r=0),
                                          showlegend=False, coloraxis_showscale=False,
                                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    fig_bd.update_traces(texttemplate="%{text:.1f}", textposition="inside")
                    st.plotly_chart(fig_bd, use_container_width=True)

                # Key metrics
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Price",      f"₹{row.price:,.2f}")
                m1.metric("1D Change",  f"{row.change_1d:+.2f}%")
                m1.metric("5D Change",  f"{row.change_5d:+.2f}%")
                m2.metric("RSI",        f"{row.rsi:.1f}")
                m2.metric("Stop Loss",  f"₹{row.stop_loss:,.2f}  ({row.risk_pct:.1f}% risk)")
                m2.metric("Target",     f"₹{row.target:,.2f}  ({row.reward_pct:.1f}% upside)")
                m3.metric("Median Return",  f"{row.median_return:+.2f}%")
                m3.metric("Profit Factor",  f"{row.profit_factor:.2f}")
                m3.metric("Max Drawdown",   f"{row.max_drawdown:.1f}%")
                m4.metric("Strategies",     row.active_strategies)
                m4.metric("Backtest Trades",f"{row.avg_trades}")
                m4.metric("Market Regime",  row.get("market_regime","?"))

                # Per-strategy table
                sigs = row.signals or {}
                bts  = row.backtest or {}
                wts  = row.strategy_weights or {}
                rows = []
                for name, val in sigs.items():
                    fired = "🟢 BUY" if val==1 else ("🔴 SELL" if val==-1 else "⬜ None")
                    b = bts.get(name, {})
                    rows.append({
                        "Strategy":    name,
                        "Signal":      fired,
                        "Weight":      f"{wts.get(name,0):.2f}",
                        "Win Rate":    f"{b.get('win_rate',0):.0f}%",
                        "Avg Return":  f"{b.get('avg_return',0):+.2f}%",
                        "Median Ret":  f"{b.get('median_return',0):+.2f}%",
                        "Prof. Factor":f"{b.get('profit_factor',0):.2f}",
                        "SL Exits":    b.get("sl_exits",0),
                        "Tgt Exits":   b.get("target_exits",0),
                        "Trades":      b.get("trades",0),
                    })
                if lsw:
                    st.warning("⚠️ Backtest sample size is small (< 5 trades). Treat win rate with caution.")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Chart
                hist_df = price_history(row.ticker, 90)
                if not hist_df.empty:
                    st.plotly_chart(candlestick(hist_df, row.ticker,
                                                sl=row.stop_loss, tgt=row.target),
                                    use_container_width=True)

                # Paper trade form
                st.markdown("**📝 Paper Trade**")
                pb1,pb2,pb3,pb4 = st.columns(4)
                qty   = pb1.number_input("Qty",   min_value=1, value=10, key=f"qty_{row.ticker}")
                price = pb2.number_input("Price (₹)", value=float(row.price), key=f"px_{row.ticker}")
                sl_in = pb3.number_input("Stop Loss (₹)", value=float(row.stop_loss), key=f"sl_{row.ticker}")
                tg_in = pb4.number_input("Target (₹)",   value=float(row.target),    key=f"tg_{row.ticker}")
                notes = st.text_input("Notes", value=row.active_strategies, key=f"nt_{row.ticker}")
                rec_id = str(row.get("id",""))
                if st.button(f"🟢 Paper Buy {row.ticker}", key=f"buy_{row.ticker}"):
                    paper_buy(row.ticker, price, qty, sl_in, tg_in, notes, rec_id)
                    st.success(f"✅ Paper bought {qty} × {row.ticker} @ ₹{price:.2f} | SL ₹{sl_in:.2f} | Target ₹{tg_in:.2f}")
                    st.balloons()

    if not buys.empty:
        st.subheader("🟢 Buy Recommendations")
        render_signals(buys, "BUY")

    if not sells.empty:
        st.divider()
        st.subheader("🔴 Sell / Exit Recommendations")
        render_signals(sells, "SELL")

# ─────────────────────────────────────────────
#  PAGE: PAPER PORTFOLIO
# ─────────────────────────────────────────────

elif page == "💼 My Paper Portfolio":
    st.title("💼 My Paper Portfolio")

    port = load_portfolio()
    open_pos   = port[port.status=="OPEN"].copy()   if not port.empty else pd.DataFrame()
    closed_pos = port[port.status=="CLOSED"].copy() if not port.empty else pd.DataFrame()

    # Portfolio summary metrics
    total_inv = total_cur = 0
    if not open_pos.empty:
        for _, row in open_pos.iterrows():
            lp = live_price(row.ticker)
            total_inv += row.buy_price * row.quantity
            total_cur += (lp if lp>0 else row.buy_price) * row.quantity

    open_pnl   = total_cur - total_inv
    closed_pnl = float(closed_pos.pnl_inr.sum()) if not closed_pos.empty else 0
    total_pnl  = open_pnl + closed_pnl
    win_ct     = len(closed_pos[closed_pos.pnl_pct>0]) if not closed_pos.empty else 0
    win_rate   = win_ct/len(closed_pos)*100 if not closed_pos.empty else 0

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Open Positions", len(open_pos))
    c2.metric("Open P&L",  f"₹{open_pnl:+,.0f}",
               delta=f"{open_pnl/total_inv*100:+.1f}%" if total_inv else "–")
    c3.metric("Realised P&L", f"₹{closed_pnl:+,.0f}")
    c4.metric("Total P&L",    f"₹{total_pnl:+,.0f}")
    c5.metric("Win Rate (closed)", f"{win_rate:.0f}%", delta=f"{win_ct}/{len(closed_pos)} trades")

    st.divider()

    # ── Open positions
    st.subheader("📂 Open Positions")
    if alerts:
        st.warning(f"⚠️ {len(alerts)} position(s) need attention — see alerts above!")
    if open_pos.empty:
        st.info("No open positions. Go to 'Today's Signals' to paper buy a stock.")
    else:
        for _, row in open_pos.iterrows():
            lp = live_price(row.ticker)
            if lp > 0:
                pnl_pct = (lp - row.buy_price)/row.buy_price*100
                pnl_inr = (lp - row.buy_price)*row.quantity
            else:
                lp = row.buy_price; pnl_pct = pnl_inr = 0

            sl      = float(row.get("entry_stop_loss") or 0)
            tgt     = float(row.get("entry_target") or 0)
            sl_hit  = sl > 0 and lp <= sl
            tgt_hit = tgt > 0 and lp >= tgt

            status_icon = "🚨" if sl_hit else ("🎯" if tgt_hit else ("🟢" if pnl_pct>=0 else "🔴"))

            with st.expander(
                f"{status_icon} **{row.ticker}**  |  {row.quantity} shares  |  "
                f"Buy ₹{row.buy_price:,.2f}  →  Live ₹{lp:,.2f}  |  "
                f"P&L {pnl_pct:+.2f}%  (₹{pnl_inr:+,.0f})",
                expanded=sl_hit or tgt_hit
            ):
                if sl_hit:
                    st.error(f"🚨 Stop-loss breached! Live ₹{lp:,.2f} ≤ SL ₹{sl:,.2f}")
                if tgt_hit:
                    st.success(f"🎯 Target hit! Live ₹{lp:,.2f} ≥ Target ₹{tgt:,.2f}")

                col1,col2,col3 = st.columns(3)
                col1.metric("Buy Price",  f"₹{row.buy_price:,.2f}")
                col1.metric("Buy Date",   str(row.buy_date))
                col1.metric("Invested",   f"₹{row.buy_price*row.quantity:,.0f}")
                col2.metric("Live Price", f"₹{lp:,.2f}")
                col2.metric("Stop Loss",  f"₹{sl:,.2f}" if sl else "Not set")
                col2.metric("Target",     f"₹{tgt:,.2f}" if tgt else "Not set")
                col3.metric("P&L %", f"{pnl_pct:+.2f}%")
                col3.metric("P&L ₹", f"₹{pnl_inr:+,.0f}")
                if sl > 0:
                    sl_dist = (lp - sl)/lp*100
                    col3.metric("Distance to SL", f"{sl_dist:.1f}%",
                                delta_color="inverse" if sl_dist < 2 else "normal")

                if row.notes:
                    st.caption(f"Signal: {row.notes}")

                hist_df = price_history(row.ticker, 60)
                if not hist_df.empty:
                    st.plotly_chart(
                        candlestick(hist_df, row.ticker,
                                    buy_price=row.buy_price, sl=sl or None, tgt=tgt or None),
                        use_container_width=True
                    )

                s1, s2, s3 = st.columns(3)
                sell_px    = s1.number_input("Sell Price (₹)", value=float(lp), key=f"spx_{row.id}")
                exit_reason = s2.selectbox("Exit Reason", ["manual","sl_hit","target_hit","other"], key=f"er_{row.id}")
                if s3.button(f"🔴 Paper Sell {row.ticker}", key=f"sell_{row.id}"):
                    paper_sell(row.id, sell_px, row.buy_price, row.quantity, exit_reason)
                    st.success(f"✅ Closed {row.ticker} @ ₹{sell_px:.2f}")
                    st.rerun()

    st.divider()

    # ── Closed trades
    st.subheader("✅ Closed Trades")
    if closed_pos.empty:
        st.info("No closed trades yet.")
    else:
        # Equity curve
        closed_sorted = closed_pos.sort_values("sell_date").copy()
        closed_sorted["cum_pnl"] = closed_sorted.pnl_inr.cumsum()
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=closed_sorted.sell_date, y=closed_sorted.cum_pnl,
            fill="tozeroy", line=dict(color="#26a69a"), name="Cumulative P&L"
        ))
        fig_eq.add_hline(y=0, line_color="#ef5350", line_dash="dash")
        fig_eq.update_layout(height=260, margin=dict(t=10,b=10,l=0,r=0),
                               title="Cumulative P&L (₹)",
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_eq, use_container_width=True)

        disp = closed_pos[["ticker","buy_date","sell_date","buy_price",
                             "sell_price","quantity","pnl_pct","pnl_inr",
                             "exit_reason"]].copy()
        disp.columns = ["Ticker","Buy Date","Sell Date","Buy ₹","Sell ₹",
                         "Qty","P&L %","P&L ₹","Exit Reason"]
        st.dataframe(
            disp.style
                .applymap(lambda v: "color:#26a69a" if isinstance(v,(int,float)) and v>0
                          else ("color:#ef5350" if isinstance(v,(int,float)) and v<0 else ""),
                          subset=["P&L %","P&L ₹"])
                .format({"P&L %":"{:+.2f}%", "P&L ₹":"₹{:+,.0f}",
                          "Buy ₹":"₹{:.2f}", "Sell ₹":"₹{:.2f}"}),
            use_container_width=True, hide_index=True
        )

# ─────────────────────────────────────────────
#  PAGE: SIGNAL HISTORY
# ─────────────────────────────────────────────

elif page == "📅 Signal History":
    st.title("📅 Signal History")
    days_back = st.slider("Past N days", 7, 60, 14)
    hist = load_recs(days_back=days_back)

    if hist.empty:
        st.info("No history yet."); st.stop()

    act_f = st.selectbox("Action filter", ["All","BUY","SELL"])
    if act_f != "All":
        hist = hist[hist.action==act_f]

    # Daily signal count chart
    daily = hist.groupby(["date","action"]).size().reset_index(name="count")
    fig = px.bar(daily, x="date", y="count", color="action",
                  color_discrete_map={"BUY":"#26a69a","SELL":"#ef5350"},
                  title="Daily signal count")
    fig.update_layout(height=250, margin=dict(t=30,b=10,l=0,r=0),
                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    # Composite score over time
    fig2 = px.scatter(hist, x="date", y="composite_score", color="action",
                       color_discrete_map={"BUY":"#26a69a","SELL":"#ef5350"},
                       hover_data=["ticker","score_label"],
                       title="Composite score distribution")
    fig2.update_layout(height=250, margin=dict(t=30,b=10,l=0,r=0),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

    show_cols = ["date","ticker","action","composite_score","score_label",
                  "win_rate","avg_return","median_return","profit_factor",
                  "rsi","active_strategies","market_regime"]
    show_cols = [c for c in show_cols if c in hist.columns]
    st.dataframe(hist[show_cols], use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
#  PAGE: STRATEGY STATS
# ─────────────────────────────────────────────

elif page == "📈 Strategy Stats":
    st.title("📈 Strategy Performance Stats")

    res = sb.table("recommendations").select(
        "ticker,backtest,action,win_rate,avg_return,composite_score,profit_factor,max_drawdown,median_return"
    ).execute()
    all_recs = pd.DataFrame(res.data) if res.data else pd.DataFrame()

    if all_recs.empty:
        st.info("No data yet. Run the agent first."); st.stop()

    # Per-strategy aggregation
    rows = []
    for _, r in all_recs.iterrows():
        try:
            bt = json.loads(r.backtest) if isinstance(r.backtest,str) else r.backtest
            for name, stats in bt.items():
                rows.append({
                    "Strategy":     name,
                    "Win Rate":     stats.get("win_rate",0),
                    "Avg Return":   stats.get("avg_return",0),
                    "Median Return":stats.get("median_return",0),
                    "Profit Factor":stats.get("profit_factor",0),
                    "Max Drawdown": stats.get("max_drawdown",0),
                    "SL Exits":     stats.get("sl_exits",0),
                    "Target Exits": stats.get("target_exits",0),
                    "Trades":       stats.get("trades",0),
                })
        except: continue

    if not rows:
        st.info("No backtest data yet."); st.stop()

    bt_df = pd.DataFrame(rows)
    agg = bt_df.groupby("Strategy").agg(
        Win_Rate       = ("Win Rate",      "mean"),
        Avg_Return     = ("Avg Return",    "mean"),
        Median_Return  = ("Median Return", "mean"),
        Profit_Factor  = ("Profit Factor", "mean"),
        Max_Drawdown   = ("Max Drawdown",  "mean"),
        Total_Trades   = ("Trades",        "sum"),
        SL_Exits       = ("SL Exits",      "sum"),
        Target_Exits   = ("Target Exits",  "sum"),
    ).reset_index()

    # Warn about low-sample strategies
    for _, row in agg.iterrows():
        if row.Total_Trades < 20:
            st.warning(f"⚠️ **{row.Strategy}**: only {int(row.Total_Trades)} total backtest trades — interpret stats cautiously.")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(agg, x="Strategy", y="Win_Rate",
                      color="Win_Rate", color_continuous_scale="teal",
                      text="Win_Rate", title="Win Rate % by Strategy")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=300, showlegend=False,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.bar(agg, x="Strategy", y="Profit_Factor",
                       color="Profit_Factor", color_continuous_scale="RdYlGn",
                       text="Profit_Factor", title="Profit Factor by Strategy (>1 = profitable)")
        fig2.add_hline(y=1.0, line_dash="dash", line_color="#ef5350")
        fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig2.update_layout(height=300, showlegend=False,
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    # Exit type breakdown
    exit_data = []
    for _, row in agg.iterrows():
        total_exits = row.Total_Trades
        if total_exits > 0:
            exit_data.append({"Strategy": row.Strategy, "Type": "SL",     "Count": row.SL_Exits})
            exit_data.append({"Strategy": row.Strategy, "Type": "Target",  "Count": row.Target_Exits})
            exit_data.append({"Strategy": row.Strategy, "Type": "Timeout",
                               "Count": max(0, total_exits - row.SL_Exits - row.Target_Exits)})
    if exit_data:
        fig3 = px.bar(pd.DataFrame(exit_data), x="Strategy", y="Count", color="Type",
                       color_discrete_map={"SL":"#ef5350","Target":"#26a69a","Timeout":"#ffa726"},
                       barmode="stack", title="Exit Reason Breakdown")
        fig3.update_layout(height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

    # Summary table
    st.dataframe(agg.style.format({
        "Win_Rate":"{:.1f}%","Avg_Return":"{:+.2f}%","Median_Return":"{:+.2f}%",
        "Profit_Factor":"{:.2f}","Max_Drawdown":"{:.1f}%",
    }), use_container_width=True, hide_index=True)

    # Score vs realised P&L (only if there are closed paper trades linked to recs)
    closed = load_portfolio()
    if not closed.empty:
        closed = closed[(closed.status=="CLOSED") & closed.recommendation_id.notna()].copy()
    if not closed.empty:
        st.divider()
        st.subheader("🔗 Composite Score vs Realised P&L")
        st.caption("Each dot = a paper trade you closed, linked back to its original recommendation score.")
        rec_ids = closed.recommendation_id.astype(str).tolist()
        rec_res = sb.table("recommendations").select("id,composite_score,score_label").in_("id", rec_ids).execute()
        if rec_res.data:
            rec_df = pd.DataFrame(rec_res.data)
            rec_df["id"] = rec_df["id"].astype(str)
            closed["recommendation_id"] = closed["recommendation_id"].astype(str)
            merged = closed.merge(rec_df, left_on="recommendation_id", right_on="id", how="inner")
            if not merged.empty:
                fig_sc = px.scatter(merged, x="composite_score", y="pnl_pct",
                                     color="pnl_pct", color_continuous_scale="RdYlGn",
                                     hover_data=["ticker","score_label","exit_reason"],
                                     title="Composite Score vs Realised P&L %",
                                     labels={"composite_score":"Composite Score","pnl_pct":"P&L %"})
                fig_sc.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_sc.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_sc, use_container_width=True)

    # Your paper trading stats
    port_all2 = load_portfolio()
    if not port_all2.empty:
        cl2 = port_all2[port_all2.status=="CLOSED"]
        if not cl2.empty and "pnl_pct" in cl2.columns:
            st.divider()
            st.subheader("📊 Your Paper Trading Stats")
            total = len(cl2)
            wins  = len(cl2[cl2.pnl_pct>0])
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Total Trades",   total)
            c2.metric("Win Rate",       f"{wins/total*100:.0f}%")
            c3.metric("Avg P&L",        f"{cl2.pnl_pct.mean():+.2f}%")
            c4.metric("Median P&L",     f"{cl2.pnl_pct.median():+.2f}%")
            c5.metric("Best Trade",     f"{cl2.pnl_pct.max():+.2f}%")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ Paper trading & educational purposes only. Not financial advice. "
    "Always do your own research. Past backtest results do not guarantee future performance."
)
