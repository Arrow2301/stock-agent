"""
============================================================
  Indian Stock Agent — Dashboard v3
  New pages:
  ─ 🤖 Optimizer: champion vs challenger, param history,
       promotion controls, trial leaderboard, param diff
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
    if st.session_state.get("authenticated"): return True
    st.title("🇮🇳 Indian Stock Agent")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if pwd == st.secrets.get("DASHBOARD_PASSWORD", "stockagent123"):
            st.session_state.authenticated = True; st.rerun()
        else:
            st.error("Incorrect password")
    return False

if not check_password():
    st.stop()

# ─────────────────────────────────────────────
#  DATA HELPERS
# ─────────────────────────────────────────────

def score_color(s):
    if s>=80: return "#00c853"
    if s>=65: return "#69f0ae"
    if s>=50: return "#ffeb3b"
    if s>=35: return "#ffa726"
    return "#ef5350"

def score_badge(s, label):
    col = score_color(s)
    return (f'<span style="background:{col};color:#111;padding:3px 10px;'
            f'border-radius:12px;font-weight:700;font-size:1.1em">{s:.0f} — {label}</span>')

@st.cache_data(ttl=300)
def load_recs(target_date=None, days_back=None):
    q = sb.table("recommendations").select("*").order("composite_score", desc=True)
    if target_date:   q = q.eq("date", target_date)
    elif days_back:   q = q.gte("date", (date.today()-timedelta(days=days_back)).isoformat())
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
        return float(df["Close"].iloc[-1]) if not df.empty else 0.0
    except: return 0.0

@st.cache_data(ttl=600)
def price_history(ticker, days=90):
    try:
        df = yf.download(ticker+".NS",
                         start=date.today()-timedelta(days=days),
                         end=date.today(), progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        df.columns = [c[0] if isinstance(c,tuple) else c for c in df.columns]
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except: return pd.DataFrame()

@st.cache_data(ttl=120)
def get_meta():
    res = sb.table("agent_meta").select("*").eq("id",1).execute()
    return res.data[0] if res.data else {}

@st.cache_data(ttl=120)
def load_agent_params(status=None):
    q = sb.table("agent_params").select("*").order("version", desc=True)
    if status: q = q.eq("status", status)
    res = q.execute()
    return pd.DataFrame(res.data) if res.data else pd.DataFrame()

@st.cache_data(ttl=120)
def load_opt_runs():
    res = sb.table("optimization_runs").select("*").order("run_date", desc=True).execute()
    return pd.DataFrame(res.data) if res.data else pd.DataFrame()

def paper_buy(ticker, price, qty, sl, tgt, notes, rec_id=None):
    sb.table("paper_portfolio").insert({
        "ticker": ticker, "buy_date": date.today().isoformat(),
        "buy_price": price, "quantity": qty,
        "entry_stop_loss": sl, "entry_target": tgt,
        "status": "OPEN", "notes": notes, "recommendation_id": rec_id,
    }).execute(); st.cache_data.clear()

def paper_sell(trade_id, sell_price, buy_price, qty, exit_reason="manual"):
    pnl_pct=round((sell_price-buy_price)/buy_price*100,2)
    pnl_inr=round((sell_price-buy_price)*qty,2)
    sb.table("paper_portfolio").update({
        "sell_date": date.today().isoformat(), "sell_price": sell_price,
        "status": "CLOSED", "pnl_pct": pnl_pct, "pnl_inr": pnl_inr,
        "exit_reason": exit_reason,
    }).eq("id", trade_id).execute(); st.cache_data.clear()

def promote_param(version, new_status):
    if new_status == "champion":
        sb.table("agent_params").update({"status":"retired"}).eq("status","champion").execute()
    if new_status in ("challenger", "retired"):
        pass
    sb.table("agent_params").update({
        "status": new_status,
        "promoted_at": date.today().isoformat(),
    }).eq("version", version).execute()
    st.cache_data.clear()

def candlestick(df, ticker, buy_price=None, sl=None, tgt=None):
    df=df.copy()
    df["ema9"]=df.Close.ewm(span=9,adjust=False).mean()
    df["ema21"]=df.Close.ewm(span=21,adjust=False).mean()
    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,open=df.Open,high=df.High,low=df.Low,close=df.Close,
                                  name=ticker,increasing_line_color="#26a69a",decreasing_line_color="#ef5350"))
    fig.add_trace(go.Scatter(x=df.index,y=df.ema9, name="EMA 9", line=dict(color="#ffb300",width=1.2)))
    fig.add_trace(go.Scatter(x=df.index,y=df.ema21,name="EMA 21",line=dict(color="#ab47bc",width=1.2)))
    if buy_price: fig.add_hline(y=buy_price,line_color="#00e676",line_dash="dash",annotation_text=f"Buy ₹{buy_price:.2f}")
    if sl:  fig.add_hline(y=sl, line_color="#ef5350",line_dash="dot",annotation_text=f"SL ₹{sl:.2f}")
    if tgt: fig.add_hline(y=tgt,line_color="#69f0ae",line_dash="dot",annotation_text=f"Tgt ₹{tgt:.2f}")
    fig.update_layout(height=360,xaxis_rangeslider_visible=False,
                       margin=dict(t=10,b=10,l=0,r=0),
                       plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                       legend=dict(orientation="h",y=1.02))
    return fig

def check_exit_alerts(port):
    alerts=[]
    if port.empty: return alerts
    for _,row in port[port.status=="OPEN"].iterrows():
        lp=live_price(row.ticker)
        if lp<=0: continue
        sl=row.get("entry_stop_loss"); tgt=row.get("entry_target")
        if sl and lp<=float(sl):
            alerts.append({"ticker":row.ticker,"type":"SL_HIT","lp":lp,"level":sl,
                            "pnl":round((lp-row.buy_price)/row.buy_price*100,2),"id":row.id,
                            "buy_price":row.buy_price,"qty":row.quantity})
        elif tgt and lp>=float(tgt):
            alerts.append({"ticker":row.ticker,"type":"TARGET_HIT","lp":lp,"level":tgt,
                            "pnl":round((lp-row.buy_price)/row.buy_price*100,2),"id":row.id,
                            "buy_price":row.buy_price,"qty":row.quantity})
    return alerts

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("🇮🇳 Stock Agent")
    meta=get_meta()
    if meta:
        st.success(f"Last run: **{meta.get('last_run','N/A')}**")
        regime=meta.get("market_regime","?")
        emoji={"BULLISH":"🟢","BEARISH":"🔴","NEUTRAL":"🟡"}.get(regime,"⬜")
        st.caption(f"Market: {emoji} **{regime}**  |  Signals: {meta.get('total_signals',0)}")
        st.caption(f"Params: {meta.get('active_param_version','defaults')}")
    else:
        st.warning("Agent hasn't run yet")
    st.divider()
    page = st.radio("Navigate", [
        "📊 Today's Signals",
        "💼 My Paper Portfolio",
        "📅 Signal History",
        "📈 Strategy Stats",
        "🤖 Optimizer",
    ])
    st.divider()
    st.caption("NSE: 9:15 AM – 3:30 PM IST")
    st.caption("Agent: 7:00 AM IST, Mon–Fri")
    st.caption("Optimizer: Sunday 11 PM IST")
    if st.button("🔄 Refresh"): st.cache_data.clear(); st.rerun()

# ─────────────────────────────────────────────
#  EXIT ALERT BANNER
# ─────────────────────────────────────────────

port_all = load_portfolio()
alerts   = check_exit_alerts(port_all)
for a in alerts:
    if a["type"]=="SL_HIT":
        st.error(f"🚨 **STOP-LOSS HIT — {a['ticker']}** | Live: ₹{a['lp']:,.2f} ≤ SL: ₹{float(a['level']):,.2f} | P&L: {a['pnl']:+.2f}%")
    else:
        st.success(f"🎯 **TARGET HIT — {a['ticker']}** | Live: ₹{a['lp']:,.2f} ≥ Target: ₹{float(a['level']):,.2f} | P&L: {a['pnl']:+.2f}%")

# ─────────────────────────────────────────────
#  PAGE: TODAY'S SIGNALS
# ─────────────────────────────────────────────

if page == "📊 Today's Signals":
    today_str = date.today().isoformat()
    st.title("📊 Today's Signals")
    st.caption(f"{datetime.today().strftime('%A, %d %B %Y')}  |  Sorted by Composite Score ↓")

    recs = load_recs(target_date=today_str)
    if recs.empty:
        st.info("No signals today. Agent runs at 7:00 AM IST — trigger manually from GitHub Actions if needed.")
        st.stop()

    buys  = recs[recs.action=="BUY"].copy()
    sells = recs[recs.action=="SELL"].copy()

    with st.expander("🔧 Filters", expanded=False):
        fc1,fc2,fc3 = st.columns(3)
        min_cs = fc1.slider("Min Composite Score", 0, 100, 30)
        min_wr = fc2.slider("Min Win Rate %", 0, 100, 40)
        min_pf = fc3.slider("Min Profit Factor", 0.0, 5.0, 0.8, 0.1)
        buys  = buys[ (buys.composite_score>=min_cs) & (buys.win_rate>=min_wr)  & (buys.profit_factor>=min_pf)]
        sells = sells[(sells.composite_score>=min_cs) & (sells.win_rate>=min_wr) & (sells.profit_factor>=min_pf)]

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🟢 Buy",  len(buys))
    c2.metric("🔴 Sell", len(sells))
    c3.metric("🏆 Top Score", f"{recs.composite_score.max():.0f}/100")
    c4.metric("Market",  meta.get("market_regime","?"))
    c5.metric("Params",  meta.get("active_param_version","defaults"))

    st.divider()

    def render_rec(df_recs, action):
        if df_recs.empty:
            st.info(f"No {action} signals match filters."); return
        for _,row in df_recs.iterrows():
            cs=row.composite_score; label=row.score_label
            lsw=row.get("low_sample_warning",False)
            warn=" ⚠️ Low sample" if lsw else ""
            with st.expander(
                f"**{row.ticker}**  |  Score {cs:.0f}/100 ({label}){warn}  |  "
                f"WR {row.win_rate:.0f}%  |  PF {row.profit_factor:.2f}  |  "
                f"Avg Ret {row.avg_return:+.2f}%",
                expanded=(cs>=65)
            ):
                st.markdown(score_badge(cs,label), unsafe_allow_html=True)
                st.markdown("")

                bd=row.score_breakdown or {}
                if bd:
                    bd_df=pd.DataFrame([
                        {"Component":"Strategy (40)","Score":bd.get("strategy",0),"Max":40},
                        {"Component":"RSI (20)",     "Score":bd.get("rsi",0),     "Max":20},
                        {"Component":"Volume (15)",  "Score":bd.get("volume",0),  "Max":15},
                        {"Component":"R:R (15)",     "Score":bd.get("rr",0),      "Max":15},
                        {"Component":"Regime (10)",  "Score":bd.get("regime",0),  "Max":10},
                    ])
                    bd_df["%"]=(bd_df.Score/bd_df.Max*100).clip(0,100)
                    fig_bd=px.bar(bd_df,x="Score",y="Component",orientation="h",
                                   color="%",color_continuous_scale="RdYlGn",range_color=[0,100],text="Score")
                    fig_bd.update_layout(height=200,margin=dict(t=0,b=0,l=0,r=0),
                                          showlegend=False,coloraxis_showscale=False,
                                          plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
                    fig_bd.update_traces(texttemplate="%{text:.1f}",textposition="inside")
                    st.plotly_chart(fig_bd, use_container_width=True)

                m1,m2,m3,m4=st.columns(4)
                m1.metric("Price",    f"₹{row.price:,.2f}")
                m1.metric("1D",       f"{row.change_1d:+.2f}%")
                m1.metric("5D",       f"{row.change_5d:+.2f}%")
                m2.metric("RSI",      f"{row.rsi:.1f}")
                m2.metric("Stop Loss",f"₹{row.stop_loss:,.2f}  ({row.risk_pct:.1f}% risk)")
                m2.metric("Target",   f"₹{row.target:,.2f}  ({row.reward_pct:.1f}% upside)")
                m3.metric("Median Ret",   f"{row.median_return:+.2f}%")
                m3.metric("Profit Factor",f"{row.profit_factor:.2f}")
                m3.metric("Max Drawdown", f"{row.max_drawdown:.1f}%")
                m4.metric("Strategies",   row.active_strategies)
                m4.metric("BT Trades",    f"{row.avg_trades}")
                m4.metric("Regime",       row.get("market_regime","?"))

                sigs=row.signals or {}; bts=row.backtest or {}; wts=row.strategy_weights or {}
                rows_t=[]
                for name,val in sigs.items():
                    fired="🟢 BUY" if val==1 else("🔴 SELL" if val==-1 else "⬜ None")
                    b=bts.get(name,{})
                    rows_t.append({"Strategy":name,"Signal":fired,
                                    "Weight":f"{wts.get(name,0):.2f}",
                                    "Win Rate":f"{b.get('win_rate',0):.0f}%",
                                    "Avg Ret":f"{b.get('avg_return',0):+.2f}%",
                                    "Median":f"{b.get('median_return',0):+.2f}%",
                                    "PF":f"{b.get('profit_factor',0):.2f}",
                                    "SL Exits":b.get("sl_exits",0),
                                    "Tgt Exits":b.get("target_exits",0),
                                    "Trades":b.get("trades",0)})
                if lsw: st.warning("⚠️ Fewer than 5 backtest trades — treat win rate with caution.")
                st.dataframe(pd.DataFrame(rows_t),use_container_width=True,hide_index=True)

                hist_df=price_history(row.ticker,90)
                if not hist_df.empty:
                    st.plotly_chart(candlestick(hist_df,row.ticker,sl=row.stop_loss,tgt=row.target),
                                    use_container_width=True)

                st.markdown("**📝 Paper Trade**")
                pb1,pb2,pb3,pb4=st.columns(4)
                qty  =pb1.number_input("Qty",min_value=1,value=10,key=f"qty_{row.ticker}")
                price=pb2.number_input("Price (₹)",value=float(row.price),key=f"px_{row.ticker}")
                sl_in=pb3.number_input("Stop Loss (₹)",value=float(row.stop_loss),key=f"sl_{row.ticker}")
                tg_in=pb4.number_input("Target (₹)",value=float(row.target),key=f"tg_{row.ticker}")
                notes=st.text_input("Notes",value=row.active_strategies,key=f"nt_{row.ticker}")
                if st.button(f"🟢 Paper Buy {row.ticker}",key=f"buy_{row.ticker}"):
                    paper_buy(row.ticker,price,qty,sl_in,tg_in,notes,str(row.get("id","")))
                    st.success(f"✅ Paper bought {qty} × {row.ticker} @ ₹{price:.2f}")
                    st.balloons()

    if not buys.empty:
        st.subheader("🟢 Buy Recommendations")
        render_rec(buys,"BUY")
    if not sells.empty:
        st.divider()
        st.subheader("🔴 Sell / Exit Recommendations")
        render_rec(sells,"SELL")

# ─────────────────────────────────────────────
#  PAGE: PAPER PORTFOLIO
# ─────────────────────────────────────────────

elif page == "💼 My Paper Portfolio":
    st.title("💼 My Paper Portfolio")
    port=load_portfolio()
    open_pos  =port[port.status=="OPEN"].copy()   if not port.empty else pd.DataFrame()
    closed_pos=port[port.status=="CLOSED"].copy() if not port.empty else pd.DataFrame()

    total_inv=total_cur=0
    if not open_pos.empty:
        for _,row in open_pos.iterrows():
            lp=live_price(row.ticker)
            total_inv+=row.buy_price*row.quantity
            total_cur+=(lp if lp>0 else row.buy_price)*row.quantity

    open_pnl  =total_cur-total_inv
    closed_pnl=float(closed_pos.pnl_inr.sum()) if not closed_pos.empty else 0
    win_ct    =len(closed_pos[closed_pos.pnl_pct>0]) if not closed_pos.empty else 0
    win_rate  =win_ct/len(closed_pos)*100 if not closed_pos.empty else 0

    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Open Positions", len(open_pos))
    c2.metric("Open P&L",  f"₹{open_pnl:+,.0f}",
               delta=f"{open_pnl/total_inv*100:+.1f}%" if total_inv else "–")
    c3.metric("Realised P&L",   f"₹{closed_pnl:+,.0f}")
    c4.metric("Total P&L",      f"₹{open_pnl+closed_pnl:+,.0f}")
    c5.metric("Win Rate",       f"{win_rate:.0f}%",delta=f"{win_ct}/{len(closed_pos)}")

    st.divider()
    st.subheader("📂 Open Positions")
    if alerts: st.warning(f"⚠️ {len(alerts)} position(s) need attention — see alerts above!")
    if open_pos.empty:
        st.info("No open positions.")
    else:
        for _,row in open_pos.iterrows():
            lp=live_price(row.ticker)
            if lp>0: pnl_pct=(lp-row.buy_price)/row.buy_price*100; pnl_inr=(lp-row.buy_price)*row.quantity
            else:    lp=row.buy_price; pnl_pct=pnl_inr=0
            sl=float(row.get("entry_stop_loss") or 0)
            tgt=float(row.get("entry_target") or 0)
            sl_hit=sl>0 and lp<=sl; tgt_hit=tgt>0 and lp>=tgt
            icon="🚨" if sl_hit else("🎯" if tgt_hit else("🟢" if pnl_pct>=0 else "🔴"))
            with st.expander(
                f"{icon} **{row.ticker}**  |  {row.quantity} shares  |  "
                f"Buy ₹{row.buy_price:,.2f} → Live ₹{lp:,.2f}  |  P&L {pnl_pct:+.2f}%",
                expanded=sl_hit or tgt_hit
            ):
                if sl_hit:  st.error(f"🚨 Stop-loss breached! ₹{lp:,.2f} ≤ SL ₹{sl:,.2f}")
                if tgt_hit: st.success(f"🎯 Target hit! ₹{lp:,.2f} ≥ Target ₹{tgt:,.2f}")
                c1,c2,c3=st.columns(3)
                c1.metric("Buy Price",  f"₹{row.buy_price:,.2f}")
                c1.metric("Buy Date",   str(row.buy_date))
                c2.metric("Live Price", f"₹{lp:,.2f}")
                c2.metric("Stop Loss",  f"₹{sl:,.2f}" if sl else "Not set")
                c2.metric("Target",     f"₹{tgt:,.2f}" if tgt else "Not set")
                c3.metric("P&L %", f"{pnl_pct:+.2f}%")
                c3.metric("P&L ₹", f"₹{pnl_inr:+,.0f}")
                if sl>0:
                    sl_dist=(lp-sl)/lp*100
                    c3.metric("Dist to SL",f"{sl_dist:.1f}%",delta_color="inverse" if sl_dist<2 else "normal")
                hist_df=price_history(row.ticker,60)
                if not hist_df.empty:
                    st.plotly_chart(candlestick(hist_df,row.ticker,
                                                buy_price=row.buy_price,sl=sl or None,tgt=tgt or None),
                                    use_container_width=True)
                s1,s2,s3=st.columns(3)
                sell_px=s1.number_input("Sell Price (₹)",value=float(lp),key=f"spx_{row.id}")
                er=s2.selectbox("Exit Reason",["manual","sl_hit","target_hit","other"],key=f"er_{row.id}")
                if s3.button(f"🔴 Sell {row.ticker}",key=f"sell_{row.id}"):
                    paper_sell(row.id,sell_px,row.buy_price,row.quantity,er)
                    st.success(f"✅ Closed {row.ticker} @ ₹{sell_px:.2f}"); st.rerun()

    st.divider()
    st.subheader("✅ Closed Trades")
    if closed_pos.empty:
        st.info("No closed trades yet.")
    else:
        cs2=closed_pos.sort_values("sell_date").copy()
        cs2["cum_pnl"]=cs2.pnl_inr.cumsum()
        fig_eq=go.Figure()
        fig_eq.add_trace(go.Scatter(x=cs2.sell_date,y=cs2.cum_pnl,
                                     fill="tozeroy",line=dict(color="#26a69a"),name="Cumulative P&L"))
        fig_eq.add_hline(y=0,line_color="#ef5350",line_dash="dash")
        fig_eq.update_layout(height=260,margin=dict(t=10,b=10,l=0,r=0),title="Cumulative P&L (₹)",
                               plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_eq,use_container_width=True)
        disp=closed_pos[["ticker","buy_date","sell_date","buy_price","sell_price",
                           "quantity","pnl_pct","pnl_inr","exit_reason"]].copy()
        disp.columns=["Ticker","Buy Date","Sell Date","Buy ₹","Sell ₹","Qty","P&L %","P&L ₹","Exit"]
        st.dataframe(disp.style
                     .applymap(lambda v:"color:#26a69a" if isinstance(v,(int,float)) and v>0
                                else("color:#ef5350" if isinstance(v,(int,float)) and v<0 else ""),
                                subset=["P&L %","P&L ₹"])
                     .format({"P&L %":"{:+.2f}%","P&L ₹":"₹{:+,.0f}",
                               "Buy ₹":"₹{:.2f}","Sell ₹":"₹{:.2f}"}),
                     use_container_width=True,hide_index=True)

# ─────────────────────────────────────────────
#  PAGE: SIGNAL HISTORY
# ─────────────────────────────────────────────

elif page == "📅 Signal History":
    st.title("📅 Signal History")
    days_back=st.slider("Past N days",7,60,14)
    hist=load_recs(days_back=days_back)
    if hist.empty: st.info("No history yet."); st.stop()

    act_f=st.selectbox("Action filter",["All","BUY","SELL"])
    if act_f!="All": hist=hist[hist.action==act_f]

    daily=hist.groupby(["date","action"]).size().reset_index(name="count")
    fig=px.bar(daily,x="date",y="count",color="action",
                color_discrete_map={"BUY":"#26a69a","SELL":"#ef5350"},title="Daily signal count")
    fig.update_layout(height=250,margin=dict(t=30,b=10,l=0,r=0),
                       plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)

    fig2=px.scatter(hist,x="date",y="composite_score",color="action",
                     color_discrete_map={"BUY":"#26a69a","SELL":"#ef5350"},
                     hover_data=["ticker","score_label"],title="Composite score distribution")
    fig2.update_layout(height=250,margin=dict(t=30,b=10,l=0,r=0),
                        plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2,use_container_width=True)

    show_cols=["date","ticker","action","composite_score","score_label",
                "win_rate","avg_return","median_return","profit_factor",
                "rsi","active_strategies","market_regime","param_version"]
    show_cols=[c for c in show_cols if c in hist.columns]
    st.dataframe(hist[show_cols],use_container_width=True,hide_index=True)

# ─────────────────────────────────────────────
#  PAGE: STRATEGY STATS
# ─────────────────────────────────────────────

elif page == "📈 Strategy Stats":
    st.title("📈 Strategy Performance Stats")
    res=sb.table("recommendations").select(
        "ticker,backtest,action,win_rate,avg_return,composite_score,profit_factor,max_drawdown,median_return"
    ).execute()
    all_recs=pd.DataFrame(res.data) if res.data else pd.DataFrame()
    if all_recs.empty: st.info("No data yet."); st.stop()

    rows=[]
    for _,r in all_recs.iterrows():
        try:
            bt=json.loads(r.backtest) if isinstance(r.backtest,str) else r.backtest
            for name,stats in bt.items():
                rows.append({"Strategy":name,"Win Rate":stats.get("win_rate",0),
                              "Avg Return":stats.get("avg_return",0),
                              "Median Return":stats.get("median_return",0),
                              "Profit Factor":stats.get("profit_factor",0),
                              "Max Drawdown":stats.get("max_drawdown",0),
                              "SL Exits":stats.get("sl_exits",0),
                              "Target Exits":stats.get("target_exits",0),
                              "Trades":stats.get("trades",0)})
        except: continue

    if not rows: st.info("No backtest data yet."); st.stop()
    bt_df=pd.DataFrame(rows)
    agg=bt_df.groupby("Strategy").agg(
        Win_Rate=("Win Rate","mean"),Avg_Return=("Avg Return","mean"),
        Median_Return=("Median Return","mean"),Profit_Factor=("Profit Factor","mean"),
        Max_Drawdown=("Max Drawdown","mean"),Total_Trades=("Trades","sum"),
        SL_Exits=("SL Exits","sum"),Target_Exits=("Target Exits","sum"),
    ).reset_index()

    for _,row in agg.iterrows():
        if row.Total_Trades<20:
            st.warning(f"⚠️ **{row.Strategy}**: only {int(row.Total_Trades)} trades — interpret cautiously.")

    c1,c2=st.columns(2)
    with c1:
        fig=px.bar(agg,x="Strategy",y="Win_Rate",color="Win_Rate",
                    color_continuous_scale="teal",text="Win_Rate",title="Win Rate %")
        fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
        fig.update_layout(height=300,showlegend=False,
                           plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig2=px.bar(agg,x="Strategy",y="Profit_Factor",color="Profit_Factor",
                     color_continuous_scale="RdYlGn",text="Profit_Factor",
                     title="Profit Factor (>1 = profitable)")
        fig2.add_hline(y=1.0,line_dash="dash",line_color="#ef5350")
        fig2.update_traces(texttemplate="%{text:.2f}",textposition="outside")
        fig2.update_layout(height=300,showlegend=False,
                            plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2,use_container_width=True)

    st.dataframe(agg.style.format({"Win_Rate":"{:.1f}%","Avg_Return":"{:+.2f}%",
                                    "Median_Return":"{:+.2f}%","Profit_Factor":"{:.2f}",
                                    "Max_Drawdown":"{:.1f}%"}),
                 use_container_width=True,hide_index=True)

    closed=load_portfolio()
    if not closed.empty:
        closed=closed[(closed.status=="CLOSED") & closed.recommendation_id.notna()].copy()
    if not closed.empty:
        st.divider(); st.subheader("🔗 Composite Score vs Realised P&L")
        rec_ids=closed.recommendation_id.astype(str).tolist()
        rec_res=sb.table("recommendations").select("id,composite_score,score_label").in_("id",rec_ids).execute()
        if rec_res.data:
            rec_df=pd.DataFrame(rec_res.data); rec_df["id"]=rec_df["id"].astype(str)
            closed["recommendation_id"]=closed["recommendation_id"].astype(str)
            merged=closed.merge(rec_df,left_on="recommendation_id",right_on="id",how="inner")
            if not merged.empty:
                fig_sc=px.scatter(merged,x="composite_score",y="pnl_pct",color="pnl_pct",
                                   color_continuous_scale="RdYlGn",
                                   hover_data=["ticker","score_label","exit_reason"],
                                   title="Composite Score vs Realised P&L %")
                fig_sc.add_hline(y=0,line_dash="dash",line_color="gray")
                fig_sc.update_layout(height=350,plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_sc,use_container_width=True)

# ─────────────────────────────────────────────
#  PAGE: OPTIMIZER
# ─────────────────────────────────────────────

elif page == "🤖 Optimizer":
    st.title("🤖 Parameter Optimizer")
    st.caption("View champion/challenger params, compare performance, promote candidates, track optimization history.")

    params_df = load_agent_params()
    opt_runs  = load_opt_runs()

    if params_df.empty:
        st.info("No optimization runs yet. The optimizer runs every Sunday at 11 PM IST, or trigger it manually from GitHub Actions.")
        st.stop()

    champion   = params_df[params_df.status=="champion"].head(1)
    challenger = params_df[params_df.status=="challenger"].head(1)
    candidates = params_df[params_df.status=="candidate"].copy()

    # ── Status banner
    col1,col2,col3 = st.columns(3)
    with col1:
        st.markdown("### 👑 Champion")
        if not champion.empty:
            r=champion.iloc[0]
            st.success(f"**v{r['version']}**  |  Score: {r['objective_score']:.4f}")
            st.caption(f"PF: {r.get('profit_factor',0):.2f}  |  WR: {r.get('win_rate',0):.1f}%  |  "
                       f"Avg Ret: {r.get('avg_return',0):+.2f}%  |  Max DD: {r.get('max_drawdown',0):.1f}%")
            st.caption(f"Promoted: {r.get('promoted_at','?')}")
        else:
            st.warning("No champion yet")
    with col2:
        st.markdown("### ⚔️ Challenger")
        if not challenger.empty:
            r=challenger.iloc[0]
            champ_score=float(champion.iloc[0]["objective_score"]) if not champion.empty else 0
            delta=r["objective_score"]-champ_score
            color="🟢" if delta>0 else "🔴"
            st.info(f"**v{r['version']}**  |  Score: {r['objective_score']:.4f}  {color} {delta:+.4f} vs champion")
            st.caption(f"PF: {r.get('profit_factor',0):.2f}  |  WR: {r.get('win_rate',0):.1f}%  |  "
                       f"Avg Ret: {r.get('avg_return',0):+.2f}%  |  Max DD: {r.get('max_drawdown',0):.1f}%")
            st.caption(f"Promoted: {r.get('promoted_at','?')}")
        else:
            st.info("No challenger yet")
    with col3:
        st.markdown("### 📋 Candidates")
        st.metric("Available", len(candidates))
        if not opt_runs.empty:
            st.caption(f"Last optimization: {opt_runs.iloc[0]['run_date']}")
            st.caption(f"Trials: {int(opt_runs.iloc[0].get('n_trials',0))}  |  Valid: {int(opt_runs.iloc[0].get('n_valid_trials',0))}")

    st.divider()

    # ── Champion vs Challenger param comparison
    if not champion.empty and not challenger.empty:
        st.subheader("📊 Champion vs Challenger — Parameter Diff")
        champ_p  = json.loads(champion.iloc[0]["params_json"]) if isinstance(champion.iloc[0]["params_json"],str) else champion.iloc[0]["params_json"]
        chall_p  = json.loads(challenger.iloc[0]["params_json"]) if isinstance(challenger.iloc[0]["params_json"],str) else challenger.iloc[0]["params_json"]
        all_keys = sorted(set(list(champ_p.keys()) + list(chall_p.keys())))
        diff_rows=[]
        for k in all_keys:
            cv=champ_p.get(k,"–"); chv=chall_p.get(k,"–")
            changed=cv!=chv
            diff_rows.append({"Parameter":k,
                               f"👑 Champion v{champion.iloc[0]['version']}":cv,
                               f"⚔️ Challenger v{challenger.iloc[0]['version']}":chv,
                               "Changed":"✅" if changed else ""})
        diff_df=pd.DataFrame(diff_rows)
        changed_only=st.checkbox("Show changed params only",value=True)
        if changed_only: diff_df=diff_df[diff_df.Changed=="✅"]
        st.dataframe(diff_df,use_container_width=True,hide_index=True)

        # Side-by-side metric comparison
        st.subheader("📈 Walk-Forward Metrics Comparison")
        metrics_to_compare=[("Objective Score","objective_score"),
                             ("Profit Factor","profit_factor"),
                             ("Win Rate %","win_rate"),
                             ("Avg Return %","avg_return"),
                             ("Max Drawdown %","max_drawdown")]
        mc1,mc2,mc3,mc4,mc5=st.columns(5)
        cols=[mc1,mc2,mc3,mc4,mc5]
        for col,(label,field) in zip(cols,metrics_to_compare):
            cv=float(champion.iloc[0].get(field,0) or 0)
            chv=float(challenger.iloc[0].get(field,0) or 0)
            delta=chv-cv
            better=(delta>0 and field!="max_drawdown") or (delta<0 and field=="max_drawdown")
            col.metric(label, f"{chv:.3f}", delta=f"{delta:+.3f}",
                        delta_color="normal" if better else "inverse")

        # Promote / demote controls
        st.divider()
        st.subheader("🎛️ Promotion Controls")
        st.warning("⚠️ Promoting the challenger replaces the current champion for all live signals. "
                   "Only do this after monitoring paper-trade performance for 2–4 weeks.")
        pr1,pr2=st.columns(2)
        with pr1:
            if st.button(f"👑 Promote Challenger v{challenger.iloc[0]['version']} → Champion",
                          type="primary"):
                promote_param(int(challenger.iloc[0]["version"]), "champion")
                st.success(f"✅ Challenger v{challenger.iloc[0]['version']} is now Champion!")
                st.rerun()
        with pr2:
            if st.button(f"🗑️ Retire Challenger v{challenger.iloc[0]['version']}"):
                promote_param(int(challenger.iloc[0]["version"]), "retired")
                st.info(f"Challenger v{challenger.iloc[0]['version']} retired.")
                st.rerun()

    st.divider()

    # ── All candidates table
    st.subheader("📋 Top Candidates from Latest Run")
    if candidates.empty:
        st.info("No candidates available.")
    else:
        cand_display=candidates[["version","objective_score","profit_factor","win_rate",
                                   "avg_return","max_drawdown","total_trades",
                                   "run_date","rank","notes"]].copy()
        cand_display.columns=["Version","Score","Profit Factor","Win Rate %",
                               "Avg Return %","Max Drawdown %","Trades",
                               "Run Date","Rank","Notes"]
        st.dataframe(cand_display.style.format({
            "Score":"{:.4f}","Profit Factor":"{:.2f}","Win Rate %":"{:.1f}%",
            "Avg Return %":"{:+.2f}%","Max Drawdown %":"{:.1f}%"
        }),use_container_width=True,hide_index=True)

        # Manually promote any candidate
        st.subheader("🔧 Manually Promote a Candidate")
        cand_versions=candidates["version"].tolist()
        sel_v=st.selectbox("Select candidate version",cand_versions)
        mc1,mc2=st.columns(2)
        if mc1.button("⚔️ Set as Challenger"):
            sb.table("agent_params").update({"status": "retired"}).eq("status", "challenger").execute()
            promote_param(int(sel_v), "challenger")
            st.success(f"v{sel_v} is now Challenger.")
            st.rerun()
          if mc2.button("👑 Promote directly to Champion"):
              promote_param(int(sel_v), "champion")
              st.success(f"v{sel_v} is now Champion.")
              st.rerun()  
    

    st.divider()

    # ── Optimization history
    st.subheader("📜 Optimization Run History")
    if opt_runs.empty:
        st.info("No optimization history yet.")
    else:
        fig_hist=px.line(opt_runs.sort_values("run_date"),x="run_date",y="best_score",
                          markers=True,title="Best Objective Score per Optimization Run")
        fig_hist.update_layout(height=260,margin=dict(t=30,b=10,l=0,r=0),
                                plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hist,use_container_width=True)

        st.dataframe(opt_runs[["run_date","n_trials","n_valid_trials","best_score",
                                 "best_profit_factor","best_win_rate","best_avg_return",
                                 "champion_version","challenger_version"]].style.format({
            "best_score":"{:.4f}","best_profit_factor":"{:.2f}",
            "best_win_rate":"{:.1f}%","best_avg_return":"{:+.2f}%"
        }),use_container_width=True,hide_index=True)

    # ── Current active params
    st.divider()
    st.subheader("⚙️ Currently Active Parameters")
    if not champion.empty:
        active_p=json.loads(champion.iloc[0]["params_json"]) if isinstance(champion.iloc[0]["params_json"],str) else champion.iloc[0]["params_json"]
        p_df=pd.DataFrame([{"Parameter":k,"Value":v} for k,v in sorted(active_p.items())])
        st.dataframe(p_df,use_container_width=True,hide_index=True)
    else:
        st.info("Using default parameters (no champion set yet). Run the optimizer to generate candidates.")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ Paper trading & educational purposes only. Not financial advice. "
    "Past backtest results do not guarantee future performance."
)
