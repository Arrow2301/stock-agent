# 🇮🇳 Indian Stock Agent — Complete Setup Guide

**100% free. 100% online. Zero local setup required.**

---

## Architecture

```
GitHub Actions (7 AM IST)
    │
    ▼  runs analyze.py
    │
    ▼  writes signals
Supabase (free PostgreSQL)
    │
    ▼  reads signals + paper trades
Streamlit Community Cloud (free dashboard)
```

---

## Step 1 — Create Supabase Database (~5 minutes)

1. Go to **[supabase.com](https://supabase.com)** → Sign up free
2. Click **"New Project"** → give it a name (e.g. `stock-agent`) → set a database password → **Create project**
3. Wait ~2 minutes for it to spin up
4. In the left sidebar, click **SQL Editor** → **New Query**
5. Copy the entire contents of `supabase_setup.sql` and paste it in → click **Run**
   - This creates 3 tables: `recommendations`, `paper_portfolio`, `agent_meta`
6. Go to **Settings → API**:
   - Copy **Project URL** → save this as `SUPABASE_URL`
   - Copy **anon / public key** → save this as `SUPABASE_KEY`

---

## Step 2 — Push to GitHub (~3 minutes)

1. Go to **[github.com](https://github.com)** → Sign up free (if you don't have an account)
2. Click **"New repository"** → name it `stock-agent` → set to **Public** → **Create**
3. Upload all the project files:
   - Click **"uploading an existing file"**
   - Drag and drop all files/folders from this project
   - Commit changes

   > **OR** use GitHub's web editor to create files one by one — paste each file's content

4. Add GitHub Secrets (your Supabase credentials):
   - In your repo → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**
   - Add `SUPABASE_URL` with your Supabase Project URL
   - Add `SUPABASE_KEY` with your Supabase anon key

---

## Step 3 — Test the Agent (~2 minutes)

1. In your GitHub repo → **Actions** tab
2. Click **"Daily Stock Analysis"** workflow
3. Click **"Run workflow"** → **Run workflow** (green button)
4. Watch it run (takes 3–5 minutes)
5. Once it finishes ✅, go to Supabase → **Table Editor** → `recommendations`
   - You should see today's buy/sell signals!

---

## Step 4 — Deploy the Dashboard (~5 minutes)

1. Go to **[share.streamlit.io](https://share.streamlit.io)** → Sign in with GitHub
2. Click **"New app"**
3. Set:
   - **Repository:** `your-username/stock-agent`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`
4. Click **"Advanced settings"** → **Secrets** → paste this (with your real values):
   ```toml
   SUPABASE_URL = "https://xxxxxxxxxxxx.supabase.co"
   SUPABASE_KEY = "your-anon-key"
   DASHBOARD_PASSWORD = "your-chosen-password"
   ```
5. Click **Deploy** → wait ~2 minutes
6. 🎉 Your dashboard is live at a URL like `https://your-username-stock-agent-dashboard-app-xxxx.streamlit.app`

---

## Daily Usage

- **Every weekday at 7:00 AM IST**, GitHub Actions automatically runs the agent
- Open your Streamlit dashboard → **📊 Today's Signals** to see that morning's recommendations
- Before market opens (9:15 AM IST), review signals and paper-buy stocks you like
- Use **💼 My Paper Portfolio** to track your paper trades and P&L in real time
- After a few weeks, check **📈 Strategy Stats** to see which strategies perform best for you

---

## Customising Your Watchlist

In `agent/analyze.py`, edit the `EXTRA_WATCHLIST`:
```python
EXTRA_WATCHLIST = [
    "IRCTC", "ZOMATO", "IRFC", "DELHIVERY", "TATAPOWER"
]
```
Commit the change → GitHub will use it in the next morning's run.

---

## Adjusting Signal Sensitivity

Also in `agent/analyze.py`:

| Variable | Default | Effect |
|---|---|---|
| `MIN_BUY_SCORE` | `2` | Lower → more signals (noisier). Higher → fewer (higher conviction) |
| `MIN_SELL_SCORE` | `2` | Same for sell signals |
| `RSI_OVERSOLD` | `40` | Raise to `45` for more RSI-based buys |
| `HOLD_DAYS` | `10` | Backtest holding period (doesn't affect live signals) |

---

## Project File Structure

```
stock-agent/
├── .github/
│   └── workflows/
│       └── daily_analysis.yml   ← GitHub Actions schedule
├── agent/
│   └── analyze.py               ← Core analysis engine
├── dashboard/
│   └── app.py                   ← Streamlit dashboard
├── .streamlit/
│   └── secrets.toml.template    ← Reference for Streamlit secrets
├── supabase_setup.sql            ← Run once in Supabase SQL editor
├── requirements.txt
└── README.md
```

---

## Troubleshooting

**Agent failed in GitHub Actions?**
- Check the Actions log for errors
- Most common: a ticker returned no data from Yahoo Finance (safe to ignore)
- Run it again manually via Actions → "Run workflow"

**Dashboard shows "No signals"?**
- The agent may not have run yet today (it runs at 7 AM IST)
- Trigger a manual run from GitHub Actions

**Live prices not loading in portfolio?**
- Yahoo Finance has occasional rate limits — refresh after a minute

---

## ⚠️ Disclaimer

This tool is for educational and paper trading purposes only.
It is not financial advice. Never invest more than you can afford to lose.
Always cross-check signals with your own research.
Start with paper trading for at least 4–6 weeks before using real money.
