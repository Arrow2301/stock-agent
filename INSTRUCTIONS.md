# Stock Agent v7 — Deployment Instructions

This zip contains the complete files you need to upgrade from v6 to v7.
Total deployment time: **about 20 minutes**, mostly waiting for Supabase
and GitHub to do their thing. No code edits required — just file copies
and one SQL paste.

## What's in this folder

```
v7/
├── INSTRUCTIONS.md                          ← you are here
├── REVIEW.md                                ← full diagnostic report (read once)
├── NEXT_STEPS.md                            ← phased roadmap (Phases 1-4)
├── supabase_migration_v7.sql                ← run once in Supabase
│
├── agent/
│   ├── analyze.py            (REPLACE)     ← v7 version, 4 strategies, hard gates
│   ├── check_alerts.py       (REPLACE)     ← OHLC breach detection + persistence
│   ├── market_data.py        (NEW)         ← shared OHLC + breach utility
│   ├── news_v2.py            (NEW)         ← junk filter + financial overrides
│   ├── holiday_calendar.py   (NEW)         ← NSE holiday list
│   └── health_check.py       (NEW)         ← daily observability script
│
├── dashboard/
│   └── app.py                (REPLACE)     ← v7 with breach-aware UI
│
└── .github/workflows/
    ├── daily_analysis.yml          (REPLACE)  ← now skips on holidays
    ├── daily_health_check.yml      (NEW)      ← runs after market close
    └── weekly_optimize.yml         (REPLACE)  ← cron disabled (see why below)
```

Files **not in this folder are unchanged** — keep your existing
`agent/backsimulate.py`, `agent/optimize.py`, `agent/telegram_alerts.py`,
`requirements.txt`, etc. as they are.

---

## Step 1 — Run the Supabase migration (~2 min)

In your Supabase project → **SQL Editor** → **New Query** → paste the
entire contents of `supabase_migration_v7.sql` → **Run**.

Adds:
- `paper_portfolio.breach_flag`, `breach_price`, `breach_date`
- `health_checks` table
- `market_holidays` table (optional override)
- `reconciliation` table (Phase 4 use)
- Unique-index guard preventing two-champion bugs in the optimiser
- One-shot purge of `ticker_run_log` rows older than 90 days

Safe to re-run; everything uses `IF NOT EXISTS`.

---

## Step 2 — Replace files in your repo (~5 min)

1. Open your `stock-agent` repo on GitHub.
2. For each file marked **(REPLACE)** above, click the existing file in
   the GitHub UI → click the pencil icon (Edit) → delete all contents →
   paste the new contents from this folder → **Commit**.
3. For each file marked **(NEW)**, click **Add file → Create new file**,
   paste the path (e.g. `agent/market_data.py`) and contents, **Commit**.

Order doesn't matter — files reference each other through Python imports
that work as long as they're all present before the next workflow run.

If you prefer the all-at-once route: clone locally, copy this whole
`v7/` folder over your repo (preserving the directory structure), commit
all changes in one push.

---

## Step 3 — (Optional) Set HF_TOKEN for FinBERT (~2 min)

The new `news_v2.py` uses VADER + financial-keyword overrides by default.
If you want true financial sentiment, get a free HuggingFace token:

1. Sign up at huggingface.co (free, no card needed).
2. Account → Settings → Access Tokens → New token (Read role).
3. In your GitHub repo: **Settings → Secrets and variables → Actions
   → New repository secret**, name `HF_TOKEN`, paste the token.

The next run will use FinBERT instead of VADER. Either way works — VADER
+ overrides catches the egregious cases (profit booking, downgrade-to-sell).

---

## Step 4 — Trigger a manual run (~5 min)

1. GitHub → **Actions** tab.
2. Pick **"Daily Stock Analysis"** → **Run workflow** → **Run workflow**.
3. Watch the log. You should see:
   - `Stocks : 192 NSE tickers` (or similar)
   - `Pipeline summary:` block at the end with new lines:
     - `Rejected: multi-strat   : N`     (the new gate count)
     - `Rejected: bearish regime: N`
   - `Final signals: ~30-50%` of what you saw on v6 — fewer is correct.

If the workflow fails, the log will tell you which file is missing or
mistyped. The most common cause is forgetting to add a (NEW) file.

---

## Step 5 — Check the dashboard (~2 min)

1. Open your Streamlit dashboard.
2. **My Paper Portfolio** → if you have any open positions where the
   stock has wicked through SL on any day since entry, the icon should
   now be 🚨 with a clear breach message including the date.
3. **Today's Signals** → notice the leaderboard is shorter and the
   News column no longer has dubious POSITIVE labels on profit-decline
   headlines.

---

## Step 6 — Verify the new health check (next morning, ~1 min)

The `daily_health_check.yml` workflow runs at 16:00 IST after market
close. The next morning, check Telegram. You should see *no message* if
all checks passed (the workflow is silent on green). If you get a
WARN/FAIL message, the detail line tells you exactly what's wrong.

You can also manually trigger it: **Actions → Daily Health Check →
Run workflow**. Look at the log for the seven check lines.

---

## What to expect after deployment

### Immediately

- **Trade count drops by ~50%.** From your historical data, the v7 hard
  gates would have cut 727 BUYs to ~350. This is intentional — the
  rejected ones were predominantly anti-predictive multi-strategy
  setups and bearish-regime longs.
- **Win rate up.** Single-strategy + non-bearish-regime BUYs in your
  historical data won 71% (vs the overall 68%). Live results should
  drift toward this once the new gates have run for a few weeks.
- **News labels much more accurate.** No more "profit booking → POSITIVE."
  ~22% of headlines are now correctly filtered as junk (option chains,
  top-gainers lists, holiday notices).

### Over the next 4-6 weeks

- The composite score's anti-prediction pattern should weaken as the
  worst-quartile signals get gated out before scoring.
- The health check will tell you if live results diverge from backsim by
  more than 20pp — your earliest-warning system for any logic divergence.

### Things that are *intentionally* still wrong in v7

These are flagged for Phase 2:
- Composite score is still hand-weighted. Phase 2 replaces it with a
  calibrated logistic regression trained on the backsim outcomes.
- The optimiser is **paused** because it currently maximises that same
  hand-weighted score. Re-enable only after the LR model exists and the
  Optuna objective is repointed at holdout AUC.
- No position sizing. You still type the qty manually. Phase 2 adds
  fixed-fractional risk sizing (1% of capital per trade).

When you're ready, see `NEXT_STEPS.md` for the staged roadmap with
specific code sketches for Phases 2-4.

---

## Rolling back (if something goes wrong)

Every changed file has its v6 equivalent in your git history. To revert
a single file: GitHub → file → History → click the v6 commit → "Restore
this version". The Supabase migration is *additive only* (adds columns
and tables, never drops), so the new schema is forward-compatible with
running v6 code if you ever need to roll back the agent code.

The one item that affects historical comparability: the `recommendations`
table written from v7 onward will show ~50% fewer rows than v6 was
producing. That's correct, not a regression — but if you're computing
"avg signals per day" charts in the dashboard, expect a step-down.

---

## What I tested

- `agent/analyze.py` — Python syntax + import sanity + `get_strategies()`
  returns the v7 four-strategy roster + Supertrend function is gone.
- `agent/news_v2.py` — 13 unit tests on the actual failing cases from
  your historical data (Suzlon profit booking, Coromandel profit falls
  75%, CG Power downgrade to Sell, etc.). 13/13 pass. Empirically tested
  against all 789 historical headlines: 22% correctly filtered as junk,
  9 confirmed POSITIVE→NEGATIVE flips on real failures.
- `agent/market_data.py` — 9 unit tests covering normal SL hit, gap-down
  SL, target hit, gap-up target, same-bar both-touched (stop-first),
  no-breach, recovered intraday wick (the original bug), missing data,
  and SL-only / target-only modes. 9/9 pass.
- `agent/check_alerts.py` — Python syntax. Live test requires Supabase
  + open positions, so verify on your first run.
- `dashboard/app.py` — Python syntax. Streamlit-side rendering verified
  on first dashboard load.

---

## Need help

If a workflow fails or the dashboard breaks, the most useful thing to
share back is the *full log* from the failing GitHub Actions run, plus
which step it failed at. Most v7 issues will be one of:
- Missing (NEW) file that wasn't committed
- Stale Supabase credentials in GitHub Secrets
- Schema migration not yet applied (you'll see "column X does not exist")

In all three cases the fix is: add the file / re-paste the secret / re-run
the migration SQL.
