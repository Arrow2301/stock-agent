#!/usr/bin/env python3
"""
============================================================
  Phase 3 — Calibrated LightGBM Scoring Model
  (replaces Phase 2's calibrated LR)

  WHAT CHANGED FROM PHASE 2
  ─────────────────────────
  ─ Switched LogisticRegression → LightGBM. LR produced AUC 0.553
    on the holdout — essentially a linear projection of features
    that are weakly informative individually. LightGBM models
    interactions natively, which is where the real edge in
    momentum-vs-mean-reversion detection lives.
  ─ Removed StandardScaler. Tree models don't need feature scaling
    and scaling actually hurts split quality on long-tailed
    financial features (e.g. atr_pct).
  ─ Added 3 India VIX features: vix_level, vix_change_5d,
    vix_zscore_60d. VIX leads NSE by 1-3 days as a regime signal
    and is orthogonal to anything derivable from a single ticker's
    price chart.
  ─ Kept CalibratedClassifierCV(method='isotonic'). LightGBM's raw
    probabilities are typically miscalibrated; we still want
    p_win to be interpretable as P(win) for ranking and analysis.
  ─ Added feature importance reporting after training, so the
    user can see what the model actually learned.

  TRAINING
  ────────
  Input  : synthetic_training_data table (built by build_training_data.py)
  Method : LGBMClassifier → CalibratedClassifierCV (5-fold isotonic)
  Output : pickled bytes saved to score_models, MD5 fingerprint, AUC report.

  VALIDATION
  ──────────
  Walk-forward time-based split:
    * Train on rows where signal_date < cutoff (default: 80th-pct date)
    * Test  on rows where signal_date >= cutoff
  Reports AUC, log-loss, Brier score, per-quartile holdout WR, and
  feature importance ranking. If AUC < 0.55, training is aborted and
  no model is saved (champion never degrades).

  PREDICTION (called from analyze.py — UNCHANGED API)
  ─────────────────────────────────
    from score_model import load_model, predict_p_win
    model, feats = load_model()
    p = predict_p_win(model, feats, signal_features_dict)
============================================================
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
import pickle
import warnings
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  FEATURE SCHEMA  — keep stable; analyze.py must produce
#  exactly these keys (in any order) when calling predict.
#  Phase 3 adds vix_level, vix_change_5d, vix_zscore_60d.
# ─────────────────────────────────────────────
NUMERIC_FEATURES = [
    "rsi", "macd_hist", "atr_pct",
    "change_1d", "change_5d", "change_30d",
    "pct_from_52w_high", "pct_from_52w_low",
    "vol_ratio",
    "n_firing",
    "risk_pct", "reward_pct", "rr_ratio",
    "regime_score",
    "mom_vs_nifty_30d",
    # Phase 3: India VIX
    "vix_level", "vix_change_5d", "vix_zscore_60d",
]
BINARY_FEATURES = [
    "ema_bullish",
    "has_donchian", "has_ema", "has_rsi_trend", "has_bollinger",
    "single_strat", "multi_strat",
]
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES   # 18 numeric + 7 binary = 25


# ─────────────────────────────────────────────
#  LIGHTGBM HYPERPARAMETERS
#  Conservative defaults tuned for tabular financial data
#  (~50k rows, weak feature signal, regime-shifty labels).
#  Strong regularization to combat overfitting on noise.
# ─────────────────────────────────────────────
LGBM_PARAMS = dict(
    objective         = "binary",
    n_estimators      = 300,
    learning_rate     = 0.05,
    num_leaves        = 31,         # ~2^5 splits per tree
    max_depth         = -1,         # let num_leaves cap complexity
    min_child_samples = 50,         # large minimum for noisy data
    min_split_gain    = 0.01,       # avoid splits with no real gain
    reg_alpha         = 0.1,        # L1
    reg_lambda        = 0.1,        # L2
    subsample         = 0.8,        # row sub-sampling
    subsample_freq    = 1,          # required for subsample to take effect
    colsample_bytree  = 0.8,        # feature sub-sampling
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
def get_supabase():
    if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY")):
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY must be set")
    from supabase import create_client
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def load_training_data(sb, min_rows: int = 500,
                       exclude_quality_flags: tuple[str, ...] = (
                           "borderline_timeout", "gap_stop", "very_low_vol",
                       )) -> pd.DataFrame:
    """
    Load all rows from synthetic_training_data. Paginates Supabase's
    1000-row default limit. Returns a clean DataFrame.

    `exclude_quality_flags` filters out rows whose `data_quality_flags`
    column (added in v8 generator) contains ANY of the listed flags.
    Defaults exclude the three known-noisy buckets:
      * borderline_timeout — return within ±1%, label ≈ random coin flip
      * gap_stop           — instant stop on gap-down at entry open
      * very_low_vol       — ATR < 1%, ATR-based SL is unreliable
    Pass an empty tuple to disable filtering and include everything.
    """
    rows: list[dict] = []
    page = 0
    page_size = 1000
    while True:
        r = (
            sb.table("synthetic_training_data")
            .select("*")
            .order("signal_date")
            .range(page * page_size, (page + 1) * page_size - 1)
            .execute()
        )
        if not r.data:
            break
        rows.extend(r.data)
        if len(r.data) < page_size:
            break
        page += 1
        if page > 100:
            break
    if len(rows) < min_rows:
        raise RuntimeError(
            f"Only {len(rows)} training rows; need at least {min_rows}. "
            "Run agent/build_training_data.py first."
        )
    df = pd.DataFrame(rows)
    df["signal_date"] = pd.to_datetime(df["signal_date"])

    # Quality filter (only if column exists — old datasets won't have it)
    if exclude_quality_flags and "data_quality_flags" in df.columns:
        flags = df["data_quality_flags"].fillna("").astype(str)
        excl = pd.Series(False, index=df.index)
        for f in exclude_quality_flags:
            excl |= flags.str.contains(f, na=False)
        before = len(df)
        df = df[~excl].reset_index(drop=True)
        print(f"   Filtered {before - len(df)} rows by quality flags "
              f"({', '.join(exclude_quality_flags)})")

    return df


def feature_coverage(df: pd.DataFrame) -> dict[str, float]:
    """Report what fraction of rows have each numeric feature populated.
       Used to diagnose features that are 0%-covered (model can't learn from them)
       and ones <20%-covered (median imputation will dominate)."""
    cov: dict[str, float] = {}
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            cov[col] = 0.0
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        cov[col] = round(float(s.notna().mean()), 3)
    return cov


def prepare_xy(df: pd.DataFrame, *, coverage: dict[str, float] | None = None
               ) -> tuple[pd.DataFrame, pd.Series]:
    """Apply schema, fill NaNs, coerce booleans to int.

    Features with effectively zero coverage are zeroed out for ALL rows
    rather than median-imputed. A column where every row is NaN has no
    information and median-imputation produces 0.0 (since pandas median
    of an all-NaN series is NaN, and we coerce NaN→0). Treating such a
    column as "constant zero" is honest and lets the tree skip it.
    """
    X = df.copy()
    for col in ALL_FEATURES:
        if col not in X.columns:
            X[col] = 0
    for col in BINARY_FEATURES:
        X[col] = X[col].fillna(0).astype(int)

    if coverage is None:
        coverage = feature_coverage(df)

    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        cov = coverage.get(col, 0.0)
        if cov < 0.05:
            # < 5% coverage → treat the whole column as zero; median is meaningless.
            X[col] = 0.0
            continue
        med = X[col].median()
        if pd.isna(med):
            med = 0.0
        X[col] = X[col].fillna(med)
    y = X["was_win"].astype(int)
    return X[ALL_FEATURES], y


# ─────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────
def train_model(df: pd.DataFrame, holdout_cutoff: str | None = None) -> dict:
    """
    Train a calibrated LightGBM. Returns a dict with the fitted pipeline,
    the feature list, validation metrics, feature importance, and a fingerprint.
    """
    from lightgbm import LGBMClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

    # ── Walk-forward holdout split. If no cutoff provided, take last 20% of dates.
    df = df.sort_values("signal_date").reset_index(drop=True)
    if holdout_cutoff:
        cutoff = pd.Timestamp(holdout_cutoff)
    else:
        cutoff = df.signal_date.quantile(0.80)
    train_mask = df.signal_date < cutoff
    test_mask  = df.signal_date >= cutoff
    df_train, df_test = df[train_mask], df[test_mask]
    if len(df_train) < 200 or len(df_test) < 50:
        raise RuntimeError(
            f"Insufficient split: train={len(df_train)}, test={len(df_test)}. "
            "Need ≥200 train and ≥50 test."
        )

    # Coverage is computed on the FULL dataset so train and test see the
    # same imputation policy. Without this, a feature might be 0%-covered
    # in train but populated in test (or vice-versa) and we'd still
    # silently median-impute it.
    coverage = feature_coverage(df)
    X_train, y_train = prepare_xy(df_train, coverage=coverage)
    X_test,  y_test  = prepare_xy(df_test,  coverage=coverage)

    # ── Diagnostic LightGBM (used for feature importance + raw AUC).
    # We fit a separate LGBM here because CalibratedClassifierCV wraps the
    # estimator inside a list of fold-fits, making feature importance
    # extraction awkward. This costs an extra 10-15 sec but is worth it
    # for the diagnostic.
    diag = LGBMClassifier(**LGBM_PARAMS)
    diag.fit(X_train, y_train)
    p_test_raw = diag.predict_proba(X_test)[:, 1]
    raw_auc    = roc_auc_score(y_test, p_test_raw)

    importance = sorted(
        zip(ALL_FEATURES, diag.feature_importances_),
        key=lambda x: -x[1],
    )

    # ── Calibrated model: this is what we save and serve.
    cv_folds = min(5, max(2, len(df_train) // 200))
    base = LGBMClassifier(**LGBM_PARAMS)
    pipe = CalibratedClassifierCV(estimator=base, cv=cv_folds, method="isotonic")
    pipe.fit(X_train, y_train)

    # ── Holdout metrics (on the calibrated model — what users actually see)
    p_test    = pipe.predict_proba(X_test)[:, 1]
    auc       = roc_auc_score(y_test, p_test)
    ll        = log_loss(y_test, p_test, labels=[0, 1])
    brier     = brier_score_loss(y_test, p_test)
    base_rate = y_test.mean()

    # Per-quartile holdout WR — the most important sanity check.
    # If the model is real, Q4 (highest predicted P) should win
    # noticeably more than Q1.
    qdf = pd.DataFrame({"p": p_test, "y": y_test.values})
    try:
        qdf["q"] = pd.qcut(qdf["p"], q=4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
    except ValueError:
        # Many ties in p → fewer than 4 bins. Fall back to qcut without labels
        # so pandas auto-numbers them, then stringify.
        qdf["q"] = pd.qcut(qdf["p"], q=4, duplicates="drop").astype(str)
    quartile_wr = qdf.groupby("q", observed=True).agg(
        n  = ("y", "count"),
        wr = ("y", lambda x: round(x.mean() * 100, 1)),
        mean_p = ("p", lambda x: round(x.mean(), 3)),
    ).to_dict("index")

    metrics = {
        "model_kind":     "lightgbm_calibrated",
        "n_train":        int(len(df_train)),
        "n_test":         int(len(df_test)),
        "cutoff":         str(cutoff.date()),
        "auc":            round(float(auc), 4),
        "auc_raw":        round(float(raw_auc), 4),     # uncalibrated reference
        "log_loss":       round(float(ll), 4),
        "brier":          round(float(brier), 4),
        "base_rate":      round(float(base_rate), 4),
        "quartile_wr":    {str(k): v for k, v in quartile_wr.items()},
        "feature_importance_top10": [
            {"feature": f, "importance": int(i)} for f, i in importance[:10]
        ],
    }

    if auc < 0.55:
        # Failed gate. Skip the expensive full-data refit; main() will print
        # metrics + feature importance and abort without saving. The existing
        # champion model stays untouched.
        return {
            "pipeline":     None,
            "features":     ALL_FEATURES,
            "blob":         b"",
            "fingerprint":  "",
            "metrics":      metrics,
            "importance":   importance,
            "passed_gate":  False,
        }

    # Refit on the full dataset (train + test) for production deploy.
    # Reuse the SAME coverage mask so the served model agrees with what
    # we just validated.
    X_full, y_full = prepare_xy(df, coverage=coverage)
    pipe_full = CalibratedClassifierCV(
        estimator=LGBMClassifier(**LGBM_PARAMS),
        cv=cv_folds,
        method="isotonic",
    )
    pipe_full.fit(X_full, y_full)

    blob = pickle.dumps({"pipeline": pipe_full, "features": ALL_FEATURES})
    fingerprint = hashlib.md5(blob).hexdigest()[:12]

    return {
        "pipeline":     pipe_full,
        "features":     ALL_FEATURES,
        "blob":         blob,
        "fingerprint":  fingerprint,
        "metrics":      metrics,
        "importance":   importance,    # full list, not just top 10
        "passed_gate":  True,
    }


def save_model(sb, trained: dict) -> None:
    """Persist the model + metrics to score_models table."""
    import base64
    sb.table("score_models").insert({
        "fingerprint":   trained["fingerprint"],
        "trained_at":    datetime.utcnow().isoformat(),
        "pickle_b64":    base64.b64encode(trained["blob"]).decode(),
        "metrics":       trained["metrics"],
        "feature_list":  trained["features"],
        "status":        "champion",
    }).execute()
    # Demote any older champions
    sb.table("score_models").update({"status": "retired"}).neq(
        "fingerprint", trained["fingerprint"]
    ).eq("status", "champion").execute()


# ─────────────────────────────────────────────
#  PREDICTION (used by analyze.py — UNCHANGED API)
# ─────────────────────────────────────────────
_model_cache = {"loaded": None}

def load_model(sb=None) -> tuple[object, list[str]] | None:
    """
    Lazily load the current champion model. Returns (pipeline, feature_list)
    or None if no champion exists yet (analyze.py should fall back to the
    legacy composite_score in that case).
    """
    if _model_cache["loaded"] is not None:
        return _model_cache["loaded"]
    if sb is None:
        sb = get_supabase()
    try:
        r = (
            sb.table("score_models")
            .select("pickle_b64, feature_list, fingerprint")
            .eq("status", "champion")
            .order("trained_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception:
        return None
    if not r.data:
        return None
    import base64
    blob = base64.b64decode(r.data[0]["pickle_b64"])
    bundle = pickle.loads(blob)
    pipeline = bundle["pipeline"]
    feats    = bundle["features"]
    _model_cache["loaded"] = (pipeline, feats)
    print(f"  ℹ️  Score model loaded: {r.data[0]['fingerprint']}")
    return pipeline, feats


def predict_p_win(pipeline, feats: list[str], signal_features: dict) -> float:
    """
    Return P(win) for one signal. signal_features is a dict that must contain
    all keys in `feats`. Missing keys default to 0; unexpected keys are ignored.
    Works for both Phase 2 (LR) and Phase 3 (LGBM) models — predict_proba is
    stable across both.
    """
    row = {f: signal_features.get(f, 0) for f in feats}
    X = pd.DataFrame([row])
    # Coerce dtypes
    for col in BINARY_FEATURES:
        if col in X.columns:
            X[col] = X[col].fillna(0).astype(int)
    for col in NUMERIC_FEATURES:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    p = pipeline.predict_proba(X[feats])[0, 1]
    return float(p)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default=None,
                    help="Train/test cutoff date (default: 80th-percentile signal_date)")
    ap.add_argument("--no-save", action="store_true",
                    help="Train and report, but don't write model to Supabase")
    args = ap.parse_args()

    print("\n📈 Phase 3 — Calibrated LightGBM scoring model")
    print(f"   Loading training data from Supabase...")
    sb = get_supabase()
    df = load_training_data(sb)
    print(f"   Loaded {len(df)} rows  ({df.signal_date.min().date()} → {df.signal_date.max().date()})")

    base_rate = df.was_win.mean()
    print(f"   Base win rate    : {base_rate*100:.1f}%")
    print(f"   Date span        : {(df.signal_date.max() - df.signal_date.min()).days} days")
    print(f"   Tickers covered  : {df.ticker.nunique()}")
    print(f"   Regime mix       : {dict(df.regime.value_counts())}")
    if "vix_level" in df.columns:
        n_vix = df.vix_level.notna().sum()
        print(f"   Rows w/ VIX data : {n_vix}/{len(df)} ({n_vix*100//len(df)}%)")

    # Coverage report — features < 5% coverage will be zeroed out (not median-imputed).
    cov = feature_coverage(df)
    low_cov = [(f, c) for f, c in cov.items() if c < 0.50]
    if low_cov:
        print(f"\n   ── Numeric features with low coverage ──")
        for f, c in sorted(low_cov, key=lambda x: x[1]):
            tag = " (DROPPED — coverage < 5%)" if c < 0.05 else ""
            print(f"      {f:<22}  {c*100:>5.1f}% covered{tag}")
    print()

    print("   Training calibrated LightGBM...")
    trained = train_model(df, holdout_cutoff=args.cutoff)
    m = trained["metrics"]
    print(f"\n   ── Holdout metrics ──")
    print(f"      Train rows           : {m['n_train']}")
    print(f"      Holdout rows         : {m['n_test']}  (after {m['cutoff']})")
    print(f"      Base rate            : {m['base_rate']*100:.1f}%")
    print(f"      AUC (calibrated)     : {m['auc']:.3f}")
    print(f"      AUC (raw LGBM)       : {m['auc_raw']:.3f}")
    print(f"      Log-loss             : {m['log_loss']:.3f}")
    print(f"      Brier score          : {m['brier']:.3f}")
    print(f"      Quartile WR (sorted by predicted P):")
    for q, vals in m["quartile_wr"].items():
        print(f"        {q}:  n={vals['n']:>4}  WR={vals['wr']:>5.1f}%  mean_p={vals['mean_p']}")
    print(f"\n   ── Top features by gain importance ──")
    for entry in m["feature_importance_top10"]:
        print(f"      {entry['feature']:<22}  {entry['importance']}")

    if not trained["passed_gate"]:
        print(f"\n   ❌ AUC {m['auc']:.3f} < 0.55 — failed gate.")
        print(f"      Aborting save. Existing champion remains active.")
        sys.exit(1)

    print(f"\n   Fingerprint           : {trained['fingerprint']}")

    if args.no_save:
        print("   --no-save flag set; not persisting.")
    else:
        save_model(sb, trained)
        print(f"   ✅ Saved as champion model")


if __name__ == "__main__":
    main()
