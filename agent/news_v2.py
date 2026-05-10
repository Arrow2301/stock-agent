#!/usr/bin/env python3
"""
============================================================
  News Sentiment v2 — drop-in replacement for the relevant
  parts of agent/analyze.py
  ──────────────────────────────────────────────────────────
  The previous news pipeline had three problems demonstrated
  in the recommendations data:

  1. VADER does not understand financial English. Across 62
     "profit"-containing headlines in your DB, 43 were labelled
     POSITIVE and 0 NEGATIVE — even ones that literally said
     "profit falls 75% YoY" or "Profit Booking Halts Momentum".
     VADER scores "profit" as a strong positive token regardless
     of what is happening *to* it.

  2. The relevance filter was too loose. 63 headlines were
     "Option Chain - Live data" pages, 9 were "Share Latest News"
     templates, 7 were "Top gainers/losers" market wraps, 3 were
     stock-market-holiday notices. None were real news, all
     passed the existing relevance filter, all got scored.

  3. Predictive power was zero. Pearson(news_score, win) = +0.082
     across 231 backsim trades. POSITIVE labels won 72.7% vs
     NEUTRAL 70.3% — within sampling noise.

  This module fixes problems 2 and 3, and *partially* fixes
  problem 1 by preferring FinBERT (financially-trained) over
  VADER (general-purpose) and providing a financial-keyword
  override layer for clear bearish phrases that even FinBERT
  occasionally misses.

  Drop-in usage in agent/analyze.py
  ─────────────────────────────────
  Replace the existing fetch_news_sentiment() and the helper
  _headline_is_relevant() with the implementations below.
  Everything else (the run() loop, apply_score_multipliers, etc.)
  stays the same.
============================================================
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

try:
    from gnews import GNews
    _GNEWS_OK = True
except ImportError:
    _GNEWS_OK = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_OK = True
except ImportError:
    _VADER_OK = False

import requests


HF_FINBERT_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"


# ─────────────────────────────────────────────
#  JUNK HEADLINE FILTER — the cheapest, biggest win
#  These patterns appeared 100+ times in the live DB
#  and are not actually news about the company.
# ─────────────────────────────────────────────
JUNK_PATTERNS = [
    # Automated derivatives data pages — very common, not news
    r"option chain",
    r"options? data",
    r"open interest surge",
    r"open interest data",
    # Multi-stock summaries that happen to mention the ticker
    r"top (?:gainers?|losers?)",
    r"top \d+ stocks?",
    r"stocks? to (?:watch|buy)",
    r"(?:nifty|sensex) (?:gainers?|losers?)",
    r"market wrap",
    r"market roundup",
    # Market-hours / holiday notices
    r"stock market holiday",
    r"(?:nse|bse)(?:[, ]+(?:nse|bse))? (?:will|to) remain shut",
    r"trading holiday",
    # SEO templates
    r"share latest news",
    r"share target",
    r"share price history",
    r"latest news today",
    # Other stocks listed alongside
    r"and more:",
    r"and other (?:stocks?|shares?)",
]

_JUNK_RE = re.compile("|".join(JUNK_PATTERNS), re.IGNORECASE)


def _company_name_variants(company_name: str) -> list[str]:
    """Generate name variants to recognise abbreviated forms in headlines.
    'HCL Technologies' should match 'HCL Tech' headlines. 'Tata Consultancy
    Services' should match 'TCS'."""
    if not company_name:
        return []
    cn = _normalize(company_name)
    for suffix in [" limited", " ltd", " corporation", " corp", " company"]:
        if cn.endswith(suffix):
            cn = cn[: -len(suffix)].strip()
    out = {cn}
    words = cn.split()
    if len(words) >= 2:
        out.add(words[0])              # 'HCL'
        out.add(" ".join(words[:2]))   # 'HCL Tech', 'Tata Consultancy'
        if len(words) >= 3:
            out.add(" ".join(words[:3]))
        # Acronym (TCS for Tata Consultancy Services)
        acronym = "".join(w[0] for w in words if w)
        if len(acronym) >= 3:
            out.add(acronym)
    # Truncations: 'Hindustan Unilever' → 'hindustan', 'unilever'
    for w in words:
        if len(w) >= 5:
            out.add(w)
    return [v for v in out if v and len(v) >= 3]


def headline_is_relevant_strict(headline: str, ticker: str, company_name: str) -> bool:
    """
    Stricter than the v1 relevance check:
    - Reject any junk-pattern match (option chain, top gainers, market holiday, etc.)
    - Allow if any reasonable name variant appears, OR the ticker appears
      as a standalone word.
    """
    if not headline:
        return False
    if _JUNK_RE.search(headline):
        return False
    norm = _normalize(headline)
    if not norm:
        return False
    # Try multiple variants of the company name
    for variant in _company_name_variants(company_name):
        if variant in norm:
            return True
    # Standalone ticker
    t = (ticker or "").lower()
    if t and re.search(rf"\b{re.escape(t)}\b", norm):
        return True
    return False


# ─────────────────────────────────────────────
#  FINANCIAL OVERRIDE LEXICON
#  When these tokens appear in a headline, force the label
#  to NEGATIVE regardless of what VADER says, because VADER
#  systematically gets them wrong.
# ─────────────────────────────────────────────
BEARISH_OVERRIDES = [
    r"profit booking",
    r"profit fell",
    r"profit fall(?:s|ing)?",
    r"profit drop(?:s|ped)?",
    r"profit decline(?:s|d)?",
    r"profit slump(?:s|ed)?",
    r"profit slip(?:s|ped)?",
    r"profit dip(?:s|ped)?",
    r"profit (?:halve|halved|halving)",
    r"profit miss(?:ed|es)?",
    r"profit warning",
    r"profit plunge(?:s|d)?",
    r"net loss",
    r"posts loss",
    r"reported a loss",
    r"q[1-4](?: results?)? miss",
    r"miss(?:es|ed) (?:estimates?|expectations?|street)",
    r"weak (?:guidance|outlook|q[1-4])",
    r"guidance cut",
    r"cut(?:s|ting)? guidance",
    # Downgrade — allow up to 60 chars between 'downgrade' and 'to sell/reduce'
    # so 'Downgrades CG Power and Industrial Solutions to Sell' is caught.
    r"downgrade(?:s|d)?\b[^.]{0,60}\bto (?:sell|reduce|underweight|underperform)",
    r"downgraded? from",
    r"target (?:cut|lowered|reduced)",
    r"price target.*(?:cut|lowered|reduced)",
    r"shares (?:fall|fell|drop|dropped|plunge|plunged|crash|crashed|tumble|tumbled|slide|slid|sink|sank|slump|slumped)",
    r"stock (?:fall|fell|drop|dropped|plunge|plunged|crash|crashed|tumble|tumbled|slump|slumped)",
    r"sell rating",
    r"reduce rating",
    r"\bbearish\b",
]

_BEARISH_RE = re.compile("|".join(BEARISH_OVERRIDES), re.IGNORECASE)


BULLISH_OVERRIDES = [
    r"profit (?:up|rises?|grew|grows?|jumps?|surges?|doubles?|soars?)",
    r"profit beat",
    r"q[1-4](?: results?)? beat",
    r"beat(?:s|en)? (?:estimates?|expectations?|street)",
    r"raise(?:s|d) guidance",
    r"upgrade(?:s|d)? to (?:buy|outperform|overweight)",
    r"target (?:raised|hiked|increased)",
    r"price target.*(?:raised|hiked)",
    r"buy rating",
    r"\bbullish\b",
    r"shares (?:rise|risen|rose|jump|jumped|surge|surged|soar|soared|rally|rallied|climb|climbed)",
    r"stock (?:rises?|rose|jumps?|jumped|surge|surged|soars?|soared|rally|rallied|climbs?|climbed)",
    r"hits? (?:record|all[- ]time) high",
    r"52[- ]week high",
]

_BULLISH_RE = re.compile("|".join(BULLISH_OVERRIDES), re.IGNORECASE)


# ─────────────────────────────────────────────
#  RELEVANCE — strict version
# ─────────────────────────────────────────────
def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9&+ ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()





# ─────────────────────────────────────────────
#  HEADLINE FETCH (mostly unchanged; tighter dedup)
# ─────────────────────────────────────────────
def _dedup_headline(h: str) -> str:
    """Aggressive normalisation for dedup: lowercase, strip punctuation,
    drop publisher suffix '... - PublisherName'."""
    h = re.sub(r"\s*-\s*[\w. ]+$", "", h or "")  # drop trailing publisher
    h = re.sub(r"[^a-z0-9 ]+", " ", h.lower())
    return re.sub(r"\s+", " ", h).strip()


def fetch_news_headlines(ticker: str, company_name: str, n: int = 5) -> list[str]:
    if not _GNEWS_OK:
        return []
    queries: list[str] = []
    if company_name:
        queries.extend([f'"{company_name}" stock', f'"{company_name}" share'])
    queries.extend([f'"{ticker}" NSE', f'"{ticker}" share'])

    seen_keys: set[str] = set()
    kept: list[str] = []
    try:
        gn = GNews(language="en", country="IN", period="7d", max_results=max(8, n * 2))
        for query in queries:
            for row in gn.get_news(query) or []:
                headline = (row.get("title") or "").strip()
                if len(headline) <= 10:
                    continue
                key = _dedup_headline(headline)
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                if headline_is_relevant_strict(headline, ticker, company_name):
                    kept.append(headline)
                    if len(kept) >= n:
                        return kept
        return kept[:n]
    except Exception:
        return []


# ─────────────────────────────────────────────
#  SCORING — FinBERT preferred, VADER as fallback,
#  then financial-keyword override layer.
# ─────────────────────────────────────────────
def _label_from_score(s: float) -> str:
    if s >= 0.12: return "POSITIVE"
    if s <= -0.12: return "NEGATIVE"
    return "NEUTRAL"


def _finbert_score(headline: str, hf_token: str) -> Optional[tuple[float, str]]:
    if not headline or not hf_token:
        return None
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            r = requests.post(HF_FINBERT_URL, headers=headers,
                              json={"inputs": headline[:512]}, timeout=25)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    raw = data[0] if isinstance(data[0], list) else data
                    sd = {str(it.get("label", "")).lower(): float(it.get("score", 0.0))
                          for it in raw if isinstance(it, dict)}
                    net = sd.get("positive", 0.0) - sd.get("negative", 0.0)
                    return round(net, 3), _label_from_score(net)
            elif r.status_code == 503 and attempt < 2:
                try:
                    wait = min(float(r.json().get("estimated_time", 20)), 30)
                except Exception:
                    wait = 10
                time.sleep(wait)
                continue
            else:
                break
        except requests.Timeout:
            if attempt < 2:
                time.sleep(5)
                continue
            break
        except Exception:
            break
    return None


def _vader_score(headline: str) -> Optional[tuple[float, str]]:
    if not headline or not _VADER_OK:
        return None
    try:
        an = SentimentIntensityAnalyzer()
        c = float(an.polarity_scores(headline).get("compound", 0.0))
        return round(c, 3), _label_from_score(c)
    except Exception:
        return None


def _apply_financial_override(headline: str, base_score: float, base_label: str) -> tuple[float, str]:
    """
    If the headline contains explicit bearish/bullish financial phrases,
    override the model's classification. This is the layer that fixes
    'profit booking', 'profit falls 75%', 'downgrade to sell' etc.
    """
    if _BEARISH_RE.search(headline):
        # Force a clearly negative score (-0.4) regardless of what VADER said.
        return -0.4, "NEGATIVE"
    if _BULLISH_RE.search(headline):
        # Only override if the model wasn't already firmly positive
        if base_score < 0.3:
            return 0.4, "POSITIVE"
    return base_score, base_label


def score_one_headline(headline: str, hf_token: str = "") -> Optional[tuple[float, str, str]]:
    """
    Returns (score, label, source) where source is 'finbert', 'vader',
    or 'override'. Returns None if neither backend produced a result.
    """
    # Prefer FinBERT
    result = _finbert_score(headline, hf_token) if hf_token else None
    source = "finbert" if result else None
    if result is None:
        result = _vader_score(headline)
        source = "vader" if result else None
    if result is None:
        # No model available — try pure override
        if _BEARISH_RE.search(headline): return -0.4, "NEGATIVE", "override"
        if _BULLISH_RE.search(headline): return  0.4, "POSITIVE", "override"
        return None

    s, lbl = result
    s2, lbl2 = _apply_financial_override(headline, s, lbl)
    if lbl2 != lbl:
        source = f"{source}+override"
    return s2, lbl2, source


def fetch_news_sentiment(ticker: str, company_name: str, hf_token: str = "") -> dict:
    empty = {
        "news_score": 0.0,
        "news_sentiment": "NEUTRAL",
        "news_headline": None,
        "news_headlines": [],
        "news_count": 0,
        "news_multiplier": 1.0,
        "news_source": "disabled",
        "news_alert": False,
    }

    headlines = fetch_news_headlines(ticker, company_name)
    if not headlines:
        return empty

    scored = []
    for h in headlines:
        r = score_one_headline(h, hf_token)
        if r is None:
            continue
        scored.append((h, r[0], r[1], r[2]))

    if not scored:
        return {**empty, "news_headlines": headlines,
                "news_count": len(headlines),
                "news_headline": headlines[0],
                "news_source": "headlines_only"}

    # Sort by absolute strength so the most-extreme headline shows first
    scored.sort(key=lambda x: abs(x[1]), reverse=True)
    headlines_ordered = [h for h, _, _, _ in scored]
    avg_score = round(float(sum(it[1] for it in scored) / len(scored)), 3)
    sentiment = _label_from_score(avg_score)

    # Multiplier: keep the same ±6% envelope as before, but apply *only*
    # when news_score is meaningful. NEUTRAL → multiplier exactly 1.0.
    # This is more conservative than v1, which always nudged the score.
    if avg_score >= 0.35:    multiplier = 1.06
    elif avg_score >= 0.20:  multiplier = 1.03
    elif avg_score <= -0.35: multiplier = 0.94
    elif avg_score <= -0.20: multiplier = 0.97
    else:                    multiplier = 1.00

    sources = sorted({it[3] for it in scored})
    return {
        "news_score": avg_score,
        "news_sentiment": sentiment,
        "news_headline": headlines_ordered[0],
        "news_headlines": headlines_ordered,
        "news_count": len(headlines_ordered),
        "news_multiplier": multiplier,
        "news_source": "+".join(sources),
        "news_alert": False,
    }
