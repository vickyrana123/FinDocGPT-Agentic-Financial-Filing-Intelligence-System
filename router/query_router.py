"""
query_router.py

Decides whether a question should be answered via:
  - live data tools (e.g. current stock price)
  - static RAG over SEC filings
  - both

Two-stage decision:
  1. Rule-based keyword hints (fast, free, deterministic)
  2. LLM classification fallback for ambiguous queries
"""

import re
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from llm_client import call_llm

LIVE_KEYWORDS = [
    "current price", "current stock", "stock price", "today", "right now",
    "as of now", "live", "trading at", "how much is", "what's the price"
]

FILING_KEYWORDS = [
    "10-k", "10-q", "8-k", "filed", "filing", "reported", "annual report",
    "quarterly report", "risk factors", "management discussion", "md&a",
    "net income", "balance sheet", "cash flow statement", "revenue in"
]

TICKER_PATTERN = re.compile(r'\b[A-Z]{1,5}\b')

# Only trust an all-caps match if it's actually one of our known tickers -
# this avoids false positives on unrelated acronyms like SEC, CEO, USD.
KNOWN_TICKERS = {"AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "META", "NVDA", "NFLX"}

COMPANY_TO_TICKER = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "tesla": "TSLA",
    "amazon": "AMZN",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "meta": "META",
    "facebook": "META",
    "nvidia": "NVDA",
    "netflix": "NFLX",
}


def rule_based_route(query: str) -> str | None:
    q = query.lower()
    live_hit = any(kw in q for kw in LIVE_KEYWORDS)
    filing_hit = any(kw in q for kw in FILING_KEYWORDS)

    if live_hit and not filing_hit:
        return "live"
    if filing_hit and not live_hit:
        return "filings"
    if live_hit and filing_hit:
        return "both"
    return None  # ambiguous -> fall through to LLM


def llm_route(query: str) -> str:
    prompt = f"""Classify this financial question into exactly one category:

- "live": needs current/real-time market data (e.g. stock price today)
- "filings": needs information from SEC filings (10-K, 10-Q, historical financials, risk factors, MD&A)
- "both": needs both live data AND filings context

Question: "{query}"

Respond with ONLY one word: live, filings, or both."""

    try:
        answer = call_llm(prompt, temperature=0.0, max_tokens=10, timeout=30).strip().lower()
        for candidate in ("live", "filings", "both"):
            if candidate in answer:
                return candidate
        return "filings"
    except Exception:
        return "filings"


def route_query(query: str) -> str:
    rule_result = rule_based_route(query)
    return rule_result if rule_result else llm_route(query)


def extract_ticker(query: str) -> str | None:
    """Extract a ticker symbol from the query.
    First tries direct ticker matches (only against KNOWN_TICKERS, to avoid
    false positives on unrelated all-caps acronyms like SEC, CEO, USD).
    Falls back to matching known company names using word boundaries (so
    'meta' doesn't accidentally match inside 'metadata')."""
    candidates = [c for c in TICKER_PATTERN.findall(query) if c in KNOWN_TICKERS]
    if candidates:
        return candidates[0]

    q_lower = query.lower()
    for name, ticker in COMPANY_TO_TICKER.items():
        if re.search(rf"\b{re.escape(name)}\b", q_lower):
            return ticker

    return None


if __name__ == "__main__":
    test_queries = [
        "What is Apple's current stock price?",
        "What were Apple's risk factors in their latest 10-K?",
        "How does TSLA's current price compare to their reported revenue growth?",
        "What does the SEC require in a 10-K filing?",  # should NOT extract "SEC" as ticker now
    ]
    for q in test_queries:
        print(f"\nQ: {q}")
        print(f"Route: {route_query(q)}")
        print(f"Ticker: {extract_ticker(q)}")