"""
query_router.py

Decides whether a question should be answered via:
  - live data tools (e.g. current stock price)
  - static RAG over SEC filings
  - both
  - general conversation (greetings, thanks, capability questions)

Two-stage decision for live/filings/both:
  1. Rule-based keyword hints (fast, free, deterministic)
  2. LLM classification fallback for ambiguous queries

General conversation detection happens first and short-circuits everything
else - see is_general_query().
"""

import re
import random
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
    "net income", "balance sheet", "cash flow statement", "revenue in",
    "grew", "growth", "grow",
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

# Single source of truth for praise/thanks detection - previously this was
# defined separately in GENERAL_PATTERNS (used for ROUTING) and again
# inline inside general_response() (used only for reply selection). They
# drifted apart when one got updated and the other didn't, causing
# "great response from your side" to fail routing as general even though
# the reply-selection regex would have handled it correctly. Now both
# reference this one constant.
PRAISE_WORDS = r'thanks|thank you|thx|appreciate it|great answer|good job|well done|awesome|perfect|got it|makes sense|great|nice|cool|good'

GENERAL_PATTERNS = [
    r'^\s*(hi|hello|hey|yo|good morning|good afternoon|good evening)\b',
    rf'\b({PRAISE_WORDS})\b',
    r'^\s*(bye|goodbye|see you|later)\b',
    r'\b(what can you do|who are you|what is this|help me understand what you do)\b',
]

GREETING_RESPONSES = [
    "Hi there! I'm FinDocGPT — I can answer questions about SEC filings (10-Ks, 10-Qs) for Apple, Microsoft, Tesla, Amazon, and NVIDIA, plus check live stock prices. What would you like to know?",
    "Hello! Ask me about a company's filings, current stock price, or both — happy to help.",
]

THANKS_RESPONSES = [
    "You're welcome! Let me know if you have any other questions.",
    "Glad that helped! Feel free to ask anything else about these filings or live prices.",
    "Anytime! I'm here if you want to dig into anything else.",
]

CAPABILITY_RESPONSES = [
    "I can answer questions from SEC 10-K/10-Q filings (Apple, Microsoft, Tesla, Amazon, NVIDIA), look up live stock prices, or combine both in one answer — just ask naturally!",
]

FAREWELL_RESPONSES = [
    "Goodbye! Come back anytime you have questions about these filings or stock prices.",
]


def is_general_query(query: str) -> bool:
    """
    Detects greetings, thanks, farewells, and capability questions.
    Guarded against false positives: if the message also contains live or
    filing keywords (e.g. "thanks, what's Apple's price?"), it's NOT
    treated as general - the real question still gets routed normally.
    """
    q = query.lower().strip()
    if any(kw in q for kw in LIVE_KEYWORDS) or any(kw in q for kw in FILING_KEYWORDS):
        return False
    return any(re.search(p, q) for p in GENERAL_PATTERNS)


def general_response(query: str) -> str:
    q = query.lower()
    if re.search(rf'\b({PRAISE_WORDS})\b', q):
        return random.choice(THANKS_RESPONSES)
    if re.search(r'^\s*(bye|goodbye|see you|later)\b', q):
        return random.choice(FAREWELL_RESPONSES)
    if re.search(r'\b(what can you do|who are you|what is this)\b', q):
        return random.choice(CAPABILITY_RESPONSES)
    return random.choice(GREETING_RESPONSES)


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
    if is_general_query(query):
        return "general"
    rule_result = rule_based_route(query)
    return rule_result if rule_result else llm_route(query)


def _extract_ticker_from_text(text: str) -> str | None:
    """Core extraction logic - separated out so history lookup (added
    separately) can reuse it without recursion."""
    candidates = [c for c in TICKER_PATTERN.findall(text) if c in KNOWN_TICKERS]
    if candidates:
        return candidates[0]

    text_lower = text.lower()
    for name, ticker in COMPANY_TO_TICKER.items():
        if re.search(rf"\b{re.escape(name)}\b", text_lower):
            return ticker

    return None


def extract_ticker(query: str, history: list[dict] | None = None) -> str | None:
    """
    Extract a ticker from the current query. If none is found and
    conversation history is available, falls back to the most recently
    mentioned ticker - lets "what about their revenue?" work as a
    follow-up instead of failing with no ticker found.
    """
    ticker = _extract_ticker_from_text(query)
    if ticker:
        return ticker

    if history:
        for turn in reversed(history):
            prior_ticker = _extract_ticker_from_text(turn.get("question", ""))
            if prior_ticker:
                return prior_ticker

    return None


if __name__ == "__main__":
    test_queries = [
        "What is Apple's current stock price?",
        "What were Apple's risk factors in their latest 10-K?",
        "How does TSLA's current price compare to their reported revenue growth?",
        "What does the SEC require in a 10-K filing?",
        "great response from your side",   # should now be "general"
        "netflix is grew in 2024?",         # should now be "filings"
        "great",                             # should be "general"
    ]
    for q in test_queries:
        print(f"\nQ: {q}")
        print(f"Route: {route_query(q)}")
        print(f"Ticker: {extract_ticker(q)}")