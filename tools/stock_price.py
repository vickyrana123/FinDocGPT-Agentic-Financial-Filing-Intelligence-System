"""
stock_price.py

Live stock price tool with two data sources:
1. yfinance (primary) - free, no key, but scrapes Yahoo Finance and gets
   rate-limited, especially from shared cloud IPs like Render's free tier.
2. Finnhub (fallback) - free tier, official API with a key, used only when
   yfinance fails. Redundant data sources for a "live" feature is the
   right pattern regardless - don't depend on a single unofficial source.

Includes a cache to reduce how often either source gets hit at all.
"""

import os
import time
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB_URL = "https://finnhub.io/api/v1/quote"

CACHE_TTL_SECONDS = 300  # 5 min - live-to-the-second precision isn't needed here,
                          # and a longer cache meaningfully reduces rate-limit risk
_price_cache: dict[str, tuple[float, dict]] = {}

MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 2


def _try_yfinance(ticker: str) -> dict | None:
    """Returns a result dict on success, or None to signal 'try the fallback'."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            price = info.get("currentPrice") or info.get("regularMarketPrice")
            previous_close = info.get("previousClose")

            if price is None:
                return None  # not a rate-limit error, just no data - let fallback try

            change = round(price - previous_close, 2) if previous_close else None
            change_pct = round((change / previous_close) * 100, 2) if change and previous_close else None

            return {
                "ticker": ticker,
                "company_name": info.get("longName", ticker),
                "price": price,
                "currency": info.get("currency", "USD"),
                "change": change,
                "change_pct": change_pct,
                "previous_close": previous_close,
                "source": "yfinance",
            }
        except Exception as e:
            is_rate_limit = "too many requests" in str(e).lower() or "rate limit" in str(e).lower()
            if is_rate_limit and attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            return None  # give up on yfinance, let the fallback try


def _try_finnhub(ticker: str) -> dict | None:
    if not FINNHUB_API_KEY:
        return None

    try:
        response = requests.get(
            FINNHUB_URL,
            params={"symbol": ticker, "token": FINNHUB_API_KEY},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        price = data.get("c")  # current price
        previous_close = data.get("pc")

        if not price:
            return None

        change = round(price - previous_close, 2) if previous_close else None
        change_pct = round((change / previous_close) * 100, 2) if change and previous_close else None

        return {
            "ticker": ticker,
            "company_name": ticker,  # Finnhub's quote endpoint doesn't include name
            "price": price,
            "currency": "USD",
            "change": change,
            "change_pct": change_pct,
            "previous_close": previous_close,
            "source": "finnhub",
        }
    except Exception:
        return None


def get_stock_price(ticker: str) -> dict:
    ticker = ticker.upper()

    cached = _price_cache.get(ticker)
    if cached and (time.time() - cached[0]) < CACHE_TTL_SECONDS:
        return cached[1]

    result = _try_yfinance(ticker) or _try_finnhub(ticker)

    if result is None:
        result = {"error": f"Could not retrieve live price for '{ticker}' from any source."}

    _price_cache[ticker] = (time.time(), result)
    return result


def format_price_result(result: dict) -> str:
    if "error" in result:
        return result["error"]

    if result["change"] is not None:
        direction = "up" if result["change"] >= 0 else "down"
        return (
            f"{result['company_name']} ({result['ticker']}): "
            f"{result['price']} {result['currency']} "
            f"({direction} {abs(result['change'])}, {abs(result['change_pct'])}% "
            f"from previous close of {result['previous_close']})"
        )

    return f"{result['company_name']} ({result['ticker']}): {result['price']} {result['currency']}"


if __name__ == "__main__":
    for t in ["AAPL", "MSFT", "TSLA"]:
        result = get_stock_price(t)
        print(format_price_result(result), f"  [source: {result.get('source', 'n/a')}]")