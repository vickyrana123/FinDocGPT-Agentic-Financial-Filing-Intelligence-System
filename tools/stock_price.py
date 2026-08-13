"""
stock_price.py

A live data "tool" the agent can call for current stock price info.

Includes a short-lived in-memory cache and a small retry with backoff,
since yfinance scrapes Yahoo Finance's public endpoints (no official API
key), which rate-limits aggressively - especially from cloud provider IPs
like Render's, which are shared across many apps. Caching identical
ticker requests within a short window is the real fix, not just retrying
harder against a limit that's often IP-wide, not just per-request.
"""

import time
import yfinance as yf

CACHE_TTL_SECONDS = 60
_price_cache: dict[str, tuple[float, dict]] = {}  # ticker -> (timestamp, result)

MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 2


def get_stock_price(ticker: str) -> dict:
    ticker = ticker.upper()

    cached = _price_cache.get(ticker)
    if cached and (time.time() - cached[0]) < CACHE_TTL_SECONDS:
        return cached[1]

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            price = info.get("currentPrice") or info.get("regularMarketPrice")
            previous_close = info.get("previousClose")
            currency = info.get("currency", "USD")
            company_name = info.get("longName", ticker)

            if price is None:
                result = {"error": f"Could not retrieve live price for '{ticker}'."}
                _price_cache[ticker] = (time.time(), result)
                return result

            change = None
            change_pct = None
            if previous_close:
                change = round(price - previous_close, 2)
                change_pct = round((change / previous_close) * 100, 2)

            result = {
                "ticker": ticker,
                "company_name": company_name,
                "price": price,
                "currency": currency,
                "change": change,
                "change_pct": change_pct,
                "previous_close": previous_close,
            }
            _price_cache[ticker] = (time.time(), result)
            return result

        except Exception as e:
            last_error = e
            is_rate_limit = "too many requests" in str(e).lower() or "rate limit" in str(e).lower()
            if is_rate_limit and attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            break

    error_msg = str(last_error).lower()
    if "too many requests" in error_msg or "rate limit" in error_msg:
        friendly = f"Live price for '{ticker}' is temporarily rate-limited by the data provider. Please try again in a minute."
    else:
        friendly = f"Failed to fetch price for '{ticker}': {last_error}"

    result = {"error": friendly}
    _price_cache[ticker] = (time.time(), result)  # cache the error too, briefly - avoids hammering during an active rate-limit window
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
        print(format_price_result(result))