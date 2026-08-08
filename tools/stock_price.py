"""
stock_price.py

A live data "tool" the agent can call for current stock price info.
"""

import yfinance as yf


def get_stock_price(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        previous_close = info.get("previousClose")
        currency = info.get("currency", "USD")
        company_name = info.get("longName", ticker)

        if price is None:
            return {"error": f"Could not retrieve live price for '{ticker}'."}

        change = None
        change_pct = None
        if previous_close:
            change = round(price - previous_close, 2)
            change_pct = round((change / previous_close) * 100, 2)

        return {
            "ticker": ticker.upper(),
            "company_name": company_name,
            "price": price,
            "currency": currency,
            "change": change,
            "change_pct": change_pct,
            "previous_close": previous_close,
        }
    except Exception as e:
        return {"error": f"Failed to fetch price for '{ticker}': {e}"}


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