"""
agent.py

Top-level entrypoint: routes the query, calls the right tool(s),
and returns the final answer along with routing metadata (so the
frontend can display the REAL route decision instead of re-guessing
from keywords).
"""

from tools.stock_price import get_stock_price, format_price_result
from router.query_router import route_query, extract_ticker, general_response
from retrieval.rag_chain import ask as query_filings

def answer_query(query: str) -> dict:
    """
    Returns a dict with the answer text plus routing metadata:
    {"answer": str, "route": str, "ticker": str | None}
    """
    route = route_query(query)

    if route == "general":
        return {
            "answer": general_response(query),
            "route": route,
            "ticker": None,
        }

    ticker = extract_ticker(query)

    if route == "live":
        if not ticker:
            return {
                "answer": "I couldn't find a ticker in your question. Try including one, e.g. 'AAPL current price'.",
                "route": route,
                "ticker": None,
            }
        return {
            "answer": format_price_result(get_stock_price(ticker)),
            "route": route,
            "ticker": ticker,
        }

    elif route == "filings":
        return {
        "answer": query_filings(query, ticker=ticker),
        "route": route,
        "ticker": ticker,
    }

    elif route == "both":
        live_part = format_price_result(get_stock_price(ticker)) if ticker else "No ticker found for live data."
        filing_part = query_filings(query, ticker=ticker)
        return {
        "answer": f"Live data:\n{live_part}\n\nFrom filings:\n{filing_part}",
        "route": route,
        "ticker": ticker,
        }

    return {
        "answer": "Sorry, I couldn't determine how to answer that.",
        "route": "unknown",
        "ticker": None,
    }
    
if __name__ == "__main__":
    test_queries = [
        "What is Apple's current stock price?",
        "What were Apple's risk factors in their latest 10-K?",
    ]
    for q in test_queries:
        print(f"\nQ: {q}")
        print("Routing and answering (filings questions can take 30-90s via Ollama)...")
        result = answer_query(q)
        print(f"Route: {result['route']}  |  Ticker: {result['ticker']}")
        print(f"A: {result['answer']}\n")