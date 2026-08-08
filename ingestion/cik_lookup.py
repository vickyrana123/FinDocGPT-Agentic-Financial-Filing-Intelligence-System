"""
cik_lookup.py

SEC EDGAR identifies companies by CIK (Central Index Key), not by name or
ticker. This module downloads SEC's official ticker->CIK mapping once and
lets us look up any company by ticker symbol.

Why this matters (industry note):
Every downstream ingestion step needs the CIK, so this is the entry point
of the whole pipeline. Getting this wrong (e.g. hardcoding CIKs) makes the
system brittle and non-scalable to new companies.
"""

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

USER_AGENT = os.getenv("SEC_EDGAR_USER_AGENT")
if not USER_AGENT:
    raise EnvironmentError(
        "SEC_EDGAR_USER_AGENT not set. Add it to your .env file, "
        "e.g. 'Your Name your_email@example.com'"
    )

HEADERS = {"User-Agent": USER_AGENT}

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


def get_ticker_to_cik_map() -> dict:
    """
    Downloads SEC's full ticker -> CIK mapping.
    Returns a dict like {"AAPL": "0000320193", "MSFT": "0000789019", ...}
    CIKs are zero-padded to 10 digits, which is the format EDGAR's
    filing endpoints expect.
    """
    resp = requests.get(TICKER_MAP_URL, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    # raw format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    ticker_to_cik = {
        entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
        for entry in raw.values()
    }
    return ticker_to_cik


def get_cik(ticker: str, ticker_map: dict | None = None) -> str:
    """
    Look up a single company's CIK by ticker symbol.
    Pass in a pre-fetched ticker_map if looking up multiple companies,
    to avoid re-downloading the full list each time.
    """
    if ticker_map is None:
        ticker_map = get_ticker_to_cik_map()

    ticker = ticker.upper()
    if ticker not in ticker_map:
        raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR company list.")

    return ticker_map[ticker]


if __name__ == "__main__":
    # Quick manual test - run this file directly to sanity check it works
    companies = ["AAPL", "MSFT", "TSLA", "AMZN", "NVDA"]
    ticker_map = get_ticker_to_cik_map()

    print("Ticker -> CIK lookup:")
    for t in companies:
        cik = get_cik(t, ticker_map)
        print(f"  {t}: {cik}")