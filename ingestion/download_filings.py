"""
download_filings.py

Downloads 10-K (annual) and 10-Q (quarterly) filings for a list of companies
from SEC EDGAR, and saves the raw HTML to data/raw_filings/.

Design notes (industry practice):
- We save raw files to disk before parsing, so parsing can be re-run/debugged
  without re-hitting the network every time (parsing bugs are common;
  network calls are slow and rate-limited).
- Filenames encode metadata (ticker, filing type, fiscal year) so later
  pipeline stages don't need to re-fetch metadata to know what a file is.
- SEC EDGAR rate limit is ~10 requests/second - we throttle conservatively
  to stay well under that and avoid getting blocked.
"""

import os
import time
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

from cik_lookup import get_cik, get_ticker_to_cik_map

load_dotenv()

USER_AGENT = os.getenv("SEC_EDGAR_USER_AGENT")
HEADERS = {"User-Agent": USER_AGENT}

RAW_DATA_DIR = Path(__file__).parent.parent / "data" / "raw_filings"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_BASE_URL = "https://www.sec.gov/Archives/edgar/data"

REQUEST_DELAY_SECONDS = 0.2  # ~5 req/sec, safely under SEC's 10/sec limit


def get_filing_list(cik: str, form_types=("10-K", "10-Q"), limit_per_type=3) -> list[dict]:
    """
    Fetches the list of recent filings for a company (by CIK) and filters
    to the form types we care about.

    Returns a list of dicts with the info needed to construct the download URL:
    accession_number, filing_date, form, primary_document.
    """
    url = SUBMISSIONS_URL.format(cik=cik)
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    recent = data["filings"]["recent"]
    company_name = data.get("name", "unknown")

    filings = []
    counts = {ft: 0 for ft in form_types}

    # SEC returns parallel arrays (same index = same filing), not a list of objects
    for i, form in enumerate(recent["form"]):
        if form in form_types and counts[form] < limit_per_type:
            filings.append({
                "company_name": company_name,
                "form": form,
                "filing_date": recent["filingDate"][i],
                "accession_number": recent["accessionNumber"][i],
                "primary_document": recent["primaryDocument"][i],
            })
            counts[form] += 1

        if all(c >= limit_per_type for c in counts.values()):
            break

    return filings


def download_filing(cik: str, filing: dict, ticker: str) -> Path:
    """
    Downloads a single filing document and saves it to data/raw_filings/.
    Returns the local file path.
    """
    accession_no_dashes = filing["accession_number"].replace("-", "")
    doc_url = (
        f"{ARCHIVE_BASE_URL}/{int(cik)}/{accession_no_dashes}/"
        f"{filing['primary_document']}"
    )

    resp = requests.get(doc_url, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    # Filename encodes metadata for easy identification downstream
    filename = f"{ticker}_{filing['form']}_{filing['filing_date']}.html"
    out_path = RAW_DATA_DIR / filename
    out_path.write_bytes(resp.content)

    return out_path


def download_all(tickers: list[str], form_types=("10-K", "10-Q"), limit_per_type=3):
    """
    Main entry point: downloads filings for a list of tickers.
    Also writes a manifest.json summarizing what was downloaded, which
    Phase 2 (parsing) will read to know what files exist and their metadata.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ticker_map = get_ticker_to_cik_map()
    manifest = []

    for ticker in tickers:
        print(f"\n[{ticker}] Looking up CIK...")
        try:
            cik = get_cik(ticker, ticker_map)
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue

        print(f"[{ticker}] CIK={cik}. Fetching filing list...")
        filings = get_filing_list(cik, form_types, limit_per_type)
        time.sleep(REQUEST_DELAY_SECONDS)

        for filing in filings:
            print(f"  Downloading {filing['form']} filed {filing['filing_date']}...")
            try:
                local_path = download_filing(cik, filing, ticker)
                manifest.append({
                    "ticker": ticker,
                    "cik": cik,
                    "company_name": filing["company_name"],
                    "form": filing["form"],
                    "filing_date": filing["filing_date"],
                    "local_path": str(local_path),
                })
            except requests.HTTPError as e:
                print(f"    FAILED: {e}")
            time.sleep(REQUEST_DELAY_SECONDS)

    manifest_path = RAW_DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. {len(manifest)} filings downloaded. Manifest: {manifest_path}")

    return manifest


if __name__ == "__main__":
    # Start small - 5 well-known companies, latest 10-K + last 2 10-Qs each.
    # Expand this list once the pipeline is proven to work end-to-end.
    COMPANIES = ["AAPL", "MSFT", "TSLA", "AMZN", "NVDA"]

    download_all(COMPANIES, form_types=("10-K", "10-Q"), limit_per_type=2)