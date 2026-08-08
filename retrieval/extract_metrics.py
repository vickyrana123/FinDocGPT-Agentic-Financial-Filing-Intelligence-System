"""
extract_metrics.py

Extracts structured financial metrics (Revenue, Net Income, EPS) from
SEC filings using the LLM, and aggregates them into a clean Pandas
DataFrame - the "structured extraction" mode of FinDocGPT.
"""

import json
import re
from pathlib import Path
import pandas as pd
import requests
import chromadb
from chromadb.utils import embedding_functions

import sys
sys.path.append(str(Path(__file__).parent))


def call_ollama_for_extraction(prompt: str, timeout: int = 400) -> str:
    """
    Uses the 3B model (llama3.2) - testing showed the smaller 1B model is
    too weak for numeric extraction from dense financial text; it returns
    valid JSON but with everything null instead of actually finding the
    values. The 3B model extracts real numbers correctly but is slower,
    so we compensate by shrinking context size (fewer chunks, tighter
    truncation) rather than downgrading the model.
    """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "num_predict": 200,
                "temperature": 0.1,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["response"]


def call_with_retry(prompt: str, retries: int = 1) -> str:
    """
    Retries once on timeout - the first call in a batch is often slow due
    to model load time, so a single retry after a timeout frequently
    succeeds once the model is warm.
    """
    last_error = None
    for attempt in range(retries + 1):
        try:
            return call_ollama_for_extraction(prompt)
        except requests.exceptions.ReadTimeout as e:
            last_error = e
            print(f"    [RETRY] Attempt {attempt + 1} timed out, "
                  f"{'retrying...' if attempt < retries else 'giving up.'}")
    raise last_error


CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "sec_filings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_collection = None


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        _collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
    return _collection


EXTRACTION_PROMPT_TEMPLATE = """You are a financial data extraction assistant. \
Extract the following metrics from the CONTEXT below, which is from {ticker}'s \
{form} filed {filing_date}.

Metrics to extract:
- total_revenue (in millions USD, as a number only, no symbols/commas)
- net_income (in millions USD, as a number only)
- eps_diluted (diluted earnings per share, as a number only)
- fiscal_year (the fiscal year this data refers to)

Respond with ONLY valid JSON in this exact format, nothing else. \
Do not add any explanation, markdown formatting, or text before or after the JSON:
{{
  "total_revenue": <number or null>,
  "net_income": <number or null>,
  "eps_diluted": <number or null>,
  "fiscal_year": <number or null>
}}

If a metric is not found in the context, use null. Do not guess or estimate.

CONTEXT:
{context}

JSON:"""


def extract_json_from_response(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def extract_metrics_for_filing(ticker: str, form: str, filing_date: str) -> dict:
    collection = get_collection()

    where_filter = {
        "$and": [
            {"ticker": ticker},
            {"form": form},
            {"filing_date": filing_date},
            {"$or": [
                {"section": "Financial Statements"},
                {"section": "MD&A"},
            ]},
        ]
    }

    results = collection.query(
        query_texts=["total revenue net sales net income earnings per share diluted"],
        n_results=4,
        where=where_filter,
    )

    docs = results["documents"][0]

    if not docs:
        return {
            "ticker": ticker, "form": form, "filing_date": filing_date,
            "total_revenue": None, "net_income": None,
            "eps_diluted": None, "fiscal_year": None,
            "extraction_status": "no_relevant_chunks_found",
            "flags": "",
        }

    context = "\n\n".join(docs)

    MAX_CONTEXT_CHARS = 2500
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        ticker=ticker, form=form, filing_date=filing_date, context=context
    )

    try:
        raw_response = call_with_retry(prompt)
    except requests.exceptions.ReadTimeout:
        print(f"    [WARN] Timed out extracting {ticker} {form} ({filing_date}) - skipping.")
        return {
            "ticker": ticker, "form": form, "filing_date": filing_date,
            "total_revenue": None, "net_income": None,
            "eps_diluted": None, "fiscal_year": None,
            "extraction_status": "timeout",
            "flags": "",
        }

    parsed = extract_json_from_response(raw_response)

    if parsed is None:
        print(f"    [DEBUG] Raw LLM response that failed to parse:\n    {raw_response[:300]}\n")
        return {
            "ticker": ticker, "form": form, "filing_date": filing_date,
            "total_revenue": None, "net_income": None,
            "eps_diluted": None, "fiscal_year": None,
            "extraction_status": "failed_to_parse_json",
            "flags": "",
        }

    parsed.update({
        "ticker": ticker, "form": form, "filing_date": filing_date,
        "extraction_status": "success",
    })

    flags = []
    revenue = parsed.get("total_revenue")
    net_income = parsed.get("net_income")

    if revenue is not None and not (1_000 <= revenue <= 1_000_000):
        flags.append("revenue_out_of_plausible_range")
    if net_income is not None and not (-200_000 <= net_income <= 500_000):
        flags.append("net_income_out_of_plausible_range")

    parsed["flags"] = ", ".join(flags) if flags else ""

    return parsed


def build_metrics_table(filings: list[dict]) -> pd.DataFrame:
    rows = []
    for f in filings:
        print(f"Extracting metrics for {f['ticker']} {f['form']} ({f['filing_date']})...")
        try:
            row = extract_metrics_for_filing(f["ticker"], f["form"], f["filing_date"])
        except Exception as e:
            print(f"    [ERROR] Unexpected failure: {e} - skipping this filing.")
            row = {
                "ticker": f["ticker"], "form": f["form"], "filing_date": f["filing_date"],
                "total_revenue": None, "net_income": None,
                "eps_diluted": None, "fiscal_year": None,
                "extraction_status": "unexpected_error",
                "flags": "",
            }
        rows.append(row)

    df = pd.DataFrame(rows)
    column_order = [
        "ticker", "form", "filing_date", "fiscal_year",
        "total_revenue", "net_income", "eps_diluted",
        "extraction_status", "flags",
    ]
    return df[[c for c in column_order if c in df.columns]]


if __name__ == "__main__":
    manifest_path = Path(__file__).parent.parent / "data" / "raw_filings" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    ten_ks = [
        {"ticker": m["ticker"], "form": m["form"], "filing_date": m["filing_date"]}
        for m in manifest if m["form"] == "10-K"
    ]

    print(f"Found {len(ten_ks)} 10-K filings to extract from.\n")

    df = build_metrics_table(ten_ks)

    print("\n" + "=" * 70)
    print("EXTRACTED METRICS TABLE")
    print("=" * 70)
    print(df.to_string(index=False))

    out_path = Path(__file__).parent.parent / "data" / "processed" / "metrics_table.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")