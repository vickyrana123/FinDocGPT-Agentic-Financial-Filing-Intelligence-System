"""
query_test.py

Standalone script to test retrieval quality against the ChromaDB
vectorstore - BEFORE we add LLM generation on top.
"""

from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "sec_filings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def search(query: str, n_results: int = 5, ticker: str = None, section=None):
    """
    section can be a single string (e.g. "Risk Factors") or a list of
    strings (e.g. ["Financial Statements", "MD&A"]) to match any of them.
    """
    collection = get_collection()

    conditions = []
    if ticker:
        conditions.append({"ticker": ticker})
    if section:
        if isinstance(section, list):
            conditions.append({"section": {"$in": section}})
        else:
            conditions.append({"section": section})

    if len(conditions) == 1:
        where_filter = conditions[0]
    elif len(conditions) > 1:
        where_filter = {"$and": conditions}
    else:
        where_filter = None

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter,
    )
    return results


def print_results(query: str, results: dict):
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"{'='*70}")

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        print("No results found.")
        return

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        relevance = max(0, (1 - dist)) * 100
        print(f"\n[{i+1}] Relevance: {relevance:.1f}%  |  "
              f"{meta['ticker']} {meta['form']} ({meta['filing_date']})  |  "
              f"Section: {meta['section']}")
        print(f"    {doc[:300]}...")


if __name__ == "__main__":
    results = search("What are the main risks related to supply chain?")
    print_results("What are the main risks related to supply chain?", results)

    results = search(
        "revenue growth and financial performance",
        ticker="AAPL",
        section="MD&A",
    )
    print_results("[AAPL, MD&A only] revenue growth and financial performance", results)

    results = search("What is the CEO's favorite color?")
    print_results("What is the CEO's favorite color?", results)