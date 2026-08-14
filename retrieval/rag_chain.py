"""
rag_chain.py

Combines retrieval (ChromaDB) with generation (local LLM via Ollama) to
produce grounded, cited answers to questions about SEC filings.
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from query_test import search

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from llm_client import call_llm

import requests

METRIC_KEYWORDS = [
    "net income", "revenue", "total revenue", "net sales", "earnings per share",
    "eps", "diluted eps", "operating income", "gross margin", "total assets",
    "cash flow",
]

def is_metric_query(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in METRIC_KEYWORDS)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

SYSTEM_PROMPT = """You are a financial document analyst assistant. You answer \
questions ONLY using the provided context extracted from SEC filings (10-K/10-Q).

Rules you must follow:
1. Only use information present in the CONTEXT below. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say clearly: \
"This information is not available in the provided filing excerpts."
3. Write your answer as clear, natural prose — as a human analyst would explain it \
in a report. Do NOT interrupt sentences with inline citation tags. Do NOT copy \
fragments of the source text verbatim if they are incomplete sentences; restate \
the idea fully and coherently in your own words.
4. After your full answer, add a blank line, then a "Sources:" section listing \
each source you used on its own line in the format: [Ticker, Form, Filing Date, Section]
5. Be concise and factual. Do not speculate or extrapolate beyond what's stated.

Example of the expected format:

Apple faces several supply chain risks, including potential shortages and price \
increases for key components, which could materially affect its business and \
financial results. The company also faces risks from design and manufacturing \
defects that could harm its reputation.

Sources:
[AAPL, 10-K, 2025-10-31, Risk Factors]
"""

def build_context(chunks: list[str], metadatas: list[dict]) -> str:
    parts = []
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
        source_tag = f"[{meta['ticker']}, {meta['form']}, {meta['filing_date']}, {meta['section']}]"
        parts.append(f"--- Source {i+1} {source_tag} ---\n{chunk}")
    return "\n\n".join(parts)

def call_ollama(prompt: str) -> str:
    """Delegates to the centralized llm_client, which switches between
    Ollama (local dev) and Groq (deployed) based on LLM_PROVIDER."""
    return call_llm(prompt, temperature=0.2, max_tokens=800, timeout=400)

def ask_with_context(question: str, n_results: int = 5, ticker: str = None, section: str = None) -> dict:
    """
    Same RAG pipeline as ask(), but also returns the raw retrieved chunks
    - needed for evaluation (faithfulness scoring requires comparing the
    answer against the actual context it was generated from, not just
    the final text).
    """

    if section is None and ticker and is_metric_query(question):
        section = ["Financial Statements", "MD&A"]
        n_results = max(n_results, 6)

    results = search(question, n_results=n_results, ticker=ticker, section=section)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        return {
            "answer": "No relevant filing content was found for this question.",
            "contexts": [],
        }

    # Relevance gate: ChromaDB always returns its top-k nearest chunks,
    # even if none are actually relevant to the question - it has no
    # concept of "not relevant enough," only "closest available." Without
    # this check, a genuinely off-topic or nonsensical query still gets
    # handed real (but irrelevant) chunks, and the LLM will often just
    # summarize whatever it received instead of recognizing the mismatch.
    # Lower distance = more similar. Threshold calibrated empirically:
    # genuinely relevant queries scored 0.63-0.78 distance, genuinely
    # irrelevant/off-topic queries scored 1.53-1.59 - a wide, clean gap,
    # so 1.0 sits safely in the middle with margin on both sides.
    RELEVANCE_THRESHOLD = 1.0
    relevant_pairs = [
        (doc, meta) for doc, meta, dist in zip(docs, metas, distances)
        if dist < RELEVANCE_THRESHOLD
    ]

    if not relevant_pairs:
        return {
            "answer": "This information is not available in the provided filing excerpts.",
            "contexts": [],
        }

    docs = [d for d, m in relevant_pairs]
    metas = [m for d, m in relevant_pairs]

    context = build_context(docs, metas)

    full_prompt = f"""{SYSTEM_PROMPT}

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    answer = call_ollama(full_prompt)
    return {"answer": answer, "contexts": docs}

def ask(question: str, n_results: int = 5, ticker: str = None, section: str = None) -> str:
    """
    Full RAG pipeline: retrieve relevant chunks, build a grounded prompt,
    generate an answer via Ollama. Thin wrapper around ask_with_context()
    for callers that only need the answer text.
    """
    return ask_with_context(question, n_results, ticker, section)["answer"]

if __name__ == "__main__":
    print("FinDocGPT - RAG test (type 'quit' to exit)\n")

    test_questions = [
        "What are Apple's main supply chain risks?",
        "What was Apple's revenue growth in the latest 10-K?",
    ]

    for q in test_questions:
        print(f"\nQ: {q}")
        print("Thinking... (this can take 30-90 seconds on CPU, please wait)")
        answer = ask(q)
        print(f"A: {answer}")
        print("-" * 70)

    while True:
        q = input("\nAsk a question (or 'quit'): ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue
        print("Thinking...")
        answer = ask(q)
        print(f"\nA: {answer}")