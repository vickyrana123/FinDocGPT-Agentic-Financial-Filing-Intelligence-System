# FinDocGPT — Agentic Financial Filing Intelligence System

A hybrid RAG + agentic system that answers questions over SEC filings (10-K/10-Q) and augments them with live market data, using a query router to decide which data source(s) a question actually needs — not a single-mode chatbot bolted onto a vector database.

**Live capabilities:**
- Natural-language Q&A over SEC filings, grounded with citations
- Live stock price lookups (no static/stale data)
- Hybrid answers combining both (e.g. "how does current price compare to reported revenue growth?")
- Structured financial metric extraction into comparison tables
- Custom evaluation harness measuring faithfulness, relevancy, and hallucination resistance

Everything runs **fully local and free** — local LLM (Ollama), local embeddings (sentence-transformers), local vector store (ChromaDB), no OpenAI key required anywhere in the stack.

---

## Architecture

```
                         ┌─────────────────────┐
                         │   User Question      │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │   Query Router        │
                         │  (rules → LLM fallback)│
                         └──────────┬───────────┘
                    ┌───────────────┼───────────────┐
                    │               │               │
              ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
              │   live     │   │  filings   │   │   both     │
              └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
                    │               │               │
            ┌───────▼──────┐ ┌──────▼───────┐       │
            │  yfinance     │ │  ChromaDB     │◄──────┘
            │  (live price) │ │  + Ollama     │
            └───────────────┘ │  (RAG + gen)  │
                               └───────────────┘
                                    │
                         ┌──────────▼───────────┐
                         │  Grounded, cited      │
                         │  answer               │
                         └──────────────────────┘
```

**Retrieval design:** filings are parsed and chunked *section-aware* (Risk Factors, MD&A, Financial Statements, etc. — detected via SEC's standardized "Item" headers), not split by fixed character count. Retrieval is filtered by metadata (ticker, form type, filing date, section) before semantic search runs, rather than relying on embedding similarity alone to find the right company/filing.

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| LLM | Ollama (Llama 3.2) | Local, free, no API key |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Local, fast, no API key |
| Vector DB | ChromaDB | Metadata filtering + persistence |
| Live data | yfinance | Free, no API key |
| Orchestration | Custom router (rule-based + LLM fallback) | Faster/cheaper than always calling an LLM to route |
| Backend | FastAPI | Async-friendly, typed |
| Frontend | Custom HTML/CSS/JS | No framework overhead for a single-page terminal UI |
| Data source | SEC EDGAR (public API) | Free, official filing source |

---

## Project Structure

```
findocgpt/
├── ingestion/        # SEC filing download, HTML parsing, section-aware chunking
├── retrieval/         # Embeddings, ChromaDB, RAG chain, structured metric extraction
├── tools/             # Live data tools (stock price)
├── router/             # Query routing (rule-based + LLM classification)
├── api/                # FastAPI backend
├── ui/                 # Frontend
├── eval/               # Evaluation harness (faithfulness, relevancy, hallucination tests)
├── data/
│   ├── raw_filings/    # Downloaded SEC filings (gitignored)
│   └── processed/      # Chunks, extracted metrics, evaluation results (gitignored)
└── core_agent.py       # Top-level entrypoint tying router + tools + RAG together
```

---

## Setup

```bash
git clone <your-repo-url>
cd findocgpt
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env           # add your SEC EDGAR user-agent email
ollama pull llama3.2
```

**Build the knowledge base (one-time):**
```bash
cd ingestion
python download_filings.py
python parse_and_chunk.py
cd ../retrieval
python build_vectorstore.py
```

**Run it:**
```bash
cd api
uvicorn main:app --reload
```
Open `ui/index.html` in a browser.

---

## Example Queries

| Question | Route |
|---|---|
| "What is Apple's current stock price?" | `live` |
| "What were Apple's risk factors in their latest 10-K?" | `filings` |
| "How does Tesla's current price compare to their reported revenue growth?" | `both` |

---

## Evaluation

Rather than depending on the `ragas` library (which is built primarily around OpenAI), evaluation is implemented directly via LLM-as-judge prompting against the same local Ollama model — keeping the entire stack free and local, consistent with the rest of the project. Two metrics, matching RAGAS's methodology:

- **Faithfulness** — does the answer only make claims supported by the retrieved context?
- **Answer relevancy** — does the answer actually address the question asked?
- **Hallucination resistance** — for questions the filings don't cover, does the system correctly decline instead of fabricating an answer?

**Results (9-question test set: 5 factual, 4 out-of-scope):**

```
Factual questions:
  Average faithfulness: ~80 / 100
  Average relevancy:    ~76 / 100

Hallucination-check questions:
  Correctly declined: 4/4 (100%)
```

### Real bugs found and fixed during evaluation

The evaluation process surfaced actual retrieval bugs — this is the point of building an eval harness, and it's worth documenting the process, not just the final score:

1. **Cross-company contamination**: a question about Amazon's cloud segment returned an answer that suddenly pivoted to discussing NVIDIA, because retrieval wasn't filtered by ticker. *Fixed* by scoping retrieval to the ticker already extracted by the router.
2. **Cross-filing-type blending for numeric queries**: "Microsoft's net income in their most recent 10-K" pulled narrative chunks from a 10-Q's Risk Factors section (which *discussed* a net income change) instead of the actual financial statement figure, producing a different number on different runs. *Fixed* by detecting metric-seeking questions and restricting retrieval to `Financial Statements`/`MD&A` sections specifically.

### Known limitations (documented honestly, not hidden)

- **LLM-judge scoring noise**: using a 3B local model as an evaluator (rather than a larger frontier model) produces some run-to-run scoring inconsistency on functionally identical answers. This is a known trade-off of a fully local eval pipeline.
- **Structured metric extraction unit errors**: the LLM-based financial metric extractor (Phase 3) occasionally misreads unit scale (e.g. raw dollars vs. millions) on dense tables. A plausibility-range validator catches gross errors but can't catch subtler wrong-but-plausible values — a production system would need cross-validation against SEC's structured XBRL data or a human review step.
- **Small local models require more careful prompting** than larger models to get consistent output; this shaped several design decisions (e.g. capping output length, lowering temperature for factual queries, strict JSON-only extraction prompts).

---

## What This Project Demonstrates

- Section-aware document chunking (not naive fixed-size splitting)
- Metadata-filtered retrieval (precision over pure semantic similarity)
- A genuine agent (tool-calling + routing), not just a RAG wrapper
- Structured output extraction with validation, not blind trust in LLM output
- A real evaluation methodology, including documenting the bugs it found
- Fully local, free, reproducible — no paid API dependencies anywhere in the stack