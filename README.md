# FinDocGPT — Agentic Financial Filing Intelligence System

A hybrid RAG + agentic system that answers questions over SEC filings (10-K/10-Q) and augments them with live market data, using a query router to decide which data source(s) a question actually needs — not a single-mode chatbot bolted onto a vector database.

**🔗 Live demo:** `https://findocgpt-j88a.onrender.com`
*(Free tier — may take 30-60s to wake up if idle for 15+ minutes)*

**Capabilities:**
- Natural-language Q&A over SEC filings, grounded with citations
- Live stock price lookups (no static/stale data)
- Hybrid answers combining both (e.g. "how does current price compare to reported revenue growth?")
- Structured financial metric extraction into comparison tables
- Custom evaluation harness measuring faithfulness, relevancy, and hallucination resistance

The LLM provider is abstracted behind a single client module (`llm_client.py`): **Ollama** for local development (fully free, fully offline), **Groq** for the deployed version (also free, hosted, no cold-start model loading). Embeddings, vector storage, and live market data are free/local in both environments — no OpenAI key required anywhere in the stack.

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
            │  (live price) │ │  (retrieval)   │
            └───────────────┘ └──────┬────────┘
                                      │
                              ┌───────▼────────┐
                              │  llm_client.py  │
                              │ Ollama (local)  │
                              │ Groq (deployed) │
                              └───────┬────────┘
                                      │
                         ┌────────────▼──────────┐
                         │  Grounded, cited        │
                         │  answer                 │
                         └────────────────────────┘
```

**Retrieval design:** filings are parsed and chunked *section-aware* (Risk Factors, MD&A, Financial Statements, etc. — detected via SEC's standardized "Item" headers), not split by fixed character count. Retrieval is filtered by metadata (ticker, form type, filing date, section) before semantic search runs, rather than relying on embedding similarity alone to find the right company/filing.

---

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| LLM | Ollama (local) / Groq (deployed) | Free either way, swapped via one config value |
| Embeddings | ChromaDB's ONNX-based default function (`all-MiniLM-L6-v2`) | Same model as sentence-transformers, lighter runtime — critical for fitting free-tier memory limits |
| Vector DB | ChromaDB | Metadata filtering + persistence; pre-built and committed to the repo so the deployed instance doesn't need to re-embed on every cold start |
| Live data | yfinance | Free, no API key |
| Orchestration | Custom router (rule-based + LLM fallback) | Faster/cheaper than always calling an LLM to route |
| Backend | FastAPI | Serves both the API and the static frontend from one process |
| Frontend | Custom HTML/CSS/JS | No framework overhead for a single-page terminal UI |
| Data source | SEC EDGAR (public API) | Free, official filing source |
| Hosting | Render (free tier) | Persistent service (not serverless) — required since retrieval needs a warm, resident vector store |

---

## Project Structure

```
findocgpt/
├── ingestion/          # SEC filing download, HTML parsing, section-aware chunking
├── retrieval/           # Embeddings, ChromaDB, RAG chain, structured metric extraction
├── tools/               # Live data tools (stock price)
├── router/               # Query routing (rule-based + LLM classification)
├── api/                  # FastAPI backend (also serves the frontend statically)
├── frontend/              # UI (single-page HTML/CSS/JS)
├── eval/                 # Evaluation harness (faithfulness, relevancy, hallucination tests)
├── llm_client.py          # Centralized LLM provider switch (Ollama ↔ Groq)
├── core_agent.py          # Top-level entrypoint tying router + tools + RAG together
├── chroma_db/              # Pre-built vector store (committed — see Deployment notes below)
└── data/
    ├── raw_filings/        # Downloaded SEC filings (gitignored)
    └── processed/          # Chunks, extracted metrics, evaluation results (gitignored)
```

---

## Setup (local development)

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
Open `http://127.0.0.1:8000` — FastAPI serves the frontend directly, no separate step needed.

---

## Deployment

Deployed on **Render's free tier**, with the LLM switched to **Groq** via `LLM_PROVIDER=groq` (set as an environment variable in Render's dashboard, never committed to the repo).

### Why this needed real changes, not just "upload and go"

Deploying surfaced three real bugs that never appeared in local testing — worth documenting since finding and fixing infrastructure bugs is as much a part of this project as the RAG logic itself:

1. **Frontend hardcoded to `localhost`** — the UI defaulted its API base URL to `http://127.0.0.1:8000`, which obviously breaks once served from a real domain. *Fixed* by defaulting to `window.location.origin` instead, so the frontend always calls whatever origin it's actually being served from.

2. **Out-of-memory crashes on the free tier (512MB limit)** — the original embedding setup loaded full PyTorch via `sentence-transformers`, which alone can exceed 512MB once FastAPI and ChromaDB are also running. *Fixed* by switching to ChromaDB's built-in ONNX-runtime-based embedding function — same underlying model, much lighter memory footprint, no `torch` dependency required at all.

3. **Ephemeral filesystem on free tier** — Render's free instances wipe any files written at runtime on every restart, which would mean re-embedding thousands of chunks from scratch on every cold start. *Fixed* by committing the pre-built `chroma_db/` directory to the repo itself — since it's part of the deployed image (not a runtime write), it survives every restart with zero rebuild cost.

---

## Example Queries

| Question | Route |
|---|---|
| "What is Apple's current stock price?" | `live` |
| "What were Apple's risk factors in their latest 10-K?" | `filings` |
| "How does Tesla's current price compare to their reported revenue growth?" | `both` |

---

## Evaluation

Rather than depending on the `ragas` library (which is built primarily around OpenAI), evaluation is implemented directly via LLM-as-judge prompting — keeping the entire stack free, whether pointed at local Ollama or deployed Groq. Two metrics, matching RAGAS's methodology, plus a custom hallucination check:

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

1. **Cross-company contamination**: a question about Amazon's cloud segment returned an answer that suddenly pivoted to discussing NVIDIA, because retrieval wasn't filtered by ticker. *Fixed* by scoping retrieval to the ticker already extracted by the router.
2. **Cross-filing-type blending for numeric queries**: "Microsoft's net income in their most recent 10-K" pulled narrative chunks from a 10-Q's Risk Factors section (which *discussed* a net income change) instead of the actual financial statement figure, producing a different number on different runs. *Fixed* by detecting metric-seeking questions and restricting retrieval to `Financial Statements`/`MD&A` sections specifically.

### Known limitations (documented honestly, not hidden)

- **LLM-judge scoring noise**: using a small local model as an evaluator produces some run-to-run scoring inconsistency on functionally identical answers — a known trade-off of a fully local/free eval pipeline.
- **Structured metric extraction unit errors**: the LLM-based financial metric extractor (Phase 3) occasionally misreads unit scale (e.g. raw dollars vs. millions) on dense tables. A plausibility-range validator catches gross errors but can't catch subtler wrong-but-plausible values — a production system would need cross-validation against SEC's structured XBRL data or a human review step.
- **Free-tier hosting trade-offs**: cold starts after 15 minutes of inactivity (~30-60s), and the memory ceiling that shaped the embedding-function choice above.

---

## What This Project Demonstrates

- Section-aware document chunking (not naive fixed-size splitting)
- Metadata-filtered retrieval (precision over pure semantic similarity)
- A genuine agent (tool-calling + routing), not just a RAG wrapper
- Structured output extraction with validation, not blind trust in LLM output
- A real evaluation methodology, including documenting the bugs it found
- A real deployment, including documenting the infrastructure bugs *that* surfaced
- Provider-agnostic LLM design — swapping from local Ollama to hosted Groq required editing one file
- Fully free at every layer, in both local and deployed environments