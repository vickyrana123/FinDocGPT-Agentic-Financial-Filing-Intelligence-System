"""
main.py

FastAPI backend exposing the FinDocGPT agent over HTTP.
Run with: uvicorn api.main:app --reload
"""

import sys
from pathlib import Path
from fastapi.staticfiles import StaticFiles

sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core_agent import answer_query

app = FastAPI(
    title="FinDocGPT API",
    description="Agentic financial document + live market data assistant",
    version="1.0.0",
)

@app.on_event("startup")
def warm_up_ollama():
    """
    Local-only optimization: warms Ollama's model into memory before the
    first user request. Skipped entirely when deployed with a hosted
    provider (Groq), since there's no local model-loading cost to avoid.
    """
    import os
    if os.getenv("LLM_PROVIDER", "ollama") != "ollama":
        print("Using hosted LLM provider - skipping local warm-up.")
        return

    import requests
    try:
        print("Warming up Ollama model...")
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2", "prompt": "Hello", "stream": False, "keep_alive": "30m"},
            timeout=120,
        )
        print("Ollama model warmed up and ready.")
    except Exception as e:
        print(f"Warm-up failed (Ollama may not be running yet): {e}")
        
# Allow requests from any frontend origin for now.
# When deploying to production, restrict this to your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    history: list[dict] | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    route: str
    ticker: str | None = None


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = answer_query(question, history=request.history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")
    
    return QueryResponse(
        question=question,
        answer=result["answer"],
        route=result["route"],
        ticker=result["ticker"],
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer = answer_query(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

    return QueryResponse(question=question, answer=answer)

# Serve the frontend directly so the whole app launches with one command.
# Must be mounted LAST - Starlette checks routes in the order they were
# added, so /health and /ask above still get matched first; this mount
# only catches everything else (i.e. the frontend files).
app.mount(
    "/",
    StaticFiles(directory=str(Path(__file__).parent.parent / "frontend"), html=True),
    name="frontend",
)