"""
main.py

FastAPI backend exposing the FinDocGPT agent over HTTP.
Run with: uvicorn api.main:app --reload
"""

import sys
from pathlib import Path

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
    Sends a trivial prompt to Ollama on server startup so the model is
    already loaded into memory before the first real user question
    arrives. Without this, the first request after starting the server
    pays a slow model-load cost (which is what just caused the timeout).
    """
    import requests
    try:
        print("Warming up Ollama model...")
        requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": "Hello",
                "stream": False,
                "keep_alive": "30m",
            },
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
        result = answer_query(question)
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