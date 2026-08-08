"""
run_evaluation.py

Evaluates FinDocGPT's filings RAG pipeline using two RAGAS-style metrics,
implemented directly via LLM-as-judge prompting against the local Ollama
model (rather than the `ragas` package, to avoid its OpenAI-centric
dependency chain and keep the whole stack local/free).

Revision notes (from a real first run):
1. Faithfulness prompt now explicitly treats a correct "not available"
   refusal as maximally faithful, not unsupported - a refusal makes zero
   claims, so by definition it can't contain unsupported ones. The first
   version of this prompt scored refusals as 0, which was a bug in the
   evaluation, not the RAG system.
2. Hallucination-check detection now uses an LLM judge classification
   instead of exact-substring matching, since the model doesn't always
   phrase a refusal identically - substring matching was flipping
   True/False across runs on the same question due to paraphrasing.
3. Full raw answer text is now printed to console (not just the judge's
   paraphrase) so results can be independently verified, not just trusted.
"""

import re
import json
import requests
from pathlib import Path
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent.parent / "retrieval"))
from rag_chain import ask_with_context

from eval_dataset import TEST_QUESTIONS

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

def call_judge(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.0,
                "num_predict": 100,
                "num_ctx": 4096,
            },
        },
        timeout=240,
    )
    response.raise_for_status()
    return response.json()["response"]

def parse_score(judge_response: str) -> int | None:
    match = re.search(r"SCORE:\s*(\d+)", judge_response, re.IGNORECASE)
    return int(match.group(1)) if match else None

def score_faithfulness(answer: str, contexts: list[str]) -> tuple[int | None, str]:
    context_block = "\n\n".join(contexts)
    prompt = f"""You are evaluating whether an AI-generated ANSWER is faithful to \
the provided CONTEXT - meaning every factual claim in the answer is actually \
supported by the context, with nothing fabricated or added from outside knowledge.

IMPORTANT: If the ANSWER declines to answer or states that the information is \
not available in the context, this counts as FULLY FAITHFUL (score 95-100) - \
a refusal makes no claims at all, so it cannot contain unsupported ones. Only \
score low if the answer makes specific factual claims that the context does \
not support.

CONTEXT:
{context_block}

ANSWER:
{answer}

Rate faithfulness from 0 to 100.

Respond in exactly this format:
SCORE: <number>
REASON: <one sentence>"""

    response = call_judge(prompt)
    score = parse_score(response)
    return score, response.strip()

def score_relevancy(question: str, answer: str) -> tuple[int | None, str]:
    prompt = f"""You are evaluating whether an ANSWER actually addresses the \
QUESTION asked - not whether it's factually correct, just whether it's on-topic \
and directly responsive. A clear, honest "this information is not available" \
response IS relevant if it directly addresses the question's topic, even \
though it doesn't provide the requested fact.

QUESTION:
{question}

ANSWER:
{answer}

Rate relevancy from 0 to 100.

Respond in exactly this format:
SCORE: <number>
REASON: <one sentence>"""

    response = call_judge(prompt)
    score = parse_score(response)
    return score, response.strip()

def classify_declined(question: str, answer: str) -> bool:
    """
    Replaces brittle exact-substring matching with an LLM judge classification.
    The model doesn't always phrase a refusal identically (e.g. sometimes
    "not available in the provided filing excerpts", sometimes "I don't have
    that information", sometimes "the filings do not discuss this") - a
    classification prompt handles paraphrasing that a fixed string can't.
    """
    prompt = f"""Did the ANSWER decline to answer because the information isn't \
available in the filings, OR did it provide a substantive answer/claim (even a \
speculative or fabricated one)?

QUESTION: {question}

ANSWER: {answer}

Respond with exactly one word: DECLINED or ANSWERED."""

    response = call_judge(prompt).strip().upper()
    return "DECLINED" in response

def run_evaluation():
    rows = []

    for item in TEST_QUESTIONS:
        print(f"\n[{item['id']}] ({item['category']}) {item['question']}")
        result = ask_with_context(item["question"], ticker=item.get("ticker"))        
        answer = result["answer"]
        contexts = result["contexts"]

        print(f"    [RAW ANSWER] {answer}")

        if item["category"] == "hallucination_check":
            try:
                correctly_declined = classify_declined(item["question"], answer)
            except requests.exceptions.ReadTimeout:
                print("    [WARN] Judge timed out - recording as unscored, continuing batch.")
                correctly_declined = None
            print(f"    Correctly declined: {correctly_declined}")
        else:
            if not contexts:
                print("    [WARN] No context retrieved - skipping scoring.")
                rows.append({
                    "id": item["id"], "category": item["category"], "question": item["question"],
                    "answer": answer[:500], "faithfulness": None, "relevancy": None,
                    "correctly_declined": None,
                })
                continue

            try:
                faith_score, faith_reason = score_faithfulness(answer, contexts)
                rel_score, rel_reason = score_relevancy(item["question"], answer)
                print(f"    Faithfulness: {faith_score}  |  Relevancy: {rel_score}")
                print(f"    [Faithfulness reason] {faith_reason}")
                print(f"    [Relevancy reason]    {rel_reason}")
            except requests.exceptions.ReadTimeout:
                print("    [WARN] Judge timed out - recording as unscored, continuing batch.")
                faith_score, rel_score = None, None

            rows.append({
                "id": item["id"],
                "category": item["category"],
                "question": item["question"],
                "answer": answer[:500],
                "faithfulness": faith_score,
                "relevancy": rel_score,
                "correctly_declined": None,
            })

    df = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    factual = df[df["category"] == "factual"]
    hallucination = df[df["category"] == "hallucination_check"]

    if not factual.empty:
        avg_faith = factual["faithfulness"].dropna().mean()
        avg_rel = factual["relevancy"].dropna().mean()
        print(f"Factual questions ({len(factual)}):")
        print(f"  Average faithfulness: {avg_faith:.1f} / 100")
        print(f"  Average relevancy:    {avg_rel:.1f} / 100")

    if not hallucination.empty:
        decline_rate = hallucination["correctly_declined"].sum() / len(hallucination) * 100
        print(f"\nHallucination-check questions ({len(hallucination)}):")
        print(f"  Correctly declined: {hallucination['correctly_declined'].sum()}/{len(hallucination)} ({decline_rate:.0f}%)")

    out_path = Path(__file__).parent.parent / "data" / "processed" / "evaluation_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")

if __name__ == "__main__":
    run_evaluation()
    
