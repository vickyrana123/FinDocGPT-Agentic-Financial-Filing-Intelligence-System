"""
llm_client.py

Single point of contact for all LLM calls in the project. Switches between
local Ollama (development) and Groq's hosted API (deployment) based on an
environment variable - this means deploying doesn't require hunting down
every individual Ollama call across the codebase, and switching providers
again in the future only means editing this one file.

Set LLM_PROVIDER=groq in your environment (e.g. on Render) to use Groq.
Defaults to "ollama" for local development if unset.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# --- Ollama (local dev) config ---
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

# --- Groq (deployed) config ---
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "groq/compound-mini"  # llama-3.1-8b-instant requires phone
                                     # verification not yet completed on this
                                     # account; compound-mini is accessible
                                     # without it and stays free


def call_llm(prompt: str, temperature: float = 0.2, max_tokens: int = 800, timeout: int = 60) -> str:
    """
    Sends a prompt to whichever provider is configured and returns the
    generated text. Same interface regardless of provider, so calling
    code never needs to know which one is active.
    """
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise EnvironmentError("GROQ_API_KEY not set - required when LLM_PROVIDER=groq.")

        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        if not response.ok:
            raise RuntimeError(f"Groq API error {response.status_code}: {response.text}")
        return response.json()["choices"][0]["message"]["content"]

    else:  # ollama (local dev default)
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": 4096,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()["response"]