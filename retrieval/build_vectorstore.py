"""
build_vectorstore.py

Loads data/processed/chunks.json, generates embeddings for each chunk using
a local sentence-transformer model, and stores them in a persistent ChromaDB
collection along with metadata (ticker, form, filing_date, section).
"""

import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNKS_FILE = PROCESSED_DIR / "chunks.json"
COLLECTION_NAME = "sec_filings"

EXCLUDED_SECTIONS = {"Preamble"}
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 100


def load_chunks() -> list[dict]:
    chunks = json.loads(CHUNKS_FILE.read_text())
    filtered = [c for c in chunks if c["section"] not in EXCLUDED_SECTIONS]
    print(f"Loaded {len(chunks)} total chunks, "
          f"{len(filtered)} after excluding {EXCLUDED_SECTIONS}")
    return filtered


def build_vectorstore():
    chunks = load_chunks()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"description": "SEC 10-K/10-Q filing chunks"},
    )

    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]

        ids = [f"{c['ticker']}_{c['form']}_{c['filing_date']}_{c['section']}_{c['chunk_index']}"
               for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = [
            {
                "ticker": c["ticker"],
                "company_name": c["company_name"],
                "form": c["form"],
                "filing_date": c["filing_date"],
                "section": c["section"],
            }
            for c in batch
        ]

        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"  Embedded {min(i + BATCH_SIZE, total)}/{total} chunks...")

    print(f"\nDone. {total} chunks embedded and stored in ChromaDB at {CHROMA_DIR}")
    print(f"Collection '{COLLECTION_NAME}' now has {collection.count()} items.")


if __name__ == "__main__":
    build_vectorstore()