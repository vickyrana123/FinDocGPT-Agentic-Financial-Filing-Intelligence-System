"""
parse_and_chunk.py

Converts raw SEC filing HTML (data/raw_filings/) into clean, section-tagged
text chunks (data/processed/), ready for embedding in Phase 2.

Pipeline:
1. Strip HTML -> readable plain text (BeautifulSoup)
2. Detect standard 10-K/10-Q "Item" section headers (Item 1A, Item 7, etc.)
   and tag every paragraph with which section it belongs to
3. Chunk text within each section (not across section boundaries -
   this matters: a chunk spanning Risk Factors -> MD&A would confuse
   retrieval later)
4. Save as structured JSON: one record per chunk, with metadata
   (ticker, form, filing_date, section, chunk_text)

Why section-aware chunking (not just fixed-size splitting):
Naive chunking (e.g. every 500 characters) ignores document structure -
a chunk might start mid-sentence in the Risk Factors section and end in
the middle of the MD&A section, making it useless for a query like
"what does the filing say about competition risk?" This is the #1
reason generic RAG on financial documents performs poorly.
"""

import json
import re
from pathlib import Path
from bs4 import BeautifulSoup

RAW_DIR = Path(__file__).parent.parent / "data" / "raw_filings"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Standard 10-K/10-Q item headers we want to detect and tag.
# Regex is deliberately loose (case-insensitive, flexible spacing) because
# SEC filings are inconsistently formatted across companies and years.
SECTION_PATTERNS = {
    "Business": r"item\s*1\.?\s*business",
    "Risk Factors": r"item\s*1a\.?\s*risk\s*factors",
    "Unresolved Staff Comments": r"item\s*1b\.?\s*unresolved\s*staff\s*comments",
    "Properties": r"item\s*2\.?\s*properties",
    "Legal Proceedings": r"item\s*3\.?\s*legal\s*proceedings",
    "MD&A": r"item\s*7\.?\s*management.?s\s*discussion",
    "Quantitative Disclosures": r"item\s*7a\.?\s*quantitative",
    "Financial Statements": r"item\s*8\.?\s*financial\s*statements",
    "Controls and Procedures": r"item\s*9a\.?\s*controls\s*and\s*procedures",
}

CHUNK_SIZE = 1200          # characters per chunk - sized for typical embedding models
CHUNK_OVERLAP = 200        # overlap so context isn't lost at chunk boundaries
MIN_CHUNK_LENGTH = 100     # skip tiny fragments (headers, noise)


def html_to_clean_text(html_path: Path) -> str:
    """Strips HTML down to readable plain text."""
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    # Remove elements that never contain useful content
    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Collapse excessive whitespace/blank lines left over from HTML structure
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def tag_sections(text: str) -> list[dict]:
    """
    Finds all section header matches across the WHOLE document (not
    line-by-line), then splits the document into blocks based on where
    each header occurs.
    """
    matches = []  # (start_position, section_name)
    for section_name, pattern in SECTION_PATTERNS.items():
        for m in re.finditer(pattern, text, re.IGNORECASE):
            matches.append((m.start(), section_name))

    matches.sort(key=lambda x: x[0])

    if not matches:
        return [{"section": "Preamble", "text": text}]

    blocks = []

    if matches[0][0] > 0:
        blocks.append({"section": "Preamble", "text": text[: matches[0][0]]})

    for i, (start, section_name) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        blocks.append({"section": section_name, "text": text[start:end]})

    return blocks


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    """
    Simple sliding-window chunker with overlap, snapped to sentence
    boundaries where possible so chunks don't cut mid-sentence.
    """
    if len(text) <= chunk_size:
        return [text] if len(text) >= MIN_CHUNK_LENGTH else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to snap to the last sentence boundary within this chunk
        last_period = chunk.rfind(". ")
        if last_period > chunk_size * 0.5:  # only snap if it doesn't shrink chunk too much
            end = start + last_period + 1
            chunk = text[start:end]

        if len(chunk.strip()) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk.strip())

        start = end - overlap  # move forward, keeping overlap

    return chunks

def process_filing(manifest_entry: dict) -> list[dict]:
    """
    Full pipeline for one filing: HTML -> clean text -> section-tagged ->
    chunked. Returns a list of chunk records ready to save.
    """
    html_path = Path(manifest_entry["local_path"])
    if not html_path.exists():
        print(f"  SKIP (file missing): {html_path}")
        return []

    clean_text = html_to_clean_text(html_path)
    section_blocks = tag_sections(clean_text)

    # A section name (e.g. "Risk Factors") can appear as more than one
    # block if it's referenced multiple times in the filing. We track a
    # running counter PER SECTION NAME across all its blocks, so
    # chunk_index stays globally unique instead of restarting at 0 per
    # block and colliding.
    section_running_index = {}

    records = []
    for block in section_blocks:
        section_chunks = chunk_text(block["text"])
        section_name = block["section"]
        start_index = section_running_index.get(section_name, 0)

        for offset, chunk in enumerate(section_chunks):
            records.append({
                "ticker": manifest_entry.get("ticker", "UNKNOWN"),
                "company_name": manifest_entry["company_name"],
                "form": manifest_entry["form"],
                "filing_date": manifest_entry["filing_date"],
                "section": section_name,
                "chunk_index": start_index + offset,
                "text": chunk,
            })

        section_running_index[section_name] = start_index + len(section_chunks)

    return records

def run():
    manifest_path = RAW_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_records = []

    for entry in manifest:
        label = f"{entry.get('ticker', '?')}_{entry['form']}_{entry['filing_date']}"
        print(f"Processing {label}...")
        records = process_filing(entry)
        print(f"  -> {len(records)} chunks across "
              f"{len(set(r['section'] for r in records))} sections")
        all_records.extend(records)

    out_path = PROCESSED_DIR / "chunks.json"
    out_path.write_text(json.dumps(all_records, indent=2))

    print(f"\nDone. {len(all_records)} total chunks saved to {out_path}")

    # Quick summary so you can sanity-check section coverage
    section_counts = {}
    for r in all_records:
        section_counts[r["section"]] = section_counts.get(r["section"], 0) + 1
    print("\nChunks per section:")
    for section, count in sorted(section_counts.items(), key=lambda x: -x[1]):
        print(f"  {section}: {count}")

if __name__ == "__main__":
    run()