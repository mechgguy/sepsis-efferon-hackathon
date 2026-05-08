"""
extract.py  —  Natural language query → structured evidence table

Usage:
    python extract.py --query "What is relationship between lactate and 28-day mortality in septic shock?"
    python extract.py --query "..." --top-k 10 --candidates 30 --mode hybrid --format table
"""
import argparse
import json
import os
import sys
from pathlib import Path
# Add rag_python to path if running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import retrieve as _retrieve_print  # noqa: F401 — keep for reference
from openai import OpenAI
from config import get_client, COLLECTION_NAME
from embedder import Embedder
from reranker import Reranker
from weaviate.classes.query import MetadataQuery
from ingestion_rag.rag_python.schema import RETRIEVE_PROPS, PROP_TO_CHUNK_KEY
# ── Weaviate retrieval (returns list[dict], no print) ─────────────────────────

_PROPS = RETRIEVE_PROPS  



def _to_record(obj) -> dict:
    p = obj.properties
    record = {chunk_key: p.get(wv_key, 0 if wv_key == "pageNumber" else "")
              for wv_key, chunk_key in PROP_TO_CHUNK_KEY.items()}
    record["_id"] = str(obj.uuid)
    record["_score"] = obj.metadata.score if obj.metadata else None
    return record


def retrieve_chunks(
    query: str,
    mode: str = "hybrid",
    top_k: int = 8,
    candidates: int = 30,
    rerank: bool = True,
) -> list[dict]:
    embedder = Embedder()
    vector = embedder.embed(query)
    client = get_client()
    try:
        collection = client.collections.get(COLLECTION_NAME)
        if mode == "bm25":
            resp = collection.query.bm25(
                query=query, limit=candidates,
                return_properties=_PROPS,
                return_metadata=MetadataQuery(score=True),
            )
        elif mode == "vector":
            resp = collection.query.near_vector(
                near_vector=vector, limit=candidates,
                return_properties=_PROPS,
                return_metadata=MetadataQuery(distance=True),
            )
        else:
            resp = collection.query.hybrid(
                query=query, vector=vector, limit=candidates,
                return_properties=_PROPS,
                return_metadata=MetadataQuery(score=True),
            )
        results = [_to_record(o) for o in resp.objects]
    finally:
        client.close()

    if rerank and results:
        results = Reranker().rerank(query, results, top_k)
    else:
        results = results[:top_k]
    return results


# ── Prompt construction ───────────────────────────────────────────────────────


SYSTEM_PROMPT = """You are a clinical evidence extraction assistant for a Sepsis Atlas.

Your task: given retrieved text chunks from scientific papers and a clinical query,
extract structured evidence records and return ONLY valid JSON.

Each record must follow this schema:
{
  "study": "<Author Year or paper title>",
  "population": "<patient population description>",
  "sample_size": "<N= ...>",
  "predictor": "<variable of interest>",
  "outcome": "<outcome definition e.g. 28-day mortality>",
  "timing": "<when measurement was taken>",
  "method": "<statistical method used>",
  "effect_size": "<OR/HR/AUC/cutoff value with 95% CI if available>",
  "performance": "<sensitivity, specificity, p-value, AUC if available>",
  "notes": "<any important caveats or adjustments>",
  "source_anchor": "<exact short quote from source text that supports this record, ≤30 words>"
  "page": "<page number from source chunk header>"
}

Rules:
- Extract ONLY values explicitly stated in the source text. Never infer or hallucinate.
- Use "not reported" for fields absent from the text.
- source_anchor must be a verbatim short excerpt (≤30 words) from the chunk.
- Return a JSON array of records. No preamble, no markdown fences, just the raw JSON array.
- If a chunk cites another study's results, set "study" to that cited author/year, NOT the source chunk's paper.
- Never report the same predictor+outcome+effect_size combination twice.
- If no relevant data found, return empty array: []
"""

SYSTEM_PROMPT= SYSTEM_PROMPT = """You are a clinical evidence extraction assistant supporting a sepsis registry analysis.

CONTEXT:
A clinical registry contains sepsis patients treated with hemoadsorption but has no control group.
The goal is to estimate expected mortality for similar untreated patients using published literature.

YOUR TASK:
Extract structured evidence records from the retrieved chunks that can inform expected mortality estimation.
Focus on:
- Associations between clinical variables and mortality
- Prognostic biomarkers (lactate, IL-6, lymphocytes, procalcitonin, etc.)
- Severity scores (SOFA, APACHE II/III) and their mortality associations
- Statistical modeling approaches used for mortality prediction
- Cohort characteristics that allow comparability to a hemoadsorption registry

Return ONLY a valid JSON array. No preamble, no markdown fences.

Each record must follow this schema:
{
  "study":          "<Author Year or paper title>",
  "population":     "<patient population: sepsis subtype, ICU setting, inclusion criteria>",
  "sample_size":    "<N= ...>",
  "predictor":      "<variable: biomarker, score, or clinical parameter>",
  "outcome":        "<outcome definition e.g. 28-day mortality, ICU mortality>",
  "timing":         "<when predictor was measured e.g. ICU admission, 24h, sequential>",
  "method":         "<statistical method: logistic regression, Cox, ROC, AUROC, etc.>",
  "effect_size":    "<OR/HR/AUC/cutoff with 95% CI if available>",
  "performance":    "<sensitivity, specificity, p-value, AUC if available>",
  "cohort_details": "<age, comorbidities, vasopressor use, ventilation — for comparability>",
  "notes":          "<caveats, adjustments, exclusion criteria, or limitations>",
  "source_anchor":  "<verbatim excerpt ≤30 words from source text supporting this record>",
  "page":           "<page number from source chunk header>"
}

Rules:
- Extract ONLY values explicitly stated in the source text. Never infer or hallucinate.
- Use "not reported" for absent fields.
- If a chunk cites another study's results, set "study" to that cited author/year.
- Never report the same predictor+outcome+effect_size combination twice.
- If no relevant data found, return [].
"""

def build_user_prompt(query: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[CHUNK {i} | source: {chunk['title']} | page: {chunk['page_number']} | chunkIndex: {chunk['chunkIndex']}]\n"
            f"{chunk['compressedContent']}\n"
        )
    context = "\n---\n".join(context_parts)
    return (
        f"CLINICAL QUERY: {query}\n\n"
        f"SOURCE CHUNKS:\n{context}\n\n"
        "Extract all relevant evidence records as a JSON array."
    )


# ── LLM call ─────────────────────────────────────────────────────────────────

import requests

def extract_records(query: str, chunks: list[dict]) -> list[dict]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    completion = client.chat.completions.create(
        model="google/gemini-2.5-pro",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(query, chunks)},
        ],
        temperature=0,
        max_tokens=4096,
    )
    raw = (completion.choices[0].message.content or "").strip()

# REPLACE the fence-stripping block with:
    if "```" in raw:
        parts = raw.split("```")
        # find the json part
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                raw = part[4:].strip()
                break
            elif part.startswith("[") or part.startswith("{"):
                raw = part
            break

    try:
        return json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"[WARN] JSON parse failed: {e}\nRaw:\n{raw}", file=sys.stderr)
        return []

# ── Output formatters ─────────────────────────────────────────────────────────

def format_table(records: list[dict]) -> str:
    """Render as plain-text table (dash-separated)."""
    if not records:
        return "No relevant evidence found."
    lines = []
    sep = "-" * 72
    for r in records:
        lines.append(sep)
        for key, val in r.items():
            label = key.replace("_", " ").title().ljust(14)
            lines.append(f"{label}: {val}")
    lines.append(sep)
    return "\n".join(lines)


def format_csv(records: list[dict]) -> str:
    if not records:
        return ""
    import csv, io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
    return buf.getvalue()


def format_markdown(records: list[dict]) -> str:
    """Render as markdown table."""
    if not records:
        return "No relevant evidence found."
    keys = list(records[0].keys())
    header = "| " + " | ".join(k.replace("_", " ").title() for k in keys) + " |"
    sep = "| " + " | ".join(["---"] * len(keys)) + " |"
    rows = []
    for r in records:
        row = "| " + " | ".join(str(r.get(k, "")).replace("|", "\\|") for k in keys) + " |"
        rows.append(row)
    return "\n".join([header, sep] + rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract structured evidence table from Sepsis Atlas.")
    parser.add_argument("--query", required=True, help="Natural language clinical question")
    parser.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=30)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--format", choices=["table", "json", "csv", "markdown"], default="table")
    parser.add_argument("--out", type=str, default=None, help="Optional output file path")
    args = parser.parse_args()

    print(f"[1/3] Retrieving chunks for: {args.query!r}", file=sys.stderr)
    chunks = retrieve_chunks(
        args.query, args.mode, args.top_k, args.candidates, not args.no_rerank
    )
    print(f"      → {len(chunks)} chunks retrieved", file=sys.stderr)

    print("[2/3] Extracting structured records via LLM...", file=sys.stderr)
    records = extract_records(args.query, chunks)
    print(f"      → {len(records)} records extracted", file=sys.stderr)

    print("[3/3] Formatting output...\n", file=sys.stderr)

    if args.format == "json":
        output = json.dumps(records, indent=2)
    elif args.format == "csv":
        output = format_csv(records)
    elif args.format == "markdown":
        output = format_markdown(records)
    else:
        output = format_table(records)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"Saved to {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()