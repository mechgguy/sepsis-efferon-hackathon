"""
pipeline/extract.py — RAG-powered clinical evidence extraction.

Retrieves relevant chunks from Weaviate, then calls an LLM to extract
structured evidence records as JSON.

Provider priority (reads from .env or environment):
  1. OPENROUTER_API_KEY  → OpenRouter  (default model: google/gemini-2.5-pro)
  2. OPENAI_API_KEY      → OpenAI      (default model: gpt-4o)

Override the model with LLM_MODEL env var.

CLI:
  python -m pipeline.extract --query "lactate and 28-day mortality in septic shock"
  python -m pipeline.extract --query "..." --mode hybrid --top-k 8 --format markdown
"""
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from pipeline.weaviate.retrieve import retrieve

load_dotenv()

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical evidence extraction assistant supporting a sepsis registry analysis.

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


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_user_prompt(query: str, chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[CHUNK {i} | source: {chunk['title']} "
            f"| page: {chunk['page_number']} "
            f"| chunkIndex: {chunk['chunkIndex']}]\n"
            f"{chunk['compressedContent']}\n"
        )
    context = "\n---\n".join(parts)
    return (
        f"CLINICAL QUERY: {query}\n\n"
        f"SOURCE CHUNKS:\n{context}\n\n"
        "Extract all relevant evidence records as a JSON array."
    )


# ── LLM client factory ────────────────────────────────────────────────────────

def _get_client_and_model() -> tuple[OpenAI, str]:
    """Return (OpenAI client, model name) based on available env keys."""
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if openrouter_key:
        client = OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        )
        model = os.environ.get("LLM_MODEL", "google/gemini-2.5-pro")
        return client, model

    if openai_key:
        client = OpenAI(api_key=openai_key)
        model = os.environ.get("LLM_MODEL", "gpt-4o")
        return client, model

    raise RuntimeError(
        "No LLM API key found. "
        "Set OPENROUTER_API_KEY or OPENAI_API_KEY in your .env file."
    )


# ── Main extraction function ──────────────────────────────────────────────────

def extract_records(query: str, chunks: list[dict]) -> list[dict]:
    """Call LLM to extract structured evidence records from retrieved chunks."""
    client, model = _get_client_and_model()

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(query, chunks)},
        ],
        temperature=0,
        max_tokens=4096,
    )
    raw = (completion.choices[0].message.content or "").strip()

    # Strip markdown fences if model wraps output in ```json ... ```
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip()
            if part.startswith("json"):
                raw = part[4:].strip()
                break
            if part.startswith("[") or part.startswith("{"):
                raw = part
                break

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[WARN] JSON parse failed: {e}\nRaw:\n{raw}", file=sys.stderr)
        return []


# ── Output formatters ─────────────────────────────────────────────────────────

def format_table(records: list[dict]) -> str:
    if not records:
        return "No relevant evidence found."
    sep = "-" * 72
    lines = []
    for r in records:
        lines.append(sep)
        for key, val in r.items():
            lines.append(f"{key.replace('_', ' ').title().ljust(14)}: {val}")
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
    if not records:
        return "No relevant evidence found."
    keys = list(records[0].keys())
    header = "| " + " | ".join(k.replace("_", " ").title() for k in keys) + " |"
    sep    = "| " + " | ".join(["---"] * len(keys)) + " |"
    rows   = [
        "| " + " | ".join(str(r.get(k, "")).replace("|", "\\|") for k in keys) + " |"
        for r in records
    ]
    return "\n".join([header, sep] + rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract structured clinical evidence from Sepsis Atlas."
    )
    parser.add_argument("--query",      required=True, help="Clinical question")
    parser.add_argument("--mode",       choices=["bm25", "vector", "hybrid"], default="hybrid")
    parser.add_argument("--top-k",      type=int, default=8)
    parser.add_argument("--candidates", type=int, default=30)
    parser.add_argument("--no-rerank",  action="store_true")
    parser.add_argument("--format",     choices=["table", "json", "csv", "markdown"],
                        default="table")
    parser.add_argument("--out",        help="Optional output file path")
    args = parser.parse_args()

    print(f"[1/3] Retrieving chunks for: {args.query!r}", file=sys.stderr)
    chunks = retrieve(args.query, args.mode, args.top_k, args.candidates, not args.no_rerank)
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
