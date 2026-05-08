"""Structure-aware chunking for ParsedPaper objects."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pipeline.ocr import ParsedPaper
from pipeline.config import PARSED_CACHE_DIR

DEFAULT_MAX_CHARS = 1800

SECTION_LABELS = {
    "abstract", "introduction", "background", "methods",
    "materials and methods", "results", "discussion",
    "conclusions", "conclusion", "references",
    "disclosures", "acknowledgements", "funding",
}


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def extract_year(paper_id: str) -> int | None:
    match = re.search(r"(19|20)\d{2}$", paper_id)
    return int(match.group(0)) if match else None


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def split_sentences(text: str) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|(?<=\.)\s+(?=\d+\.)", cleaned)
    return [p.strip() for p in parts if p.strip()]


def split_long_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text.strip()
    while len(remaining) > limit:
        split_at = remaining.rfind(" ", 0, limit)
        if split_at <= 0:
            split_at = limit
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def pack_units(units: list[str], limit: int, joiner: str) -> list[str]:
    packed: list[str] = []
    current = ""
    for unit in units:
        candidate = unit if not current else f"{current}{joiner}{unit}"
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            packed.append(current)
            current = ""
        if len(unit) <= limit:
            current = unit
        else:
            packed.extend(split_long_text(unit, limit))
    if current:
        packed.append(current)
    return packed


def chunk_section_body(body: str, available_chars: int) -> list[dict[str, Any]]:
    clean_body = body.strip()
    if not clean_body:
        return [{"text": "", "split_strategy": "heading", "paragraph_id": None}]
    if len(clean_body) <= available_chars:
        return [{"text": clean_body, "split_strategy": "heading", "paragraph_id": None}]

    paragraphs = split_paragraphs(clean_body)
    if len(paragraphs) > 1:
        para_chunks = pack_units(paragraphs, available_chars, "\n\n")
        if para_chunks:
            return [
                {"text": chunk, "split_strategy": "paragraph", "paragraph_id": i}
                for i, chunk in enumerate(para_chunks)
            ]

    sentences = split_sentences(clean_body)
    if len(sentences) > 1:
        sent_chunks = pack_units(sentences, available_chars, " ")
        if sent_chunks:
            return [{"text": chunk, "split_strategy": "sentence", "paragraph_id": None}
                    for chunk in sent_chunks]

    return [{"text": chunk, "split_strategy": "char", "paragraph_id": None}
            for chunk in split_long_text(clean_body, available_chars)]


def build_chunks_from_paper(paper: ParsedPaper, max_chars: int = DEFAULT_MAX_CHARS) -> list[dict[str, Any]]:
    """Return structure-aware chunks with full metadata including page numbers."""
    paper_id = paper.paper_id
    doc_meta = {
        "document_id": paper_id,
        "paper_id": paper_id,
        "year": extract_year(paper_id),
    }
    chunks: list[dict[str, Any]] = []

    for section_index, section in enumerate(paper.sections):
        heading = normalize_text(section.heading or "Untitled")
        heading_prefix = f"## {heading}"
        available_chars = max(300, max_chars - len(heading_prefix) - 1)
        body_chunks = chunk_section_body(section.text, available_chars)

        for part_index, body_chunk in enumerate(body_chunks):
            chunk_text = (
                f"{heading_prefix}\n{body_chunk['text']}"
                if body_chunk["text"]
                else heading_prefix
            )
            chunks.append({
                "id": f"{paper_id}_s{section_index}_p{part_index}",
                "text": chunk_text,
                "metadata": {
                    **doc_meta,
                    "section": heading,
                    "section_index": section_index,
                    "part_index": part_index,
                    "page_number": section.page_start,
                    "chunk_type": "section",
                    "split_strategy": body_chunk["split_strategy"],
                },
            })

    return chunks


def write_chunks_json(paper: ParsedPaper, max_chars: int = DEFAULT_MAX_CHARS) -> Path:
    """Write {paper_id}_chunks.json to data/parsed_papers/."""
    path = PARSED_CACHE_DIR / f"{paper.paper_id}_chunks.json"
    chunks = build_chunks_from_paper(paper, max_chars)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    return path


def write_sections_md(paper: ParsedPaper) -> Path:
    """Write sections-only .md (no tables) to data/parsed_papers/."""
    path = PARSED_CACHE_DIR / f"{paper.paper_id}.md"
    lines: list[str] = []
    for s in paper.sections:
        lines.append(f"## {s.heading}\n")
        lines.append(s.text)
        lines.append("\n\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def write_tables_json(paper: ParsedPaper) -> Path:
    """Write (or overwrite) tables-only JSON to data/parsed_papers/."""
    from dataclasses import asdict
    path = PARSED_CACHE_DIR / f"{paper.paper_id}_tables.json"
    with open(path, "w") as f:
        json.dump([asdict(t) for t in paper.tables], f, indent=2)
    return path
