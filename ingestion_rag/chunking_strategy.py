"""Minimal markdown-only chunking script.

This version focuses only on the first step you asked for:
take a parsed markdown paper, split it into structure-aware chunks,
and attach metadata to each chunk so the output is easy to inspect.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MARKDOWN_PATH = REPO_ROOT / "data/parsed_papers/Baloch_2022.md"
DEFAULT_MAX_CHARS = 1800

SECTION_LABELS = {
    "abstract",
    "introduction",
    "background",
    "methods",
    "materials and methods",
    "results",
    "discussion",
    "conclusions",
    "conclusion",
    "references",
    "disclosures",
    "acknowledgements",
    "funding",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk a parsed markdown paper into structure-aware chunks."
    )
    parser.add_argument(
        "--markdown-path",
        type=Path,
        default=DEFAULT_MARKDOWN_PATH,
        help="Path to the parsed markdown paper.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults next to the markdown file.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Maximum characters per chunk before fallback splitting.",
    )
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def extract_year(paper_id: str) -> int | None:
    match = re.search(r"(19|20)\d{2}$", paper_id)
    return int(match.group(0)) if match else None


def parse_markdown_sections(markdown_path: Path) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_heading = "preamble"
    current_level = 1
    current_lines: list[str] = []
    saw_heading = False

    with markdown_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            match = re.match(r"^(#{1,6})\s+(.*)$", line)
            if match:
                if current_lines or saw_heading:
                    sections.append(
                        {
                            "heading": current_heading,
                            "level": current_level,
                            "text": "\n".join(current_lines).strip(),
                        }
                    )
                current_heading = normalize_text(match.group(2))
                current_level = len(match.group(1))
                current_lines = []
                saw_heading = True
            else:
                current_lines.append(line)

    if current_lines or saw_heading:
        sections.append(
            {
                "heading": current_heading,
                "level": current_level,
                "text": "\n".join(current_lines).strip(),
            }
        )

    return [section for section in sections if section["heading"] or section["text"]]


def infer_title(sections: list[dict[str, Any]], paper_id: str) -> str:
    for section in sections:
        heading = normalize_text(section["heading"])
        if not heading or heading.lower() == "preamble":
            continue
        if heading.lower() not in SECTION_LABELS:
            return heading

    for section in sections:
        heading = normalize_text(section["heading"])
        if heading and heading.lower() != "preamble":
            return heading

    return paper_id


def split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]


def split_sentences(text: str) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|(?<=\.)\s+(?=\d+\.)", cleaned)
    return [part.strip() for part in parts if part.strip()]


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
        return [{"text": "", "split_strategy": "heading", "fallback": False, "paragraph_id": None}]

    if len(clean_body) <= available_chars:
        return [{"text": clean_body, "split_strategy": "heading", "fallback": False, "paragraph_id": None}]

    paragraphs = split_paragraphs(clean_body)
    if len(paragraphs) > 1:
        paragraph_chunks = pack_units(paragraphs, available_chars, "\n\n")
        if paragraph_chunks:
            return [
                {
                    "text": chunk,
                    "split_strategy": "paragraph",
                    "fallback": False,
                    "paragraph_id": index,
                }
                for index, chunk in enumerate(paragraph_chunks)
            ]

    sentences = split_sentences(clean_body)
    if len(sentences) > 1:
        sentence_chunks = pack_units(sentences, available_chars, " ")
        if sentence_chunks:
            return [
                {
                    "text": chunk,
                    "split_strategy": "sentence",
                    "fallback": True,
                    "paragraph_id": None,
                }
                for chunk in sentence_chunks
            ]

    return [
        {
            "text": chunk,
            "split_strategy": "sentence",
            "fallback": True,
            "paragraph_id": None,
        }
        for chunk in split_long_text(clean_body, available_chars)
    ]


def build_document_metadata(paper_id: str, title: str, markdown_path: Path) -> dict[str, Any]:
    return {
        "document_id": paper_id,
        "paper_id": paper_id,
        "title": title,
        "year": extract_year(paper_id),
        "source_markdown": str(markdown_path.relative_to(REPO_ROOT)),
    }


def build_chunks(markdown_path: Path, max_chars: int) -> list[dict[str, Any]]:
    paper_id = markdown_path.stem
    sections = parse_markdown_sections(markdown_path)
    title = infer_title(sections, paper_id)
    document_metadata = build_document_metadata(paper_id, title, markdown_path)

    chunks: list[dict[str, Any]] = []

    for section_index, section in enumerate(sections):
        heading = normalize_text(section["heading"] or "Untitled")
        level = section["level"]
        body = section["text"]
        heading_prefix = f"{'#' * level} {heading}".strip()
        available_chars = max(300, max_chars - len(heading_prefix) - 1)
        body_chunks = chunk_section_body(body, available_chars)

        for part_index, body_chunk in enumerate(body_chunks):
            chunk_text = heading_prefix
            if body_chunk["text"]:
                chunk_text = f"{heading_prefix}\n{body_chunk['text']}"

            chunks.append(
                {
                    "id": f"{paper_id}_section_{section_index}_{part_index}",
                    "text": chunk_text,
                    "metadata": {
                        **document_metadata,
                        "chunk_type": "section",
                        "section": heading,
                        "level": level,
                        "section_index": section_index,
                        "part_index": part_index,
                        "split_strategy": body_chunk["split_strategy"],
                        "paragraph_id": body_chunk["paragraph_id"],
                        "fallback": body_chunk["fallback"],
                    },
                }
            )

    return chunks


def default_output_path(markdown_path: Path) -> Path:
    return markdown_path.parent / f"{markdown_path.stem}_markdown_chunks.json"


def write_output(chunks: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(chunks, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    output_path = args.output_path or default_output_path(args.markdown_path)
    chunks = build_chunks(args.markdown_path, args.max_chars)
    write_output(chunks, output_path)

    print(f"Wrote {len(chunks)} markdown chunks")
    print(f"  json: {output_path}")


if __name__ == "__main__":
    main()
