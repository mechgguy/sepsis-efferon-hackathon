# Backward-compat shim — canonical code lives in pipeline/chunking.py
from pipeline.chunking import (
    build_chunks_from_paper,
    write_chunks_json,
    write_sections_md,
    write_tables_json,
    normalize_text,
    extract_year,
    chunk_section_body,
    split_paragraphs,
    split_sentences,
    split_long_text,
    pack_units,
    DEFAULT_MAX_CHARS,
    SECTION_LABELS,
)

__all__ = [
    "build_chunks_from_paper", "write_chunks_json", "normalize_text",
    "extract_year", "chunk_section_body", "split_paragraphs",
    "split_sentences", "split_long_text", "pack_units",
    "DEFAULT_MAX_CHARS", "SECTION_LABELS",
]
