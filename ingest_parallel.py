# ingest_parallel.py
"""
Thread-safe parallel ingest. Each thread gets its own DocumentConverter
instance so Docling's global state is never shared.

Usage:
    from ingest_parallel import parse_all_parallel
    papers = parse_all_parallel("papers/", max_workers=3)
"""
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from ingest import (
    build_converter, parse_pdf, ParsedPaper,
    ParsedSection, ParsedTable, PARSED_CACHE_DIR,
)
import json
from dataclasses import asdict

# thread-local storage: one converter per thread
_local = threading.local()

def _get_thread_converter():
    if not hasattr(_local, "converter"):
        _local.converter = build_converter()
    return _local.converter


def _parse_one(pdf_path: Path) -> tuple[Path, ParsedPaper | Exception]:
    """Parse a single PDF. Uses cache if available; otherwise uses thread-local converter."""
    try:
        cache_path = PARSED_CACHE_DIR / f"{pdf_path.stem}.json"
        if cache_path.exists():
            # cache hit — safe to load from any thread (read-only)
            return pdf_path, parse_pdf(pdf_path)

        # cache miss — use thread-local converter instead of global
        converter = _get_thread_converter()
        result = converter.convert(str(pdf_path))
        doc = result.document

        sections, current_heading, current_text = [], "preamble", []
        for item, _ in doc.iterate_items():
            t = type(item).__name__
            if t == "SectionHeaderItem":
                if current_text:
                    sections.append(ParsedSection(current_heading, " ".join(current_text)))
                current_heading = item.text
                current_text = []
            elif t in ("TextItem", "ListItem"):
                current_text.append(item.text)
        if current_text:
            sections.append(ParsedSection(current_heading, " ".join(current_text)))

        tables, section_at_table, heading_cursor, table_idx = [], {}, "preamble", 0
        for item, _ in doc.iterate_items():
            t = type(item).__name__
            if t == "SectionHeaderItem":
                heading_cursor = item.text
            elif t == "TableItem":
                section_at_table[table_idx] = heading_cursor
                table_idx += 1

        for i, table in enumerate(doc.tables):
            tables.append(ParsedTable(
                index=i,
                preceding_heading=section_at_table.get(i, "unknown"),
                markdown=table.export_to_markdown(doc),
                page_start=1,
            ))

        paper = ParsedPaper(
            paper_id=pdf_path.stem,
            sections=sections,
            tables=tables,
            full_markdown=doc.export_to_markdown(),
        )
        with open(cache_path, "w") as f:
            json.dump(asdict(paper), f)

        return pdf_path, paper

    except Exception as e:
        return pdf_path, e


def parse_all_parallel(
    papers_dir: str | Path,
    max_workers: int = 3,
    progress_cb=None,
) -> tuple[list[ParsedPaper], list[tuple[Path, Exception]]]:
    """
    Parse all PDFs in papers_dir using a thread pool.

    Args:
        papers_dir:   directory containing PDFs
        max_workers:  parallel threads (keep ≤4 on CPU-only machines)
        progress_cb:  optional callable(completed, total, filename) for UI updates

    Returns:
        (successes, failures) where failures = [(path, exception), ...]
    """
    papers_dir = Path(papers_dir)
    pdfs = sorted(papers_dir.glob("*.pdf"))
    total = len(pdfs)
    successes, failures = [], []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_parse_one, p): p for p in pdfs}
        for i, future in enumerate(as_completed(futures), 1):
            path, result = future.result()
            if isinstance(result, Exception):
                failures.append((path, result))
            else:
                successes.append(result)
            if progress_cb:
                progress_cb(i, total, path.name)

    return successes, failures