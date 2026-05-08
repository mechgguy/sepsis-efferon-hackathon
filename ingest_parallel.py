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
from ingest import FIGURE_DIR, ParsedFigure
# thread-local storage: one converter per thread
_local = threading.local()

def _get_thread_converter():
    if not hasattr(_local, "converter"):
        _local.converter = build_converter()
    return _local.converter


def _parse_one(pdf_path: Path) -> tuple[Path, ParsedPaper | Exception]:
    try:
        return pdf_path, parse_pdf(pdf_path, converter=_get_thread_converter())
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