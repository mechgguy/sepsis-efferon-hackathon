"""Thread-safe parallel PDF parsing. Each thread owns its own Docling converter."""
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline.ocr import build_converter, parse_pdf, ParsedPaper

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
        papers_dir:  directory containing PDFs
        max_workers: parallel threads (keep ≤4 on CPU-only machines)
        progress_cb: optional callable(completed, total, filename) for UI updates

    Returns:
        (successes, failures) where failures = [(path, exception), ...]
    """
    papers_dir = Path(papers_dir)
    pdfs = sorted(papers_dir.glob("*.pdf"))
    total = len(pdfs)
    successes: list[ParsedPaper] = []
    failures: list[tuple[Path, Exception]] = []

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
