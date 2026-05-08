# Backward-compat shim — canonical code lives in pipeline/ocr.py
from pipeline.ocr import (
    ParsedPaper,
    ParsedSection,
    ParsedTable,
    ParsedFigure,
    build_converter,
    parse_pdf,
    parse_all,
)
from pipeline.config import PARSED_CACHE_DIR, FIGURE_DIR

__all__ = [
    "ParsedPaper", "ParsedSection", "ParsedTable", "ParsedFigure",
    "build_converter", "parse_pdf", "parse_all",
    "PARSED_CACHE_DIR", "FIGURE_DIR",
]
