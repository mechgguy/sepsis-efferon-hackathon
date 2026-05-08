# Backward-compat shim — canonical code lives in pipeline/ocr_parallel.py
from pipeline.ocr_parallel import parse_all_parallel

__all__ = ["parse_all_parallel"]
