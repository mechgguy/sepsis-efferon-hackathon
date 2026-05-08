# schema.py
from weaviate.classes.config import DataType

# Each entry: field_name → (weaviate_type, extractor_fn(meta, chunk))
# meta = chunk["metadata"], chunk = raw JSON chunk dict
CHUNK_SCHEMA = {
    "chunkId":           (DataType.TEXT, lambda m, c: c["id"]),
    "title":             (DataType.TEXT, lambda m, c: m.get("section", m.get("title", ""))),
    "chunkIndex":        (DataType.INT,  lambda m, c: m.get("section_index", m.get("part_index", 0))),
    "chapterIndex":      (DataType.INT,  lambda m, c: m.get("part_index", 0)),
    "compressedContent": (DataType.TEXT, lambda m, c: c["text"]),
    "pageNumber":        (DataType.INT,  lambda m, c: m.get("page_number", 0)),
    "shortSummary":      (DataType.TEXT, lambda m, c: ""),
    "fullSummary":       (DataType.TEXT, lambda m, c: ""),
}

# Fields to fetch from Weaviate at query time (exclude write-only or unused)
RETRIEVE_PROPS = ["chunkId", "title", "chapterIndex", "chunkIndex", "compressedContent", "pageNumber", "shortSummary", "fullSummary"]

# Mapping from Weaviate field name → key used in chunk dicts passed to prompt
PROP_TO_CHUNK_KEY = {
    "compressedContent": "compressedContent",
    "title":             "title",
    "chunkIndex":        "chunkIndex",
    "chapterIndex":      "chapterIndex",
    "pageNumber":        "page_number",   # rename on retrieval
    "shortSummary":      "shortSummary",
    "fullSummary":       "fullSummary",
    "chunkId":           "chunkId",
}