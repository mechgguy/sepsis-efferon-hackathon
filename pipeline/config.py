import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ARTICLES_DIR = REPO_ROOT / "materials" / "articles"
PARSED_CACHE_DIR = REPO_ROOT / "data" / "parsed_papers"
PARSED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR = PARSED_CACHE_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Weaviate
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8090"))
COLLECTION_NAME = "RagDocumentChunk"

# Models
BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
