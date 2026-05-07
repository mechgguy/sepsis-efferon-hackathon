# Minimal Python RAG Implementation Plan

Local, single-user. Chunks are pre-split and provided as input.

---

## 1. Objective

- Accept pre-split chunks as a JSON file
- Embed each chunk with BGE-M3
- Write to Weaviate (`RagDocumentChunk`)
- Retrieve via BM25, vector, or hybrid
- Rerank results with a cross-encoder

---

## 2. File Structure

```
rag_python/
├── requirements.txt
├── config.py      — Weaviate connection
├── embedder.py    — BGE-M3 wrapper
├── reranker.py    — BGE cross-encoder reranker
├── ingest.py      — embed chunks and write to Weaviate
├── verify.py      — check chunks and vectors exist
└── retrieve.py    — BM25 / vector / hybrid search + reranking
```

---

## 3. Dependencies

```
weaviate-client==3.26.7
sentence-transformers==3.3.1
torch>=2.0
# reranker uses CrossEncoder from sentence-transformers — no extra package needed
```

---

## 4. Configuration (`config.py`)

```
WEAVIATE_HOST        — default: "localhost:8080"
WEAVIATE_HTTP_SCHEME — default: "http"
WEAVIATE_API_KEY     — default: ""
BGE_MODEL_NAME       — default: "BAAI/bge-m3"
BATCH_SIZE           — default: 32
```

Exports `get_client()` returning a connected `weaviate.Client`.

---

## 5. Input Format

Each chunk file is a JSON array. Each element:

```json
{
  "id": "Baloch_2022_section_3_0",
  "text": "## Introduction\nThe pediatric intensive care unit...",
  "metadata": {
    "document_id": "Baloch_2022",
    "title": "Comparison of Pediatric Sequential...",
    "section": "Introduction",
    "section_index": 3,
    "part_index": 0
  }
}
```

Fields used by ingest:

| Field | Used as |
|-------|---------|
| `id` | stored as `chunkId` for traceability |
| `text` | embedding input + stored as `compressedContent` |
| `metadata.section` | stored as `title` |
| `metadata.section_index` | stored as `chunkIndex` |
| `metadata.part_index` | stored as `chapterIndex` |

All other metadata fields are ignored.

---

## 6. Reranker (`reranker.py`)

Model: `BAAI/bge-reranker-v2-m3` — cross-encoder, multilingual, from the same BGE family as the embedder.

A cross-encoder scores each (query, chunk) pair jointly instead of comparing independent vectors.
This is slower but more accurate, so it runs on a small candidate set after initial retrieval.

```python
class Reranker:
    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]: ...
```

Internally:
```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-v2-m3")
pairs = [(query, chunk["compressedContent"]) for chunk in chunks]
scores = model.predict(pairs)          # one float per pair
ranked = sorted(zip(scores, chunks), reverse=True)
return [chunk for _, chunk in ranked[:top_k]]
```

Model loads on first call, reused for the process lifetime.

---

## 7. Embedder (`embedder.py`)

Model: `BAAI/bge-m3` — 1024-dim, mean pooling, L2 normalized.
Downloaded to `~/.cache/huggingface/` on first run (~2.27 GB).

The `text` field is embedded as-is. It already contains the section heading and body.

```python
class Embedder:
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

Model loads on first call, reused for the process lifetime.

---

## 7. Ingestion (`ingest.py`)

```bash
python ingest.py --chunks Baloch_2022_markdown_chunks.json
```

Flow:
```
1. Load JSON file → list of chunk objects
2. Embed all texts in batches of 32
3. For each chunk, POST to Weaviate RagDocumentChunk:
     title            ← metadata.section
     chunkIndex       ← metadata.section_index
     chapterIndex     ← metadata.part_index
     compressedContent ← text
     shortSummary     ← ""
     fullSummary      ← ""
   with vector = BGE-M3 embedding of text
4. Print: chunk count, elapsed time
```

No `RagDocument` parent. No cross-references. Single collection, flat writes.

---

## 8. Verify (`verify.py`)

```bash
python verify.py [--limit 10]
```

Fetches the first N `RagDocumentChunk` objects. Prints:
- Chunk count returned
- `title` and `chunkIndex` of each
- Vector dimension and L2 norm for the first 5

---

## 9. Retrieval (`retrieve.py`)

```bash
python retrieve.py --query "some question" [--mode bm25|vector|hybrid] [--top-k 5] [--candidates 20] [--no-rerank]
```

Default mode: `hybrid`. Reranking is on by default.

**Step 1 — fetch candidates from Weaviate**

`--candidates` controls how many chunks are fetched (default: 20). More candidates = better
reranker recall at the cost of more cross-encoder inference.

**BM25:**
```python
client.query.get("RagDocumentChunk", fields)
    .with_bm25(query=query_text)
    .with_limit(candidates)
    .do()
```

**Vector:**
```python
client.query.get("RagDocumentChunk", fields)
    .with_near_vector({"vector": embedder.embed(query_text)})
    .with_limit(candidates)
    .do()
```

**Hybrid:** run both, merge by Weaviate object ID:
```python
seen = set()
merged = []
for chunk in bm25_results + vector_results:
    if chunk["_additional"]["id"] not in seen:
        seen.add(chunk["_additional"]["id"])
        merged.append(chunk)
```

**Step 2 — rerank**

```python
results = reranker.rerank(query, candidates, top_k=top_k)
```

The reranker scores each (query, `compressedContent`) pair and returns the top `--top-k` chunks
ordered by cross-encoder score.

Pass `--no-rerank` to skip step 2 and return the raw Weaviate ranking.

Fields returned:
```graphql
title chunkIndex compressedContent _additional { id score }
```

Output per result:
```
[1] Introduction (chunkIndex=3, rerank_score=0.94)
    content: The pediatric intensive care unit plays an important role...
```

---

## 10. Run

```bash
pip install -r rag_python/requirements.txt

python rag_python/ingest.py --chunks docs/Baloch_2022_markdown_chunks.json

python rag_python/verify.py

# hybrid retrieval with reranking (default)
python rag_python/retrieve.py --query "mortality prediction in PICU" --mode hybrid

# fetch 20 candidates, return top 5 after reranking
python rag_python/retrieve.py --query "mortality prediction in PICU" --candidates 20 --top-k 5

# skip reranking
python rag_python/retrieve.py --query "mortality prediction in PICU" --no-rerank
```
