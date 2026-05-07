# Python RAG Pipeline — Implementation Guide

Local, single-user RAG system that embeds medical-paper chunks with **BGE-M3**, stores them in **Weaviate**, and retrieves them with BM25 / vector / hybrid search, then reranks with a **BGE cross-encoder**.

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [File Structure](#2-file-structure)
3. [Data Flow](#3-data-flow)
4. [Input Format](#4-input-format)
5. [Weaviate Schema](#5-weaviate-schema)
6. [File-by-File Breakdown](#6-file-by-file-breakdown)
   - [config.py](#61-configpy)
   - [embedder.py](#62-embedderpy)
   - [reranker.py](#63-rerankerpy)
   - [ingest.py](#64-ingestpy)
   - [verify.py](#65-verifypy)
   - [retrieve.py](#66-retrievepy)
7. [Step-by-Step Run Guide](#7-step-by-step-run-guide)
8. [Retrieval Modes Explained](#8-retrieval-modes-explained)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                              │
│                                                                  │
│  ┌──────────────┐     ┌────────────┐     ┌───────────────────┐  │
│  │  chunks.json │────▶│ embedder   │────▶│   Weaviate DB     │  │
│  │  (24 chunks) │     │ (BGE-M3)   │     │ RagDocumentChunk  │  │
│  └──────────────┘     │ 1024-dim   │     │ (vectors + BM25)  │  │
│                       └────────────┘     └────────┬──────────┘  │
│                                                   │             │
│  ┌──────────────┐     ┌────────────┐     ┌────────▼──────────┐  │
│  │   Answer     │◀────│  Reranker  │◀────│    Retrieval      │  │
│  │  (top-k      │     │ BGE cross- │     │  BM25 / vector /  │  │
│  │   chunks)    │     │  encoder   │     │     hybrid        │  │
│  └──────────────┘     └────────────┘     └───────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Models used:**

| Role | Model | Size | Notes |
|------|-------|------|-------|
| Embedder | `BAAI/bge-m3` | ~2.3 GB | 1024-dim, multilingual, L2-normalized |
| Reranker | `BAAI/bge-reranker-v2-m3` | ~1.1 GB | Cross-encoder, same BGE family |

Both models are downloaded to `~/.cache/huggingface/` on first run and cached permanently.

---

## 2. File Structure

```
ingestion_rag/
├── docker-compose.yml          ← Weaviate container
├── output/
│   └── chunks/
│       └── Baloch_2022_markdown_chunks.json   ← input chunks
└── rag_python/
    ├── requirements.txt
    ├── config.py               ← connection + constants
    ├── embedder.py             ← BGE-M3 wrapper
    ├── reranker.py             ← BGE cross-encoder reranker
    ├── ingest.py               ← embed + write to Weaviate
    ├── verify.py               ← confirm chunks + vectors exist
    └── retrieve.py             ← BM25 / vector / hybrid + rerank
```

---

## 3. Data Flow

### Ingestion flow

```
chunks.json
    │
    ▼
[ load JSON array ]
    │
    ▼  (embed_batch, BATCH_SIZE=32)
[ BGE-M3 ] ──── produces ───▶ [ 1024-dim float vector per chunk ]
    │
    ▼
[ Weaviate batch.dynamic() ]
    │
    ├── chunk["id"]            ──▶  chunkId  (property)
    ├── metadata["section"]    ──▶  title
    ├── metadata["section_index"] ▶ chunkIndex
    ├── metadata["part_index"] ──▶  chapterIndex
    ├── chunk["text"]          ──▶  compressedContent
    └── vector                 ──▶  self_provided vector index
```

### Retrieval flow

```
query string
    │
    ├──(BM25)──▶  tokenise + keyword match  ──────────────────┐
    │                                                          ▼
    ├──(vector)─▶  BGE-M3 embed query  ──▶  near_vector  ─────▶  up to N candidates
    │                                                          │
    └──(hybrid)─▶  both above via Weaviate hybrid  ───────────┘
                                                               │
                                                  ┌────────────▼─────────────┐
                                                  │  BGE reranker            │
                                                  │  score(query, content)   │
                                                  │  for each candidate      │
                                                  └────────────┬─────────────┘
                                                               │
                                                         top-k results
```

---

## 4. Input Format

Each chunk file is a JSON array. Every element follows this shape:

```json
{
  "id": "Baloch_2022_section_3_0",
  "text": "## Introduction\nThe pediatric intensive care unit (PICU) plays an important role...",
  "metadata": {
    "document_id": "Baloch_2022",
    "paper_id": "Baloch_2022",
    "title": "Comparison of Pediatric Sequential Organ Failure Assessment...",
    "year": 2022,
    "source_markdown": "data/parsed_papers/Baloch_2022.md",
    "chunk_type": "section",
    "section": "Introduction",
    "level": 2,
    "section_index": 3,
    "part_index": 0,
    "split_strategy": "sentence",
    "paragraph_id": null,
    "fallback": true
  }
}
```

Fields used by the pipeline:

| JSON field | Mapped to (Weaviate) | Notes |
|---|---|---|
| `id` | `chunkId` | stored as-is for traceability |
| `text` | `compressedContent` | embedding input + stored content |
| `metadata.section` | `title` | section heading |
| `metadata.section_index` | `chunkIndex` | position of section in doc |
| `metadata.part_index` | `chapterIndex` | sub-part within section |

---

## 5. Weaviate Schema

Collection name: `RagDocumentChunk`

```
┌─────────────────────────────────────────────────────────┐
│  RagDocumentChunk                                        │
├──────────────────┬──────────┬────────────────────────────┤
│ Property         │ Type     │ Purpose                    │
├──────────────────┼──────────┼────────────────────────────┤
│ chunkId          │ TEXT     │ original chunk "id" field  │
│ title            │ TEXT     │ section heading (BM25)     │
│ chunkIndex       │ INT      │ section position           │
│ chapterIndex     │ INT      │ sub-part index             │
│ compressedContent│ TEXT     │ full chunk text (BM25)     │
│ shortSummary     │ TEXT     │ reserved (empty for now)   │
│ fullSummary      │ TEXT     │ reserved (empty for now)   │
├──────────────────┼──────────┼────────────────────────────┤
│ vector           │ FLOAT[]  │ 1024-dim BGE-M3 embedding  │
│                  │          │ self_provided (no module)  │
└──────────────────┴──────────┴────────────────────────────┘
```

Vector config: `Configure.Vectors.self_provided()` — Weaviate stores the vector as-is without running any internal vectorizer. The pipeline is responsible for passing the vector at insert time.

UUIDs are generated **deterministically** from the chunk ID:
```python
from weaviate.util import generate_uuid5
uuid = generate_uuid5("Baloch_2022_section_3_0")
# → always the same UUID for the same string
# → re-running ingest never creates duplicates
```

---

## 6. File-by-File Breakdown

### 6.1 `config.py`

Central place for all tuneable constants. Everything else imports from here.

```python
import os
import weaviate

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-m3")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

COLLECTION_NAME = "RagDocumentChunk"

def get_client() -> weaviate.WeaviateClient:
    return weaviate.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT)
```

All values can be overridden with environment variables without touching code:
```bash
WEAVIATE_HOST=192.168.1.10 BATCH_SIZE=16 python ingest.py --chunks ...
```

---

### 6.2 `embedder.py`

Wraps `BAAI/bge-m3` via `sentence-transformers`. The model is loaded once on first call and reused for the process lifetime (lazy singleton pattern).

```python
from sentence_transformers import SentenceTransformer
from config import BGE_MODEL_NAME, BATCH_SIZE

class Embedder:
    _model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(BGE_MODEL_NAME)
        return self._model

    def embed(self, text: str) -> list[float]:
        # used at query time — single text
        return self._get_model().encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # used at ingest time — all chunks at once
        return self._get_model().encode(
            texts, batch_size=BATCH_SIZE, normalize_embeddings=True, show_progress_bar=True
        ).tolist()
```

**Why L2 normalization?**  
BGE-M3 is trained for cosine similarity. Normalizing to unit length means cosine similarity collapses to dot product, which Weaviate's HNSW index computes efficiently.

**Why batch?**  
`encode()` with `batch_size=32` processes chunks in groups, keeping GPU/CPU memory bounded while still benefiting from vectorized matrix ops.

---

### 6.3 `reranker.py`

Uses `BAAI/bge-reranker-v2-m3`, a **cross-encoder** — it takes a `(query, document)` pair and outputs a single relevance score. This is more accurate than vector similarity but slower, so it only runs on the small candidate set returned by Weaviate.

```python
from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL_NAME

class Reranker:
    _model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(RERANKER_MODEL_NAME)
        return self._model

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        pairs = [(query, chunk["compressedContent"]) for chunk in chunks]
        scores = self._get_model().predict(pairs)       # one float per pair
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        result = []
        for score, chunk in ranked[:top_k]:
            chunk = dict(chunk)
            chunk["_rerank_score"] = float(score)       # attached for display
            result.append(chunk)
        return result
```

**Bi-encoder(not) vs cross-encoder:**

```
Bi-encoder (BGE-M3)                Cross-encoder (reranker)
─────────────────────               ──────────────────────────
query  ──▶ vector A                 (query, doc) ──▶ score
doc    ──▶ vector B                
similarity = cosine(A, B)          

Fast (pre-computed vectors)        Slow (joint inference per pair)
Good recall, weaker precision      Better precision, limited scale
Used for: fetching N candidates    Used for: reranking top N
```

---

### 6.4 `ingest.py`

Entry point for loading chunks into Weaviate.

```bash
python ingest.py --chunks ../output/chunks/Baloch_2022_markdown_chunks.json
```

Full flow:

```python
def ingest(chunks_path: str):
    # 1. Load JSON
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    # 2. Embed all texts in one batched call
    print(f"Embedding {len(chunks)} chunks with BGE-M3...")
    t0 = time.time()
    embedder = Embedder()
    vectors = embedder.embed_batch([c["text"] for c in chunks])

    # 3. Connect and ensure collection exists
    client = get_client()
    try:
        ensure_collection(client)
        collection = client.collections.get(COLLECTION_NAME)

        # 4. Batch write to Weaviate
        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, vectors):
                meta = chunk.get("metadata", {})
                batch.add_object(
                    uuid=generate_uuid5(chunk["id"]),   # deterministic
                    properties={
                        "chunkId":           chunk["id"],
                        "title":             meta.get("section", ""),
                        "chunkIndex":        meta.get("section_index", 0),
                        "chapterIndex":      meta.get("part_index", 0),
                        "compressedContent": chunk["text"],
                        "shortSummary":      "",
                        "fullSummary":       "",
                    },
                    vector=vector,
                )
    finally:
        client.close()

    print(f"Ingested {len(chunks)} chunks in {time.time() - t0:.1f}s")
```

`batch.dynamic()` lets Weaviate auto-tune batch size and concurrency. Errors per object are captured without aborting the whole batch.

Collection creation (called once, skipped if already exists):

```python
def ensure_collection(client):
    if client.collections.exists(COLLECTION_NAME):
        return
    client.collections.create(
        name=COLLECTION_NAME,
        properties=[...],
        vector_config=Configure.Vectors.self_provided(),
    )
```

---

### 6.5 `verify.py`

Sanity check after ingestion. Confirms chunks are stored and vectors have the right shape.

```bash
python verify.py --limit 10
```

```python
def verify(limit: int = 10):
    client = get_client()
    try:
        # List all collections — useful for debugging
        existing = [c.name for c in client.collections.list_all().values()]
        print(f"Collections in Weaviate: {existing or '(none)'}")

        if COLLECTION_NAME not in existing:
            print(f"Collection '{COLLECTION_NAME}' not found — run ingest.py first.")
            return

        collection = client.collections.get(COLLECTION_NAME)
        response = collection.query.fetch_objects(limit=limit, include_vector=True)

        print(f"Chunk count returned: {len(response.objects)}")
        for i, obj in enumerate(response.objects):
            props = obj.properties
            vec = obj.vector.get("default") if obj.vector else None
            line = f"[{i+1}] title={props.get('title')!r}, chunkIndex={props.get('chunkIndex')}"
            if vec is not None and i < 5:
                norm = math.sqrt(sum(x * x for x in vec))
                line += f", dim={len(vec)}, L2_norm={norm:.4f}"
            print(line)
    finally:
        client.close()
```

Expected output after a successful ingest:
```
Collections in Weaviate: ['RagDocumentChunk']
Chunk count returned: 10
[1] title='preamble', chunkIndex=0, dim=1024, L2_norm=1.0000
[2] title='Abstract', chunkIndex=2, dim=1024, L2_norm=1.0000
...
```

`L2_norm=1.0000` confirms the vectors are correctly normalized.

---

### 6.6 `retrieve.py`

Retrieval with three modes, optional reranking.

```bash
python retrieve.py --query "mortality prediction in PICU" --mode hybrid
```

**Step 1 — fetch candidates from Weaviate**

```python
# BM25 — keyword matching
def _bm25(collection, query, limit):
    resp = collection.query.bm25(
        query=query, limit=limit,
        return_properties=_PROPS,
        return_metadata=MetadataQuery(score=True),
    )
    return [_to_record(o) for o in resp.objects]

# Vector — semantic similarity
def _vector(collection, vector, limit):
    resp = collection.query.near_vector(
        near_vector=vector, limit=limit,
        return_properties=_PROPS,
        return_metadata=MetadataQuery(distance=True),
    )
    return [_to_record(o) for o in resp.objects]

# Hybrid — Weaviate native fusion of BM25 + vector
def _hybrid(collection, query, vector, limit):
    resp = collection.query.hybrid(
        query=query, vector=vector, limit=limit,
        return_properties=_PROPS,
        return_metadata=MetadataQuery(score=True),
    )
    return [_to_record(o) for o in resp.objects]
```

**Step 2 — rerank**

```python
if rerank and results:
    results = Reranker().rerank(query, results, top_k)
else:
    results = results[:top_k]
```

**Output format:**
```
[1] Introduction (chunkIndex=3, rerank_score=0.9421)
    content: The pediatric intensive care unit (PICU) plays an important role...

[2] Abstract (chunkIndex=2, rerank_score=0.8803)
    content: Objective: To assess and compare the diagnostic accuracy of the...
```

---

## 7. Step-by-Step Run Guide

### Prerequisites

- Python 3.10+
- Docker + Docker Compose
- ~4 GB disk space for models
- Active internet on first run (model download)

---

### Step 1 — Install Python dependencies

```bash
cd ingestion_rag/rag_python
python -m venv ../../venv        # skip if you already have a venv
source ../../venv/bin/activate
pip install -r requirements.txt
```

---

### Step 2 — Start Weaviate

```bash
cd ingestion_rag
docker compose up -d
```

Weaviate will be available at `http://localhost:8080`. Confirm it's up:

```bash
curl http://localhost:8080/v1/.well-known/ready
# → {"status":"OK"}
```

The `docker-compose.yml` runs a single-node Weaviate instance with anonymous access and a persistent named volume (`weaviate_data`), so data survives container restarts.

```yaml
services:
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.37.2
    ports:
      - "8080:8080"    # REST + GraphQL
      - "50051:50051"  # gRPC (used by v4 client)
    volumes:
      - weaviate_data:/var/lib/weaviate
    environment:
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
```

> **Port 50051 matters.** The Weaviate v4 Python client uses gRPC for search queries. If port 50051 is blocked, searches will fail even if REST works.

---

### Step 3 — Ingest chunks

```bash
cd rag_python

# If model download hangs (see Troubleshooting §9.1):
HF_HUB_DISABLE_XET=1 python ingest.py --chunks ../output/chunks/Baloch_2022_markdown_chunks.json

# Normal run:
python ingest.py --chunks ../output/chunks/Baloch_2022_markdown_chunks.json
```

Expected output:
```
Embedding 24 chunks with BGE-M3...
Batches: 100%|████████████████████| 1/1 [00:03<00:00, 3.21s/it]
Ingested 24 chunks in 28.4s
```

First run is slow (~2–5 min) because BGE-M3 downloads 2.27 GB. Subsequent runs are fast (model is cached).

---

### Step 4 — Verify

```bash
python verify.py --limit 10
```

Expected output:
```
Collections in Weaviate: ['RagDocumentChunk']
Chunk count returned: 10
[1] title='preamble', chunkIndex=0, dim=1024, L2_norm=1.0000
[2] title='Comparison of Pediatric...', chunkIndex=1, dim=1024, L2_norm=1.0000
[3] title='Abstract', chunkIndex=2, dim=1024, L2_norm=1.0000
[4] title='Abstract', chunkIndex=2, dim=1024, L2_norm=1.0000
[5] title='Introduction', chunkIndex=3, dim=1024, L2_norm=1.0000
[6] title='Introduction', chunkIndex=3
[7] title='Materials And Methods', chunkIndex=4
...
```

`dim=1024` and `L2_norm=1.0000` on the first 5 confirms the embeddings are correct.

---

### Step 5 — Retrieve

```bash
# Default: hybrid search + reranking, top 5
python retrieve.py --query "mortality prediction in PICU"

# BM25 only, no reranking
python retrieve.py --query "PRISM III score accuracy" --mode bm25 --no-rerank

# Vector only, fetch 30 candidates, return top 3
python retrieve.py --query "organ failure in children" --mode vector --candidates 30 --top-k 3

# Hybrid with more candidates for better reranker recall
python retrieve.py --query "p-SOFA vs PRISM comparison" --candidates 20 --top-k 5
```

---

### Full command sequence (copy-paste)

```bash
# 1. Start Weaviate
cd ~/Sepsis_hackathon/ingestion_rag
docker compose up -d

# 2. Wait for readiness
curl -s http://localhost:8080/v1/.well-known/ready

# 3. Activate venv and install
source ~/venv/bin/activate
cd rag_python
pip install -r requirements.txt

# 4. Ingest (with xet fix)
HF_HUB_DISABLE_XET=1 python ingest.py \
  --chunks ../output/chunks/Baloch_2022_markdown_chunks.json

# 5. Verify
python verify.py --limit 10

# 6. Retrieve
python retrieve.py --query "mortality prediction in PICU" --mode hybrid
```

---

## 8. Retrieval Modes Explained

```
Mode       What it does                         Best for
─────────  ───────────────────────────────────  ─────────────────────────────
bm25       Keyword frequency (TF-IDF style)     Exact terms, acronyms, codes
vector     Cosine similarity on BGE-M3 vectors  Semantic / paraphrase queries
hybrid     Weaviate fuses both scores           General use — recommended
```

**Why hybrid + reranker?**

```
Query: "pediatric organ failure prognosis"

BM25 results             Vector results          After reranking
────────────────         ──────────────────      ──────────────────────
1. Discussion            1. Abstract             1. Introduction (0.94)
2. Methods               2. Introduction         2. Abstract (0.88)
3. References            3. Discussion           3. Discussion (0.81)
```

BM25 catches exact keywords; vector catches semantic matches; the reranker jointly reads the query and each chunk to find the best fit, often surface results that neither alone would rank first.

**Candidate count vs top-k:**

```
--candidates 20  →  Weaviate returns 20 chunks
                         │
                   reranker scores all 20
                         │
--top-k 5        →  return best 5

More candidates = better recall for reranker, but more cross-encoder inference.
```

---

## 9. Troubleshooting

### 9.1 Model download hangs at 0%

**Symptom:**
```
pytorch_model.bin:   0%| | 0.00/2.27G [00:00<?, ?B/s]
DeprecationWarning: hf_xet.download_files() is deprecated.
```

**Cause:** The `hf_xet` transfer protocol is broken in your environment.

**Fix:**
```bash
HF_HUB_DISABLE_XET=1 python ingest.py --chunks ...
```

Or pre-download the model first, then run normally:
```bash
HF_HUB_DISABLE_XET=1 huggingface-cli download BAAI/bge-m3
```

---

### 9.2 Collection not found

**Symptom:**
```
WeaviateQueryError: could not find class RagDocumentChunk in schema
```

**Cause:** Ingest was never run, or it failed before creating the collection.

**Fix:**
```bash
python verify.py  # will list what collections exist
python ingest.py --chunks ../output/chunks/Baloch_2022_markdown_chunks.json
```

---

### 9.3 gRPC connection refused on port 50051

**Symptom:**
```
grpc._channel._InactiveRpcError: StatusCode.UNAVAILABLE
```

**Cause:** The Weaviate container's port 50051 is not exposed or blocked.

**Fix:** Confirm the `docker-compose.yml` exposes `50051:50051` and re-up:
```bash
docker compose down && docker compose up -d
```

---

### 9.4 BM25 returns 0 results

**Cause:** Weaviate BM25 tokenises on whitespace and punctuation. Very short queries or queries with only stop-words may return empty.

**Fix:** Use hybrid mode (default) which falls back to vector when BM25 finds nothing:
```bash
python retrieve.py --query "mortality" --mode hybrid
```

---

### 9.5 Wipe and re-ingest

If you need to start fresh:
```bash
# Delete the collection from Weaviate
python - <<'EOF'
from config import get_client, COLLECTION_NAME
c = get_client()
c.collections.delete(COLLECTION_NAME)
c.close()
print("Deleted.")
EOF

# Then re-ingest
python ingest.py --chunks ../output/chunks/Baloch_2022_markdown_chunks.json
```
