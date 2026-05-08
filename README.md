# Sepsis Atlas

End-to-end pipeline for ingesting clinical sepsis research papers, building a vector index, and running hybrid RAG retrieval — all through a unified Streamlit interface.

---

## 📺 Product Demo

![Sepsis Atlas Demo](demo/frontToken_burners.gif)

<details>
  <summary>▶ Click to expand Video Demo</summary>
  <video src="demo/frontToken_burners.gif" width="100%" controls title="Fronend Dashboard Walkthrough"></video>
</details>

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Pipeline Stages](#pipeline-stages)
5. [Setup](#setup)
6. [Running the Apps](#running-the-apps)
7. [CLI Reference](#cli-reference)
8. [Chunk Schema](#chunk-schema)
9. [Configuration](#configuration)
10. [Caching & Reset](#caching--reset)
11. [Known Limitations](#known-limitations)

---

## Overview

Sepsis Atlas ingests PDF research papers, parses them with layout-aware OCR, splits them into structured chunks, embeds them with a sentence transformer, and stores them in Weaviate for hybrid BM25 + vector retrieval with cross-encoder reranking.

The system has two Streamlit frontends:

- **`apps/ingest_app.py`** — parse papers, manage Weaviate ingestion, query via chat
- **`frontend/dashboard_viewer_graph.py`** — SEPSIS ATLAS DASHBOARD and viewer for attribute analytics and full-text traceability across all papers

---

## Architecture

### End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  INPUT                                                                  │
│  materials/articles/*.pdf                                               │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — OCR + PARSING          pipeline/ocr.py                       │
│                                                                         │
│  Docling layout-aware PDF parser                                        │
│  ├─ Extracts section headings + body text with page numbers             │
│  ├─ Extracts tables as Markdown with preceding heading                  │
│  └─ Extracts figures as cropped PNG images (PyMuPDF)                    │
│                                                                         │
│  Output → data/parsed_papers/{paper_id}.json   (full parse cache)       │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — MARKDOWN + TABLE EXPORT    pipeline/chunking.py              │
│                                                                         │
│  ├─ Sections exported as .md (no tables)                                │
│  └─ Tables exported as JSON list                                        │
│     └─ Manual rotation correction available in UI                       │
│                                                                         │
│  Output → data/parsed_papers/{paper_id}.md                              │
│           data/parsed_papers/{paper_id}_tables.json                     │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — CHUNKING + METADATA ENRICHMENT    pipeline/chunking.py       │
│                                                                         │
│  Structure-aware splitting strategy:                                    │
│  ├─ heading  → section fits in one chunk                                │
│  ├─ paragraph → split by blank lines, pack into max_chars               │
│  ├─ sentence  → split by sentence boundary                              │
│  └─ char      → hard split fallback                                     │
│                                                                         │
│  Each chunk carries:                                                    │
│  id, text, section, section_index, part_index, page_number, year        │
│                                                                         │
│  Output → data/parsed_papers/{paper_id}_chunks.json                     │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 4 — EMBEDDING + WEAVIATE INGESTION    pipeline/weaviate/         │
│                                                                         │
│  Embedder (sentence-transformers/all-MiniLM-L6-v2 by default)           │
│  ├─ Encodes all chunk texts in batches                                  │
│  └─ L2-normalised float vectors                                         │
│                                                                         │
│  Weaviate (localhost:8080 via Docker)                                   │
│  └─ Collection: RagDocumentChunk                                        │
│     Fields: chunkId, title, chunkIndex, chapterIndex,                   │
│             compressedContent, pageNumber, shortSummary, fullSummary    │
│                                                                         │
│  Output → Weaviate persistent volume (weaviate_data)                    │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 5 — RETRIEVAL    pipeline/weaviate/retrieve.py                   │
│                                                                         │
│  Step 1 — Fetch candidates from Weaviate                                │
│  ├─ bm25    → keyword match (inverted index)                            │
│  ├─ vector  → nearest neighbour by cosine similarity                    │
│  └─ hybrid  → Weaviate fusion of BM25 + vector scores                   │
│                                                                         │
│  Step 2 — Rerank (optional)                                             │
│  └─ BAAI/bge-reranker-v2-m3 cross-encoder scores each (query, chunk)    │
│                                                                         │
│  Returns → list[dict] of top-K chunks with scores                       │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 6 — FRONTEND    apps/                                            │
│                                                                         │
│  apps/ingest_app.py                                                     │
│  ├─ Ingest tab      → parse, rotate tables, chunk papers                │
│  ├─ Documents tab   → ingest chunks to Weaviate                         │
│  └─ Chat tab        → hybrid RAG query interface                        │
│                                                                         │
│                                                                         │
│  frontend/dashboard_viewer_graph.py                                     │
│  └─ Attribute analytics, traceability search, paper browser             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Module Dependency Graph

```
apps/ingest_app.py
  ├── pipeline.config          (paths, env vars)
  ├── pipeline.ocr             (parse_pdf, ParsedPaper)
  ├── pipeline.ocr_parallel    (parse_all_parallel)
  ├── pipeline.chunking        (build_chunks_from_paper, write_*)
  ├── pipeline.weaviate.ingest (ingest_chunks, ingest_file)
  └── pipeline.weaviate.retrieve (retrieve)

apps/dashboard.py
  └── pipeline.config          (PARSED_CACHE_DIR)

pipeline.ocr
  └── pipeline.config          (PARSED_CACHE_DIR, FIGURE_DIR)

pipeline.ocr_parallel
  └── pipeline.ocr             (build_converter, parse_pdf)

pipeline.chunking
  ├── pipeline.ocr             (ParsedPaper)
  └── pipeline.config          (PARSED_CACHE_DIR)

pipeline.weaviate.ingest
  ├── pipeline.config          (COLLECTION_NAME, PARSED_CACHE_DIR)
  ├── pipeline.weaviate.config (get_client)
  ├── pipeline.weaviate.embedder (Embedder)
  └── pipeline.weaviate.schema (CHUNK_SCHEMA)

pipeline.weaviate.retrieve
  ├── pipeline.config          (COLLECTION_NAME)
  ├── pipeline.weaviate.config (get_client)
  ├── pipeline.weaviate.embedder (Embedder)
  └── pipeline.weaviate.reranker (Reranker)

# Legacy shims (root level) — all re-export from pipeline.*
ingest.py           → pipeline.ocr + pipeline.config
ingest_parallel.py  → pipeline.ocr_parallel
chunking_strategy.py → pipeline.chunking
```

---

### Retrieval Flow (Chat Tab)

```
User query
    │
    ▼
Embedder.embed(query)  →  float[384]
    │
    ├──── BM25 ────────────────────────────────┐
    │     Weaviate keyword search              │
    │                                          │
    ├──── Vector ──────────────────────────────┤  top-N candidates
    │     cosine similarity                    │
    │                                          │
    └──── Hybrid (default) ────────────────────┘
          Weaviate score fusion
              │
              ▼
         [Optional] Reranker
         CrossEncoder(query, chunk_text) per candidate
         → sorted by cross-encoder score
              │
              ▼
         top-K chunks returned
              │
              ▼
         Formatted as assistant message
         displayed in Chat tab
```

---

## Directory Structure

```
Sepsis_hackathon/
│
├── materials/
│   └── articles/                  ← Input PDFs
│       ├── Baloch_2022.pdf
│       ├── Seymour_2016.pdf
│       └── ...
│
├── data/
│   └── parsed_papers/             ← All pipeline outputs (auto-created)
│       ├── {paper_id}.json        ← Full Docling parse cache
│       ├── {paper_id}.md          ← Sections text only (no tables)
│       ├── {paper_id}_tables.json ← Tables list (overwritten on rotation)
│       ├── {paper_id}_chunks.json ← Embedding-ready chunks
│       └── figures/               ← Extracted figure images
│           └── {paper_id}_fig0.png
│
├── pipeline/                      ← Canonical pipeline package
│   ├── __init__.py
│   ├── config.py                  ← Centralized paths + env config
│   ├── ocr.py                     ← PDF → ParsedPaper (Docling)
│   ├── ocr_parallel.py            ← Thread-safe parallel parsing
│   ├── chunking.py                ← Chunking + file writers
│   └── weaviate/                  ← Weaviate integration subpackage
│       ├── __init__.py
│       ├── config.py              ← get_client()
│       ├── embedder.py            ← BGE sentence-transformer
│       ├── reranker.py            ← BGE cross-encoder reranker
│       ├── schema.py              ← Collection schema + field extractors
│       ├── ingest.py              ← Chunk → Weaviate ingestion
│       ├── retrieve.py            ← BM25 / vector / hybrid retrieval
│       └── verify.py              ← Inspect Weaviate contents
│
├── apps/                          ← Streamlit frontends
│   ├── ingest_app.py              ← Main app (Ingest + Documents + Chat)
│   └── dashboard.py               ← Attribute analytics dashboard
│
├── docker-compose.yml             ← Weaviate container
├── requirements.txt               ← All dependencies
├── README.md
│
├── ingestion_rag/                 ← Legacy (superseded by pipeline/)
│   ├── rag_python/                ← Old RAG scripts (kept for reference)
│   │   ├── extract.py             ← LLM evidence extraction (OpenRouter)
│   │   └── ...
│   └── docker-compose.yml         ← Old compose (use root one instead)
│
└── Legacy shims (root level — keep test_ingest.py working)
    ├── ingest.py              → pipeline.ocr
    ├── ingest_parallel.py     → pipeline.ocr_parallel
    ├── chunking_strategy.py   → pipeline.chunking
    └── test_ingest.py         ← Original ingest viewer (still functional)
```

---

## Pipeline Stages

### Stage 1 — OCR / Parsing (`pipeline/ocr.py`)

Uses [Docling](https://github.com/DS4SD/docling) for layout-aware PDF parsing.

**Input:** Any PDF in `materials/articles/`

**Output files in `data/parsed_papers/`:**
- `{paper_id}.json` — full parse cache (never overwritten; delete to force re-parse)

**ParsedPaper dataclass:**
```
ParsedPaper
  ├── paper_id: str
  ├── sections: list[ParsedSection]
  │     └── heading, text, page_start
  ├── tables: list[ParsedTable]
  │     └── index, preceding_heading, markdown, page_start
  ├── figures: list[ParsedFigure]
  │     └── index, caption, page_start, image_path
  └── full_markdown: str
```

Parallel parsing uses one Docling converter per thread to avoid shared state.

---

### Stage 2 — Markdown + Table Export (`pipeline/chunking.py`)

**Output files:**
- `{paper_id}.md` — sections text only, no tables, used for LLM context
- `{paper_id}_tables.json` — list of table objects

Tables can be manually rotated (corrected) in the Ingest app UI. Rotating auto-overwrites `_tables.json` and regenerates `_chunks.json`.

---

### Stage 3 — Chunking (`pipeline/chunking.py`)

`build_chunks_from_paper(paper, max_chars=1800)` splits each section body using a cascade strategy:

```
Section body
  │
  ├── fits in max_chars?  →  single chunk  (split_strategy: "heading")
  │
  ├── multiple paragraphs?  →  pack paragraphs  (split_strategy: "paragraph")
  │
  ├── multiple sentences?   →  pack sentences   (split_strategy: "sentence")
  │
  └── none of the above    →  hard char split  (split_strategy: "char")
```

**Output file:** `{paper_id}_chunks.json`

---

### Stage 4 — Embedding + Weaviate Ingestion (`pipeline/weaviate/`)

The Weaviate collection `RagDocumentChunk` stores one object per chunk:

```
RagDocumentChunk
  ├── chunkId           ← chunk["id"]
  ├── title             ← metadata["section"]
  ├── chunkIndex        ← metadata["section_index"]
  ├── chapterIndex      ← metadata["part_index"]
  ├── compressedContent ← chunk["text"]
  ├── pageNumber        ← metadata["page_number"]
  ├── shortSummary      ← "" (reserved)
  └── fullSummary       ← "" (reserved)
  + vector              ← BGE embedding of chunk["text"]
```

Weaviate uses `self_provided` vectors — embeddings are computed client-side before upload.

---

### Stage 5 — Retrieval (`pipeline/weaviate/retrieve.py`)

```python
retrieve(
    query: str,
    mode: str = "hybrid",     # "hybrid" | "bm25" | "vector"
    top_k: int = 5,           # final results after reranking
    candidates: int = 20,     # candidates fetched from Weaviate
    rerank: bool = True,      # apply cross-encoder reranker
) -> list[dict]
```

Returns a list of chunk dicts with `compressedContent`, `title`, `page_number`, `_score`, `_rerank_score`.

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd Sepsis_hackathon
pip install -r requirements.txt
```

Core dependencies:

| Library | Purpose |
|---|---|
| `docling` | Layout-aware PDF parsing + table extraction |
| `PyMuPDF` | Figure extraction from PDF pages |
| `streamlit` | App UI |
| `weaviate-client>=4.6.0` | Weaviate v4 Python client |
| `sentence-transformers` | BGE embedder + reranker |
| `torch` | Backend for sentence-transformers |
| `plotly` | Charts in dashboard |
| `openai` | OpenRouter LLM calls (extract.py only) |

### 2. Start Weaviate

```bash
docker-compose up -d
```

Weaviate runs at `http://localhost:8080`. Data is persisted in the `weaviate_data` Docker volume.

To stop:
```bash
docker-compose down
```

To wipe all indexed data:
```bash
docker-compose down -v
```

---

## Running the Apps

### Main App (Ingest + Documents + Chat)

```bash
streamlit run apps/ingest_app.py
```

Opens at `http://localhost:8501`

```
┌─────────────────────────────────────────────────────────┐
│  📄 Ingest   📚 Documents   💬 Chat                     │  ← tabs
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [Ingest tab]                                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │ Select paper: [ Baloch_2022.pdf          ▾ ]    │    │
│  │ [ Parse / Load ]                                │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 📦 Batch Parse All Papers                       │    │
│  │ Parallel workers: [──●──────] 3                 │    │
│  │ [ 🚀 Parse All Papers ]                         │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 📦 Chunk All Parsed Papers                      │    │
│  │ [ 🔪 Chunk All Papers ]                         │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ── Paper Viewer ───────────────────────────────────    │
│  Sections | Tables | Figures | Chunks | Full Markdown   │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  [Documents tab]                                        │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 📄 Baloch_2022_chunks.json    22 chunks         │    │
│  │                            [ Ingest ]           │    │
│  ├─────────────────────────────────────────────────┤    │
│  │ 📄 Seymour_2016_chunks.json   31 chunks         │    │
│  │                            [ Ingest ]           │    │
│  └─────────────────────────────────────────────────┘    │
│  [ ⬆ Ingest All to Weaviate ]                           │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  [Chat tab]                                             │
│  ⚙ Settings  ▾                                          │
│  Mode: ● hybrid  ○ bm25  ○ vector                       │
│  Candidates: [──●──] 20    Top-K: [──●──] 5             │
│  Rerank: [●]                                            │
│                                                         │
│  ┌── assistant ─────────────────────────────────────┐   │
│  │  **Results (hybrid · reranked)**                 │   │
│  │  [1] Introduction  ·  score 0.94                 │   │
│  │  > The pediatric intensive care unit plays...    │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  ⚙ hybrid · rerank on · top-5 of 20 candidates          │
│  [ Ask something about the documents...     ] [send]    │
└─────────────────────────────────────────────────────────┘
```

**Chat input is page-level** — it stays pinned at the bottom regardless of which tab is active. Responses are stored in session state and displayed in the Chat tab.

If Weaviate is not running, the Documents and Chat features show a warning but the Ingest tab continues to work.

---

### Dashboard (Attribute Analytics)

```bash
streamlit run apps/dashboard.py
```

```
┌─────────────────────────────────────────────────────────────┐
│  🌐 Sepsis Global Attribute Atlas                           │
├────────────────┬────────────────────────────────────────────┤
│  ⚙️ Sidebar    │  📊 Analytics  🔍 Traceability  📖 Atlas    │
│  Active Papers │                                            │
│  [✓] Baloch   │  ┌── Clinical Coverage ──┐  ┌── Density ──┐ │
│  [✓] Seymour  │  │ Sunburst: Domain →    │  │ Horizontal  │ │
│  [✓] ...      │  │ Attribute → paper_id  │  │ bar by paper│ │
│               │  └───────────────────────┘  └─────────────┘ │
│               │                                             │
│               │  [Traceability tab]                         │
│               │  Search: [ lactate          ]               │
│               │  Found 12 mentions                          │
│               │  ▸ Baloch_2022 | Results                    │
│               │  ▸ Seymour_2016 | Methods                   │
│               │                                             │
│               │  [Atlas tab]                                │
│               │  Paper: [ Baloch_2022 ▾ ]                   │
│               │  ● Abstract   ### Abstract                  │
│               │  ○ Methods    Objective: To assess...       │
│               │  ○ Results                                  │
└────────────────┴────────────────────────────────────────────┘
```

Discovers clinical attributes (severity scores, biomarkers, demographics, treatments, outcomes) by keyword scan of all parsed paper sections.

---

### Legacy Ingest Viewer

```bash
streamlit run test_ingest.py
```

The original single-paper viewer. Still fully functional — it imports from the root-level shims which delegate to `pipeline.*`.

---

## CLI Reference

All pipeline modules can be run directly with `-m`:

### Ingest chunks into Weaviate

```bash
# All *_chunks.json from data/parsed_papers/ (most common)
python -m pipeline.weaviate.ingest --all

# Single file
python -m pipeline.weaviate.ingest --chunks data/parsed_papers/Baloch_2022_chunks.json

# Custom folder
python -m pipeline.weaviate.ingest --folder /path/to/chunks/
```

### Retrieve

```bash
rm data/parsed_papers/{paper_id}_tables.json
```

## File Format Details

## Methods and Files

`tablejsonviewer.py` Interactive view of markdown and json (non networkX) files. Gives best visualisation and analysis with `./data/parsed_papers`

`graph_viewer.py` Interactive view of json files stored in networkx format in streamlit

`dashboard_viewer_graph.py` Sepsis Global Attribute Atlas
![alt text](image.png)

### `{paper_id}.md`

Sections text only. No tables. Used for text chunking and LLM context.

```markdown
## Abstract
Sepsis is defined as life-threatening organ dysfunction...

## Methods
...
```

### `{paper_id}_tables.json`

List of table objects. Overwritten when you rotate a table in the UI.
# Default: hybrid retrieval with reranking
python -m pipeline.weaviate.retrieve --query "SOFA score and 28-day mortality"

# With options
python -m pipeline.weaviate.retrieve \
  --query "lactate clearance septic shock" \
  --mode hybrid \
  --candidates 30 \
  --top-k 8 \
  --no-rerank
```

### Verify Weaviate index

```bash
python -m pipeline.weaviate.verify
python -m pipeline.weaviate.verify --limit 20
```

---

## Chunk Schema

Every file matching `data/parsed_papers/*_chunks.json` contains a JSON array. Each element:

```json
{
  "id": "Baloch_2022_s3_p0",
  "text": "## Introduction\nThe pediatric intensive care unit...",
  "metadata": {
    "document_id":    "Baloch_2022",
    "paper_id":       "Baloch_2022",
    "year":           2022,
    "section":        "Introduction",
    "section_index":  3,
    "part_index":     0,
    "page_number":    2,
    "chunk_type":     "section",
    "split_strategy": "paragraph"
  }
}
```

ID format: `{paper_id}_s{section_index}_p{part_index}`

How metadata maps to Weaviate fields:

| Chunk metadata field | Weaviate field | Notes |
|---|---|---|
| `id` | `chunkId` | stable identifier |
| `section` | `title` | section heading |
| `section_index` | `chunkIndex` | position in paper |
| `part_index` | `chapterIndex` | sub-chunk position |
| `text` | `compressedContent` | full chunk text + heading prefix |
| `page_number` | `pageNumber` | 1-based |

---

## Configuration

All paths and service settings are defined in `pipeline/config.py` and can be overridden with environment variables:

```python
# Paths (derived from repo root — always absolute)
REPO_ROOT        = Path(__file__).resolve().parent.parent
ARTICLES_DIR     = REPO_ROOT / "materials" / "articles"
PARSED_CACHE_DIR = REPO_ROOT / "data" / "parsed_papers"
FIGURE_DIR       = PARSED_CACHE_DIR / "figures"

# Weaviate
WEAVIATE_HOST    = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT    = int(os.getenv("WEAVIATE_PORT", "8080"))
COLLECTION_NAME  = "RagDocumentChunk"

# Models
BGE_MODEL_NAME      = os.getenv("BGE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
BATCH_SIZE          = int(os.getenv("BATCH_SIZE", "32"))
```

To switch to a higher-quality embedding model:

```bash
export BGE_MODEL_NAME="BAAI/bge-m3"
streamlit run apps/ingest_app.py
```

---

## Caching & Reset

### Re-parse a single paper

```bash
rm data/parsed_papers/Baloch_2022.json
# Then Parse / Load in the UI or re-run pipeline
```

### Reset a rotated table correction

```bash
rm data/parsed_papers/Baloch_2022_tables.json
```

### Regenerate chunks (after table correction or config change)

```bash
rm data/parsed_papers/Baloch_2022_chunks.json
# Then Chunk All Papers in the UI
```

### Wipe the Weaviate index completely

```bash
docker-compose down -v
docker-compose up -d
```

---

## Known Limitations

- **Rotated tables** — Docling does not auto-detect rotated tables. Use the ↕ button in the viewer to transpose. The fix is saved to `_tables.json` and propagates to chunks automatically.
- **Figure captions** — Captions adjacent to figures are not reliably extracted; `caption` field is empty in most cases.
- **Superscripts / subscripts** — Flattened to plain text by Docling.
- **Weaviate required for Chat/Documents** — Both tabs degrade to a warning if Weaviate is not reachable.
- **Models download on first run** — The BGE embedding model (~90 MB for MiniLM, ~2.3 GB for bge-m3) and reranker (~600 MB for bge-reranker-v2-m3) are downloaded from HuggingFace on first use.
- **CPU-only inference** — Both embedder and reranker run on CPU by default. Set `device="cuda"` in `pipeline/weaviate/embedder.py` and `reranker.py` to use GPU.
