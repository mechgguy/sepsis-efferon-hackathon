# Minimal Streamlit App Plan

Mirrors the layout and feel of the existing chatbot panel. Local only.

---

## 1. Objective

- Two tabs: **Chat** and **Documents** — same structure as current panel
- Chat tab: scrollable message history (user / assistant bubbles) + query input at bottom
- Documents tab: upload chunks JSON, ingest, show document list with status
- Retrieval: BM25 / vector / hybrid + reranker, results surfaced as assistant messages

---

## 2. File Structure

```
streamlit_app/
├── requirements.txt
└── app.py
```

Imports `rag_python/` directly as functions.

---

## 3. Dependencies

```
streamlit>=1.35.0
weaviate-client==3.26.7
sentence-transformers==3.3.1
torch>=2.0
```

---

## 4. Layout

```
┌─────────────────────────────────────────────────┐
│  [ Chat ]  [ Documents ]                        │  ← st.tabs
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌── user ────────────────────────────────────┐ │  ← st.chat_message("user")
│  │  what is the p-SOFA cut-off for mortality? │ │
│  └────────────────────────────────────────────┘ │
│                                                 │
│  ┌── assistant ───────────────────────────────┐ │  ← st.chat_message("assistant")
│  │  **Results (hybrid · reranked)**           │ │
│  │                                            │ │
│  │  ▸ [1] Results  (score 0.94)               │ │  ← st.expander, first open
│  │      For the prediction of 30-day          │ │
│  │      mortality at the cut-off value of     │ │
│  │      p-SOFA>2, the sensitivity was 93.87%  │ │
│  │                                            │ │
│  │  ▸ [2] Discussion  (score 0.87)            │ │  ← collapsed
│  │  ▸ [3] Abstract  (score 0.81)              │ │  ← collapsed
│  └────────────────────────────────────────────┘ │
│                                                 │
├─────────────────────────────────────────────────┤
│  ⚙ hybrid · rerank on · top-5 of 20 candidates │  ← st.caption (settings summary)
│  [ Ask something about the document... ] [send] │  ← st.chat_input
└─────────────────────────────────────────────────┘
```

**Documents tab:**

```
┌─────────────────────────────────────────────────┐
│  [ Chat ]  [ Documents ]                        │
├─────────────────────────────────────────────────┤
│  Upload chunks JSON                             │
│  [ Drop file here or Browse ]                   │
│                                                 │
│  ──────────────────────────────────────────     │
│  📄 Baloch_2022_markdown_chunks.json            │
│     22 chunks · Ingested ✓                      │
│     [ View chunks ]                             │
│                                                 │
│  ──────────────────────────────────────────     │
│  (empty state: "No documents yet. Upload a      │
│   chunks JSON file to get started.")            │
└─────────────────────────────────────────────────┘
```

---

## 5. Session State

```python
st.session_state.messages        # list of {role, content} — chat history
st.session_state.documents       # list of {filename, chunk_count, ingested: bool}
st.session_state.settings        # {mode, candidates, top_k, rerank}
```

`messages` persists across reruns within the session.
Uploading a new file appends to `documents`; it does not clear previous documents.

---

## 6. Chat Tab

### Message history

```python
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
```

### Assistant message format

Each assistant response is a markdown string assembled from the retrieved chunks:

```
**Results (hybrid · reranked)**

[1] Introduction  ·  score 0.94
> The pediatric intensive care unit plays an important role...

[2] Results  ·  score 0.87
> We included 286 children...
```

Rendered via `st.markdown()` inside `st.chat_message("assistant")`.
Each chunk result is a blockquote so it visually separates from the heading.

### Query input

```python
query = st.chat_input("Ask something about the document...")
```

`st.chat_input` is page-level — it always appears pinned at the bottom regardless of the
active tab. This matches the existing panel where the input is always visible.

### On submit

```python
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("Retrieving..."):
        chunks = retrieve(query, **st.session_state.settings)
    response = format_response(chunks, st.session_state.settings)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
```

### Settings (above input)

A `st.popover` or `st.expander` labelled "⚙ Settings" exposes:

```python
mode       = st.radio("Mode", ["hybrid", "bm25", "vector"], horizontal=True)
candidates = st.slider("Candidates", 5, 50, 20)
top_k      = st.slider("Top-K", 1, 20, 5)
rerank     = st.toggle("Rerank", value=True)
```

Below the expander, a single `st.caption` line summarises the current settings:

```
⚙ hybrid · rerank on · top-5 of 20 candidates
```

---

## 7. Documents Tab

### File uploader

```python
uploaded = st.file_uploader("Upload chunks JSON", type="json", label_visibility="collapsed")
```

On upload:
1. Parse JSON
2. Show document name, chunk count, a preview of the first chunk title
3. Show **Ingest** button

### Ingest button

```python
if st.button("Ingest", type="primary"):
    with st.status("Ingesting...", expanded=True) as status:
        st.write(f"Embedding {len(chunks)} chunks...")
        ingest_chunks(chunks)
        status.update(label="Done", state="complete")
    st.session_state.documents.append({
        "filename": uploaded.name,
        "chunk_count": len(chunks),
        "ingested": True
    })
```

`st.status` with `expanded=True` mirrors the progress feedback in the existing panel
(extraction progress bar, "Extracting... N%" text).

### Document list

Mirrors the Documents tab in the existing panel — one card per ingested file:

```python
for doc in st.session_state.documents:
    with st.container(border=True):
        col1, col2 = st.columns([8, 2])
        with col1:
            st.markdown(f"📄 **{doc['filename']}**")
            status = "✓ Ingested" if doc["ingested"] else "⏳ Pending"
            st.caption(f"{doc['chunk_count']} chunks · {status}")
        with col2:
            if st.button("View", key=doc["filename"]):
                st.session_state.viewing = doc["filename"]
```

Empty state (no documents yet):

```python
st.info("No documents yet. Upload a chunks JSON file to get started.")
```

### Chunk viewer

When "View" is clicked, show chunks in an expander below the document list:

```python
if st.session_state.get("viewing"):
    with st.expander(f"Chunks — {st.session_state.viewing}", expanded=True):
        for i, chunk in enumerate(chunks):
            st.markdown(f"**[{i}] {chunk['metadata']['section']}**")
            st.markdown(chunk["text"])
            st.divider()
```

---

## 8. Model Loading

```python
@st.cache_resource
def get_embedder():
    return Embedder()

@st.cache_resource
def get_reranker():
    return Reranker()

@st.cache_resource
def get_client():
    return weaviate.Client(url="http://localhost:8080")
```

Models and client load once. A `st.spinner("Loading models...")` is shown on first run only.

---

## 9. Weaviate Health Check

```python
try:
    get_client().is_ready()
except Exception:
    st.error("Weaviate not reachable at localhost:8080. Start the container first.")
    st.stop()
```

Runs on every page load. If Weaviate is down, the app stops immediately with a clear message.

---

## 10. Run

```bash
pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py
```

Opens at `http://localhost:8501`.
