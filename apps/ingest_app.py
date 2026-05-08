"""
apps/ingest_app.py — Sepsis Atlas unified Streamlit app.
Run with: streamlit run apps/ingest_app.py

Tabs:
  📄 Ingest   — PDF parsing, table correction, chunking
  📚 Documents — Weaviate ingestion of chunked papers
  💬 Chat      — Hybrid RAG retrieval over ingested chunks
"""
import sys
import copy
import re
import json
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from pipeline.config import ARTICLES_DIR, PARSED_CACHE_DIR
from pipeline.ocr import parse_pdf, ParsedPaper, ParsedTable, ParsedFigure
from pipeline.ocr_parallel import parse_all_parallel
from pipeline.chunking import (
    build_chunks_from_paper,
    write_chunks_json,
    write_sections_md,
    write_tables_json,
)

# ── Weaviate (graceful degradation if not running) ────────────────────────────

weaviate_available = False
try:
    from pipeline.weaviate.config import get_client
    from pipeline.weaviate.ingest import ingest_chunks, ingest_file
    from pipeline.weaviate.retrieve import retrieve as _retrieve
    _c = get_client()
    _c.is_ready()
    _c.close()
    weaviate_available = True
except Exception:
    pass


# ── Markdown table display helpers ────────────────────────────────────────────

_STAT_KEYWORDS = [
    "p value", "p-value", "risk difference", "95% ci", "95%ci",
    "hazard ratio", "odds ratio", "relative risk", "confidence interval",
    "percentage points", "mean ±", "median", "interquartile",
]


def _parse_md_table(md: str) -> list[list[str]]:
    rows = []
    for line in md.strip().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if re.match(r"^\|[-| :]+\|$", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) > 1:
            rows.append(cells)
    return rows


def _rows_to_md(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    header = "| " + " | ".join(padded[0]) + " |"
    sep    = "| " + " | ".join(["---"] * width) + " |"
    body   = "\n".join("| " + " | ".join(r) + " |" for r in padded[1:])
    return "\n".join([header, sep, body])


def _rows_to_df(rows: list[list[str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    seen: dict[str, int] = {}
    headers = []
    for col in padded[0]:
        if col in seen:
            seen[col] += 1
            headers.append(f"{col}.{seen[col]}")
        else:
            seen[col] = 0
            headers.append(col)
    return pd.DataFrame(padded[1:], columns=headers)


def _transpose(rows: list[list[str]], reverse: bool) -> list[list[str]]:
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    t = [list(col) for col in zip(*padded)]
    return [r[::-1] for r in t] if reverse else t


def _apply_mode(md: str, mode: str) -> list[list[str]]:
    rows = _parse_md_table(md)
    if mode == "down":
        return _transpose(rows, reverse=True)
    if mode == "up":
        return _transpose(rows, reverse=False)
    return rows


def _is_rotated(table: ParsedTable) -> tuple[bool, str]:
    rows = _parse_md_table(table.markdown.lower())
    if not rows:
        return False, ""
    first_row = " ".join(rows[0])
    for kw in _STAT_KEYWORDS:
        if kw in first_row:
            return True, f"stat keyword '{kw}' in row 0"
    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)
    if n_cols > n_rows * 2:
        return True, f"cols ({n_cols}) >> rows ({n_rows})"
    if len(rows) >= 3:
        first_col = [r[0] for r in rows[1:] if r]
        numeric = sum(1 for c in first_col if re.search(r"\d", c))
        first_row_text = all(not re.search(r"^\s*[\d.]+\s*$", c) for c in rows[0])
        if first_col and numeric / len(first_col) > 0.6 and first_row_text:
            return True, f"first col {numeric}/{len(first_col)} numeric, first row all-text"
    return False, ""


def _rotate_and_save(paper: ParsedPaper, table_idx: int, mode: str) -> ParsedPaper:
    paper = copy.deepcopy(paper)
    t = paper.tables[table_idx]
    rows = _apply_mode(t.markdown, mode)
    t.markdown = _rows_to_md(rows)
    write_tables_json(paper)
    write_chunks_json(paper)
    return paper


# ── Chat helpers ──────────────────────────────────────────────────────────────

def _format_response(chunks: list[dict], mode: str, rerank: bool) -> str:
    rerank_label = "reranked" if rerank else "no rerank"
    lines = [f"**Results ({mode} · {rerank_label})**\n"]
    for i, chunk in enumerate(chunks, 1):
        score = chunk.get("_rerank_score") or chunk.get("_score")
        score_str = f"  ·  score {score:.3f}" if score is not None else ""
        lines.append(f"**[{i}] {chunk['title']}{score_str}**")
        lines.append(f"> {chunk['compressedContent'][:600]}")
        lines.append("")
    return "\n".join(lines)


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "paper": None,
        "table_mode": {},
        "messages": [],
        "settings": {"mode": "hybrid", "candidates": 20, "top_k": 5, "rerank": True},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Sepsis Atlas", layout="wide")
_init_state()

st.title("🩺 Sepsis Atlas")
if not weaviate_available:
    st.caption(
        "⚠️ Weaviate not reachable — Documents ingestion and Chat disabled. "
        "Start it: `docker-compose up -d`"
    )

pdfs = sorted(ARTICLES_DIR.glob("*.pdf")) if ARTICLES_DIR.exists() else []

tab_ingest, tab_documents, tab_chat = st.tabs(["📄 Ingest", "📚 Documents", "💬 Chat"])

# ═══════════════════════════════════════════════════════════════
#  INGEST TAB
# ═══════════════════════════════════════════════════════════════

with tab_ingest:
    st.subheader("Parse / Load Paper")
    if not pdfs:
        st.error(f"No PDFs found in {ARTICLES_DIR}")
    else:
        selected = st.selectbox("Select paper", pdfs, format_func=lambda p: p.name)
        if st.button("Parse / Load", type="primary"):
            with st.spinner("Parsing (uses cache if available)..."):
                paper = parse_pdf(selected)
                write_sections_md(paper)
                write_tables_json(paper)
                write_chunks_json(paper)
                st.session_state["paper"] = paper
                st.session_state["table_mode"] = {}
            st.success(f"Loaded {paper.paper_id}")

    st.divider()
    st.subheader("📦 Batch Parse All Papers")

    cached   = [p for p in pdfs if (PARSED_CACHE_DIR / f"{p.stem}.json").exists()]
    uncached = [p for p in pdfs if p not in cached]
    st.caption(f"{len(cached)} cached · {len(uncached)} not yet parsed · {len(pdfs)} total")
    workers = st.slider("Parallel workers", 1, 6, 3,
                        help="Each worker builds its own Docling converter. Keep ≤4 on CPU-only.")

    if st.button("🚀 Parse All Papers", type="primary"):
        prog = st.progress(0)
        status_txt = st.empty()
        log_box = st.empty()
        log: list[str] = []

        def _cb(done, total, filename):
            prog.progress(done / total)
            status_txt.text(f"[{done}/{total}] {filename}")
            log.append(f"✅ {filename}")
            log_box.markdown("\n".join(log[-10:]))

        with st.spinner("Batch parsing in progress..."):
            successes, failures = parse_all_parallel(
                ARTICLES_DIR, max_workers=workers, progress_cb=_cb
            )
            for p in successes:
                write_sections_md(p)
                write_tables_json(p)
                write_chunks_json(p)

        prog.progress(1.0)
        status_txt.text("Done.")
        st.success(f"Parsed {len(successes)} papers.")
        if failures:
            st.error(f"{len(failures)} failed:")
            for fp, err in failures:
                st.code(f"{fp.name}: {err}")

    st.divider()
    st.subheader("📦 Chunk All Parsed Papers")

    parsed_jsons = [p for p in pdfs if (PARSED_CACHE_DIR / f"{p.stem}.json").exists()]
    unchunked    = [p for p in parsed_jsons
                    if not (PARSED_CACHE_DIR / f"{p.stem}_chunks.json").exists()]
    st.caption(f"{len(parsed_jsons)} parsed · {len(unchunked)} not yet chunked")

    if st.button("🔪 Chunk All Papers", type="primary"):
        if not parsed_jsons:
            st.warning("No parsed papers found. Parse first.")
        else:
            prog2 = st.progress(0)
            log2: list[str] = []
            box2 = st.empty()
            for i, p in enumerate(parsed_jsons, 1):
                paper = parse_pdf(p)
                out = write_chunks_json(paper)
                log2.append(f"✅ {p.name} → {out.name}")
                box2.markdown("\n".join(log2[-10:]))
                prog2.progress(i / len(parsed_jsons))
            st.success(f"Chunked {len(parsed_jsons)} papers.")

    st.divider()

    paper: ParsedPaper | None = st.session_state.get("paper")
    if paper is None:
        st.info("Select a PDF above and click Parse / Load.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    pid = paper.paper_id
    c1.metric("Paper ID", pid[:40] + "…" if len(pid) > 40 else pid)
    c2.metric("Sections", len(paper.sections))
    c3.metric("Tables",   len(paper.tables))
    c4.metric("Figures",  len(paper.figures))
    st.caption(
        f"📄 `{PARSED_CACHE_DIR / paper.paper_id}.md`  "
        f"🗃 `{PARSED_CACHE_DIR / paper.paper_id}_tables.json`"
    )
    st.divider()

    _CYCLE = ["original", "down", "up"]
    t_sec, t_tbl, t_fig, t_ch, t_raw = st.tabs(
        ["Sections", "Tables", "Figures", "Chunks", "Full Markdown"]
    )

    with t_sec:
        if not paper.sections:
            st.warning("No sections parsed.")
        for s in paper.sections:
            label = f"**{s.heading}** &nbsp; `p.{s.page_start}` &nbsp; ({len(s.text)} chars)"
            with st.expander(label):
                st.markdown(s.text[:2000] + ("…" if len(s.text) > 2000 else ""))

    with t_tbl:
        if not paper.tables:
            st.warning("No tables parsed.")
        for t in paper.tables:
            rotated, reason = _is_rotated(t)
            badge = " 🔴" if rotated else ""
            label = (f"Table {t.index}{badge} &nbsp; `p.{t.page_start}` "
                     f"&nbsp; — *{t.preceding_heading}*")
            with st.expander(label, expanded=rotated):
                tkey = f"{paper.paper_id}_t{t.index}"
                mode = st.session_state["table_mode"].get(tkey, "original")
                hcol, btn_col = st.columns([6, 1])
                with hcol:
                    if rotated:
                        st.warning(f"Rotation detected: {reason}")
                with btn_col:
                    next_mode = _CYCLE[(_CYCLE.index(mode) + 1) % len(_CYCLE)]
                    if st.button(f"↕ →{next_mode}", key=f"cycle_{tkey}"):
                        st.session_state["table_mode"][tkey] = next_mode
                        if next_mode != "original":
                            updated = _rotate_and_save(paper, t.index, next_mode)
                            st.session_state["paper"] = updated
                            st.toast(f"Table {t.index} rotated ({next_mode}) → saved")
                        st.rerun()
                st.caption(f"mode: {mode}")
                st.dataframe(
                    _rows_to_df(_apply_mode(t.markdown, mode)),
                    use_container_width=True,
                )

    with t_fig:
        if not getattr(paper, "figures", []):
            st.warning("No figures parsed.")
        for fig in paper.figures:
            with st.expander(f"Figure {fig.index} · p.{fig.page_start}"):
                if fig.caption:
                    st.caption(fig.caption)
                if fig.image_path and Path(fig.image_path).exists():
                    st.image(fig.image_path, use_container_width=True)
                else:
                    st.error(f"Missing image: {fig.image_path}")

    with t_ch:
        if st.button("Generate Chunks Preview"):
            chunks = build_chunks_from_paper(paper, max_chars=500)
            st.write(f"Generated {len(chunks)} chunks.")
            for c in chunks:
                meta = c["metadata"]
                with st.expander(f"📄 p.{meta['page_number']} | {meta['section']}"):
                    st.json(meta)
                    st.markdown(c["text"])

    with t_raw:
        st.text_area("full_markdown", paper.full_markdown[:10000], height=600,
                     label_visibility="collapsed")
        if len(paper.full_markdown) > 10000:
            st.caption(f"Showing first 10,000 of {len(paper.full_markdown)} chars.")

# ═══════════════════════════════════════════════════════════════
#  DOCUMENTS TAB
# ═══════════════════════════════════════════════════════════════

with tab_documents:
    if not weaviate_available:
        st.warning("Weaviate is not running. Start it with `docker-compose up -d`.")

    chunk_files = sorted(PARSED_CACHE_DIR.glob("*_chunks.json"))
    st.subheader(f"Chunked Papers ({len(chunk_files)} available)")

    if not chunk_files:
        st.info("No chunked papers yet. Go to the Ingest tab → Chunk All Papers.")
    else:
        if weaviate_available and st.button("⬆ Ingest All to Weaviate", type="primary"):
            with st.status("Ingesting all chunks...", expanded=True) as wv_status:
                total = 0
                for cf in chunk_files:
                    chunks = json.loads(cf.read_text())
                    st.write(f"  {cf.name}: {len(chunks)} chunks")
                    ingest_chunks(chunks)
                    total += len(chunks)
                wv_status.update(label=f"Done — {total} chunks ingested", state="complete")

        for cf in chunk_files:
            with st.container(border=True):
                col1, col2 = st.columns([8, 2])
                with col1:
                    try:
                        n_chunks = len(json.loads(cf.read_text()))
                    except Exception:
                        n_chunks = "?"
                    st.markdown(f"📄 **{cf.name}**")
                    st.caption(f"{n_chunks} chunks")
                with col2:
                    if weaviate_available and st.button("Ingest", key=f"ingest_{cf.name}"):
                        with st.spinner(f"Ingesting {cf.name}..."):
                            ingest_file(cf)
                        st.toast(f"✓ Ingested {cf.name}")

    st.divider()
    st.subheader("Upload External Chunks JSON")
    uploaded = st.file_uploader("Upload chunks JSON", type="json",
                                label_visibility="collapsed")
    if uploaded:
        up_chunks = json.loads(uploaded.read())
        st.write(f"{len(up_chunks)} chunks")
        if up_chunks:
            st.caption(f"First section: `{up_chunks[0]['metadata'].get('section', '?')}`")
        if weaviate_available and st.button("Ingest Uploaded File", type="primary"):
            with st.status("Ingesting...", expanded=True) as wv_status2:
                st.write(f"Embedding {len(up_chunks)} chunks...")
                ingest_chunks(up_chunks)
                wv_status2.update(label="Done", state="complete")

# ═══════════════════════════════════════════════════════════════
#  CHAT TAB
# ═══════════════════════════════════════════════════════════════

with tab_chat:
    if not weaviate_available:
        st.warning("Weaviate is not running. Start it with `docker-compose up -d`.")

    with st.expander("⚙ Settings", expanded=False):
        s = st.session_state["settings"]
        s["mode"] = st.radio(
            "Mode", ["hybrid", "bm25", "vector"], horizontal=True,
            index=["hybrid", "bm25", "vector"].index(s["mode"]),
        )
        col_a, col_b = st.columns(2)
        s["candidates"] = col_a.slider("Candidates", 5, 50, s["candidates"])
        s["top_k"]      = col_b.slider("Top-K", 1, 20, s["top_k"])
        s["rerank"]     = st.toggle("Rerank", value=s["rerank"])

    st.caption(
        f"⚙ {st.session_state['settings']['mode']} · "
        f"{'rerank on' if st.session_state['settings']['rerank'] else 'rerank off'} · "
        f"top-{st.session_state['settings']['top_k']} of "
        f"{st.session_state['settings']['candidates']} candidates"
    )

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Page-level chat input — always pinned at the bottom of every tab
query = st.chat_input("Ask something about the documents...")
if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    if not weaviate_available:
        response = "⚠️ Weaviate is not running. Start it with `docker-compose up -d`."
    else:
        s = st.session_state["settings"]
        with st.spinner("Retrieving..."):
            try:
                chunks = _retrieve(
                    query,
                    s["mode"],
                    s["top_k"],
                    s["candidates"],
                    s["rerank"],
                )
                response = _format_response(chunks, s["mode"], s["rerank"])
            except Exception as e:
                response = f"⚠️ Retrieval error: {e}"
    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.rerun()
