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
import glob
import os
from dataclasses import asdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
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

from dotenv import load_dotenv
load_dotenv()

llm_available = bool(
    os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
)
if llm_available:
    try:
        from pipeline.extract import chat_answer
    except Exception:
        llm_available = False

# ── Dashboard domain map & data loading ──────────────────────────────────────

DOMAIN_MAP = {
    "Severity Scores": ["sofa", "apache", "saps", "qsofa", "mews"],
    "Biomarkers":      ["lactate", "il-6", "crp", "procalcitonin", "leukocyte", "creatinine", "bilirubin", "plt"],
    "Demographics":    ["age", "sex", "male", "female", "bmi", "weight", "ethnicity", "comorbidity"],
    "Treatment":       ["vasopressor", "norepinephrine", "fluid", "antibiotic", "ventilation", "dialysis"],
    "Outcomes":        ["mortality", "death", "discharge", "icu stay", "readmission"],
}


def _extract_attributes(text: str) -> list[dict]:
    found = []
    text_lower = text.lower()
    for domain, keywords in DOMAIN_MAP.items():
        for kw in keywords:
            if kw in text_lower:
                found.append({"Domain": domain, "Attribute": kw.capitalize()})
    return found


@st.cache_data
def _load_dashboard_data():
    all_sections, all_attributes, all_links = [], [], []
    if not PARSED_CACHE_DIR.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for f_path in glob.glob(str(PARSED_CACHE_DIR / "*_chunks.json")):
        try:
            data = json.loads(Path(f_path).read_text())
            if isinstance(data, str):
                data = json.loads(data)
            if not isinstance(data, list):
                continue
            for chunk in data:
                if not isinstance(chunk, dict):
                    continue
                content  = chunk.get("text", "")
                metadata = chunk.get("metadata", {})
                paper_id = metadata.get("paper_id", "Unknown")
                heading  = metadata.get("section", "Untitled")
                page_num = metadata.get("page_number", "N/A")
                all_sections.append({"paper_id": paper_id, "heading": heading,
                                      "text": content, "page": page_num})
                found = _extract_attributes(content)
                unique_attrs = list({item["Attribute"] for item in found})
                for item in found:
                    all_attributes.append({"paper_id": paper_id, "Domain": item["Domain"],
                                           "Attribute": item["Attribute"],
                                           "Section": heading, "Page": page_num})
                for i in range(len(unique_attrs)):
                    for j in range(i + 1, len(unique_attrs)):
                        all_links.append({"source": unique_attrs[i],
                                          "target": unique_attrs[j],
                                          "paper_id": paper_id})
        except Exception as e:
            print(f"[WARN] {f_path}: {e}")
    df_text  = pd.DataFrame(all_sections)
    df_attr  = pd.DataFrame(all_attributes)
    df_links = pd.DataFrame(all_links, columns=["source", "target", "paper_id"])
    return df_text, df_attr, df_links


def _format_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant chunks found."
    lines = [f"**{len(chunks)} relevant passage(s):**\n"]
    for i, c in enumerate(chunks, 1):
        score = c.get("_rerank_score") or c.get("_score")
        score_str = f"  ·  score {score:.3f}" if score is not None else ""
        lines.append(f"**[{i}] {c.get('title', '?')}{score_str}**")
        lines.append(f"> {c.get('compressedContent', '')[:500]}")
        lines.append("")
    return "\n".join(lines)


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


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "paper": None,
        "table_mode": {},
        "messages": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Sepsis Atlas", layout="wide")
_init_state()

st.markdown("""
<style>
/* Chat tab: right column as styled panel */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child {
    background: #44546a;
    border: 2px solid #4effd0;
    border-radius: 12px;
    padding: 16px !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child * {
    color: #ffffff;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child
  div[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.07);
    border-radius: 8px;
    border: 1px solid rgba(78,255,208,0.2);
    margin-bottom: 8px;
}
div[data-testid="stChatInput"] textarea {
    background: rgba(68,84,106,0.8);
    border: 1px solid #4effd0;
    color: #ffffff;
}
div[data-testid="stChatInput"] button { background: #4effd0; color: #44546a; }
button[data-testid="stTab"][aria-selected="true"] {
    border-bottom: 2px solid #4effd0 !important;
    color: #4effd0 !important;
}
* { scrollbar-color: #4effd0 rgba(255,255,255,0.1); scrollbar-width: thin; }
</style>
""", unsafe_allow_html=True)

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
#  CHAT TAB — Dashboard (left) + Chatbot panel (right)
# ═══════════════════════════════════════════════════════════════

with tab_chat:
    df_text, df_attr, df_links = _load_dashboard_data()

    col_dash, col_chat = st.columns([3, 1.3])

    # ── LEFT: full 5-tab dashboard ────────────────────────────
    with col_dash:
        if df_text.empty:
            st.info(f"No chunked papers in `{PARSED_CACHE_DIR}`. Use the Ingest tab to parse and chunk papers.")
        else:
            paper_list = sorted(df_text["paper_id"].unique())
            selected_papers = st.multiselect(
                "Filter papers", paper_list, default=paper_list, key="dash_paper_filter"
            )
            f_text  = df_text[df_text["paper_id"].isin(selected_papers)]
            f_attr  = df_attr[df_attr["paper_id"].isin(selected_papers)]
            f_links = (df_links[df_links["paper_id"].isin(selected_papers)]
                       if not df_links.empty else pd.DataFrame(columns=["source", "target", "paper_id"]))

            tab_viz, tab_search, tab_explain, tab_consensus, tab_atlas = st.tabs([
                "📊 Attribute Analytics", "🔍 Traceability",
                "🧠 Explainability & Graph", "🤝 Cross-Study Consensus", "📖 Literature Atlas",
            ])

            with tab_viz:
                st.subheader("🎯 Clinical Attribute Coverage")
                if not f_attr.empty:
                    fig_sun = px.sunburst(f_attr, path=["Domain", "Attribute", "paper_id"],
                                          color="Domain", template="plotly_dark")
                    fig_sun.update_layout(height=700)
                    st.plotly_chart(fig_sun, use_container_width=True)
                st.divider()
                st.subheader("📊 Information Density")
                if not f_attr.empty:
                    density = f_attr.groupby(["paper_id", "Domain"]).size().reset_index(name="Mention Count")
                    fig_bar = px.bar(density, x="Mention Count", y="paper_id", color="Domain",
                                     orientation="h", template="plotly_dark")
                    fig_bar.update_layout(height=max(400, len(selected_papers) * 30))
                    st.plotly_chart(fig_bar, use_container_width=True)

            with tab_search:
                st.subheader("📋 Extracted Clinical Variables & Evidence")
                search_var = st.text_input("🔍 Search Variable Name", placeholder="e.g., Lactate")
                display_df = f_attr.copy()
                if search_var:
                    display_df = display_df[display_df["Attribute"].str.contains(search_var, case=False, na=False)]
                registry_table = display_df.rename(columns={
                    "Attribute": "Variable", "paper_id": "Source Paper",
                    "Section": "Section Heading", "Page": "Pg #",
                })
                if not registry_table.empty:
                    st.dataframe(
                        registry_table[["Domain", "Variable", "Source Paper", "Pg #", "Section Heading"]],
                        use_container_width=True, hide_index=True,
                        column_config={
                            "Pg #":            st.column_config.NumberColumn("Pg #", format="%d"),
                            "Variable":        st.column_config.TextColumn("Variable", width="medium"),
                            "Source Paper":    st.column_config.TextColumn("Source Paper", width="medium"),
                            "Section Heading": st.column_config.TextColumn("Evidence Context", width="large"),
                        },
                    )

            with tab_explain:
                c1, c2 = st.columns(2)
                with c1:
                    st.write("**Attribute Consensus Matrix**")
                    if not f_attr.empty:
                        matrix_data = f_attr.groupby(["Attribute", "paper_id"]).size().unstack(fill_value=0)
                        truncated_ids = [pid[:30] + "…" if len(pid) > 30 else pid
                                         for pid in matrix_data.columns]
                        fig_heat = px.imshow(matrix_data, color_continuous_scale="Viridis", aspect="auto")
                        fig_heat.update_layout(
                            xaxis=dict(tickmode="array",
                                       tickvals=list(range(len(matrix_data.columns))),
                                       ticktext=truncated_ids, tickangle=45, automargin=True),
                            yaxis=dict(tickmode="linear", dtick=1, automargin=True),
                            height=800, margin=dict(l=150, b=150),
                        )
                        st.plotly_chart(fig_heat, use_container_width=True)
                with c2:
                    st.write("**Clinical Relationship Graph**")
                    if not f_links.empty:
                        edge_df = f_links.groupby(["source", "target"]).size().reset_index(name="weight")
                        G = nx.from_pandas_edgelist(edge_df, "source", "target", ["weight"])
                        pos = nx.spring_layout(G, k=0.6, seed=42)
                        edge_x, edge_y = [], []
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
                        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                                                line=dict(width=1, color="#555"), mode="lines")
                        node_trace = go.Scatter(
                            x=[pos[n][0] for n in G.nodes()],
                            y=[pos[n][1] for n in G.nodes()],
                            mode="markers+text", text=list(G.nodes()),
                            textposition="top center",
                            marker=dict(size=12, color="#4effd0"),
                        )
                        fig_graph = go.Figure(
                            data=[edge_trace, node_trace],
                            layout=go.Layout(showlegend=False, height=800,
                                             xaxis_visible=False, yaxis_visible=False,
                                             paper_bgcolor="rgba(0,0,0,0)",
                                             plot_bgcolor="rgba(0,0,0,0)"),
                        )
                        st.plotly_chart(fig_graph, use_container_width=True)

            with tab_consensus:
                st.subheader("🤝 Cross-Study Consensus Cloud")
                if not f_attr.empty:
                    consensus_df = (
                        f_attr.groupby("Attribute")
                        .agg(Study_Count=("paper_id", "nunique"),
                             Total_Mentions=("Attribute", "count"))
                        .reset_index()
                    )
                    rng = np.random.default_rng(42)
                    consensus_df["x_pos"] = np.linspace(0, 10, len(consensus_df))
                    consensus_df["y_pos"] = rng.uniform(2, 5, len(consensus_df))
                    fig_cloud = px.scatter(
                        consensus_df, x="x_pos", y="y_pos",
                        size="Total_Mentions", color="Total_Mentions",
                        text="Attribute", size_max=60,
                        hover_data=["Study_Count"], template="plotly_dark",
                    )
                    fig_cloud.update_layout(showlegend=False, height=600,
                                            xaxis_visible=False, yaxis_visible=False)
                    st.plotly_chart(fig_cloud, use_container_width=True)

            with tab_atlas:
                if not f_text.empty:
                    paper_choice = st.selectbox("Select Paper", sorted(f_text["paper_id"].unique()))
                    paper_content = f_text[f_text["paper_id"] == paper_choice]
                    l, r = st.columns([1, 3])
                    with l:
                        sec_choice = st.radio("Sections", paper_content["heading"].tolist())
                    with r:
                        row = paper_content[paper_content["heading"] == sec_choice].iloc[0]
                        st.markdown(f"### {sec_choice}")
                        st.caption(f"Source: {row['paper_id']} | Page: {row['page']}")
                        st.write(row["text"])

    # ── RIGHT: styled chatbot panel ───────────────────────────
    with col_chat:
        st.markdown(
            "<h3 style='color:#4effd0;margin-top:0;font-size:1rem;letter-spacing:0.05em;'>💬 CHAT</h3>",
            unsafe_allow_html=True,
        )
        if not weaviate_available:
            st.warning("Weaviate offline.\n\n`docker-compose up -d`")

        if st.button("＋ New conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if not st.session_state["messages"]:
            st.markdown(
                "<p style='color:#aaaaaa;font-size:0.85rem;font-style:italic;'>"
                "Ask a clinical question about the loaded papers…"
                "</p>",
                unsafe_allow_html=True,
            )

# Page-level chat input — pinned at bottom
query = st.chat_input(
    "Ask about sepsis evidence…",
    disabled=not weaviate_available,
)
if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    if not weaviate_available:
        answer = "⚠️ Weaviate is not running. Start it with `docker-compose up -d`."
    else:
        with st.spinner("Searching literature…"):
            try:
                chunks = _retrieve(query, mode="hybrid", top_k=8, candidates=30, rerank=True)
                if llm_available and chunks:
                    answer = chat_answer(query, chunks)
                else:
                    answer = _format_chunks(chunks)
            except Exception as e:
                answer = f"⚠️ Retrieval error: {e}"
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.rerun()
