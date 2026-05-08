"""
test_ingest.py  —  run with: streamlit run ingest_viewer.py

Storage layout (written automatically):
  data/parsed_papers/{paper_id}.md          ← sections text only, no tables
  data/parsed_papers/{paper_id}_tables.json ← tables list, updated on rotate
  data/parsed_papers/{paper_id}.json        ← full docling cache (untouched)
"""
import re
import json
import copy
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st
from ingest import (
    parse_pdf,
    ParsedPaper,
    ParsedTable,
    ParsedFigure,
    PARSED_CACHE_DIR,
)
from ingest_parallel import parse_all_parallel
from chunking_strategy import build_chunks_from_paper


# ── Markdown table utils ──────────────────────────────────────────────────────

_STAT_HEADER_KEYWORDS = [
    "p value", "p-value", "risk difference", "95% ci", "95%ci",
    "hazard ratio", "odds ratio", "relative risk", "confidence interval",
    "percentage points", "mean ±", "median", "interquartile",
]

def _parse_markdown_table(md: str) -> list[list[str]]:
    rows = []
    for line in md.strip().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if re.match(r"^\|[-| :]+\|$", line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) <= 1:
            continue
        rows.append(cells)
    return rows


def _rows_to_markdown(rows: list[list[str]]) -> str:
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


def transpose_rows(rows: list[list[str]], reverse: bool) -> list[list[str]]:
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]
    transposed = [list(col) for col in zip(*padded)]
    return [r[::-1] for r in transposed] if reverse else transposed


def apply_mode(md: str, mode: str) -> list[list[str]]:
    rows = _parse_markdown_table(md)
    if mode == "down":
        return transpose_rows(rows, reverse=True)
    elif mode == "up":
        return transpose_rows(rows, reverse=False)
    return rows


def is_table_rotated(table: ParsedTable) -> tuple[bool, str]:
    md = table.markdown.lower()
    rows = _parse_markdown_table(md)
    if not rows:
        return False, ""
    first_row_text = " ".join(rows[0])
    for kw in _STAT_HEADER_KEYWORDS:
        if kw in first_row_text:
            return True, f"stat keyword '{kw}' in row 0"
    n_rows, n_cols = len(rows), max(len(r) for r in rows)
    if n_cols > n_rows * 2:
        return True, f"cols ({n_cols}) >> rows ({n_rows})"
    if len(rows) >= 3:
        first_col = [r[0] for r in rows[1:] if r]
        numeric_count = sum(1 for c in first_col if re.search(r"\d", c))
        first_row_all_text = all(not re.search(r"^\s*[\d.]+\s*$", c) for c in rows[0])
        if first_col and numeric_count / len(first_col) > 0.6 and first_row_all_text:
            return True, f"first col {numeric_count}/{len(first_col)} numeric, first row all-text"
    return False, ""


# ── File writers ──────────────────────────────────────────────────────────────

def write_sections_md(paper: ParsedPaper) -> Path:
    """Write sections-only .md (no tables). Called once on load."""
    path = PARSED_CACHE_DIR / f"{paper.paper_id}.md"
    lines = []
    for s in paper.sections:
        lines.append(f"## {s.heading}\n")
        lines.append(s.text)
        lines.append("\n\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def write_tables_json(paper: ParsedPaper) -> Path:
    """Write (or overwrite) tables-only JSON. Called on load and on rotate."""
    path = PARSED_CACHE_DIR / f"{paper.paper_id}_tables.json"
    with open(path, "w") as f:
        json.dump([asdict(t) for t in paper.tables], f, indent=2)
    return path


def rotate_and_save(paper: ParsedPaper, table_idx: int, mode: str) -> ParsedPaper:
    paper = copy.deepcopy(paper)
    t = paper.tables[table_idx]
    rows = apply_mode(t.markdown, mode)
    t.markdown = _rows_to_markdown(rows)
    write_tables_json(paper)
    write_chunks_json(paper) # <--- ADD THIS to keep chunks in sync with rotations
    return paper

def write_chunks_json(paper: ParsedPaper) -> Path:
    """Generate chunks from paper and write to {paper_id}_chunks.json."""
    path = PARSED_CACHE_DIR / f"{paper.paper_id}_chunks.json"
    
    # You can adjust max_chars here (e.g., 1500 or 1800)
    chunks = build_chunks_from_paper(paper, max_chars=1500)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    return path

# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Sepsis Atlas — Ingest Viewer", layout="wide")
st.title("📄 Ingest Viewer")
st.caption("Sections → .md | Tables → _tables.json. Rotating auto-saves _tables.json.")

papers_dir = Path("materials/articles")
pdfs = sorted(papers_dir.glob("*.pdf")) if papers_dir.exists() else []

if not pdfs:
    st.error("No PDFs found in " + str(papers_dir))
    st.stop()

selected = st.selectbox("Select paper", pdfs, format_func=lambda p: p.name)

if st.button("Parse / Load", type="primary"):
    with st.spinner("Parsing (uses cache if available)..."):
        paper = parse_pdf(selected)
        write_sections_md(paper)
        write_tables_json(paper)
        write_chunks_json(paper)  # <--- ADD THIS
        st.session_state["paper"] = paper
        st.session_state["table_mode"] = {}

st.divider()
st.subheader("📦 Batch Ingest All Papers")

cached = [p for p in pdfs if (PARSED_CACHE_DIR / f"{p.stem}.json").exists()]
uncached = [p for p in pdfs if p not in cached]
st.caption(f"{len(cached)} cached · {len(uncached)} not yet parsed · {len(pdfs)} total")

workers = st.slider("Parallel workers", min_value=1, max_value=6, value=3,
                    help="Each worker builds its own Docling converter. Keep ≤4 on CPU-only.")

if st.button("🚀 Parse All Papers", type="primary"):
    progress_bar = st.progress(0)
    status_text  = st.empty()
    results_box  = st.empty()
    log: list[str] = []

    def on_progress(done, total, filename):
        progress_bar.progress(done / total)
        status_text.text(f"[{done}/{total}] {filename}")
        log.append(f"✅ {filename}")
        results_box.markdown("\n".join(log[-10:]))  # show last 10

    with st.spinner("Batch parsing in progress..."):
        successes, failures = parse_all_parallel(
            papers_dir, max_workers=workers, progress_cb=on_progress
        )
        for paper in successes:
            write_sections_md(paper)
            write_tables_json(paper)

    progress_bar.progress(1.0)
    status_text.text("Done.")
    st.success(f"Parsed {len(successes)} papers.")
    if failures:
        st.error(f"{len(failures)} failed:")
        for path, err in failures:
            st.code(f"{path.name}: {err}")

st.divider()

st.subheader("📦 Chunk All Parsed Papers")

parsed_jsons = [p for p in pdfs if (PARSED_CACHE_DIR / f"{p.stem}.json").exists()]
unchunked = [p for p in parsed_jsons if not (PARSED_CACHE_DIR / f"{p.stem}_chunks.json").exists()]
st.caption(f"{len(parsed_jsons)} parsed · {unchunked and len(unchunked) or 0} not yet chunked")

if st.button("🔪 Chunk All Papers", type="primary"):
    if not parsed_jsons:
        st.warning("No parsed papers found. Parse first.")
    else:
        prog = st.progress(0)
        log2: list[str] = []
        box2 = st.empty()
        for i, p in enumerate(parsed_jsons, 1):
            paper = parse_pdf(p)
            path = write_chunks_json(paper)
            log2.append(f"✅ {p.name} → {path.name}")
            box2.markdown("\n".join(log2[-10:]))
            prog.progress(i / len(parsed_jsons))
        st.success(f"Chunked {len(parsed_jsons)} papers.")

st.divider()

paper: ParsedPaper | None = st.session_state.get("paper")

if paper is None:
    st.info("Select a PDF and click Parse / Load.")
    st.stop()

if "table_mode" not in st.session_state:
    st.session_state["table_mode"] = {}

c1, c2, c3, c4 = st.columns(4)
c1.metric("Paper ID", paper.paper_id[:40] + "…" if len(paper.paper_id) > 40 else paper.paper_id)
c2.metric("Sections", len(paper.sections))
c3.metric("Tables", len(paper.tables))  
c4.metric("Figures", len(paper.figures))

md_path = PARSED_CACHE_DIR / f"{paper.paper_id}.md"
tbl_path = PARSED_CACHE_DIR / f"{paper.paper_id}_tables.json"
st.caption(f"📄 `{md_path}` &nbsp;&nbsp; 🗃 `{tbl_path}`")

st.divider()

_CYCLE = ["original", "down", "up"]

tab_sections, tab_tables, tab_figures, tab_chunks, tab_raw = st.tabs(
    ["Sections", "Tables", "Figures","Chunks", "Full Markdown"]
)

with tab_sections:
    if not paper.sections:
        st.warning("No sections parsed.")
    for s in paper.sections:
        label = f"**{s.heading}** &nbsp; `p.{s.page_start}` &nbsp; ({len(s.text)} chars)"
        with st.expander(label):
            st.markdown(s.text[:2000] + ("…" if len(s.text) > 2000 else ""))

with tab_tables:
    if not paper.tables:
        st.warning("No tables parsed.")

    for t in paper.tables:
        rotated, reason = is_table_rotated(t)
        badge = " 🔴" if rotated else ""
        label = (
            f"Table {t.index}{badge} &nbsp; `p.{t.page_start}` &nbsp; "
            f"— *{t.preceding_heading}*"
        )
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
                    new_mode = next_mode
                    st.session_state["table_mode"][tkey] = new_mode
                    if new_mode != "original":
                        # auto-save: rotate in memory + overwrite _tables.json
                        updated = rotate_and_save(paper, t.index, new_mode)
                        st.session_state["paper"] = updated
                        st.toast(f"Table {t.index} rotated ({new_mode}) → saved to _tables.json")
                    st.rerun()

            st.caption(f"mode: {mode}")
            rows = apply_mode(t.markdown, mode)
            st.dataframe(_rows_to_df(rows), use_container_width=True)

with tab_figures:
    if not getattr(paper, "figures", []):
        st.warning("No figures parsed.")

    for fig in paper.figures:
        label = f"Figure {fig.index} · p.{fig.page_start}"

        with st.expander(label):
            if fig.caption:
                st.caption(fig.caption)

            if fig.image_path and Path(fig.image_path).exists():
                st.image(fig.image_path, use_container_width=True)
                st.code(fig.image_path)
            else:
                st.error(f"Missing image: {fig.image_path}")
with tab_chunks:
    if st.button("Generate Chunks for Testing"):
        
        chunks = build_chunks_from_paper(paper, max_chars=500)
        
        st.write(f"Generated {len(chunks)} chunks.")
        
        for c in chunks:
            with st.expander(f"📄 Page {c['metadata']['page_number']} | {c['metadata']['section']}"):
                st.json(c["metadata"])
                st.markdown(c["text"])
                
with tab_raw:
    st.text_area(
        "full_markdown",
        paper.full_markdown[:10000],
        height=600,
        label_visibility="collapsed"
    )
    if len(paper.full_markdown) > 10000:
        st.caption(f"Showing first 10 000 of {len(paper.full_markdown)} chars.")