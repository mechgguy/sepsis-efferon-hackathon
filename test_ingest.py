"""
ingest_viewer.py  —  run with: streamlit run ingest_viewer.py

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
from ingest import parse_pdf, ParsedPaper, ParsedTable, PARSED_CACHE_DIR


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
    """
    Apply rotation to table in-memory, overwrite _tables.json, return updated paper.
    Does NOT touch the main .json (docling cache).
    """
    paper = copy.deepcopy(paper)
    t = paper.tables[table_idx]
    rows = apply_mode(t.markdown, mode)
    t.markdown = _rows_to_markdown(rows)
    write_tables_json(paper)
    return paper


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Sepsis Atlas — Ingest Viewer", layout="wide")
st.title("📄 Ingest Viewer")
st.caption("Sections → .md | Tables → _tables.json. Rotating auto-saves _tables.json.")

papers_dir = Path("papers")
pdfs = sorted(papers_dir.glob("*.pdf")) if papers_dir.exists() else []

if not pdfs:
    st.error("No PDFs found in ./papers/")
    st.stop()

selected = st.selectbox("Select paper", pdfs, format_func=lambda p: p.name)

if st.button("Parse / Load", type="primary"):
    with st.spinner("Parsing (uses cache if available)..."):
        paper = parse_pdf(selected)
        write_sections_md(paper)
        write_tables_json(paper)
        st.session_state["paper"] = paper
        st.session_state["table_mode"] = {}

paper: ParsedPaper | None = st.session_state.get("paper")

if paper is None:
    st.info("Select a PDF and click Parse / Load.")
    st.stop()

if "table_mode" not in st.session_state:
    st.session_state["table_mode"] = {}

c1, c2, c3 = st.columns(3)
c1.metric("Paper ID", paper.paper_id[:40] + "…" if len(paper.paper_id) > 40 else paper.paper_id)
c2.metric("Sections", len(paper.sections))
c3.metric("Tables", len(paper.tables))

md_path = PARSED_CACHE_DIR / f"{paper.paper_id}.md"
tbl_path = PARSED_CACHE_DIR / f"{paper.paper_id}_tables.json"
st.caption(f"📄 `{md_path}` &nbsp;&nbsp; 🗃 `{tbl_path}`")

st.divider()

_CYCLE = ["original", "down", "up"]

tab_sections, tab_tables, tab_raw = st.tabs(["Sections", "Tables", "Full Markdown"])

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

with tab_raw:
    st.text_area(
        "full_markdown",
        paper.full_markdown[:10000],
        height=600,
        label_visibility="collapsed"
    )
    if len(paper.full_markdown) > 10000:
        st.caption(f"Showing first 10 000 of {len(paper.full_markdown)} chars.")