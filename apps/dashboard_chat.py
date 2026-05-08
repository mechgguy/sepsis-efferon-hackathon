"""
apps/dashboard_chat.py — Sepsis Atlas: Dashboard + Chat
Run with: streamlit run apps/dashboard_chat.py

Left:  Full 5-tab attribute visualization
Right: Chatbot panel (styled per chatbot_panel_layout.md)
"""
import sys
import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import glob
import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline.config import PARSED_CACHE_DIR

# ── Chat helpers ──────────────────────────────────────────────────────────────

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


# ── Weaviate / LLM (graceful degradation) ────────────────────────────────────

weaviate_available = False
try:
    from pipeline.weaviate.config import get_client
    from pipeline.weaviate.retrieve import retrieve as _retrieve
    _c = get_client()
    _c.is_ready()
    _c.close()
    weaviate_available = True
except Exception:
    pass

llm_available = bool(
    os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
)
if llm_available:
    try:
        from pipeline.extract import extract_records
    except Exception:
        llm_available = False

# ── Attribute domain map ──────────────────────────────────────────────────────

DOMAIN_MAP = {
    "Severity Scores": ["sofa", "apache", "saps", "qsofa", "mews"],
    "Biomarkers":      ["lactate", "il-6", "crp", "procalcitonin", "leukocyte", "creatinine", "bilirubin", "plt"],
    "Demographics":    ["age", "sex", "male", "female", "bmi", "weight", "ethnicity", "comorbidity"],
    "Treatment":       ["vasopressor", "norepinephrine", "fluid", "antibiotic", "ventilation", "dialysis"],
    "Outcomes":        ["mortality", "death", "discharge", "icu stay", "readmission"],
}


def extract_attributes(text: str) -> list[dict]:
    found = []
    text_lower = text.lower()
    for domain, keywords in DOMAIN_MAP.items():
        for kw in keywords:
            if kw in text_lower:
                found.append({"Domain": domain, "Attribute": kw.capitalize()})
    return found


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
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

                found = extract_attributes(content)
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


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


# ── Page config & CSS ─────────────────────────────────────────────────────────

st.set_page_config(page_title="Sepsis Atlas", layout="wide")
_init_state()

st.markdown("""
<style>
/* ── Global ── */
.stApp { background: #1a2234; }
* { scrollbar-color: #4effd0 rgba(255,255,255,0.1); scrollbar-width: thin; }

/* ── Sidebar (paper filter) ── */
section[data-testid="stSidebar"] > div:first-child {
    background: #44546a;
    border-right: 2px solid #4effd0;
}
section[data-testid="stSidebar"] * { color: #ffffff !important; }

/* ── Chatbot panel (right column) ── */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child {
    background: #44546a;
    border: 2px solid #4effd0;
    border-radius: 12px;
    padding: 16px !important;
    min-height: 80vh;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child * {
    color: #ffffff;
}

/* ── Chat messages inside right panel ── */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child
  div[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.07);
    border-radius: 8px;
    border: 1px solid rgba(78,255,208,0.2);
    margin-bottom: 8px;
}

/* ── Chat input ── */
div[data-testid="stChatInput"] textarea {
    background: rgba(68,84,106,0.8);
    border: 1px solid #4effd0;
    color: #ffffff;
}
div[data-testid="stChatInput"] button {
    background: #4effd0;
    color: #44546a;
}

/* ── Active tab ── */
button[data-testid="stTab"][aria-selected="true"] {
    border-bottom: 2px solid #4effd0 !important;
    color: #4effd0 !important;
}

/* ── Progress bars ── */
div[data-testid="stProgress"] > div { background: #4effd0; }

/* ── Headings ── */
h1, h2, h3 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────

df_text, df_attr, df_links = load_data()

# ── Sidebar filter ────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Global Filter")
    if df_text.empty:
        st.warning("No data loaded.")
        selected_papers = []
    else:
        paper_list = sorted(df_text["paper_id"].unique())
        selected_papers = st.multiselect("Active Papers", paper_list, default=paper_list)
    st.divider()
    st.markdown("### 💬 Conversation")
    if st.button("＋ New conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()
    if not weaviate_available:
        st.warning("Weaviate offline — chat disabled.\n\n`docker-compose up -d`")

# ── Filter data ───────────────────────────────────────────────────────────────

if not df_text.empty and selected_papers:
    f_text  = df_text[df_text["paper_id"].isin(selected_papers)]
    f_attr  = df_attr[df_attr["paper_id"].isin(selected_papers)]
    f_links = (df_links[df_links["paper_id"].isin(selected_papers)]
               if not df_links.empty else pd.DataFrame(columns=["source", "target", "paper_id"]))
else:
    f_text  = pd.DataFrame()
    f_attr  = pd.DataFrame()
    f_links = pd.DataFrame(columns=["source", "target", "paper_id"])

# ── Main two-column layout ────────────────────────────────────────────────────

col_dash, col_chat = st.columns([3, 1.3])

# ═══════════════════════════════════════════════════════════════
#  LEFT — FULL 5-TAB DASHBOARD
# ═══════════════════════════════════════════════════════════════

with col_dash:
    st.markdown("## 🩺 Sepsis Global Attribute Atlas")

    if df_text.empty:
        st.error(f"No chunked JSON files found in `{PARSED_CACHE_DIR}`.")
        st.info("Run: **Ingest** tab → Parse + Chunk all papers.")
    else:
        tab_viz, tab_search, tab_explain, tab_consensus, tab_atlas = st.tabs([
            "📊 Attribute Analytics",
            "🔍 Traceability",
            "🧠 Explainability & Graph",
            "🤝 Cross-Study Consensus",
            "📖 Literature Atlas",
        ])

        # ── ANALYTICS ────────────────────────────────────────────
        with tab_viz:
            st.subheader("🎯 Clinical Attribute Coverage")
            if not f_attr.empty:
                fig_sun = px.sunburst(f_attr, path=["Domain", "Attribute", "paper_id"],
                                      color="Domain", template="plotly_dark")
                fig_sun.update_layout(height=700)
                st.plotly_chart(fig_sun, use_container_width=True)
            else:
                st.info("No attribute data for selected papers.")

            st.divider()
            st.subheader("📊 Information Density")
            if not f_attr.empty:
                density = f_attr.groupby(["paper_id", "Domain"]).size().reset_index(name="Mention Count")
                fig_bar = px.bar(density, x="Mention Count", y="paper_id", color="Domain",
                                 orientation="h", template="plotly_dark")
                fig_bar.update_layout(height=max(400, len(selected_papers) * 30))
                st.plotly_chart(fig_bar, use_container_width=True)

        # ── TRACEABILITY ──────────────────────────────────────────
        with tab_search:
            st.subheader("📋 Extracted Clinical Variables & Evidence")
            search_var = st.text_input("🔍 Search Variable Name", placeholder="e.g., Lactate")
            display_df = f_attr.copy()
            if search_var:
                display_df = display_df[display_df["Attribute"].str.contains(search_var, case=False, na=False)]
            registry_table = display_df.rename(columns={
                "Attribute": "Variable",
                "paper_id":  "Source Paper",
                "Section":   "Section Heading",
                "Page":      "Pg #",
            })
            if not registry_table.empty:
                st.dataframe(
                    registry_table[["Domain", "Variable", "Source Paper", "Pg #", "Section Heading"]],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Pg #":            st.column_config.NumberColumn("Pg #", format="%d"),
                        "Variable":        st.column_config.TextColumn("Variable", width="medium"),
                        "Source Paper":    st.column_config.TextColumn("Source Paper", width="medium"),
                        "Section Heading": st.column_config.TextColumn("Evidence Context", width="large"),
                    },
                )
            else:
                st.info("No variables found.")

        # ── EXPLAINABILITY ────────────────────────────────────────
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
                        height=800,
                        margin=dict(l=150, b=150),
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("No attribute data.")

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
                else:
                    st.info("No co-occurrence links.")

        # ── CONSENSUS CLOUD ───────────────────────────────────────
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
                    hover_data=["Study_Count"],
                    template="plotly_dark",
                )
                fig_cloud.update_layout(showlegend=False, height=600,
                                        xaxis_visible=False, yaxis_visible=False)
                st.plotly_chart(fig_cloud, use_container_width=True)
            else:
                st.info("No attribute data.")

        # ── LITERATURE ATLAS ──────────────────────────────────────
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
            else:
                st.info("No text data for selected papers.")

# ═══════════════════════════════════════════════════════════════
#  RIGHT — CHATBOT PANEL
# ═══════════════════════════════════════════════════════════════

with col_chat:
    st.markdown(
        "<h3 style='color:#4effd0; margin-top:0; font-size:1rem; letter-spacing:0.05em;'>💬 CHAT</h3>",
        unsafe_allow_html=True,
    )

    if not weaviate_available:
        st.warning("Weaviate offline.\n\n`docker-compose up -d`")

    # Message history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state["messages"]:
        st.markdown(
            "<p style='color:#aaaaaa; font-size:0.85rem; font-style:italic;'>"
            "Ask a clinical question about the loaded papers…"
            "</p>",
            unsafe_allow_html=True,
        )

# ── Chat input (page-level, pinned at bottom) ─────────────────────────────────

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
                    records = extract_records(query, chunks)
                    if records:
                        lines = [f"**Found {len(records)} evidence record(s):**\n"]
                        for r in records:
                            lines.append(
                                f"**{r.get('study', '?')}** — {r.get('predictor', '?')} → "
                                f"{r.get('outcome', '?')}\n"
                                f"> Effect: {r.get('effect_size', 'not reported')}  "
                                f"| Method: {r.get('method', 'not reported')}\n"
                                f"> {r.get('source_anchor', '')}"
                            )
                        answer = "\n\n".join(lines)
                    else:
                        answer = _format_chunks(chunks)
                else:
                    answer = _format_chunks(chunks)
            except Exception as e:
                answer = f"⚠️ Error: {e}"

    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.rerun()
