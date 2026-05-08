"""
apps/dashboard.py — Sepsis Global Attribute Atlas
Run with: streamlit run apps/dashboard.py
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
import re

import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline.config import PARSED_CACHE_DIR

DOMAIN_MAP = {
    "Severity Scores": ["sofa", "apache", "saps", "qsofa", "mews"],
    "Biomarkers":      ["lactate", "il-6", "crp", "procalcitonin", "leukocyte",
                        "creatinine", "bilirubin", "plt"],
    "Demographics":    ["age", "sex", "male", "female", "bmi", "weight",
                        "ethnicity", "comorbidity"],
    "Treatment":       ["vasopressor", "norepinephrine", "fluid", "antibiotic",
                        "ventilation", "dialysis"],
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


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_sections: list[dict] = []
    all_attributes: list[dict] = []

    for f_path in PARSED_CACHE_DIR.glob("*.json"):
        # Skip non-paper JSONs (tables, chunks, nx)
        if any(f_path.name.endswith(s) for s in ("_tables.json", "_chunks.json", "_nx.json")):
            continue
        try:
            data = json.loads(f_path.read_text(encoding="utf-8"))
            paper_id = data.get("paper_id", "Unknown")
            for sec in data.get("sections", []):
                content = sec.get("text", "")
                all_sections.append({
                    "paper_id": paper_id,
                    "heading":  sec.get("heading", "Untitled"),
                    "text":     content,
                })
                for item in extract_attributes(content):
                    all_attributes.append({
                        "paper_id":  paper_id,
                        "Domain":    item["Domain"],
                        "Attribute": item["Attribute"],
                        "Section":   sec.get("heading"),
                    })
        except Exception:
            continue

    return pd.DataFrame(all_sections), pd.DataFrame(all_attributes)


st.set_page_config(page_title="Sepsis Textual Attribute Atlas", layout="wide")
st.title("🌐 Sepsis Global Attribute Atlas")
st.markdown(
    "Discovers clinical attributes directly from paper sections — "
    "a wider lens than use-case-specific stats."
)

df_text, df_attr = load_data()

if df_text.empty:
    st.error(f"No text data found in {PARSED_CACHE_DIR}. "
             "Run the Ingest app first to parse papers.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Global Filter")
    all_papers = sorted(df_text["paper_id"].unique())
    selected_papers = st.multiselect("Active Papers", all_papers, default=all_papers)

f_text = df_text[df_text["paper_id"].isin(selected_papers)]
f_attr = df_attr[df_attr["paper_id"].isin(selected_papers)] if not df_attr.empty else df_attr

tab_viz, tab_search, tab_atlas = st.tabs(
    ["📊 Attribute Analytics", "🔍 Attribute Traceability", "📖 Literature Atlas"]
)

with tab_viz:
    if f_attr.empty:
        st.info("No attributes found in selected papers.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Clinical Attribute Coverage")
            fig_sun = px.sunburst(
                f_attr, path=["Domain", "Attribute", "paper_id"],
                color="Domain", title="Attribute Distribution in Raw Text",
            )
            st.plotly_chart(fig_sun, use_container_width=True)
        with col2:
            st.subheader("Information Density")
            density = (
                f_attr.groupby(["paper_id", "Domain"])
                .size()
                .reset_index(name="Mention Count")
            )
            fig_bar = px.bar(
                density, x="Mention Count", y="paper_id", color="Domain",
                orientation="h", title="Attribute Mentions per Study",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

with tab_search:
    st.subheader("🔍 Deep Attribute Traceability")
    query = st.text_input("Search for a clinical attribute (e.g. 'Lactate')", "")
    if query:
        results = f_text[f_text["text"].str.contains(query, case=False, na=False)]
        st.write(f"Found **{len(results)}** mentions across selected papers.")
        for _, row in results.iterrows():
            with st.expander(f"📄 {row['paper_id']} | {row['heading']}"):
                display_text = re.sub(
                    f"({re.escape(query)})", r"**\1**", row["text"], flags=re.IGNORECASE
                )
                st.markdown(display_text)

with tab_atlas:
    paper_choice = st.selectbox("Select Paper", sorted(f_text["paper_id"].unique()))
    paper_content = f_text[f_text["paper_id"] == paper_choice]
    l, r = st.columns([1, 2])
    with l:
        sec_choice = st.radio("Sections", paper_content["heading"].tolist())
    with r:
        st.markdown(f"### {sec_choice}")
        match = paper_content[paper_content["heading"] == sec_choice]["text"].values
        if len(match):
            st.write(match[0])
