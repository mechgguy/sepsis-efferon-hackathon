import streamlit as st
import pandas as pd
import json
import os
import glob
import plotly.express as px
import re

st.set_page_config(page_title="Sepsis Textual Attribute Atlas", layout="wide")

# --- PATH CONFIG ---
TEXT_DIR = os.path.join(os.getcwd(), "data/parsed_papers/")

# --- ATTRIBUTE DISCOVERY LOGIC ---
# This expands your visualization by finding specific attributes inside raw text
DOMAIN_MAP = {
    "Severity Scores": ['sofa', 'apache', 'saps', 'qsofa', 'mews'],
    "Biomarkers": ['lactate', 'il-6', 'crp', 'procalcitonin', 'leukocyte', 'creatinine', 'bilirubin', 'plt'],
    "Demographics": ['age', 'sex', 'male', 'female', 'bmi', 'weight', 'ethnicity', 'comorbidity'],
    "Treatment": ['vasopressor', 'norepinephrine', 'fluid', 'antibiotic', 'ventilation', 'dialysis'],
    "Outcomes": ['mortality', 'death', 'discharge', 'icu stay', 'readmission']
}

def extract_attributes(text):
    """Scans text for keywords and returns identified attributes."""
    found = []
    text_lower = text.lower()
    for domain, keywords in DOMAIN_MAP.items():
        for kw in keywords:
            if kw in text_lower:
                found.append({"Domain": domain, "Attribute": kw.capitalize()})
    return found

# --- DATA LOADING ---
@st.cache_data
def load_raw_text_data():
    all_sections = []
    all_attributes = []
    
    files = glob.glob(os.path.join(TEXT_DIR, "*.json"))
    for f_path in files:
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                paper_id = data.get("paper_id", "Unknown")
                for sec in data.get("sections", []):
                    content = sec.get("text", "")
                    section_data = {
                        "paper_id": paper_id,
                        "heading": sec.get("heading", "Untitled"),
                        "text": content
                    }
                    all_sections.append(section_data)
                    
                    # Extract attributes for visualization
                    found = extract_attributes(content)
                    for item in found:
                        all_attributes.append({
                            "paper_id": paper_id,
                            "Domain": item["Domain"],
                            "Attribute": item["Attribute"],
                            "Section": sec.get("heading")
                        })
        except: continue
    return pd.DataFrame(all_sections), pd.DataFrame(all_attributes)

df_text, df_attr = load_raw_text_data()

# --- UI ---
st.title("🌐 Sepsis Global Attribute Atlas")
st.markdown("This view discovers clinical attributes directly from the paper sections, providing a wider lens than the use-case specific stats.")

if df_text.empty:
    st.error(f"No text data found in {TEXT_DIR}")
else:
    with st.sidebar:
        st.header("⚙️ Global Filter")
        selected_papers = st.multiselect("Active Papers", sorted(df_text['paper_id'].unique()), default=sorted(df_text['paper_id'].unique()))
    
    f_text = df_text[df_text['paper_id'].isin(selected_papers)]
    f_attr = df_attr[df_attr['paper_id'].isin(selected_papers)]

    # --- TABS ---
    tab_viz, tab_search, tab_atlas = st.tabs(["📊 Attribute Analytics", "🔍 Attribute Traceability", "📖 Literature Atlas"])

    with tab_viz:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Clinical Attribute Coverage")
            # Sunburst of discovered attributes across papers
            fig_sun = px.sunburst(
                f_attr, path=['Domain', 'Attribute', 'paper_id'],
                color='Domain', title="Attribute Distribution in Raw Text"
            )
            st.plotly_chart(fig_sun, width='stretch')

        with col2:
            st.subheader("Information Density")
            # Replacement for Forest Plot: Frequency of attribute mentions per paper
            density = f_attr.groupby(['paper_id', 'Domain']).size().reset_index(name='Mention Count')
            fig_bar = px.bar(
                density, x='Mention Count', y='paper_id', color='Domain',
                orientation='h', title="Attribute Mentions per Study"
            )
            st.plotly_chart(fig_bar, width='stretch')

    with tab_search:
        st.subheader("🔍 Deep Attribute Traceability")
        query = st.text_input("Search for a specific clinical attribute (e.g., 'Lactate')", "")
        
        if query:
            results = f_text[f_text['text'].str.contains(query, case=False, na=False)]
            st.write(f"Found **{len(results)}** mentions across selected papers.")
            for _, row in results.iterrows():
                with st.expander(f"📄 {row['paper_id']} | Section: {row['heading']}"):
                    # Highlight keyword
                    display_text = re.sub(f"({query})", r"**\1**", row['text'], flags=re.IGNORECASE)
                    st.markdown(display_text)

    with tab_atlas:
        paper_choice = st.selectbox("Select Paper", sorted(f_text['paper_id'].unique()))
        paper_content = f_text[f_text['paper_id'] == paper_choice]
        l, r = st.columns([1, 2])
        with l:
            sec_choice = st.radio("Sections", paper_content['heading'].tolist())
        with r:
            st.markdown(f"### {sec_choice}")
            st.write(paper_content[paper_content['heading'] == sec_choice]['text'].values[0])