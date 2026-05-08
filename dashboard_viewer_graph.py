import streamlit as st
import pandas as pd
import json
import os
import glob
import plotly.express as px
import re

st.set_page_config(page_title="Sepsis Global Attribute Atlas", layout="wide")

# --- PATH CONFIG ---
TEXT_DIR = os.path.join(os.getcwd(), "data/parsed_papers/")
STATS_DIR = os.path.join(os.getcwd(), "data/mortality_counterfactuals/")

# --- UNIVERSAL CATEGORIZATION LOGIC ---
def categorize_domain(row):
    """Categorizes every extraction into a clinical domain."""
    var = str(row.get('variable', '')).lower()
    text = str(row.get('evidence_text', '')).lower()
    
    # Severity Scores
    if any(x in var for x in ['sofa', 'apache', 'saps', 'qsofa', 'score']):
        return "Severity Scores"
    # Outcomes
    elif any(x in var for x in ['mortality', 'death', 'survivor', 'outcome', 'discharge']):
        return "Clinical Outcomes"
    # Demographics
    elif any(x in var for x in ['age', 'sex', 'male', 'female', 'bmi', 'weight', 'height', 'ethnicity']):
        return "Demographics"
    # Lab Values / Biomarkers
    elif any(x in var for x in ['lactate', 'il-6', 'crp', 'procalcitonin', 'leukocyte', 'creatinine', 'bilirubin', 'plt']):
        return "Lab Biomarkers"
    # Comorbidities
    elif any(x in var for x in ['diabetes', 'hypertension', 'cancer', 'renal', 'cardiac', 'history']):
        return "Comorbidities"
    else:
        return "Other Clinical Parameters"

def categorize_stat_type(row):
    """Identifies the type of value (Predictive vs Descriptive)"""
    var = str(row.get('variable', '')).upper()
    val = str(row.get('value', '')).upper()
    if any(x in var for x in ['HR', 'OR', 'RR', 'ODDS', 'HAZARD']):
        return "Predictive (Ratio)"
    if "AUC" in var or "ROC" in var:
        return "Accuracy (AUC)"
    if "P-VALUE" in var or "P=" in var or "P <" in var:
        return "Significance (P-value)"
    return "Descriptive (Mean/Median)"

# --- DATA LOADING ---
@st.cache_data
def load_all_data():
    all_text = []
    all_stats = []
    
    # Load Raw Text
    for f_path in glob.glob(os.path.join(TEXT_DIR, "*.json")):
        try:
            with open(f_path, "r") as f:
                data = json.load(f)
                for sec in data.get("sections", []):
                    all_text.append({
                        "paper_id": data.get("paper_id", "Unknown"),
                        "heading": sec.get("heading", "Untitled"),
                        "text": sec.get("text", "")
                    })
        except: continue

    # Load Extractions
    for f_path in glob.glob(os.path.join(STATS_DIR, "*_results.json")):
        try:
            with open(f_path, "r") as f:
                data = json.load(f)
                for ext in data.get("extractions", []):
                    ext['source_paper'] = data.get("paper_id", "Unknown")
                    all_stats.append(ext)
        except: continue
        
    return pd.DataFrame(all_text), pd.DataFrame(all_stats)

df_text, df_stats = load_all_data()

# --- UI ---
st.title("🌐 Sepsis Global Evidence & Attribute Atlas")
st.markdown("Exploring all extracted clinical variables, biomarkers, and outcomes across the literature.")

if df_stats.empty:
    st.error("No extraction data found. Please run the extraction script first.")
else:
    # Process Stats for Visualization
    df_stats['Domain'] = df_stats.apply(categorize_domain, axis=1)
    df_stats['Stat_Type'] = df_stats.apply(categorize_stat_type, axis=1)
    # Extract numbers for plotting
    df_stats['num_val'] = pd.to_numeric(df_stats['value'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')

    tab_viz, tab_table, tab_atlas = st.tabs(["📊 Visual Analytics", "📋 Comprehensive Table", "📖 Literature Atlas"])

    with tab_viz:
        st.subheader("Clinical Evidence Map (All Attributes)")
        c1, c2 = st.columns(2)
        
        with c1:
            st.write("**Attribute Distribution**")
            # Hierarchy: Domain -> Stat Type -> Variable
            fig_sun = px.sunburst(
                df_stats.fillna("Unknown"),
                path=['Domain', 'Stat_Type', 'variable'],
                color='Domain',
                color_discrete_sequence=px.colors.qualitative.Vivid,
                maxdepth=3
            )
            st.plotly_chart(fig_sun, use_container_width=True)
            st.caption("Click segments to drill down into specific clinical domains.")

        with c2:
            st.write("**Impact Factors (HR/OR Only)**")
            # Show predictive impact for all attributes (Lactate, Age, etc.)
            forest_df = df_stats[df_stats['Stat_Type'] == "Predictive (Ratio)"].dropna(subset=['num_val'])
            if not forest_df.empty:
                fig_forest = px.scatter(
                    forest_df, x='num_val', y='variable', color='Domain',
                    hover_data=['source_paper', 'value'],
                    title="Comparison of Effect Sizes across Domains"
                )
                fig_forest.add_vline(x=1.0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_forest, use_container_width=True)
            else:
                st.info("No predictive ratios (HR/OR) found to plot.")

    with tab_table:
        st.subheader("Global Evidence Registry")
        # Domain filter
        domain_filter = st.multiselect("Filter by Domain", df_stats['Domain'].unique(), default=df_stats['Domain'].unique())
        filtered_stats = df_stats[df_stats['Domain'].isin(domain_filter)]
        
        search = st.text_input("🔍 Search any variable (e.g., 'IL-6', 'SOFA', 'Male')")
        if search:
            filtered_stats = filtered_stats[filtered_stats['variable'].str.contains(search, case=False, na=False)]
            
        st.dataframe(filtered_stats[['Domain', 'variable', 'value', 'Stat_Type', 'source_paper']], use_container_width=True, hide_index=True)

    with tab_atlas:
        if not df_text.empty:
            paper_choice = st.selectbox("Select Paper", df_text['paper_id'].unique())
            paper_content = df_text[df_text['paper_id'] == paper_choice]
            
            l_col, r_col = st.columns([1, 2])
            with l_col:
                selected_sec = st.radio("Sections", paper_content['heading'].tolist())
            with r_col:
                st.markdown(f"### {selected_sec}")
                st.write(paper_content[paper_content['heading'] == selected_sec]['text'].values[0])