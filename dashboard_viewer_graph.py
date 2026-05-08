import streamlit as st
import pandas as pd
import json
import os
import glob
import plotly.express as px
import re
import networkx as nx

st.set_page_config(page_title="Sepsis Global Attribute Atlas", layout="wide")

# --- PATH CONFIG ---
TEXT_DIR = os.path.join(os.getcwd(), "data/parsed_papers/")
STATS_DIR = os.path.join(os.getcwd(), "data/mortality_counterfactuals/")

# --- UNIVERSAL CATEGORIZATION LOGIC ---
def categorize_domain(row):
    """Categorizes every extraction into a clinical domain."""
    var = str(row.get('variable', '')).lower()
    if any(x in var for x in ['sofa', 'apache', 'saps', 'qsofa', 'score']):
        return "Severity Scores"
    elif any(x in var for x in ['mortality', 'death', 'survivor', 'outcome', 'discharge']):
        return "Clinical Outcomes"
    elif any(x in var for x in ['age', 'sex', 'male', 'female', 'bmi', 'weight', 'height', 'ethnicity']):
        return "Demographics"
    elif any(x in var for x in ['lactate', 'il-6', 'crp', 'procalcitonin', 'leukocyte', 'creatinine', 'bilirubin', 'plt']):
        return "Lab Biomarkers"
    elif any(x in var for x in ['diabetes', 'hypertension', 'cancer', 'renal', 'cardiac', 'history']):
        return "Comorbidities"
    return "Other Clinical Parameters"

def categorize_stat_type(row):
    """Identifies the type of value (Predictive vs Descriptive)"""
    var = str(row.get('variable', '')).upper()
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

if df_stats.empty:
    st.error("No extraction data found.")
else:
    # Processing
    df_stats['Domain'] = df_stats.apply(categorize_domain, axis=1)
    df_stats['Stat_Type'] = df_stats.apply(categorize_stat_type, axis=1)
    df_stats['num_val'] = pd.to_numeric(df_stats['value'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Global Filter")
        paper_list = sorted(df_stats['source_paper'].unique())
        selected_papers = st.multiselect("Active Studies", paper_list, default=paper_list)
        st.divider()
        st.info(f"Loaded {len(df_stats)} extractions across {len(paper_list)} papers.")

    filtered_stats = df_stats[df_stats['source_paper'].isin(selected_papers)]

    # --- TABS --- tab_list
    tab_viz, tab_table, tab_trace, tab_atlas, tab_explain = st.tabs([
        "📊 Visual Analytics", "📋 Registry", "🔍 Traceability", "📖 Atlas", "🧠 Explanation"
    ])

    with tab_viz:
        st.subheader("Clinical Evidence Map")
        c1, c2 = st.columns(2)
        with c1:
            fig_sun = px.sunburst(filtered_stats.fillna("Unknown"), 
                                  path=['Domain', 'Stat_Type', 'variable'], color='Domain')
            st.plotly_chart(fig_sun, use_container_width=True)
        with c2:
            forest_df = filtered_stats[filtered_stats['Stat_Type'] == "Predictive (Ratio)"].dropna(subset=['num_val'])
            if not forest_df.empty:
                fig_forest = px.scatter(forest_df, x='num_val', y='variable', color='Domain', hover_data=['source_paper'])
                fig_forest.add_vline(x=1.0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_forest, use_container_width=True)

    with tab_table:
        st.subheader("Global Evidence Registry")
        search = st.text_input("🔍 Search variable name")
        table_df = filtered_stats
        if search:
            table_df = table_df[table_df['variable'].str.contains(search, case=False, na=False)]
        st.dataframe(table_df[['Domain', 'variable', 'value', 'source_paper']], use_container_width=True)

    with tab_trace:
        st.subheader("Traceability: From Data to Source")
        st.markdown("Select a variable below to see exactly where it was mentioned in the literature.")
        
        # Selectable index for traceability
        trace_var = st.selectbox("Select Extracted Variable", filtered_stats['variable'].unique())
        
        occurrences = filtered_stats[filtered_stats['variable'] == trace_var]
        
        for _, row in occurrences.iterrows():
            with st.expander(f"📄 {row['source_paper']} | Value: {row['value']}", expanded=True):
                col_left, col_right = st.columns([1, 3])
                with col_left:
                    st.metric("Stat Type", row['Stat_Type'])
                    st.caption(f"Domain: {row['Domain']}")
                with col_right:
                    st.markdown("**Evidence Snippet:**")
                    # Highlight the value in the text if possible
                    evidence = row.get('evidence_text', "No specific evidence text stored.")
                    st.info(f"\"{evidence}\"")
                    
                    # Cross-reference with the full atlas
                    if st.button(f"View full context in {row['source_paper']}", key=f"btn_{_}"):
                        st.session_state.target_paper = row['source_paper']
                        st.warning("Switch to 'Literature Atlas' tab to see full section.")

    with tab_atlas:
            st.subheader("📖 Literature Atlas")
            
            # 1. Check if we actually have text data loaded
            if df_text.empty:
                st.error("No parsed text found in `data/parsed_papers/`. Please check your file paths.")
            else:
                # 2. Get list of papers that actually have text available
                available_text_papers = sorted(df_text['paper_id'].unique().tolist())
                
                # 3. Handle the 'initial_paper' logic safely
                # We check session state first (from Traceability click), then fallback to first available
                target = st.session_state.get('target_paper', available_text_papers[0])
                
                # If the targeted paper isn't in our text list, reset to the first available text paper
                if target not in available_text_papers:
                    target = available_text_papers[0]
                
                # Find the index for the selectbox
                try:
                    default_idx = available_text_papers.index(target)
                except ValueError:
                    default_idx = 0

                # 4. Render Selectbox
                paper_choice = st.selectbox(
                    "Select Paper for Deep Reading", 
                    available_text_papers, 
                    index=default_idx,
                    key="atlas_paper_select"
                )

                # 5. Filter and Display Content
                paper_content = df_text[df_text['paper_id'] == paper_choice]
                
                if paper_content.empty:
                    st.warning(f"No text sections found for {paper_choice}.")
                else:
                    st.divider()
                    l_col, r_col = st.columns([1, 2])
                    
                    with l_col:
                        st.write("**Sections**")
                        # Use a radio button to navigate headings
                        headings = paper_content['heading'].unique().tolist()
                        selected_sec = st.radio(
                            "Go to:", 
                            headings,
                            key="atlas_section_radio"
                        )
                    
                    with r_col:
                        # Get the text for the selected heading
                        content_row = paper_content[paper_content['heading'] == selected_sec]
                        if not content_row.empty:
                            st.markdown(f"### {selected_sec}")
                            st.write(content_row['text'].values[0])
                        else:
                            st.info("Select a section to view content.")
    
    with tab_explain:
        st.subheader("Clinical Knowledge Graph")
        st.markdown("""
        This graph visualizes the **consensus** between different studies. 
        Nodes represent biomarkers; the thickness of lines represents how many papers support that association.
        """)
        
        # Simple logic to create a relationship dataframe
        edges = filtered_stats.groupby(['variable', 'Domain']).size().reset_index(name='Weight')
        
        # You can use a Plotly Scatter plot to simulate a network if you don't want extra libraries
        fig_network = px.scatter(
            edges, x="Domain", y="variable", size="Weight", color="Domain",
            title="Predictor Prevalence across Clinical Domains",
            labels={'Weight': 'Number of Citations'}
        )
        st.plotly_chart(fig_network, use_container_width=True)
        
        st.divider()
        st.subheader("Logic Consistency Check")
        # Show a summary of how often the AI finds specific metric types
        logic_df = filtered_stats['Stat_Type'].value_counts()
        st.bar_chart(logic_df)