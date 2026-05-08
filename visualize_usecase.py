import streamlit as st
import pandas as pd
import json
import os
import glob
import plotly.express as px

st.set_page_config(page_title="Sepsis Counterfactual Explorer", layout="wide")

# --- PATH CONFIG ---
# Points to where your AI-extracted JSONs live
DATA_DIR = "data/mortality_counterfactuals"

def categorize_outcome(row):
    """Categorizes the outcome definition (Requirement: Use Case 1)"""
    text = (str(row.get('variable', '')) + " " + str(row.get('evidence_text', ''))).lower()
    if "28" in text: return "28-Day Mortality"
    if "90" in text: return "90-Day Mortality"
    if "hospital" in text: return "In-Hospital Mortality"
    if "icu" in text: return "ICU Mortality"
    return "General Mortality"

def categorize_metric(row):
    """Categorizes the statistical method (Requirement: Use Case 1)"""
    var = str(row.get('variable', '')).upper()
    if "HR" in var or "HAZARD" in var: return "Hazard Ratio (HR)"
    if "OR" in var or "ODDS" in var: return "Odds Ratio (OR)"
    if "AUC" in var or "ROC" in var: return "AUC/C-Statistic"

@st.cache_data
def load_all_extractions():
    """Loads PaperExtractions format JSONs into a flat list for dataframes."""
    all_data = []
    if not os.path.exists(DATA_DIR):
        return []
    
    # Grab all results.json files
    json_files = glob.glob(os.path.join(DATA_DIR, "*_results.json"))
    
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # Ensure we are looking at the 'extractions' key from your new schema
                if "extractions" in data:
                    paper_id = data.get("paper_id", "Unknown")
                    year = data.get("year", "N/A")
                    
                    for ext in data["extractions"]:
                        # Flatten the structure for the table
                        ext['source_paper'] = paper_id
                        ext['year'] = year
                        all_data.append(ext)
            except Exception as e:
                continue
    return all_data

# --- DATA LOADING ---
all_evidence = load_all_extractions()

# --- UI HEADER ---
st.title("🟢 Use Case 1: Mortality Benchmarking")
st.markdown("""
This view aggregates **Hazard Ratios (HR)** and **Odds Ratios (OR)** extracted from clinical literature 
to help estimate the expected mortality (Counterfactual) for your patient registry.
""")

if not all_evidence:
    st.warning(f"No extraction files found in `{DATA_DIR}`. Run your extraction script first!")
else:
    # Create Dataframe
    df = pd.DataFrame(all_evidence)

    # --- DATA CLEANING FOR PLOTS ---
    # Extract numeric values for Forest Plot (e.g., "1.044" -> 1.044)
    df['numeric_value'] = pd.to_numeric(df['value'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')

    # --- SIDEBAR FILTERS ---
    with st.sidebar:
        st.header("📋 Study Control Panel")
        all_papers = sorted(list(df['source_paper'].unique()))
        selected_paper = st.multiselect("Filter by Paper", all_papers, default=all_papers)
        
        st.divider()
        st.write("**Quick Metrics**")
        st.metric("Total Associations", len(df))
        
        # Identify Prognostic variables (HR, OR, AUC)
        prognostic_mask = df['variable'].str.contains('HR|OR|AUC|P-value', case=False, na=False)
        st.metric("Statistical Predictors", prognostic_mask.sum())
        
        # Identify Cohort Characteristics (Age, Sex, BMI, etc.)
        cohort_mask = df['variable'].str.contains('Age|Sex|Male|BMI|Baseline', case=False, na=False)
        st.metric("Cohort Metrics", cohort_mask.sum())

    # Filter Data based on sidebar
    filtered_df = df[df['source_paper'].isin(selected_paper)]

    # --- MAIN CONTENT ---
    tab1, tab2, tab3 = st.tabs(["📈 Visual Analytics", "📋 Evidence Table", "🔍 Deep Traceability"])

    with tab1:
        st.subheader("Statistical Evidence Landscape")
        
        viz_col1, viz_col2 = st.columns(2)
        
        # with viz_col1:
        #     st.write("**Association Hierarchy**")
        #     # Sunburst: Paper -> Variable
        #     fig_sun = px.sunburst(
        #         filtered_df, 
        #         path=['source_paper', 'variable'], 
        #         color='source_paper',
        #         title="Evidence Distribution by Study"
        #     )
        #     st.plotly_chart(fig_sun, width='stretch')

        with viz_col1:
            st.write("**Literature Evidence Hierarchy**")
            
            # 1. Prepare levels for the Sunburst
            # We create helper columns to ensure the levels match your prompt's requirements
            sun_df = filtered_df.copy()
            
            # Categorize outcome (e.g., if '28' is in variable/text, it's 28-day Mortality)
            sun_df['outcome_type'] = sun_df['variable'].apply(
                lambda x: "28-day Mort" if "28" in str(x) else ("90-day Mort" if "90" in str(x) else "General Outcome")
            )
            
            # Categorize Metric Type (OR vs HR vs AUC)
            sun_df['metric_type'] = sun_df['variable'].apply(
                lambda x: "Hazard Ratio" if "HR" in str(x) else ("Odds Ratio" if "OR" in str(x) else "Other Metric")
            )

            # 2. Build the Sunburst
            # Levels: Source -> Outcome Definition -> Metric Type -> Variable Name
            fig_sun = px.sunburst(
                sun_df, 
                path=['source_paper', 'outcome_type', 'metric_type', 'variable'], 
                color='outcome_type',
                color_discrete_map={'28-day Mort': '#636EFA', '90-day Mort': '#EF553B', 'General Outcome': '#00CC96'},
                title="Prognostic Evidence Map (Use Case 1)"
            )
            
            fig_sun.update_traces(textinfo="label+percent entry")
            st.plotly_chart(fig_sun, width='stretch')

        with viz_col2:
            st.write("**Effect Size Distribution (Forest Plot)**")
            # Only plot HR/OR values
            forest_df = filtered_df[filtered_df['variable'].str.contains('HR|OR', case=False, na=False)].dropna(subset=['numeric_value'])
            if not forest_df.empty:
                fig_forest = px.scatter(
                    forest_df, 
                    x='numeric_value', 
                    y='variable', 
                    color='source_paper',
                    hover_data=['value', 'evidence_text'],
                    labels={'numeric_value': 'Effect Size (Value > 1.0 = Higher Risk)'}
                )
                fig_forest.add_vline(x=1.0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_forest, width='stretch')
            else:
                st.info("No Hazard/Odds Ratios found for plotting.")

    # with tab1:
    #     st.subheader("📊 Visualizing Use Case 1 Requirements")
        
    #     # Pre-processing for the specific levels requested
    #     viz_df = filtered_df.copy()
    #     viz_df['Outcome_Def'] = viz_df.apply(categorize_outcome, axis=1)
    #     viz_df['Stat_Method'] = viz_df.apply(categorize_metric, axis=1)
        
    #     col_a, col_b = st.columns([1, 1])
        
    #     with col_a:
    #         st.markdown("### 1. Coverage of Evidence")
    #         st.caption("Levels: Study ➔ Outcome Definition ➔ Stat Method ➔ Variable")
    #         fig_sun = px.sunburst(
    #             viz_df,
    #             path=['source_paper', 'Outcome_Def', 'Stat_Method', 'variable'],
    #             color='Outcome_Def',
    #             color_discrete_sequence=px.colors.qualitative.Bold
    #         )
    #         st.plotly_chart(fig_sun, use_container_width=True)

    #     with col_b:
    #         st.markdown("### 2. Direction of Risk (Forest Plot)")
    #         st.caption("Points right of 1.0 = Higher risk of mortality in standard care.")
            
    #         # Filter for numeric HR/OR
    #         forest_df = viz_df[viz_df['Stat_Method'].str.contains('Ratio')].copy()
    #         forest_df['val_num'] = pd.to_numeric(forest_df['value'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    #         forest_df = forest_df.dropna(subset=['val_num'])
            
    #         if not forest_df.empty:
    #             fig_forest = px.scatter(
    #                 forest_df, x='val_num', y='variable', color='source_paper',
    #                 labels={'val_num': 'Effect Size (Odds/Hazard Ratio)'}
    #             )
    #             fig_forest.add_vline(x=1.0, line_dash="dash", line_color="red")
    #             st.plotly_chart(fig_forest, use_container_width=True)
    #         else:
    #             st.info("No HR/OR values found to plot.")

    with tab2:
        st.subheader("Extracted Clinical Variables & Effects")
        
        search = st.text_input("🔍 Search Variables", placeholder="e.g., Lactate, SOFA, APACHE")
        if search:
            # Added fillna to prevent search crashes
            filtered_df = filtered_df[
                filtered_df['variable'].str.contains(search, case=False, na=False) | 
                filtered_df['evidence_text'].str.contains(search, case=False, na=False)
            ]

        display_cols = ['variable', 'value', 'source_paper', 'year', 'evidence_text']
        actual_cols = [c for c in display_cols if c in filtered_df.columns]
        
        st.dataframe(
            filtered_df[actual_cols], 
            width='stretch', 
            hide_index=True
        )

    with tab3:
        st.subheader("Evidence Grounding")
        st.info("Direct AI extractions mapped to original source text.")
        
        # Sort by variable for easier reading
        trace_df = filtered_df.sort_values('variable')
        
        for _, row in trace_df.iterrows():
            with st.expander(f"📑 {row['variable']} | {row['source_paper']}"):
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.metric("Extracted", row['value'])
                with c2:
                    st.markdown(f"**Contextual Evidence:**\n> {row['evidence_text']}")
                    if 'year' in row:
                        st.caption(f"Study Year: {row['year']}")
# --- DOWNLOAD FOR ANALYSIS ---
if all_evidence:
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        "Download Evidence CSV",
        csv,
        "sepsis_mortality_evidence.csv",
        "text/csv",
        key='download-csv'
    )