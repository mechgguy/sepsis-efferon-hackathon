import streamlit as st
import pandas as pd
import json
import os
import glob
import plotly.express as px

st.set_page_config(page_title="Sepsis Counterfactual Explorer", layout="wide")

# --- PATH CONFIG ---
DATA_DIR = "data/mortality_counterfactuals"

# --- CATEGORIZATION LOGIC ---

def categorize_domain(row):
    """Categorizes variables into clinical domains."""
    var = str(row.get('variable', '')).lower()
    if any(x in var for x in ['sofa', 'apache', 'saps', 'qsofa', 'score']):
        return "Severity Scores"
    elif any(x in var for x in ['mortality', 'death', 'survivor', 'outcome']):
        return "Clinical Outcomes"
    elif any(x in var for x in ['age', 'sex', 'male', 'female', 'bmi', 'weight']):
        return "Demographics"
    elif any(x in var for x in ['lactate', 'il-6', 'crp', 'procalcitonin', 'creatinine', 'bilirubin', 'plt']):
        return "Lab Biomarkers"
    return "Other Parameters"

def categorize_outcome(row):
    text = (str(row.get('variable', '')) + " " + str(row.get('evidence_text', ''))).lower()
    if "28" in text: return "28-Day Mort"
    if "90" in text: return "90-Day Mort"
    if "hospital" in text: return "In-Hosp Mort"
    return "Gen Mortality"

def categorize_metric(row):
    var = str(row.get('variable', '')).upper()
    if "HR" in var or "HAZARD" in var: return "Hazard Ratio (HR)"
    if "OR" in var or "ODDS" in var: return "Odds Ratio (OR)"
    if "AUC" in var or "ROC" in var: return "AUC/C-Stat"
    return "Descriptive/Other"

@st.cache_data
def load_all_extractions():
    all_data = []
    if not os.path.exists(DATA_DIR): return []
    json_files = glob.glob(os.path.join(DATA_DIR, "*_results.json"))
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if "extractions" in data:
                    for ext in data["extractions"]:
                        ext['source_paper'] = data.get("paper_id", "Unknown")
                        ext['year'] = data.get("year", "N/A")
                        all_data.append(ext)
            except: continue
    return all_data

# --- DATA LOADING ---
all_evidence = load_all_extractions()

st.title("🟢 Use Case 1: Mortality Benchmarking")

if not all_evidence:
    st.warning(f"No extraction files found in `{DATA_DIR}`.")
else:
    df = pd.DataFrame(all_evidence)
    df['numeric_value'] = pd.to_numeric(df['value'].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    
    # Apply Hierarchy
    df['Domain'] = df.apply(categorize_domain, axis=1)
    df['Outcome_Def'] = df.apply(categorize_outcome, axis=1)
    df['Stat_Method'] = df.apply(categorize_metric, axis=1)

    with st.sidebar:
        st.header("📋 Study Control Panel")
        selected_paper = st.multiselect("Filter by Paper", sorted(df['source_paper'].unique()), default=sorted(df['source_paper'].unique()))
        st.divider()
        st.metric("Total Associations", len(df))
        st.metric("Predictive Ratios", df['Stat_Method'].str.contains('Ratio').sum())

    filtered_df = df[df['source_paper'].isin(selected_paper)]

    tab1, tab2, tab3 = st.tabs(["📈 Visual Analytics", "📋 Evidence Table", "🔍 Deep Traceability"])

    with tab1:
        st.subheader("Statistical Evidence Landscape")
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.write("**Domain & Outcome Hierarchy**")
            # Hierarchy: Domain -> Outcome -> Metric -> Variable
            # We sanitize to "N/A" to prevent Sunburst render errors
            sun_df = filtered_df.copy().fillna("N/A")
            fig_sun = px.sunburst(
                sun_df, 
                path=['Domain', 'Outcome_Def', 'Stat_Method', 'variable'], 
                color='Domain',
                title="Evidence Coverage Map",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_sun, use_container_width=True)

        with viz_col2:
            st.write("**Impact Factors (Forest Plot)**")
            forest_df = filtered_df[filtered_df['Stat_Method'].str.contains('Ratio')].dropna(subset=['numeric_value'])
            if not forest_df.empty:
                fig_forest = px.scatter(
                    forest_df, x='numeric_value', y='variable', color='Domain',
                    hover_data=['source_paper', 'value', 'Outcome_Def'],
                    labels={'numeric_value': 'Effect Size (HR/OR)'}
                )
                fig_forest.add_vline(x=1.0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_forest, use_container_width=True)
            else:
                st.info("No numeric HR/OR values found.")

    with tab2:
        st.subheader("Extracted Clinical Variables & Effects")
        search = st.text_input("🔍 Search Variables", placeholder="e.g., Lactate, SOFA, Age")
        display_df = filtered_df.copy()
        if search:
            display_df = display_df[display_df['variable'].str.contains(search, case=False, na=False)]
        
        st.dataframe(
            display_df[['Domain', 'variable', 'value', 'Stat_Method', 'source_paper', 'evidence_text']], 
            use_container_width=True, hide_index=True
        )

    with tab3:
        st.subheader("Evidence Grounding")
        for _, row in filtered_df.sort_values('Domain').iterrows():
            with st.expander(f"[{row['Domain']}] {row['variable']} | {row['source_paper']}"):
                c1, c2 = st.columns([1, 4])
                c1.metric("Value", row['value'])
                c2.markdown(f"**Source Text:**\n> {row['evidence_text']}")

    # Download Button
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download CSV", csv, "sepsis_evidence.csv", "text/csv")