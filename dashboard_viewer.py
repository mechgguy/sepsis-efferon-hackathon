# Views json files
import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="Sepsis Atlas", layout="wide")

# --- PATH CONFIG ---
RESULTS_DIR = os.path.join(os.getcwd(), "./data/parsed_papers/")
# RESULTS_DIR = os.path.join(os.getcwd(), "./data/mortality_counterfactuals/")

st.title("🏥 Sepsis Biomarker Atlas")

# --- DEBUG SIDEBAR ---
with st.sidebar:
    st.header("🛠 System Check")
    if os.path.exists(RESULTS_DIR):
        files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]
        st.success(f"Found {len(files)} JSON files")
    else:
        st.error(f"Directory NOT FOUND: {RESULTS_DIR}")

# --- DATA LOADING ---
@st.cache_data
def load_all_data():
    all_rows = []
    if not os.path.exists(RESULTS_DIR):
        return pd.DataFrame()
    
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]
    for filename in files:
        file_path = os.path.join(RESULTS_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                paper_id = data.get("paper_id", "Unknown")
                
                # Flatten the 'sections' list into individual rows
                sections = data.get("sections", [])
                if sections:
                    for section in sections:
                        row = {
                            "paper_id": paper_id,
                            "heading": section.get("heading"),
                            "text": section.get("text"),
                            "page_start": section.get("page_start")
                        }
                        all_rows.append(row)
                else:
                    # If no sections, just add the paper_id info
                    all_rows.append({"paper_id": paper_id, "heading": "N/A", "text": "No content"})
                    
            except Exception as e:
                st.sidebar.warning(f"Error loading {filename}: {e}")
    
    return pd.DataFrame(all_rows)

df = load_all_data()

# --- DISPLAY ---
if df.empty:
    st.warning(f"No valid data found in `{RESULTS_DIR}`.")
else:
    # Sidebar Filters
    papers = df['paper_id'].unique()
    selected_paper = st.sidebar.selectbox("📄 Select Paper", papers)
    
    # Filter data for selected paper
    paper_df = df[df['paper_id'] == selected_paper]
    
    # Top Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Papers", len(papers))
    c2.metric("Total Sections", len(df))
    c3.metric("Sections in Current Paper", len(paper_df))

    st.divider()

    # Layout: List of sections on the left, Content on the right
    col_list, col_content = st.columns([1, 2])

    with col_list:
        st.subheader("Sections")
        # Use a radio or selectbox to pick a specific heading to read
        selected_heading = st.radio(
            "Go to section:", 
            paper_df['heading'].tolist(),
            index=0 if not paper_df.empty else None
        )

    with col_content:
        st.subheader("Content")
        if selected_heading:
            content = paper_df[paper_df['heading'] == selected_heading]['text'].values[0]
            st.markdown(f"### {selected_heading}")
            st.write(content)
        
    # Optional: Show the raw table view at the bottom
    with st.expander("📊 View Raw Flattened Data"):
        # st.dataframe(df, use_container_width=True)
        st.dataframe(df, width='stretch')