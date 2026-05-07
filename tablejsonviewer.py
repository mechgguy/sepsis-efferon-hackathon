import streamlit as st
import pandas as pd
import json
import os
import re

st.set_page_config(page_title="Sepsis Evidence Explorer", layout="wide")

# --- PATH CONFIG ---
# Directory where your table-heavy JSON files are stored
TABLES_DIR = os.path.join(os.getcwd(), "./data/parsed_papers/")

def parse_markdown_table(md_string):
    """Parses markdown table and handles duplicate column names."""
    try:
        lines = [line.strip() for line in md_string.split('\n') if '|' in line]
        if len(lines) < 3: return None
        
        # 1. Extract and clean headers
        raw_headers = [cell.strip() for cell in lines[0].split('|') if cell.strip()]
        
        # 2. De-duplicate headers (Fixes the ValueError)
        final_headers = []
        counts = {}
        for h in raw_headers:
            if h in counts:
                counts[h] += 1
                final_headers.append(f"{h}_{counts[h]}")
            else:
                counts[h] = 0
                final_headers.append(h)
        
        # 3. Extract data rows
        data = []
        for line in lines[2:]:  # Skip header and '---' separator
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            # Ensure row matches header length, or pad it
            if len(cells) > 0:
                if len(cells) > len(final_headers):
                    cells = cells[:len(final_headers)]
                elif len(cells) < len(final_headers):
                    cells += [""] * (len(final_headers) - len(cells))
                data.append(cells)
        
        return pd.DataFrame(data, columns=final_headers)
    except Exception as e:
        # Fallback: if parsing fails, we don't crash the whole app
        return None

@st.cache_data
def load_table_data():
    all_evidence = []
    if not os.path.exists(TABLES_DIR):
        return []
    
    files = [f for f in os.listdir(TABLES_DIR) if f.endswith(".json")]
    for filename in files:
        with open(os.path.join(TABLES_DIR, filename), "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for entry in data:
                    entry['source_file'] = filename
                    all_evidence.append(entry)
            except:
                continue
    return all_evidence

evidence_data = load_table_data()

# --- UI LAYOUT ---
st.title("📊 Sepsis Clinical Evidence Explorer")
st.markdown("Extracting insights from structured clinical trial tables.")

if not evidence_data:
    st.info(f"Please place your JSON files in `{TABLES_DIR}`")
else:
    # --- SEARCH & FILTER ---
    search_query = st.text_input("🔍 Search for a biomarker or metric (e.g., 'p-SOFA', 'Mortality', 'Lactate')")
    
    # Filter logic
    filtered_evidence = [
        e for e in evidence_data 
        if search_query.lower() in e['markdown'].lower() or search_query.lower() in e['preceding_heading'].lower()
    ]

    # --- TABS FOR DIFFERENT VIEWS ---
    tab1, tab2 = st.tabs(["📝 Evidence Cards", "📋 Raw Table View"])

    with tab1:
        if not filtered_evidence:
            st.warning("No matches found.")
        else:
            for idx, item in enumerate(filtered_evidence):
                with st.container():
                    st.subheader(f"Finding {idx+1}: {item['preceding_heading']}")
                    
                    # Highlight significant p-values in the card
                    p_values = re.findall(r"p[<=]\s?0\.0\d+", item['markdown'])
                    if p_values:
                        st.markdown(f"**Significant Values Detected:** {', '.join(set(p_values))}")
                    
                    # Render the actual table
                    df_table = parse_markdown_table(item['markdown'])
                    if df_table is not None:
                        st.dataframe(df_table, hide_index=True)
                    else:
                        st.markdown(item['markdown'])
                    
                    st.caption(f"Source: {item['source_file']} | Page: {item.get('page_start', 'N/A')}")
                    st.divider()

    with tab2:
        st.subheader("Aggregated Table Indices")
        summary_df = pd.DataFrame(evidence_data)[['source_file', 'preceding_heading', 'page_start']]
        st.dataframe(summary_df, width='stretch')

# --- ANALYTICS SIDEBAR ---
with st.sidebar:
    st.header("📈 Quick Analytics")
    if evidence_data:
        total_tables = len(evidence_data)
        st.metric("Tables Extracted", total_tables)
        
        # Check for SOFA mention frequency
        sofa_mentions = sum(1 for e in evidence_data if 'sofa' in e['markdown'].lower())
        st.metric("SOFA/p-SOFA Mentions", sofa_mentions)
        
        st.markdown("---")
        st.write("**Top Study Keywords:**")
        # Simple keyword counter
        all_text = " ".join([e['preceding_heading'] for e in evidence_data]).lower()
        keywords = ["mortality", "shock", "sofa", "icu", "correlation", "characteristics"]
        for k in keywords:
            count = all_text.count(k)
            if count > 0:
                st.write(f"- {k.title()}: {count}")