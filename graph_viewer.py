import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="Sepsis Knowledge Graph", layout="wide")

# --- PATH CONFIG ---
TABLES_DIR = os.path.join(os.getcwd(), "./data/agenticnx_jsons/")

st.title("🕸️ Sepsis Knowledge Graph Explorer")
st.caption(f"Reading from: {TABLES_DIR}")

def load_graph_data():
    all_nodes = []
    all_links = []
    
    if not os.path.exists(TABLES_DIR):
        return [], []
    
    json_files = [f for f in os.listdir(TABLES_DIR) if f.endswith(".json")]
    
    for filename in json_files:
        with open(os.path.join(TABLES_DIR, filename), "r") as f:
            try:
                data = json.load(f)
                # Matches your 'head' output: Dictionary with 'nodes' key
                if isinstance(data, dict) and "nodes" in data:
                    paper_id = filename.replace("_nx.json", "")
                    
                    # Process Nodes
                    for node in data["nodes"]:
                        node["origin_paper"] = paper_id
                        all_nodes.append(node)
                    
                    # Process Links (NetworkX default key is 'links')
                    links = data.get("links", [])
                    for link in links:
                        link["origin_paper"] = paper_id
                        all_links.append(link)
            except Exception as e:
                continue
    return all_nodes, all_links

# Load Data
nodes, links = load_graph_data()

if not nodes:
    st.error("No graph data detected!")
    st.info("Ensure your Agentic script has finished running.")
    if os.path.exists(TABLES_DIR):
        st.write("Files found:", os.listdir(TABLES_DIR))
    st.stop()

# --- INTERFACE ---
search = st.text_input("🔍 Search Clinical Variables (e.g., 'mortality', 'sofa')", "").lower()

# Filter logic: Variables often have 'label' or 'id'
filtered_nodes = [
    n for n in nodes 
    if n.get("type") == "variable" and 
    (search in str(n.get("label", "")).lower() or search in str(n.get("id", "")).lower())
]

tab1, tab2, tab3 = st.tabs(["📊 Variable Insights", "🔗 Relationships", "📦 Raw Graph Data"])

with tab1:
    if not filtered_nodes:
        st.warning("No variables match your search.")
    else:
        for n in filtered_nodes:
            with st.expander(f"📍 {n.get('label', n.get('id'))} ({n.get('origin_paper')})"):
                col1, col2 = st.columns(2)
                col1.metric("Value", n.get("value", "N/A"))
                col2.write(f"**Node Type:** {n.get('type')}")
                
                # Find associated reasoning from links
                related_links = [l for l in links if l.get('target') == n.get('id')]
                if related_links:
                    st.info(f"**Evidence:** {related_links[0].get('relation', 'Connected')} - {related_links[0].get('strength', 'N/A')} strength")

with tab2:
    if links:
        st.subheader("Extracted Relationships")
        links_df = pd.DataFrame(links)
        st.table(links_df[['source', 'target', 'relation', 'origin_paper']].head(20))

with tab3:
    st.subheader("Global Node Registry")
    st.dataframe(pd.DataFrame(nodes), width='stretch')

# --- SIDEBAR ---
with st.sidebar:
    st.header("📈 Graph Metrics")
    st.metric("Total Entities", len(nodes))
    st.metric("Total Links", len(links))
    if st.button("♻️ Force Refresh Data"):
        st.cache_data.clear()
        st.rerun()