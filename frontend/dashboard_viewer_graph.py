import streamlit as st
import pandas as pd
import json
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import re
import numpy as np

st.set_page_config(page_title="Sepsis Global Attribute Atlas", layout="wide")

# --- PATH CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data", "parsed_papers"))

# --- ATTRIBUTE DISCOVERY LOGIC ---
DOMAIN_MAP = {
    "Severity Scores": ['sofa', 'apache', 'saps', 'qsofa', 'mews'],
    "Biomarkers": ['lactate', 'il-6', 'crp', 'procalcitonin', 'leukocyte', 'creatinine', 'bilirubin', 'plt'],
    "Demographics": ['age', 'sex', 'male', 'female', 'bmi', 'weight', 'ethnicity', 'comorbidity'],
    "Treatment": ['vasopressor', 'norepinephrine', 'fluid', 'antibiotic', 'ventilation', 'dialysis'],
    "Outcomes": ['mortality', 'death', 'discharge', 'icu stay', 'readmission']
}

def extract_attributes(text):
    found = []
    text_lower = text.lower()
    for domain, keywords in DOMAIN_MAP.items():
        for kw in keywords:
            if kw in text_lower:
                found.append({"Domain": domain, "Attribute": kw.capitalize()})
    return found

@st.cache_data
def load_raw_text_data():
    all_sections = []
    all_attributes = []
    all_links = [] 
    
    if not os.path.exists(TEXT_DIR):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    files = glob.glob(os.path.join(TEXT_DIR, "*.json"))
    for f_path in files:
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, str):
                    data = json.loads(data)
                if not isinstance(data, list):
                    continue

                for chunk in data:
                    if isinstance(chunk, str):
                        continue
                        
                    content = chunk.get("text", "")
                    metadata = chunk.get("metadata", {})
                    
                    paper_id = metadata.get("paper_id", "Unknown")
                    heading = metadata.get("section", "Untitled")
                    page_num = metadata.get("page_number", "N/A")

                    all_sections.append({
                        "paper_id": paper_id, 
                        "heading": heading, 
                        "text": content,
                        "page": page_num
                    })
                    
                    found = extract_attributes(content)
                    unique_attrs_in_sec = list({item["Attribute"] for item in found})
                    
                    for item in found:
                        all_attributes.append({
                            "paper_id": paper_id,
                            "Domain": item["Domain"],
                            "Attribute": item["Attribute"],
                            "Section": heading,
                            "Page": page_num
                        })
                    
                    for i in range(len(unique_attrs_in_sec)):
                        for j in range(i + 1, len(unique_attrs_in_sec)):
                            all_links.append({
                                "source": unique_attrs_in_sec[i],
                                "target": unique_attrs_in_sec[j],
                                "paper_id": paper_id
                            })
        except Exception as e:
            print(f"Error processing {f_path}: {e}")
            continue

    df_text = pd.DataFrame(all_sections)
    df_attr = pd.DataFrame(all_attributes)
    df_links = pd.DataFrame(all_links, columns=['source', 'target', 'paper_id'])
    
    return df_text, df_attr, df_links

df_text, df_attr, df_links = load_raw_text_data()

# --- UI LOGIC ---
if df_text.empty:
    st.error(f"No text data found in {TEXT_DIR}. Check if JSON files match the expected format.")
else:
    with st.sidebar:
        st.header("⚙️ Global Filter")
        paper_list = sorted(df_text['paper_id'].unique())
        selected_papers = st.multiselect("Active Papers", paper_list, default=paper_list)

    f_text = df_text[df_text['paper_id'].isin(selected_papers)]
    f_attr = df_attr[df_attr['paper_id'].isin(selected_papers)]
    f_links = df_links[df_links['paper_id'].isin(selected_papers)] if not df_links.empty else pd.DataFrame(columns=['source', 'target', 'paper_id'])

    tab_viz, tab_search, tab_explain, tab_consensus, tab_atlas = st.tabs([
        "📊 Attribute Analytics", "🔍 Traceability", "🧠 Explainability & Graph", "🤝 Cross-Study Consensus", "📖 Literature Atlas"
    ])

    # --- TAB: TRACEABILITY ---
    with tab_search:
        st.subheader("📋 Extracted Clinical Variables & Evidence")
        st.info("This registry maps discovered clinical variables back to their exact location in the source literature. Use the search bar to find specific evidence for biomarkers or scores.")
        
        search_var = st.text_input("🔍 Search Variable Name", placeholder="e.g., Lactate")
        display_df = f_attr.copy()
        if search_var:
            display_df = display_df[display_df['Attribute'].str.contains(search_var, case=False, na=False)]

        registry_table = display_df.rename(columns={
            "Attribute": "Variable",
            "paper_id": "Source Paper",
            "Section": "Section Heading",
            "Page": "Pg #"
        })

        st.dataframe(
            registry_table[['Domain', 'Variable', 'Source Paper', 'Pg #', 'Section Heading']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Pg #": st.column_config.NumberColumn("Pg #", format="%d"),
                "Variable": st.column_config.TextColumn("Variable", width="medium"),
                "Source Paper": st.column_config.TextColumn("Source Paper", width="medium"),
                "Section Heading": st.column_config.TextColumn("Evidence Context", width="large"),
            }
        )

    # --- TAB: ANALYTICS ---
    with tab_viz:
        st.subheader("🎯 Clinical Attribute Coverage")
        st.write("This sunburst chart shows the hierarchy of clinical data. **Inner rings** represent broad domains (e.g., Biomarkers), **middle rings** show specific attributes, and **outer rings** identify the source papers contributing to that data point.")
        fig_sun = px.sunburst(f_attr, path=['Domain', 'Attribute', 'paper_id'], color='Domain', template="plotly_dark")
        fig_sun.update_layout(height=700)
        st.plotly_chart(fig_sun, use_container_width=True)

        st.divider()
        st.subheader("📊 Information Density")
        st.write("This chart tracks how many clinical variables were extracted per paper. High-density bars indicate 'feature-rich' studies that are likely central to sepsis benchmarking.")
        density = f_attr.groupby(['paper_id', 'Domain']).size().reset_index(name='Mention Count')
        fig_bar = px.bar(density, x='Mention Count', y='paper_id', color='Domain', orientation='h', template="plotly_dark")
        fig_bar.update_layout(height=max(400, len(selected_papers) * 30))
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- TAB: EXPLAINABILITY ---
    with tab_explain:
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Attribute Consensus Matrix**")
            st.caption("Heatmap showing the frequency of attribute mentions across papers. Bright spots highlight high-consensus variables used across the entire study corpus.")
            if not f_attr.empty:
                matrix_data = f_attr.groupby(['Attribute', 'paper_id']).size().unstack(fill_value=0)
                fig_heat = px.imshow(matrix_data, color_continuous_scale="Viridis", aspect="auto")
                truncated_paper_ids = [pid[:30] + '...' if len(pid) > 30 else pid for pid in matrix_data.columns]
                fig_heat.update_layout(
                    xaxis=dict(tickmode='array', tickvals=list(range(len(matrix_data.columns))), ticktext=truncated_paper_ids, tickangle=45, automargin=True),
                    yaxis=dict(tickmode='linear', dtick=1, automargin=True),
                    height=800, margin=dict(l=150, b=150)
                )
                st.plotly_chart(fig_heat, use_container_width=True)
        with c2:
            st.write("**Clinical Relationship Graph**")
            st.caption("Network analysis of 'Co-occurrence'. Nodes (variables) are linked if they appear in the same section, revealing how biomarkers like Lactate relate to Outcomes like Mortality.")
            if not f_links.empty:
                edge_df = f_links.groupby(['source', 'target']).size().reset_index(name='weight')
                G = nx.from_pandas_edgelist(edge_df, 'source', 'target', ['weight'])
                pos = nx.spring_layout(G, k=0.6)
                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
                
                edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#555'), mode='lines')
                node_trace = go.Scatter(x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()], 
                                        mode='markers+text', text=list(G.nodes()), textposition="top center",
                                        marker=dict(size=12, color='skyblue'))
                fig_graph = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, height=800, xaxis_visible=False, yaxis_visible=False))
                st.plotly_chart(fig_graph, use_container_width=True)

    # --- TAB: CONSENSUS CLOUD ---
    with tab_consensus:
        st.subheader("🤝 Cross-Study Consensus Cloud")
        st.write("A 'Clinical Importance' map. **Larger, brighter bubbles** represent attributes mentioned most frequently across the highest number of unique studies.")
        if not f_attr.empty:
            consensus_df = f_attr.groupby('Attribute').agg({'paper_id': 'nunique', 'Attribute': 'count'}).rename(columns={'paper_id': 'Study_Count', 'Attribute': 'Total_Mentions'}).reset_index()
            consensus_df['x_pos'] = np.linspace(0, 10, len(consensus_df))
            consensus_df['y_pos'] = np.random.uniform(2, 5, len(consensus_df))
            fig_cloud = px.scatter(consensus_df, x='x_pos', y='y_pos', size='Total_Mentions', color='Total_Mentions', text='Attribute', size_max=60, hover_data=['Study_Count'], template="plotly_dark")
            fig_cloud.update_layout(showlegend=False, height=600, xaxis_visible=False, yaxis_visible=False)
            st.plotly_chart(fig_cloud, use_container_width=True)

    # --- TAB: ATLAS ---
    with tab_atlas:
        st.subheader("📖 Literature Atlas")
        st.info("The original source of truth. Select a paper and section to read the raw text and verify the extracted metadata.")
        paper_choice = st.selectbox("Select Paper", sorted(f_text['paper_id'].unique()))
        paper_content = f_text[f_text['paper_id'] == paper_choice]
        l, r = st.columns([1, 3])
        with l:
            sec_choice = st.radio("Sections", paper_content['heading'].tolist())
        with r:
            row = paper_content[paper_content['heading'] == sec_choice].iloc[0]
            st.markdown(f"### {sec_choice}")
            st.caption(f"Source: {row['paper_id']} | Page: {row['page']}")
            st.write(row['text'])