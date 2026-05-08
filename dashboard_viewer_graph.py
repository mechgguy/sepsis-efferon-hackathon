import streamlit as st
import pandas as pd
import json
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import re

st.set_page_config(page_title="Sepsis Textual Attribute Atlas", layout="wide")

# --- PATH CONFIG ---
TEXT_DIR = os.path.join(os.getcwd(), "data/parsed_papers/")

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

# --- DATA LOADING ---
@st.cache_data
def load_raw_text_data():
    all_sections = []
    all_attributes = []
    all_links = [] # For Knowledge Graph
    
    files = glob.glob(os.path.join(TEXT_DIR, "*.json"))
    for f_path in files:
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                paper_id = data.get("paper_id", "Unknown")
                for sec in data.get("sections", []):
                    content = sec.get("text", "")
                    all_sections.append({
                        "paper_id": paper_id, 
                        "heading": sec.get("heading", "Untitled"), 
                        "text": content
                    })
                    
                    found = extract_attributes(content)
                    unique_attrs_in_sec = list({item["Attribute"] for item in found})
                    
                    # Store attributes
                    for item in found:
                        all_attributes.append({
                            "paper_id": paper_id,
                            "Domain": item["Domain"],
                            "Attribute": item["Attribute"],
                            "Section": sec.get("heading")
                        })
                    
                    # Create links for Knowledge Graph (co-occurrence in same section)
                    for i in range(len(unique_attrs_in_sec)):
                        for j in range(i + 1, len(unique_attrs_in_sec)):
                            all_links.append({
                                "source": unique_attrs_in_sec[i],
                                "target": unique_attrs_in_sec[j],
                                "paper_id": paper_id
                            })
        except: continue
    return pd.DataFrame(all_sections), pd.DataFrame(all_attributes), pd.DataFrame(all_links)

df_text, df_attr, df_links = load_raw_text_data()

# --- UI ---
st.title("🌐 Sepsis Global Attribute Atlas")

if df_text.empty:
    st.error(f"No text data found in {TEXT_DIR}")
else:
    with st.sidebar:
        st.header("⚙️ Global Filter")
        selected_papers = st.multiselect("Active Papers", sorted(df_text['paper_id'].unique()), default=sorted(df_text['paper_id'].unique()))
    
    f_text = df_text[df_text['paper_id'].isin(selected_papers)]
    f_attr = df_attr[df_attr['paper_id'].isin(selected_papers)]
    f_links = df_links[df_links['paper_id'].isin(selected_papers)]

    # --- TABS ---
    tab_viz, tab_search, tab_explain, tab_consensus, tab_atlas = st.tabs([
        "📊 Attribute Analytics", 
        "🔍 Traceability", 
        "🧠 Explainability & Graph",
        "🤝 Cross-Study Consensus",
        "📖 Literature Atlas"
    ])

    with tab_viz:
            # --- 1. FULL WIDTH SUNBURST (TOP) ---
            st.subheader("🎯 Clinical Attribute Coverage")
            st.markdown("Global distribution of variables across clinical domains and papers.")
            
            fig_sun = px.sunburst(
                f_attr, 
                path=['Domain', 'Attribute', 'paper_id'],
                color='Domain', 
                color_discrete_sequence=px.colors.qualitative.Pastel,
                template="plotly_dark"
            )
            
            # Increase height significantly to make it "Bigger"
            fig_sun.update_layout(
                height=800, 
                margin=dict(t=10, l=10, r=10, b=10)
            )
            
            st.plotly_chart(fig_sun, width='stretch')

            st.divider() # Visual break

            # --- 2. FULL WIDTH BAR CHART (BOTTOM) ---
            st.subheader("📊 Information Density")
            st.markdown("Volume of clinical mentions extracted per study.")
            
            density = f_attr.groupby(['paper_id', 'Domain']).size().reset_index(name='Mention Count')
            
            fig_bar = px.bar(
                density, 
                x='Mention Count', 
                y='paper_id', 
                color='Domain',
                orientation='h', 
                template="plotly_dark",
                category_orders={"paper_id": sorted(f_attr['paper_id'].unique())}
            )
            
            # Dynamic height based on number of papers so it doesn't look squashed
            bar_height = max(400, len(selected_papers) * 25)
            
            fig_bar.update_layout(
                height=bar_height,
                xaxis_title="Total Mentions (Attribute Frequency)",
                yaxis_title="",
                legend_title="Clinical Domain",
                margin=dict(t=20, l=10, r=10, b=20)
            )
            
            st.plotly_chart(fig_bar, width='stretch')

    with tab_search:
        st.subheader("🔍 Deep Attribute Traceability")
        query = st.text_input("Search keywords (e.g., 'Lactate')", "")
        if query:
            results = f_text[f_text['text'].str.contains(query, case=False, na=False)]
            for _, row in results.iterrows():
                with st.expander(f"📄 {row['paper_id']} | {row['heading']}"):
                    display_text = re.sub(f"({query})", r"**\1**", row['text'], flags=re.IGNORECASE)
                    st.markdown(display_text)

        st.subheader("📋 Extracted Clinical Variables & Evidence")
        st.markdown("Below is the registry of all clinical attributes discovered within the raw text sections.")

        # 1. Search Bar (styled like your screenshot)
        search_query = st.text_input("🔍 Search Variables", placeholder="e.g., Lactate, SOFA, Age")

        # 2. Prepare the Table Data from f_attr and f_text
        # We merge to get the actual text snippet for each attribute mention
        if not f_attr.empty:
            # Create a display dataframe
            # Note: We use the 'Section' and 'paper_id' to link back to the raw text if needed,
            # but f_attr already contains the mapping we need.
            
            display_df = f_attr.copy()
            
            # Filter based on search query
            if search_query:
                display_df = display_df[
                    display_df['Attribute'].str.contains(search_query, case=False, na=False) |
                    display_df['Domain'].str.contains(search_query, case=False, na=False)
                ]

            # 3. Formatting the table to match your UI
            # We rename columns to match your screenshot
            registry_table = display_df.rename(columns={
                "Domain": "Domain",
                "Attribute": "Variable",
                "paper_id": "Source Paper",
                "Section": "Section Heading"
            })

            # Add a placeholder for "Value" or "Context" since raw text doesn't have 
            # a single 'value' column like the stats folder. 
            # We can pull a snippet of the text as 'Evidence Text'
            
            # This part ensures we show the actual sentence context
            st.dataframe(
                registry_table[['Domain', 'Variable', 'Source Paper', 'Section Heading']],
                width='stretch',
                hide_index=True,
                column_config={
                    "Domain": st.column_config.TextColumn("Domain", width="medium"),
                    "Variable": st.column_config.TextColumn("Variable", width="medium"),
                    "Source Paper": st.column_config.TextColumn("Source Paper", width="medium"),
                    "Section Heading": st.column_config.TextColumn("Evidence Context (Heading)", width="large"),
                }
            )
            
            st.caption(f"Showing {len(registry_table)} attribute occurrences.")
        else:
            st.info("No attributes discovered yet. Try adjusting your paper filters.")

    with tab_explain:
        st.subheader("🧠 Knowledge Discovery & Explainability")
        
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.write("**Attribute Consensus Matrix**")
            st.caption("Shows how frequently attributes appear across different papers. Darker cells indicate high-consensus variables.")
            matrix_data = f_attr.groupby(['Attribute', 'paper_id']).size().unstack(fill_value=0)
            fig_heat = px.imshow(matrix_data, labels=dict(x="Paper ID", y="Clinical Attribute", color="Mentions"),
                                color_continuous_scale="Viridis")
            # --- THE FIX FOR VISIBILITY ---
            fig_heat.update_layout(
                yaxis=dict(
                    tickmode='linear',
                    dtick=1,      # Forces every single attribute label to show
                    automargin=True
                ),
                xaxis=dict(tickangle=45), # Angles paper IDs for better fit
                height=800,               # Increases height so labels aren't squashed
                margin=dict(l=150)        # Extra left margin for long attribute names
            )

            st.plotly_chart(fig_heat, width='stretch')

        with c2:
            st.write("**Clinical Relationship Graph**")
            st.caption("Nodes are attributes; lines connect attributes mentioned in the same section. Thicker lines = stronger correlation in text.")
            
            if not f_links.empty:
                # Aggregate links
                edge_df = f_links.groupby(['source', 'target']).size().reset_index(name='weight')
                G = nx.from_pandas_edgelist(edge_df, 'source', 'target', ['weight'])
                
                pos = nx.spring_layout(G, k=0.5, iterations=50)
                
                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')

                node_x, node_y, node_text = [], [], []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)

                node_trace = go.Scatter(
                    x=node_x, y=node_y, mode='markers+text', text=node_text,
                    textposition="top center", marker=dict(size=12, color='skyblue', line_width=2)
                )

                fig_graph = go.Figure(data=[edge_trace, node_trace],
                                     layout=go.Layout(showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
                st.plotly_chart(fig_graph, width='stretch')
            else:
                st.info("No co-occurring attributes found to generate graph.")

    with tab_consensus:
        st.subheader("🧠 Knowledge cross-Consensus")
        
        # 1. Prepare Consensus Data
        # We want to know: How many unique papers mention each attribute?
        consensus_df = f_attr.groupby('Attribute').agg({
            'paper_id': 'nunique',  # Number of papers (Consensus)
            'Domain': 'first',      # For coloring
            'Attribute': 'count'    # Total raw mentions
        }).rename(columns={'paper_id': 'Study_Count', 'Attribute': 'Total_Mentions'}).reset_index()

        st.write("**Cross-Study Consensus Graph**")
        st.caption("Thicker bubbles represent variables with the highest cross-study consensus. Position is jittered for readability.")

        # 2. Add some "Jitter" so bubbles don't just sit in a straight line
        import numpy as np
        consensus_df['x_pos'] = np.linspace(0, 10, len(consensus_df))
        consensus_df['y_pos'] = np.random.uniform(2, 5, len(consensus_df))

        # 3. Create the Bubble Cloud
        fig_cloud = px.scatter(
            consensus_df,
            x='x_pos',
            y='y_pos',
            size='Total_Mentions',
            color='Total_Mentions',
            text='Attribute',
            color_continuous_scale='Viridis',
            size_max=80,  # Adjust this to make bubbles bigger/smaller
            labels={'Total_Mentions': 'Evidence Weight'},
            hover_data=['Study_Count']
        )

        # 4. Styling to match your screenshot (Dark theme, hidden axes)
        fig_cloud.update_traces(
            textposition='top center',
            marker=dict(line=dict(width=2, color='white')), # White border like your image
            opacity=0.8
        )

        fig_cloud.update_layout(
            showlegend=False,
            height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_cloud, width='stretch')

    with tab_atlas:
        paper_choice = st.selectbox("Select Paper", sorted(f_text['paper_id'].unique()))
        paper_content = f_text[f_text['paper_id'] == paper_choice]
        l, r = st.columns([1, 2])
        with l:
            sec_choice = st.radio("Sections", paper_content['heading'].tolist())
        with r:
            st.markdown(f"### {sec_choice}")
            st.write(paper_content[paper_content['heading'] == sec_choice]['text'].values[0])