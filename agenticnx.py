import os
import json
import requests
import networkx as nx
from typing import Literal, List, get_args
from pydantic import BaseModel

# =========================
# 1. GRAPH SCHEMA
# =========================
SepsisVariable = Literal[
    "mortality_icu", "mortality_hospital", "sofa_score", "sap_3_score",
    "lactate_levels", "sepsis_2_criteria", "sepsis_3_criteria",
    "sample_size", "age", "vasopressor_use", "mechanical_ventilation"
]

# =========================
# 2. NEXUS AGENT LOGIC
# =========================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "google/gemini-2.0-flash-001"

def call_nexus_agent(paper_json):
    """Deep analysis of JSON sections to find facts and inter-file connections."""
    allowed = get_args(SepsisVariable)
    
    prompt = f"""
    Analyze the following paper JSON. 
    1. Extract clinical variables matching: {allowed}
    2. Identify correlations (e.g., "Lactate > 4 correlated with high mortality").
    3. Map these into a structured JSON for a Knowledge Graph.

    INPUT JSON:
    {json.dumps(paper_json)[:15000]} 

    OUTPUT FORMAT:
    {{
      "nodes": [
        {{"id": "var_name", "type": "variable", "value": "...", "label": "Human Label"}},
        {{"id": "Besen_2016", "type": "paper", "label": "Study Root"}}
      ],
      "edges": [
        {{"source": "Besen_2016", "target": "var_name", "relation": "measured"}},
        {{"source": "lactate_levels", "target": "mortality_icu", "relation": "predicts", "strength": "high"}}
      ]
    }}
    """
    
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"}
    }
    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    return res.json()['choices'][0]['message']['content']

# =========================
# 3. PROCESSING PIPELINE
# =========================
def transform_to_nx(input_json_path, output_json_path):
    with open(input_json_path, 'r') as f:
        raw_input = json.load(f)

    # Check if the JSON is a list and take the first element if it is
    if isinstance(raw_input, list):
        if len(raw_input) > 0:
            paper_data = raw_input[0]
        else:
            print(f"⚠️ Skipping empty list: {input_json_path}")
            return
    else:
        paper_data = raw_input    
    
    print(f"📡 Processing {paper_data.get('paper_id')}...")
    
    # Get Agentic Graph structure
    graph_raw = call_nexus_agent(paper_data)
    graph_data = json.loads(graph_raw)
    
    # Initialize NetworkX
    G = nx.DiGraph()
    
    # Add Nodes
    for node in graph_data.get("nodes", []):
        G.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
    
    # Add Edges (Connections between content)
    for edge in graph_data.get("edges", []):
        G.add_edge(edge["source"], edge["target"], 
                   relation=edge.get("relation"), 
                   strength=edge.get("strength"))
    
    # Export to Node-Link JSON format
    nx_json = nx.node_link_data(G)
    with open(output_json_path, 'w') as f:
        json.dump(nx_json, f, indent=2)
    print(f"✅ Saved NX Graph to {output_json_path}")

# =========================
# 4. RUNNER
# =========================
if __name__ == "__main__":
    # Update these paths to your directory
    INPUT_DIR = "/home/mechguy/Documents/projects/sepsisefferon/token_burners/Sepsis_hackathon/data/parsed_papers"
    OUTPUT_DIR = "/home/mechguy/Documents/projects/sepsisefferon/token_burners/Sepsis_hackathon/data/agenticnx_jsons"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".json"):
            transform_to_nx(
                os.path.join(INPUT_DIR, filename),
                os.path.join(OUTPUT_DIR, filename.replace(".json", "_nx.json"))
            )