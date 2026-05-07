import os
import json
import requests
import fitz  # PyMuPDF
import networkx as nx
from networkx.readwrite import json_graph

# --- CONFIG ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
INPUT_FOLDER = "./materials/articles"
OUTPUT_FOLDER = "./data/nx_jsons"
# Using a model strong at reasoning for medical data
MODEL = "google/gemini-2.0-flash-001" 

def get_medical_graph_from_llm(text, filename):
    prompt = f"""
    Analyze this medical paper: {filename}.
    Extract a Knowledge Graph in NetworkX 'node-link' JSON format.
    
    NODES should be:
    - Scoring Systems (e.g., p-SOFA, PRISM III)
    - Clinical Outcomes (e.g., 30-day mortality)
    - Demographics (e.g., Median Age)
    - Statistical Metrics (e.g., AUC, Sensitivity, Specificity)

    EDGES should represent:
    - Predictive relationships (Predicts, Correlates with)
    - Statistical findings (Resulted in AUC of X, p-value of Y)

    Return ONLY JSON:
    {{
      "directed": true,
      "multigraph": false,
      "nodes": [{{"id": "node_name", "type": "category", "value": "optional_value"}}],
      "links": [{{"source": "A", "target": "B", "relation": "label", "p_value": "numeric_if_any"}}]
    }}

    TEXT CONTENT:
    {text[:15000]}
    """

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": { "type": "json_object" }
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for file_name in os.listdir(INPUT_FOLDER):
        if file_name.endswith(".pdf"):
            print(f"🔬 Extracting clinical graph from: {file_name}")
            try:
                # 1. Extract Text
                doc = fitz.open(os.path.join(INPUT_FOLDER, file_name))
                text = " ".join([page.get_text() for page in doc])
                
                # 2. Get LLM Graph
                raw_json = get_medical_graph_from_llm(text, file_name)
                graph_json = json.loads(raw_json)
                
                # 3. Save individual NX file
                output_path = os.path.join(OUTPUT_FOLDER, file_name.replace(".pdf", "_nx.json"))
                if os.path.exists(output_path):
                    print(f"⏩ Skipping {file_name} (Already processed)")
                    continue
                else:
                    with open(output_path, "w") as f:
                        json.dump(graph_json, f, indent=2)
                
                print(f"✅ Created {output_path}")

            except Exception as e:
                print(f"❌ Error in {file_name}: {e}")

if __name__ == "__main__":
    main()