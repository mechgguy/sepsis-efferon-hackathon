import glob
import networkx as nx
from networkx.readwrite import json_graph

all_graphs = []

# 1. Loop through your converted folder
for file_path in glob.glob("networkx_output/*.json"):
    with open(file_path, 'r') as f:
        data = json.load(f)
        # Convert JSON back to a NetworkX Graph object
        G = json_graph.node_link_graph(data)
        all_graphs.append(G)

# 2. Merge all 30 graphs into one
# This combines nodes with the same ID into a single point
combined_G = nx.compose_all(all_graphs)

# 3. Save the master graph
combined_data = json_graph.node_link_data(combined_G)
with open("global_knowledge_graph.json", "w") as f:
    json.dump(combined_data, f)

print(f"Merged {len(all_graphs)} papers into one graph with {combined_G.number_of_nodes()} nodes.")