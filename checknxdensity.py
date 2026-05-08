import networkx as nx
import json
from networkx.readwrite import json_graph

with open('./data/nx_jsons/Suttapanit_2022_nx.json') as f:
    d = json.load(f)
    G = json_graph.node_link_graph(d)
    
print(f"Nodes: {G.nodes()}")
print(f"Predictive Relations: {len(G.edges())}")