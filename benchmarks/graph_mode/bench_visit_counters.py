import sys
sys.path.insert(0, '/workspace')
from deepdiff import DeepDiff
from deepdiff.graph_model import build_graph_index
from deepdiff.graph_engine import GraphDeltaEngine

def make_chain(n, payload):
    nodes = [{'payload': payload, 'next': None, 'alt': None} for _ in range(n + 1)]
    for i in range(n):
        nodes[i]['next'] = nodes[i + 1]
        nodes[i]['alt'] = nodes[i + 1]
    return nodes[0]

p = 'x' * 1024
t1 = make_chain(10000, p); t2 = make_chain(10000, p)
node = t2
for i in range(5):
    node['payload'] = 'y' * 1024
    node['marker'] = i
    node = node['next']
d = DeepDiff(t1, t2, reference_mode='graph')
idx1 = build_graph_index(t1); idx2 = build_graph_index(t2)
print('t1 nodes/edges:', idx1.node_count, idx1.edge_count)
print('t2 nodes/edges:', idx2.node_count, idx2.edge_count)
engine = GraphDeltaEngine(d.graph_patch)
out = engine.apply(t1)
print('verify visits: nodes', engine.verified_nodes, 'edges', engine.verified_edges)
print('materialize: nodes', engine.materialized_nodes, 'edges', engine.materialized_edges)
assert engine.materialized_nodes == idx2.node_count
assert engine.materialized_edges == idx2.edge_count
print('LINEAR: each target node/edge materialized exactly once; no exponential path expansion')
