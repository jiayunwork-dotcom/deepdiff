import sys, time, resource, json
sys.path.insert(0, '/workspace')
from deepdiff import DeepDiff, Delta
from deepdiff.graph_model import build_graph_index, graphs_equivalent

def make_chain(n, payload):
    nodes = [{'payload': payload, 'next': None, 'alt': None} for _ in range(n + 1)]
    for i in range(n):
        nodes[i]['next'] = nodes[i + 1]
        nodes[i]['alt'] = nodes[i + 1]
    return nodes[0]

def make_backedge(n, payload):
    root = make_chain(n, payload)
    tail = root
    while tail['next'] is not None:
        tail = tail['next']
    tail['back'] = root
    return root

def change(root, n_scalar, n_edge, psize):
    node, i = root, 0
    while node is not None and i < n_scalar:
        node['payload'] = 'y' * psize; i += 1; node = node['next']
    node, i = root, 0
    while node is not None and i < n_edge:
        node['marker'] = i; i += 1; node = node['next']

def run(builder, n, psize):
    p = 'x' * psize
    t1 = builder(n, p); t2 = builder(n, p)
    change(t2, 5, 5, psize)
    idx = build_graph_index(t1)
    t0 = time.perf_counter(); d = DeepDiff(t1, t2, reference_mode='graph'); td = time.perf_counter()-t0
    t0 = time.perf_counter(); delta = Delta(d, bidirectional=True); tb = time.perf_counter()-t0
    t0 = time.perf_counter(); out = delta + t1; ta = time.perf_counter()-t0
    t0 = time.perf_counter(); back = t2 - delta; tr = time.perf_counter()-t0
    assert graphs_equivalent(build_graph_index(out), build_graph_index(t2))
    assert graphs_equivalent(build_graph_index(back), build_graph_index(t1))
    dump = json.dumps(d.graph_patch)
    return dict(nodes=idx.node_count, edges=idx.edge_count,
        diff_s=round(td,3), delta_s=round(tb,3), apply_s=round(ta,3), reverse_s=round(tr,3),
        patch_bytes=len(dump),
        rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,1),
        untouched_1kib_copied=('u'*1024) in dump,
        reports={k: len(v) for k, v in d.graph_result.changes.items() if v})

results = []
for n in (10000, 100000):
    for psize in (1024, 8192):
        for kind, builder in (('chain', make_chain), ('backedge', make_backedge)):
            r = run(builder, n, psize); r.update(case=kind, n_target=n, psize=psize)
            results.append(r)
            print(json.dumps(r), flush=True)
with open('/tmp/graph_scripts/bench_results.json', 'w') as f:
    json.dump(results, f, indent=1)
print('DONE')
