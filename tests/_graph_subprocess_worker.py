"""
Subprocess worker for graph reference mode round-trip tests.

Rebuilds the same cyclic/shared base and target graphs from scratch (the raw
graphs cannot be JSON-sent because they contain cycles) and applies deltas
restored from plain JSON text and from flat-dict rows.
"""
import json
import sys

sys.path.insert(0, __import__('os').path.dirname(
    __import__('os').path.dirname(__import__('os').path.abspath(__file__))))

from deepdiff import Delta
from deepdiff.serialization import json_loads
from deepdiff.graph_model import build_graph_index, graphs_equivalent


def main():
    payload = json.load(sys.stdin)
    shared = {'v': 1, 'w': [1, 2], 'name': 'keepme'}
    t1 = {'a': shared, 'b': shared, 'c': []}
    t1['c'].append(t1)
    target = {'v': 9, 'w': [1, 2, 0], 'name': 'keepme'}
    t2 = {'a': target, 'b': target, 'c': []}
    t2['c'].append(t2)

    delta_json = Delta(payload['dump'], deserializer=json_loads, bidirectional=True)
    delta_flat = Delta(flat_dict_list=payload['flat'], bidirectional=True)

    out_j = delta_json + t1
    back_j = t2 - delta_json
    out_f = delta_flat + t1
    back_f = t2 - delta_flat

    src_ids = {id(v) for v in t1.values()}
    disjoint = all(id(v) not in src_ids for v in out_j.values())

    result = {
        'json_forward': graphs_equivalent(build_graph_index(out_j), build_graph_index(t2)),
        'json_reverse': graphs_equivalent(build_graph_index(back_j), build_graph_index(t1)),
        'flat_forward': graphs_equivalent(build_graph_index(out_f), build_graph_index(t2)),
        'flat_reverse': graphs_equivalent(build_graph_index(back_f), build_graph_index(t1)),
        'disjoint_from_input': disjoint,
        'cycle_preserved': out_j['c'][0] is out_j,
        'alias_preserved': out_j['a'] is out_j['b'],
    }
    print(json.dumps(result))


if __name__ == '__main__':
    main()
