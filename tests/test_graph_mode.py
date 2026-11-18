"""
Tests for graph reference mode (``DeepDiff(..., reference_mode="graph")``).

The small graphs and their expected node/edge/value facts are declared
explicitly at the top of each test instead of being generated from the
implementation.  Topology checks use
:func:`deepdiff.graph_model.graphs_equivalent` and ``is`` assertions so
sharing and cycles are verified directly - plain ``==`` would not distinguish
an alias from a copied equal subtree.
"""
import copy
import json
import os
import subprocess
import sys

import pytest

from deepdiff import DeepDiff, Delta
from deepdiff.delta import DeltaError
from deepdiff.graph_model import (
    GraphError, UnsupportedGraphTypeError, GraphConflictError,
    GraphPatchError, build_graph_index, graphs_equivalent,
    canonical_signatures,
)
from deepdiff.serialization import json_dumps, json_loads


HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def topo_equal(left, right):
    return graphs_equivalent(build_graph_index(left), build_graph_index(right))


def assert_roundtrip(t1, t2):
    """Forward and reverse delta application with full topology checks."""
    diff = DeepDiff(t1, t2, reference_mode='graph')
    delta = Delta(diff, bidirectional=True)
    forward = delta + t1
    reverse = t2 - delta
    assert topo_equal(forward, t2)
    assert topo_equal(reverse, t1)
    # The result must not share any container with the inputs.
    _assert_disjoint(forward, t1)
    _assert_disjoint(reverse, t2)
    return diff, delta, forward, reverse


def _container_ids(root):
    index = build_graph_index(root)
    return set(index.nodes)  # ids are local; used only for disjointness below


def _assert_disjoint(built, source):
    source_containers = []
    index = build_graph_index(source)
    # Map physical python ids of source containers.
    src_py_ids = set(index.obj_to_id.keys())
    built_index = build_graph_index(built)
    # Every materialized container in built must be a different python object.
    seen = set()
    stack = [built]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        assert id(obj) not in src_py_ids, "result shares a container with the input"
        if isinstance(obj, dict):
            stack.extend(v for v in obj.values() if isinstance(v, (dict, list)))
        else:
            stack.extend(v for v in obj if isinstance(v, (dict, list)))


# ---------------------------------------------------------------------------
# Graph index basics: explicitly listed nodes / edges / values
# ---------------------------------------------------------------------------


class TestGraphIndex:

    def test_shared_vs_copied_node_counts(self):
        shared = {'v': 1}
        shared_graph = {'a': shared, 'b': shared}
        copied_graph = {'a': {'v': 1}, 'b': {'v': 1}}
        idx_shared = build_graph_index(shared_graph)
        idx_copied = build_graph_index(copied_graph)
        # Nodes: root + one shared child (2) vs root + two children (3).
        assert idx_shared.node_count == 2
        assert idx_copied.node_count == 3
        # Edges: root->child twice + child->scalar (3) vs four edges.
        assert idx_shared.edge_count == 3
        assert idx_copied.edge_count == 4
        assert not graphs_equivalent(idx_shared, idx_copied)

    def test_self_loop_vs_mutual_loop(self):
        self_loop = {}
        self_loop['self'] = self_loop
        a, b = {}, {}
        a['b'] = b
        b['a'] = a
        idx_self = build_graph_index(self_loop)
        idx_mutual = build_graph_index(a)
        assert idx_self.node_count == 1 and idx_self.edge_count == 1
        assert idx_mutual.node_count == 2 and idx_mutual.edge_count == 2
        assert not graphs_equivalent(idx_self, idx_mutual)
        # Isomorphic self loops allocated separately are equivalent.
        other = {}
        other['self'] = other
        assert graphs_equivalent(idx_self, build_graph_index(other))

    def test_dict_insertion_order_irrelevant(self):
        assert graphs_equivalent(
            build_graph_index({'a': 1, 'b': 2}),
            build_graph_index({'b': 2, 'a': 1}))

    def test_list_order_matters(self):
        assert not graphs_equivalent(
            build_graph_index([1, 2]),
            build_graph_index([2, 1]))

    def test_scalar_type_sensitivity(self):
        assert not graphs_equivalent(
            build_graph_index({'a': 1}), build_graph_index({'a': True}))
        assert not graphs_equivalent(
            build_graph_index({'a': 1}), build_graph_index({'a': 1.0}))

    def test_empty_containers(self):
        assert graphs_equivalent(build_graph_index({}), build_graph_index({}))
        assert not graphs_equivalent(build_graph_index([]), build_graph_index({}))


# ---------------------------------------------------------------------------
# Diff semantics on hand-built small graphs
# ---------------------------------------------------------------------------


class TestGraphDiffSemantics:

    def test_equal_graphs_produce_empty_diff(self):
        s1, s2 = {'k': 1}, {'k': 1}
        t1 = {'a': s1, 'b': s1}
        t2 = {'a': s2, 'b': s2}
        assert not DeepDiff(t1, t2, reference_mode='graph')

    def test_shared_vs_copied_is_a_diff(self):
        shared = {'v': 1}
        t1 = {'a': shared, 'b': shared}
        t2 = {'a': {'v': 1}, 'b': {'v': 1}}
        diff = DeepDiff(t1, t2, reference_mode='graph')
        assert diff
        assert_roundtrip(t1, t2)

    def test_shared_field_change_reported_once(self):
        shared = {'v': 1, 'w': 2}
        t1 = {'a': shared, 'b': shared}
        target = {'v': 9, 'w': 2}
        t2 = {'a': target, 'b': target}
        diff = DeepDiff(t1, t2, reference_mode='graph')
        changes = diff['graph_values_changed']
        assert len(changes) == 1  # not once per expanded path
        entry = changes[0]
        assert entry['old_value'] == 1 and entry['new_value'] == 9
        # The aliases of the shared node are discoverable from the entry.
        alias_nodes = {edge[0] for edge in entry['affected_aliases']}
        assert 0 in alias_nodes
        assert_roundtrip(t1, t2)

    def test_alias_merge_and_split(self):
        x = {'k': 1}
        merge_t1 = {'a': x, 'b': {'k': 1}}
        merge_t2 = {'a': x, 'b': x}
        diff = DeepDiff(merge_t1, merge_t2, reference_mode='graph')
        assert any(key in diff for key in
                   ('graph_alias_added', 'graph_dictionary_item_added',
                    'graph_node_added', 'graph_node_removed'))
        assert_roundtrip(merge_t1, merge_t2)
        # Split is the inverse direction.
        split_t1 = {'a': x, 'b': x}
        split_t2 = {'a': {'k': 1}, 'b': {'k': 1}}
        assert_roundtrip(split_t1, split_t2)

    def test_self_loop_and_mutual_loop_changes(self):
        loop = {}
        loop['self'] = loop
        a, b = {}, {}
        a['b'] = b
        b['a'] = a
        assert_roundtrip(loop, a)
        # Break a self loop into a plain field.
        plain = {'self': 1}
        assert_roundtrip(loop, plain)

    def test_root_self_loop(self):
        r1 = []
        r1.append(r1)
        r2 = []
        r2.append(r2)
        assert not DeepDiff(r1, r2, reference_mode='graph')
        r3 = []
        r3.append([1])
        r3[0].append(r3)
        assert_roundtrip(r1, r3)

    def test_list_insert_delete_reconnect(self):
        assert_roundtrip([1, 2, 3], [1, 9, 2, 3])
        assert_roundtrip([1, 2, 3], [1, 3])
        # Insertion keeps the surviving shared nodes aliased.
        node = {'v': 1}
        assert_roundtrip([node, node], [{'x': 0}, node, node])

    def test_container_scalar_swap(self):
        assert_roundtrip({'a': {'x': 1}}, {'a': 5})
        assert_roundtrip({'a': 5}, {'a': {'x': 1}})

    def test_empty_container_swap(self):
        assert_roundtrip({'a': {}}, {'a': []})

    def test_is_assertions_on_rebuilt_topology(self):
        shared = {'v': 1}
        t1 = {'a': shared, 'b': shared, 'c': []}
        t1['c'].append(t1)
        target = {'v': 2}
        t2 = {'a': target, 'b': target, 'c': []}
        t2['c'].append(t2)
        _, _, forward, _ = assert_roundtrip(t1, t2)
        # Sharing and the cycle survive as identity relationships.
        assert forward['a'] is forward['b']
        assert forward['c'][0] is forward
        # And the rebuilt objects are not the input objects.
        assert forward['a'] is not t1['a']
        assert forward is not t1


# ---------------------------------------------------------------------------
# Tree view / text view surface
# ---------------------------------------------------------------------------


class TestGraphReportSurface:

    def test_tree_view_exposes_graph_change_objects(self):
        t1 = {'a': [1, 2]}
        t2 = {'a': [1, 3]}
        tree = DeepDiff(t1, t2, reference_mode='graph', view='tree')
        # Same-type scalars at stable list positions are value changes.
        levels = tree['graph_values_changed']
        (level,) = levels
        assert level.path().startswith('root[n')
        assert level.path(use_t2=True).startswith('root[n')
        assert level.t1 == 2 and level.t2 == 3
        assert level.report_type == 'graph_values_changed'
        # A true insert still produces the iterable add/remove reports.
        insert_tree = DeepDiff([1, 2], [1, 9, 2], reference_mode='graph', view='tree')
        assert 'graph_iterable_item_added' in insert_tree

    def test_text_and_tree_views_agree_on_keys(self):
        t1 = {'a': 1, 'b': [1]}
        t2 = {'a': 2, 'b': [1, 2]}
        text = DeepDiff(t1, t2, reference_mode='graph')
        tree = DeepDiff(t1, t2, reference_mode='graph', view='tree')
        assert set(text.keys()) == set(tree.keys())

    def test_pretty_output(self):
        diff = DeepDiff({'a': 1}, {'a': 2}, reference_mode='graph', view='tree')
        text = diff.pretty()
        assert 'Graph scalar' in text


# ---------------------------------------------------------------------------
# Validation and parameter rejection
# ---------------------------------------------------------------------------


class TestGraphValidation:

    @pytest.mark.parametrize('bad', [
        float('nan'), float('inf'), float('-inf'),
        (1, 2), {1: 'x'}, object(), {1, 2}, b'bytes',
    ])
    def test_unsupported_types_rejected(self, bad):
        with pytest.raises((GraphError, UnsupportedGraphTypeError)):
            DeepDiff({'x': bad}, {'x': 1}, reference_mode='graph')

    def test_non_string_dict_key_rejected(self):
        with pytest.raises(UnsupportedGraphTypeError):
            DeepDiff({1: 'a'}, {'a': 1}, reference_mode='graph')

    def test_scalar_root_rejected(self):
        with pytest.raises(UnsupportedGraphTypeError):
            DeepDiff(5, 6, reference_mode='graph')

    def test_incompatible_parameter_combos(self):
        with pytest.raises(ValueError):
            DeepDiff({}, {}, reference_mode='graph', iterable_keys={'root': 'id'})
        with pytest.raises(ValueError):
            DeepDiff({}, {}, reference_mode='graph', ignore_order=True)
        with pytest.raises(ValueError):
            DeepDiff({}, {}, reference_mode='graph', group_by='x')
        with pytest.raises(ValueError):
            DeepDiff({}, {}, reference_mode='graph',
                     iterable_compare_func=lambda a, b, l: True)

    def test_semantic_changing_options_listed(self):
        with pytest.raises(ValueError) as exc:
            DeepDiff({}, {}, reference_mode='graph', significant_digits=2)
        assert 'significant_digits' in str(exc.value)

    def test_bad_reference_mode_value(self):
        with pytest.raises(ValueError):
            DeepDiff({}, {}, reference_mode='weird')

    def test_default_mode_unchanged(self):
        # No reference_mode: the classic tree engine and result keys apply.
        diff = DeepDiff({'a': 1}, {'a': 2})
        assert 'values_changed' in diff
        assert not any(k.startswith('graph_') for k in diff)


# ---------------------------------------------------------------------------
# Old-value verification and non-mutation guarantees
# ---------------------------------------------------------------------------


class TestGraphVerification:

    def test_equal_values_different_alias_rejected(self):
        shared = {'v': 1}
        t1 = {'a': shared, 'b': shared}
        t2 = {'a': {'v': 1}, 'b': {'v': 1}}
        delta = Delta(DeepDiff(t1, t2, reference_mode='graph'), bidirectional=True)
        wrong_base = {'a': {'v': 1}, 'b': {'v': 1}}  # copied, not shared
        with pytest.raises(GraphConflictError) as exc:
            delta + wrong_base
        assert exc.value.location

    def test_wrong_scalar_old_value_rejected(self):
        delta = Delta(
            DeepDiff({'a': 1}, {'a': 2}, reference_mode='graph'),
            bidirectional=True)
        with pytest.raises(GraphConflictError):
            delta + {'a': 99}

    def test_wrong_cycle_edge_rejected(self):
        loop = {}
        loop['self'] = loop
        # The patch expects a self loop; supply an acyclic graph instead.
        delta = Delta(
            DeepDiff(loop, {'self': 1}, reference_mode='graph'),
            bidirectional=True)
        with pytest.raises(GraphConflictError):
            delta + {'self': 1}

    def test_no_partial_result_and_input_unchanged_on_failure(self):
        shared = {'v': 1}
        t1 = {'a': shared, 'b': shared}
        t2 = {'a': {'v': 2}, 'b': {'v': 2}}
        delta = Delta(DeepDiff(t1, t2, reference_mode='graph'), bidirectional=True)
        wrong = {'a': {'v': 1}, 'b': {'v': 1}}
        snapshot = copy.deepcopy(wrong)
        with pytest.raises(GraphConflictError):
            delta + wrong
        assert wrong == snapshot

    def test_delta_reusable_after_failure(self):
        t1, t2 = {'a': 1}, {'a': 2}
        delta = Delta(DeepDiff(t1, t2, reference_mode='graph'), bidirectional=True)
        with pytest.raises(GraphConflictError):
            delta + {'a': 99}
        first = delta + t1
        second = delta + t1
        assert first == second and first is not second

    def test_bidirectional_required(self):
        with pytest.raises(ValueError):
            Delta(DeepDiff({'a': 1}, {'a': 2}, reference_mode='graph'))

    def test_apply_rebased_rejected(self):
        delta = Delta(
            DeepDiff({'a': 1}, {'a': 2}, reference_mode='graph'),
            bidirectional=True)
        with pytest.raises(DeltaError):
            delta.apply_rebased({'a': 1})


# ---------------------------------------------------------------------------
# Serialization: JSON and flat-dict, including cross-process
# ---------------------------------------------------------------------------


class TestGraphSerialization:

    def _cyclic_shared_case(self):
        shared = {'v': 1, 'w': [1, 2], 'name': 'keepme'}
        t1 = {'a': shared, 'b': shared, 'c': []}
        t1['c'].append(t1)
        target = {'v': 9, 'w': [1, 2, 0], 'name': 'keepme'}
        t2 = {'a': target, 'b': target, 'c': []}
        t2['c'].append(t2)
        return t1, t2

    def test_json_roundtrip_in_process(self):
        t1, t2 = self._cyclic_shared_case()
        delta = Delta(
            DeepDiff(t1, t2, reference_mode='graph'),
            bidirectional=True, serializer=json_dumps)
        dump = delta.dumps()
        assert isinstance(dump, str)
        restored = Delta(dump, deserializer=json_loads, bidirectional=True)
        assert topo_equal(restored + t1, t2)
        assert topo_equal(t2 - restored, t1)

    def test_flat_dict_roundtrip(self):
        t1, t2 = self._cyclic_shared_case()
        delta = Delta(DeepDiff(t1, t2, reference_mode='graph'), bidirectional=True)
        flat = delta.to_flat_dicts()
        # Flat rows are plain JSON-safe data.
        json.dumps(flat)
        restored = Delta(flat_dict_list=copy.deepcopy(flat), bidirectional=True)
        assert topo_equal(restored + t1, t2)
        assert topo_equal(t2 - restored, t1)

    def test_cross_process_roundtrip(self):
        t1, t2 = self._cyclic_shared_case()
        delta = Delta(
            DeepDiff(t1, t2, reference_mode='graph'),
            bidirectional=True, serializer=json_dumps)
        payload = json.dumps({'dump': delta.dumps(), 'flat': delta.to_flat_dicts()})
        worker = os.path.join(HERE, '_graph_subprocess_worker.py')
        env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(HERE)))
        proc = subprocess.run(
            [sys.executable, '-I', worker], input=payload, text=True,
            capture_output=True, env=env, check=True)
        result = json.loads(proc.stdout)
        assert result == {
            'json_forward': True, 'json_reverse': True,
            'flat_forward': True, 'flat_reverse': True,
            'disjoint_from_input': True,
            'cycle_preserved': True, 'alias_preserved': True,
        }

    @pytest.mark.parametrize('mutation,expected', [
        (lambda p: p['_graph_patch'].update(version=99), 'version'),
        (lambda p: p['_graph_patch']['t2_topology']['nodes'][0]['edges'][0]
            .__setitem__(1, {'node': 999}), 'dangling'),
        (lambda p: p['_graph_patch']['t1_topology']['nodes']
            .append(dict(p['_graph_patch']['t1_topology']['nodes'][0])), 'duplicate'),
        (lambda p: p['_graph_patch']['t1_topology']['nodes'][0].update(kind='set'),
            'node type'),
        (lambda p: p['_graph_patch'].pop('fingerprints'), 'missing section'),
    ])
    def test_corrupt_json_patch_rejected(self, mutation, expected):
        t1, t2 = {'a': [1, 2]}, {'a': [1, 3]}
        delta = Delta(
            DeepDiff(t1, t2, reference_mode='graph'),
            bidirectional=True, serializer=json_dumps)
        bad = json.loads(delta.dumps())
        mutation(bad)
        with pytest.raises((GraphPatchError, DeltaError)):
            Delta(json_dumps(bad), deserializer=json_loads, bidirectional=True)

    def test_corrupt_flat_patch_rejected(self):
        t1, t2 = {'a': 1}, {'a': 2}
        delta = Delta(DeepDiff(t1, t2, reference_mode='graph'), bidirectional=True)
        flat = copy.deepcopy(delta.to_flat_dicts())
        flat[0]['value']['version'] = 99
        with pytest.raises((GraphPatchError, DeltaError)):
            Delta(flat_dict_list=flat, bidirectional=True)

    def test_pickle_roundtrip(self):
        t1, t2 = self._cyclic_shared_case()
        delta = Delta(DeepDiff(t1, t2, reference_mode='graph'), bidirectional=True)
        restored = Delta(delta.dumps(), bidirectional=True)
        assert topo_equal(restored + t1, t2)
        assert topo_equal(t2 - restored, t1)

    def test_to_json_and_flat_rows(self):
        t1, t2 = {'a': [1, 2]}, {'a': [1, 3]}
        diff = DeepDiff(t1, t2, reference_mode='graph')
        # to_json() must not drop the graph-specific report keys.
        assert 'graph_values_changed' in json.loads(diff.to_json())
        delta = Delta(diff, bidirectional=True)
        rows = delta.to_flat_rows()
        restored = Delta(flat_rows_list=rows, bidirectional=True)
        assert topo_equal(restored + t1, t2)

    def test_equality_independent_of_addresses_and_allocation_order(self):
        def first():
            a = {'k': 1}
            b = {'k': 2}
            return {'a': a, 'b': b, 'c': [a, b]}

        def second():
            # Reverse allocation order and reverse field insertion order.
            b = {'k': 2}
            a = {'k': 1}
            return {'c': [a, b], 'b': b, 'a': a}

        assert not DeepDiff(first(), second(), reference_mode='graph')

    def test_corrupt_pickle_patch_rejected(self):
        t1, t2 = {'a': 1}, {'a': 2}
        delta = Delta(DeepDiff(t1, t2, reference_mode='graph'), bidirectional=True)
        # The envelope is validated at construction even for pickle input.
        class Bad(dict):
            pass
        corrupt = dict(delta.diff)
        corrupt['_graph_patch'] = dict(corrupt['_graph_patch'], version=42)
        with pytest.raises(GraphPatchError):
            Delta(corrupt, bidirectional=True)

    def test_patch_does_not_copy_unchanged_payload(self):
        payload = 'x' * 1024
        untouched = 'u' * 1024
        shared = {'payload': payload, 'untouched': untouched, 'next': None}
        t1 = {'a': shared, 'b': shared}
        target = {'payload': 'y' * 1024, 'untouched': untouched, 'next': None}
        t2 = {'a': target, 'b': target}
        delta = Delta(DeepDiff(t1, t2, reference_mode='graph'), bidirectional=True)
        dump = json.dumps(delta._graph_patch)
        # An unchanged 1 KiB scalar is never embedded; the changed scalar's
        # old/new values are the only payloads the patch may carry.
        assert untouched not in dump
        assert 'y' * 1024 in dump


# ---------------------------------------------------------------------------
# Backwards compatibility: existing modes keep their behavior
# ---------------------------------------------------------------------------


class TestBackwardsCompatibility:

    def test_regular_delta_json_behavior(self):
        t1, t2 = [1, 2, 3, 5], [1, 2, 3, 4, 5]
        diff = DeepDiff(t1, t2)
        delta = Delta(diff, serializer=json_dumps)
        restored = Delta(delta.dumps(), deserializer=json_loads)
        assert restored + t1 == t2

    def test_iterable_keys_mode_unchanged(self):
        t1 = [{'id': 1, 'v': 2}]
        t2 = [{'id': 1, 'v': 3}]
        diff = DeepDiff(t1, t2, iterable_keys={'root': 'id'}, view='tree')
        delta = Delta(diff, bidirectional=True)
        assert delta + t1 == t2
        assert t2 - delta == t1
