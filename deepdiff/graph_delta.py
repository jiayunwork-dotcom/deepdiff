"""
Graph reference mode: difference engine and patch construction.

The engine works exclusively on the node/edge indexes from
:mod:`deepdiff.graph_model`, never on the expanded object tree.  A shared tail
with exponentially many root paths is therefore compared once.

Reporting model
---------------

All changes are expressed against stable patch node ids (the deterministic BFS
discovery order of each snapshot) rather than process addresses.  The report
keys are:

* ``graph_values_changed``          - a scalar edge changed value/type;
* ``graph_dictionary_item_added`` / ``graph_dictionary_item_removed``;
* ``graph_iterable_item_added`` / ``graph_iterable_item_removed``
  (list insert/delete, including reconnect after an insert);
* ``graph_alias_added`` / ``graph_alias_removed``
  (an edge started/stopped pointing at an existing node: alias merges/splits
  and cycle edge changes);
* ``graph_node_added`` / ``graph_node_removed``
  (whole container subgraphs created/deleted).

Every tree-view entry is a :class:`GraphChange` exposing public ``path()``,
``node_id`` / ``other_node_id`` and endpoint relationship attributes.  A change
to a shared node is reported exactly once; its ``affected_aliases`` lists every
incoming edge of the node so the impacted sharing relationships are discoverable
without re-walking the graph.

Patch locality
--------------

The serialized patch carries the full compact graph topology of both snapshots
(ids, kinds, labels, node references) but only the scalar payloads that actually
participate in a change.  Topology metadata is O(V + E) small integers and
labels - never the 1 KiB/8 KiB unchanged payloads - and the whole before/after
user graphs are never embedded.  See :mod:`deepdiff.graph_engine` for how the
metadata drives application and old-value verification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from deepdiff.graph_model import (
    DICT_TYPE, LIST_TYPE, GraphIndex,
    Correspondence,
    build_graph_index, compute_correspondence,
    scalar_equal, scalar_kind, encode_scalar,
    node_fingerprint, graph_fingerprint, canonical_signatures,
    structural_signatures,
)

GRAPH_PATCH_VERSION = 1
GRAPH_PATCH_KEY = '_graph_patch'

# Report keys (also registered in deepdiff.model.REPORT_KEYS).
GRAPH_VALUES_CHANGED = 'graph_values_changed'
GRAPH_DICT_ADDED = 'graph_dictionary_item_added'
GRAPH_DICT_REMOVED = 'graph_dictionary_item_removed'
GRAPH_LIST_ADDED = 'graph_iterable_item_added'
GRAPH_LIST_REMOVED = 'graph_iterable_item_removed'
GRAPH_ALIAS_ADDED = 'graph_alias_added'
GRAPH_ALIAS_REMOVED = 'graph_alias_removed'
GRAPH_NODE_ADDED = 'graph_node_added'
GRAPH_NODE_REMOVED = 'graph_node_removed'

GRAPH_REPORT_KEYS = (
    GRAPH_VALUES_CHANGED,
    GRAPH_DICT_ADDED, GRAPH_DICT_REMOVED,
    GRAPH_LIST_ADDED, GRAPH_LIST_REMOVED,
    GRAPH_ALIAS_ADDED, GRAPH_ALIAS_REMOVED,
    GRAPH_NODE_ADDED, GRAPH_NODE_REMOVED,
)


# ---------------------------------------------------------------------------
# Public tree-view change object
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class GraphChange:
    """
    A single graph-mode change as exposed by the tree view.

    Attributes
    ----------
    report_type:
        One of the ``GRAPH_*`` report keys.
    path_t1 / path_t2:
        Public, process independent locators of the edge (or node) on each side,
        rendered from node ids and labels, e.g. ``root[n3]['name']``.
    node_id / other_node_id:
        Stable patch node ids of the endpoint container(s); ``-1`` marks a
        side where the endpoint does not exist.
    label_t1 / label_t2:
        Edge labels (field name or list index) on each side; ``None`` when the
        edge does not exist on that side.
    t1 / t2:
        Scalar payloads for value changes and node add/removes; ``notpresent``
        style ``None`` is *not* used - check ``has_t1`` / ``has_t2`` instead so
        a genuine ``None`` value stays distinguishable.
    endpoint_t1 / endpoint_t2:
        For alias changes, the node id the edge points at on each side; this is
        the public endpoint relationship requested for sharing/cycle changes.
    affected_aliases:
        Incoming edges ``[(node_id, label), ...]`` of the shared node involved,
        so a single reported change still reveals every alias it influences.
    """

    report_type: str
    path_t1: str
    path_t2: str
    node_id: int = -1
    other_node_id: int = -1
    label_t1: Optional[Any] = None
    label_t2: Optional[Any] = None
    t1: Any = None
    t2: Any = None
    has_t1: bool = False
    has_t2: bool = False
    endpoint_t1: Optional[int] = None
    endpoint_t2: Optional[int] = None
    affected_aliases: List[Tuple[int, Any]] = field(default_factory=list)
    verbose_level: int = 1

    # DiffLevel-compatible surface used by the generic rendering machinery.
    @property
    def identity(self):  # pragma: no cover - compatibility only
        return None

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def path(self, root: str = "root", force: Optional[str] = None,
             use_t2: bool = False, **_kwargs) -> str:
        return self.path_t2 if use_t2 else self.path_t1

    def stable_path(self, root: str = "root") -> str:
        return self.path_t1

    def __repr__(self) -> str:
        return f"<GraphChange {self.report_type}: {self.path_t1} -> {self.path_t2}>"


# ---------------------------------------------------------------------------
# Path rendering from node ids
# ---------------------------------------------------------------------------


def edge_path(index: Optional[GraphIndex], node_id: int, label: Any) -> str:
    """Render ``root[n3]['name']`` style locators from ids and labels."""
    kind = index.nodes[node_id].kind if index is not None else LIST_TYPE
    if kind == DICT_TYPE:
        return f"root[n{node_id}][{label!r}]"
    return f"root[n{node_id}][{label}]"


def node_path(index: Optional[GraphIndex], node_id: int) -> str:
    return f"root[n{node_id}]"


# ---------------------------------------------------------------------------
# Difference engine
# ---------------------------------------------------------------------------


class GraphDiffResult:
    """Holds the tree-view changes and the derived JSON-safe patch."""

    def __init__(self):
        self.changes: Dict[str, List[GraphChange]] = {key: [] for key in GRAPH_REPORT_KEYS}
        self.patch: Optional[Dict[str, Any]] = None


def diff_graphs(t1: Any, t2: Any) -> GraphDiffResult:
    """Compare two supported object graphs and build the change list + patch."""
    index1 = build_graph_index(t1, side="t1")
    index2 = build_graph_index(t2, side="t2")
    # Value-sensitive signatures feed the verification fingerprints;
    # structural-only signatures drive node correspondence so scalar edits
    # stay edge-level changes instead of subgraph remove/add.
    verify_sigs1 = canonical_signatures(index1)
    verify_sigs2 = canonical_signatures(index2)
    signatures1 = structural_signatures(index1)
    signatures2 = structural_signatures(index2)

    result = GraphDiffResult()
    # A root container type swap cannot share any node correspondence: report
    # the whole roots as removed/added subgraphs.
    if index1.nodes[index1.root_id].kind != index2.nodes[index2.root_id].kind:
        correspondence = Correspondence(
            t1_to_t2={}, t2_to_t1={},
            unmatched_t1=sorted(index1.nodes), unmatched_t2=sorted(index2.nodes))
        _diff_subgraphs(index1, index2, correspondence, result)
    else:
        correspondence = compute_correspondence(
            index1, index2, signatures1, signatures2)
        _diff_edges(index1, index2, correspondence, signatures1, signatures2, result)
        _diff_subgraphs(index1, index2, correspondence, result)

    result.patch = build_graph_patch(index1, index2, correspondence,
                                     verify_sigs1, verify_sigs2, result)
    return result


def _diff_edges(index1: GraphIndex, index2: GraphIndex, correspondence,
                sigs1, sigs2, result: GraphDiffResult) -> None:
    """Compare the outgoing edges of every matched container pair."""
    for nid1, nid2 in correspondence.t1_to_t2.items():
        node1 = index1.nodes[nid1]
        node2 = index2.nodes[nid2]
        if node1.kind != node2.kind:
            # Container type swap: the whole node pair is handled as subgraph
            # remove/add by the caller through unmatched children.
            continue
        if node1.kind == DICT_TYPE:
            _diff_dict_edges(index1, index2, nid1, nid2, node1, node2,
                             correspondence, result)
        else:
            _diff_list_edges(index1, index2, nid1, nid2, node1, node2,
                             sigs1, sigs2, correspondence, result)


def _diff_dict_edges(index1, index2, nid1, nid2, node1, node2,
                     correspondence, result):
    edges1 = {label: target for label, target in node1.edges}
    edges2 = {label: target for label, target in node2.edges}
    for label in sorted(set(edges1) | set(edges2)):
        target1 = edges1.get(label)
        target2 = edges2.get(label)
        path1 = edge_path(index1, nid1, label)
        path2 = edge_path(index2, nid2, label)
        if target1 is None:
            _report_added_edge(index2, nid2, label, target2, path1, path2,
                               GRAPH_DICT_ADDED, correspondence, result)
        elif target2 is None:
            _report_removed_edge(index1, nid1, label, target1, path1, path2,
                                 GRAPH_DICT_REMOVED, result)
        else:
            _compare_targets(
                index1, index2, nid1, nid2, label, label,
                target1, target2, correspondence, result, path1, path2)


def _diff_list_edges(index1, index2, nid1, nid2, node1, node2,
                     sigs1, sigs2, correspondence, result):
    # Reuse the same LCS alignment the correspondence used, so reporting and
    # matching never disagree.
    from deepdiff.graph_model import _lcs_alignment, _list_color
    seq1 = [_list_color(target, sigs1) for _, target in node1.edges]
    seq2 = [_list_color(target, sigs2) for _, target in node2.edges]
    alignment = _lcs_alignment(seq1, seq2)
    for i, j in alignment:
        if i is None:
            label2, target2 = node2.edges[j]
            path2 = edge_path(index2, nid2, label2)
            path1 = edge_path(index1, nid1, label2)  # symmetric label space
            _report_added_edge(index2, nid2, label2, target2, path1, path2,
                               GRAPH_LIST_ADDED, correspondence, result)
        elif j is None:
            label1, target1 = node1.edges[i]
            path1 = edge_path(index1, nid1, label1)
            path2 = edge_path(index2, nid2, label1)
            _report_removed_edge(index1, nid1, label1, target1, path1, path2,
                                 GRAPH_LIST_REMOVED, result)
        else:
            label1, target1 = node1.edges[i]
            label2, target2 = node2.edges[j]
            path1 = edge_path(index1, nid1, label1)
            path2 = edge_path(index2, nid2, label2)
            _compare_targets(
                index1, index2, nid1, nid2, label1, label2,
                target1, target2, correspondence, result, path1, path2)


def _compare_targets(index1, index2, nid1, nid2, label1, label2,
                     target1, target2, correspondence, result, path1, path2):
    if target1.is_node != target2.is_node:
        # Container <-> scalar swap at an existing edge: expressed as remove of
        # the old edge plus add of the new edge (the node itself is reported by
        # the subgraph stage if it becomes unreachable).
        _report_removed_edge(index1, nid1, label1, target1, path1, path2,
                             GRAPH_DICT_REMOVED if index1.nodes[nid1].kind == DICT_TYPE
                             else GRAPH_LIST_REMOVED, result)
        _report_added_edge(index2, nid2, label2, target2, path1, path2,
                           GRAPH_DICT_ADDED if index2.nodes[nid2].kind == DICT_TYPE
                           else GRAPH_LIST_ADDED, correspondence, result)
        return
    if not target1.is_node:
        if not scalar_equal(target1.value, target2.value):
            _emit_value_change(index1, index2, nid1, nid2, label1, label2,
                               target1, target2, path1, path2, result)
        return
    # Both are node references.
    mapped = correspondence.t1_to_t2.get(target1.node_id)
    if mapped == target2.node_id:
        return  # same structural endpoint
    # The edge endpoint changed.  If the t2 endpoint is itself a matched node
    # (another t1 node maps to it), this is an alias relationship change.
    if target2.node_id in correspondence.t2_to_t1:
        _emit_alias_change(index1, index2, nid1, nid2, label1, label2,
                           target1, target2, path1, path2, result,
                           removed=False)
    else:
        # t2 endpoint is brand new: edge add.  The old edge disappears with its
        # node only if that node becomes unreachable (handled at subgraph stage).
        _report_added_edge(index2, nid2, label2, target2, path1, path2,
                           GRAPH_DICT_ADDED if index2.nodes[nid2].kind == DICT_TYPE
                           else GRAPH_LIST_ADDED, correspondence, result)


def _emit_value_change(index1, index2, nid1, nid2, label1, label2, target1, target2,
                       path1, path2, result):
    change = GraphChange(
        report_type=GRAPH_VALUES_CHANGED,
        path_t1=path1, path_t2=path2,
        node_id=nid1, other_node_id=nid2,
        label_t1=label1, label_t2=label2,
        t1=target1.value, t2=target2.value,
        has_t1=True, has_t2=True,
        # The incoming-edge multiset of the node carrying the scalar makes the
        # sharing relationships a single shared-node change influences visible.
        affected_aliases=sorted(set(index1.nodes[nid1].incoming)),
    )
    result.changes[GRAPH_VALUES_CHANGED].append(change)


def _emit_alias_change(index1, index2, nid1, nid2, label1, label2,
                       target1, target2, path1, path2, result, removed):
    # One semantic event per changed edge; alias merge (two edges converge on
    # one node) and alias split (one node fans out) are visible via
    # affected_aliases on both endpoints.
    aliases = sorted(set(index2.nodes[target2.node_id].incoming))
    change = GraphChange(
        report_type=GRAPH_ALIAS_ADDED if not removed else GRAPH_ALIAS_REMOVED,
        path_t1=path1, path_t2=path2,
        node_id=nid1, other_node_id=nid2,
        label_t1=label1, label_t2=label2,
        endpoint_t1=target1.node_id, endpoint_t2=target2.node_id,
        affected_aliases=aliases,
    )
    result.changes[change.report_type].append(change)


def _report_added_edge(index2, nid2, label, target, path1, path2,
                       report_type, correspondence, result):
    aliases = (sorted(set(index2.nodes[target.node_id].incoming))
               if target.is_node else [])
    change = GraphChange(
        report_type=report_type,
        path_t1=path1, path_t2=path2,
        node_id=-1, other_node_id=nid2,
        label_t1=None, label_t2=label,
        t2=target.value if not target.is_node else None,
        # Every added scalar edge carries its payload so the delta can create
        # it without a base counterpart.
        has_t1=False, has_t2=not target.is_node,
        endpoint_t2=target.node_id if target.is_node else None,
        affected_aliases=aliases,
    )
    result.changes[report_type].append(change)


def _report_removed_edge(index1, nid1, label, target, path1, path2,
                         report_type, result):
    aliases = (sorted(set(index1.nodes[target.node_id].incoming))
               if target.is_node else [])
    change = GraphChange(
        report_type=report_type,
        path_t1=path1, path_t2=path2,
        node_id=nid1, other_node_id=-1,
        label_t1=label, label_t2=None,
        t1=target.value if not target.is_node else None,
        has_t1=not target.is_node, has_t2=False,
        endpoint_t1=target.node_id if target.is_node else None,
        affected_aliases=aliases,
    )
    result.changes[report_type].append(change)


def _diff_subgraphs(index1, index2, correspondence, result):
    """Report whole container nodes that exist on one side only."""
    for nid in correspondence.unmatched_t2:
        node = index2.nodes[nid]
        change = GraphChange(
            report_type=GRAPH_NODE_ADDED,
            path_t1="<absent>", path_t2=node_path(index2, nid),
            node_id=-1, other_node_id=nid,
            endpoint_t2=nid,
            affected_aliases=sorted(set(node.incoming)),
        )
        result.changes[GRAPH_NODE_ADDED].append(change)
    for nid in correspondence.unmatched_t1:
        node = index1.nodes[nid]
        change = GraphChange(
            report_type=GRAPH_NODE_REMOVED,
            path_t1=node_path(index1, nid), path_t2="<absent>",
            node_id=nid, other_node_id=-1,
            endpoint_t1=nid,
            affected_aliases=sorted(set(node.incoming)),
        )
        result.changes[GRAPH_NODE_REMOVED].append(change)


# ---------------------------------------------------------------------------
# Patch construction
# ---------------------------------------------------------------------------


def _build_list_alignments(index1, index2, correspondence):
    """
    For every matched list-node pair record the LCS label correspondence as
    ``[t1_index_or_null, t2_index_or_null]`` rows.  This lets the application
    engine resolve unchanged scalar values across list insert/delete shifts
    without re-running the structural matcher.
    """
    from deepdiff.graph_model import _lcs_alignment, _list_color
    sigs1 = structural_signatures(index1)
    sigs2 = structural_signatures(index2)
    alignments = []
    for nid1, nid2 in sorted(correspondence.t1_to_t2.items()):
        node1 = index1.nodes[nid1]
        node2 = index2.nodes[nid2]
        if node1.kind != LIST_TYPE or node2.kind != LIST_TYPE:
            continue
        seq1 = [_list_color(target, sigs1) for _, target in node1.edges]
        seq2 = [_list_color(target, sigs2) for _, target in node2.edges]
        rows = []
        for i, j in _lcs_alignment(seq1, seq2):
            rows.append([i, j])
        alignments.append({'t1_node': nid1, 't2_node': nid2, 'rows': rows})
    return alignments


def _topology(index: GraphIndex, full_scalar_nodes: Optional[set] = None) -> Dict[str, Any]:
    """
    JSON-safe compact topology: kinds and labeled edges, no scalar payloads -
    except for edges under nodes in ``full_scalar_nodes`` (the created
    subgraph, which has no base counterpart to copy unchanged scalars from).
    """
    full_scalar_nodes = full_scalar_nodes or set()
    nodes = []
    for nid in sorted(index.nodes):
        node = index.nodes[nid]
        edges = []
        for label, target in node.edges:
            if target.is_node:
                edges.append([label, {'node': target.node_id}])
            elif nid in full_scalar_nodes:
                edges.append([label, encode_scalar(target.value)])
            else:
                edges.append([label, {'scalar': scalar_kind(target.value)}])
        nodes.append({
            'id': nid,
            'kind': node.kind,
            'edges': edges,
            'incoming': [[source, label] for source, label in node.incoming],
        })
    return {'root': index.root_id, 'nodes': nodes, 'stats': dict(index.stats)}


def build_graph_patch(index1, index2, correspondence, signatures1, signatures2,
                      result: GraphDiffResult) -> Dict[str, Any]:
    """
    Assemble the versioned, JSON-safe graph patch.

    It combines compact topology of both sides, the node correspondence, the
    changed scalar payloads, per-node fingerprints of the *changed* nodes plus a
    whole-graph fingerprint (for old-value verification), and the human-facing
    change list.

    Unchanged scalar payloads are never copied for nodes that exist on both
    sides.  The only full scalar payloads embedded in the patch are those inside
    a *created* subgraph (forward: t2-only nodes; reverse: t1-only nodes),
    because such a subgraph has no base counterpart to copy the values from.
    """
    changed_t1_nodes = {c.node_id for c in _all_changes(result) if c.node_id >= 0}
    changed_t2_nodes = {c.other_node_id for c in _all_changes(result) if c.other_node_id >= 0}
    fingerprints1 = {str(nid): node_fingerprint(index1, nid, signatures1)
                     for nid in changed_t1_nodes}
    fingerprints2 = {str(nid): node_fingerprint(index2, nid, signatures2)
                     for nid in changed_t2_nodes}

    created_t2 = set(correspondence.unmatched_t2)
    removed_t1 = set(correspondence.unmatched_t1)

    scalar_changes = []
    for change in result.changes[GRAPH_VALUES_CHANGED]:
        scalar_changes.append({
            't1_node': change.node_id, 't2_node': change.other_node_id,
            't1_label': change.label_t1, 't2_label': change.label_t2,
            'old': encode_scalar(change.t1), 'new': encode_scalar(change.t2),
        })

    edge_added, edge_removed = [], []
    for report_type, bucket, index_used, side in (
        (GRAPH_DICT_ADDED, edge_added, index2, 't2'),
        (GRAPH_LIST_ADDED, edge_added, index2, 't2'),
        (GRAPH_DICT_REMOVED, edge_removed, index1, 't1'),
        (GRAPH_LIST_REMOVED, edge_removed, index1, 't1'),
    ):
        for change in result.changes[report_type]:
            bucket.append(_edge_entry(change, index_used, side))

    alias_events = []
    for report_type in (GRAPH_ALIAS_ADDED, GRAPH_ALIAS_REMOVED):
        for change in result.changes[report_type]:
            alias_events.append({
                'event': 'added' if report_type == GRAPH_ALIAS_ADDED else 'removed',
                't1_node': change.node_id, 't2_node': change.other_node_id,
                't1_label': change.label_t1, 't2_label': change.label_t2,
                't1_endpoint': change.endpoint_t1,
                't2_endpoint': change.endpoint_t2,
            })

    node_events = []
    for change in result.changes[GRAPH_NODE_ADDED]:
        node_events.append({'event': 'node_added', 't2_node': change.other_node_id})
    for change in result.changes[GRAPH_NODE_REMOVED]:
        node_events.append({'event': 'node_removed', 't1_node': change.node_id})

    list_alignments = _build_list_alignments(index1, index2, correspondence)

    return {
        'version': GRAPH_PATCH_VERSION,
        'mode': 'graph',
        't1_topology': _topology(index1, full_scalar_nodes=removed_t1),
        't2_topology': _topology(index2, full_scalar_nodes=created_t2),
        'correspondence': {
            't1_to_t2': [[a, b] for a, b in sorted(correspondence.t1_to_t2.items())],
        },
        'list_alignments': list_alignments,
        'scalar_changes': scalar_changes,
        'edge_added': edge_added,
        'edge_removed': edge_removed,
        'alias_events': alias_events,
        'node_events': node_events,
        'fingerprints': {
            't1': graph_fingerprint(index1, signatures1),
            't2': graph_fingerprint(index2, signatures2),
            't1_nodes': fingerprints1,
            't2_nodes': fingerprints2,
        },
    }


def _edge_entry(change: GraphChange, index: GraphIndex, side: str) -> Dict[str, Any]:
    if side == 't2':
        node_id, label = change.other_node_id, change.label_t2
        path = change.path_t2
    else:
        node_id, label = change.node_id, change.label_t1
        path = change.path_t1
    entry = {'node': node_id, 'label': label, 'path': path}
    if change.has_t2 and side == 't2':
        entry['value'] = encode_scalar(change.t2)
    elif change.has_t1 and side == 't1':
        entry['value'] = encode_scalar(change.t1)
    if (side == 't2' and change.endpoint_t2 is not None) or \
       (side == 't1' and change.endpoint_t1 is not None):
        entry['endpoint'] = change.endpoint_t2 if side == 't2' else change.endpoint_t1
    return entry


def _all_changes(result: GraphDiffResult):
    for changes in result.changes.values():
        for change in changes:
            yield change


def _embed_full_scalars(topology: Dict[str, Any], full_scalar_nodes: set) -> Dict[str, Any]:
    """
    Mark a topology's scalar edges as full payloads.

    Used when reversing a patch: a node that was only removed in the forward
    direction becomes a created node in the reverse, so its scalar edges must
    carry values.  The values are recovered from the scalar change / edge event
    sections where present; removed subgraph scalar edges already carry full
    tagged payloads when the forward patch embedded them.
    """
    import copy
    cloned = copy.deepcopy(topology)
    for node in cloned['nodes']:
        if node['id'] not in full_scalar_nodes:
            continue
        for edge in node['edges']:
            target = edge[1]
            if isinstance(target, dict) and 'scalar' in target:
                # A bare kind marker can only appear when the value lives on an
                # unchanged matched edge; for an unmatched node this must not
                # happen, so flag it loudly rather than inventing a value.
                raise ValueError(
                    "Cannot reverse graph patch: node n{} is created in the reverse direction but its "
                    "scalar edge {!r} only carries a kind marker.".format(node['id'], edge[0]))
    return cloned


def reverse_graph_patch(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Return the inverse patch by swapping the two snapshot roles."""
    # Nodes that become "created" when roles swap: t1 nodes with no t2 mapping.
    mapped_t1 = {a for a, _ in patch['correspondence']['t1_to_t2']}
    t1_node_ids = {node['id'] for node in patch['t1_topology']['nodes']}
    created_after_swap = t1_node_ids - mapped_t1
    reversed_t2_topology = _embed_full_scalars(patch['t1_topology'], created_after_swap)
    reversed_patch = {
        'version': patch['version'],
        'mode': 'graph',
        't1_topology': patch['t2_topology'],
        't2_topology': reversed_t2_topology,
        'correspondence': {
            't1_to_t2': [[b, a] for a, b in patch['correspondence']['t1_to_t2']],
        },
        'list_alignments': [
            {'t1_node': entry['t2_node'], 't2_node': entry['t1_node'],
             'rows': [[j, i] for i, j in entry['rows']]}
            for entry in patch.get('list_alignments', [])
        ],
        'scalar_changes': [
            {
                't1_node': entry['t2_node'], 't2_node': entry['t1_node'],
                't1_label': entry['t2_label'], 't2_label': entry['t1_label'],
                'old': entry['new'], 'new': entry['old'],
            }
            for entry in patch['scalar_changes']
        ],
        'edge_added': patch['edge_removed'],
        'edge_removed': patch['edge_added'],
        'alias_events': [
            {
                'event': 'removed' if event['event'] == 'added' else 'added',
                't1_node': event['t2_node'], 't2_node': event['t1_node'],
                't1_label': event['t2_label'], 't2_label': event['t1_label'],
                't1_endpoint': event['t2_endpoint'],
                't2_endpoint': event['t1_endpoint'],
            }
            for event in patch['alias_events']
        ],
        'node_events': [
            ({'event': 'node_added', 't2_node': event['t1_node']}
             if event['event'] == 'node_removed'
             else {'event': 'node_removed', 't1_node': event['t2_node']})
            for event in patch['node_events']
        ],
        'fingerprints': {
            't1': patch['fingerprints']['t2'],
            't2': patch['fingerprints']['t1'],
            't1_nodes': patch['fingerprints']['t2_nodes'],
            't2_nodes': patch['fingerprints']['t1_nodes'],
        },
    }
    return reversed_patch
