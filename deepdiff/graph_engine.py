"""
Application engine for graph reference mode deltas.

A graph patch is applied by *reconstructing* the target snapshot from the
patch topology rather than mutating the caller's objects.  This gives three
guarantees at once:

* the input graph and the :class:`~deepdiff.delta.Delta` are never mutated, on
  success or on failure (all verification runs on the untouched input, and the
  allocation of the result only starts after every check passed);
* the result shares no container with the input or the delta internals;
* aliasing and cycles inside the result reproduce the target topology exactly
  because each patch node id is materialized exactly once and every edge stores
  the id of its single materialization.

Old-value verification
----------------------

The patch does not keep a copy of the base snapshot.  Instead it stores compact
facts (see :func:`deepdiff.graph_model.graph_fingerprint` and
:func:`deepdiff.graph_model.node_fingerprint`):

* global facts - root kind, node/edge counts and the refined 1-WL color
  multiset - reject graphs with a different global topology role distribution,
  which is exactly how an equal-but-differently-aliased base graph is caught
  (copying a shared tail changes node/edge counts and color multiplicities);
* local facts - kind, color, edge count and the incoming-edge multiset of every
  node touched by an event - reject local divergence (an edited/moved edge, a
  changed alias) in O(changed nodes), without re-hashing user payloads.

Every failure raises :class:`GraphConflictError` carrying a public locator; a
partial result is never returned.
"""
from __future__ import annotations

from typing import Any, Dict

from deepdiff.graph_model import (
    DICT_TYPE, LIST_TYPE, TAG_NODE_REF, GraphConflictError, GraphPatchError,
    build_graph_index, node_fingerprint, graph_fingerprint,
    decode_scalar, canonical_signatures,
)
from deepdiff.graph_delta import GRAPH_PATCH_VERSION


class GraphDeltaEngine:
    """Applies one validated graph patch in one direction to one base graph."""

    def __init__(self, patch: Dict[str, Any], reverse: bool = False):
        self.patch = reverse_patch(patch) if reverse else patch
        self.reverse = reverse
        # Diagnostics counters (also used by the scale checks): how many nodes
        # and edges the verification/application stages touch.
        self.verified_nodes = 0
        self.verified_edges = 0
        self.materialized_nodes = 0
        self.materialized_edges = 0
        self._base_index = None
        self._t2_to_t1_map: Dict[int, int] = {}
        self._t1_node_map: Dict[int, dict] = {}
        self._t2_node_map: Dict[int, dict] = {}
        self._t1_topology_nodes: Dict[int, dict] = {}
        self._t2_topology_nodes: Dict[int, dict] = {}
        self._scalar_change_map: Dict[tuple, dict] = {}
        self._edge_added_map: Dict[tuple, dict] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def apply(self, root: Any) -> Any:
        validate_patch(self.patch)
        base_index = build_graph_index(root, side="base")
        self._base_index = base_index
        # Precompute lookup indexes once so per-edge resolution stays O(1).
        self._t2_to_t1_map = {b: a for a, b in self.patch['correspondence']['t1_to_t2']}
        self._t1_node_map = {entry['t1_node']: entry for entry in self.patch['list_alignments']}
        self._t2_node_map = {entry['t2_node']: entry for entry in self.patch['list_alignments']}
        # Node-id -> topology entry maps for both sides, built once.
        self._t1_topology_nodes = {n['id']: n for n in self.patch['t1_topology']['nodes']}
        self._t2_topology_nodes = {n['id']: n for n in self.patch['t2_topology']['nodes']}
        # O(1) event lookup tables so scalar resolution never scans the patch.
        self._scalar_change_map = {
            (entry['t2_node'], entry['t2_label']): entry
            for entry in self.patch['scalar_changes']}
        self._edge_added_map = {
            (entry['node'], entry['label']): entry
            for entry in self.patch['edge_added']}
        self._verify(base_index)
        return self._reconstruct()

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def _verify(self, base_index) -> None:
        patch = self.patch
        base_signatures = canonical_signatures(base_index)

        # ---- global facts -------------------------------------------------
        expected_global = patch['fingerprints']['t1']
        actual_global = graph_fingerprint(base_index, base_signatures)
        if base_index.node_count != expected_global['node_count'] \
                or base_index.edge_count != expected_global['edge_count']:
            raise GraphConflictError(
                'root',
                f"base graph has {base_index.node_count} nodes/{base_index.edge_count} edges but the "
                f"patch recorded {expected_global['node_count']} nodes/{expected_global['edge_count']} "
                f"edges; the base topology does not match the snapshot the delta was made from.")
        if actual_global['root_kind'] != expected_global['root_kind']:
            raise GraphConflictError(
                'root',
                f"root kind {actual_global['root_kind']!r} does not match the recorded "
                f"{expected_global['root_kind']!r}.")
        if actual_global['color_multiset'] != expected_global['color_multiset']:
            raise GraphConflictError(
                'root',
                "structural color multiset differs from the recorded base: the base graph has equal "
                "values but different sharing/topology (or different local structure).")
        self.verified_nodes += base_index.node_count
        self.verified_edges += base_index.edge_count

        # ---- topology shape vs patch t1 topology -------------------------
        if set(base_index.nodes) != set(self._t1_topology_nodes):
            raise GraphConflictError('root', 'base node id set differs from the recorded topology.')
        for nid, node in base_index.nodes.items():
            recorded = self._t1_topology_nodes[nid]
            if recorded['kind'] != node.kind:
                raise GraphConflictError(f"n{nid}", f"kind {node.kind!r} != recorded {recorded['kind']!r}.")
            if len(recorded['edges']) != len(node.edges):
                raise GraphConflictError(f"n{nid}", 'edge count does not match the recorded base.')
            for (label, target), (recorded_label, recorded_target) in zip(node.edges, recorded['edges']):
                if label != recorded_label:
                    raise GraphConflictError(f"n{nid}", f"edge label {label!r} != recorded {recorded_label!r}.")
                if target.is_node != ('node' in recorded_target):
                    raise GraphConflictError(f"n{nid}", 'edge target kind (node vs scalar) mismatch.')
                if target.is_node and target.node_id != recorded_target['node']:
                    raise GraphConflictError(
                        f"n{nid}", f"edge {label!r} points at n{target.node_id}, recorded "
                        f"n{recorded_target['node']}: an alias or cycle edge differs.")
                if not target.is_node:
                    recorded_kind = recorded_target.get('scalar')
                    if recorded_kind is None:
                        # Full tagged payload (newly created subgraph in the
                        # opposite topology never occurs here under t1, but stay
                        # strict rather than silently accepting it).
                        from deepdiff.graph_model import decode_scalar as _decode
                        recorded_kind = scalar_kind_of(_decode(recorded_target))
                    if scalar_kind_of(target.value) != recorded_kind:
                        raise GraphConflictError(
                            f"n{nid}", f"edge {label!r} scalar type {scalar_kind_of(target.value)!r} != "
                            f"recorded {recorded_kind!r}.")

        # ---- local fingerprints of touched nodes -------------------------
        recorded_nodes = patch['fingerprints']['t1_nodes']
        for nid_str, expected_fp in recorded_nodes.items():
            nid = int(nid_str)
            actual_fp = node_fingerprint(base_index, nid, base_signatures)
            self.verified_nodes += 1
            if actual_fp['kind'] != expected_fp['kind']:
                raise GraphConflictError(f"n{nid}", 'container kind mismatch.')
            if actual_fp['color'] != expected_fp['color']:
                raise GraphConflictError(
                    f"n{nid}", 'structural color mismatch: local topology changed since the delta was made.')
            if actual_fp['edge_count'] != expected_fp['edge_count']:
                raise GraphConflictError(f"n{nid}", 'edge count mismatch.')
            if actual_fp['incoming'] != [list(item) for item in expected_fp['incoming']]:
                raise GraphConflictError(
                    f"n{nid}", 'incoming-edge multiset mismatch: a sharing/alias relationship differs.')

        # ---- scalar old values of every changed edge ---------------------
        for entry in patch['scalar_changes']:
            self._check_scalar_edge(base_index, entry['t1_node'], entry['t1_label'],
                                    entry['old'], f"n{entry['t1_node']}")
        for entry in patch['edge_removed']:
            if 'value' in entry:
                self._check_scalar_edge(base_index, entry['node'], entry['label'],
                                        entry['value'], f"n{entry['node']}")

    @staticmethod
    def _check_scalar_edge(base_index, node_id, label, expected_tagged, location):
        node = base_index.nodes.get(node_id)
        if node is None:
            raise GraphConflictError(location, 'node does not exist in the base graph.')
        target = _edge_target(node, label, location)
        if target is None or target.is_node:
            raise GraphConflictError(location, f"edge {label!r} is not a scalar edge as recorded.")
        expected = decode_scalar(expected_tagged)
        if type(target.value) is not type(expected) or target.value != expected:
            raise GraphConflictError(
                location, f"old scalar value at {label!r} is {target.value!r}, expected {expected!r}.")

    # ------------------------------------------------------------------
    # Reconstruction of the target snapshot
    # ------------------------------------------------------------------

    def _reconstruct(self) -> Any:
        topology = self.patch['t2_topology']
        materialized: Dict[int, Any] = {}
        # Phase 1: allocate every container exactly once.  This is what makes
        # sharing and cycles expressible - edges only ever reference ids.
        for node_data in topology['nodes']:
            nid = node_data['id']
            materialized[nid] = {} if node_data['kind'] == DICT_TYPE else []
            self.materialized_nodes += 1
        # Phase 2: wire the edges.  Forward references and back edges resolve
        # because every id already has a materialization.
        for node_data in topology['nodes']:
            container = materialized[node_data['id']]
            for label, target_payload in node_data['edges']:
                if TAG_NODE_REF in target_payload:
                    value = materialized[target_payload[TAG_NODE_REF]]
                elif 'scalar' in target_payload:
                    # The compact topology only stores the scalar *kind*; the
                    # real value comes from an event payload or the base graph.
                    value = self._resolve_scalar(node_data['id'], label,
                                                 target_payload['scalar'])
                else:  # full tagged scalar (only present on event-bearing edges)
                    value = decode_scalar(target_payload)
                if node_data['kind'] == DICT_TYPE:
                    container[label] = value
                else:
                    container.append(value)
                self.materialized_edges += 1
        return materialized[topology['root']]

    def _resolve_scalar(self, node_id: int, label: Any, kind_hint: str) -> Any:
        # 1) explicit scalar change wins.
        change = self._scalar_change_map.get((node_id, label))
        if change is not None:
            return decode_scalar(change['new'])
        # 2) added edges carry their payload in the patch.
        added = self._edge_added_map.get((node_id, label))
        if added is not None and 'value' in added:
            return decode_scalar(added['value'])
        # 3) unchanged scalar: fetch from the base graph through the
        # correspondence and the recorded list alignment for shifted indexes.
        t1_node = self._t2_to_t1_map.get(node_id)
        if t1_node is not None:
            t1_label = self._t1_label_for(node_id, label)
            if t1_label is not None:
                node_data = self._t1_topology_nodes.get(t1_node)
                payload = _edge_payload(node_data, t1_label) if node_data else None
                if payload is not None and 'scalar' in payload:
                    return _base_scalar_value(self._base_index, t1_node, t1_label, kind_hint)
        raise GraphConflictError(
            f"n{node_id}", f"cannot resolve scalar edge {label!r}: the patch is missing its payload.")

    def _t1_label_for(self, t2_node: int, t2_label: Any):
        """Map a t2 edge label to its t1 label via the cached alignment."""
        node_data = self._t2_topology_nodes.get(t2_node)
        if node_data is None:
            return None
        if node_data['kind'] == DICT_TYPE:
            return t2_label
        alignment = self._t2_node_map.get(t2_node)
        if alignment is None:
            return None
        for t1_index, t2_index in alignment['rows']:
            if t2_index == t2_label:
                return t1_index
        return None
    # The base index is rebuilt lazily only when unchanged scalars must be
    # copied; apply() already built one, so hand it over to avoid a second walk.
    _base_index = None


# Module level helpers -------------------------------------------------------


def _base_scalar_value(base_index, node_id, label, kind_hint):
    node = base_index.nodes[node_id]
    target = _edge_target(node, label, f"n{node_id}")
    if target is None or target.is_node:
        raise GraphConflictError(f"n{node_id}", f"recorded scalar edge {label!r} not found in base.")
    if scalar_kind_of(target.value) != kind_hint:
        raise GraphConflictError(f"n{node_id}", f"scalar kind mismatch at {label!r}.")
    return target.value


def scalar_kind_of(value: Any) -> str:
    from deepdiff.graph_model import scalar_kind
    return scalar_kind(value)


def _edge_target(node, label, location):
    for edge_label, target in node.edges:
        if edge_label == label:
            return target
    return None


def _edge_payload(node_data, label):
    for edge_label, payload in node_data['edges']:
        if edge_label == label:
            return payload
    return None


def reverse_patch(patch: Dict[str, Any]) -> Dict[str, Any]:
    from deepdiff.graph_delta import reverse_graph_patch
    return reverse_graph_patch(patch)


# ---------------------------------------------------------------------------
# Strict patch validation (used both at Delta load and at apply time)
# ---------------------------------------------------------------------------


def validate_patch(patch: Any) -> None:
    """
    Validate the envelope and internal references of a graph patch.

    Rejects unknown versions, duplicate node definitions, dangling or illegal
    references, wrong node types and malformed scalar tags.  Never executes any
    path text and never falls back to the tree-based delta format.
    """
    if not isinstance(patch, dict):
        raise GraphPatchError("Graph patch must be a dictionary.")
    if patch.get('version') != GRAPH_PATCH_VERSION:
        raise GraphPatchError(
            f"Unknown graph patch version {patch.get('version')!r}; expected {GRAPH_PATCH_VERSION}.")
    if patch.get('mode') != 'graph':
        raise GraphPatchError("Graph patch mode marker is missing or invalid.")
    for required in ('t1_topology', 't2_topology', 'correspondence', 'scalar_changes',
                     'edge_added', 'edge_removed', 'alias_events', 'node_events',
                     'fingerprints'):
        if required not in patch:
            raise GraphPatchError(f"Graph patch is missing required section {required!r}.")
    if 'list_alignments' not in patch or not isinstance(patch['list_alignments'], list):
        raise GraphPatchError("Graph patch is missing the 'list_alignments' list.")

    t1_ids = _validate_topology(patch['t1_topology'], 't1_topology')
    t2_ids = _validate_topology(patch['t2_topology'], 't2_topology')

    correspondence = patch['correspondence']
    if not isinstance(correspondence, dict) or not isinstance(correspondence.get('t1_to_t2'), list):
        raise GraphPatchError("Graph patch correspondence must be a dict containing 't1_to_t2'.")
    seen_left, seen_right = set(), set()
    for pair in correspondence['t1_to_t2']:
        if not (isinstance(pair, list) and len(pair) == 2):
            raise GraphPatchError("Correspondence entries must be [t1_node, t2_node] pairs.")
        a, b = pair
        if a not in t1_ids or b not in t2_ids:
            raise GraphPatchError(f"Correspondence pair {pair!r} references an unknown node.")
        if a in seen_left or b in seen_right:
            raise GraphPatchError(f"Correspondence pair {pair!r} duplicates a node mapping.")
        seen_left.add(a)
        seen_right.add(b)

    _validate_event_lists(patch, t1_ids, t2_ids)
    _validate_fingerprints(patch)


def _validate_topology(topology: Any, name: str) -> set:
    if not isinstance(topology, dict):
        raise GraphPatchError(f"{name} must be a dictionary.")
    if not isinstance(topology.get('nodes'), list):
        raise GraphPatchError(f"{name}.nodes must be a list.")
    ids = set()
    for node in topology['nodes']:
        if not isinstance(node, dict):
            raise GraphPatchError(f"{name} nodes must be dictionaries.")
        nid = node.get('id')
        if isinstance(nid, bool) or not isinstance(nid, int) or nid < 0:
            raise GraphPatchError(f"{name} node ids must be non negative integers.")
        if nid in ids:
            raise GraphPatchError(f"{name} contains a duplicate node definition for n{nid}.")
        ids.add(nid)
        if node.get('kind') not in (DICT_TYPE, LIST_TYPE):
            raise GraphPatchError(f"n{nid} in {name} has an invalid node type {node.get('kind')!r}.")
        if not isinstance(node.get('edges'), list) or not isinstance(node.get('incoming'), list):
            raise GraphPatchError(f"n{nid} in {name} must carry edges and incoming lists.")
    # References are validated after every id is known so forward references
    # (legal in cyclic graphs) and back edges are treated the same.
    for node in topology['nodes']:
        nid = node['id']
        for label, target in node['edges']:
            if node['kind'] == DICT_TYPE and not isinstance(label, str):
                raise GraphPatchError(f"n{nid} dict edge labels must be strings.")
            if node['kind'] == LIST_TYPE and (isinstance(label, bool) or not isinstance(label, int) or label < 0):
                raise GraphPatchError(f"n{nid} list edge labels must be non negative integers.")
            _validate_target_ref(target, ids, name, nid)
        for source, label in node['incoming']:
            if source not in ids:
                raise GraphPatchError(f"n{nid} incoming edge in {name} dangles from unknown n{source}.")
    root = topology.get('root')
    if root not in ids:
        raise GraphPatchError(f"{name} root id {root!r} is not a defined node.")
    if not isinstance(topology.get('stats'), dict):
        raise GraphPatchError(f"{name} stats must be a dictionary.")
    return ids


def _validate_target_ref(target, ids, name, nid):
    if not isinstance(target, dict) or len(target) != 1:
        raise GraphPatchError(f"n{nid} edge target in {name} must be a single-key tagged object.")
    if 'node' in target:
        ref = target['node']
        if isinstance(ref, bool) or not isinstance(ref, int) or ref not in ids:
            raise GraphPatchError(f"n{nid} edge in {name} has an illegal/dangling node reference {ref!r}.")
    elif 'scalar' in target:
        if target['scalar'] not in ('int', 'float', 'str', 'bool', 'null'):
            raise GraphPatchError(f"n{nid} edge in {name} has an invalid scalar kind {target['scalar']!r}.")
    else:
        # Full tagged scalar payload (edges of newly created subgraphs).
        try:
            decode_scalar(target)
        except GraphPatchError as exc:
            raise GraphPatchError(f"n{nid} edge target in {name}: {exc}") from None


def _validate_event_lists(patch, t1_ids, t2_ids):
    for section in ('scalar_changes', 'edge_added', 'edge_removed', 'alias_events', 'node_events'):
        if not isinstance(patch[section], list):
            raise GraphPatchError(f"Graph patch section {section!r} must be a list.")
    for alignment in patch['list_alignments']:
        if alignment.get('t1_node') not in t1_ids or alignment.get('t2_node') not in t2_ids:
            raise GraphPatchError("list alignment references an unknown node.")
        if not isinstance(alignment.get('rows'), list):
            raise GraphPatchError("list alignment rows must be a list.")
        for row in alignment['rows']:
            if not (isinstance(row, list) and len(row) == 2):
                raise GraphPatchError("list alignment rows must be [t1_index, t2_index] pairs.")
            i, j = row
            if (i is not None and (isinstance(i, bool) or not isinstance(i, int) or i < 0)) or \
               (j is not None and (isinstance(j, bool) or not isinstance(j, int) or j < 0)):
                raise GraphPatchError("list alignment indexes must be non negative integers or null.")
    for entry in patch['scalar_changes']:
        for node_key, ids in (('t1_node', t1_ids), ('t2_node', t2_ids)):
            if entry.get(node_key) not in ids:
                raise GraphPatchError(f"scalar change references unknown {node_key} {entry.get(node_key)!r}.")
        decode_scalar(entry['old'])
        decode_scalar(entry['new'])
    for section, ids, node_key in (
        ('edge_added', t2_ids, 'node'), ('edge_removed', t1_ids, 'node'),
    ):
        for entry in patch[section]:
            if entry.get(node_key) not in ids:
                raise GraphPatchError(f"{section} references unknown node {entry.get(node_key)!r}.")
            if 'value' in entry:
                decode_scalar(entry['value'])
            if 'endpoint' in entry and entry['endpoint'] not in ids:
                raise GraphPatchError(f"{section} has a dangling endpoint reference.")
    for entry in patch['alias_events']:
        if entry.get('event') not in ('added', 'removed'):
            raise GraphPatchError("alias event must be 'added' or 'removed'.")
        if entry.get('t1_node') not in t1_ids or entry.get('t2_node') not in t2_ids:
            raise GraphPatchError("alias event references an unknown node.")
        for key, ids in (('t1_endpoint', t1_ids), ('t2_endpoint', t2_ids)):
            if entry.get(key) is not None and entry[key] not in ids:
                raise GraphPatchError(f"alias event {key} dangles.")
    for entry in patch['node_events']:
        if entry.get('event') == 'node_added' and entry.get('t2_node') not in t2_ids:
            raise GraphPatchError("node_added references an unknown t2 node.")
        if entry.get('event') == 'node_removed' and entry.get('t1_node') not in t1_ids:
            raise GraphPatchError("node_removed references an unknown t1 node.")


def _validate_fingerprints(patch):
    fps = patch['fingerprints']
    for side in ('t1', 't2'):
        if not isinstance(fps.get(side), dict):
            raise GraphPatchError(f"fingerprint {side!r} must be a dictionary.")
        for key in ('root_kind', 'node_count', 'edge_count', 'color_multiset'):
            if key not in fps[side]:
                raise GraphPatchError(f"fingerprint {side!r} is missing {key!r}.")
    for side in ('t1_nodes', 't2_nodes'):
        if not isinstance(fps.get(side), dict):
            raise GraphPatchError(f"fingerprint {side!r} must be a dictionary.")
        for nid_str, fp in fps[side].items():
            int(nid_str)  # raises ValueError on junk keys
            for key in ('kind', 'color', 'edge_count', 'incoming'):
                if key not in fp:
                    raise GraphPatchError(f"node fingerprint in {side} is missing {key!r}.")
