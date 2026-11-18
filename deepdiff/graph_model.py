"""
Graph reference mode (``DeepDiff(..., reference_mode="graph")``).

This module compares *object graphs* rather than acyclic trees.  Two containers
that share one child are distinguished from two separately allocated but equal
children, and cycles are first class data.  Everything in here is self
contained and depends only on the standard library so the delta engine and the
(de)serializers can reuse it without import cycles.

Graph model
-----------

Only ``dict`` and ``list`` instances are *nodes* (they have an identity).
JSON-ish scalars (``str``, ``int``, ``bool``, ``None`` and finite ``float``) are
*values* carried by edges; scalars never need object identity because the mode
does not distinguish separately allocated equal scalars.  Dictionary keys must
be strings.  Every edge therefore goes from a container node to either another
container node or a scalar value and carries a label:

* a dictionary field name (``str``);
* a list index (``int``).

Node correspondence across snapshots is never derived from process addresses
(``id()``) nor from a content hash alone.  See :func:`compute_correspondence`
for the deterministic structural algorithm and the module level notes in
:mod:`deepdiff.graph_delta` for the locality / matching / verification trade
offs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GraphError(ValueError):
    """Base class for all ``reference_mode="graph"`` validation errors."""


class UnsupportedGraphTypeError(GraphError):
    """Raised when a graph contains a type graph mode does not support."""


class GraphPatchError(GraphError):
    """Raised when a serialized graph patch is corrupt or has an unknown version."""


class GraphConflictError(GraphError):
    """
    Raised when a graph delta cannot be applied because the supplied base graph
    does not satisfy the facts the patch recorded about its base snapshot.

    ``location`` is a public, process independent locator (a patch node id, an
    edge label pair or ``"root"``).
    """

    def __init__(self, location: str, message: str):
        self.location = location
        super().__init__(f"Graph delta conflict at {location}: {message}")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DICT_TYPE = 'dict'
LIST_TYPE = 'list'

# How many 1-WL refinement rounds run before list LCS alignment and fingerprint
# checks.  Linear in nothing; a fixed small bound keeps cost at O(rounds*E) and
# the stored verification metadata compact while separating virtually all
# practical structures.  The remaining ambiguity (true automorphisms) is
# resolved deterministically by discovery order.
WL_ROUNDS = 8

# Scalar value tags used inside the JSON-safe patch representation.  The tags
# live inside the patch envelope, so user dictionaries can never be confused
# with control information.
TAG_INT = 'int'
TAG_FLOAT = 'float'
TAG_STR = 'str'
TAG_BOOL = 'bool'
TAG_NULL = 'null'
TAG_NODE_REF = 'node'

# ---------------------------------------------------------------------------
# Scalar handling
# ---------------------------------------------------------------------------


def is_supported_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def scalar_kind(value: Any) -> str:
    """Return the JSON-safe type tag of a supported scalar."""
    if value is None:
        return TAG_NULL
    if isinstance(value, bool):
        return TAG_BOOL
    if isinstance(value, int):
        return TAG_INT
    if isinstance(value, float):
        return TAG_FLOAT
    if isinstance(value, str):
        return TAG_STR
    raise UnsupportedGraphTypeError(f"Unsupported scalar type {type(value).__name__}.")


def scalar_equal(left: Any, right: Any) -> bool:
    """Type sensitive scalar equality (``1`` != ``True`` != ``1.0``)."""
    if type(left) is not type(right):
        return False
    return left == right


def encode_scalar(value: Any) -> Dict[str, Any]:
    """Encode a supported scalar as a tagged JSON-safe ``{tag: value}`` dict."""
    return {scalar_kind(value): value}


def decode_scalar(payload: Any) -> Any:
    """Inverse of :func:`encode_scalar` with strict validation."""
    if not isinstance(payload, dict) or len(payload) != 1:
        raise GraphPatchError("Scalar values must be single-key tagged objects.")
    (tag, value), = payload.items()
    if tag == TAG_NULL:
        if value is not None:
            raise GraphPatchError("The 'null' tag must carry a null payload.")
        return None
    if tag == TAG_BOOL:
        if not isinstance(value, bool):
            raise GraphPatchError("The 'bool' tag must carry a boolean payload.")
        return value
    if tag == TAG_INT:
        if isinstance(value, bool) or not isinstance(value, int):
            raise GraphPatchError("The 'int' tag must carry an integer payload.")
        return value
    if tag == TAG_FLOAT:
        if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value):
            raise GraphPatchError("The 'float' tag must carry a finite float payload.")
        return value
    if tag == TAG_STR:
        if not isinstance(value, str):
            raise GraphPatchError("The 'str' tag must carry a string payload.")
        return value
    raise GraphPatchError(f"Unknown scalar tag {tag!r}.")


def encode_target(target: "EdgeTarget") -> Dict[str, Any]:
    """Encode an edge target (node reference or scalar) into JSON-safe form."""
    if target.is_node:
        return {TAG_NODE_REF: target.node_id}
    return encode_scalar(target.value)


def decode_target(payload: Any) -> "EdgeTarget":
    if isinstance(payload, dict) and len(payload) == 1 and TAG_NODE_REF in payload:
        ref = payload[TAG_NODE_REF]
        if isinstance(ref, bool) or not isinstance(ref, int) or ref < 0:
            raise GraphPatchError("Node references must be non negative integers.")
        return EdgeTarget.node(ref)
    return EdgeTarget.scalar(decode_scalar(payload))


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeTarget:
    """Where an edge points: either a node id or an inline scalar value."""

    is_node: bool
    node_id: Optional[int] = None
    value: Any = None

    @staticmethod
    def node(node_id: int) -> "EdgeTarget":
        return EdgeTarget(True, node_id=node_id)

    @staticmethod
    def scalar(value: Any) -> "EdgeTarget":
        return EdgeTarget(False, value=value)

    def sort_key(self) -> Tuple[int, Any]:
        # Ordering used to make every emitted structure deterministic: nodes
        # first (by id), then scalars (by tag/value).
        if self.is_node:
            return (0, self.node_id)
        return (1, (scalar_kind(self.value), self.value))


@dataclass
class GraphNode:
    node_id: int
    kind: str                      # DICT_TYPE / LIST_TYPE
    # Ordered outgoing edges: ``(label, target)``.  Labels are str for dicts,
    # int for lists; dict edges are stored in sorted key order.
    edges: List[Tuple[Any, EdgeTarget]] = field(default_factory=list)
    # Incoming edges as ``(source_node_id, label)``; also kept sorted.  It is the
    # incoming-edge multiset that lets us describe sharing and locate every
    # alias of a node without enumerating exponentially many root paths.
    incoming: List[Tuple[int, Any]] = field(default_factory=list)


@dataclass
class GraphIndex:
    """The node/edge representation of one reachable object graph."""

    root_id: int
    nodes: Dict[int, GraphNode]
    # id() of the original python container -> node id, only valid while the
    # input objects stay alive and never serialized.
    obj_to_id: Dict[int, int]
    stats: Dict[str, int]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return self.stats['edges']


def build_graph_index(root: Any, side: str = "t1") -> GraphIndex:
    """
    Walk ``root`` once and build its node/edge index.

    The walk is iterative (explicit stack) so cyclic and deeply nested graphs
    never hit Python's recursion limit.  Re-encountering a container simply
    reuses its node id, which is exactly what makes shared subtrees and cycles
    representable without any expansion into a tree.
    """
    nodes: Dict[int, GraphNode] = {}
    obj_to_id: Dict[int, int] = {}
    edge_count = 0

    def assign_id(obj: Any) -> int:
        obj_id = id(obj)
        existing = obj_to_id.get(obj_id)
        if existing is not None:
            return existing
        node_id = len(nodes)
        kind = DICT_TYPE if isinstance(obj, dict) else LIST_TYPE
        obj_to_id[obj_id] = node_id
        nodes[node_id] = GraphNode(node_id=node_id, kind=kind)
        return node_id

    def validate_scalar(value: Any, path: str) -> None:
        if not is_supported_scalar(value):
            raise UnsupportedGraphTypeError(
                f"Graph reference mode supports dict, list, str, int, bool, None and finite float only; "
                f"got {type(value).__name__} on {side} at {path}.")

    _validate_root(root, side)
    root_id = assign_id(root)
    scheduled = {root_id}
    # The stack holds (python container, node id, path string).  Each physical
    # container is expanded exactly once: a re-encounter (shared child or a
    # back edge) only emits another edge reference.
    stack: List[Tuple[Any, int, str]] = [(root, root_id, "root")]
    while stack:
        obj, node_id, path = stack.pop()
        scheduled.discard(node_id)
        node = nodes[node_id]
        if node.edges:
            # Already expanded; keep the guard so cycles are obviously safe.
            continue
        if isinstance(obj, dict):
            items = sorted(obj.items(), key=lambda kv: kv[0])
            for key, child in items:
                if not isinstance(key, str):
                    raise UnsupportedGraphTypeError(
                        f"Graph reference mode only supports string dictionary keys; got {key!r} "
                        f"on {side} at {path}.")
                child_path = f"{path}[{key!r}]"
                target = _make_target(child, child_path, validate_scalar, assign_id, nodes, stack, scheduled)
                node.edges.append((key, target))
                edge_count += 1
                if target.is_node:
                    nodes[target.node_id].incoming.append((node_id, key))
        else:  # list
            for index, child in enumerate(obj):
                child_path = f"{path}[{index}]"
                target = _make_target(child, child_path, validate_scalar, assign_id, nodes, stack, scheduled)
                node.edges.append((index, target))
                edge_count += 1
                if target.is_node:
                    nodes[target.node_id].incoming.append((node_id, index))

    for node in nodes.values():
        node.incoming.sort()
    stats = {
        'nodes': len(nodes),
        'edges': edge_count,
        'dict_nodes': sum(1 for n in nodes.values() if n.kind == DICT_TYPE),
        'list_nodes': sum(1 for n in nodes.values() if n.kind == LIST_TYPE),
    }
    return GraphIndex(root_id=root_id, nodes=nodes, obj_to_id=obj_to_id, stats=stats)


def _validate_root(root: Any, side: str) -> None:
    if isinstance(root, (dict, list)):
        return
    if is_supported_scalar(root):
        raise UnsupportedGraphTypeError(
            f"The root object on {side} must be a dict or list in graph reference mode; "
            f"got {type(root).__name__}.")
    raise UnsupportedGraphTypeError(
        f"Graph reference mode supports dict, list, str, int, bool, None and finite float only; "
        f"got {type(root).__name__} on {side}.")


def _make_target(child, path, validate_scalar, assign_id, nodes, stack, scheduled):
    if isinstance(child, (dict, list)):
        child_id = assign_id(child)
        # A node is scheduled/expanded once: the first edge that discovers a
        # container schedules it, later edges (sharing or back edges) only
        # record the reference.
        if child_id not in scheduled and not nodes[child_id].edges:
            scheduled.add(child_id)
            stack.append((child, child_id, path))
        return EdgeTarget.node(child_id)
    validate_scalar(child, path)
    return EdgeTarget.scalar(child)


# ---------------------------------------------------------------------------
# Canonical structural signatures (cross-process 1-WL colors)
# ---------------------------------------------------------------------------


def canonical_signatures(index: GraphIndex, rounds: int = WL_ROUNDS,
                         include_values: bool = True) -> List[str]:
    """
    Compute *canonical* per-node color signatures: short content-addressed
    strings that describe the same structural role identically across graphs
    and processes.  These are the cross-process counterparts of the local
    integer color ids used internally by classic 1-WL refinement, which are
    only meaningful inside one graph.

    Each round folds the node kind, edge labels and the previous signatures of
    its neighbors into an interned nested key; an interned key is only ever
    reduced to a digest *once*, so a 100k-node graph performs a number of
    hashes proportional to the number of distinct structural roles rather than
    rounds * V.  Equal roles yield equal digests in every process because the
    interned keys contain only kinds, labels, neighbor role strings and scalar
    facts - never python addresses.

    ``include_values=False`` produces *structural-only* signatures: scalar
    edges contribute their type but never their value, so a node keeps the
    same signature when only a leaf scalar value changes.  Correspondence and
    list alignment use this variant so value edits are reported as edge-level
    value changes rather than wholesale subgraph remove/add.  Verification
    fingerprints use the value-sensitive default.
    """
    import hashlib

    intern: Dict[Any, str] = {}

    def digest_key(key: Any) -> str:
        cached = intern.get(key)
        if cached is not None:
            return cached
        text = repr(key)
        digest = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        intern[key] = digest
        return digest

    # Round 0 labels: node kind, edge count and direct scalar facts.
    signatures: List[str] = [''] * len(index.nodes)
    for nid in sorted(index.nodes):
        node = index.nodes[nid]
        scalar_labels = tuple(
            (label, scalar_kind(target.value)) +
            ((target.value,) if include_values else ())
            for label, target in node.edges if not target.is_node)
        signatures[nid] = digest_key((node.kind, len(node.edges), scalar_labels))

    for _ in range(rounds):
        updated: List[str] = [''] * len(index.nodes)
        for nid in sorted(index.nodes):
            node = index.nodes[nid]
            edge_key = tuple(
                (label, 'n', signatures[target.node_id]) if target.is_node
                else ((label, 's', scalar_kind(target.value)) +
                      ((target.value,) if include_values else ()))
                for label, target in node.edges)
            incoming_key = tuple(
                (label, signatures[source]) for source, label in node.incoming)
            updated[nid] = digest_key((signatures[nid], edge_key, incoming_key))
        signatures = updated
    return signatures


def structural_signatures(index: GraphIndex, rounds: int = WL_ROUNDS) -> List[str]:
    """Canonical signatures ignoring scalar values; used for node matching."""
    return canonical_signatures(index, rounds=rounds, include_values=False)


def canonical_color_multiset(signatures: List[str]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for sig in signatures:
        summary[sig] = summary.get(sig, 0) + 1
    return summary


# ---------------------------------------------------------------------------
# Deterministic node correspondence
# ---------------------------------------------------------------------------


@dataclass
class Correspondence:
    """
    The structural node matching between two snapshots.

    ``t1_to_t2`` / ``t2_to_t1`` are partial bijections of node ids; nodes only
    present on one side are simply absent from the mappings.
    """

    t1_to_t2: Dict[int, int]
    t2_to_t1: Dict[int, int]
    unmatched_t1: List[int]
    unmatched_t2: List[int]


def _lcs_alignment(seq1: List[Any], seq2: List[Any]) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    Standard LCS over *comparable edge symbols*, returning an ordered list of
    aligned index pairs and unaligned indexes.  List symbols are wrapped so
    scalar positions compare on full value (not just type): equal symbols are
    matched, a changed scalar becomes a delete+add instead of a false match.
    """
    m, n = len(seq1), len(seq2)
    table = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if seq1[i] == seq2[j]:
                table[i][j] = table[i + 1][j + 1] + 1
            else:
                table[i][j] = max(table[i + 1][j], table[i][j + 1])
    alignment: List[Tuple[Optional[int], Optional[int]]] = []
    i = j = 0
    while i < m and j < n:
        if seq1[i] == seq2[j]:
            alignment.append((i, j))
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            alignment.append((i, None))
            i += 1
        else:
            alignment.append((None, j))
            j += 1
    while i < m:
        alignment.append((i, None))
        i += 1
    while j < n:
        alignment.append((None, j))
        j += 1
    return alignment


def compute_correspondence(index1: GraphIndex, index2: GraphIndex,
                           sigs1: List[str], sigs2: List[str]) -> Correspondence:
    """
    Compute a deterministic structural correspondence between two graphs.

    ``sigs1`` / ``sigs2`` are the *canonical* per-node signatures from
    :func:`canonical_signatures`: identical structural roles produce identical
    strings across different graphs and processes, unlike per-graph local
    integer color ids which are only meaningful inside one graph.

    Algorithm
    ---------
    Rooted BFS from the two roots simultaneously.  When a matched container
    pair is expanded:

    * dictionary edges are joined on their field names; scalar children are
      compared directly and container children become matched node pairs;
    * list edges are aligned with an LCS over the canonical signatures, so
      inserts/deletes reconnect the surviving elements instead of producing
      positional noise;
    * when several same-signature candidates exist (a genuine automorphism,
      e.g. two identical records), the first still-unmatched candidate in
      discovery order wins.  Discovery order is derived from labels and list
      positions, never from memory addresses, so the choice is reproducible
      across processes and every automorphism produces the same canonical
      result.

    Nodes unreachable through matched edges (deleted/created subgraphs) are
    collected afterwards in BFS discovery order.  The algorithm is O(V + E +
    sum of LCS tables); it never expands a node twice and never enumerates the
    exponential set of root paths of a shared tail.
    """
    t1_to_t2: Dict[int, int] = {}
    t2_to_t1: Dict[int, int] = {}
    queue: List[Tuple[int, int]] = []
    # Roots only correspond when their container kind matches; a root type swap
    # is handled by the caller as full remove + add.
    if index1.nodes[index1.root_id].kind == index2.nodes[index2.root_id].kind:
        queue.append((index1.root_id, index2.root_id))
        t1_to_t2[index1.root_id] = index2.root_id
        t2_to_t1[index2.root_id] = index1.root_id
    head = 0
    while head < len(queue):
        nid1, nid2 = queue[head]
        head += 1
        node1 = index1.nodes[nid1]
        node2 = index2.nodes[nid2]
        if node1.kind != node2.kind:
            # Type-swapped containers cannot share child correspondence.
            continue
        if node1.kind == DICT_TYPE:
            edges1 = {label: target for label, target in node1.edges}
            edges2 = {label: target for label, target in node2.edges}
            for label in sorted(set(edges1) | set(edges2)):
                t1_target = edges1.get(label)
                t2_target = edges2.get(label)
                pair = _pair_targets(t1_target, t2_target, t1_to_t2, t2_to_t1,
                                     sigs1, sigs2, index1, index2)
                if pair is not None:
                    queue.append(pair)
        else:
            seq1 = [_list_color(target, sigs1) for _, target in node1.edges]
            seq2 = [_list_color(target, sigs2) for _, target in node2.edges]
            alignment = _lcs_alignment(seq1, seq2)
            edges1 = node1.edges
            edges2 = node2.edges
            for i, j in alignment:
                t1_target = edges1[i][1] if i is not None else None
                t2_target = edges2[j][1] if j is not None else None
                pair = _pair_targets(t1_target, t2_target, t1_to_t2, t2_to_t1,
                                     sigs1, sigs2, index1, index2)
                if pair is not None:
                    queue.append(pair)

    unmatched_t1 = [nid for nid in sorted(index1.nodes) if nid not in t1_to_t2]
    unmatched_t2 = [nid for nid in sorted(index2.nodes) if nid not in t2_to_t1]
    return Correspondence(t1_to_t2, t2_to_t1, unmatched_t1, unmatched_t2)


def _list_color(target: EdgeTarget, signatures: List[str]) -> Tuple[str, Any]:
    if target.is_node:
        return ('n', signatures[target.node_id])
    # Only the scalar *kind* participates in list alignment: a changed value
    # at a stable position becomes a value change rather than delete+add.
    return ('s', scalar_kind(target.value))


def _pair_targets(t1_target: Optional[EdgeTarget], t2_target: Optional[EdgeTarget],
                  t1_to_t2: Dict[int, int], t2_to_t1: Dict[int, int],
                  sigs1: List[str], sigs2: List[str],
                  index1: "GraphIndex", index2: "GraphIndex") -> Optional[Tuple[int, int]]:
    """Register one matched edge-target pair; return a node pair to enqueue."""
    if t1_target is None or t2_target is None:
        return None
    if t1_target.is_node != t2_target.is_node:
        return None
    if not t1_target.is_node:
        # Scalars are compared at the edge-diff stage; they carry no identity.
        return None
    c1, c2 = t1_target.node_id, t2_target.node_id
    if c1 in t1_to_t2 or c2 in t2_to_t1:
        # Already joined (possibly via another alias edge): keep the existing
        # correspondence; this is how an alias merge/split becomes visible.
        return None
    if index1.nodes[c1].kind != index2.nodes[c2].kind:
        # Different container types cannot be the same structural node.
        return None
    # Nodes reached through the same labeled edge from a matched parent are
    # paired positionally; signatures are reserved for list LCS tie breaking
    # and for verification.  Pairing even when the local signatures differ is
    # what lets an edited node (added/removed edge, changed scalar) stay the
    # *same* node instead of cascading into a whole-subgraph remove/add.
    t1_to_t2[c1] = c2
    t2_to_t1[c2] = c1
    return (c1, c2)


# ---------------------------------------------------------------------------
# Graph equality (topological equivalence)
# ---------------------------------------------------------------------------


def graphs_equivalent(index1: GraphIndex, index2: GraphIndex) -> bool:
    """
    Return True when the two graphs are topologically equivalent:

    * same root container type;
    * same nodes, edges, labels and list order up to isomorphism;
    * same scalar types and values;
    * same sharing and cycle structure (aliases and back edges included);
    * dictionary insertion order is irrelevant (fields are matched by name).
    """
    if index1.node_count != index2.node_count or index1.edge_count != index2.edge_count:
        return False
    sigs1 = canonical_signatures(index1)
    sigs2 = canonical_signatures(index2)
    correspondence = compute_correspondence(index1, index2, sigs1, sigs2)
    if len(correspondence.t1_to_t2) != index1.node_count:
        return False
    return _check_edges_under_correspondence(index1, index2, correspondence)


def _check_edges_under_correspondence(index1: GraphIndex, index2: GraphIndex,
                                      correspondence: Correspondence) -> bool:
    mapping = correspondence.t1_to_t2
    for nid1, node1 in index1.nodes.items():
        node2 = index2.nodes.get(mapping.get(nid1))  # type: ignore[arg-type]
        if node2 is None or node1.kind != node2.kind or len(node1.edges) != len(node2.edges):
            return False
        for (label1, target1), (label2, target2) in zip(node1.edges, node2.edges):
            if label1 != label2:
                return False
            if target1.is_node != target2.is_node:
                return False
            if target1.is_node:
                if mapping[target1.node_id] != target2.node_id:
                    return False
            elif not scalar_equal(target1.value, target2.value):
                return False
    return True


# ---------------------------------------------------------------------------
# Fingerprints (compact verification metadata, no full base snapshot)
# ---------------------------------------------------------------------------


def node_fingerprint(index: GraphIndex, node_id: int, signatures: List[str]) -> Dict[str, Any]:
    """
    Compact per-node facts recorded by a patch for old-value verification:

    * the canonical structural signature (cross-process topology role);
    * container kind and outgoing edge count;
    * the full incoming-edge multiset, expressed through node signatures and
      labels, which captures sharing/alias facts without storing payloads.
    """
    node = index.nodes[node_id]
    incoming = []
    for source_id, label in node.incoming:
        incoming.append([signatures[source_id], label])
    incoming.sort(key=lambda item: repr(item))
    return {
        'kind': node.kind,
        'color': signatures[node_id],
        'edge_count': len(node.edges),
        'incoming': incoming,
    }


def graph_fingerprint(index: GraphIndex, signatures: List[str]) -> Dict[str, Any]:
    """Whole-graph facts used to reject globally incompatible base graphs."""
    return {
        'root_kind': index.nodes[index.root_id].kind,
        'node_count': index.node_count,
        'edge_count': index.edge_count,
        'color_multiset': canonical_color_multiset(signatures),
    }
