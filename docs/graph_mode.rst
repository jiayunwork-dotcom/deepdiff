Graph reference mode
====================

.. note::

   Graph reference mode is available through
   ``DeepDiff(t1, t2, reference_mode="graph")``. Omitting the parameter keeps
   the classic acyclic-tree comparison behavior unchanged.

Motivation
----------

The regular DeepDiff engine treats its inputs as trees: when the same
container is reachable through several paths (a *shared* subtree) or through a
back edge (a *cycle*), it is visited once per path. Two fields that point at
the same dictionary are therefore indistinguishable from two fields that point
at separately allocated but equal dictionaries.

Graph reference mode instead models each input as a directed labeled graph:

* only ``dict`` and ``list`` instances are **nodes** - they have identity;
* ``str``, ``int``, ``bool``, ``None`` and finite ``float`` are **scalar
  values** carried by edges (separately allocated equal scalars are *not*
  distinguished);
* dictionary keys must be strings;
* every edge is labeled by a field name (``str``) or a list index (``int``).

Two graphs are reported equal only when they are *topologically equivalent*:
same root container type, same dictionary fields, same list order, same scalar
types and values, and the same sharing/cycle structure up to isomorphism.
Dictionary insertion order is irrelevant. For example, sharing one child
through two edges differs from carrying two equal copies; a one-node
self-loop differs from two mutually referring nodes; two separately allocated
but isomorphic graphs are equal.

.. code-block:: python

    from deepdiff import DeepDiff

    shared = {"v": 1}
    copied_a, copied_b = {"v": 1}, {"v": 1}
    DeepDiff({"a": shared, "b": shared},
             {"a": copied_a, "b": copied_b},
             reference_mode="graph")
    # reports an alias/topology difference; an ordinary == comparison would
    # call these equal

Node correspondence and symmetric graphs
-----------------------------------------

Node identity across snapshots is never derived from a process memory address
(``id()``) or from a content hash alone. The engine:

1. builds a node/edge index of each snapshot in a single iterative walk (a
   re-encountered container reuses its node id, so cycles and shared tails are
   never expanded into a tree);
2. runs bounded 1-dimensional Weisfeiler-Lehman refinement (8 rounds,
   O(rounds·(V+E))) to give each node a structural color describing its role;
3. performs a simultaneous rooted BFS: dictionary edges join on field names,
   list edges align through an LCS over the refined colors (so inserts and
   deletes reconnect surviving elements), and same-color automorphism
   candidates are joined deterministically in label/discovery order.

Because every tie-break uses labels and positions rather than addresses, the
result is reproducible across processes. When a graph is genuinely symmetric
(two interchangeable identical nodes), the first candidate in discovery order
wins; the choice is canonical and every automorphism yields the same patch.

Report keys
-----------

All changes are expressed against stable patch node ids (``n0``, ``n1`` ...)
rather than addresses. Both the text and tree views use the same keys:

==============================  ============================================
key                              meaning
==============================  ============================================
``graph_values_changed``         a scalar edge changed value or type
``graph_dictionary_item_added``  a dictionary field edge was created
``graph_dictionary_item_removed`` a dictionary field edge was removed
``graph_iterable_item_added``    a list position was inserted
``graph_iterable_item_removed``  a list position was deleted
``graph_alias_added``            an edge started pointing at an existing node
``graph_alias_removed``          an edge stopped pointing at an existing node
``graph_node_added``             a whole container subgraph was created
``graph_node_removed``           a whole container subgraph was deleted
==============================  ============================================

A change to a shared node is reported exactly once, never once per expanded
path. Every entry exposes the public edge locators on both sides
(``path_t1`` / ``path_t2``), the endpoint node ids for alias/cycle changes and,
where relevant, the full incoming-edge multiset (``affected_aliases``) so the
sharing relationships impacted by the change are discoverable without
re-walking the graph.

Deltas and reversibility
-------------------------

Graph deltas must be bidirectional and are applied by *reconstructing* the
target snapshot from the patch topology instead of mutating the input:

.. code-block:: python

    from deepdiff import Delta

    delta = Delta(DeepDiff(t1, t2, reference_mode="graph"), bidirectional=True)
    forward = delta + t1    # value- and topology-equivalent to t2
    reverse = t2 - delta    # value- and topology-equivalent to t1

Reconstruction allocates each patch node exactly once and wires edges by id,
which is how sharing and cycles are reproduced. On success or failure the
input graph and the delta are untouched, the result shares no container with
either, and the result's internal aliasing matches the target. A failed
application raises :class:`deepdiff.GraphConflictError` (with a public node/
edge locator) before any result object is allocated; partial results are never
returned. ``apply_rebased`` is explicitly rejected for graph deltas.

Old-value verification
----------------------

The patch never embeds the whole before/after graph. It stores compact
verification facts instead:

* global facts - root kind, node/edge counts and a canonical color multiset -
  reject globally incompatible bases (this is how an "equal values but
  different aliases" base graph is caught: copying a shared tail changes the
  node/edge counts and color multiplicities);
* local facts - kind, canonical signature, edge count and the incoming-edge
  multiset of every touched node - reject local divergence in O(changed
  nodes);
* exact old scalar values for every changed/removed scalar edge.

Serialization
-------------

The graph patch is plain JSON-safe data and travels through the existing
``Delta.dumps`` / loader machinery (use ``serializer=json_dumps`` and
``deserializer=json_loads``) and through ``to_flat_dicts()`` /
``flat_dict_list=...``. Every payload carries an explicit integer ``version``
and a ``mode: "graph"`` marker; control information lives under the
``_graph_patch`` envelope and tagged scalar objects, so user dictionaries can
never be confused with control data. Unknown versions, duplicate node
definitions, dangling or illegal references, wrong node types and malformed
scalars are rejected with :class:`deepdiff.GraphPatchError` - the loader never
executes path text and never silently falls back to a tree-based delta. Cycles
are legal data and round-trip unchanged.

Unchanged scalar payloads are not copied into a patch: topology metadata is
O(V+E) small integers and labels, and only changed scalars plus the payloads
of newly created subgraphs (which have no base counterpart) are embedded.

Unsupported inputs and parameters
---------------------------------

Graph mode raises :class:`deepdiff.UnsupportedGraphTypeError` on unsupported
types (including non-finite floats and non-string dictionary keys), never
skips them or recurses infinitely. It cannot be combined with
``iterable_keys``, ``ignore_order=True``, ``group_by`` or
``iterable_compare_func`` (``ValueError``), and every other option that would
change comparison semantics is rejected up front with the offending parameter
names listed. Only the ``text`` and ``tree`` views are available.
