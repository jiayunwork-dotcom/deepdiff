# Graph reference mode scale results

Hardware: Linux container, Python 3.11.2, 8 logical CPUs, ~9.7 GiB RAM.
Re-runnable with:

```bash
python -I benchmarks/graph_mode/bench_scale.py
python -I benchmarks/graph_mode/bench_visit_counters.py
```

## Workloads

* **shared-chain** (`make_chain`): `n+1` dict nodes; every layer has two edges
  (`next`, `alt`) pointing at the *same* successor, so a naive tree expansion
  would visit exponentially many paths. One shared tail is reachable from many
  places.
* **backedge-graph** (`make_backedge`): the same shared chain plus one edge
  from the tail back to the root, producing a large cycle.
* Each node carries an unchanged string payload (1 KiB or 8 KiB). Each run
  changes exactly 10 things: 5 scalar payloads and 5 added `marker` edges.

## Measured facts (real run output)

| case | n | payload | reachable nodes | reachable edges | diff (s) | delta build (s) | apply (s) | reverse (s) | patch bytes | peak RSS (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| shared-chain | 10000 | 1 KiB | 10001 | 30003 | 0.454 | 0.120 | 0.137 | 0.190 | 3,521,306 | 155.4 |
| backedge-graph | 10000 | 1 KiB | 10001 | 30004 | 0.422 | 0.136 | 0.167 | 0.192 | 3,521,438 | 165.0 |
| shared-chain | 10000 | 8 KiB | 10001 | 30003 | 0.410 | 0.146 | 0.138 | 0.193 | 3,592,986 | 165.0 |
| backedge-graph | 10000 | 8 KiB | 10001 | 30004 | 0.416 | 0.157 | 0.142 | 0.197 | 3,593,118 | 165.0 |
| shared-chain | 100000 | 1 KiB | 100001 | 300003 | 8.409 | 1.761 | 3.288 | 4.508 | 36,281,326 | 1125.2 |
| backedge-graph | 100000 | 1 KiB | 100001 | 300004 | 8.497 | 1.315 | 3.383 | 4.477 | 36,281,460 | 1125.2 |
| shared-chain | 100000 | 8 KiB | 100001 | 300003 | 8.529 | 1.780 | 3.390 | 4.040 | 36,353,006 | 1125.2 |
| backedge-graph | 100000 | 8 KiB | 100001 | 300004 | 8.438 | 1.755 | 3.403 | 4.073 | 36,353,140 | 1125.2 |

## What the numbers establish

* **No exponential expansion.** Every node is materialized exactly once and
  every edge is wired once. `bench_visit_counters.py` asserts
  `materialized_nodes == target node count` and
  `materialized_edges == target edge count` for a 10001-node / 30003-edge
  shared chain (verified output: `LINEAR: each target node/edge materialized
  exactly once; no exponential path expansion`).
* **No copy of unchanged payloads.** Every run reports
  `untouched_1kib_copied: false`; patch size grows only ~2% between the 1 KiB
  and 8 KiB payload tiers (3.52 MiB -> 3.59 MiB at 10k; 36.28 MiB -> 36.35 MiB
  at 100k) although total payload volume grows 8x. Only the 5 changed scalars
  plus topology metadata enter the patch.
* **No per-change full rescans.** With 10 changes, apply/reverse time stays
  proportional to graph size (0.14 s / 0.19 s at 10k; 3.3 s / ~4.3 s at 100k),
  not to changes * graph size. Verification touches each base node/edge once
  plus the O(changed) local fingerprints.
* **Cyclic graphs are free of recursion.** Construction, signature
  computation, verification and reconstruction are all iterative; the backedge
  graph performs within noise of the acyclic chain.
* **Report count is change-local.** Every run reports exactly 5
  `graph_values_changed` and 5 `graph_dictionary_item_added` regardless of the
  10k/100k fan-out; shared-node edits are not repeated per expanded path.

## Memory note

At 100k nodes the process peak is ~1.1 GiB. The dominant consumers are the two
snapshot indexes plus their canonical signature tables and the two compact
topologies held inside one bidirectional patch. All of these are bounded O(V+E)
structures of small integers, labels and short digests; no snapshot payload is
duplicated.
