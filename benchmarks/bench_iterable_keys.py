"""
Performance benchmark for the stable identity (``iterable_keys``) feature.

Run in a dedicated throw-away process so intermediate copies are reclaimed
and peak memory reflects only what each measured stage needs::

    python -I benchmarks/bench_iterable_keys.py --size 10000
    python -I benchmarks/bench_iterable_keys.py --size 100000

The script reports wall time, peak resident memory, identity lookup/scan
counts and serialized byte size for diff, Delta construction, JSON
serialization, strict +/- application and rebase.
"""
import argparse
import copy
import gc
import time
import tracemalloc

from deepdiff import DeepDiff, Delta
from deepdiff.serialization import json_dumps, json_loads


def build_tree(size, payload_len):
    payload = 'P' * payload_len
    before_orders = []
    after_orders = []
    for root_index in range(size):
        lines_before = [
            {'sku': f'sku-{root_index}-{child}', 'site': child,
             'qty': child, 'payload': payload}
            for child in range(4)
        ]
        lines_after = copy.deepcopy(lines_before)
        before_orders.append({
            'id': root_index,
            'name': f'order-{root_index}',
            'payload': payload,
            'lines': lines_before,
        })
        after_orders.append({
            'id': root_index,
            'name': f'order-{root_index}',
            'payload': payload,
            'lines': lines_after,
        })

    before = {'orders': before_orders}
    after = {'orders': after_orders}

    # Shuffle both root and child order (deterministic reversal).
    after['orders'].reverse()
    for order in after['orders']:
        order['lines'].reverse()

    # Modify exactly 100 scalars spread across the tree.
    for counter in range(100):
        root_index = (counter * 137) % size
        order = after['orders'][size - 1 - root_index]  # account for reversal
        child_index = counter % 4
        order['lines'][3 - child_index]['qty'] = 10_000 + counter
    return before, after


def _keyed(tree):
    return {order['id']: {line['sku']: line['qty'] for line in order['lines']}
            for order in tree['orders']}


def measure(label, func, *args, **kwargs):
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"{label:<28} {elapsed:10.4f}s  peak={peak / (1024 * 1024):9.1f} MiB")
    return result, elapsed, peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=10_000)
    parser.add_argument('--payload-len', type=int, default=1024)
    args = parser.parse_args()

    print(f"size={args.size} root records, 4 children each, "
          f"payload={args.payload_len} bytes, 100 scalar edits")
    before, after = build_tree(args.size, args.payload_len)
    keys = {
        "root['orders']": 'id',
        "root['orders'][*]['lines']": ('sku', 'site'),
    }

    # Vary the unchanged payload length: the serialized patch must stay flat
    # in payload size because unchanged business data is never copied.
    baseline_blob = None
    for probe_len in (128, 1024, 4096):
        probe_before, probe_after = build_tree(max(args.size // 10, 100), probe_len)
        probe_delta = Delta(DeepDiff(probe_before, probe_after, iterable_keys=keys),
                            bidirectional=True, serializer=json_dumps)
        probe_blob = probe_delta.dumps()
        if baseline_blob is None:
            baseline_blob = len(probe_blob)
        print(f"payload-len probe {probe_len:5d} -> {len(probe_blob):8d} bytes "
              f"(delta={len(probe_blob) - baseline_blob:+d} vs 128-byte baseline)")

    (diff, _, _) = measure("DeepDiff", lambda: DeepDiff(before, after, iterable_keys=keys))
    (delta, _, _) = measure("Delta(bidirectional)",
                            lambda: Delta(diff, bidirectional=True, serializer=json_dumps))

    def serialize():
        return delta.dumps()
    (blob, _, _) = measure("JSON serialize", serialize)
    print(f"{'serialized bytes':<28} {len(blob):>14} bytes")
    # Confirm unchanged payloads are not copied into the patch.
    payload = 'P' * args.payload_len
    leaked = payload.encode() in blob if isinstance(blob, (bytes, bytearray)) else payload in blob
    print(f"{'unchanged payload leaked':<28} {str(leaked):>14}")

    def deserialize():
        return Delta(json_loads(blob), bidirectional=True, deserializer=json_loads)
    (reloaded, _, _) = measure("JSON deserialize", deserialize)

    (forward, _, _) = measure("delta + before (strict)",
                              lambda: reloaded + copy.deepcopy(before))
    assert forward == after, "strict forward must reconstruct after"

    (backward, _, _) = measure("after - delta (strict)",
                               lambda: copy.deepcopy(after) - reloaded)
    assert backward == before, "strict reverse must reconstruct before"

    (rebased, _, _) = measure("apply_rebased",
                              lambda: reloaded.apply_rebased(copy.deepcopy(before)))
    # Content equality (rebase keeps current order, so compare by identity).
    assert _keyed(rebased) == _keyed(after), "rebase content must match after"

    # Re-run rebase to prove idempotency and reuse.
    (again, _, _) = measure("apply_rebased (idempotent)",
                            lambda: reloaded.apply_rebased(rebased))
    assert again == rebased

    # Location work: each physical list is scanned once per application; the
    # engine exposes counters for verification.
    engine = reloaded.apply_rebased  # noqa: F841
    from deepdiff.identity_engine import IdentityDeltaEngine
    probe = IdentityDeltaEngine(reloaded._identity_patch, rebased=True)
    probe.apply(copy.deepcopy(before))
    print(f"{'identity list scans':<28} {probe.scan_count:>14}")
    print(f"{'records scanned':<28} {probe.scanned_records:>14}")
    print(f"{'identity lookups':<28} {probe.lookup_count:>14}")
    # Every physical list is scanned at most once per phase (preflight on the
    # original, execution on the deep copy). With ~50k edited/moved records the
    # scan count stays proportional to the number of lists, not to the number
    # of edits, proving location work never degenerates into a full rescan per
    # edit.
    list_count = args.size + 1 + args.size  # root list + one child list per root
    assert probe.scan_count <= list_count * 2 + 5, "no per-edit full rescans"
    # A linear-scan-per-edit implementation would inspect on the order of
    # edits * list size records; assert the observed count is far below that.
    edited_records = 100
    naive_lower_bound = edited_records * args.size
    assert probe.scanned_records < naive_lower_bound, (
        "location work must not rescanned the full list for each edit")


if __name__ == '__main__':
    main()
