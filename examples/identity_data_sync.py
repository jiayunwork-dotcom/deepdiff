"""
Data synchronization example using stable identity lists.

This module is documentation-as-code: it shows the full lifecycle of an
``iterable_keys`` based patch shared between two processes. Run it directly::

    python -I examples/identity_data_sync.py
"""
from __future__ import annotations

import copy
import json

from deepdiff import DeepDiff, Delta, DeltaConflictError
from deepdiff.serialization import json_dumps, json_loads


# The upstream canonical document and a local copy that has diverged.
UPSTREAM_BEFORE = {
    'orders': [
        {'id': 1, 'customer': 'Ada', 'lines': [
            {'sku': 'book', 'site': 1, 'qty': 1},
        ]},
        {'id': 2, 'customer': 'Lin', 'lines': []},
    ]
}

UPSTREAM_AFTER = {
    'orders': [
        # order 2 moved to the front, order 1 gained a line and changed owner
        {'id': 2, 'customer': 'Lin', 'lines': []},
        {'id': 1, 'customer': 'Ada L.', 'lines': [
            {'sku': 'pen', 'site': 2, 'qty': 4},
            {'sku': 'book', 'site': 1, 'qty': 1},
        ]},
        {'id': 3, 'customer': 'Grace', 'lines': []},
    ]
}

ITERABLE_KEYS = {
    "root['orders']": 'id',
    "root['orders'][*]['lines']": ('sku', 'site'),
}


def build_patch() -> str:
    """Produce a portable JSON patch that another process can load."""
    diff = DeepDiff(UPSTREAM_BEFORE, UPSTREAM_AFTER, iterable_keys=ITERABLE_KEYS)
    delta = Delta(diff, bidirectional=True, serializer=json_dumps)
    return delta.dumps()


def main() -> None:
    patch_blob = build_patch()
    print("serialized patch (bytes):", len(patch_blob))

    # A second "process" only sees the JSON patch and its own local document.
    local = copy.deepcopy(UPSTREAM_BEFORE)
    # The local branch reordered records and edited an untouched field.
    local['orders'].reverse()
    local['orders'][1]['local_note'] = 'do not lose me'

    delta = Delta(json.loads(patch_blob), bidirectional=True,
                  deserializer=json_loads)
    merged = delta.apply_rebased(local)

    by_id = {order['id']: order for order in merged['orders']}
    print("merged order ids:", [order['id'] for order in merged['orders']])
    print("upstream owner applied:", by_id[1]['customer'] == 'Ada L.')
    print("local note preserved:", by_id[1].get('local_note') == 'do not lose me')
    print("new order present:", 3 in by_id)

    # A conflicting local edit on the exact same field is reported precisely.
    conflict_local = copy.deepcopy(UPSTREAM_BEFORE)
    conflict_local['orders'][0]['customer'] = 'someone-else'
    try:
        delta.apply_rebased(conflict_local)
    except DeltaConflictError as error:
        print("conflict:", error.kind, "at", error.stable_path,
              "| expected", error.expected, "| current", error.current)

    # Strict application against the true base restores the full document,
    # including its exact order, and subtraction reverses it.
    assert (delta + copy.deepcopy(UPSTREAM_BEFORE)) == UPSTREAM_AFTER
    assert (copy.deepcopy(UPSTREAM_AFTER) - delta) == UPSTREAM_BEFORE
    print("strict forward/reverse round trip: ok")


if __name__ == '__main__':
    main()
