"""
Cross-process round trip for identity aware deltas.

Run as a standalone script (``python -I tests/_identity_subprocess_worker.py``)
or invoked through :mod:`tests.test_iterable_keys_subprocess`. It writes a JSON
serialized identity delta to a temp file, reloads it in a fresh interpreter and
verifies forward/reverse application, rebase and conflict reporting.
"""
import json
import sys


def main(payload_path: str) -> int:
    from deepdiff import DeepDiff, Delta, DeltaConflictError
    from deepdiff.serialization import json_dumps, json_loads

    before = {
        'orders': [
            {'id': 1, 'name': 'a', 'lines': [{'sku': 'x', 'site': 1, 'qty': 1}]},
            {'id': 2, 'name': 'b', 'lines': []},
        ]
    }
    after = {
        'orders': [
            {'id': 2, 'name': 'b', 'lines': []},
            {'id': 1, 'name': 'a2',
             'lines': [{'sku': 'x', 'site': 1, 'qty': 1}, {'sku': 'z', 'site': 3, 'qty': 9}]},
            {'id': 3, 'name': 'c', 'lines': []},
        ]
    }
    keys = {"root['orders']": 'id', "root['orders'][*]['lines']": ('sku', 'site')}
    delta = Delta(DeepDiff(before, after, iterable_keys=keys),
                  bidirectional=True, serializer=json_dumps)
    with open(payload_path, 'w') as handle:
        handle.write(delta.dumps())

    # Fresh "process": reload purely from the on-disk JSON document.
    with open(payload_path) as handle:
        reloaded = Delta(handle.read(), bidirectional=True, deserializer=json_loads)

    import copy
    assert (reloaded + copy.deepcopy(before)) == after, "forward application failed"
    assert (copy.deepcopy(after) - reloaded) == before, "reverse application failed"
    rebased = reloaded.apply_rebased(copy.deepcopy(before))
    assert {o['id'] for o in rebased['orders']} == {1, 2, 3}
    assert reloaded.apply_rebased(rebased) == rebased, "rebase must be idempotent"

    divergent = copy.deepcopy(before)
    divergent['orders'][0]['name'] = 'diverged'
    try:
        reloaded.apply_rebased(divergent)
    except DeltaConflictError as exc:
        assert exc.kind == 'value_mismatch'
        assert exc.stable_path == "root['orders'][int=1]['name']"
    else:  # pragma: no cover - assertion failure path
        raise AssertionError("expected DeltaConflictError")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
