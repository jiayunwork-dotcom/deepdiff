"""
Fresh-process worker for mixed identity/plain delta round trips.

Invoked by ``tests/test_iterable_keys_mixed.py``. It receives a JSON document
holding the before/after snapshots and a JSON serialized delta blob, rebuilds
the delta in a brand new interpreter and verifies forward, reverse, rebase,
idempotency and conflict behavior purely from the on-disk document.
"""
import copy
import json
import sys


def main(payload_path: str) -> int:
    from deepdiff import Delta, DeltaConflictError
    from deepdiff.serialization import json_loads

    with open(payload_path) as handle:
        payload = json.load(handle)
    before = payload['before']
    after = payload['after']

    delta = Delta(payload['blob'], bidirectional=True, deserializer=json_loads)

    assert (delta + copy.deepcopy(before)) == after, "forward application failed"
    assert (copy.deepcopy(after) - delta) == before, "reverse application failed"
    rebased = delta.apply_rebased(copy.deepcopy(before))
    assert rebased == after, "rebase failed"
    assert delta.apply_rebased(copy.deepcopy(rebased)) == rebased, "not idempotent"

    divergent = copy.deepcopy(before)
    divergent['batch'] = 999
    try:
        delta.apply_rebased(divergent)
    except DeltaConflictError:
        pass
    else:  # pragma: no cover - assertion failure path
        raise AssertionError("conflict not reported after fresh-process reload")
    print("mixed paths fresh-process round trip OK")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
