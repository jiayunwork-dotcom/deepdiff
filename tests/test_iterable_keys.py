"""
Tests for the stable identity feature (``iterable_keys``).

The expected facts in this module were written by hand from small example
trees before any identity-locator based helpers existed; the assertions only
use literal expected stable paths / identities and plain dict comparisons.
"""
import copy
import json
import os
import subprocess
import sys

import pytest

from deepdiff import DeepDiff, Delta, DeltaConflictError
from deepdiff.identity import (
    IdentityRegistry, DuplicateIdentityError, IdentityKeyError,
    NotARecordListError, IterableKeysDeclarationError, IdentityError,
    parse_identity_segment, format_identity_segment,
    stable_path_elements, format_stable_path,
    stable_path_to_flat_tokens, flat_tokens_to_stable_path,
)
from deepdiff.serialization import json_dumps, json_loads


ORDERS_KEYS = {
    "root['orders']": 'id',
    "root['orders'][*]['lines']": ('sku', 'site'),
}


def _orders_before():
    return {
        'orders': [
            {'id': 1, 'name': 'a', 'lines': [
                {'sku': 'x', 'site': 1, 'qty': 1},
                {'sku': 'y', 'site': 2, 'qty': 2},
            ]},
            {'id': 2, 'name': 'b', 'lines': []},
        ]
    }


def _orders_after():
    return {
        'orders': [
            {'id': 2, 'name': 'b', 'lines': []},
            {'id': 1, 'name': 'a2', 'lines': [
                {'sku': 'y', 'site': 2, 'qty': 20},
                {'sku': 'x', 'site': 1, 'qty': 1},
                {'sku': 'z', 'site': 3, 'qty': 9},
            ]},
            {'id': 3, 'name': 'c', 'lines': []},
        ]
    }


# ---------------------------------------------------------------------------
# Hand written expected facts
# ---------------------------------------------------------------------------

class TestHandWrittenFacts:
    def test_complete_fact_list_two_level_composite_keys(self):
        before = _orders_before()
        after = _orders_after()
        dd = DeepDiff(before, after, iterable_keys=ORDERS_KEYS)

        # --- record moves (root level) ----------------------------------
        moved = dd['iterable_identity_moved']
        # order id 2 moved index 1 -> 0
        assert "root['orders'][int=2]" in moved
        assert moved["root['orders'][int=2]"]['index_t1'] == 1
        assert moved["root['orders'][int=2]"]['index_t2'] == 0
        # order id 1 moved index 0 -> 1
        assert moved["root['orders'][int=1]"]['index_t1'] == 0
        assert moved["root['orders'][int=1]"]['index_t2'] == 1

        # --- record add / remove ----------------------------------------
        added = dd['iterable_identity_item_added']
        assert "root['orders'][int=3]" in added
        assert added["root['orders'][int=3]"]['value'] == {'id': 3, 'name': 'c', 'lines': []}
        assert added["root['orders'][int=3]"]['path_t1'] is None
        assert added["root['orders'][int=3]"]['path_t2'] == "root['orders'][2]"
        # nested new line
        line_added_path = "root['orders'][int=1]['lines'][str='z',int=3]"
        assert line_added_path in added
        assert added[line_added_path]['value'] == {'sku': 'z', 'site': 3, 'qty': 9}

        # --- pure content changes keep stable paths ---------------------
        values = dd['values_changed']
        assert values["root['orders'][int=1]['name']"]['old_value'] == 'a'
        assert values["root['orders'][int=1]['name']"]['new_value'] == 'a2'
        qty_path = "root['orders'][int=1]['lines'][str='y',int=2]['qty']"
        assert values[qty_path]['old_value'] == 2
        assert values[qty_path]['new_value'] == 20
        # physical locations on both sides are exposed too
        assert values[qty_path]['path_t1'] == "root['orders'][0]['lines'][1]['qty']"
        assert values[qty_path]['path_t2'] == "root['orders'][1]['lines'][0]['qty']"

        # nested moves are reported even when content changed as well
        nested_moved = moved["root['orders'][int=1]['lines'][str='y',int=2]"]
        assert nested_moved['index_t1'] == 1
        assert nested_moved['index_t2'] == 0

    def test_pure_reorder_is_not_add_remove(self):
        before = {'orders': [{'id': 1, 'v': 1}, {'id': 2, 'v': 2}, {'id': 3, 'v': 3}]}
        after = {'orders': [{'id': 3, 'v': 3}, {'id': 1, 'v': 1}, {'id': 2, 'v': 2}]}
        dd = DeepDiff(before, after, iterable_keys={"root['orders']": 'id'})
        assert 'iterable_identity_item_added' not in dd
        assert 'iterable_identity_item_removed' not in dd
        assert 'values_changed' not in dd
        moved = dd['iterable_identity_moved']
        assert set(moved) == {
            "root['orders'][int=1]",
            "root['orders'][int=2]",
            "root['orders'][int=3]",
        }

    def test_move_and_change_keeps_both_facts(self):
        before = {'orders': [{'id': 1, 'v': 'a'}]}
        after = {'orders': [{'id': 1, 'v': 'b'}]}
        # single record can't move; use two records and change one
        before = {'orders': [{'id': 1, 'v': 'a'}, {'id': 2, 'v': 'x'}]}
        after = {'orders': [{'id': 2, 'v': 'x'}, {'id': 1, 'v': 'b'}]}
        dd = DeepDiff(before, after, iterable_keys={"root['orders']": 'id'})
        assert "root['orders'][int=1]" in dd['iterable_identity_moved']
        assert "root['orders'][int=2]" in dd['iterable_identity_moved']
        assert dd['values_changed']["root['orders'][int=1]['v']"]['new_value'] == 'b'

    def test_key_change_is_remove_plus_insert(self):
        before = {'orders': [{'id': 1, 'v': 'a'}]}
        after = {'orders': [{'id': 9, 'v': 'a'}]}
        dd = DeepDiff(before, after, iterable_keys={"root['orders']": 'id'})
        assert "root['orders'][int=1]" in dd['iterable_identity_item_removed']
        assert "root['orders'][int=9]" in dd['iterable_identity_item_added']
        assert 'iterable_identity_moved' not in dd

    def test_int_and_string_identities_are_distinct(self):
        before = {'orders': [{'id': 1}]}
        after = {'orders': [{'id': 1}]}
        # same type and value -> no diff
        assert DeepDiff(before, after, iterable_keys={"root['orders']": 'id'}) == {}
        # int 1 vs string '1' -> different identities, treated as remove+add
        before = {'orders': [{'id': 1}]}
        after = {'orders': [{'id': '1'}]}
        dd = DeepDiff(before, after, iterable_keys={"root['orders']": 'id'})
        assert "root['orders'][str='1']" in dd['iterable_identity_item_added']
        assert "root['orders'][int=1]" in dd['iterable_identity_item_removed']
        assert 'iterable_identity_moved' not in dd

    def test_absent_vs_null_distinction(self):
        # a field that is null is different from a missing field
        before = {'orders': [{'id': 1, 'note': None}]}
        after = {'orders': [{'id': 1}]}
        dd = DeepDiff(before, after, iterable_keys={"root['orders']": 'id'})
        removed = dd['dictionary_item_removed']
        assert "root['orders'][int=1]['note']" in removed

    def test_tree_view_locates_identity_and_both_positions(self):
        before = _orders_before()
        after = _orders_after()
        dd = DeepDiff(before, after, view='tree', iterable_keys=ORDERS_KEYS)
        moves = list(dd['iterable_identity_moved'])
        order1 = next(c for c in moves if c.stable_path() == "root['orders'][int=1]")
        assert order1.identity == (('int', 1),)
        assert order1.physical_path(use_t2=False) == "root['orders'][0]"
        assert order1.physical_path(use_t2=True) == "root['orders'][1]"

    def test_undeclared_list_keeps_ordered_comparison(self):
        before = {'plain': [1, 2, 3], 'declared': [{'id': 1}]}
        after = {'plain': [3, 2, 1], 'declared': [{'id': 1}]}
        dd = DeepDiff(before, after, iterable_keys={"root['declared']": 'id'})
        # the plain list still uses index based semantics
        assert 'values_changed' in dd or 'iterable_item_added' in dd


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_missing_key_reports_side_and_path(self):
        with pytest.raises(IdentityKeyError) as exc:
            DeepDiff({'o': [{}]}, {'o': [{'id': 1}]},
                     iterable_keys={"root['o']": 'id'})
        assert 't1' in str(exc.value) and "root['o'][0]" in str(exc.value)

    def test_duplicate_identity_reports_indexes(self):
        with pytest.raises(DuplicateIdentityError) as exc:
            DeepDiff({'o': [{'id': 1}, {'id': 1}]}, {'o': []},
                     iterable_keys={"root['o']": 'id'})
        assert 'indexes 0 and 1' in str(exc.value)

    def test_non_list_target_rejected(self):
        with pytest.raises(NotARecordListError):
            DeepDiff({'o': {}}, {'o': {}}, iterable_keys={"root['o']": 'id'})

    def test_non_dict_element_rejected(self):
        with pytest.raises(NotARecordListError):
            DeepDiff({'o': [1]}, {'o': [2]}, iterable_keys={"root['o']": 'id'})

    def test_bool_and_empty_string_key_rejected(self):
        with pytest.raises(IdentityKeyError):
            DeepDiff({'o': [{'id': True}]}, {'o': []},
                     iterable_keys={"root['o']": 'id'})
        with pytest.raises(IdentityKeyError):
            DeepDiff({'o': [{'id': ''}]}, {'o': []},
                     iterable_keys={"root['o']": 'id'})

    @pytest.mark.parametrize('bad', ['', (), [], 123, None])
    def test_bad_key_declaration_rejected(self, bad):
        with pytest.raises(IterableKeysDeclarationError):
            IdentityRegistry({"root['o']": bad})

    def test_incompatible_options_rejected(self):
        with pytest.raises(ValueError):
            DeepDiff([], [], iterable_keys={"root['x']": 'id'}, ignore_order=True)
        with pytest.raises(ValueError):
            DeepDiff([], [], iterable_keys={"root['x']": 'id'}, group_by='id')
        with pytest.raises(ValueError):
            DeepDiff([], [], iterable_keys={"root['x']": 'id'},
                     iterable_compare_func=lambda x, y, level: True)

    def test_nan_float_rejected(self):
        with pytest.raises(IdentityError):
            DeepDiff({'o': [{'id': 1, 'f': float('nan')}]},
                     {'o': [{'id': 1, 'f': 1.0}]},
                     iterable_keys={"root['o']": 'id'})

    def test_cycle_rejected(self):
        a = {'id': 1}
        a['self'] = a
        with pytest.raises(IdentityError):
            DeepDiff({'o': [a]}, {'o': []}, iterable_keys={"root['o']": 'id'})

    def test_children_of_different_parents_do_not_clash(self):
        before = {'orders': [
            {'id': 1, 'lines': [{'sku': 'x'}]},
            {'id': 2, 'lines': [{'sku': 'x'}]},
        ]}
        after = copy.deepcopy(before)
        # same child identity 'x' under two parents is perfectly valid
        assert DeepDiff(before, after, iterable_keys={
            "root['orders']": 'id',
            "root['orders'][*]['lines']": 'sku',
        }) == {}

    def test_declaration_absent_on_both_sides_allowed(self):
        dd = DeepDiff({'a': 1}, {'a': 2}, iterable_keys={"root['missing']": 'id'})
        assert 'values_changed' in dd

    def test_inputs_and_declaration_not_mutated(self):
        before = _orders_before()
        after = _orders_after()
        declaration = copy.deepcopy(ORDERS_KEYS)
        b1, a1 = copy.deepcopy(before), copy.deepcopy(after)
        DeepDiff(before, after, iterable_keys=declaration)
        assert before == b1 and after == a1 and declaration == ORDERS_KEYS


# ---------------------------------------------------------------------------
# Strict delta application and reverse
# ---------------------------------------------------------------------------

class TestStrictDelta:
    def test_forward_reconstructs_full_after_with_order(self):
        before, after = _orders_before(), _orders_after()
        delta = Delta(DeepDiff(before, after, iterable_keys=ORDERS_KEYS), bidirectional=True)
        result = delta + copy.deepcopy(before)
        assert result == after
        assert [o['id'] for o in result['orders']] == [2, 1, 3]

    def test_reverse_reconstructs_full_before_with_order(self):
        before, after = _orders_before(), _orders_after()
        delta = Delta(DeepDiff(before, after, iterable_keys=ORDERS_KEYS), bidirectional=True)
        result = copy.deepcopy(after) - delta
        assert result == before
        assert [o['id'] for o in result['orders']] == [1, 2]

    def test_bidirectional_required(self):
        before, after = _orders_before(), _orders_after()
        with pytest.raises(ValueError):
            Delta(DeepDiff(before, after, iterable_keys=ORDERS_KEYS))

    def test_strict_application_conflicts_on_divergence(self):
        before, after = _orders_before(), _orders_after()
        delta = Delta(DeepDiff(before, after, iterable_keys=ORDERS_KEYS), bidirectional=True)
        divergent = copy.deepcopy(before)
        divergent['orders'][0]['name'] = 'locally-changed'
        with pytest.raises(DeltaConflictError) as exc:
            delta + divergent
        assert exc.value.kind == 'value_mismatch'

    def test_identity_not_found_conflict(self):
        before, after = _orders_before(), _orders_after()
        delta = Delta(DeepDiff(before, after, iterable_keys=ORDERS_KEYS), bidirectional=True)
        # remove the record the field change targets
        divergent = copy.deepcopy(before)
        divergent['orders'] = [o for o in divergent['orders'] if o['id'] != 1]
        with pytest.raises(DeltaConflictError) as exc:
            delta + divergent
        assert exc.value.kind == 'identity_not_found'


# ---------------------------------------------------------------------------
# Rebase
# ---------------------------------------------------------------------------

class TestRebase:
    def _delta(self):
        return Delta(DeepDiff(_orders_before(), _orders_after(),
                              iterable_keys=ORDERS_KEYS), bidirectional=True)

    def test_non_overlapping_local_edits_are_preserved(self):
        delta = self._delta()
        current = {
            'orders': [
                # local brand new record stays untouched
                {'id': 9, 'name': 'local', 'lines': [], 'local_only': True},
                # keep the lines the patch modifies, but add a local field
                {'id': 1, 'name': 'a',
                 'lines': [{'sku': 'x', 'site': 1, 'qty': 1},
                           {'sku': 'y', 'site': 2, 'qty': 2}],
                 'note': 'keep-me'},
                # a local edit on a different field of record 2
                {'id': 2, 'name': 'b', 'lines': [], 'tag': 'local'},
            ]
        }
        snapshot = copy.deepcopy(current)
        result = delta.apply_rebased(current)
        by_id = {o['id']: o for o in result['orders']}
        # patch applied
        assert by_id[1]['name'] == 'a2'
        # local additions and edits preserved
        assert by_id[1]['note'] == 'keep-me'
        assert by_id[2]['tag'] == 'local'
        assert by_id[9]['local_only'] is True
        # input untouched
        assert current == snapshot

    def test_same_edit_is_accepted(self):
        delta = self._delta()
        # current already has the patched new value
        current = copy.deepcopy(_orders_after())
        result = delta.apply_rebased(current)
        assert result == current

    def test_divergent_edit_conflicts_with_details(self):
        delta = self._delta()
        current = copy.deepcopy(_orders_before())
        current['orders'][0]['name'] = 'different'
        with pytest.raises(DeltaConflictError) as exc:
            delta.apply_rebased(current)
        err = exc.value
        assert err.kind == 'value_mismatch'
        assert err.stable_path == "root['orders'][int=1]['name']"
        assert err.expected == 'a'
        assert err.current == 'different'

    def test_add_path_accepts_when_equal_else_conflicts(self):
        before = {'orders': [{'id': 1}]}
        after = {'orders': [{'id': 1, 'extra': 5}]}
        delta = Delta(DeepDiff(before, after, iterable_keys={"root['orders']": 'id'}),
                      bidirectional=True)
        # already present with same value -> accept
        current = copy.deepcopy(after)
        assert delta.apply_rebased(current) == current
        # present with different value -> conflict
        conflict = copy.deepcopy(before)
        conflict['orders'][0]['extra'] = 99
        with pytest.raises(DeltaConflictError) as exc:
            delta.apply_rebased(conflict)
        assert exc.value.kind == 'already_present'

    def test_delete_path_accepts_absence_and_verifies_value(self):
        before = {'orders': [{'id': 1, 'extra': 5}]}
        after = {'orders': [{'id': 1}]}
        delta = Delta(DeepDiff(before, after, iterable_keys={"root['orders']": 'id'}),
                      bidirectional=True)
        # already absent -> accept
        assert delta.apply_rebased(copy.deepcopy(after)) == after
        # different value present -> conflict
        conflict = {'orders': [{'id': 1, 'extra': 7}]}
        with pytest.raises(DeltaConflictError):
            delta.apply_rebased(conflict)

    def test_ordering_rules_survivors_then_appended(self):
        before = {'orders': [{'id': 1}, {'id': 2}]}
        after = {'orders': [{'id': 2}, {'id': 1}, {'id': 3}]}
        delta = Delta(DeepDiff(before, after, iterable_keys={"root['orders']": 'id'}),
                      bidirectional=True)
        current = {'orders': [{'id': 1}, {'id': 2}, {'id': 8}]}
        result = delta.apply_rebased(current)
        # current survivors keep relative order, patch-new identity 3 appended
        assert [o['id'] for o in result['orders']] == [1, 2, 8, 3]

    def test_idempotent_same_direction(self):
        delta = self._delta()
        once = delta.apply_rebased(copy.deepcopy(_orders_before()))
        twice = delta.apply_rebased(once)
        assert once == twice

    def test_reverse_rebase(self):
        delta = self._delta()
        result = delta.apply_rebased(copy.deepcopy(_orders_after()), reverse=True)
        by_id = {o['id']: o for o in result['orders']}
        assert by_id[1]['name'] == 'a'
        assert 3 not in by_id

    def test_no_partial_result_on_conflict(self):
        delta = self._delta()
        current = copy.deepcopy(_orders_before())
        # diverge on a field the patch also changes
        current['orders'][0]['name'] = 'divergent'
        snapshot = copy.deepcopy(current)
        with pytest.raises(DeltaConflictError):
            delta.apply_rebased(current)
        assert current == snapshot

    def test_reusable_after_conflict(self):
        delta = self._delta()
        bad = copy.deepcopy(_orders_before())
        bad['orders'][0]['name'] = 'x'
        with pytest.raises(DeltaConflictError):
            delta.apply_rebased(bad)
        # same delta still applies cleanly to the true base
        assert (delta + copy.deepcopy(_orders_before())) == _orders_after()

    def test_removed_subtree_concurrent_descendant_change_conflicts(self):
        # deleting a whole record must verify the entire subtree; a divergent
        # descendant can not be silently dropped
        before = {'orders': [{'id': 1, 'meta': {'nested': 1}}]}
        after = {'orders': []}
        delta = Delta(DeepDiff(before, after, iterable_keys={"root['orders']": 'id'}),
                      bidirectional=True)
        divergent = {'orders': [{'id': 1, 'meta': {'nested': 2}}]}
        with pytest.raises(DeltaConflictError):
            delta.apply_rebased(divergent)

    def test_parent_replacement_concurrent_child_change_conflicts(self):
        before = {'orders': [{'id': 1, 'v': {'a': 1}}]}
        after = {'orders': [{'id': 1, 'v': {'a': 2}}]}
        delta = Delta(DeepDiff(before, after, iterable_keys={"root['orders']": 'id'}),
                      bidirectional=True)
        divergent = {'orders': [{'id': 1, 'v': {'a': 99}}]}
        with pytest.raises(DeltaConflictError):
            delta.apply_rebased(divergent)


# ---------------------------------------------------------------------------
# Serialization round trips
# ---------------------------------------------------------------------------

class TestSerialization:
    def _delta(self):
        return Delta(DeepDiff(_orders_before(), _orders_after(),
                              iterable_keys=ORDERS_KEYS), bidirectional=True)

    def test_json_roundtrip_in_new_data(self):
        delta = self._delta()
        blob = json_dumps(delta.diff)
        loaded = json_loads(blob)
        rebuilt = Delta(loaded, bidirectional=True)
        assert (rebuilt + copy.deepcopy(_orders_before())) == _orders_after()
        assert (copy.deepcopy(_orders_after()) - rebuilt) == _orders_before()

    def test_json_roundtrip_rebase_and_conflict(self):
        delta = Delta(DeepDiff(_orders_before(), _orders_after(),
                               iterable_keys=ORDERS_KEYS),
                      bidirectional=True, serializer=json_dumps)
        blob = delta.dumps()
        rebuilt = Delta(blob, bidirectional=True, deserializer=json_loads)
        result = rebuilt.apply_rebased(copy.deepcopy(_orders_before()))
        assert {o['id'] for o in result['orders']} == {1, 2, 3}
        bad = copy.deepcopy(_orders_before())
        bad['orders'][0]['name'] = 'diverged'
        with pytest.raises(DeltaConflictError):
            rebuilt.apply_rebased(bad)

    def test_flat_dict_roundtrip(self):
        delta = self._delta()
        flat = delta.to_flat_dicts()
        rebuilt = Delta(flat_dict_list=copy.deepcopy(flat), bidirectional=True)
        assert (rebuilt + copy.deepcopy(_orders_before())) == _orders_after()
        assert (copy.deepcopy(_orders_after()) - rebuilt) == _orders_before()
        result = rebuilt.apply_rebased(copy.deepcopy(_orders_before()))
        assert rebuilt.apply_rebased(result) == result

    def test_corrupt_metadata_rejected(self):
        delta = self._delta()
        corrupt = copy.deepcopy(delta.diff)
        corrupt['_identity_patch']['version'] = 999
        with pytest.raises(Exception):
            Delta(corrupt, bidirectional=True)

    def test_flat_without_metadata_rejected(self):
        delta = self._delta()
        flat = [row for row in delta.to_flat_dicts()
                if row['action'] != 'identity_metadata']
        with pytest.raises(Exception):
            Delta(flat_dict_list=flat, bidirectional=True)

    def test_unchanged_payload_not_copied(self):
        payload = 'P' * 1024
        before = {'orders': [{'id': 1, 'payload': payload, 'name': 'a'},
                             {'id': 2, 'payload': payload, 'name': 'b'}]}
        after = {'orders': [{'id': 2, 'payload': payload, 'name': 'b'},
                            {'id': 1, 'payload': payload, 'name': 'a2'}]}
        delta = Delta(DeepDiff(before, after, iterable_keys={"root['orders']": 'id'}),
                      bidirectional=True)
        blob = json_dumps(delta.diff)
        assert payload not in blob


# ---------------------------------------------------------------------------
# Stable path parsing helpers (no eval)
# ---------------------------------------------------------------------------

class TestStablePathHelpers:
    def test_identity_segment_roundtrip(self):
        identity = (('str', 'a'), ('int', 1))
        segment = format_identity_segment(identity)
        assert segment == "[str='a',int=1]"
        assert parse_identity_segment(segment) == identity

    def test_full_stable_path_roundtrip(self):
        path = "root['orders'][int=7]['lines'][str='z',int=3]['qty']"
        elements = stable_path_elements(path)
        assert elements == [
            ('field', 'orders'),
            ('identity', (('int', 7),)),
            ('field', 'lines'),
            ('identity', (('str', 'z'), ('int', 3))),
            ('field', 'qty'),
        ]
        assert format_stable_path(elements) == path

    def test_flat_token_roundtrip(self):
        path = "root['orders'][int=7]['lines'][str='z',int=3]"
        tokens = stable_path_to_flat_tokens(path)
        assert flat_tokens_to_stable_path(tokens) == path
        # JSON-safe
        assert json.loads(json.dumps(tokens)) == tokens

    def test_malformed_segment_rejected_without_eval(self):
        for bad in ["[int=abc]", "[str='a',int=]", "[weird=1]", "[]"]:
            with pytest.raises(IdentityError):
                parse_identity_segment(bad)


class TestSubprocessRoundTrip:
    def test_fresh_process_json_roundtrip(self, tmp_path):
        payload = tmp_path / 'identity_delta.json'
        worker = os.path.join(os.path.dirname(__file__), '_identity_subprocess_worker.py')
        env = dict(os.environ)
        # Ensure the worker imports the in-tree package rather than an
        # installed copy.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env['PYTHONPATH'] = repo_root + os.pathsep + env.get('PYTHONPATH', '')
        completed = subprocess.run(
            [sys.executable, '-I', worker, str(payload)],
            capture_output=True, text=True, env=env,
        )
        assert completed.returncode == 0, completed.stderr
        assert payload.exists() and payload.stat().st_size > 0


class TestRootAndContainerCases:
    def test_root_level_identity_list(self):
        before = [{'id': 1, 'v': 'a'}]
        after = [{'id': 1, 'v': 'b'}, {'id': 2, 'v': 'c'}]
        delta = Delta(DeepDiff(before, after, iterable_keys={'root': 'id'}),
                      bidirectional=True)
        assert (delta + copy.deepcopy(before)) == after
        assert (copy.deepcopy(after) - delta) == before
        assert delta.apply_rebased(copy.deepcopy(before)) == after

    def test_nested_move_does_not_relocate_child_change(self):
        before = {'orders': [
            {'id': 1, 'lines': [{'sku': 'a', 'q': 1}]},
            {'id': 2, 'lines': [{'sku': 'b', 'q': 2}]},
        ]}
        after = {'orders': [
            {'id': 2, 'lines': [{'sku': 'b', 'q': 2}]},
            {'id': 1, 'lines': [{'sku': 'a', 'q': 99}]},
        ]}
        keys = {"root['orders']": 'id', "root['orders'][*]['lines']": 'sku'}
        delta = Delta(DeepDiff(before, after, iterable_keys=keys), bidirectional=True)
        result = delta + copy.deepcopy(before)
        # the qty change stays attached to order 1 / sku a despite the move
        target = next(o for o in result['orders'] if o['id'] == 1)
        line = next(item for item in target['lines'] if item['sku'] == 'a')
        assert line['q'] == 99

    def test_pickle_roundtrip(self):
        before = _orders_before()
        after = _orders_after()
        delta = Delta(DeepDiff(before, after, iterable_keys=ORDERS_KEYS), bidirectional=True)
        rebuilt = Delta(delta.dumps(), bidirectional=True)
        assert (rebuilt + copy.deepcopy(before)) == after
        assert (copy.deepcopy(after) - rebuilt) == before

    def test_absent_distinct_from_null_in_conflict(self):
        before = {'orders': [{'id': 1}]}
        after = {'orders': [{'id': 1, 'note': None}]}
        delta = Delta(DeepDiff(before, after, iterable_keys={"root['orders']": 'id'}),
                      bidirectional=True)
        # field currently holds a non-null value -> conflict
        conflict = {'orders': [{'id': 1, 'note': 'x'}]}
        with pytest.raises(DeltaConflictError) as exc:
            delta.apply_rebased(conflict)
        assert exc.value.expected is not None
        # absent is represented by a sentinel, never None
        from deepdiff.identity_engine import ABSENT
        assert ABSENT is not None


class TestInvalidCurrentTree:
    def _delta(self):
        t1 = {'o': [{'id': 1, 'v': 1}]}
        t2 = {'o': [{'id': 1, 'v': 2}]}
        return Delta(DeepDiff(t1, t2, iterable_keys={"root['o']": 'id'}),
                     bidirectional=True)

    def test_duplicate_identity_in_current_conflicts(self):
        delta = self._delta()
        current = {'o': [{'id': 1, 'v': 1}, {'id': 1, 'v': 1}]}
        with pytest.raises(DeltaConflictError) as exc:
            delta.apply_rebased(current)
        assert exc.value.kind == 'duplicate_identity'

    def test_missing_key_in_current_conflicts(self):
        delta = self._delta()
        current = {'o': [{'v': 1}]}
        with pytest.raises(DeltaConflictError) as exc:
            delta.apply_rebased(current)
        assert exc.value.kind == 'bad_identity'

    def test_missing_identity_for_update_conflicts(self):
        delta = self._delta()
        current = {'o': [{'id': 9, 'v': 1}]}
        with pytest.raises(DeltaConflictError) as exc:
            delta.apply_rebased(current)
        assert exc.value.kind == 'identity_not_found'
        assert exc.value.stable_path == "root['o'][int=1]['v']"

    def test_delta_still_usable_after_these_conflicts(self):
        delta = self._delta()
        with pytest.raises(DeltaConflictError):
            delta.apply_rebased({'o': [{'id': 9, 'v': 1}]})
        base = {'o': [{'id': 1, 'v': 1}]}
        assert (delta + copy.deepcopy(base)) == {'o': [{'id': 1, 'v': 2}]}


class TestWildcardSemantics:
    def test_wildcard_matches_exactly_one_element_layer(self):
        # a[*] are themselves lists, not dict records -> reject
        nested_lists = {'a': [[{'id': 1}]]}
        with pytest.raises(NotARecordListError):
            DeepDiff(nested_lists, nested_lists,
                     iterable_keys={"root['a'][*]": 'id'})

    def test_trailing_wildcard_identifies_record_list(self):
        records = {'a': [{'id': 1}, {'id': 2}]}
        assert DeepDiff(records, records,
                        iterable_keys={"root['a'][*]": 'id'}) == {}

    def test_nested_wildcard_scoped_per_parent(self):
        before = {
            'orders': [
                {'id': 1, 'lines': [{'sku': 'a'}]},
                {'id': 2, 'lines': [{'sku': 'a'}]},
            ]
        }
        # identical child identity under different parents is not a duplicate
        assert DeepDiff(before, copy.deepcopy(before), iterable_keys={
            "root['orders']": 'id',
            "root['orders'][*]['lines']": 'sku',
        }) == {}
