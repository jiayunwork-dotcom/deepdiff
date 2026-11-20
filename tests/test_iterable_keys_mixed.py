"""
Tests for mixed ``iterable_keys`` and ordinary path handling.

Every expected tree, change fact and conflict path in this module is written
by hand from small literal snapshots; no identity locator under test is used
to generate the expected values.

The scenarios cover:

* the two originally reported regressions (plain root field dropped from the
  delta; ordinary ordered sub-list wrongly treated as a record list);
* root level field adds/removes, type changes and absent vs ``None``;
* ordinary list insert/delete/reorder with duplicate values;
* two declared identity layers with an ordinary list in between;
* a declared identity sub-list reached through an ordinary list ``[*]``;
* an outer identity record moved while an inner edit stays attached to it;
* a single conflict inside an otherwise applicable mixed delta;
* forward/reverse idempotency and delta reuse after a failed application;
* JSON and flat-dict round trips plus legacy and corrupt metadata handling.
"""
import copy
import json
import os
import subprocess
import sys

import pytest

from deepdiff import DeepDiff, Delta, DeltaConflictError
from deepdiff.identity import NotARecordListError, IdentityError
from deepdiff.serialization import json_dumps, json_loads


RECORDS_KEYS = {"root['records']": 'id'}


# ---------------------------------------------------------------------------
# Hand written expected facts
# ---------------------------------------------------------------------------

class TestReportedRegressions:
    def test_plain_root_field_travels_through_delta_and_rebase(self):
        before = {"records": [{"id": 1, "v": 10}], "batch": 1}
        after = {"records": [{"id": 1, "v": 11}], "batch": 2}
        diff = DeepDiff(before, after, iterable_keys=RECORDS_KEYS)

        # Hand written diff facts: exactly two leaf changes, one identity
        # located and one plain root field.
        assert set(diff['values_changed']) == {
            "root['records'][int=1]['v']",
            "root['batch']",
        }
        assert diff['values_changed']["root['batch']"] == {
            'old_value': 1, 'new_value': 2}

        delta = Delta(diff, bidirectional=True)
        assert delta._identity_patch['version'] == 2

        assert (delta + copy.deepcopy(before)) == after
        assert (copy.deepcopy(after) - delta) == before
        assert delta.apply_rebased(copy.deepcopy(before)) == after
        assert delta.apply_rebased(copy.deepcopy(after), reverse=True) == before

    def test_ordinary_sublist_keeps_ordered_semantics(self):
        before = {"records": [{"id": 1, "values": [1, 2]}]}
        after = {"records": [{"id": 1, "values": [0, 1, 2]}]}
        # Must not raise NotARecordListError any more.
        diff = DeepDiff(before, after, iterable_keys=RECORDS_KEYS)
        # The ordinary list reports a classic ordered insertion, not an
        # identity report.
        assert "root['records'][int=1]['values'][0]" in diff['iterable_item_added']
        assert 'iterable_identity_item_added' not in diff

        delta = Delta(diff, bidirectional=True)
        # The structural change is stored once as a replaced subtree.
        plain = delta._identity_patch['plain_lists']
        assert len(plain) == 1
        assert plain[0]['path'] == "root['records'][int=1]['values']"
        assert plain[0]['old_value'] == [1, 2]
        assert plain[0]['new_value'] == [0, 1, 2]

        assert (delta + copy.deepcopy(before)) == after
        assert (copy.deepcopy(after) - delta) == before
        assert delta.apply_rebased(copy.deepcopy(before)) == after

    def test_declaration_does_not_leak_into_child_lists(self):
        # The parent declaration must not be inherited by an undeclared
        # sibling/child list even when its elements are not dicts.
        before = {"records": [{"id": 1, "values": ["a", "b"]}], "plain": [1, 2]}
        after = {"records": [{"id": 1, "values": ["a", "b", "c"]}], "plain": [2, 1]}
        diff = DeepDiff(before, after, iterable_keys=RECORDS_KEYS)
        assert 'iterable_identity_item_added' not in diff
        delta = Delta(diff, bidirectional=True)
        assert (delta + copy.deepcopy(before)) == after
        assert (copy.deepcopy(after) - delta) == before


class TestTreeAndTextViewAgreement:
    def test_text_and_tree_carry_the_same_changes(self):
        before = {"records": [{"id": 1, "v": 10, "values": [1, 2]}], "batch": 1}
        after = {"records": [{"id": 1, "v": 11, "values": [2, 1]}], "batch": 2}
        text = DeepDiff(before, after, iterable_keys=RECORDS_KEYS)
        tree = DeepDiff(before, after, view='tree', iterable_keys=RECORDS_KEYS)

        text_paths = {
            path for report, bucket in text.items()
            if isinstance(bucket, dict) and report != 'iterable_identity_moved'
            for path in bucket
        }
        tree_paths = {
            level.stable_path()
            for report, levels in tree.items()
            if report != 'iterable_identity_moved'
            for level in levels
            if level.stable_path() is not None
        }
        # Every tree level has a matching text entry.
        assert tree_paths <= text_paths
        # The plain root field is present in both views.
        assert "root['batch']" in text_paths
        assert any(level.stable_path() == "root['batch']"
                   for level in tree['values_changed'])

    def test_plain_reorder_is_not_marked_as_identity_move(self):
        before = {"records": [{"id": 1, "values": [1, 2, 3]}]}
        after = {"records": [{"id": 1, "values": [3, 2, 1]}]}
        diff = DeepDiff(before, after, iterable_keys=RECORDS_KEYS)
        assert 'iterable_identity_moved' not in diff
        assert 'iterable_identity_item_added' not in diff
        # Ordinary ordered comparison reports the position based value edits.
        assert "root['records'][int=1]['values'][0]" in diff['values_changed']


class TestRootFieldAndScalarChanges:
    def test_add_remove_type_change_and_absent_vs_null(self):
        before = {"a": 1, "n": None, "gone": "x", "records": [{"id": 1}]}
        after = {"a": "1", "n": 1, "added": None, "records": [{"id": 1}]}
        diff = DeepDiff(before, after, iterable_keys=RECORDS_KEYS)

        assert "root['a']" in diff['type_changes']
        assert "root['n']" in diff['type_changes']
        assert "root['added']" in diff['dictionary_item_added']
        assert "root['gone']" in diff['dictionary_item_removed']

        delta = Delta(diff, bidirectional=True)
        assert (delta + copy.deepcopy(before)) == after
        assert (copy.deepcopy(after) - delta) == before
        assert delta.apply_rebased(copy.deepcopy(before)) == after
        # reverse rebase lands back on the exact original snapshot
        assert delta.apply_rebased(copy.deepcopy(after), reverse=True) == before

    def test_unrelated_sibling_branches_are_preserved(self):
        before = {"records": [{"id": 1, "v": 1}], "sibling": {"keep": [1, {"k": 2}]}}
        after = {"records": [{"id": 1, "v": 2}], "sibling": {"keep": [1, {"k": 2}]}}
        delta = Delta(DeepDiff(before, after, iterable_keys=RECORDS_KEYS),
                      bidirectional=True)
        current = copy.deepcopy(before)
        current['sibling']['keep'].append("local")
        result = delta.apply_rebased(current)
        assert result['sibling'] == {"keep": [1, {"k": 2}, "local"]}
        assert result['records'] == after['records']


class TestPlainListSemantics:
    def test_insert_delete_and_duplicate_values(self):
        before = {"xs": [1, 1, 2, 3]}
        after = {"xs": [1, 2, 2, 3, 4]}
        diff = DeepDiff(before, after, iterable_keys={})
        delta = Delta(diff, bidirectional=True)
        assert (delta + copy.deepcopy(before)) == after
        assert (copy.deepcopy(after) - delta) == before

    def test_structurally_changed_plain_list_is_verified_as_subtree(self):
        before = {"records": [{"id": 1, "values": [1, 2]}]}
        after = {"records": [{"id": 1, "values": [0, 1, 2]}]}
        delta = Delta(DeepDiff(before, after, iterable_keys=RECORDS_KEYS),
                      bidirectional=True)

        # Current list already equals the target -> accepted (idempotent).
        target_current = {"records": [{"id": 1, "values": [0, 1, 2]}]}
        assert delta.apply_rebased(copy.deepcopy(target_current)) == target_current

        # Current list equals the recorded old list -> updated.
        assert delta.apply_rebased(copy.deepcopy(before)) == after

        # Any third content conflicts instead of writing at drifted indexes.
        drifted = {"records": [{"id": 1, "values": [9, 9]}]}
        with pytest.raises(DeltaConflictError) as exc:
            delta.apply_rebased(copy.deepcopy(drifted))
        assert exc.value.kind == 'value_mismatch'
        assert exc.value.stable_path == "root['records'][int=1]['values']"

    def test_leaf_edit_inside_unchosen_structure_plain_list(self):
        # A plain list whose length/order did not change keeps index addressed
        # leaf edits instead of a whole subtree replacement.
        before = {"records": [{"id": 1, "vals": [10, 20, 30]}]}
        after = {"records": [{"id": 1, "vals": [10, 21, 30]}]}
        diff = DeepDiff(before, after, iterable_keys=RECORDS_KEYS)
        delta = Delta(diff, bidirectional=True)
        assert 'plain_lists' not in delta._identity_patch
        assert (delta + copy.deepcopy(before)) == after
        assert (copy.deepcopy(after) - delta) == before


class TestNestedIdentityLayers:
    def test_two_identity_layers_with_plain_list_between(self):
        before = {"orders": [{
            "id": 1, "tags": [9, 8],
            "lines": [{"sku": "x", "q": 1}],
        }]}
        after = {"orders": [{
            "id": 1, "tags": [9, 8, 7],
            "lines": [{"sku": "x", "q": 2}, {"sku": "y", "q": 3}],
        }]}
        keys = {
            "root['orders']": 'id',
            "root['orders'][*]['lines']": ('sku',),
        }
        diff = DeepDiff(before, after, iterable_keys=keys)
        delta = Delta(diff, bidirectional=True)
        assert (delta + copy.deepcopy(before)) == after
        assert (copy.deepcopy(after) - delta) == before
        assert delta.apply_rebased(copy.deepcopy(before)) == after
        # The tags list is ordinary; it travels as a replaced subtree.
        paths = {entry['path'] for entry in delta._identity_patch.get('plain_lists', [])}
        assert "root['orders'][int=1]['tags']" in paths

    def test_identity_sublist_through_ordinary_list_wildcard(self):
        before = {"groups": [
            {"items": [{"id": 1, "v": 1}]},
            {"items": [{"id": 2, "v": 2}]},
        ]}
        after = {"groups": [
            {"items": [{"id": 1, "v": 10}]},
            {"items": [{"id": 2, "v": 2}, {"id": 3, "v": 3}]},
        ]}
        keys = {"root['groups'][*]['items']": 'id'}
        diff = DeepDiff(before, after, iterable_keys=keys)
        delta = Delta(diff, bidirectional=True)
        assert (delta + copy.deepcopy(before)) == after
        assert (copy.deepcopy(after) - delta) == before
        assert delta.apply_rebased(copy.deepcopy(before)) == after


class TestMovesAndConflicts:
    def _move_delta(self):
        before = {"records": [
            {"id": 1, "v": 1, "child": {"x": 1}},
            {"id": 2, "v": 2, "child": {"x": 2}},
        ]}
        after = {"records": [
            {"id": 2, "v": 2, "child": {"x": 2}},
            {"id": 1, "v": 1, "child": {"x": 99}},
        ]}
        delta = Delta(
            DeepDiff(before, after, iterable_keys=RECORDS_KEYS),
            bidirectional=True)
        return before, after, delta

    def test_outer_move_keeps_inner_change_on_original_record(self):
        before, after, delta = self._move_delta()
        result = delta + copy.deepcopy(before)
        assert result == after
        record_one = next(r for r in result['records'] if r['id'] == 1)
        assert record_one['child']['x'] == 99
        assert (copy.deepcopy(after) - delta) == before

    def test_single_conflict_aborts_without_touching_inputs(self):
        before, after, delta = self._move_delta()
        divergent = copy.deepcopy(before)
        divergent['records'][0]['child']['x'] = 777
        snapshot = copy.deepcopy(divergent)
        with pytest.raises(DeltaConflictError):
            delta + divergent
        assert divergent == snapshot  # current untouched
        # The delta itself remains usable.
        assert (delta + copy.deepcopy(before)) == after

    def test_idempotent_in_both_directions(self):
        before, after, delta = self._move_delta()
        once = delta + copy.deepcopy(before)
        twice = delta + once
        assert once == twice == after
        back_once = copy.deepcopy(after) - delta
        back_twice = back_once - delta
        assert back_once == back_twice == before

    def test_rebase_idempotent_and_locally_edited_fields_preserved(self):
        before, after, delta = self._move_delta()
        current = copy.deepcopy(before)
        current['records'][1]['note'] = 'local'
        first = delta.apply_rebased(current)
        second = delta.apply_rebased(copy.deepcopy(first))
        assert first == second
        assert next(r for r in first['records'] if r['id'] == 2)['note'] == 'local'
        assert next(r for r in first['records'] if r['id'] == 1)['child']['x'] == 99

    def test_reverse_rebase_round_trip(self):
        before, after, delta = self._move_delta()
        # Rebase deliberately preserves the current display order, so compare
        # records by identity instead of requiring the original physical order.
        result = delta.apply_rebased(copy.deepcopy(after), reverse=True)
        by_id = {r['id']: r for r in result['records']}
        expected = {r['id']: r for r in before['records']}
        assert by_id == expected


class TestInputImmutability:
    def test_inputs_and_delta_are_not_mutated(self):
        before = {"records": [{"id": 1, "v": 10, "values": [1, 2]}], "batch": 1}
        after = {"records": [{"id": 1, "v": 11, "values": [0, 1, 2]}], "batch": 2}
        declaration = copy.deepcopy(RECORDS_KEYS)
        b_snapshot, a_snapshot = copy.deepcopy(before), copy.deepcopy(after)
        diff = DeepDiff(before, after, iterable_keys=declaration)
        delta = Delta(diff, bidirectional=True)
        patch_snapshot = copy.deepcopy(delta.diff)
        _ = delta + copy.deepcopy(before)
        _ = copy.deepcopy(after) - delta
        _ = delta.apply_rebased(copy.deepcopy(before))
        assert before == b_snapshot
        assert after == a_snapshot
        assert declaration == RECORDS_KEYS
        assert delta.diff == patch_snapshot


class TestSerialization:
    def _delta(self):
        before = {"records": [{"id": 1, "v": 10, "values": [1, 2]}], "batch": 1}
        after = {"records": [{"id": 1, "v": 11, "values": [0, 1, 2]}], "batch": 2}
        return before, after, Delta(
            DeepDiff(before, after, iterable_keys=RECORDS_KEYS),
            bidirectional=True)

    def test_json_roundtrip_keeps_every_change(self):
        before, after, delta = self._delta()
        rebuilt = Delta(json_loads(json_dumps(delta.diff)), bidirectional=True)
        assert (rebuilt + copy.deepcopy(before)) == after
        assert (copy.deepcopy(after) - rebuilt) == before
        assert rebuilt.apply_rebased(copy.deepcopy(before)) == after
        assert rebuilt._identity_patch['plain_lists'] == \
            delta._identity_patch['plain_lists']

    def test_flat_dict_roundtrip_keeps_every_change(self):
        before, after, delta = self._delta()
        flat = delta.to_flat_dicts()
        actions = {row['action'] for row in flat}
        assert 'identity_plain_list' in actions
        rebuilt = Delta(flat_dict_list=copy.deepcopy(flat), bidirectional=True)
        assert (rebuilt + copy.deepcopy(before)) == after
        assert (copy.deepcopy(after) - rebuilt) == before
        assert rebuilt.apply_rebased(copy.deepcopy(before)) == after
        assert rebuilt.apply_rebased(
            rebuilt.apply_rebased(copy.deepcopy(before))) == after

    def test_legacy_v1_patch_still_loads(self):
        legacy = {
            '_identity_patch': {
                'version': 1,
                'declarations': {"root['records']": ["id"]},
                'lists': {},
                'changes': [{
                    'action': 'values_changed',
                    'path': "root['records'][int=1]['v']",
                    'old_value': 10, 'new_value': 11,
                }],
            }
        }
        delta = Delta(legacy, bidirectional=True)
        base = {"records": [{"id": 1, "v": 10}]}
        assert (delta + copy.deepcopy(base)) == {"records": [{"id": 1, "v": 11}]}

    def test_legacy_v1_sidecar_applies_in_both_directions_and_flat(self):
        legacy = {
            '_identity_patch': {
                'version': 1,
                'declarations': {"root['records']": ["id"]},
                'lists': {},
                'changes': [{
                    'action': 'values_changed',
                    'path': "root['records'][int=1]['v']",
                    'old_value': 10, 'new_value': 11,
                }],
            },
            'values_changed': {
                "root['batch']": {'old_value': 1, 'new_value': 2},
            },
        }
        base = {"records": [{"id": 1, "v": 10}], "batch": 1}
        target = {"records": [{"id": 1, "v": 11}], "batch": 2}
        delta = Delta(legacy, bidirectional=True)
        assert (delta + copy.deepcopy(base)) == target
        assert (copy.deepcopy(target) - delta) == base
        rebuilt = Delta(flat_dict_list=copy.deepcopy(delta.to_flat_dicts()),
                        bidirectional=True)
        assert (rebuilt + copy.deepcopy(base)) == target
        assert (copy.deepcopy(target) - rebuilt) == base

    @pytest.mark.parametrize('mutator', [
        lambda d: d['_identity_patch'].__setitem__('version', 999),
        lambda d: d['_identity_patch']['plain_lists'][0].__delitem__('new_value'),
        lambda d: d['_identity_patch'].__setitem__('declarations', []),
    ])
    def test_corrupt_or_unknown_metadata_is_rejected(self, mutator):
        before, after, delta = self._delta()
        corrupt = copy.deepcopy(delta.diff)
        mutator(corrupt)
        with pytest.raises(Exception):
            Delta(corrupt, bidirectional=True)

    def test_flat_rows_without_metadata_are_rejected(self):
        before, after, delta = self._delta()
        flat = [row for row in delta.to_flat_dicts()
                if row['action'] != 'identity_metadata']
        with pytest.raises(Exception):
            Delta(flat_dict_list=flat, bidirectional=True)

    def test_stable_path_parsing_does_not_evaluate_text(self):
        from deepdiff.identity import stable_path_elements
        for evil in ["root[__import__('os')]", "root[1.2.3]", "root[x]",
                     "root[int=abc]", "root[weird=1]"]:
            with pytest.raises(IdentityError):
                stable_path_elements(evil)

    def test_fresh_process_json_roundtrip(self):
        before, after, delta = self._delta()
        payload = {
            'before': before,
            'after': after,
            'blob': json_dumps(delta.diff),
        }
        worker = os.path.join(os.path.dirname(__file__),
                              '_mixed_paths_subprocess_worker.py')
        payload_path = os.path.join(os.path.dirname(__file__),
                                    '_mixed_paths_payload.json')
        with open(payload_path, 'w') as handle:
            json.dump(payload, handle)
        try:
            completed = subprocess.run(
                [sys.executable, worker, payload_path],
                capture_output=True, text=True,
                env={k: v for k, v in os.environ.items()
                     if k != 'PYTHONNOUSERSITE'},
            )
            assert completed.returncode == 0, completed.stderr
        finally:
            if os.path.exists(payload_path):
                os.remove(payload_path)


class TestValidationStillEnforced:
    def test_missing_duplicate_and_illegal_keys_still_rejected(self):
        from deepdiff.identity import IdentityKeyError, DuplicateIdentityError
        with pytest.raises(IdentityKeyError):
            DeepDiff({"r": [{}]}, {"r": [{"id": 1}]},
                     iterable_keys={"root['r']": 'id'})
        with pytest.raises(DuplicateIdentityError):
            DeepDiff({"r": [{"id": 1}, {"id": 1}]}, {"r": []},
                     iterable_keys={"root['r']": 'id'})
        with pytest.raises(IdentityKeyError):
            DeepDiff({"r": [{"id": True}]}, {"r": []},
                     iterable_keys={"root['r']": 'id'})

    def test_non_list_declared_target_still_rejected(self):
        with pytest.raises(NotARecordListError):
            DeepDiff({"r": {}}, {"r": {}},
                     iterable_keys={"root['r']": 'id'})
