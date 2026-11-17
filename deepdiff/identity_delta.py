"""
Build the JSON-safe identity patch consumed by :mod:`deepdiff.identity_engine`.

The patch is derived from the tree view of an identity mode DeepDiff run. It
contains:

* ``declarations``: the normalized ``iterable_keys`` mapping;
* ``lists``: per identity list the before/after identity sequences plus only
  the records that were actually added or removed (full unchanged records are
  never copied);
* ``changes``: leaf level field changes addressed by stable path;
* ``version``: a format marker used to reject corrupt/unknown metadata.

Only touched subtrees are stored, so a 1 KiB payload on an unchanged record
never enters the patch.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from deepdiff.identity import (
    identity_to_json, registry_to_json, IdentityRegistry,
)
from deepdiff.identity_engine import (
    VALUE_CHANGED, TYPE_CHANGED, DICT_ADDED, DICT_REMOVED,
    ATTR_ADDED, ATTR_REMOVED, SET_ITEM_ADDED, SET_ITEM_REMOVED,
)

IDENTITY_PATCH_VERSION = 1
IDENTITY_PATCH_KEY = '_identity_patch'

# Tree report key -> internal change action.
_TREE_ACTION_MAP = {
    'values_changed': VALUE_CHANGED,
    'type_changes': TYPE_CHANGED,
    'dictionary_item_added': DICT_ADDED,
    'dictionary_item_removed': DICT_REMOVED,
    'attribute_added': ATTR_ADDED,
    'attribute_removed': ATTR_REMOVED,
    'set_item_added': SET_ITEM_ADDED,
    'set_item_removed': SET_ITEM_REMOVED,
}


def build_identity_patch(diff) -> Dict[str, Any]:
    """
    Build a JSON-safe identity patch from an identity-mode :class:`DeepDiff`
    instance (any view).
    """
    registry: IdentityRegistry = diff.identity_registry
    tree = diff.tree

    lists: Dict[str, Dict[str, Any]] = {}
    changes: List[Dict[str, Any]] = []

    # ---- structural record changes --------------------------------------
    for report_type, side in (
        ('iterable_identity_item_removed', 'removed'),
        ('iterable_identity_item_added', 'added'),
    ):
        for level in tree.get(report_type, []):
            list_path = level.up.stable_path() if hasattr(level.up, 'stable_path') else None
            if list_path is None:
                list_path = level.up.path(force='fake')
            descriptor = lists.setdefault(list_path, {
                'added': [],
                'removed': [],
                'before_identity': None,
                'after_identity': None,
            })
            identity = level.identity
            entry = {'identity': identity_to_json(identity)}
            if side == 'removed':
                entry['old_value'] = level.t1
            else:
                entry['new_value'] = level.t2
            descriptor[side].append(entry)

    # ---- before/after identity sequences --------------------------------
    # Recover all sequences in a single linear walk so that pure moves (which
    # produce no add/remove entry) still carry ordering data and nested lists
    # never trigger a repeated tree traversal.
    all_list_paths = set(lists.keys())
    for level in tree.get('iterable_identity_moved', []):
        list_path = level.up.stable_path() if hasattr(level.up, 'stable_path') else None
        if list_path is None:
            list_path = level.up.path(force='fake')
        all_list_paths.add(list_path)
    sequences = collect_identity_sequences(diff, all_list_paths)
    for list_path, descriptor in lists.items():
        descriptor['before_identity'], descriptor['after_identity'] = sequences[list_path]

    # Lists that only experienced moves (no add/remove) still need descriptors
    # carrying their identity sequences.
    for list_path in all_list_paths:
        if list_path not in lists:
            before_seq, after_seq = sequences[list_path]
            lists[list_path] = {
                'added': [],
                'removed': [],
                'before_identity': before_seq,
                'after_identity': after_seq,
            }

    # ---- leaf field changes ---------------------------------------------
    for report_type, action in _TREE_ACTION_MAP.items():
        for level in tree.get(report_type, []):
            if level.identity is None:
                # Non identity changes are handled by the regular delta path.
                continue
            stable = level.stable_path()
            entry: Dict[str, Any] = {'action': action, 'path': stable}
            if action in (VALUE_CHANGED, TYPE_CHANGED):
                entry['old_value'] = level.t1
                entry['new_value'] = level.t2
                if action == TYPE_CHANGED:
                    entry['old_type'] = type(level.t1).__name__
                    entry['new_type'] = type(level.t2).__name__
            elif action in (DICT_ADDED, ATTR_ADDED, SET_ITEM_ADDED):
                entry['new_value'] = level.t2
            else:
                entry['old_value'] = level.t1
            changes.append(entry)

    return {
        'version': IDENTITY_PATCH_VERSION,
        'declarations': registry_to_json(registry),
        'lists': lists,
        'changes': changes,
    }


def collect_identity_sequences(diff, list_paths):
    """
    Compute the before/after identity sequences for every declared list path
    in a single linear walk of each snapshot.

    Nested list paths (e.g. ``root['orders'][int=1]['lines']``) are reached
    while traversing their parent records, so each physical list is visited
    exactly once per side instead of re-walking the tree once per path.
    """
    from deepdiff.identity import (
        identity_of, identity_to_json, ABSENT,
        format_stable_path,
    )
    registry = diff.identity_registry
    wanted = set(list_paths)
    result: Dict[str, Tuple[Any, Any]] = {}

    def walk_snapshot(snapshot, side):
        sequences: Dict[str, Any] = {}

        def recurse(obj, elements):
            # If the current element chain resolves to a declared list, record
            # its identity sequence.
            if isinstance(obj, list):
                lookup = tuple(
                    ('*', 'GET') if kind == 'identity' else (value, 'GET')
                    for kind, value in elements
                )
                spec = registry.keys_for(lookup)
                if spec is not None:
                    stable = format_stable_path(elements)
                    if stable in wanted:
                        sequences[stable] = [
                            identity_to_json(identity_of(record, spec, stable, side, index))
                            for index, record in enumerate(obj)
                        ]
            if isinstance(obj, dict):
                for key, value in obj.items():
                    recurse(value, elements + [('field', key)])
            elif isinstance(obj, list):
                # To descend into records we need the spec that applies to
                # this list.
                lookup = tuple(
                    ('*', 'GET') if kind == 'identity' else (value, 'GET')
                    for kind, value in elements
                )
                spec = registry.keys_for(lookup)
                for index, record in enumerate(obj):
                    if spec is not None and isinstance(record, dict):
                        identity = identity_of(record, spec, format_stable_path(elements),
                                               side, index)
                        recurse(record, elements + [('identity', identity)])
                    elif isinstance(record, (dict, list)):
                        recurse(record, elements)

        recurse(snapshot, [])
        return sequences

    before_sequences = walk_snapshot(diff.t1, 't1')
    after_sequences = walk_snapshot(diff.t2, 't2')
    for path in wanted:
        result[path] = (
            before_sequences.get(path, ABSENT),
            after_sequences.get(path, ABSENT),
        )
    return result


def _sequences_for_path(diff, list_stable_path: str):
    """Single-path convenience wrapper around :func:`collect_identity_sequences`."""
    return collect_identity_sequences(diff, [list_stable_path])[list_stable_path]


def reverse_identity_patch(patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the reversed identity patch: before/after roles swap, additions
    become removals and vice versa, and field changes flip old/new values.
    """
    reversed_lists: Dict[str, Dict[str, Any]] = {}
    for list_path, descriptor in patch['lists'].items():
        reversed_lists[list_path] = {
            'added': [
                {'identity': entry['identity'], 'new_value': entry['old_value']}
                for entry in descriptor.get('removed', [])
            ],
            'removed': [
                {'identity': entry['identity'], 'old_value': entry['new_value']}
                for entry in descriptor.get('added', [])
            ],
            'before_identity': descriptor.get('after_identity'),
            'after_identity': descriptor.get('before_identity'),
        }
    reversed_changes = []
    for change in patch.get('changes', []):
        action = change['action']
        reversed_action = {
            VALUE_CHANGED: VALUE_CHANGED,
            TYPE_CHANGED: TYPE_CHANGED,
            DICT_ADDED: DICT_REMOVED,
            DICT_REMOVED: DICT_ADDED,
            ATTR_ADDED: ATTR_REMOVED,
            ATTR_REMOVED: ATTR_ADDED,
            SET_ITEM_ADDED: SET_ITEM_REMOVED,
            SET_ITEM_REMOVED: SET_ITEM_ADDED,
        }[action]
        entry = {'action': reversed_action, 'path': change['path']}
        if action in (VALUE_CHANGED, TYPE_CHANGED):
            entry['old_value'] = change['new_value']
            entry['new_value'] = change['old_value']
            if action == TYPE_CHANGED:
                entry['old_type'] = change['new_type']
                entry['new_type'] = change['old_type']
        elif reversed_action in (DICT_ADDED, ATTR_ADDED, SET_ITEM_ADDED):
            entry['new_value'] = change['old_value']
        else:
            entry['old_value'] = change['new_value']
        reversed_changes.append(entry)
    return {
        'version': patch['version'],
        'declarations': patch['declarations'],
        'lists': reversed_lists,
        'changes': reversed_changes,
    }
