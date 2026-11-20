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
    PLAIN_LISTS_KEY,
)

IDENTITY_PATCH_VERSION = 2
IDENTITY_PATCH_VERSION_LEGACY = 1
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

    # ---- ordinary ordered lists with structural changes -----------------
    # Any undeclared list that gained, lost or reordered elements is stored
    # once as a whole replaced subtree (old snapshot + new snapshot) instead
    # of index based opcodes. Leaf value changes that happen *inside* an
    # unaffected list keep their normal, index addressed entries.
    plain_lists, structurally_changed_plain_paths = collect_plain_list_replacements(diff)

    # ---- leaf field changes ---------------------------------------------
    # Every leaf change the tree view carries is captured here, including
    # changes on ordinary fields/lists that have no identity ancestor (for
    # example ``root['batch']``) and changes inside an ordinary ordered
    # sub-list of a record (for example
    # ``root['records'][int=1]['values'][0]``). A mixed diff therefore keeps
    # one complete change set instead of dropping the plain half on the floor.
    from deepdiff.identity import IDENTITY_PATH_SEGMENT_RE
    for report_type, action in _TREE_ACTION_MAP.items():
        for level in tree.get(report_type, []):
            stable = level.stable_path()
            if stable is None:
                # Unrepresentable levels are still exposed through the regular
                # delta dictionaries for the classic engine.
                continue
            if _is_inside_structurally_changed_plain_list(stable, structurally_changed_plain_paths):
                # The enclosing plain list is already carried as a replaced
                # subtree; its per-index edits must not be replayed twice.
                continue
            entry: Dict[str, Any] = {'action': action, 'path': stable}
            entry['identity_aware'] = bool(IDENTITY_PATH_SEGMENT_RE.search(stable))
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

    patch = {
        'version': IDENTITY_PATCH_VERSION,
        'declarations': registry_to_json(registry),
        'lists': lists,
        'changes': changes,
    }
    if plain_lists:
        patch[PLAIN_LISTS_KEY] = plain_lists
    return patch


def collect_plain_list_replacements(diff):
    """
    Find every ordinary (non declared) list whose structure changed
    (insertion, deletion or reorder) and return:

    * a list of ``{'path', 'old_value', 'new_value', 'identity_aware'}``
      entries carrying the full before/after subtree of each such list;
    * the set of stable path prefixes whose descendants are fully covered
      by a subtree replacement and therefore must not emit leaf edits.
    """
    from copy import deepcopy
    from deepdiff.identity import (
        stable_path_elements, format_stable_path, path_has_identity,
    )
    registry = diff.identity_registry
    tree = diff.tree
    # Map the stable path of each structurally changed ordinary list
    # container to its concrete t1/t2 values, straight from the tree levels,
    # so no identity based re-navigation is required.
    containers: Dict[str, Tuple[Any, Any]] = {}

    def _is_declared_list(level):
        chain = diff._element_chain(level)
        return registry.keys_for(chain) is not None

    # Tree level structural reports for ordinary lists.
    for report_type in ('iterable_item_added', 'iterable_item_removed', 'iterable_item_moved'):
        for level in tree.get(report_type, []):
            container = level.up
            if isinstance(container.t1, list) and not _is_declared_list(container):
                stable = container.stable_path()
                if stable is not None:
                    containers.setdefault(stable, (container.t1, container.t2))
    # difflib opcode paths (basic hashable element lists) are regular paths;
    # only keep those whose container is not a declared identity list.
    for opcode_path in getattr(diff, '_iterable_opcodes', {}).keys():
        elements = _regular_path_to_stable_elements(opcode_path)
        if elements is None:
            continue
        lookup = tuple(
            ('*', 'GET') if kind == 'identity' else (value, 'GET')
            for kind, value in elements
        )
        if registry.keys_for(lookup) is not None:
            continue
        stable = format_stable_path(elements)
        if stable in containers:
            continue
        old_value, old_found = _extract_regular(diff.t1, elements)
        new_value, new_found = _extract_regular(diff.t2, elements)
        if old_found or new_found:
            containers[stable] = (old_value, new_value)

    entries = []
    covered_prefixes = set()
    for stable in sorted(containers):
        old_value, new_value = containers[stable]
        entries.append({
            'path': stable,
            'old_value': deepcopy(old_value),
            'new_value': deepcopy(new_value),
            'identity_aware': path_has_identity(stable_path_elements(stable)),
        })
        covered_prefixes.add(stable)
    return entries, covered_prefixes


def _regular_path_to_stable_elements(path: str):
    """Parse a regular DeepDiff path into field/index descriptor tuples."""
    from deepdiff.path import _path_to_elements, GET
    try:
        raw = _path_to_elements(path, root_element=None)
    except Exception:
        return None
    elements = []
    for value, action in raw:
        if action != GET:
            return None
        if isinstance(value, int) and not isinstance(value, bool):
            elements.append(('index', value))
        elif isinstance(value, str):
            elements.append(('field', value))
        else:
            return None
    return elements


def _extract_regular(root, elements):
    """
    Navigate a field/index-only descriptor chain over a concrete snapshot.

    Returns ``(value, found)`` so that a genuine ``None`` stored at the path
    stays distinguishable from a missing path.
    """
    current = root
    for kind, value in elements:
        if kind == 'field':
            if not isinstance(current, dict) or value not in current:
                return None, False
            current = current[value]
        else:
            if not isinstance(current, list) or value >= len(current):
                return None, False
            current = current[value]
    from copy import deepcopy
    return deepcopy(current), True


def _is_inside_structurally_changed_plain_list(stable_path: str, covered_prefixes) -> bool:
    for prefix in covered_prefixes:
        if stable_path == prefix or stable_path.startswith(prefix + '['):
            return True
    return False


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
                    ('*', 'GET') if kind in ('identity', 'index') else (value, 'GET')
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
                    ('*', 'GET') if kind in ('identity', 'index') else (value, 'GET')
                    for kind, value in elements
                )
                spec = registry.keys_for(lookup)
                for index, record in enumerate(obj):
                    if spec is not None and isinstance(record, dict):
                        identity = identity_of(record, spec, format_stable_path(elements),
                                               side, index)
                        recurse(record, elements + [('identity', identity)])
                    elif isinstance(record, (dict, list)):
                        # An ordinary (non declared) list: keep its physical
                        # layer in the element chain so declarations reached
                        # through a ``[*]`` wildcard (for example
                        # ``root['groups'][*]['items']``) still resolve.
                        recurse(record, elements + [('index', index)])

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
    # Ordinary ordered list subtrees simply swap their old/new snapshots.
    reversed_plain_lists = [
        {
            'path': entry['path'],
            'old_value': entry['new_value'],
            'new_value': entry['old_value'],
            'identity_aware': entry.get('identity_aware', True),
        }
        for entry in patch.get(PLAIN_LISTS_KEY, [])
    ]
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
        from deepdiff.identity import path_has_identity as _path_has_identity, stable_path_elements as _spe
        entry['identity_aware'] = change.get('identity_aware',
                                             _path_has_identity(_spe(change['path'])))
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
    reversed_patch = {
        'version': patch['version'],
        'declarations': patch['declarations'],
        'lists': reversed_lists,
        'changes': reversed_changes,
    }
    if PLAIN_LISTS_KEY in patch:
        reversed_patch[PLAIN_LISTS_KEY] = reversed_plain_lists
    return reversed_patch
