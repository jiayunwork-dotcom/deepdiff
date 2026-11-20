"""
Application engine for identity aware deltas.

The engine consumes a JSON-safe identity patch (produced by
``deepdiff.identity_delta.build_identity_patch``) and applies it either:

* strictly (``Delta.__add__`` / ``__rsub__``): the input must be exactly the
  snapshot the delta was made from, otherwise a :class:`DeltaConflictError`
  is raised; or
* rebased (``Delta.apply_rebased``): records are located by identity, edits
  that still apply are replayed and divergent edits raise a conflict while
  unrelated local edits are preserved.

All public functions are non mutating: the input tree is deep copied before
anything is changed and a brand new object is returned. Identity indexes are
built once per physical list and reused for every change touching that list.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple

from deepdiff.identity import (
    Identity, IdentityRegistry, IdentityError,
    stable_path_elements, identity_from_json,
    registry_from_json, ABSENT,
)

# Internal action names used inside an identity patch.
VALUE_CHANGED = 'values_changed'
TYPE_CHANGED = 'type_changes'
DICT_ADDED = 'dictionary_item_added'
DICT_REMOVED = 'dictionary_item_removed'
ATTR_ADDED = 'attribute_added'
ATTR_REMOVED = 'attribute_removed'
SET_ITEM_ADDED = 'set_item_added'
SET_ITEM_REMOVED = 'set_item_removed'
# An ordinary (non declared) ordered list changed structurally: the whole
# list is replaced as one verified subtree instead of replaying index edits.
PLAIN_LIST_REPLACED = 'plain_list_replaced'

# Top level key of the per-list plain-subtree section of a version 2 patch.
PLAIN_LISTS_KEY = 'plain_lists'

CONFLICT_MISSING = 'missing'
CONFLICT_PRESENT = 'already_present'
CONFLICT_VALUE = 'value_mismatch'
CONFLICT_NOT_FOUND = 'identity_not_found'
CONFLICT_BAD_IDENTITY = 'bad_identity'
CONFLICT_DUPLICATE_IDENTITY = 'duplicate_identity'
CONFLICT_TYPE = 'type_mismatch'


class DeltaConflictError(ValueError):
    """
    Raised when an identity aware delta can not be applied because the target
    tree diverged from (or does not satisfy) what the patch expects.

    Attributes
    ----------
    stable_path:
        The identity based path of the conflicting location, or the list path
        when an identity itself could not be located.
    kind:
        One of the ``CONFLICT_*`` constants of this module.
    expected:
        The value the patch expected. ``ABSENT`` means the path was expected
        to be absent, which is deliberately distinguishable from ``None``.
    current:
        The value currently found. ``ABSENT`` means the path is absent.
    """

    def __init__(self, stable_path: str, kind: str, expected: Any, current: Any, message: Optional[str] = None):
        self.stable_path = stable_path
        self.kind = kind
        self.expected = expected
        self.current = current
        if message is None:
            message = (
                f"Delta conflict ({kind}) at {stable_path}: expected {_render(expected)} "
                f"but found {_render(current)}.")
        super().__init__(message)


def _render(value: Any) -> str:
    if value is ABSENT:
        return '<absent>'
    return repr(value)


class _IdentityMap:
    """Cached identity -> index mapping for one physical list instance."""

    __slots__ = ('key_spec', 'list_path', 'side', 'identity_to_index')

    def __init__(self, records: List[Any], key_spec: Tuple[str, ...], list_path: str, side: str):
        self.key_spec = key_spec
        self.list_path = list_path
        self.side = side
        mapping: Dict[Identity, int] = {}
        seen: Dict[Identity, int] = {}
        for index, record in enumerate(records):
            from deepdiff.identity import identity_of, IdentityError as _IdentityError
            try:
                identity = identity_of(record, key_spec, list_path, side, index)
            except _IdentityError as exc:
                # A malformed current tree (missing/illegal key) is a conflict,
                # not an internal validation error, during application.
                raise DeltaConflictError(
                    list_path, CONFLICT_BAD_IDENTITY, ABSENT, record, str(exc)) from None
            if identity in seen:
                raise DeltaConflictError(
                    list_path, CONFLICT_DUPLICATE_IDENTITY,
                    expected=identity, current=index,
                    message=(f"Duplicate identity {identity!r} in {list_path} on {side}: indexes "
                             f"{seen[identity]} and {index}."))
            mapping[identity] = index
            seen[identity] = index
        self.identity_to_index = mapping


class IdentityDeltaEngine:
    """
    Applies one identity patch to one target tree.

    A fresh engine is created for every application so that the cached
    identity maps never leak between calls and the delta stays reusable.
    """

    def __init__(self, patch: Mappingish, reverse: bool = False, rebased: bool = False):
        self.patch = patch
        self.rebase = rebased
        self.reverse = reverse
        try:
            self.registry: IdentityRegistry = registry_from_json(patch['declarations'])
        except IdentityError as exc:
            raise DeltaConflictError('root', CONFLICT_BAD_IDENTITY, ABSENT, ABSENT, str(exc)) from None
        # Per list descriptors; when reversing, the before/after roles and the
        # identity sequences are swapped by the builder already.
        self.lists: Dict[str, Dict[str, Any]] = patch['lists']
        self._map_cache: Dict[Tuple[str, int], _IdentityMap] = {}
        self._spec_cache: Dict[str, Tuple[str, ...]] = {}
        # Memoizes record identity per (list_path, object id) within one apply.
        self._identity_cache: Dict[Tuple[str, int], Identity] = {}
        # Number of identity lookups performed (for diagnostics/benchmarks).
        self.lookup_count = 0
        # Number of full list scans performed while building identity maps.
        self.scan_count = 0
        # Number of records inspected while building identity maps.
        self.scanned_records = 0

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def apply(self, root: Any) -> Any:
        # Preflight on the untouched input first: any conflict aborts before
        # we allocate a mutable copy, so a failed call never returns partial
        # results and never mutates the caller's object.
        self._preflight(root)
        result = deepcopy(root)
        result = self._execute(result)
        return result

    # ------------------------------------------------------------------
    # Identity index caching
    # ------------------------------------------------------------------

    def _key_spec_for_list(self, list_stable_path: str) -> Tuple[str, ...]:
        cached = self._spec_cache.get(list_stable_path)
        if cached is not None:
            return cached
        elements = stable_path_elements(list_stable_path)
        lookup_elements = tuple(
            ('*', 'GET') if kind == 'identity' else (value, 'GET')
            for kind, value in elements
        )
        spec = self.registry.keys_for(lookup_elements)
        if spec is None:  # pragma: no cover - builder always produces valid paths
            raise DeltaConflictError(
                list_stable_path, CONFLICT_BAD_IDENTITY, ABSENT, ABSENT,
                f"No iterable_keys declaration matches {list_stable_path}.")
        self._spec_cache[list_stable_path] = spec
        return spec

    def _identity_map(self, records: List[Any], list_path: str, side: str) -> _IdentityMap:
        cache_key = (list_path, id(records))
        cached = self._map_cache.get(cache_key)
        if cached is not None:
            return cached
        spec = self._key_spec_for_list(list_path)
        index = _IdentityMap(records, spec, list_path, side)
        self._map_cache[cache_key] = index
        self.scan_count += 1
        self.scanned_records += len(records)
        return index

    def _locate_record(self, container: List[Any], list_path: str, identity: Identity,
                       side: str, raise_if_missing: bool = True) -> Optional[int]:
        index_map = self._identity_map(container, list_path, side)
        self.lookup_count += 1
        index = index_map.identity_to_index.get(identity)
        if index is None and raise_if_missing:
            raise DeltaConflictError(
                list_path, CONFLICT_NOT_FOUND, expected=identity, current=ABSENT,
                message=(f"Record with identity {identity!r} declared at {list_path} could not be "
                         f"located on the current tree."))
        return index

    # ------------------------------------------------------------------
    # Generic navigation
    # ------------------------------------------------------------------

    def _navigate(self, root: Any, elements: List[Tuple[str, Any]], for_write: bool):
        """
        Walk ``elements`` (field / identity descriptors) from ``root``.

        Returns ``(parent, container_kind, key, current_value)`` where
        ``container_kind`` is ``'dict'`` or ``'list'``. Identity segments
        resolve through cached identity maps. When ``for_write`` is True the
        parent containers encountered are expected to exist (the engine only
        creates final leaf values, never whole intermediate branches).
        """
        current = root
        parent = None
        kind = None
        key = None
        for position, (seg_kind, seg_value) in enumerate(elements):
            if seg_kind == 'field':
                parent = current
                kind = 'dict'
                key = seg_value
                if isinstance(current, dict):
                    current = current.get(seg_value, ABSENT)
                elif hasattr(current, seg_value):
                    current = getattr(current, seg_value)
                else:
                    current = ABSENT
            elif seg_kind == 'index':
                # Physical position inside an ordinary ordered list. The
                # segment is only reached for mixed paths; the whole-list
                # subtree check in preflight guarantees the index still means
                # the same element at application time.
                list_path = _path_prefix(elements, position)
                if not isinstance(current, list):
                    raise DeltaConflictError(
                        list_path, CONFLICT_TYPE,
                        expected=list, current=type(current))
                parent = current
                kind = 'list'
                key = seg_value
                if seg_value < len(current):
                    current = current[seg_value]
                else:
                    current = ABSENT
            else:  # identity segment
                list_path = _path_prefix(elements, position)
                if not isinstance(current, list):
                    raise DeltaConflictError(
                        list_path, CONFLICT_TYPE,
                        expected=list, current=type(current))
                identity = seg_value
                idx = self._locate_record(current, list_path, identity, 'current',
                                          raise_if_missing=for_write)
                parent = current
                kind = 'list'
                key = idx
                if idx is None:
                    current = ABSENT
                    # Descending further is impossible; keep ABSENT all the way.
                    for remaining in elements[position + 1:]:
                        parent = current
                        kind = 'dict'
                        key = remaining[1]
                    break
                current = current[idx]
        return parent, kind, key, current

    # ------------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------------

    def _preflight(self, root: Any) -> None:
        # List structural changes first so record removals/additions are
        # verified before descendant edits.
        for list_path, descriptor in self.lists.items():
            self._preflight_list(root, list_path, descriptor)
        # Ordinary ordered lists that changed structurally are verified as a
        # whole subtree before any leaf edit is considered.
        for entry in self.patch.get(PLAIN_LISTS_KEY, []):
            self._preflight_plain_list(root, entry)
        # Field level changes (values_changed / type_changes / dict add/remove).
        for change in self.patch.get('changes', []):
            self._preflight_change(root, change)

    def _preflight_plain_list(self, root: Any, entry: Dict[str, Any]) -> None:
        elements = stable_path_elements(entry['path'])
        parent, kind, key, current = self._navigate(root, elements, for_write=False)
        old_value = entry['old_value']
        new_value = entry['new_value']
        if current is ABSENT:
            raise DeltaConflictError(entry['path'], CONFLICT_MISSING,
                                     expected=old_value, current=ABSENT)
        if not isinstance(current, list):
            raise DeltaConflictError(entry['path'], CONFLICT_TYPE,
                                     expected=list, current=type(current))
        if self.rebase and values_equal(current, new_value):
            return  # already applied -> accept
        if not values_equal(current, old_value):
            raise DeltaConflictError(
                entry['path'], CONFLICT_VALUE,
                expected=old_value, current=current,
                message=(f"Ordinary list at {entry['path']} diverged from the recorded subtree; "
                         f"refusing to replay index based edits onto a drifted list."))

    def _path_goes_through_missing_identity(self, root: Any, elements: List[Tuple[str, Any]]) -> bool:
        """True if an identity segment on this path cannot resolve in ``root``."""
        current = root
        for position, (seg_kind, seg_value) in enumerate(elements):
            if seg_kind == 'identity':
                if not isinstance(current, list):
                    return True
                list_path = _path_prefix(elements, position)
                # Reuse the cached map instead of scanning a second time.
                index_map = self._identity_map(current, list_path, 'current')
                if seg_value not in index_map.identity_to_index:
                    return True
                current = current[index_map.identity_to_index[seg_value]]
            elif seg_kind == 'index':
                if not isinstance(current, list) or seg_value >= len(current):
                    return False
                current = current[seg_value]
            else:
                if isinstance(current, dict):
                    if seg_value not in current:
                        return False
                    current = current[seg_value]
                else:
                    return False
        return False

    def _preflight_list(self, root: Any, list_path: str, descriptor: Dict[str, Any]) -> None:
        elements = stable_path_elements(list_path)
        parent, kind, key, container = self._navigate_to_container(root, elements)
        if container is ABSENT:
            # The whole list is absent on the current tree: every list change
            # is a conflict unless the patch itself created that list (which
            # would be represented as a dictionary/container change, not here).
            raise DeltaConflictError(list_path, CONFLICT_MISSING, ABSENT, ABSENT)
        if not isinstance(container, list):
            raise DeltaConflictError(list_path, CONFLICT_TYPE, expected=list, current=type(container))

        side = 'current'
        identity_map = self._identity_map(container, list_path, side)

        # Removed records: the whole subtree must still equal the recorded
        # old subtree (strict or rebase), or be gone already so a same
        # direction replay stays idempotent.
        for record in descriptor.get('removed', []):
            identity = identity_from_json(record['identity'])
            index = identity_map.identity_to_index.get(identity)
            old_value = record['old_value']
            if index is None:
                continue  # already removed -> idempotent replay
            current_value = container[index]
            if not values_equal(current_value, old_value):
                raise DeltaConflictError(
                    self._record_path(list_path, identity), CONFLICT_VALUE,
                    expected=old_value, current=current_value,
                    message=(f"Cannot remove record {identity!r} at {list_path}: its subtree diverged "
                             f"from the recorded old value."))
        # Added records.
        for record in descriptor.get('added', []):
            identity = identity_from_json(record['identity'])
            new_value = record['new_value']
            index = identity_map.identity_to_index.get(identity)
            if index is None:
                continue  # will be appended/inserted at execution time
            current_value = container[index]
            if values_equal(current_value, new_value):
                continue  # already applied -> idempotent replay
            raise DeltaConflictError(
                self._record_path(list_path, identity), CONFLICT_PRESENT,
                expected=ABSENT, current=current_value)
        # Matched records still present must keep resolvable identities; bad
        # keys / duplicates are raised above while building the map.

    def _preflight_change(self, root: Any, change: Dict[str, Any]) -> None:
        action = change['action']
        path = change['path']
        elements = stable_path_elements(path)
        parent, kind, key, current = self._navigate(root, elements, for_write=False)
        if action in (VALUE_CHANGED, TYPE_CHANGED):
            expected_old = change['old_value']
            new_value = change['new_value']
            if current is ABSENT:
                # Distinguish a missing record from a merely absent field by
                # checking whether the path navigates through an identity list.
                if self._path_goes_through_missing_identity(root, elements):
                    raise DeltaConflictError(path, CONFLICT_NOT_FOUND,
                                             expected=expected_old, current=ABSENT)
                raise DeltaConflictError(path, CONFLICT_MISSING, expected=expected_old, current=ABSENT)
            # Same direction application must be idempotent in both engines:
            # a leaf already holding the patch target is accepted, while a
            # third value still conflicts with the recorded old value.
            if values_equal(current, new_value):
                return
            if not values_equal(current, expected_old):
                raise DeltaConflictError(path, CONFLICT_VALUE, expected=expected_old, current=current)
        elif action in (DICT_ADDED, ATTR_ADDED, SET_ITEM_ADDED):
            new_value = change['new_value']
            if current is ABSENT:
                return
            if values_equal(current, new_value):
                return
            raise DeltaConflictError(path, CONFLICT_PRESENT, expected=ABSENT, current=current)
        elif action in (DICT_REMOVED, ATTR_REMOVED, SET_ITEM_REMOVED):
            expected_old = change['old_value']
            if current is ABSENT:
                return
            if not values_equal(current, expected_old):
                raise DeltaConflictError(path, CONFLICT_VALUE, expected=expected_old, current=current)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute(self, root: Any) -> Any:
        # Apply list changes, recording the final identity sequences so that
        # the resulting order matches the patch target (strict) or follows the
        # rebase ordering rules (rebase).
        replaced_root = root
        for list_path, descriptor in self.lists.items():
            outcome = self._execute_list(replaced_root, list_path, descriptor)
            if outcome is not None:
                replaced_root = outcome
        # Replace structurally changed ordinary ordered lists as whole
        # subtrees (preflight already proved the current subtree matches the
        # recorded old one, or the rebase target).
        for entry in self.patch.get(PLAIN_LISTS_KEY, []):
            outcome = self._execute_plain_list(replaced_root, entry)
            if outcome is not None:
                replaced_root = outcome
        for change in self.patch.get('changes', []):
            self._execute_change(replaced_root, change)
        return replaced_root

    def _execute_plain_list(self, root: Any, entry: Dict[str, Any]) -> Optional[Any]:
        elements = stable_path_elements(entry['path'])
        if not elements:
            # The ordinary list is the root object itself.
            return deepcopy(entry['new_value'])
        parent, kind, key, current = self._navigate(root, elements, for_write=True)
        if self.rebase and current is not ABSENT and values_equal(current, entry['new_value']):
            return None
        self._replace_container(parent, kind, key, deepcopy(entry['new_value']))
        return None

    def _execute_list(self, root: Any, list_path: str, descriptor: Dict[str, Any]) -> Optional[Any]:
        """Apply one identity list change; return a new root when the list is
        the root object, otherwise ``None``."""
        elements = stable_path_elements(list_path)
        parent, kind, key, container = self._navigate_to_container(root, elements)
        # Preflight guarantees the target exists and is a list at this point.
        assert isinstance(container, list)
        removed = {identity_from_json(r['identity']): r['old_value']
                   for r in descriptor.get('removed', [])}
        added = {identity_from_json(r['identity']): r['new_value']
                 for r in descriptor.get('added', [])}

        if self.rebase:
            # Surviving records keep their current relative order; new
            # identities are appended in the order the patch target records
            # them. Already-present added identities keep their position.
            new_list: List[Any] = []
            present_added = set()
            for record in container:
                identity = self._identity_of(record, list_path)
                if identity in removed:
                    continue  # deleted by the patch
                new_list.append(record)
                if identity in added:
                    present_added.add(identity)
            for record in descriptor.get('added', []):
                identity = identity_from_json(record['identity'])
                if identity not in present_added:
                    new_list.append(deepcopy(record['new_value']))
            # Patch-internal moves never reorder current in rebase mode: the
            # surviving order above already reflects the current display.
        else:
            # Strict mode: reconstruct exactly the target identity sequence,
            # preserving full target order including pure moves.
            target_sequence = [identity_from_json(payload)
                               for payload in descriptor['after_identity']]
            current_records = {}
            for record in container:
                current_records[self._identity_of(record, list_path)] = record
            new_list = []
            for identity in target_sequence:
                if identity in added:
                    new_list.append(deepcopy(added[identity]))
                else:
                    new_list.append(current_records[identity])
        if parent is None:
            # The identity list is the root object itself.
            return new_list
        self._replace_container(parent, kind, key, new_list)
        return None

    def _identity_of(self, record: Any, list_path: str) -> Identity:
        cache_key = (list_path, id(record))
        cached = self._identity_cache.get(cache_key)
        if cached is not None:
            return cached
        spec = self._key_spec_for_list(list_path)
        from deepdiff.identity import identity_of, IdentityError as _IdentityError
        try:
            result = identity_of(record, spec, list_path, 'current', -1)
        except _IdentityError as exc:
            raise DeltaConflictError(
                list_path, CONFLICT_BAD_IDENTITY, ABSENT, record, str(exc)) from None
        self._identity_cache[cache_key] = result
        return result

    def _execute_change(self, root: Any, change: Dict[str, Any]) -> None:
        action = change['action']
        path = change['path']
        elements = stable_path_elements(path)
        parent, kind, key, current = self._navigate(root, elements, for_write=True)
        if action in (VALUE_CHANGED, TYPE_CHANGED):
            new_value = change['new_value']
            if current is not ABSENT and values_equal(current, new_value):
                return
            self._set_value(parent, kind, key, new_value)
        elif action in (DICT_ADDED, ATTR_ADDED, SET_ITEM_ADDED):
            new_value = change['new_value']
            if current is not ABSENT and values_equal(current, new_value):
                return
            self._set_value(parent, kind, key, new_value)
        elif action in (DICT_REMOVED, ATTR_REMOVED, SET_ITEM_REMOVED):
            if current is ABSENT:
                return
            self._del_value(parent, kind, key)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _navigate_to_container(self, root: Any, elements: List[Tuple[str, Any]]):
        if not elements:
            return None, 'root', None, root
        parent, kind, key, current = self._navigate(root, elements, for_write=True)
        return parent, kind, key, current

    @staticmethod
    def _set_value(parent, kind, key, value) -> None:
        if kind == 'dict':
            if isinstance(parent, dict):
                parent[key] = value
            else:
                setattr(parent, key, value)
        else:  # list index (never None after preflight in strict mode)
            parent[key] = value

    @staticmethod
    def _del_value(parent, kind, key) -> None:
        if kind == 'dict':
            if isinstance(parent, dict):
                del parent[key]
            else:
                del parent.__dict__[key]
        else:
            del parent[key]

    @staticmethod
    def _replace_container(parent, kind, key, new_list) -> None:
        if parent is None:
            raise DeltaConflictError('root', CONFLICT_TYPE, ABSENT, ABSENT,
                                     'Replacing the root object is not supported inside an identity patch.')
        if kind == 'dict':
            if isinstance(parent, dict):
                parent[key] = new_list
            else:
                setattr(parent, key, new_list)
        else:
            parent[key] = new_list

    @staticmethod
    def _record_path(list_path: str, identity: Identity) -> str:
        from deepdiff.identity import format_stable_path
        return format_stable_path(stable_path_elements(list_path) + [('identity', identity)])


def _path_prefix(elements: Sequence[Tuple[str, Any]], count: int) -> str:
    from deepdiff.identity import format_stable_path
    return format_stable_path(list(elements[:count]))


def values_equal(left: Any, right: Any) -> bool:
    """
    Type sensitive equality matching DeepDiff's default semantics: values of
    different types are never equal (so ``1`` != ``True`` != ``1.0``), absent
    is only equal to absent.
    """
    if left is ABSENT or right is ABSENT:
        return left is right
    if type(left) is not type(right):
        return False
    return left == right


# Type alias only used for annotations above without importing Mapping at top
# in environments that trim the standard library (kept simple intentionally).
from collections.abc import Mapping as Mappingish  # noqa: E402
