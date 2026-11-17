"""
Stable identity support for declared record lists (``iterable_keys``).

This module is intentionally self contained: it only depends on the standard
library so that it can be reused by the diff engine, the delta/rebase engine
and the (de)serializers without import cycles.

Terminology
-----------

identity
    A tuple of ``(type, value)`` pairs built from the configured key fields of
    a record.  Both the Python type and the value participate in identity, so
    ``1`` and ``"1"`` are different identities.  Key values are restricted to
    non empty ``str`` or non ``bool`` ``int``.

stable path
    A path that locates an object independently of list indexes.  It uses the
    same dictionary field syntax as the regular DeepDiff paths but list
    elements are addressed by their identity instead of their physical
    position, for example ``root['orders'][id=7]['lines'][sku='a',site=1]``.

identity declaration
    A mapping of a list path (optionally containing ``[*]`` wildcards) to
    either a single key field name or a non empty tuple of key field names.

The module never mutates the objects it inspects.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from deepdiff.path import GET, stringify_element, _path_to_elements

# Matches a stable-path identity segment such as "[int=7]" or "[str='a',int=1]".
IDENTITY_PATH_SEGMENT_RE = re.compile(r"\[(?:str|int)=")

# A single component of an identity. Booleans are explicitly excluded because
# ``isinstance(True, int)`` is True in Python.
IdentityComponent = Tuple[str, Any]
Identity = Tuple[IdentityComponent, ...]
KeySpec = Tuple[str, ...]

# Sentinel meaning "the path is absent on this side" - distinct from ``None``
# which is a perfectly valid JSON scalar.
ABSENT = object()

WILDCARD = '*'
IDENTITY_PATH_MARKER = '__identity__'

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class IdentityError(ValueError):
    """Base class for all ``iterable_keys`` related validation errors."""


class DuplicateIdentityError(IdentityError):
    """Raised when the same identity occurs more than once in one list."""


class IdentityKeyError(IdentityError):
    """Raised when a record is missing a configured key field or the key value
    is not a non empty ``str`` / non ``bool`` ``int``."""


class NotARecordListError(IdentityError):
    """Raised when a declared path does not point at a list."""


class IterableKeysDeclarationError(IdentityError):
    """Raised when the ``iterable_keys`` declaration itself is malformed."""


# ---------------------------------------------------------------------------
# Declaration handling
# ---------------------------------------------------------------------------


def normalize_key_spec(keys: Any, list_path: str) -> KeySpec:
    """Normalize a user supplied key definition into a tuple of field names."""
    if isinstance(keys, str):
        if not keys:
            raise IterableKeysDeclarationError(
                f"The key field name declared for {list_path!r} must be a non empty string.")
        return (keys,)
    if isinstance(keys, (tuple, list)):
        if not keys:
            raise IterableKeysDeclarationError(
                f"The key field tuple declared for {list_path!r} must not be empty.")
        result = []
        for key in keys:
            if not isinstance(key, str) or not key:
                raise IterableKeysDeclarationError(
                    f"Every key field name declared for {list_path!r} must be a non empty string, got {key!r}.")
            result.append(key)
        return tuple(result)
    raise IterableKeysDeclarationError(
        f"The key definition for {list_path!r} must be a non empty string or a non empty tuple of "
        f"strings, got {keys!r}.")


def _parse_declaration_path(path: str) -> Tuple[Tuple[Any, ...], ...]:
    """Parse a declaration path into element tuples ``(value, is_wildcard)``."""
    if not isinstance(path, str) or not path.startswith('root'):
        raise IterableKeysDeclarationError(
            f"iterable_keys paths must be strings starting with 'root', got {path!r}.")
    elements = _path_to_elements(path, root_element=None)
    parsed = []
    for elem, action in elements:
        if action != GET:
            raise IterableKeysDeclarationError(
                f"iterable_keys only supports dictionary/list subscript paths; {path!r} is not valid.")
        parsed.append((elem, elem == WILDCARD))
    return tuple(parsed)


class IdentityRegistry:
    """
    Holds the normalized ``iterable_keys`` declarations and answers "which key
    fields apply to the list reached by this element chain?" questions while a
    tree is being walked.

    A declaration path may contain ``[*]`` which matches exactly one layer of
    list elements.
    """

    def __init__(self, iterable_keys: Optional[Mapping[str, Union[str, Tuple[str, ...], List[str]]]]):
        self.declarations: Dict[str, KeySpec] = {}
        # Parsed declarations: list of (elements, key_spec, original_path)
        self._parsed: List[Tuple[Tuple[Tuple[Any, bool], ...], KeySpec, str]] = []
        if iterable_keys is None:
            return
        if not isinstance(iterable_keys, Mapping):
            raise IterableKeysDeclarationError(
                "iterable_keys must be a dictionary mapping list paths to key field names.")
        for list_path, keys in iterable_keys.items():
            key_spec = normalize_key_spec(keys, list_path)
            parsed = _parse_declaration_path(list_path)
            self.declarations[list_path] = key_spec
            self._parsed.append((parsed, key_spec, list_path))

    def __bool__(self) -> bool:
        return bool(self._parsed)

    def keys_for(self, elements: Sequence[Tuple[Any, str]]) -> Optional[KeySpec]:
        """
        Given the element chain (``(value, action)`` tuples, no root element)
        leading to a container, return the key spec if that container is a
        declared identity list, otherwise ``None``.

        A declaration matches when its elements are a prefix of the supplied
        chain, so nested containers can resolve the declaration of an
        enclosing identity list before their own suffix has been walked.
        """
        if not self._parsed:
            return None
        best: Optional[Tuple[int, KeySpec]] = None
        for parsed, key_spec, original_path in self._parsed:
            if len(parsed) > len(elements):
                continue
            for (decl_elem, is_wildcard), (elem, action) in zip(parsed, elements):
                if action != GET:
                    break
                if not is_wildcard and decl_elem != elem:
                    break
            else:
                # Prefer the longest (most specific) matching declaration.
                if best is None or len(parsed) > best[0]:
                    best = (len(parsed), key_spec)
        return None if best is None else best[1]

    def items(self):
        return self.declarations.items()


# ---------------------------------------------------------------------------
# Tree validation (acyclic, finite floats, JSON-ish scalars)
# ---------------------------------------------------------------------------


def _is_valid_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def validate_tree(root: Any, side: str, registry: IdentityRegistry) -> None:
    """
    Validate the structural preconditions of identity mode:

    * only ``dict`` / ``list`` containers and JSON scalars,
    * no cyclic references,
    * every float is finite.

    ``side`` is ``"t1"`` or ``"t2"`` and is used in error messages.
    """
    seen: set = set()

    def walk(obj: Any, path_elements: Tuple[Any, ...]) -> None:
        if isinstance(obj, (dict, list)):
            obj_id = id(obj)
            if obj_id in seen:
                raise IdentityError(
                    f"Cyclic reference detected on {side} at {format_stable_prefix(path_elements)}.")
            seen.add(obj_id)
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if not isinstance(key, str):
                        raise IdentityError(
                            f"Identity mode only supports string dictionary keys; got {key!r} "
                            f"on {side} at {format_stable_prefix(path_elements)}.")
                    walk(value, path_elements + (('field', key),))
            else:
                for index, value in enumerate(obj):
                    walk(value, path_elements + (('index', index),))
        elif not _is_valid_scalar(obj):
            raise IdentityError(
                f"Identity mode supports dict, list and JSON scalars (str, int, float, bool, None) only; "
                f"got {type(obj).__name__} on {side} at {format_stable_prefix(path_elements)}.")

    walk(root, ())


def format_stable_prefix(elements: Sequence[Tuple[str, Any]]) -> str:
    """Best-effort human readable path for error messages."""
    out = ['root']
    for kind, value in elements:
        if kind == 'field':
            out.append('[' + stringify_element(value, quote_str="'{}'") + ']')
        elif kind == 'index':
            out.append(f'[{value}]')
        elif kind == 'identity':
            out.append(format_identity_segment(value))
    return ''.join(out)


# ---------------------------------------------------------------------------
# Identity extraction
# ---------------------------------------------------------------------------


def validate_key_value(value: Any, field: str, list_path: str, side: str, index: int) -> IdentityComponent:
    """Validate one key component and return its ``(type_name, value)`` pair."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise IdentityKeyError(
            f"Identity key field {field!r} of the record at {list_path}[{index}] on {side} must be a "
            f"non empty str or a non bool int, got {value!r} ({type(value).__name__}).")
    if isinstance(value, str) and not value:
        raise IdentityKeyError(
            f"Identity key field {field!r} of the record at {list_path}[{index}] on {side} must be a "
            f"non empty string.")
    return (type(value).__name__, value)


def identity_of(record: Any, key_spec: KeySpec, list_path: str, side: str, index: int) -> Identity:
    """Extract the identity tuple of a record, raising on missing/bad keys."""
    if not isinstance(record, dict):
        raise NotARecordListError(
            f"Every element of the identity list at {list_path} on {side} must be a dict, got "
            f"{type(record).__name__} at index {index}.")
    components = []
    for field in key_spec:
        if field not in record:
            raise IdentityKeyError(
                f"Record at {list_path}[{index}] on {side} is missing identity key field {field!r}.")
        components.append(validate_key_value(record[field], field, list_path, side, index))
    return tuple(components)


def format_identity_segment(identity: Identity) -> str:
    """Render an identity as the bracketed stable-path segment."""
    parts = []
    for type_name, value in identity:
        if type_name == 'str':
            rendered = stringify_element(value, quote_str="'{}'")
        else:
            rendered = repr(value)
        parts.append(f"{type_name}={rendered}")
    return '[' + ','.join(parts) + ']'


def parse_identity_segment(text: str) -> Identity:
    """
    Parse a stable-path identity segment such as ``[str='a',int=1]``.

    No ``eval`` is used: strings are quoted literals parsed with
    ``literal_eval`` (which only evaluates literals, never names/calls) and
    integers go through ``int`` after a strict digit check.
    """
    from ast import literal_eval
    if not (text.startswith('[') and text.endswith(']')):
        raise IdentityError(f"Malformed identity path segment: {text!r}")
    body = text[1:-1]
    if not body:
        raise IdentityError(f"Malformed identity path segment: {text!r}")
    components: List[IdentityComponent] = []
    for chunk in split_top_level_commas(body):
        if '=' not in chunk:
            raise IdentityError(f"Malformed identity path segment: {text!r}")
        type_name, _, raw = chunk.partition('=')
        type_name = type_name.strip()
        raw = raw.strip()
        if type_name == 'str':
            try:
                value = literal_eval(raw)
            except (ValueError, SyntaxError) as exc:
                raise IdentityError(f"Malformed string identity in {text!r}: {exc}") from None
            if not isinstance(value, str) or not value:
                raise IdentityError(f"String identities must be non empty strings in {text!r}.")
        elif type_name == 'int':
            digits = raw
            sign = ''
            if digits and digits[0] in '+-':
                sign, digits = digits[0], digits[1:]
            if not digits or not digits.isdigit():
                raise IdentityError(f"Malformed int identity in {text!r}.")
            value = int(sign + digits)
        else:
            raise IdentityError(f"Unsupported identity type {type_name!r} in {text!r}.")
        components.append((type_name, value))
    return tuple(components)


def split_top_level_commas(text: str) -> List[str]:
    """Split on commas that are not inside quoted strings."""
    parts: List[str] = []
    current: List[str] = []
    quote: Optional[str] = None
    for char in text:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
            current.append(char)
        elif char == ',':
            parts.append(''.join(current))
            current = []
        else:
            current.append(char)
    parts.append(''.join(current))
    return parts


# ---------------------------------------------------------------------------
# Identity maps (one per physical list instance)
# ---------------------------------------------------------------------------


class IdentityListIndex:
    """
    A cached ``identity -> physical position`` map for one concrete list
    instance, built in a single linear scan and reused for every change that
    touches the same list during one diff or one delta application.
    """

    __slots__ = ('list_path', 'key_spec', 'side', 'identity_to_index', 'duplicates')

    def __init__(self, records: Sequence[Any], key_spec: KeySpec, list_path: str, side: str):
        self.list_path = list_path
        self.key_spec = key_spec
        self.side = side
        self.identity_to_index: Dict[Identity, int] = {}
        self.duplicates: List[Tuple[Identity, int]] = []
        for index, record in enumerate(records):
            identity = identity_of(record, key_spec, list_path, side, index)
            if identity in self.identity_to_index:
                self.duplicates.append((identity, index))
            else:
                self.identity_to_index[identity] = index

    def check_duplicates(self) -> None:
        if self.duplicates:
            identity, index = self.duplicates[0]
            first_index = self.identity_to_index[identity]
            raise DuplicateIdentityError(
                f"Duplicate identity {render_identity(identity)} in the list at {self.list_path} on "
                f"{self.side}: indexes {first_index} and {index}.")

    def get(self, identity: Identity) -> Optional[int]:
        return self.identity_to_index.get(identity)


def render_identity(identity: Identity) -> str:
    return '(' + ', '.join(f"{type_name}={value!r}" for type_name, value in identity) + ')'


def ensure_record_list(obj: Any, list_path: str, side: str) -> None:
    if not isinstance(obj, list):
        raise NotARecordListError(
            f"The declared identity list path {list_path} on {side} does not point at a list but at "
            f"{type(obj).__name__}.")


def build_identity_index(records: Any, key_spec: KeySpec, list_path: str, side: str) -> IdentityListIndex:
    ensure_record_list(records, list_path, side)
    index = IdentityListIndex(records, key_spec, list_path, side)
    index.check_duplicates()
    return index


# ---------------------------------------------------------------------------
# Stable path parsing / formatting
# ---------------------------------------------------------------------------


def stable_path_elements(path: str) -> List[Tuple[str, Any]]:
    """
    Parse a stable path into a list of descriptive tuples:

    * ``('field', 'name')`` for dictionary fields
    * ``('identity', Identity)`` for identity list segments

    Raises ``IdentityError`` on malformed input. Never uses ``eval``.
    Results are cached because the same stable path is parsed repeatedly
    during a delta application.
    """
    cached = _STABLE_PATH_CACHE.get(path)
    if cached is not None:
        return list(cached)
    if not isinstance(path, str) or not path.startswith('root'):
        raise IdentityError(f"Stable paths must start with 'root', got {path!r}.")
    rest = path[4:]
    elements: List[Tuple[str, Any]] = []
    i = 0
    length = len(rest)
    while i < length:
        char = rest[i]
        if char == '.':
            raise IdentityError(
                f"Stable paths only support dictionary fields and identity lists; {path!r} is not valid.")
        if char != '[':
            raise IdentityError(f"Malformed stable path {path!r}.")
        end = rest.find(']', i + 1)
        if end == -1:
            raise IdentityError(f"Malformed stable path {path!r}.")
        segment = rest[i:end + 1]
        inner = segment[1:-1]
        if '=' in inner:
            elements.append(('identity', parse_identity_segment(segment)))
        else:
            # Plain dictionary field. Reuse the safe standard parser for one
            # bracket segment instead of inventing another literal parser.
            from ast import literal_eval
            try:
                value = literal_eval(inner)
            except (ValueError, SyntaxError) as exc:
                raise IdentityError(f"Malformed stable path {path!r}: {exc}") from None
            if not isinstance(value, str):
                raise IdentityError(
                    f"Non identity list segments in stable paths must be string dictionary fields; "
                    f"got {value!r} in {path!r}.")
            elements.append(('field', value))
        i = end + 1
    _STABLE_PATH_CACHE[path] = tuple(elements)
    return elements


_STABLE_PATH_CACHE: Dict[str, Tuple[Tuple[str, Any], ...]] = {}


def format_stable_path(elements: Sequence[Tuple[str, Any]]) -> str:
    out = ['root']
    for kind, value in elements:
        if kind == 'field':
            out.append('[' + stringify_element(value, quote_str="'{}'") + ']')
        elif kind == 'identity':
            out.append(format_identity_segment(value))
        else:  # pragma: no cover - defensive
            raise IdentityError(f"Unknown stable path element kind {kind!r}.")
    return ''.join(out)


def physical_elements_from_stable(elements: Sequence[Tuple[str, Any]]) -> Tuple[Any, ...]:
    """Convert stable elements to the element format used by the path module."""
    return tuple(value for kind, value in elements if kind in ('field', 'identity'))


def stable_path_to_flat_tokens(path: str) -> List[Any]:
    """
    Convert a stable path string into JSON-safe flat-dict tokens.

    Dictionary fields become plain strings; identity segments become tagged
    dictionaries of the form ``{"__identity__": [[type, value], ...]}`` so the
    distinction between a field and an identity survives a flat round trip
    without any use of ``eval``.
    """
    tokens: List[Any] = []
    for kind, value in stable_path_elements(path):
        if kind == 'field':
            tokens.append(value)
        else:
            tokens.append({IDENTITY_PATH_MARKER: identity_to_json(value)})
    return tokens


def flat_tokens_to_stable_path(tokens: Sequence[Any]) -> str:
    """Inverse of :func:`stable_path_to_flat_tokens`."""
    elements: List[Tuple[str, Any]] = []
    for token in tokens:
        if isinstance(token, Mapping) and IDENTITY_PATH_MARKER in token:
            elements.append(('identity', identity_from_json(token[IDENTITY_PATH_MARKER])))
        else:
            elements.append(('field', token))
    return format_stable_path(elements)


def identity_to_json(identity: Identity) -> List[List[Any]]:
    """JSON-safe representation of an identity."""
    return [[type_name, value] for type_name, value in identity]


def identity_from_json(payload: Sequence[Sequence[Any]]) -> Identity:
    result = []
    for pair in payload:
        type_name, value = pair
        if type_name == 'str':
            if not isinstance(value, str) or not value:
                raise IdentityError("Corrupt identity metadata: string identities must be non empty.")
        elif type_name == 'int':
            if isinstance(value, bool) or not isinstance(value, int):
                raise IdentityError("Corrupt identity metadata: int identities must be ints.")
        else:
            raise IdentityError(f"Corrupt identity metadata: unknown identity type {type_name!r}.")
        result.append((type_name, value))
    return tuple(result)


def registry_to_json(registry: IdentityRegistry) -> Dict[str, List[str]]:
    return {path: list(key_spec) for path, key_spec in registry.declarations.items()}


def registry_from_json(payload: Any) -> IdentityRegistry:
    if not isinstance(payload, Mapping):
        raise IdentityError("Corrupt identity metadata: the declaration must be a dictionary.")
    return IdentityRegistry({path: tuple(key_spec) for path, key_spec in payload.items()})
