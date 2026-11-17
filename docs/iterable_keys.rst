.. _iterable_keys_label:

Stable Identity for Lists of Records (``iterable_keys``)
=========================================================

When synchronizing nested JSON documents, list positions change frequently
even though the records themselves keep their identity. The
``iterable_keys`` parameter tells DeepDiff which field(s) uniquely identify
the records of a given list, so comparisons, deltas and rebases follow the
record identity instead of a physical index.

It is incompatible with ``ignore_order=True``, ``group_by`` and
``iterable_compare_func``; combining them raises a ``ValueError``.

Declaration
-----------

Pass a mapping of list paths to either a single key field name or a non empty
tuple of key field names:

.. code-block:: python

    iterable_keys = {
        "root['orders']": "id",
        "root['orders'][*]['lines']": ("sku", "site"),
    }

* Paths use the usual DeepDiff dictionary field syntax.
* ``[*]`` matches exactly one layer of elements inside its parent list.
* A declaration may point at a list that only exists on one side; a
  declaration whose path never appears on either side is simply unused.
* A path that does not resolve to a list raises
  :class:`deepdiff.identity.NotARecordListError`.

Identity rules
--------------

* List elements must be dictionaries.
* Every key component must be a non empty ``str`` or a non ``bool`` ``int``.
* Identity is determined by both type and value, so ``1`` and ``"1"`` are
  different identities.
* A missing or illegal key raises :class:`deepdiff.identity.IdentityKeyError`
  naming the side (``t1``/``t2``) and the concrete path.
* A duplicate identity inside one list raises
  :class:`deepdiff.identity.DuplicateIdentityError`.
* Identities of child lists are scoped to their parent record, so the same
  child identity under different parents never clashes.
* Inputs are restricted to acyclic trees of ``dict``/``list`` containers and
  JSON scalars, and every float must be finite. Inputs and the declaration
  object are never modified.

What the diff reports
---------------------

Records with the same identity are compared recursively. The text and tree
views use three additional report keys:

* ``iterable_identity_moved``: the record changed physical position. This is
  reported independently of content changes, so a record that moved *and*
  changed keeps both facts and is never expanded into a removal plus an
  addition.
* ``iterable_identity_item_added`` / ``iterable_identity_item_removed``:
  records whose identity only exists on one side. Changing a key field is
  therefore reported as removing the old identity and inserting the new one.

Stable paths address records by identity, for example
``root['orders'][int=1]['lines'][str='y',int=2]['qty']``, while every entry
also exposes the physical path on each side (``path_t1`` / ``path_t2``) so
both the stable identity and the two physical locations remain visible.
Lists without a declaration keep the existing ordered comparison.

In the tree view each affected :class:`~deepdiff.model.DiffLevel` exposes:

* ``level.stable_path()`` – identity based path;
* ``level.physical_path(use_t2=False|True)`` – physical path on each side;
* ``level.identity`` – the identity tuple.

Deltas and rebasing
-------------------

Identity deltas must be bidirectional:

.. code-block:: python

    delta = Delta(DeepDiff(before, after, iterable_keys=iterable_keys),
                  bidirectional=True)
    assert before + delta == after          # exact forward application
    assert after - delta == before          # exact reverse application

The identity patch is stored under the internal ``_identity_patch`` key inside
the delta dictionary. It only carries touched subtrees and the identity
sequences needed to restore order, so unchanged business payload (for example a
1 KiB string on every record) is never copied into the delta.

``Delta.apply_rebased(current, reverse=False)`` applies the patch onto a
divergent tree and returns a new object without mutating the input:

* records are located by identity, independent of their physical index;
* a value that still equals the patch old value is updated; one that already
  equals the new value is accepted; anything else raises
  :class:`deepdiff.DeltaConflictError`;
* additions apply when the path is absent (or already hold the added value);
  removals apply when absent or when the whole recorded subtree still matches;
* whole-record deletions, container replacements and type changes verify the
  entire subtree, so concurrent edits inside a deleted branch conflict rather
  than being silently dropped;
* surviving records keep their current relative order and patch-new records
  are appended in patch target order; the patch's own moves never reorder the
  current tree;
* replaying the same direction is idempotent and any single conflict aborts
  the whole application before a partial result is produced.

:class:`deepdiff.DeltaConflictError` exposes public fields ``stable_path``,
``kind``, ``expected`` and ``current``. Absence is represented by an internal
sentinel so it stays distinguishable from a real ``None`` value.

Serialization
-------------

Identity metadata travels through the existing serializers. Using the JSON
serializer/deserializer (or the default restricted pickle) and reloading in
another process preserves the ability to apply, subtract, rebase and locate
conflicts. ``to_flat_dicts()`` round trips through ``Delta(flat_dict_list=...)``
as well; identity segments become tagged JSON-safe tokens. Corrupt or
unknown identity metadata is rejected explicitly rather than silently
degrading to an index based patch, and paths are never evaluated with
``eval``. Deltas and diffs created without ``iterable_keys`` keep their
original behavior unchanged.
