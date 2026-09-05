"""The two ways the runtime half is stricter than plain assignability.

pycroscope decides whether a value is assignable to a type the way the type system
does, which lets an `int` pass for `float` and cannot see the type arguments of a
NumPy array. Both are places where a declared type and a produced value drift apart
in practice, so after pycroscope has accepted a value, `mismatch` walks it once more
against the expected type and rejects those two things: a number that is not an
instance of the numeric class named, and an array whose dtype or dimensionality is
not the one named. Anything it does not understand it leaves to pycroscope's verdict.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import MutableSequence
from collections.abc import MutableSet
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
import sys
from types import UnionType
from typing import TYPE_CHECKING
from typing import Any
from typing import Union
from typing import get_args
from typing import get_origin

from pycroscope.runtime import CanAssignError
from pycroscope.runtime import KnownValue
from pycroscope.runtime import Relation
from pycroscope.runtime import has_relation
from pycroscope.runtime import type_from_runtime

if TYPE_CHECKING:
    from pycroscope.checker import Checker

# Numeric classes a value has to be an instance of, rather than merely promotable to.
_NUMERIC = (float, int, complex)
_SEQUENCES = (list, set, frozenset, Sequence, MutableSequence, AbstractSet, MutableSet)
_MAPPINGS = (dict, Mapping, MutableMapping)


def mismatch(value: object, expected: Any, *, checker: Checker, path: str = 'value') -> str | None:
    """Return why `value` is not of `expected` under the stricter reading, or `None`.

    Only called for a value pycroscope has already found assignable, so this never
    has to accept anything; it only adds rejections, each naming where in the value
    it found the problem.
    """
    if expected is Any or expected is object:
        return None
    origin = get_origin(expected)
    if origin is Union or origin is UnionType:
        return _mismatch_union(value, expected, checker=checker, path=path)
    if expected in _NUMERIC:
        return _mismatch_number(value, expected, path=path)
    if origin in _SEQUENCES and isinstance(value, (list, tuple, set, frozenset)):
        (item_type,) = get_args(expected) or (Any,)
        return _mismatch_items(value, [item_type] * len(value), checker=checker, path=path)
    if origin is tuple and isinstance(value, tuple):
        return _mismatch_tuple(value, expected, checker=checker, path=path)
    if origin in _MAPPINGS and isinstance(value, dict):
        return _mismatch_mapping(value, expected, checker=checker, path=path)
    ndarray = _ndarray_class()
    if ndarray is not None and isinstance(value, ndarray) and _is_array_type(expected, ndarray):
        return _mismatch_array(value, expected, checker=checker, path=path)
    return None


def _mismatch_union(value: object, expected: Any, *, checker: Checker, path: str) -> str | None:
    """Accept a union if some member accepts the value under both readings."""
    for member in get_args(expected):
        if _assignable(member, KnownValue(value), checker) and (
            mismatch(value, member, checker=checker, path=path) is None
        ):
            return None
    return f'{path} is {_describe(value)}, which is not of any of {expected}'


def _mismatch_number(value: object, expected: type, *, path: str) -> str | None:
    """Require an instance of the numeric class named, not one promotable to it."""
    if isinstance(value, expected) and not (expected is int and isinstance(value, bool)):
        return None
    return f'{path} is {_describe(value)}, not {expected.__name__}'


def _mismatch_items(items, item_types, *, checker: Checker, path: str) -> str | None:
    """Check each item of a sequence against its type."""
    for index, (item, item_type) in enumerate(zip(items, item_types, strict=True)):
        problem = mismatch(item, item_type, checker=checker, path=f'{path}[{index}]')
        if problem is not None:
            return problem
    return None


def _mismatch_tuple(value: tuple, expected: Any, *, checker: Checker, path: str) -> str | None:
    """Check a tuple item by item, whether it is fixed or variadic."""
    args = get_args(expected)
    if len(args) == 2 and args[1] is Ellipsis:
        return _mismatch_items(value, [args[0]] * len(value), checker=checker, path=path)
    if args == ((),):
        return None
    # pycroscope has already matched the lengths.
    return _mismatch_items(value, args, checker=checker, path=path)


def _mismatch_mapping(value: dict, expected: Any, *, checker: Checker, path: str) -> str | None:
    """Check a mapping key by key and value by value."""
    key_type, value_type = get_args(expected) or (Any, Any)
    for key, item in value.items():
        problem = mismatch(key, key_type, checker=checker, path=f'{path} key {key!r}')
        if problem is not None:
            return problem
        problem = mismatch(item, value_type, checker=checker, path=f'{path}[{key!r}]')
        if problem is not None:
            return problem
    return None


def _mismatch_array(value: Any, expected: Any, *, checker: Checker, path: str) -> str | None:
    """Check an array as the array type it actually is.

    A runtime array carries its dtype and shape, so unlike most generics its type
    arguments can be recovered. pycroscope compares two array *types* argument by
    argument, which it cannot do for the array itself.
    """
    numpy = sys.modules['numpy']
    shape = tuple[()] if value.ndim == 0 else tuple[(int,) * value.ndim]  # type: ignore[misc]
    actual = numpy.ndarray[shape, numpy.dtype[value.dtype.type]]
    if _assignable(expected, type_from_runtime(actual), checker):
        return None
    return (
        f'{path} is an array of dtype {value.dtype} with {value.ndim} dimension(s), '
        f'which is not {expected}'
    )


def _is_array_type(expected: Any, ndarray: type) -> bool:
    """Tell whether `expected` is a parametrised `ndarray`, also through `NDArray`.

    `numpy.typing.NDArray` is a type alias, and on Python 3.12 and later a
    subscripted alias reports the alias rather than `ndarray` as its origin.
    """
    origin = get_origin(expected)
    aliased = getattr(origin, '__value__', None)
    if aliased is not None:
        origin = get_origin(aliased)
    return origin is ndarray


def _ndarray_class() -> type | None:
    """Return `numpy.ndarray` if NumPy has been imported, without importing it."""
    numpy = sys.modules.get('numpy')
    return None if numpy is None else numpy.ndarray


def _assignable(expected: Any, actual: Any, checker: Checker) -> bool:
    """Ask pycroscope whether `actual`, a Value, is assignable to `expected`, a type."""
    relation = has_relation(type_from_runtime(expected), actual, Relation.ASSIGNABLE, checker)
    return not isinstance(relation, CanAssignError)


def _describe(value: object) -> str:
    """Name a value by its class and repr, the way the failure messages read."""
    return f'{type(value).__name__} {value!r}'
