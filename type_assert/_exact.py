"""The ways the runtime half is stricter than plain assignability.

pycroscope decides whether a value is assignable to a type the way the type system
does, which lets an `int` pass for `float`, cannot see the type arguments of a NumPy
array, and reads a container it does not recognise as bare. Each is a place where a
declared type and a produced value drift apart in practice, so after pycroscope has
accepted a value, `mismatch` walks it once more against the expected type and rejects
a number that is not an instance of the numeric class named, an array whose dtype or
dimensionality is not the one named, and an item of the wrong type in any sequence,
set or mapping, the built-in ones and a class of your own alike. Anything it does not
understand it leaves to pycroscope's verdict.
"""

from __future__ import annotations

from collections.abc import Iterable
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
#: Container protocols whose parameters say what the container holds. A class of
#: your own is walked only through one of these, and each can be iterated more than
#: once; an `Iterator` cannot, and reading it would consume the value under test.
_WALKABLE = (Sequence, AbstractSet, Mapping)
#: Of those, the ones walked item by item rather than as key and value pairs.
_ITEMWISE = (Sequence, AbstractSet)
#: Never walked as a sequence of characters.
_TEXT = (str, bytes, bytearray)


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
    if origin is tuple and isinstance(value, tuple):
        return _mismatch_tuple(value, expected, checker=checker, path=path)
    contents = _contents(expected)
    if contents is not None:
        return _mismatch_contents(value, contents, checker=checker, path=path)
    ndarray = _ndarray_class()
    if ndarray is not None and isinstance(value, ndarray) and _is_array_type(expected, ndarray):
        return _mismatch_array(value, expected, checker=checker, path=path)
    return None


def _contents(expected: Any) -> tuple[Any, ...] | None:
    """Return the item types a container type holds, or `None` if it is not one.

    One type for a sequence or set, two for a mapping. A class of your own reaches
    its container base through `_inherited`, so `MultiBlock[PolyData]` answers with
    `PolyData` the way `list[PolyData]` does.
    """
    origin = get_origin(expected)
    if origin in _MAPPINGS:
        return get_args(expected) or (Any, Any)
    if origin in _SEQUENCES:
        return get_args(expected) or (Any,)
    return _inherited(expected)


def _inherited(expected: Any) -> tuple[Any, ...] | None:
    """Return the item types a parametrised class passes to its container base.

    The arguments are matched to the class's own parameters and substituted into the
    base it lists, so a class that reorders or wraps them is followed correctly rather
    than assumed to hold its first argument.
    """
    origin = get_origin(expected)
    if not isinstance(origin, type) or not issubclass(origin, _WALKABLE):
        return None
    return _substituted(origin, dict(zip(_parameters(origin), get_args(expected), strict=False)))


def _parameters(cls: type) -> tuple[Any, ...]:
    """Return the type variables `cls` takes, in the order it takes them.

    Paired with the arguments loosely: a class subscripted with the wrong number of
    them leaves a type variable unsubstituted, which `_substituted` refuses rather
    than guessing at.

    Inheriting a `collections.abc` alias does not make a class a `Generic` subclass,
    so typing records no parameters for it; the aliases it lists carry them instead.
    """
    declared = getattr(cls, '__parameters__', ())
    if declared:
        return tuple(declared)
    found: list[Any] = []
    for base in getattr(cls, '__orig_bases__', ()):
        found += [
            parameter
            for parameter in getattr(base, '__parameters__', ())
            if parameter not in found
        ]
    return tuple(found)


def _substituted(cls: type, arguments: Mapping[Any, Any]) -> tuple[Any, ...] | None:
    """Return `cls`'s item types, following the generic bases it declares.

    Recurses so that a class deriving from another class of your own is followed to
    whichever container base finally fixes the item type.
    """
    for base in getattr(cls, '__orig_bases__', ()):
        origin = get_origin(base)
        if origin is None:
            continue
        args = tuple(arguments.get(argument, argument) for argument in get_args(base))
        if origin in _MAPPINGS or origin in _SEQUENCES:
            # A parameter the class does not pass down leaves nothing to check against.
            return None if any(argument in arguments for argument in args) else args
        if isinstance(origin, type):
            found = _substituted(origin, dict(zip(_parameters(origin), args, strict=False)))
            if found is not None:
                return found
    return None


def _mismatch_contents(
    value: object, contents: tuple[Any, ...], *, checker: Checker, path: str
) -> str | None:
    """Check the items of a container against the types it says it holds."""
    if isinstance(value, Mapping):
        key_type, value_type = contents if len(contents) == 2 else (Any, Any)
        return _mismatch_mapping(value, key_type, value_type, checker=checker, path=path)
    if len(contents) != 1 or isinstance(value, _TEXT) or not isinstance(value, _ITEMWISE):
        return None
    items = list(value)
    return _mismatch_items(items, [contents[0]] * len(items), checker=checker, path=path)


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


def _mismatch_items(
    items: Iterable[Any], item_types: Iterable[Any], *, checker: Checker, path: str
) -> str | None:
    """Check each item of a sequence against its type."""
    for index, (item, item_type) in enumerate(zip(items, item_types, strict=True)):
        problem = _mismatch_item(item, item_type, checker=checker, path=f'{path}[{index}]')
        if problem is not None:
            return problem
    return None


def _mismatch_item(item: object, expected: Any, *, checker: Checker, path: str) -> str | None:
    """Check one item of a container, by assignability and then by the stricter reading.

    The assignability half is what pycroscope does for a container it recognises and
    skips for one it does not, so asking here is what lets a class of your own be
    checked at all.
    """
    if not _assignable(expected, KnownValue(item), checker):
        return f'{path} is {_describe(item)}, which is not {_name(expected)}'
    return mismatch(item, expected, checker=checker, path=path)


def _name(expected: Any) -> str:
    """Name a type the way the failure messages read."""
    return getattr(expected, '__name__', None) or str(expected)


def _mismatch_tuple(value: tuple, expected: Any, *, checker: Checker, path: str) -> str | None:
    """Check a tuple item by item, whether it is fixed or variadic."""
    args = get_args(expected)
    if len(args) == 2 and args[1] is Ellipsis:
        return _mismatch_items(value, [args[0]] * len(value), checker=checker, path=path)
    if args == ((),):
        return None
    # pycroscope has already matched the lengths.
    return _mismatch_items(value, args, checker=checker, path=path)


def _mismatch_mapping(
    value: Mapping[Any, Any], key_type: Any, value_type: Any, *, checker: Checker, path: str
) -> str | None:
    """Check a mapping key by key and value by value."""
    for key, item in value.items():
        problem = _mismatch_item(key, key_type, checker=checker, path=f'{path} key {key!r}')
        if problem is not None:
            return problem
        problem = _mismatch_item(item, value_type, checker=checker, path=f'{path}[{key!r}]')
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
    scalar = value.dtype.type
    actual = numpy.ndarray[shape, numpy.dtype[scalar]]

    expected = _with_defaults(expected, numpy)
    concrete = _concrete_dtype(expected, numpy)
    if concrete is not None:
        # The dtype is named by one of the classes in `numpy.dtypes` rather than as
        # `dtype[<scalar>]`. Rebuilding it from the scalar type would not produce
        # that class -- a variable-width string dtype has no scalar of its own -- so
        # compare the array's dtype object against the class, and let the ordinary
        # comparison below carry the dimensionality check.
        if isinstance(value.dtype, concrete):
            expected = numpy.ndarray[get_args(expected)[0], numpy.dtype[scalar]]
        else:
            return f'{path} is an array of dtype {value.dtype}, which is not {concrete.__name__}'

    if _assignable(expected, type_from_runtime(actual), checker):
        return None
    return (
        f'{path} is an array of dtype {value.dtype} with {value.ndim} dimension(s), '
        f'which is not {expected}'
    )


def _with_defaults(expected: Any, numpy: Any) -> Any:
    """Fill in the type arguments an array type left to their defaults.

    `ndarray` declares defaults for both of its parameters, so `ndarray[tuple[int,
    int]]` is a two-dimensional array of any dtype. The defaults live in the stub,
    which the runtime class does not carry, so they are restated here.
    """
    if get_origin(expected) is not numpy.ndarray:
        return expected
    args = get_args(expected)
    if len(args) == 1:
        return numpy.ndarray[args[0], numpy.dtype[Any]]
    return expected


def _concrete_dtype(expected: Any, numpy: Any) -> type | None:
    """Return the `numpy.dtypes` class an array type names as its dtype, if it names one.

    `ndarray[S, dtype[int8]]` parametrises `dtype`, which is a generic alias rather
    than a class; `ndarray[S, Int8DType]` names the dtype class itself.
    """
    if get_origin(expected) is not numpy.ndarray:
        return None
    args = get_args(expected)
    if len(args) != 2:
        return None
    dtype = args[1]
    if isinstance(dtype, type) and issubclass(dtype, numpy.dtype):
        return dtype
    return None


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


#: Longest repr kept in a message, before the middle is elided.
_REPR_WIDTH = 60


def _describe(value: object) -> str:
    """Name a value by its class and repr, the way the failure messages read.

    The repr is flattened to one line and shortened, since an object that reprs as a
    table would otherwise bury the sentence it sits in.
    """
    shown = ' '.join(repr(value).split())
    if len(shown) > _REPR_WIDTH:
        half = _REPR_WIDTH // 2 - 2
        shown = f'{shown[:half]} ... {shown[-half:]}'
    return f'{type(value).__name__} {shown}'
