"""What `assert_types` accepts and rejects at runtime.

The static half is exercised by the plugin tests, which run a real checker. These
cover the runtime half on its own, where the interesting cases are containers: a
checker that only samples them would pass several of these wrongly.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import MutableSequence
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
import re
from typing import Any
from typing import Generic
from typing import Literal
from typing import Optional
from typing import Protocol
from typing import TypeVar
from typing import Union
from typing import runtime_checkable

import numpy as np
import numpy.typing as npt
import pytest

from type_assert import assert_types

_T = TypeVar('_T')
_K = TypeVar('_K')
_V = TypeVar('_V')


class Box(MutableSequence[_T]):
    """A sequence of your own, parametrised the ordinary way."""

    def __init__(self, items):
        self._items = list(items)

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        self._items[index] = value

    def __delitem__(self, index):
        del self._items[index]

    def __len__(self):
        return len(self._items)

    def insert(self, index, value):
        self._items.insert(index, value)


class Deeper(Box[_T]):
    """A class deriving from another class of your own."""


class Pair(Sequence[_V], Generic[_K, _V]):
    """A sequence whose item type is its second parameter, not its first."""

    def __init__(self, items):
        self._items = list(items)

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return len(self._items)


class Fixed(Sequence[int]):
    """A sequence that fixes its base argument rather than passing one down."""

    def __init__(self, items):
        self._items = list(items)

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return len(self._items)


class _Plain:
    """A base naming no type argument."""


class Mixed(_Plain, Sequence[_T]):
    """A sequence listing a plain base alongside the one that fixes its items."""

    def __init__(self, items):
        self._items = list(items)

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return len(self._items)


class Opaque(Fixed, Generic[_T]):
    """A sequence whose own parameter reaches no container base."""


class Bag(AbstractSet[_T]):
    """A set of your own."""

    def __init__(self, items):
        self._items = set(items)

    def __contains__(self, value):
        return value in self._items

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


class Table(Mapping[_K, _V]):
    """A mapping of your own."""

    def __init__(self, items):
        self._items = dict(items)

    def __getitem__(self, key):
        return self._items[key]

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


class Counter(Iterator[_T]):
    """An iterator of your own, which must not be consumed to check it."""

    def __init__(self, items):
        self._items = iter(list(items))

    def __next__(self):
        return next(self._items)


class Base:
    """A base class, for testing assignability."""


class Derived(Base):
    """A subclass of `Base`."""


@runtime_checkable
class HasName(Protocol):
    """A protocol satisfied by anything with a `name`."""

    name: str


class Named:
    """Satisfies `HasName`."""

    name = 'x'


ACCEPTED = [
    pytest.param(1, int, id='int'),
    pytest.param(True, bool, id='bool'),
    pytest.param(1.5, float, id='float'),
    pytest.param('x', str, id='str'),
    pytest.param(b'x', bytes, id='bytes'),
    pytest.param(None, None, id='none'),
    pytest.param(Derived(), Base, id='subclass-is-assignable'),
    pytest.param(Derived(), Derived, id='exact-class'),
    pytest.param(1, object, id='anything-is-an-object'),
    pytest.param(object(), Any, id='anything-is-any'),
    pytest.param([], list[str], id='empty-list-vacuously-matches'),
    pytest.param({}, dict[str, int], id='empty-dict-vacuously-matches'),
    pytest.param([1, 2, 3], list[int], id='list'),
    pytest.param(['a'], list[str], id='list-of-str'),
    pytest.param({'a': 1}, dict[str, int], id='dict'),
    pytest.param({1, 2}, set[int], id='set'),
    pytest.param(frozenset({1}), frozenset[int], id='frozenset'),
    pytest.param((1, 'a'), tuple[int, str], id='fixed-tuple'),
    pytest.param((1, 2, 3), tuple[int, ...], id='variadic-tuple'),
    pytest.param((), tuple[int, ...], id='empty-variadic-tuple'),
    pytest.param([{'a': 1}], list[dict[str, int]], id='nested'),
    pytest.param([(1, 'a')], list[tuple[int, str]], id='list-of-tuples'),
    pytest.param(1, Union[int, str], id='union-first'),
    pytest.param(1, int | str, id='union-pep604'),
    pytest.param(None, int | None, id='optional-pep604'),
    pytest.param('x', Union[int, str], id='union-second'),
    pytest.param(None, Optional[int], id='optional-none'),
    pytest.param(3, Optional[int], id='optional-value'),
    pytest.param([1, None], list[int | None], id='list-with-optional'),
    pytest.param('a', Literal['a', 'b'], id='literal'),
    pytest.param([1], Sequence[int], id='abc-sequence'),
    pytest.param(len, Callable[[Any], int], id='callable'),
    pytest.param(Named(), HasName, id='protocol'),
]

REJECTED = [
    pytest.param(1, str, id='int-is-not-str'),
    pytest.param('1', int, id='str-is-not-int'),
    pytest.param(None, int, id='none-is-not-int'),
    pytest.param(1, None, id='int-is-not-none'),
    pytest.param(Base(), Derived, id='base-is-not-derived'),
    pytest.param([1], list[str], id='wrong-element-type'),
    pytest.param(['a', 'b', 1], list[str], id='bad-element-last'),
    pytest.param([1, 'a', 2], list[int], id='bad-element-middle'),
    pytest.param([None, 1], list[int], id='none-element-first'),
    pytest.param([1, None], list[int], id='none-element-last'),
    pytest.param({'a': 'b'}, dict[str, int], id='wrong-dict-value'),
    pytest.param({1: 1}, dict[str, int], id='wrong-dict-key'),
    pytest.param((1, 2), tuple[int, str], id='wrong-tuple-member'),
    pytest.param((1, 'a', 2), tuple[int, ...], id='wrong-variadic-member'),
    pytest.param([[1], ['a']], list[list[int]], id='nested-wrong'),
    pytest.param(1.5, Union[int, str], id='not-in-union'),
    pytest.param(1.5, int | str, id='not-in-pep604-union'),
    pytest.param('c', Literal['a', 'b'], id='not-in-literal'),
    pytest.param(object(), HasName, id='does-not-satisfy-protocol'),
]


# Assignable to the type system, and rejected here all the same.
PROMOTED = [
    pytest.param(1, float, id='int-is-not-float'),
    pytest.param(True, float, id='bool-is-not-float'),
    pytest.param(True, int, id='bool-is-not-int'),
    pytest.param(1, complex, id='int-is-not-complex'),
    pytest.param(1.5, complex, id='float-is-not-complex'),
    pytest.param([1, 2], list[float], id='list-of-int-is-not-list-of-float'),
    pytest.param([1.5, 1], list[float], id='promoted-element-last'),
    pytest.param((1.5, 1), tuple[float, float], id='fixed-tuple-member'),
    pytest.param((1.5, 1), tuple[float, ...], id='variadic-tuple-member'),
    pytest.param({1.5, 1}, set[float], id='set-member'),
    pytest.param({'a': 1}, dict[str, float], id='dict-value'),
    pytest.param({1: 'a'}, dict[float, str], id='dict-key'),
    pytest.param([[1.5], [1]], list[list[float]], id='nested'),
    pytest.param([1], Sequence[float], id='abc-sequence'),
    pytest.param(1, Optional[float], id='optional'),
    pytest.param(1, float | None, id='optional-pep604'),
    pytest.param(1, float | str, id='no-union-member-fits-exactly'),
    pytest.param([1.5, 1], list[float | str], id='union-element'),
]

# The stricter reading still lets through what it should.
STILL_ACCEPTED = [
    pytest.param(1.5, float, id='float-is-float'),
    pytest.param(1, int, id='int-is-int'),
    pytest.param(True, bool, id='bool-is-bool'),
    pytest.param(1j, complex, id='complex-is-complex'),
    pytest.param(1, float | int, id='union-names-the-class'),
    pytest.param(1, int | float, id='union-in-either-order'),
    pytest.param(None, float | None, id='none-for-optional-float'),
    pytest.param([1.5, 2.5], list[float], id='list-of-float'),
    pytest.param((1.5, 1), tuple[float, int], id='fixed-tuple-as-declared'),
    pytest.param({'a': 1.5}, dict[str, float], id='dict-as-declared'),
    pytest.param([[1.5]], list[list[float]], id='nested-as-declared'),
    pytest.param(np.float64(1.5), float, id='numpy-float64-is-a-float'),
    pytest.param(1, Any, id='any'),
    pytest.param(1, object, id='object'),
    pytest.param('a', Literal['a', 'b'], id='literal'),
]


@pytest.mark.parametrize(('value', 'expected'), ACCEPTED)
def test_accepts(value, expected):
    assert_types(value, expected)


@pytest.mark.parametrize(('value', 'expected'), PROMOTED)
def test_rejects_what_the_type_system_would_merely_promote(value, expected):
    with pytest.raises(AssertionError, match='does not have the expected type'):
        assert_types(value, expected)


@pytest.mark.parametrize(('value', 'expected'), STILL_ACCEPTED)
def test_the_stricter_reading_accepts_what_is_declared(value, expected):
    assert_types(value, expected)


def test_the_stricter_failure_names_where_the_problem_is():
    with pytest.raises(AssertionError) as error:
        assert_types({'a': [1.5, 1]}, dict[str, list[float]])
    assert "value['a'][1] is int 1, not float" in str(error.value)


class TestArrays:
    """A NumPy array is checked as the array type it is, dtype and dimensions included."""

    def test_the_dtype_is_checked(self):
        assert_types(np.array([1.0]), npt.NDArray[np.float64])
        with pytest.raises(AssertionError, match='dtype float64'):
            assert_types(np.array([1.0]), npt.NDArray[np.int64])

    def test_a_union_of_dtypes_accepts_each_member(self):
        assert_types(np.array([1], dtype=np.float32), npt.NDArray[np.float32 | np.float64])
        with pytest.raises(AssertionError, match='does not have the expected type'):
            assert_types(np.array([1]), npt.NDArray[np.float32 | np.float64])

    def test_a_union_of_array_types_accepts_each_member(self):
        assert_types(np.array([1]), npt.NDArray[np.float64] | npt.NDArray[np.int64])

    def test_the_number_of_dimensions_is_checked(self):
        matrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]
        assert_types(np.zeros((2, 2)), matrix)
        with pytest.raises(AssertionError, match='1 dimension'):
            assert_types(np.zeros(2), matrix)

    def test_a_scalar_array_has_no_dimensions(self):
        assert_types(np.array(1.0), np.ndarray[tuple[()], np.dtype[np.float64]])
        assert_types(np.array(1.0), npt.NDArray[np.float64])

    def test_an_unparametrised_array_type_accepts_any_array(self):
        assert_types(np.array([1.0]), np.ndarray)

    def test_arrays_inside_containers_are_checked(self):
        assert_types([np.array([1])], list[npt.NDArray[np.int64]])
        with pytest.raises(AssertionError, match=r'value\[0\] is an array of dtype float64'):
            assert_types([np.array([1.0])], list[npt.NDArray[np.int64]])

    def test_a_dtype_named_by_its_concrete_class_is_accepted(self):
        # `numpy.dtypes` names each dtype as a class of its own, which an array type
        # may use in place of `dtype[<scalar>]`.
        assert_types(np.arange(3, dtype=np.int8), np.ndarray[tuple[int], np.dtypes.Int8DType])
        with pytest.raises(AssertionError, match='dtype int8'):
            assert_types(
                np.arange(3, dtype=np.int8), np.ndarray[tuple[int], np.dtypes.Float64DType]
            )

    def test_a_dtype_with_no_scalar_type_is_accepted(self):
        # A variable-width string dtype cannot be rebuilt from a scalar type, so it
        # can only be named by its class.
        array = np.array(['a', 'bc'], dtype=np.dtypes.StringDType())
        assert_types(array, np.ndarray[tuple[int], np.dtypes.StringDType])
        with pytest.raises(AssertionError, match='dtype StringDType'):
            assert_types(array, np.ndarray[tuple[int], np.dtypes.StrDType])

    def test_an_omitted_type_argument_falls_back_to_its_default(self):
        # `ndarray` defaults its dtype parameter, so this names a two-dimensional
        # array of any dtype.
        assert_types(np.zeros((2, 3)), np.ndarray[tuple[int, int]])
        assert_types(np.zeros((2, 3), dtype=np.int8), np.ndarray[tuple[int, int]])
        with pytest.raises(AssertionError, match='1 dimension'):
            assert_types(np.zeros(3), np.ndarray[tuple[int, int]])

    def test_a_numpy_scalar_is_not_a_python_int(self):
        # Not a promotion: np.int64 does not subclass int, so pycroscope rejects it.
        with pytest.raises(AssertionError, match='not assignable'):
            assert_types(np.int64(1), int)


@pytest.mark.parametrize(('value', 'expected'), REJECTED)
def test_rejects(value, expected):
    with pytest.raises(AssertionError, match='not assignable'):
        assert_types(value, expected)


class TestQuotedTypes:
    """A type written as a string is built where the assertion is made."""

    def test_a_quoted_type_is_checked_rather_than_taken_for_any(self):
        assert_types(1, 'int')
        with pytest.raises(AssertionError, match='not assignable'):
            assert_types('x', 'int')

    def test_a_quoted_container_type_is_walked(self):
        with pytest.raises(AssertionError, match='not assignable'):
            assert_types([1, 'a'], 'list[int]')

    def test_names_local_to_the_caller_are_visible(self):
        class Local:
            """Exists only inside this test."""

        assert_types(Local(), 'Local')

    def test_a_type_that_cannot_be_built_is_an_error_not_a_pass(self):
        with pytest.raises(TypeError, match=r'cannot be built at runtime \(NameError'):
            assert_types(1, 'NoSuchName')


def test_returns_the_value_unchanged():
    value = [1, 2]
    assert assert_types(value, list[int]) is value


def test_walks_every_element_not_just_the_first():
    # A checker that samples the first element would pass this, and the overloads
    # this package exists to test are exactly the ones that differ at index 1.
    with pytest.raises(AssertionError, match='not assignable'):
        assert_types([1, 2, 3, 4, 5, 6, 7, 8, 9, None], list[int])


def test_failure_names_the_runtime_type_and_the_expected_one():
    with pytest.raises(AssertionError) as error:
        assert_types(1, str)
    message = str(error.value)
    assert "'int'" in message
    assert 'str' in message
    assert 'not assignable' in message


def test_failure_explains_which_element_is_wrong():
    with pytest.raises(AssertionError) as error:
        assert_types(['a', 1], list[str])
    assert re.search(r'element 1', str(error.value))


def test_an_iterator_is_not_consumed_to_check_it():
    # Consuming it would leave the case's own value empty.
    values = iter([1, 2, 3])
    assert_types(values, Iterator[int])
    assert list(values) == [1, 2, 3]


def test_does_not_retain_the_checked_value():
    # pycroscope's module-level checker memoises on the value, which would keep
    # every value a suite ever checked alive until the process exits.
    import gc
    import weakref

    class Holder:
        """Something weak-referenceable to track."""

    def check() -> weakref.ref:
        holder = Holder()
        assert_types(holder, Holder)
        return weakref.ref(holder)

    reference = check()
    gc.collect()
    assert reference() is None


class TestUserContainers:
    """A container class of your own is walked the way a built-in one is."""

    def test_a_matching_item_passes(self):
        assert_types(Box([1, 2]), Box[int])

    def test_an_item_of_the_wrong_type_is_rejected(self):
        with pytest.raises(AssertionError):
            assert_types(Box(['a']), Box[int])

    def test_a_none_among_the_items_is_rejected(self):
        with pytest.raises(AssertionError):
            assert_types(Box([1, None]), Box[int])

    def test_a_union_item_type_accepts_either(self):
        assert_types(Box([1, None]), Box[int | None])

    def test_the_failure_names_the_item(self):
        with pytest.raises(AssertionError, match=r'\[0\] is str'):
            assert_types(Box(['a']), Box[int])

    def test_nesting_is_followed(self):
        with pytest.raises(AssertionError):
            assert_types(Box([Box(['a'])]), Box[Box[int]])

    def test_a_base_class_of_your_own_is_followed(self):
        with pytest.raises(AssertionError):
            assert_types(Deeper(['a']), Deeper[int])

    def test_the_argument_is_matched_to_the_parameter_it_fills(self):
        # `Pair` passes its *second* argument to the base, so the first says nothing
        # about the items. Assuming the first would reject this wrongly.
        assert_types(Pair([1]), Pair[str, int])

    def test_that_parameter_is_still_checked(self):
        with pytest.raises(AssertionError):
            assert_types(Pair(['a']), Pair[str, int])

    def test_a_class_that_fixes_its_base_argument_has_no_parameter_to_check(self):
        assert_types(Fixed(['a']), Fixed)

    def test_a_plain_base_is_skipped_to_reach_the_container_one(self):
        with pytest.raises(AssertionError):
            assert_types(Mixed(['a']), Mixed[int])

    def test_a_parameter_reaching_no_container_base_is_left_alone(self):
        # Nothing says the parameter describes the items, so nothing is claimed.
        assert_types(Opaque(['a']), Opaque[int])

    def test_a_string_is_not_walked_as_a_sequence_of_characters(self):
        assert_types('ab', Sequence[str])

    def test_a_set_of_your_own_is_walked(self):
        with pytest.raises(AssertionError):
            assert_types(Bag({'a'}), Bag[int])

    def test_a_mapping_of_your_own_is_walked(self):
        with pytest.raises(AssertionError):
            assert_types(Table({'a': 'b'}), Table[str, int])

    def test_its_keys_are_checked_too(self):
        with pytest.raises(AssertionError):
            assert_types(Table({1: 2}), Table[str, int])

    def test_an_iterator_subclass_is_not_consumed(self):
        # Walking it would leave the case's own value empty, as for a plain iterator.
        values = Counter([1, 2, 3])
        assert_types(values, Counter[int])
        assert list(values) == [1, 2, 3]


def test_a_long_repr_is_shortened_in_the_failure():
    with pytest.raises(AssertionError) as failure:
        assert_types(Box(['x' * 500]), Box[int])
    (line,) = [line for line in str(failure.value).splitlines() if 'is str' in line]
    assert '...' in line
    assert len(line) < 200
