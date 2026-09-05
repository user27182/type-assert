"""What `assert_types` accepts and rejects at runtime.

The static half is exercised by the plugin tests, which run a real checker. These
cover the runtime half on its own, where the interesting cases are containers: a
checker that only samples them would pass several of these wrongly.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
import re
from typing import Any
from typing import Literal
from typing import Optional
from typing import Protocol
from typing import Union
from typing import runtime_checkable

import numpy as np
import numpy.typing as npt
import pytest

from type_assert import assert_types


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
