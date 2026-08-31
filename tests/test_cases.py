"""How a case file is parsed, run, and reported on when it is malformed."""

from __future__ import annotations

import pytest

from type_assert import CaseSkipped
from type_assert import collect_case_file
from type_assert import collect_cases

IMPORT = 'from type_assert import assert_types\n'


@pytest.fixture
def write(tmp_path):
    """Return a helper that writes a case file and parses it."""

    def _write(body, name='cases.py', *, prelude=IMPORT):
        path = tmp_path / name
        path.write_text(prelude + body, encoding='utf-8')
        return collect_case_file(path)

    return _write


def test_each_assertion_becomes_a_case(write):
    case_file = write('assert_types(len([1]), int)\nassert_types(str(1), str)\n')
    assert [case.id for case in case_file.cases] == ['len([1]) -> int', 'str(1) -> str']
    assert case_file.error is None


def test_a_case_records_its_expression_and_expected_type(write):
    case_file = write('assert_types(sorted({1}), list[int])\n')
    (case,) = case_file.cases
    assert case.expression == 'sorted({1})'
    assert case.expected == 'list[int]'


def test_formatting_is_normalised_in_the_id(write):
    case_file = write('assert_types( len( [ 1 ] ) ,   list [ int ] )\n')
    assert case_file.cases[0].id == 'len([1]) -> list[int]'


def test_a_case_knows_which_lines_it_occupies(write):
    case_file = write('X = 1\nassert_types(X, int)\n')
    assert case_file.cases[0].lines == frozenset({3})


def test_a_case_spanning_several_lines_claims_all_of_them(write):
    case_file = write('assert_types(\n    len([1]),\n    int,\n)\n')
    assert case_file.cases[0].lines == frozenset({2, 3, 4, 5})


def test_setup_lines_are_everything_that_is_not_a_case(write):
    case_file = write('X = 1\nassert_types(X, int)\nY = 2\n')
    assert case_file.setup_lines == frozenset({1, 2, 4})


def test_a_file_with_no_cases_is_all_setup(write):
    case_file = write('X = 1\n')
    assert case_file.cases == ()
    assert case_file.setup_lines == frozenset({1, 2})


def test_an_empty_file_parses(write):
    case_file = write('', prelude='')
    assert case_file.cases == ()
    assert case_file.error is None


def test_other_calls_are_setup_not_cases(write):
    case_file = write('print\nlen([1])\nassert_types(1, int)\n')
    assert len(case_file.cases) == 1


def test_the_file_name_is_available_for_test_ids(write):
    case_file = write('assert_types(1, int)\n', name='wrap.py')
    assert case_file.name == 'wrap.py'


def test_collect_cases_reads_a_directory_in_order(tmp_path):
    for name in ('b.py', 'a.py'):
        (tmp_path / name).write_text(IMPORT + 'assert_types(1, int)\n', encoding='utf-8')
    assert [case_file.name for case_file in collect_cases(tmp_path)] == ['a.py', 'b.py']


def test_collect_cases_on_an_empty_directory(tmp_path):
    assert collect_cases(tmp_path) == []


class TestRunning:
    """Executing a case against its file's setup."""

    def test_a_passing_case_runs(self, write):
        case_file = write('assert_types(len([1]), int)\n')
        case_file.run(case_file.cases[0])

    def test_a_failing_case_raises(self, write):
        case_file = write('assert_types(len([1]), str)\n')
        with pytest.raises(AssertionError, match='not assignable'):
            case_file.run(case_file.cases[0])

    def test_setup_is_available_to_the_case(self, write):
        case_file = write('def make():\n    return [1]\n\nassert_types(make(), list[int])\n')
        case_file.run(case_file.cases[0])

    def test_a_case_cannot_see_another_cases_state(self, write):
        body = 'V = []\nassert_types(V.append(1), None)\nassert_types(len(V), int)\n'
        case_file = write(body)
        case_file.run(case_file.cases[0])
        namespace = case_file.setup_namespace()
        case_file.run(case_file.cases[1])
        # The append in the first case must not be visible anywhere else.
        assert namespace['V'] == []

    def test_running_a_case_twice_starts_from_the_same_state(self, write):
        body = 'V = []\nassert_types(V.append(1), None)\n'
        case_file = write(body)
        case_file.run(case_file.cases[0])
        case_file.run(case_file.cases[0])

    def test_setup_namespace_carries_the_file_identity(self, write):
        case_file = write('assert_types(1, int)\n', name='named.py')
        namespace = case_file.setup_namespace()
        assert namespace['__name__'] == 'named'
        assert namespace['__file__'].endswith('named.py')

    def test_an_error_in_setup_propagates(self, write):
        case_file = write('raise RuntimeError("boom")\nassert_types(1, int)\n')
        with pytest.raises(RuntimeError, match='boom'):
            case_file.run(case_file.cases[0])


class TestMalformed:
    """A file the parser cannot make sense of reports rather than raises."""

    def test_a_nested_assertion_is_rejected(self, write):
        body = 'def helper():\n    assert_types(1, int)\n\nassert_types(2, int)\n'
        case_file = write(body)
        assert 'must be a statement at module level' in case_file.error
        assert 'line(s) 3' in case_file.error

    def test_an_assertion_inside_a_conditional_is_rejected(self, write):
        case_file = write('if True:\n    assert_types(1, int)\n')
        assert 'must be a statement at module level' in case_file.error

    def test_the_wrong_number_of_arguments_is_rejected(self, write):
        case_file = write('assert_types(1)\n')
        assert 'takes an expression and a type' in case_file.error

    def test_too_many_arguments_are_rejected(self, write):
        case_file = write('assert_types(1, int, str)\n')
        assert 'takes an expression and a type' in case_file.error

    def test_a_syntax_error_is_reported(self, write):
        case_file = write('def broken(\n')
        assert 'never closed' in case_file.error
        assert case_file.cases == ()

    def test_a_missing_file_is_reported(self, tmp_path):
        case_file = collect_case_file(tmp_path / 'absent.py')
        assert case_file.error is not None
        assert case_file.cases == ()


class TestSkipping:
    """`SKIP_RUNTIME` takes the runtime half out, and only that."""

    def test_a_named_case_is_skipped(self, write):
        body = "SKIP_RUNTIME = {'len([1])': 'a reason'}\nassert_types(len([1]), int)\n"
        case_file = write(body)
        with pytest.raises(CaseSkipped, match='a reason'):
            case_file.run(case_file.cases[0])

    def test_other_cases_still_run(self, write):
        body = (
            "SKIP_RUNTIME = {'len([1])': 'a reason'}\n"
            'assert_types(len([1]), int)\n'
            'assert_types(str(1), str)\n'
        )
        case_file = write(body)
        case_file.run(case_file.cases[1])

    def test_a_skip_can_be_built_conditionally(self, write):
        body = (
            'import sys\n\n'
            'SKIP_RUNTIME = {}\n'
            "if sys.platform == 'nowhere':\n"
            "    SKIP_RUNTIME['len([1])'] = 'never applies'\n"
            'assert_types(len([1]), int)\n'
        )
        case_file = write(body)
        case_file.run(case_file.cases[0])

    def test_an_empty_reason_does_not_skip(self, write):
        body = "SKIP_RUNTIME = {'len([1])': ''}\nassert_types(len([1]), int)\n"
        case_file = write(body)
        case_file.run(case_file.cases[0])

    def test_a_file_without_the_mapping_skips_nothing(self, write):
        case_file = write('assert_types(len([1]), int)\n')
        assert case_file.unknown_skips(case_file.setup_namespace()) == []

    def test_a_skip_naming_no_case_is_reported(self, write):
        body = "SKIP_RUNTIME = {'gone()': 'a reason'}\nassert_types(len([1]), int)\n"
        case_file = write(body)
        assert case_file.unknown_skips(case_file.setup_namespace()) == ['gone()']

    def test_unknown_skips_are_listed_in_order(self, write):
        body = "SKIP_RUNTIME = {'b()': 'x', 'a()': 'y'}\nassert_types(len([1]), int)\n"
        case_file = write(body)
        assert case_file.unknown_skips(case_file.setup_namespace()) == ['a()', 'b()']

    def test_a_matching_skip_is_not_reported_as_unknown(self, write):
        body = "SKIP_RUNTIME = {'len([1])': 'a reason'}\nassert_types(len([1]), int)\n"
        case_file = write(body)
        assert case_file.unknown_skips(case_file.setup_namespace()) == []
