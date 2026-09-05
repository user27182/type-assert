"""The pytest plugin, end to end, through a real pytest run.

`pytester` runs pytest in a project of its own, so these exercise what a consumer
actually gets: the entry point, the ini options, collection, and the outcome of
each generated test.
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import REPO_ROOT

CLEAN = """\
from type_assert import assert_types

assert_types(len([1]), int)
assert_types(sorted({'b', 'a'}), list[str])
"""


@pytest.fixture
def project(pytester, monkeypatch):
    """Return a helper that lays out a project with a cases directory."""
    monkeypatch.setenv('MYPYPATH', str(REPO_ROOT))
    pytester.path.joinpath('pyrightconfig.json').write_text(
        json.dumps({'extraPaths': [str(REPO_ROOT)]}), encoding='utf-8'
    )

    def _project(source, *, checkers='mypy', cases='cases', name='sample.py'):
        settings = f'type_assert_cases = {cases}\n' if cases is not None else ''
        pytester.makefile(
            '.ini',
            pytest=f'[pytest]\n{settings}type_assert_checkers = {checkers}\n',
        )
        if cases is not None:
            directory = pytester.path / cases
            directory.mkdir(parents=True, exist_ok=True)
            directory.joinpath(name).write_text(source, encoding='utf-8')
        return pytester

    return _project


def test_a_clean_case_file_passes(project):
    result = project(CLEAN).runpytest()
    # One setup test, plus a runtime and a static test for each of two cases.
    result.assert_outcomes(passed=5)


def test_each_case_is_named_after_the_claim_it_makes(project):
    result = project(CLEAN).runpytest('--collect-only', '-q')
    # Compared as plain strings: the ids contain brackets, which are wildcards to
    # the fnmatch helpers.
    collected = set(result.outlines)
    assert 'cases/sample.py::setup' in collected
    assert 'cases/sample.py::len([1]) -> int [runtime]' in collected
    assert 'cases/sample.py::len([1]) -> int [static: mypy]' in collected
    assert "cases/sample.py::sorted({'b', 'a'}) -> list[str] [runtime]" in collected


def test_a_wrong_static_type_fails_only_that_case(project):
    source = CLEAN.replace('assert_types(len([1]), int)', 'assert_types(len([1]), str)')
    result = project(source).runpytest()
    result.assert_outcomes(passed=3, failed=2)


def test_a_supertype_fails_statically_but_passes_at_runtime(project):
    # `assert_type` is exact, so this is the case assignability would wrongly allow.
    source = 'from type_assert import assert_types\n\nassert_types(len([1]), object)\n'
    result = project(source).runpytest()
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(['*static: mypy*'])


def test_a_wrong_runtime_value_fails_the_runtime_half(project):
    source = (
        'from typing import Any\n\n'
        'from type_assert import assert_types\n\n'
        'def lies() -> Any:\n'
        '    return "not an int"\n\n'
        'assert_types(lies(), Any)\n'
    )
    result = project(source).runpytest()
    # `Any` satisfies the checker, so only the runtime half has anything to say.
    result.assert_outcomes(passed=3)


def test_a_value_the_type_system_would_promote_fails_the_runtime_half(project):
    # Returning an int for a declared float satisfies every checker, so only the
    # runtime half can say that the declaration does not match what is produced.
    source = (
        'from type_assert import assert_types\n\n'
        'def rounded() -> float:\n'
        '    return 1\n\n'
        'assert_types(rounded(), float)\n'
    )
    result = project(source).runpytest()
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(['*runtime*', '*is int 1, not float*'])


def test_an_error_outside_a_case_fails_the_setup_test(project):
    source = (
        'from type_assert import assert_types\n\nBAD: int = "no"\nassert_types(len([1]), int)\n'
    )
    result = project(source).runpytest()
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(['*::setup*'])


def test_a_malformed_file_fails_its_own_test_without_aborting_collection(project):
    source = 'from type_assert import assert_types\n\ndef helper():\n    assert_types(1, int)\n'
    result = project(source).runpytest()
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(['*must be a statement at module level*'])


def test_a_syntax_error_reports_the_syntax_error(project):
    result = project('def broken(\n').runpytest()
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(['*never closed*'])


def test_cases_are_independent_of_each_other(project):
    source = (
        'from type_assert import assert_types\n\n'
        'VALUES = []\n'
        'assert_types(VALUES.append(1), None)\n'
        'assert_types(len(VALUES), int)\n'
    )
    result = project(source).runpytest('-p', 'no:randomly')
    result.assert_outcomes(passed=5)


def test_a_skipped_case_skips_only_its_runtime_half(project):
    source = (
        'from type_assert import assert_types\n\n'
        "SKIP_RUNTIME = {'len([1])': 'a reason'}\n"
        'assert_types(len([1]), int)\n'
    )
    result = project(source).runpytest('-rs')
    result.assert_outcomes(passed=2, skipped=1)
    result.stdout.fnmatch_lines(['*a reason*'])


QUOTED_NEVER = (
    'from typing import TYPE_CHECKING\n\n'
    'from type_assert import assert_types\n\n'
    'if TYPE_CHECKING:\n'
    '    from typing_extensions import Never\n\n\n'
    'def empty() -> list[Never]:\n'
    '    return []\n\n\n'
    "assert_types(empty(), 'list[Never]')\n"
)


@pytest.mark.parametrize('checker', ['mypy', 'pyright'])
def test_a_quoted_type_only_a_checker_can_build_is_checked_statically(project, checker):
    result = project(QUOTED_NEVER, checkers=checker).runpytest('-rs')
    result.assert_outcomes(passed=2, skipped=1)
    result.stdout.fnmatch_lines(['*cannot be built at runtime*'])


def test_a_case_file_named_on_the_command_line_is_collected_once_as_cases(project):
    # pytest collects an explicitly named `.py` file as a test module whatever its
    # name; importing this one would run the quoted assertion at import and fail.
    result = project(QUOTED_NEVER).runpytest('cases/sample.py', '-rs')
    result.assert_outcomes(passed=2, skipped=1, errors=0)
    result.stdout.fnmatch_lines(['*cannot be built at runtime*'])


def test_a_case_file_named_on_the_command_line_yields_the_same_tests(project):
    result = project(CLEAN).runpytest('cases/sample.py', '--collect-only', '-q')
    result.stdout.fnmatch_lines(['*::setup', '*len([[]1[]]) -> int [[]runtime[]]'])
    assert result.stdout.str().count('[runtime]') == 2


def test_a_quoted_type_the_runtime_can_build_is_held_to_by_both_halves(project):
    source = (
        'from type_assert import assert_types\n\n'
        "assert_types(len([1]), 'int')\n"
        "assert_types(len([1]), 'str')\n"
    )
    result = project(source).runpytest()
    # The right type passes both halves and the wrong one fails both: quoting weakens neither.
    result.assert_outcomes(passed=3, failed=2)


def test_a_skip_naming_no_case_fails_the_setup_test(project):
    source = (
        'from type_assert import assert_types\n\n'
        "SKIP_RUNTIME = {'gone()': 'a reason'}\n"
        'assert_types(len([1]), int)\n'
    )
    result = project(source).runpytest()
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(['*no longer applies to anything*'])


def test_nothing_is_collected_when_no_cases_directory_is_configured(project):
    result = project(CLEAN, cases=None).runpytest()
    result.assert_outcomes()


def test_files_outside_the_cases_directory_are_left_alone(project):
    pytester = project(CLEAN)
    pytester.path.joinpath('not_a_case.py').write_text(CLEAN, encoding='utf-8')
    result = pytester.runpytest()
    result.assert_outcomes(passed=5)


def test_an_unknown_checker_is_reported(project):
    result = project(CLEAN, checkers='nope').runpytest()
    result.stdout.fnmatch_lines(['*Unknown type checker*'])


@pytest.mark.parametrize('checker', ['mypy', 'pyright'])
def test_either_checker_drives_the_static_half(project, checker):
    result = project(CLEAN, checkers=checker).runpytest()
    result.assert_outcomes(passed=5)


@pytest.mark.parametrize('checker', ['mypy', 'pyright'])
def test_either_checker_catches_a_wrong_type(project, checker):
    source = 'from type_assert import assert_types\n\nassert_types(len([1]), str)\n'
    result = project(source, checkers=checker).runpytest()
    result.assert_outcomes(passed=1, failed=2)


def test_the_checker_defaults_to_mypy(pytester, monkeypatch):
    monkeypatch.setenv('MYPYPATH', str(REPO_ROOT))
    pytester.makefile('.ini', pytest='[pytest]\ntype_assert_cases = cases\n')
    directory = pytester.path / 'cases'
    directory.mkdir()
    directory.joinpath('sample.py').write_text(CLEAN, encoding='utf-8')
    result = pytester.runpytest()
    result.assert_outcomes(passed=5)


class TestSeveralCheckers:
    """A case can be held to more than one checker at once."""

    def test_each_case_gets_a_static_test_per_checker(self, project):
        result = project(CLEAN, checkers='mypy pyright').runpytest('--collect-only', '-q')
        collected = set(result.outlines)
        assert 'cases/sample.py::len([1]) -> int [static: mypy]' in collected
        assert 'cases/sample.py::len([1]) -> int [static: pyright]' in collected
        # One runtime test only: the value does not depend on who checked it.
        assert 'cases/sample.py::len([1]) -> int [runtime]' in collected

    def test_both_checkers_run(self, project):
        # setup + 2 cases x (1 runtime + 2 static) = 7
        result = project(CLEAN, checkers='mypy pyright').runpytest()
        result.assert_outcomes(passed=7)

    def test_a_wrong_type_fails_once_per_checker(self, project):
        source = 'from type_assert import assert_types\n\nassert_types(len([1]), str)\n'
        result = project(source, checkers='mypy pyright').runpytest()
        # The runtime half fails once, and each checker's static half fails.
        result.assert_outcomes(passed=1, failed=3)

    def test_the_failure_names_the_checker_that_reported_it(self, project):
        source = 'from type_assert import assert_types\n\nassert_types(len([1]), str)\n'
        result = project(source, checkers='mypy pyright').runpytest()
        result.stdout.fnmatch_lines(['*mypy reported 1 error*'])
        result.stdout.fnmatch_lines(['*pyright reported 1 error*'])

    def test_repeated_and_padded_names_are_tolerated(self, project):
        result = project(CLEAN, checkers='  mypy   mypy  ').runpytest('--collect-only', '-q')
        collected = [line for line in result.outlines if 'static' in line]
        # Named twice, so collected twice; pytest disambiguates the duplicate ids.
        assert len(collected) == 4

    def test_an_empty_setting_falls_back_to_the_default(self, project):
        result = project(CLEAN, checkers='').runpytest('--collect-only', '-q')
        assert any('static: mypy' in line for line in result.outlines)


class TestCheckerConfiguration:
    """The project's own checker configuration applies, and can be added to."""

    UNUSED_IGNORE = (
        'from type_assert import assert_types\n\nassert_types(len([1]), int)  # type: ignore\n'
    )

    def test_a_project_mypy_config_is_picked_up(self, project):
        # Nothing tells the checker about this file: it is found because the
        # checker runs from the rootdir, which is what makes project config work.
        pytester = project(self.UNUSED_IGNORE)
        pytester.path.joinpath('mypy.ini').write_text(
            '[mypy]\nwarn_unused_ignores = True\n', encoding='utf-8'
        )
        result = pytester.runpytest()
        # setup, runtime and static for the one case; the static half fails.
        result.assert_outcomes(passed=2, failed=1)
        result.stdout.fnmatch_lines(['*nused*ignore*'])

    def test_without_that_config_the_same_file_passes(self, project):
        result = project(self.UNUSED_IGNORE).runpytest()
        result.assert_outcomes(passed=3)

    def test_extra_mypy_arguments_are_passed_through(self, pytester, monkeypatch):
        monkeypatch.setenv('MYPYPATH', str(REPO_ROOT))
        pytester.makefile(
            '.ini',
            pytest=(
                '[pytest]\ntype_assert_cases = cases\n'
                'type_assert_mypy_args = --warn-unused-ignores\n'
            ),
        )
        directory = pytester.path / 'cases'
        directory.mkdir()
        directory.joinpath('sample.py').write_text(self.UNUSED_IGNORE, encoding='utf-8')
        result = pytester.runpytest()
        result.assert_outcomes(passed=2, failed=1)

    def test_extra_arguments_can_override_a_default(self, pytester, monkeypatch):
        # `--follow-imports=silent` is a default, so it has to be overridable.
        monkeypatch.setenv('MYPYPATH', str(REPO_ROOT))
        pytester.makefile(
            '.ini',
            pytest=(
                '[pytest]\ntype_assert_cases = cases\n'
                'type_assert_mypy_args = --follow-imports=normal\n'
            ),
        )
        directory = pytester.path / 'cases'
        directory.mkdir()
        directory.joinpath('sample.py').write_text(CLEAN, encoding='utf-8')
        result = pytester.runpytest()
        result.assert_outcomes(passed=5)

    def test_extra_pyright_arguments_are_passed_through(self, pytester, monkeypatch):
        monkeypatch.setenv('MYPYPATH', str(REPO_ROOT))
        pytester.path.joinpath('pyrightconfig.json').write_text(
            json.dumps({'extraPaths': [str(REPO_ROOT)]}), encoding='utf-8'
        )
        pytester.makefile(
            '.ini',
            pytest=(
                '[pytest]\ntype_assert_cases = cases\n'
                'type_assert_checkers = pyright\n'
                'type_assert_pyright_args = --level error\n'
            ),
        )
        directory = pytester.path / 'cases'
        directory.mkdir()
        directory.joinpath('sample.py').write_text(CLEAN, encoding='utf-8')
        result = pytester.runpytest()
        result.assert_outcomes(passed=5)

    def test_a_bad_argument_is_reported_rather_than_swallowed(self, pytester, monkeypatch):
        monkeypatch.setenv('MYPYPATH', str(REPO_ROOT))
        pytester.makefile(
            '.ini',
            pytest=(
                '[pytest]\ntype_assert_cases = cases\ntype_assert_mypy_args = --no-such-flag\n'
            ),
        )
        directory = pytester.path / 'cases'
        directory.mkdir()
        directory.joinpath('sample.py').write_text(CLEAN, encoding='utf-8')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*mypy failed to run*'])


def test_the_checker_runs_once_across_xdist_workers(project):
    # Each worker would otherwise type-check the cases again, into a cache of its own.
    pytester = project(CLEAN)
    result = pytester.runpytest_subprocess('-p', 'xdist', '-n', '2')
    result.assert_outcomes(passed=5)
    caches = list((pytester.path / '.mypy_cache').glob('type_assert-*'))
    assert [path.name for path in caches] == ['type_assert-mypy']
