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

    def _project(source, *, checker='mypy', cases='cases', name='sample.py'):
        settings = f'typeassert_cases = {cases}\n' if cases is not None else ''
        pytester.makefile(
            '.ini',
            pytest=f'[pytest]\n{settings}typeassert_checker = {checker}\n',
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
    assert 'cases/sample.py::len([1]) -> int [static]' in collected
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
    result.stdout.fnmatch_lines(['*[[]static[]]*'])


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


def test_an_error_outside_a_case_fails_the_setup_test(project):
    source = 'from type_assert import assert_types\n\nBAD: int = "no"\nassert_types(len([1]), int)\n'
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
    result = project(CLEAN, checker='nope').runpytest()
    result.stdout.fnmatch_lines(['*Unknown type checker*'])


@pytest.mark.parametrize('checker', ['mypy', 'pyright'])
def test_either_checker_drives_the_static_half(project, checker):
    result = project(CLEAN, checker=checker).runpytest()
    result.assert_outcomes(passed=5)


@pytest.mark.parametrize('checker', ['mypy', 'pyright'])
def test_either_checker_catches_a_wrong_type(project, checker):
    source = 'from type_assert import assert_types\n\nassert_types(len([1]), str)\n'
    result = project(source, checker=checker).runpytest()
    result.assert_outcomes(passed=1, failed=2)


def test_the_checker_defaults_to_mypy(pytester, monkeypatch):
    monkeypatch.setenv('MYPYPATH', str(REPO_ROOT))
    pytester.makefile('.ini', pytest='[pytest]\ntypeassert_cases = cases\n')
    directory = pytester.path / 'cases'
    directory.mkdir()
    directory.joinpath('sample.py').write_text(CLEAN, encoding='utf-8')
    result = pytester.runpytest()
    result.assert_outcomes(passed=5)
