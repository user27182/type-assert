"""Pytest integration: collect case files and run each case as its own test.

Registered as a pytest plugin by entry point, so a project only has to say where
its cases live::

    [tool.pytest.ini_options]
    type_assert_cases = 'tests/typing/cases'
    type_assert_checkers = 'mypy pyright'

Each case file then collects as a test file of its own: one `setup` test for the
lines that are not cases, one runtime test per case, and one static test per case
per configured checker.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ._cases import CaseSkipped
from ._cases import collect_case_file
from ._checkers import CHECKERS
from ._checkers import get_checker

if TYPE_CHECKING:
    from ._cases import Case
    from ._cases import CaseFile
    from ._checkers import Diagnostic

CASES_INI = 'type_assert_cases'
CHECKERS_INI = 'type_assert_checkers'
DEFAULT_CHECKERS = ('mypy',)

_DIAGNOSTICS = '_type_assert_diagnostics'


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the settings a project needs."""
    parser.addini(
        CASES_INI,
        'Directory of type_assert case files, relative to the rootdir.',
        default='',
    )
    parser.addini(
        CHECKERS_INI,
        'Type checkers to check the cases with, whitespace separated. Any of: '
        f'{", ".join(sorted(CHECKERS))}. Each one gets a static test of its own per '
        "case, so a case can be held to more than one checker's inference. Defaults "
        f'to {" ".join(DEFAULT_CHECKERS)}.',
        type='args',
        default=list(DEFAULT_CHECKERS),
    )
    # One per registered checker, since their flags have nothing in common. Each
    # checker already reads the project's own configuration -- it runs from the
    # rootdir, which is where every checker looks -- so this is for what that
    # configuration cannot say: a different config file, a Python version, a
    # strictness flag that should apply to the cases and nothing else.
    for name in sorted(CHECKERS):
        parser.addini(
            checker_args_ini(name),
            f'Extra command line arguments for {name}, whitespace separated.',
            type='args',
            default=[],
        )


def checker_args_ini(checker_name: str) -> str:
    """Return the ini option holding extra arguments for `checker_name`."""
    return f'type_assert_{checker_name}_args'


def configured_checkers(config: pytest.Config) -> list[str]:
    """Return the names of the checkers the cases are checked with."""
    names = [str(name) for name in config.getini(CHECKERS_INI) if str(name).strip()]
    return names or list(DEFAULT_CHECKERS)


def cases_dir(config: pytest.Config) -> Path | None:
    """Return the configured cases directory, or `None` when unset."""
    configured = str(config.getini(CASES_INI)).strip()
    if not configured:
        return None
    return (Path(config.rootpath) / configured).resolve()


def pytest_collect_file(file_path: Path, parent: pytest.Collector):
    """Collect a `.py` file in the configured cases directory as a case file."""
    directory = cases_dir(parent.config)
    if directory is None or file_path.suffix != '.py' or file_path.parent != directory:
        return None
    return CaseFileCollector.from_parent(parent, path=file_path)


def _diagnostics(config: pytest.Config, checker_name: str) -> dict[Path, list[Diagnostic]]:
    """Run one checker once per session and cache its result on the config."""
    cache = getattr(config, _DIAGNOSTICS, None)
    if cache is None:
        cache = {}
        setattr(config, _DIAGNOSTICS, cache)
    if checker_name not in cache:
        directory = cases_dir(config)
        assert directory is not None  # only reachable from a collected case file
        root = Path(config.rootpath)
        checker = get_checker(checker_name)
        # A cache directory per checker and per xdist worker: the run is cheap once
        # warm, and sharing one between concurrent workers is what makes it stale.
        worker = getattr(config, 'workerinput', {}).get('workerid', 'master')
        cache[checker_name] = checker.run(
            '.'.join(directory.relative_to(root).parts),
            root=root,
            cache_dir=root / '.mypy_cache' / f'type_assert-{checker_name}-{worker}',
            extra_args=[str(arg) for arg in config.getini(checker_args_ini(checker_name))],
        )
    return cache[checker_name]


def _report(case_file: CaseFile, checker_name: str, errors: list[Diagnostic]) -> str:
    """Return the checker's messages against the source lines they came from."""
    source = case_file.path.read_text(encoding='utf-8').splitlines()
    body = '\n'.join(
        f'{case_file.name}:{error.line}: {error.message}\n\t{source[error.line - 1].strip()}'
        for error in errors
    )
    return f'{checker_name} reported {len(errors)} error(s):\n{body}'


class CaseFileCollector(pytest.File):
    """Collects one case file as a test file."""

    def collect(self):
        """Yield the file's setup test, and per case a runtime test and a static one."""
        case_file = collect_case_file(Path(self.path))
        checkers = configured_checkers(self.config)
        yield SetupItem.from_parent(self, name='setup', case_file=case_file)
        for case in case_file.cases:
            yield RuntimeItem.from_parent(
                self, name=f'{case.id} [runtime]', case_file=case_file, case=case
            )
            for checker_name in checkers:
                yield StaticItem.from_parent(
                    self,
                    name=f'{case.id} [static: {checker_name}]',
                    case_file=case_file,
                    case=case,
                    checker_name=checker_name,
                )


class _Item(pytest.Item):
    """Shared plumbing for the tests a case file collects."""

    def __init__(self, *args, case_file: CaseFile, **kwargs) -> None:
        """Record the case file this test belongs to."""
        super().__init__(*args, **kwargs)
        self.case_file = case_file

    def reportinfo(self):
        """Locate this test in its case file."""
        return self.path, None, self.name

    def errors_on(self, checker_name: str, lines: frozenset[int]) -> list[Diagnostic]:
        """Return one checker's errors falling on `lines` of this case file."""
        reported = _diagnostics(self.config, checker_name).get(self.case_file.path.resolve(), [])
        return [diagnostic for diagnostic in reported if diagnostic.line in lines]


class SetupItem(_Item):
    """Checks a case file's setup: that it is well formed, runs, and type-checks."""

    def runtest(self) -> None:
        """Assert the file parses, its setup executes, and nothing else is wrong."""
        # Report the file's own problem before asking for a type-check of it: a file
        # the checker cannot parse fails the whole run, which would mask the reason.
        if self.case_file.error is not None:
            pytest.fail(self.case_file.error, pytrace=False)

        namespace = self.case_file.setup_namespace()
        unknown = self.case_file.unknown_skips(namespace)
        if unknown:
            listed = '\n'.join(f'  {key}' for key in unknown)
            pytest.fail(
                f'SKIP_RUNTIME names expressions that no case in this file makes, so the '
                f'skip no longer applies to anything:\n{listed}',
                pytrace=False,
            )

        for checker_name in configured_checkers(self.config):
            errors = self.errors_on(checker_name, self.case_file.setup_lines)
            if errors:
                pytest.fail(_report(self.case_file, checker_name, errors), pytrace=False)


class _CaseItem(_Item):
    """A test about one case."""

    def __init__(self, *args, case: Case, **kwargs) -> None:
        """Record the case this test is about."""
        super().__init__(*args, **kwargs)
        self.case = case


class RuntimeItem(_CaseItem):
    """Checks the value a case builds against the type the case expects."""

    def runtest(self) -> None:
        """Run this case, and only it."""
        try:
            self.case_file.run(self.case)
        except CaseSkipped as skipped:
            pytest.skip(str(skipped))


class StaticItem(_CaseItem):
    """Checks the type one checker infers for a case against the type it expects."""

    def __init__(self, *args, checker_name: str, **kwargs) -> None:
        """Record which checker this test speaks for."""
        super().__init__(*args, **kwargs)
        self.checker_name = checker_name

    def runtest(self) -> None:
        """Assert the checker reports nothing on this case's lines."""
        errors = self.errors_on(self.checker_name, self.case.lines)
        if errors:
            pytest.fail(_report(self.case_file, self.checker_name, errors), pytrace=False)
