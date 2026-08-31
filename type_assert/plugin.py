"""Pytest integration: collect case files and run each case as its own test.

Registered as a pytest plugin by entry point, so a project only has to say where
its cases live::

    [tool.pytest.ini_options]
    typeassert_cases = 'tests/typing/cases'
    typeassert_checker = 'mypy'

Each case file then collects as a test file of its own, with one test per case for
the runtime half and one for the static half, plus a `setup` test covering the
lines that are not cases.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ._cases import CaseSkipped
from ._cases import collect_case_file
from ._checkers import get_checker

if TYPE_CHECKING:
    from ._cases import Case
    from ._cases import CaseFile
    from ._checkers import Diagnostic

CASES_INI = 'typeassert_cases'
CHECKER_INI = 'typeassert_checker'
DEFAULT_CHECKER = 'mypy'

_DIAGNOSTICS = '_typeassert_diagnostics'


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the two settings a project needs."""
    parser.addini(
        CASES_INI,
        'Directory of type_assert case files, relative to the rootdir.',
        default='',
    )
    parser.addini(
        CHECKER_INI,
        f'Type checker to check the cases with. One of: mypy, pyright. '
        f'Defaults to {DEFAULT_CHECKER}.',
        default=DEFAULT_CHECKER,
    )


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


def _diagnostics(config: pytest.Config) -> dict[Path, list[Diagnostic]]:
    """Type-check the cases once per session and cache the result on the config."""
    cached = getattr(config, _DIAGNOSTICS, None)
    if cached is None:
        directory = cases_dir(config)
        assert directory is not None  # only reachable from a collected case file
        root = Path(config.rootpath)
        checker = get_checker(str(config.getini(CHECKER_INI)).strip() or DEFAULT_CHECKER)
        # A cache directory per xdist worker: the run is cheap once warm, and sharing
        # one between concurrent workers is what makes it stale.
        worker = getattr(config, 'workerinput', {}).get('workerid', 'master')
        cached = checker.run(
            '.'.join(directory.relative_to(root).parts),
            root=root,
            cache_dir=root / '.mypy_cache' / f'type_assert-{worker}',
        )
        setattr(config, _DIAGNOSTICS, cached)
    return cached


def _report(case_file: CaseFile, errors: list[Diagnostic]) -> str:
    """Return the checker's messages against the source lines they came from."""
    source = case_file.path.read_text(encoding='utf-8').splitlines()
    body = '\n'.join(
        f'{case_file.name}:{error.line}: {error.message}\n\t{source[error.line - 1].strip()}'
        for error in errors
    )
    return f'Type checking reported {len(errors)} error(s):\n{body}'


class CaseFileCollector(pytest.File):
    """Collects one case file as a test file."""

    def collect(self):
        """Yield a test for the file's setup, and two for each of its cases."""
        case_file = collect_case_file(Path(self.path))
        yield SetupItem.from_parent(self, name='setup', case_file=case_file)
        for case in case_file.cases:
            for item_type, suffix in ((RuntimeItem, 'runtime'), (StaticItem, 'static')):
                yield item_type.from_parent(
                    self, name=f'{case.id} [{suffix}]', case_file=case_file, case=case
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

    def errors_on(self, lines: frozenset[int]) -> list[Diagnostic]:
        """Return the checker's errors falling on `lines` of this case file."""
        reported = _diagnostics(self.config).get(self.case_file.path.resolve(), [])
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

        errors = self.errors_on(self.case_file.setup_lines)
        if errors:
            pytest.fail(_report(self.case_file, errors), pytrace=False)


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
    """Checks the type a checker infers for a case against the type it expects."""

    def runtest(self) -> None:
        """Assert the checker reports nothing on this case's lines."""
        errors = self.errors_on(self.case.lines)
        if errors:
            pytest.fail(_report(self.case_file, errors), pytrace=False)
