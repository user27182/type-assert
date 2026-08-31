"""The pyrefly backend."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
from typing import TYPE_CHECKING

from ._base import Checker
from ._base import CheckerError
from ._base import Diagnostic

if TYPE_CHECKING:
    from collections.abc import Sequence

# `ERROR path:line:col-col: message`, the shape of `--output-format min-text`.
_DIAGNOSTIC = re.compile(r'^ERROR (?P<path>.+?):(?P<line>\d+):\d+(?:-\d+)?: (?P<msg>.*)$')

# pyrefly says this, then checks nothing at all and reports success.
_NO_CONFIG = 'No `pyrefly.toml` found'


class PyreflyChecker(Checker):
    """Runs pyrefly and reads its one-line-per-error output."""

    name = 'pyrefly'
    distribution = 'pyrefly'

    def run(
        self,
        package: str,
        *,
        root: Path,
        cache_dir: Path | None,
        extra_args: Sequence[str] = (),
    ) -> dict[Path, list[Diagnostic]]:
        """Type-check `package` from `root` and return pyrefly's errors keyed by file.

        pyrefly discovers the project's own configuration from `root`, and unlike
        the others it will not check anything without one -- see below.
        """
        del cache_dir  # pyrefly keeps no cache of its own to point elsewhere.
        target = Path(root) / Path(*package.split('.'))
        # pyrefly reports success for a path that does not exist, where mypy and
        # pyright both refuse. Refuse here too: passing for nothing is the one
        # outcome a checker must never produce.
        if not target.exists():
            msg = f'pyrefly was asked to check {target}, which does not exist.'
            raise CheckerError(msg)
        args = [
            sys.executable,
            '-m',
            'pyrefly',
            'check',
            '--python-interpreter-path',
            sys.executable,
            *extra_args,
            # Last, because the output is parsed and has to stay parsable.
            '--output-format',
            'min-text',
            str(target),
        ]

        try:
            process = subprocess.run(args, capture_output=True, cwd=root, text=True, check=False)
        except OSError as error:  # pragma: no cover - defensive
            msg = f'Could not run pyrefly: {error}'
            raise CheckerError(msg) from error

        output = process.stdout + process.stderr
        if 'No module named pyrefly' in process.stderr:
            msg = (
                f'pyrefly failed to run:\n{" ".join(args)}\n\n{output}\n\n'
                f'Install it with: pip install type-assert[pyrefly]'
            )
            raise CheckerError(msg)

        # Without a config pyrefly ignores the paths it is given and reports success,
        # which would silently pass every case. Refuse rather than pass for nothing.
        if _NO_CONFIG in output:
            msg = (
                'pyrefly found no configuration and so checked nothing, which would '
                'pass every case without looking at it. Give the project a '
                '`pyrefly.toml`, or a `[tool.pyrefly]` table in `pyproject.toml`, '
                f'naming the cases directory in `project-includes`.\n\n{output}'
            )
            raise CheckerError(msg)

        diagnostics: dict[Path, list[Diagnostic]] = {}
        for line in output.splitlines():
            match = _DIAGNOSTIC.match(line.strip())
            if match is None:
                continue
            path = (Path(root) / match['path']).resolve()
            diagnostics.setdefault(path, []).append(
                Diagnostic(path=path, line=int(match['line']), message=match['msg'])
            )
        return diagnostics
