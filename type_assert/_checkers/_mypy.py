"""The mypy backend."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from ._base import Checker
from ._base import CheckerError
from ._base import Diagnostic

# `path:line:col: severity: message`, with the column absent on whole-file diagnostics.
_DIAGNOSTIC = re.compile(r'^(?P<path>.+?):(?P<line>\d+):(?:\d+:)? (?P<severity>\w+): (?P<msg>.*)$')


class MypyChecker(Checker):
    """Runs mypy and reads its text output."""

    name = 'mypy'
    distribution = 'mypy'

    def run(
        self,
        package: str,
        *,
        root: Path,
        cache_dir: Path | None,
        extra_args: Sequence[str] = (),
    ) -> dict[Path, list[Diagnostic]]:
        """Type-check `package` from `root` and return mypy's errors keyed by file.

        mypy discovers the project's own configuration from `root`, so nothing has
        to be restated here. Anything it cannot express goes in `extra_args`.
        """
        # `--follow-imports=silent` types the symbols the cases use without reporting
        # the host project's own diagnostics, which vary by platform and dependency
        # versions and have nothing to do with the cases. It comes before `extra_args`
        # so a project that wants a different setting can say so.
        defaults = ['--follow-imports=silent']
        if cache_dir is not None:
            defaults.append(f'--cache-dir={cache_dir}')
        # These come after, because the output is parsed and has to stay parsable.
        required = ['--no-color-output', '--no-error-summary', '--no-pretty', '--show-traceback']
        args = [
            sys.executable,
            '-m',
            'mypy',
            *defaults,
            *extra_args,
            *required,
            '--package',
            package,
        ]

        try:
            process = subprocess.run(args, capture_output=True, cwd=root, text=True, check=False)
        except OSError as error:  # pragma: no cover - defensive
            msg = f'Could not run mypy: {error}'
            raise CheckerError(msg) from error

        # mypy exits 1 when it reports diagnostics and 2 when it could not run.
        if process.returncode > 1 or process.stderr:
            hint = ''
            if 'No module named mypy' in process.stderr:
                hint = '\n\nInstall it with: pip install type-assert[mypy]'
            msg = (
                f'mypy failed to run:\n{" ".join(args)}\n\n{process.stderr}{process.stdout}{hint}'
            )
            raise CheckerError(msg)

        diagnostics: dict[Path, list[Diagnostic]] = {}
        for line in process.stdout.splitlines():
            match = _DIAGNOSTIC.match(line)
            if match is None or match['severity'] != 'error':
                continue
            path = (Path(root) / match['path']).resolve()
            diagnostics.setdefault(path, []).append(
                Diagnostic(path=path, line=int(match['line']), message=match['msg'])
            )
        return diagnostics
