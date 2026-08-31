"""The pyright backend."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from ._base import Checker
from ._base import CheckerError
from ._base import Diagnostic


class PyrightChecker(Checker):
    """Runs pyright and reads its JSON output."""

    name = 'pyright'
    distribution = 'pyright'

    def run(
        self, package: str, *, root: Path, cache_dir: Path | None
    ) -> dict[Path, list[Diagnostic]]:
        """Type-check `package` from `root` and return pyright's errors keyed by file."""
        del cache_dir  # pyright keeps no cache of its own to point elsewhere.
        # pyright reports only on the files it is given, so unlike mypy it needs no
        # equivalent of `--follow-imports=silent` to stay quiet about the host project.
        target = Path(root) / Path(*package.split('.'))
        args = [
            sys.executable,
            '-m',
            'pyright',
            '--outputjson',
            '--pythonpath',
            sys.executable,
            str(target),
        ]

        try:
            process = subprocess.run(args, capture_output=True, cwd=root, text=True, check=False)
        except OSError as error:  # pragma: no cover - defensive
            msg = f'Could not run pyright: {error}'
            raise CheckerError(msg) from error

        try:
            report = json.loads(process.stdout)
        except json.JSONDecodeError as error:
            hint = ''
            if 'No module named pyright' in process.stderr:
                hint = '\n\nInstall it with: pip install type-assert[pyright]'
            msg = (
                f'pyright failed to run:\n{" ".join(args)}\n\n'
                f'{process.stderr}{process.stdout}{hint}'
            )
            raise CheckerError(msg) from error

        diagnostics: dict[Path, list[Diagnostic]] = {}
        for entry in report.get('generalDiagnostics', []):
            if entry.get('severity') != 'error':
                continue
            path = Path(entry['file']).resolve()
            # pyright counts lines from zero; everything else here counts from one.
            line = entry['range']['start']['line'] + 1
            diagnostics.setdefault(path, []).append(
                Diagnostic(path=path, line=line, message=entry['message'].replace('\n', ' '))
            )
        return diagnostics
