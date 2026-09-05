"""Run a checker once per session when pytest-xdist spreads the cases over workers."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import TYPE_CHECKING
from typing import Callable

from ._checkers import CheckerError
from ._checkers import Diagnostic

if TYPE_CHECKING:
    import pytest

#: How long a worker waits for whichever worker is running the checker.
TIMEOUT = 600.0
POLL = 0.1


def shared_dir(config: pytest.Config) -> Path | None:
    """Return the directory every xdist worker shares, or `None` when not under xdist."""
    if not hasattr(config, 'workerinput'):
        return None
    factory = getattr(config, '_tmp_path_factory', None)
    if factory is None:  # pragma: no cover - pytest always provides it
        return None
    return Path(factory.getbasetemp()).parent / 'type_assert'


def _encode(diagnostics: dict[Path, list[Diagnostic]]) -> str:
    """Return the diagnostics as JSON."""
    return json.dumps(
        {
            str(path): [
                {'path': str(item.path), 'line': item.line, 'message': item.message}
                for item in items
            ]
            for path, items in diagnostics.items()
        }
    )


def _decode(payload: str) -> dict[Path, list[Diagnostic]]:
    """Return the diagnostics read back from JSON."""
    stored = json.loads(payload)
    if 'error' in stored:
        raise CheckerError(stored['error'])
    return {
        Path(path): [
            Diagnostic(path=Path(item['path']), line=item['line'], message=item['message'])
            for item in items
        ]
        for path, items in stored['diagnostics'].items()
    }


def run_once(
    config: pytest.Config, name: str, run: Callable[[], dict[Path, list[Diagnostic]]]
) -> dict[Path, list[Diagnostic]]:
    """Return `run()`, calling it in only one xdist worker and sharing the result.

    Without xdist, or if the workers cannot agree on a directory, `run` is called
    directly.
    """
    directory = shared_dir(config)
    if directory is None:
        return run()

    directory.mkdir(parents=True, exist_ok=True)
    result = directory / f'{name}.json'
    claim = directory / f'{name}.claim'
    deadline = time.monotonic() + TIMEOUT

    while True:
        if result.is_file():
            return _decode(result.read_text(encoding='utf-8'))
        try:
            claim.mkdir()
        except FileExistsError:
            if time.monotonic() > deadline:
                msg = f'Timed out waiting {TIMEOUT:.0f}s for another worker to run {name}.'
                raise CheckerError(msg) from None
            time.sleep(POLL)
            continue

        try:
            diagnostics = run()
        except CheckerError as error:
            _publish(result, json.dumps({'error': str(error)}))
            raise
        except BaseException as error:
            _publish(result, json.dumps({'error': f'{type(error).__name__}: {error}'}))
            raise
        _publish(result, json.dumps({'diagnostics': json.loads(_encode(diagnostics))}))
        return diagnostics


def _publish(result: Path, payload: str) -> None:
    """Write `payload` where the other workers will find it, in one step."""
    pending = result.with_suffix('.pending')
    pending.write_text(payload, encoding='utf-8')
    pending.replace(result)
