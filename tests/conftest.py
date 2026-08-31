"""Shared fixtures for the test suite."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest_plugins = ['pytester']

REPO_ROOT = Path(__file__).parent.parent


@pytest.fixture
def checker_root(tmp_path, monkeypatch):
    """Return a project root the checkers can resolve `type_assert` from.

    The project installs itself editable through an import finder, which neither
    mypy nor pyright can follow, so point both at the source tree instead. This is
    a fact about developing this package: an ordinary installation puts `type_assert`
    in site-packages, where both find it with no help.
    """
    monkeypatch.setenv('MYPYPATH', str(REPO_ROOT))
    (tmp_path / 'pyrightconfig.json').write_text(
        json.dumps({'extraPaths': [str(REPO_ROOT)]}), encoding='utf-8'
    )
    return tmp_path
