"""Sharing one checker run between pytest-xdist workers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from type_assert import CheckerError
from type_assert import Diagnostic
from type_assert._share import _decode
from type_assert._share import _encode
from type_assert._share import run_once
from type_assert._share import shared_dir


class FakeConfig:
    """Stands in for the pytest config the sharing helper reads."""

    def __init__(self, basetemp=None, worker=None):
        """Record what the helper looks for."""
        if worker is not None:
            self.workerinput = {'workerid': worker}
        if basetemp is not None:
            self._tmp_path_factory = FakeFactory(basetemp)


class FakeFactory:
    """Stands in for pytest's tmp path factory."""

    def __init__(self, basetemp):
        """Record the base directory."""
        self._basetemp = basetemp

    def getbasetemp(self):
        """Return the worker's own temporary directory."""
        return self._basetemp


@pytest.fixture
def worker(tmp_path):
    """Return a config that looks like an xdist worker."""

    def _worker(name='gw0'):
        basetemp = tmp_path / f'popen-{name}'
        basetemp.mkdir(exist_ok=True)
        return FakeConfig(basetemp=basetemp, worker=name)

    return _worker


DIAGNOSTICS = {Path('/a/b.py'): [Diagnostic(path=Path('/a/b.py'), line=3, message='boom')]}


def test_round_trips_diagnostics():
    assert _decode(json.dumps({'diagnostics': json.loads(_encode(DIAGNOSTICS))})) == DIAGNOSTICS


def test_no_shared_directory_without_xdist(tmp_path):
    assert shared_dir(FakeConfig(basetemp=tmp_path)) is None


def test_workers_agree_on_one_directory(worker):
    assert shared_dir(worker('gw0')) == shared_dir(worker('gw1'))


def test_runs_directly_when_not_under_xdist(tmp_path):
    calls = []

    def run():
        calls.append(1)
        return DIAGNOSTICS

    assert run_once(FakeConfig(basetemp=tmp_path), 'mypy', run) == DIAGNOSTICS
    assert run_once(FakeConfig(basetemp=tmp_path), 'mypy', run) == DIAGNOSTICS
    # Nothing to share without workers, so it runs every time.
    assert len(calls) == 2


def test_only_the_first_worker_runs_the_checker(worker):
    calls = []

    def run():
        calls.append(1)
        return DIAGNOSTICS

    first = run_once(worker('gw0'), 'mypy', run)
    second = run_once(worker('gw1'), 'mypy', run)
    third = run_once(worker('gw2'), 'mypy', run)

    assert first == second == third == DIAGNOSTICS
    assert len(calls) == 1


def test_each_checker_runs_once(worker):
    calls = []

    def run():
        calls.append(1)
        return DIAGNOSTICS

    run_once(worker('gw0'), 'mypy', run)
    run_once(worker('gw1'), 'pyright', run)
    run_once(worker('gw2'), 'mypy', run)
    assert len(calls) == 2


def test_a_checker_failure_reaches_the_other_workers(worker):
    def run():
        msg = 'mypy failed to run'
        raise CheckerError(msg)

    with pytest.raises(CheckerError, match='mypy failed to run'):
        run_once(worker('gw0'), 'mypy', run)

    def unreachable():  # pragma: no cover - the stored failure is raised first
        raise AssertionError

    with pytest.raises(CheckerError, match='mypy failed to run'):
        run_once(worker('gw1'), 'mypy', unreachable)


def test_an_unexpected_failure_also_reaches_the_other_workers(worker):
    def run():
        msg = 'something else'
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match='something else'):
        run_once(worker('gw0'), 'mypy', run)

    def unreachable():  # pragma: no cover - the stored failure is raised first
        raise AssertionError

    with pytest.raises(CheckerError, match='RuntimeError: something else'):
        run_once(worker('gw1'), 'mypy', unreachable)


def test_waiting_gives_up_rather_than_hanging(worker, monkeypatch):
    monkeypatch.setattr('type_assert._share.TIMEOUT', 0.0)
    config = worker('gw0')
    # A claim with no result is what a worker killed mid-run leaves behind.
    directory = shared_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'mypy.claim').mkdir()

    def unreachable():  # pragma: no cover - the wait gives up first
        raise AssertionError

    with pytest.raises(CheckerError, match='Timed out'):
        run_once(config, 'mypy', unreachable)
