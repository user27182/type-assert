"""What every type checker backend has to provide."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class CheckerError(Exception):
    """Raised when a checker could not run, as opposed to reporting diagnostics."""


@dataclass(frozen=True)
class Diagnostic:
    """One error a checker reported, at one line of one file."""

    path: Path
    line: int
    message: str


class Checker:
    """Runs a type checker over a package and returns its errors keyed by file.

    Subclasses supply the command to run and how to read its output. Both run the
    checker in a separate process, so a crash surfaces as a failed call rather
    than taking the test session down with it.
    """

    #: Name this checker is selected by.
    name: str
    #: Distribution to install to get it, when it is missing.
    distribution: str

    def run(
        self,
        package: str,
        *,
        root: Path,
        cache_dir: Path | None,
        extra_args: Sequence[str] = (),
    ) -> dict[Path, list[Diagnostic]]:
        """Type-check `package` from `root` and return its errors keyed by file.

        Running from `root` is what makes the project's own checker configuration
        apply, since that is where every checker looks for it. `extra_args` covers
        what the configuration cannot say: a different config file, a Python
        version, a strictness flag.
        """
        raise NotImplementedError
