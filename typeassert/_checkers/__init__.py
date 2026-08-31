"""Type checker backends, and the registry that selects one."""

from __future__ import annotations

from ._base import Checker
from ._base import CheckerError
from ._base import Diagnostic
from ._mypy import MypyChecker
from ._pyright import PyrightChecker

__all__ = [
    'CHECKERS',
    'Checker',
    'CheckerError',
    'Diagnostic',
    'MypyChecker',
    'PyrightChecker',
    'get_checker',
]

#: Every checker that can be selected, by name.
#:
#: `ty` is deliberately absent: it is pre-1.0 and its output format is still moving,
#: so supporting it would mean tracking those changes. Adding a backend is one module
#: implementing `Checker` plus an entry here.
CHECKERS: dict[str, type[Checker]] = {
    MypyChecker.name: MypyChecker,
    PyrightChecker.name: PyrightChecker,
}


def get_checker(name: str) -> Checker:
    """Return the checker called `name`."""
    try:
        checker = CHECKERS[name]
    except KeyError:
        known = ', '.join(sorted(CHECKERS))
        msg = f'Unknown type checker {name!r}. Available: {known}.'
        raise CheckerError(msg) from None
    return checker()
