"""Check the package is usable with no type checker installed.

Run by the `test-no-checker` CI job. Neither mypy nor pyright is a hard
dependency, so importing the package and asserting a runtime type has to work
without them, and asking for a missing checker has to fail with a message that
names it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from type_assert import CheckerError
from type_assert import assert_types
from type_assert._checkers import get_checker

for name in ('mypy', 'pyright'):
    if importlib.util.find_spec(name) is not None:
        sys.exit(f'{name} is installed; this check is meaningless')

assert_types([1, 2], list)
print('runtime half works with no checker installed')  # noqa: T201

reported = None
try:
    get_checker('mypy').run(package='x', root=Path.cwd(), cache_dir=None)
except CheckerError as error:
    reported = str(error)

if reported is None:
    sys.exit('expected a CheckerError naming the missing checker')
if 'mypy' not in reported:
    sys.exit(f'the error did not name the missing checker: {reported}')
print('missing checker reported clearly')  # noqa: T201
