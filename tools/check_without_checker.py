"""Check the package is usable with no type checker installed.

Run by the `test-no-checker` CI job. Neither mypy nor pyright is a hard
dependency, so importing the package and asserting a runtime type has to work
without them, and asking for a missing checker has to fail with a message that
names it.
"""

from __future__ import annotations

import importlib.util
import sys

from typeassert import CheckerError
from typeassert import assert_types
from typeassert._checkers import get_checker

for name in ('mypy', 'pyright'):
    if importlib.util.find_spec(name) is not None:
        sys.exit(f'{name} is installed; this check is meaningless')

assert_types([1, 2], list)
print('runtime half works with no checker installed')  # noqa: T201

try:
    get_checker('mypy').run(package='x', root=sys.path[0], cache_dir=None)
except CheckerError as error:
    assert 'mypy' in str(error), str(error)
    print(f'missing checker reported clearly: {error}')  # noqa: T201
else:
    sys.exit('expected a CheckerError naming the missing checker')
