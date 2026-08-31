"""Check a value's static type and its runtime type in one assertion.

A case file is a stack of one-line `assert_types(expression, ExpectedType)` calls.
A type checker checks the left of each pair against the right exactly; running the
line checks the value the expression actually produces. The pytest plugin turns
each line into two tests, so a disagreement between the two fails.
"""

from __future__ import annotations

from ._assertions import assert_types as assert_types
from ._cases import Case as Case
from ._cases import CaseError as CaseError
from ._cases import CaseFile as CaseFile
from ._cases import CaseSkipped as CaseSkipped
from ._cases import collect_case_file as collect_case_file
from ._cases import collect_cases as collect_cases
from ._checkers import CHECKERS as CHECKERS
from ._checkers import Checker as Checker
from ._checkers import CheckerError as CheckerError
from ._checkers import Diagnostic as Diagnostic
from ._checkers import get_checker as get_checker

try:
    from ._version import __version__ as __version__
except ImportError:  # pragma: no cover - only when running from a source tree
    __version__ = '0.0.0.dev0'
