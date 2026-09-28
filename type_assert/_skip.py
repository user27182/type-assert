"""Group cases whose runtime half is skipped."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType


class skip_runtime:  # noqa: N801
    """Skip the runtime half of the cases in a `with` block, when `condition` is true."""

    def __init__(self, condition: bool = True, *, reason: str) -> None:
        """Record whether the skip applies, and why the cases are skipped."""
        self.condition = condition
        self.reason = reason

    def __enter__(self) -> None:
        """Enter the block; the plugin reads the block rather than running it."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leave the block without suppressing an exception."""
