"""Group cases whose runtime half is skipped."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType


class skip_runtime:  # noqa: N801
    """Skip the runtime half of the cases in a `with` block, when `when` is true."""

    def __init__(self, reason: str, *, when: bool = True) -> None:
        """Record why the cases are skipped, and whether the skip applies."""
        self.reason = reason
        self.when = when

    def __enter__(self) -> None:
        """Enter the block; the plugin reads the block rather than running it."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leave the block without suppressing an exception."""
