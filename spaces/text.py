"""Text space — variable-length string observations/actions."""

from __future__ import annotations

import string
from typing import Any

from memory_horizon.spaces.space import Space

_DEFAULT_CHARSET = string.printable


class Text(Space[str]):
    """A space of variable-length Unicode strings.

    Useful for LLM-facing environments where observations are natural-language
    strings (terminal output, tool results, webpage content).

    Args:
        min_length: Minimum string length inclusive. Defaults to ``0``.
        max_length: Maximum string length inclusive. Defaults to ``1024``.
        charset: The set of allowed characters. Defaults to printable ASCII.

    Example:
        >>> space = Text(min_length=1, max_length=32)
        >>> space.seed(7)
        >>> s = space.sample()
        >>> space.contains(s)
        True
        >>> space.contains("")  # below min_length
        False
    """

    def __init__(
        self,
        min_length: int = 0,
        max_length: int = 1024,
        charset: str = _DEFAULT_CHARSET,
    ) -> None:
        if min_length < 0:
            raise ValueError(f"min_length must be >= 0, got {min_length}")
        if max_length < min_length:
            raise ValueError(f"max_length ({max_length}) must be >= min_length ({min_length})")
        if not charset:
            raise ValueError("charset must not be empty")
        self.min_length = min_length
        self.max_length = max_length
        self.charset = charset
        self._charset_list = list(charset)
        super().__init__(shape=None, dtype=None)

    def sample(self, mask: Any = None) -> str:
        """Sample a random string of length in ``[min_length, max_length]``."""
        rng = self._np_random
        length = int(rng.integers(self.min_length, self.max_length + 1))
        indices = rng.integers(0, len(self._charset_list), size=length)
        return "".join(self._charset_list[i] for i in indices)

    def contains(self, x: Any) -> bool:
        """Return whether ``x`` is a valid string in this space."""
        if not isinstance(x, str):
            return False
        if not (self.min_length <= len(x) <= self.max_length):
            return False
        if self.charset != _DEFAULT_CHARSET:
            return all(c in self.charset for c in x)
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Text):
            return False
        return (
            self.min_length == other.min_length
            and self.max_length == other.max_length
            and self.charset == other.charset
        )

    def __repr__(self) -> str:
        return f"Text(min_length={self.min_length}, max_length={self.max_length})"
