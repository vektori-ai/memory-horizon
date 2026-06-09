"""Discrete space — finite set of integers."""

from __future__ import annotations

from typing import Any

import numpy as np

from memory_horizon.spaces.space import Space


class Discrete(Space[np.intp]):
    """A space of ``n`` integers in ``{start, start+1, ..., start+n-1}``.

    Args:
        n: Number of discrete values (must be > 0).
        start: First value in the set. Defaults to ``0``.

    Example:
        >>> space = Discrete(4)
        >>> space.seed(0)
        >>> action = space.sample()
        >>> 0 <= int(action) < 4
        True
        >>> space.contains(3)
        True
        >>> space.contains(4)
        False
    """

    def __init__(self, n: int, start: int = 0) -> None:
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        self.n = n
        self.start = start
        super().__init__(shape=(), dtype=np.dtype(np.int64))

    def sample(self, mask: np.ndarray | None = None) -> np.intp:
        """Sample a single integer from the space."""
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (self.n,):
                raise ValueError(f"mask.shape must be ({self.n},), got {mask.shape}")
            valid = np.where(mask)[0]
            if len(valid) == 0:
                raise ValueError("mask contains no valid actions")
            return np.intp(self.start + self._np_random.choice(valid))
        return np.intp(self.start + self._np_random.integers(0, self.n))

    def contains(self, x: Any) -> bool:
        """Return whether ``x`` is a valid integer in this space."""
        if isinstance(x, (int, np.integer)):
            return bool(self.start <= int(x) < self.start + self.n)
        return False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Discrete):
            return False
        return self.n == other.n and self.start == other.start

    def __repr__(self) -> str:
        if self.start == 0:
            return f"Discrete({self.n})"
        return f"Discrete({self.n}, start={self.start})"
