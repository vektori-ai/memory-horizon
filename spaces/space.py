"""Abstract base class for action and observation spaces."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np

T = TypeVar("T")


class Space(Generic[T]):
    """Abstract base class for spaces.

    A *space* describes the set of valid observations or actions for an
    environment. Concrete subclasses must implement :meth:`sample` and
    :meth:`contains`.

    Example:
        >>> space = Discrete(4)
        >>> space.seed(42)
        >>> action = space.sample()
        >>> assert space.contains(action)
    """

    def __init__(self, shape: tuple[int, ...] | None = None, dtype: np.dtype | None = None) -> None:
        self._shape = shape
        self._dtype = dtype
        self._np_random: np.random.Generator = np.random.default_rng()

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Shape of a single sample from this space, or ``None`` for non-array spaces."""
        return self._shape

    @property
    def dtype(self) -> np.dtype | None:
        """NumPy dtype of samples, or ``None`` for non-array spaces."""
        return self._dtype

    @abstractmethod
    def sample(self, mask: Any = None) -> T:
        """Randomly sample a point from the space."""

    @abstractmethod
    def contains(self, x: Any) -> bool:
        """Return whether ``x`` is a valid element of this space."""

    def seed(self, seed: int | None = None) -> list[int]:
        """Seed the space's RNG."""
        self._np_random = np.random.default_rng(seed)
        return [seed] if seed is not None else []

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
