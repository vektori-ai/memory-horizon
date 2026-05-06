"""Box space — continuous N-dimensional hypercube."""

from __future__ import annotations

from typing import Any

import numpy as np

from memory_horizon.spaces.space import Space


class Box(Space[np.ndarray]):
    """A continuous N-dimensional box.

    Observations/actions lie in the closed interval ``[low, high]`` for each
    dimension. Supports infinite bounds via ``-np.inf`` / ``np.inf``.

    Args:
        low: Lower bound(s). Scalar or array-like broadcastable to ``shape``.
        high: Upper bound(s). Scalar or array-like broadcastable to ``shape``.
        shape: Shape of a single sample. Inferred from ``low``/``high`` when
            both are array-like with matching shapes.
        dtype: NumPy dtype for samples. Defaults to ``float32``.

    Example:
        >>> box = Box(low=-1.0, high=1.0, shape=(3,))
        >>> box.seed(0)
        >>> sample = box.sample()
        >>> sample.shape
        (3,)
        >>> box.contains(sample)
        True
    """

    def __init__(
        self,
        low: float | np.ndarray,
        high: float | np.ndarray,
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype | type = np.float32,
    ) -> None:
        dtype = np.dtype(dtype)
        if shape is None:
            if np.isscalar(low) and np.isscalar(high):
                raise ValueError("shape must be specified when low and high are scalars")
            low_arr = np.asarray(low, dtype=dtype)
            high_arr = np.asarray(high, dtype=dtype)
            if low_arr.shape != high_arr.shape:
                raise ValueError(
                    f"low.shape={low_arr.shape} and high.shape={high_arr.shape} must match"
                )
            shape = low_arr.shape
        else:
            low_arr = np.full(shape, low, dtype=dtype)
            high_arr = np.full(shape, high, dtype=dtype)

        if not np.all(low_arr <= high_arr):
            raise ValueError("low must be <= high for all dimensions")

        self.low = low_arr
        self.high = high_arr
        super().__init__(shape=shape, dtype=dtype)

    @property
    def bounded_below(self) -> np.ndarray:
        """Boolean mask: ``True`` where the lower bound is finite."""
        return np.isfinite(self.low)

    @property
    def bounded_above(self) -> np.ndarray:
        """Boolean mask: ``True`` where the upper bound is finite."""
        return np.isfinite(self.high)

    def sample(self, mask: Any = None) -> np.ndarray:
        """Sample uniformly within bounds, using exponential/normal for unbounded dims."""
        rng = self._np_random
        sample = np.empty(self.shape, dtype=self.dtype)

        both = self.bounded_below & self.bounded_above
        if np.any(both):
            sample[both] = rng.uniform(self.low[both], self.high[both])

        below_only = self.bounded_below & ~self.bounded_above
        if np.any(below_only):
            sample[below_only] = self.low[below_only] + rng.exponential(size=np.sum(below_only))

        above_only = ~self.bounded_below & self.bounded_above
        if np.any(above_only):
            sample[above_only] = self.high[above_only] - rng.exponential(size=np.sum(above_only))

        unbounded = ~self.bounded_below & ~self.bounded_above
        if np.any(unbounded):
            sample[unbounded] = rng.standard_normal(size=np.sum(unbounded))

        if np.issubdtype(self.dtype, np.integer):
            sample = np.floor(sample).astype(self.dtype)

        return sample

    def contains(self, x: Any) -> bool:
        """Return whether ``x`` lies within ``[low, high]``."""
        if not isinstance(x, np.ndarray):
            try:
                x = np.asarray(x, dtype=self.dtype)
            except (ValueError, TypeError):
                return False
        return bool(
            x.shape == self.shape
            and np.can_cast(x.dtype, self.dtype)
            and np.all(x >= self.low)
            and np.all(x <= self.high)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Box):
            return False
        return (
            self.shape == other.shape
            and self.dtype == other.dtype
            and np.array_equal(self.low, other.low)
            and np.array_equal(self.high, other.high)
        )

    def __repr__(self) -> str:
        return (
            f"Box(low={self.low.tolist()}, high={self.high.tolist()}, "
            f"shape={self.shape}, dtype={self.dtype})"
        )
