"""Dict space — named collection of spaces."""

from __future__ import annotations

from typing import Any

from memory_horizon.spaces.space import Space


class Dict(Space[dict[str, Any]]):
    """A named mapping from string keys to sub-spaces.

    Useful for structured observations where each key carries a semantically
    distinct modality (e.g. screenshot, cursor position, file tree).

    Args:
        spaces: Mapping from key name to :class:`Space`.

    Example:
        >>> space = Dict({"screenshot": Box(0, 255, shape=(84, 84, 3), dtype=np.uint8),
        ...               "cursor": Box(0.0, 1.0, shape=(2,))})
        >>> space.seed(0)
        >>> obs = space.sample()
        >>> space.contains(obs)
        True
    """

    def __init__(self, spaces: dict[str, Space[Any]]) -> None:
        if not spaces:
            raise ValueError("spaces dict must not be empty")
        self.spaces = dict(spaces)
        super().__init__(shape=None, dtype=None)

    def sample(self, mask: dict[str, Any] | None = None) -> dict[str, Any]:
        """Sample from all sub-spaces."""
        mask = mask or {}
        return {key: space.sample(mask.get(key)) for key, space in self.spaces.items()}

    def contains(self, x: Any) -> bool:
        """Return whether ``x`` is a valid observation for this space."""
        if not isinstance(x, dict):
            return False
        if set(x.keys()) != set(self.spaces.keys()):
            return False
        return all(self.spaces[k].contains(v) for k, v in x.items())

    def seed(self, seed: int | None = None) -> list[int]:
        """Seed all sub-spaces deterministically from ``seed``."""
        import numpy as np

        seeds: list[int] = []
        rng = np.random.default_rng(seed)
        for space in self.spaces.values():
            child_seed = int(rng.integers(0, 2**31))
            space.seed(child_seed)
            seeds.append(child_seed)
        return seeds

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dict):
            return False
        return self.spaces == other.spaces

    def __repr__(self) -> str:
        inner = ", ".join(f"{k!r}: {v!r}" for k, v in self.spaces.items())
        return f"Dict({{{inner}}})"
