"""TimeLimit wrapper — truncate episodes at a maximum step count."""

from __future__ import annotations

from typing import Any, SupportsFloat

from memory_horizon.wrappers.core import Wrapper


class TimeLimit(Wrapper):
    """Truncate an episode after ``max_episode_steps`` steps.

    Sets ``truncated=True`` when the step limit is reached. The wrapped
    environment's ``terminated`` flag is preserved unchanged.

    Args:
        env: The environment to wrap.
        max_episode_steps: Maximum number of steps per episode.

    Example:
        >>> env = TimeLimit(SomeEnv(), max_episode_steps=100)
        >>> obs, info = env.reset()
        >>> for _ in range(100):
        ...     obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        ...     if terminated or truncated:
        ...         break
        >>> truncated  # True after exactly 100 steps if not naturally terminated
        True
    """

    def __init__(self, env: Any, max_episode_steps: int) -> None:
        super().__init__(env)
        if max_episode_steps <= 0:
            raise ValueError(f"max_episode_steps must be > 0, got {max_episode_steps}")
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps: int = 0

    @property
    def max_episode_steps(self) -> int:
        """Maximum number of steps allowed per episode."""
        return self._max_episode_steps

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        self._elapsed_steps = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
        info["elapsed_steps"] = self._elapsed_steps
        info["max_episode_steps"] = self._max_episode_steps
        return obs, reward, terminated, truncated, info

    def __repr__(self) -> str:
        return f"<TimeLimit({self.env!r}, max_episode_steps={self._max_episode_steps})>"
