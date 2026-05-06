"""RecordEpisode wrapper — capture full episode trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, SupportsFloat

from memory_horizon.wrappers.core import Wrapper


@dataclass
class Transition:
    """A single step in a recorded trajectory."""

    observation: Any
    action: Any
    reward: float
    next_observation: Any
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass
class Episode:
    """A complete recorded episode."""

    initial_observation: Any
    transitions: list[Transition] = field(default_factory=list)
    seed: int | None = None

    @property
    def total_reward(self) -> float:
        """Undiscounted sum of rewards over the episode."""
        return sum(t.reward for t in self.transitions)

    @property
    def length(self) -> int:
        """Number of steps taken."""
        return len(self.transitions)

    @property
    def terminated(self) -> bool:
        """True if the last step was a natural termination."""
        return bool(self.transitions and self.transitions[-1].terminated)

    @property
    def truncated(self) -> bool:
        """True if the last step was a truncation."""
        return bool(self.transitions and self.transitions[-1].truncated)


class RecordEpisode(Wrapper):
    """Record every step of every episode into :class:`Episode` objects.

    Useful for replay, offline RL datasets, and debugging.

    Args:
        env: The environment to wrap.
        max_episodes: Cap on how many completed episodes to keep in memory.
            Older episodes are dropped when the cap is exceeded. ``None``
            keeps all episodes.

    Example:
        >>> recorder = RecordEpisode(SomeEnv(), max_episodes=10)
        >>> obs, info = recorder.reset(seed=0)
        >>> for _ in range(5):
        ...     obs, reward, terminated, truncated, info = recorder.step(
        ...         recorder.action_space.sample()
        ...     )
        >>> ep = recorder.episodes[-1]
        >>> ep.length
        5
    """

    def __init__(self, env: Any, *, max_episodes: int | None = None) -> None:
        super().__init__(env)
        self.max_episodes = max_episodes
        self.episodes: list[Episode] = []
        self._current_episode: Episode | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        obs, info = self.env.reset(seed=seed, options=options)
        self._current_episode = Episode(initial_observation=obs, seed=seed)
        return obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        if self._current_episode is None:
            raise RuntimeError("reset() must be called before step()")
        prev_obs = (
            self._current_episode.transitions[-1].next_observation
            if self._current_episode.transitions
            else self._current_episode.initial_observation
        )
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_episode.transitions.append(
            Transition(
                observation=prev_obs,
                action=action,
                reward=float(reward),
                next_observation=obs,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        )
        if terminated or truncated:
            self.episodes.append(self._current_episode)
            if self.max_episodes is not None and len(self.episodes) > self.max_episodes:
                self.episodes.pop(0)
            self._current_episode = None
        return obs, reward, terminated, truncated, info

    @property
    def current_episode(self) -> Episode | None:
        """The in-progress episode, or ``None`` if between episodes."""
        return self._current_episode
