"""Core environment and wrapper base classes."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generic, SupportsFloat, TypeVar

from memory_horizon.spaces.space import Space

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class Env(Generic[ObsType, ActType]):
    """Abstract base class for all RL environments.

    All environments must implement :meth:`reset` and :meth:`step`. The
    :attr:`action_space` and :attr:`observation_space` attributes must be set
    before any interaction. Follows the Gymnasium v26 API strictly.

    Type parameters:
        ObsType: The type of observations produced by the environment.
        ActType: The type of actions accepted by the environment.

    Example:
        >>> env = SomeConcreteEnv()
        >>> obs, info = env.reset(seed=42)
        >>> obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        >>> env.close()
    """

    action_space: Space[ActType]
    observation_space: Space[ObsType]
    metadata: dict[str, Any] = {}
    render_mode: str | None = None

    # Populated by the registry when constructed via make().
    spec: EnvSpec | None = None

    @abstractmethod
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment to an initial state and return an observation.

        Args:
            seed: Random seed for reproducibility. Pass ``None`` to use the
                environment's existing RNG.
            options: Environment-specific configuration overrides for this
                episode (e.g. start position, difficulty level).

        Returns:
            observation: The initial observation for the new episode.
            info: A dict of auxiliary information about the reset.

        Raises:
            RuntimeError: If the environment cannot be reset (e.g. a required
                service is not running).
        """

    @abstractmethod
    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics.

        When the episode ends (``terminated`` or ``truncated`` is ``True``) the
        caller must call :meth:`reset` before the next :meth:`step`.

        Args:
            action: An action in :attr:`action_space`.

        Returns:
            observation: Agent's observation of the current environment state.
            reward: Scalar reward for taking ``action`` in the current state.
            terminated: ``True`` if the episode ended naturally (goal reached
                or irrecoverable failure).
            truncated: ``True`` if the episode was cut short by an external
                constraint (time limit, safety abort).
            info: Auxiliary diagnostic information. Always a dict; may be
                empty. Contents are environment-defined and should not be used
                by agents for learning.

        Raises:
            RuntimeError: If :meth:`reset` has not been called since the last
                terminal step.
        """

    def render(self) -> Any:
        """Return a visual representation of the current environment state."""
        return None

    def close(self) -> None:
        """Clean up resources held by the environment."""

    def __enter__(self) -> "Env[ObsType, ActType]":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        if self.spec is not None:
            return f"<{type(self).__name__} id={self.spec.id!r}>"
        return f"<{type(self).__name__}>"

    def _np_random(self, seed: int | None) -> None:
        """Seed the environment's NumPy RNG and store it as ``self.np_random``."""
        import numpy as np

        self.np_random, _ = np.random.default_rng(seed), seed


class EnvSpec:
    """Immutable specification for a registered environment.

    Attributes:
        id: Unique string identifier, e.g. ``"MemoryHorizon-v1"``.
        entry_point: Dotted import path to the environment class.
        max_episode_steps: Optional hard truncation limit.
        reward_threshold: Reward considered "solved" for this environment.
        kwargs: Default keyword arguments forwarded to the constructor.
    """

    __slots__ = ("id", "entry_point", "max_episode_steps", "reward_threshold", "kwargs")

    def __init__(
        self,
        id: str,  # noqa: A002
        entry_point: str,
        *,
        max_episode_steps: int | None = None,
        reward_threshold: float | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.id = id
        self.entry_point = entry_point
        self.max_episode_steps = max_episode_steps
        self.reward_threshold = reward_threshold
        self.kwargs: dict[str, Any] = kwargs or {}

    def make(self, **override_kwargs: Any) -> Env:  # type: ignore[type-arg]
        """Instantiate the environment described by this spec."""
        module_path, class_name = self.entry_point.rsplit(":", 1)
        import importlib

        module = importlib.import_module(module_path)
        cls: type[Env] = getattr(module, class_name)  # type: ignore[type-arg]
        kwargs = {**self.kwargs, **override_kwargs}
        env = cls(**kwargs)
        env.spec = self
        if self.max_episode_steps is not None:
            from memory_horizon.wrappers.time_limit import TimeLimit

            env = TimeLimit(env, max_episode_steps=self.max_episode_steps)
        return env

    def __repr__(self) -> str:
        return (
            f"EnvSpec(id={self.id!r}, entry_point={self.entry_point!r}, "
            f"max_episode_steps={self.max_episode_steps!r})"
        )
