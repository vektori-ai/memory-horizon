"""Base Wrapper class and typed sub-wrapper helpers."""

from __future__ import annotations

from typing import Any, SupportsFloat

from memory_horizon.core import Env, EnvSpec, ObsType, ActType
from memory_horizon.spaces.space import Space


class Wrapper(Env[ObsType, ActType]):
    """Wraps an environment to allow modular transformation.

    Subclass and override any of :meth:`step`, :meth:`reset`, :meth:`render`,
    or :meth:`close` to apply a single transformation. All other calls are
    forwarded to the wrapped environment transparently.

    Args:
        env: The environment to wrap.

    Example:
        >>> class ScaleReward(Wrapper):
        ...     def __init__(self, env: Env, scale: float = 0.01) -> None:
        ...         super().__init__(env)
        ...         self.scale = scale
        ...
        ...     def step(self, action):
        ...         obs, reward, terminated, truncated, info = self.env.step(action)
        ...         return obs, float(reward) * self.scale, terminated, truncated, info
        >>> env = ScaleReward(SomeEnv(), scale=0.01)
    """

    def __init__(self, env: Env[ObsType, ActType]) -> None:
        self.env = env

    @property
    def action_space(self) -> Space[ActType]:  # type: ignore[override]
        return self.env.action_space  # type: ignore[return-value]

    @property
    def observation_space(self) -> Space[ObsType]:  # type: ignore[override]
        return self.env.observation_space  # type: ignore[return-value]

    @property
    def metadata(self) -> dict[str, Any]:  # type: ignore[override]
        return self.env.metadata

    @property
    def render_mode(self) -> str | None:  # type: ignore[override]
        return self.env.render_mode

    @property
    def spec(self) -> EnvSpec | None:  # type: ignore[override]
        return self.env.spec

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return self.env.step(action)

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    @property
    def unwrapped(self) -> Env[ObsType, ActType]:
        """Return the base environment, unwrapping all wrappers."""
        env = self.env
        while isinstance(env, Wrapper):
            env = env.env
        return env

    def __repr__(self) -> str:
        return f"<{type(self).__name__}({self.env!r})>"
