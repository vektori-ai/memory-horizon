"""Base 7: Integration Gate — all 6 skills fire simultaneously.

Composite reward (from plan):
    Store (0.15) + Synthesize (0.20) + Contradict (0.20) + Compress (0.10)
    + Abstain (0.20) + Temporal (0.15)

Synthesize and Abstain weighted higher — map to the two biggest production failure modes.

Gate logic:
    Within 10pp of oracle (+1.0) → proceed to V1 vertical envs
    Gap > 15pp → iterate weakest base env (identified by component breakdown)
"""

from __future__ import annotations

from typing import Any, Callable

from memory_horizon.base_env import MemoryHorizonEnv, make_env
from memory_horizon.generator.session_gen import SessionGenerator
from memory_horizon.types import Trajectory


def make_integration_env(
    trajectory_fn: Callable[[], Trajectory] | None = None,
    qa_fn: Callable[[str, str], str] | None = None,
    **kwargs: Any,
) -> MemoryHorizonEnv:
    """Create the Base 7 (Integration Gate) environment."""
    if trajectory_fn is None:
        gen = SessionGenerator.from_config("base")
        trajectory_fn = gen.generate

    return make_env(
        env_type="integration",
        trajectory_fn=trajectory_fn,
        qa_fn=qa_fn,
        tier1_weight=0.3,
        tier3_weight=0.7,
        system_prompt_prefix=(
            "You are the complete memory manager. All memory operations are available. "
            "Store facts, resolve conflicts, compress when needed, abstain when uncertain, "
            "and always attach temporal markers to time-sensitive events."
        ),
        **kwargs,
    )


ENV_ID = "mh-base-integration-v1"
