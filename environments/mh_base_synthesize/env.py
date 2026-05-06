"""Base 2: Synthesize — given oracle memory, produce correct answers without hallucinating.

Skill: Synthesize stored facts into correct answers
Reward:
    Tier 1 (hallucination pre-check): -2 if numeric/entity not in oracle memory
    Tier 3 (token F1 vs gold): F1 ≥ 0.7 → +1; 0.4–0.7 → partial; < 0.4 → 0
"""

from __future__ import annotations

from typing import Any, Callable

from memory_horizon.base_env import MemoryHorizonEnv, make_env
from memory_horizon.generator.session_gen import SessionGenerator
from memory_horizon.types import Trajectory


def make_synthesize_env(
    trajectory_fn: Callable[[], Trajectory] | None = None,
    **kwargs: Any,
) -> MemoryHorizonEnv:
    """Create the Base 2 (Synthesize) environment."""
    if trajectory_fn is None:
        gen = SessionGenerator.from_config("base")
        trajectory_fn = gen.generate

    return make_env(
        env_type="synthesize",
        trajectory_fn=trajectory_fn,
        tier1_weight=0.2,
        tier3_weight=0.8,
        system_prompt_prefix=(
            "You are given a memory context. Answer questions using ONLY the provided "
            "memory. Do not hallucinate. If the answer is not in memory, abstain."
        ),
        **kwargs,
    )


ENV_ID = "mh-base-synthesize-v1"
