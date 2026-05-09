"""Base 3: Contradict — resolve conflicts between old memory and new claims.

Skill: Choose UPDATE / SUPERSEDE / DECAY / KEEP_BOTH correctly
Reward (fully deterministic — no Tier 2, no Tier 3):
    +1.0  correct conflict resolution op
    -0.5  no-op UPDATE (new value same as old)
    -2.0  SUPERSEDE without moving old value to audit log
    -2.0  KEEP_BOTH without distinct keys
    -2.0  keeping two conflicting current facts without resolution

Kill criteria: Always-UPDATE reward hack detected → pause.
Dataset: Each example is a labeled conflict pair {old_memory, new_claim, correct_op, why}.
"""

from __future__ import annotations

from typing import Any, Callable

from memory_horizon.base_env import MemoryHorizonEnv, make_env
from memory_horizon.generator.session_gen import SessionGenerator
from memory_horizon.mh_types import Trajectory


def make_contradict_env(
    trajectory_fn: Callable[[], Trajectory] | None = None,
    **kwargs: Any,
) -> MemoryHorizonEnv:
    """Create the Base 3 (Contradict) environment."""
    if trajectory_fn is None:
        gen = SessionGenerator.from_config("base")
        trajectory_fn = gen.generate

    return make_env(
        env_type="contradict",
        trajectory_fn=trajectory_fn,
        tier1_weight=1.0,   # Fully deterministic — all reward from Tier 1
        tier3_weight=0.0,
        system_prompt_prefix=(
            "You manage memory conflicts. When new information contradicts existing memory, "
            "choose: UPDATE (value changed), SUPERSEDE (old archived), "
            "DECAY (reduce confidence), or KEEP_BOTH (both valid). "
            "Never silently overwrite history."
        ),
        **kwargs,
    )


ENV_ID = "mh-base-contradict-v1"
