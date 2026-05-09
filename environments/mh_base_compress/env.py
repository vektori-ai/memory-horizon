"""Base 4: Compress — lossy summarization without QA degradation.

Skill: COMPRESS existing memory blocks, retaining QA-critical information
Reward:
    Tier 1: len(compressed) < len(original) * 0.80 (must compress ≥ 20%)
    Tier 3: QA accuracy delta < 3pp. Score = max(0, 1 - delta/0.03).
             Token penalty: reward -= 0.1 * (len(compressed) / len(original))

Kill criteria: compression ratio < 5% triggers pause (no-compression collapse).
"""

from __future__ import annotations

from typing import Any, Callable

from memory_horizon.base_env import MemoryHorizonEnv, make_env
from memory_horizon.generator.session_gen import SessionGenerator
from memory_horizon.mh_types import Trajectory


def make_compress_env(
    trajectory_fn: Callable[[], Trajectory] | None = None,
    **kwargs: Any,
) -> MemoryHorizonEnv:
    """Create the Base 4 (Compress) environment."""
    if trajectory_fn is None:
        gen = SessionGenerator.from_config("base")
        trajectory_fn = gen.generate

    return make_env(
        env_type="compress",
        trajectory_fn=trajectory_fn,
        tier1_weight=0.3,
        tier3_weight=0.7,
        system_prompt_prefix=(
            "You compress memory blocks to save space. Use COMPRESS to replace "
            "existing memory entries with shorter versions. You must reduce length "
            "by at least 20% while preserving all information needed to answer "
            "questions about the content."
        ),
        **kwargs,
    )


ENV_ID = "mh-base-compress-v1"
