"""Base 6: Temporal — store events with temporal markers, reconstruct timelines.

Skill: Attach ISO timestamps / relative markers to stored events; order up to 50 events
Reward:
    Tier 1: temporal markers present in stored memory; event count matches ground truth
    Tier 3: Kendall's tau ≥ 0.9 → +1 (continuous, clipped to 0 for negatives)
             Timeline QA: "Which happened first, X or Y?" — 10 probes per example
             -1 per hallucinated timestamp not derivable from conversation

Kill criteria: held-out eval < 30% after 8 runs → pause; switch to GEPA.
"""

from __future__ import annotations

from typing import Any, Callable

from memory_horizon.base_env import MemoryHorizonEnv, make_env
from memory_horizon.generator.session_gen import SessionGenerator
from memory_horizon.types import Trajectory


def make_temporal_env(
    trajectory_fn: Callable[[], Trajectory] | None = None,
    **kwargs: Any,
) -> MemoryHorizonEnv:
    """Create the Base 6 (Temporal) environment."""
    if trajectory_fn is None:
        gen = SessionGenerator.from_config("base")
        trajectory_fn = gen.generate

    return make_env(
        env_type="temporal",
        trajectory_fn=trajectory_fn,
        tier1_weight=0.3,
        tier3_weight=0.7,
        system_prompt_prefix=(
            "You store events with precise temporal markers. "
            "Always include a timestamp (ISO-8601) or relative marker "
            "('3 days after first call', 'second complaint') in the temporal_markers field. "
            "Never hallucinate dates not mentioned in the conversation."
        ),
        **kwargs,
    )


ENV_ID = "mh-base-temporal-v1"
