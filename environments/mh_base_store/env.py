"""Base 1: Store — teach the model when and how to store information.

Skill: Output STORE_FACT / CREATE_EPISODE / INFER_IMPLICIT
Reward:
    Tier 1 (0.3 weight): schema valid + key valid + content not verbatim copy
    Tier 3 (0.7 weight): frozen QA model answers correctly using stored memory (+1 / 0)

Kill criteria: 8 runs without convergence.
"""

from __future__ import annotations

from typing import Callable

from memory_horizon.base_env import MemoryHorizonEnv, make_env
from memory_horizon.generator.session_gen import SessionGenerator
from memory_horizon.mh_types import Trajectory


def make_store_env(
    trajectory_fn: Callable[[], Trajectory] | None = None,
    qa_fn: Callable[[str, str], str] | None = None,
    **kwargs,
) -> MemoryHorizonEnv:
    """Create the Base 1 (Store) environment.

    Args:
        trajectory_fn: Callable returning a Trajectory. Defaults to built-in generator.
        qa_fn: Frozen QA model callable. Defaults to mock (token-recall heuristic).
        **kwargs: Forwarded to MemoryHorizonEnv.
    """
    if trajectory_fn is None:
        gen = SessionGenerator.from_config("base")
        trajectory_fn = gen.generate

    return make_env(
        env_type="store",
        trajectory_fn=trajectory_fn,
        qa_fn=qa_fn,
        tier1_weight=0.3,
        tier3_weight=0.7,
        system_prompt_prefix=(
            "You are training to store information from conversations into memory. "
            "For each turn, decide what to store and use STORE_FACT, CREATE_EPISODE, "
            "or INFER_IMPLICIT as appropriate."
        ),
        **kwargs,
    )


ENV_ID = "mh-base-store-v1"
ENV_DESCRIPTION = (
    "Base 1 (Store): Teaches the memory manager when and how to store information "
    "from conversations. Reward flows from a frozen QA model's ability to answer "
    "probes using only the stored memory (Memory-R1 pattern)."
)
