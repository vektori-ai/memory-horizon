"""Base 5: Abstain — refuse to hallucinate when memory lacks the answer.

Skill: ABSTAIN correctly (know when you don't know)
Reward:
    Tier 1: abstention signal present when memory lacks answer (+1 / -1)
    Tier 2 (LLM judge — only here): "Did agent correctly identify lack of info? Score 0-1."
    Tier 3: on examples where memory DOES have the answer, measure answer correctness.
             Over-abstaining (refusing when answer is available) = -1.

Kill criteria: Hallucination rate > 10% after 8 runs.
"""

from __future__ import annotations

from typing import Any, Callable

from memory_horizon.base_env import MemoryHorizonEnv, make_env
from memory_horizon.generator.session_gen import SessionGenerator
from memory_horizon.mh_types import Trajectory


def make_abstain_env(
    trajectory_fn: Callable[[], Trajectory] | None = None,
    openai_api_key: str | None = None,
    **kwargs: Any,
) -> MemoryHorizonEnv:
    """Create the Base 5 (Abstain) environment.

    Args:
        openai_api_key: API key for GPT-4o Tier 2 judge. If None, Tier 2 is skipped.
    """
    if trajectory_fn is None:
        gen = SessionGenerator.from_config("base")
        trajectory_fn = gen.generate

    return make_env(
        env_type="abstain",
        trajectory_fn=trajectory_fn,
        tier1_weight=0.4,
        tier3_weight=0.6,
        system_prompt_prefix=(
            "You answer questions from memory. If the answer is in your memory, "
            "answer precisely. If it is NOT in your memory, you MUST abstain: "
            'output {"answer": "<abstain>I don\'t have that information"}. '
            "Never guess or hallucinate."
        ),
        **kwargs,
    )


ENV_ID = "mh-base-abstain-v1"
