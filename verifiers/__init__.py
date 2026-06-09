"""Verifier suite for memory_horizon environments.

Three tiers:
    Tier 1 — deterministic, zero LLM cost (tier1_deterministic.py)
    Tier 2 — GPT-4o judge, used ONLY for Base 5 (Abstain) (tier2_llm_judge.py)
    Tier 3 — outcome-based: frozen QA model, token F1, Kendall's tau (tier3_outcome.py)

Plus a standalone hallucination detector (hallucination.py).
"""

from memory_horizon.verifiers.tier1_deterministic import Tier1Verifier, Tier1Result
from memory_horizon.verifiers.hallucination import HallucinationDetector, HallucinationResult

__all__ = [
    "Tier1Verifier",
    "Tier1Result",
    "HallucinationDetector",
    "HallucinationResult",
]
