"""Verifiers for memory_horizon environments.

    Tier 1 — deterministic, zero LLM cost (tier1_deterministic.py)
    Tier 3 — outcome-based: token F1, Kendall's tau, QA recall (tier3_outcome.py)
    hallucination.py — entity/numeric grounding check against oracle memory
"""

from memory_horizon.verifiers.tier1_deterministic import Tier1Verifier, Tier1Result
from memory_horizon.verifiers.hallucination import HallucinationDetector, HallucinationResult

__all__ = [
    "Tier1Verifier",
    "Tier1Result",
    "HallucinationDetector",
    "HallucinationResult",
]
