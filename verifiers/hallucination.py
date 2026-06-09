"""Hallucination detector for memory_horizon.

Deterministic — zero LLM cost. Used in Base 2 (Synthesize) and Base 5 (Abstain).

The core idea (from the plan): a claim is hallucinated if it contains an entity or
numeric value that cannot be traced back to the oracle memory. We do NOT use an LLM
to judge this — we use string/entity matching against the oracle memory set.

Design notes:
- This is intentionally conservative: it only flags things that are *provably*
  absent from memory, not things that are merely unlikely.
- False negatives (hallucinations we miss) are acceptable — Tier 3 catches them.
- False positives (flagging correct claims as hallucinations) are expensive: -2.
  So we use generous matching (substring, normalized form, numeric proximity).
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class HallucinationResult:
    """Result from the hallucination detector.

    Args:
        is_hallucination: True if any provably-absent claim was found.
        flagged_claims: List of (claim_text, reason) pairs.
        penalty: Score penalty to apply (-2.0 per hallucination event, capped).
        coverage: Fraction of response sentences that are grounded in oracle.
    """
    is_hallucination: bool
    flagged_claims: list[tuple[str, str]] = field(default_factory=list)
    penalty: float = 0.0
    coverage: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_hallucination": self.is_hallucination,
            "flagged_claims": self.flagged_claims,
            "penalty": self.penalty,
            "coverage": self.coverage,
        }


# ---------------------------------------------------------------------------
# Hallucination detector
# ---------------------------------------------------------------------------

class HallucinationDetector:
    """Deterministic hallucination detector against an oracle memory set.

    Usage::

        detector = HallucinationDetector()
        result = detector.check(
            response="The patient's name is Alice and she was born in 1987.",
            oracle_memory={"patient_name": "Alice", "dob": "1985"},
        )
        # result.is_hallucination → True (1987 not in oracle)

    Args:
        numeric_tolerance: Allow numeric values within this ± range (e.g., 0 = exact match).
        min_claim_length: Ignore very short claims (likely pronouns/stop words).
    """

    def __init__(
        self,
        numeric_tolerance: float = 0.0,
        min_claim_length: int = 3,
    ) -> None:
        self.numeric_tolerance = numeric_tolerance
        self.min_claim_length = min_claim_length

    def check(
        self,
        response: str,
        oracle_memory: dict[str, Any],
    ) -> HallucinationResult:
        """Check a model response against oracle memory for hallucinated claims.

        Args:
            response: The full model response text.
            oracle_memory: Ground-truth memory dict (keys → content strings).
                           Values can be strings or dicts; all are flattened to text.

        Returns:
            HallucinationResult with penalty and flagged claims.
        """
        oracle_text = _flatten_oracle(oracle_memory)
        oracle_tokens = _normalize_tokens(oracle_text)
        oracle_numbers = _extract_numbers(oracle_text)

        sentences = _split_sentences(response)
        flagged: list[tuple[str, str]] = []
        grounded = 0

        for sentence in sentences:
            if len(sentence.split()) < self.min_claim_length:
                grounded += 1
                continue

            sentence_hallucinated = False

            # Check: numeric values in sentence not in oracle.
            nums = _extract_numbers(sentence)
            for n in nums:
                if not _number_in_oracle(n, oracle_numbers, self.numeric_tolerance):
                    flagged.append((sentence, f"numeric value '{n}' not in oracle memory"))
                    sentence_hallucinated = True
                    break

            # Check: named entities (capitalized multi-word phrases) not in oracle.
            if not sentence_hallucinated:
                entities = _extract_named_entities(sentence)
                for entity in entities:
                    if not _entity_in_oracle(entity, oracle_tokens):
                        flagged.append((sentence, f"entity '{entity}' not in oracle memory"))
                        sentence_hallucinated = True
                        break

            if not sentence_hallucinated:
                grounded += 1

        coverage = grounded / len(sentences) if sentences else 1.0
        is_hallucination = len(flagged) > 0
        # Plan: -2 per hallucination event (the whole response counts as one event).
        penalty = -2.0 if is_hallucination else 0.0

        return HallucinationResult(
            is_hallucination=is_hallucination,
            flagged_claims=flagged[:10],  # cap for readability
            penalty=penalty,
            coverage=coverage,
        )

    def check_op_action(
        self,
        content: str,
        oracle_memory: dict[str, Any],
    ) -> HallucinationResult:
        """Check a stored memory content string against oracle.

        Used in Base 1 (Store) to detect if the model invented facts it stored.
        Less strict than full response check — just numeric and entity checks.
        """
        return self.check(content, oracle_memory)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flatten_oracle(oracle: dict[str, Any]) -> str:
    """Flatten oracle memory dict to one text blob."""
    parts: list[str] = []
    for v in oracle.values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, dict):
            parts.append(_flatten_oracle(v))
        elif isinstance(v, (list, tuple)):
            parts.extend(str(item) for item in v)
        else:
            parts.append(str(v))
    return " ".join(parts)


def _normalize_tokens(text: str) -> set[str]:
    """Lowercase, split on hyphens/underscores, strip punctuation, split to token set."""
    text = text.lower()
    # Split compound tokens (e.g. "ACC-7821" → "acc 7821", "user_name" → "user name")
    # before removing other punctuation, so subcomponents are individually matchable.
    text = text.replace("-", " ").replace("_", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    return set(text.split())


_NUMBER_RE = re.compile(r"\b\d[\d,\.]*\b")


def _extract_numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text)


def _number_in_oracle(
    num_str: str,
    oracle_numbers: list[str],
    tolerance: float,
) -> bool:
    if num_str in oracle_numbers:
        return True
    if tolerance == 0.0:
        return False
    try:
        n = float(num_str.replace(",", ""))
        for oracle_num in oracle_numbers:
            try:
                o = float(oracle_num.replace(",", ""))
                if abs(n - o) <= tolerance:
                    return True
            except ValueError:
                pass
    except ValueError:
        pass
    return False


# Named entity: 2+ consecutive words starting with capital letters,
# OR a single word that's fully uppercase (abbreviation).
_NAMED_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z]{2,})\b")


def _extract_named_entities(text: str) -> list[str]:
    return _NAMED_ENTITY_RE.findall(text)


def _entity_in_oracle(entity: str, oracle_tokens: set[str]) -> bool:
    entity_tokens = set(entity.lower().split())
    # Entity is "grounded" if all its tokens appear in the oracle token set.
    return entity_tokens.issubset(oracle_tokens)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Simple rule-based, no NLTK dependency."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]
