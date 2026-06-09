"""Tier 3 outcome-based verifier for memory_horizon.

Implements:
    - Frozen QA model pattern (Memory-R1) for Base 1 and Base 7
    - Token F1 scoring for Base 2 (Synthesize)
    - Kendall's tau for Base 6 (Temporal)
    - QA accuracy delta for Base 4 (Compress)

The frozen QA model is imported lazily from training.frozen_qa_model so this
module can be imported without a GPU / transformers installed. If the frozen
model is unavailable, Tier 3 falls back to token F1 only.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class Tier3Result:
    """Result from Tier 3 outcome verification.

    Args:
        score: Scalar in [-1, +1].
        method: Which scoring method was used.
        details: Method-specific breakdown.
    """
    score: float
    method: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "method": self.method,
            "passed": self.passed,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Token F1 (Base 2 — Synthesize)
# ---------------------------------------------------------------------------

def token_f1_score(prediction: str, gold: str) -> float:
    """Compute token-level F1 between prediction and gold answer.

    Normalized: lowercase, strip punctuation, split on whitespace.
    Returns F1 in [0, 1].
    """
    pred_tokens = _normalize_answer_tokens(prediction)
    gold_tokens = _normalize_answer_tokens(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common = sum((pred_counter & gold_counter).values())

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_synthesize(prediction: str, gold: str) -> Tier3Result:
    """Tier 3 for Base 2 (Synthesize).

    Scoring (from plan):
        F1 ≥ 0.7 → +1
        0.4 ≤ F1 < 0.7 → partial = F1
        F1 < 0.4 → 0
    """
    f1 = token_f1_score(prediction, gold)
    if f1 >= 0.7:
        score = 1.0
    elif f1 >= 0.4:
        score = f1
    else:
        score = 0.0

    return Tier3Result(
        score=score,
        method="token_f1",
        passed=score > 0,
        details={"f1": f1, "prediction": prediction[:200], "gold": gold[:200]},
    )


# ---------------------------------------------------------------------------
# Kendall's tau (Base 6 — Temporal)
# ---------------------------------------------------------------------------

def kendalls_tau(predicted_order: list[int], gold_order: list[int]) -> float:
    """Compute Kendall's tau between two orderings.

    Args:
        predicted_order: Model's ordering (list of event indices in predicted order).
        gold_order: Ground-truth ordering.

    Returns:
        Tau in [-1, +1]. Clipped to 0 for negatives (plan: clip negatives).
    """
    if len(predicted_order) != len(gold_order):
        return 0.0
    n = len(predicted_order)
    if n < 2:
        return 1.0

    # Build rank lookup for predicted order.
    pred_rank = {item: rank for rank, item in enumerate(predicted_order)}

    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            gi, gj = gold_order[i], gold_order[j]
            if gi not in pred_rank or gj not in pred_rank:
                continue
            if (pred_rank[gi] < pred_rank[gj]) == (i < j):
                concordant += 1
            else:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return 1.0
    tau = (concordant - discordant) / total
    return max(0.0, tau)  # clip negatives per plan


def score_temporal(
    predicted_order: list[int],
    gold_order: list[int],
    qa_accuracy: float | None = None,
) -> Tier3Result:
    """Tier 3 for Base 6 (Temporal).

    Score = Kendall's tau (continuous, clipped to 0).
    If qa_accuracy is also provided, average with tau.
    Kill criterion: held-out eval < 30% → pause (checked by curriculum.py).
    """
    tau = kendalls_tau(predicted_order, gold_order)

    if qa_accuracy is not None:
        score = (tau + qa_accuracy) / 2
    else:
        score = tau

    return Tier3Result(
        score=score,
        method="kendalls_tau",
        passed=score >= 0.5,
        details={
            "tau": tau,
            "qa_accuracy": qa_accuracy,
            "n_events": len(gold_order),
        },
    )


# ---------------------------------------------------------------------------
# QA accuracy delta (Base 4 — Compress)
# ---------------------------------------------------------------------------

def score_compress(
    original_qa_scores: list[float],
    compressed_qa_scores: list[float],
    compression_ratio: float,
) -> Tier3Result:
    """Tier 3 for Base 4 (Compress).

    Score = max(0, 1 - delta/0.03) — continuous reward.
    Token penalty: reward -= 0.1 * compression_ratio.

    Args:
        original_qa_scores: QA accuracy on the original memory (5 probes).
        compressed_qa_scores: QA accuracy on the compressed memory (5 probes).
        compression_ratio: len(compressed) / len(original).
    """
    if not original_qa_scores or not compressed_qa_scores:
        return Tier3Result(score=0.0, method="qa_delta", passed=False,
                           details={"error": "empty qa score lists"})

    original_acc = sum(original_qa_scores) / len(original_qa_scores)
    compressed_acc = sum(compressed_qa_scores) / len(compressed_qa_scores)
    delta = max(0.0, original_acc - compressed_acc)

    # Continuous reward: plan says +1 if delta < 0.03, scaled down otherwise.
    base_score = max(0.0, 1.0 - delta / 0.03)
    # Token penalty.
    penalty = 0.1 * compression_ratio
    score = max(0.0, base_score - penalty)

    return Tier3Result(
        score=score,
        method="qa_delta",
        passed=delta < 0.03,
        details={
            "original_accuracy": original_acc,
            "compressed_accuracy": compressed_acc,
            "delta_pp": delta * 100,
            "compression_ratio": compression_ratio,
            "penalty": penalty,
        },
    )


# ---------------------------------------------------------------------------
# QA model scoring (Base 1, Base 7 — Memory-R1 pattern)
# ---------------------------------------------------------------------------

def score_with_qa_model(
    question: str,
    stored_memory_context: str,
    gold_answer: str,
    qa_fn: Callable[[str, str], str] | None = None,
) -> Tier3Result:
    """Score by asking a frozen QA model to answer using only stored memory.

    This is the Memory-R1 pattern: the frozen QA model is the verifier.
    It's deterministic relative to the training run (frozen weights).

    Args:
        question: The QA probe question.
        stored_memory_context: The model's stored memory (get_context() output).
        gold_answer: Ground-truth answer.
        qa_fn: Callable(question, context) → answer string.
               If None, falls back to token F1 between context and gold.

    Returns:
        +1 if correct, 0 if wrong.
    """
    if qa_fn is None:
        # Fallback: check if gold answer tokens appear in stored context.
        gold_tokens = set(_normalize_answer_tokens(gold_answer))
        ctx_tokens = set(_normalize_answer_tokens(stored_memory_context))
        recall = len(gold_tokens & ctx_tokens) / max(len(gold_tokens), 1)
        score = 1.0 if recall >= 0.6 else 0.0
        return Tier3Result(
            score=score,
            method="qa_fallback_token_recall",
            passed=score > 0,
            details={"recall": recall, "gold": gold_answer[:100]},
        )

    predicted = qa_fn(question, stored_memory_context)
    f1 = token_f1_score(predicted, gold_answer)
    score = 1.0 if f1 >= 0.5 else 0.0

    return Tier3Result(
        score=score,
        method="frozen_qa_model",
        passed=score > 0,
        details={"predicted": predicted[:200], "gold": gold_answer[:100], "f1": f1},
    )


# ---------------------------------------------------------------------------
# Integration gate composite score (Base 7)
# ---------------------------------------------------------------------------

BASE7_WEIGHTS: dict[str, float] = {
    "store":      0.15,
    "synthesize": 0.20,
    "contradict": 0.20,
    "compress":   0.10,
    "abstain":    0.20,
    "temporal":   0.15,
}

def score_integration(component_scores: dict[str, float]) -> Tier3Result:
    """Weighted composite score for Base 7 (Integration Gate).

    Args:
        component_scores: Dict of env_type → scalar score in [-2, +1].
                          Missing components get 0.

    Returns:
        Weighted average in [-2, +1].
    """
    total = 0.0
    weight_sum = 0.0
    breakdown: dict[str, float] = {}

    for env_type, weight in BASE7_WEIGHTS.items():
        s = component_scores.get(env_type, 0.0)
        total += s * weight
        weight_sum += weight
        breakdown[env_type] = s

    score = total / weight_sum if weight_sum > 0 else 0.0

    return Tier3Result(
        score=score,
        method="integration_composite",
        passed=score >= 0.7,  # within 10pp of oracle (+1.0) per plan
        details={"weights": BASE7_WEIGHTS, "component_scores": breakdown},
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_answer_tokens(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    # Remove common stop words that inflate F1.
    _STOP = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
             "have", "has", "had", "do", "does", "did", "will", "would",
             "could", "should", "may", "might", "shall", "can", "to", "of",
             "in", "on", "at", "by", "for", "with", "as", "from", "and",
             "or", "but", "not", "no", "it", "its", "this", "that"}
    return [t for t in tokens if t not in _STOP]
