"""Vektori reward function for GRPO memory training.

Design:
    R = 0.6 * R_task + 0.4 * R_memory - P_volume - P_delete

    R_task   — token F1 between predicted answer and gold answer
               Dense signal: gives gradient even for partially correct answers.

    R_memory — max ROUGE-1 recall over all memory entries vs gold answer
               Tells the model which facts it stored were actually useful.
               Uses max() so one good entry is enough — no reward for duplicates.

    P_volume — penalises total tokens stored (teaches concision)
    P_delete — penalises deleting a key that previously helped answer a question

    Format gate: if JSON is invalid or op is unknown → return 0.0 immediately.
    This avoids wasting gradient on unparseable outputs.

Step-wise GRPO note:
    This reward is computed once per episode (at the QA phase).
    The training loop broadcasts the terminal advantage to ALL prior steps —
    every memory op in the episode gets the same gradient signal.
    See training/stepwise_grpo.py for the advantage broadcast patch.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_episode_reward(
    predicted_answer: str,
    gold_answer: str,
    memory_entries: list[dict[str, str]],
    total_tokens_stored: int = 0,
    deleted_useful_key: bool = False,
    format_valid: bool = True,
) -> tuple[float, dict[str, Any]]:
    """Compute the episode reward and a breakdown dict for logging.

    Args:
        predicted_answer:   The answer the QA model (or memory model) produced.
        gold_answer:        Ground truth answer string.
        memory_entries:     List of dicts with at least a "content" key —
                            all entries currently in the memory bank.
        total_tokens_stored: Cumulative tokens stored this episode.
        deleted_useful_key:  True if the model deleted a key that was previously
                             used to answer a question correctly.
        format_valid:        False if any action in the episode was unparseable.

    Returns:
        (reward: float, breakdown: dict)
    """
    if not format_valid:
        return 0.0, {"r_task": 0.0, "r_memory": 0.0, "p_volume": 0.0, "p_delete": 0.0}

    r_task   = token_f1(predicted_answer, gold_answer)
    r_memory = _max_rouge1_recall(memory_entries, gold_answer)
    p_volume = min(0.002 * total_tokens_stored, 0.3)  # capped so it never dominates
    p_delete = 1.0 if deleted_useful_key else 0.0

    reward = 0.6 * r_task + 0.4 * r_memory - p_volume - p_delete
    reward = max(-1.0, min(1.0, reward))

    breakdown = {
        "r_task":   round(r_task, 4),
        "r_memory": round(r_memory, 4),
        "p_volume": round(p_volume, 4),
        "p_delete": p_delete,
        "reward":   round(reward, 4),
    }
    return reward, breakdown


# ---------------------------------------------------------------------------
# verl-compatible wrapper
# Called by verl as: compute_reward(data_source, solution_str, ground_truth, extra_info)
# ---------------------------------------------------------------------------

def compute_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
) -> float:
    """verl reward function signature.

    extra_info keys expected:
        predicted_answer    str   — answer produced during QA phase
        memory_entries      list  — [{"key": ..., "content": ...}, ...]
        total_tokens_stored int
        deleted_useful_key  bool
        format_valid        bool
    """
    reward, _ = compute_episode_reward(
        predicted_answer=extra_info.get("predicted_answer", solution_str),
        gold_answer=ground_truth,
        memory_entries=extra_info.get("memory_entries", []),
        total_tokens_stored=extra_info.get("total_tokens_stored", 0),
        deleted_useful_key=extra_info.get("deleted_useful_key", False),
        format_valid=extra_info.get("format_valid", True),
    )
    return reward


# ---------------------------------------------------------------------------
# Token F1
# ---------------------------------------------------------------------------

def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1. Normalised: lowercase, strip punctuation, split on whitespace."""
    pred_tokens = _normalize_tokens(prediction)
    gold_tokens = _normalize_tokens(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_c = Counter(pred_tokens)
    gold_c = Counter(gold_tokens)
    common = sum((pred_c & gold_c).values())

    precision = common / len(pred_tokens)
    recall    = common / len(gold_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# ROUGE-1 Recall
# ---------------------------------------------------------------------------

def rouge1_recall(hypothesis: str, reference: str) -> float:
    """ROUGE-1 recall: fraction of reference tokens found in hypothesis."""
    hyp_tokens = _normalize_tokens(hypothesis)
    ref_tokens = _normalize_tokens(reference)

    if not ref_tokens:
        return 1.0
    if not hyp_tokens:
        return 0.0

    hyp_c = Counter(hyp_tokens)
    ref_c = Counter(ref_tokens)
    common = sum((hyp_c & ref_c).values())
    return common / len(ref_tokens)


def _max_rouge1_recall(memory_entries: list[dict], gold_answer: str) -> float:
    """Max ROUGE-1 recall over all memory entries against the gold answer.

    Uses max() because one well-stored entry is enough.
    No bonus for storing the same fact multiple times.
    """
    if not memory_entries:
        return 0.0
    return max(
        rouge1_recall(entry.get("content", ""), gold_answer)
        for entry in memory_entries
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_tokens(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()
