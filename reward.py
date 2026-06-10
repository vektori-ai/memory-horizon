"""Vektori reward function — loaded by verl at training time.

verl config:
    custom_reward_function.path=/root/reward.py
    custom_reward_function.name=compute_reward
"""

from __future__ import annotations

import json
import re
from collections import Counter

from memory_fs import _token_f1, _normalize

_VALID_OPS            = {"STORE_FACT", "UPDATE", "SUPERSEDE", "COMPRESS", "ABSTAIN"}
_CONTENT_OPTIONAL_OPS = {"ABSTAIN"}


def compute_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> float:
    """Per-step reward for Architecture A (one row per turn).

    Format gate: invalid/unparseable op → -0.5.
    Valid op with no content (ABSTAIN) → 0.0.
    Valid op with content → 0.6 * token_F1 + 0.4 * ROUGE1_recall - volume_penalty + 0.1 format bonus.

    The +0.1 format bonus on valid ops with content prevents the model from
    collapsing to always ABSTAIN for near-zero reward.
    """
    if not _is_valid_memory_op(solution_str):
        return -0.5

    content = _extract_content_field(solution_str)
    if not content:
        return 0.0

    # ground_truth is json.dumps(qa_probes) — extract answer strings only so
    # _token_f1 compares stored content against actual answers, not JSON metadata
    answer_ref = _extract_answers(ground_truth)

    r_task   = _token_f1(content, answer_ref)
    r_memory = _rouge1_recall(content, answer_ref)
    p_volume = min(0.002 * len(content.split()), 0.3)

    # +0.1 bonus for any valid op that stores content (anti-collapse pressure)
    return max(-1.0, min(1.0, 0.6 * r_task + 0.4 * r_memory - p_volume + 0.1))



def _extract_answers(ground_truth: str) -> str:
    """Pull answer strings out of json.dumps(qa_probes) for clean token-F1."""
    try:
        probes = json.loads(ground_truth)
        return " ".join(str(p.get("answer", "")) for p in probes if p.get("answer"))
    except (json.JSONDecodeError, TypeError, AttributeError):
        return ground_truth


def _is_valid_memory_op(solution_str: str) -> bool:
    """Returns False (→ -0.5 penalty) if output is not a valid memory op JSON."""
    cleaned = re.sub(r"<think>.*?</think>", "", solution_str, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return False
    try:
        d = json.loads(m.group())
    except json.JSONDecodeError:
        return False

    op = d.get("op", "")
    if op not in _VALID_OPS:
        return False

    if op not in _CONTENT_OPTIONAL_OPS:
        if not d.get("content"):
            return False
        # Non-ABSTAIN ops must have a path (category/entity format)
        path = d.get("path", "")
        if not path or "/" not in path:
            return False

    return True


def _extract_content_field(completion: str) -> str:
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                completion = msg.get("content", "")
                break
        else:
            completion = str(completion)

    completion = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL).strip()
    try:
        obj = json.loads(completion.strip())
        return str(obj.get("content", ""))
    except (json.JSONDecodeError, AttributeError):
        pass
    m = re.search(r'"content"\s*:\s*"([^"]+)"', completion)
    if m:
        return m.group(1)
    return ""


def _rouge1_recall(hypothesis: str, reference: str) -> float:
    hyp = _normalize(hypothesis)
    ref = _normalize(reference)
    if not ref:
        return 1.0
    if not hyp:
        return 0.0
    hyp_c = Counter(hyp)
    ref_c = Counter(ref)
    return sum((hyp_c & ref_c).values()) / len(ref)


