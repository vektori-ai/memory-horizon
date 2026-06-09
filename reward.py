"""Vektori reward function — loaded by verl at training time.

verl config:
    custom_reward_function.path=/root/reward.py
    custom_reward_function.name=compute_reward
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter

_VALID_OPS = {"STORE_FACT", "UPDATE", "SUPERSEDE", "COMPRESS", "ABSTAIN"}
_CONTENT_OPTIONAL_OPS = {"ABSTAIN"}

_STOP = {"what", "is", "the", "are", "was", "were", "a", "an", "this", "that",
         "of", "for", "in", "on", "at", "to", "does", "did", "has", "have", "had"}


def compute_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> float:
    """R = 0.6 * token_F1 + 0.4 * ROUGE1_recall - volume_penalty.

    Format gate: invalid/unparseable op → -0.5 (penalised, not neutral).
    ABSTAIN with no content → 0.0 (model chose not to store; no signal).
    """
    if not _is_valid_memory_op(solution_str):
        return -0.5

    content = _extract_content_field(solution_str)
    if not content:
        return 0.0

    r_task   = _token_f1(content, ground_truth)
    r_memory = _rouge1_recall(content, ground_truth)
    p_volume = min(0.002 * len(content.split()), 0.3)

    return max(-1.0, min(1.0, 0.6 * r_task + 0.4 * r_memory - p_volume))


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

    if op not in _CONTENT_OPTIONAL_OPS and not d.get("content"):
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


def _token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction)
    gold_tokens = _normalize(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common    = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    precision = common / len(pred_tokens)
    recall    = common / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t not in _STOP]
