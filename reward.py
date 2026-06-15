"""Vektori reward function — loaded by verl at training time.

verl config:
    custom_reward_function.path=/root/reward.py
    custom_reward_function.name=compute_reward

Role: per-step FORMAT gate only.
  invalid JSON / missing op   → -0.5
  valid ABSTAIN               →  0.0
  valid op with content       → +0.1

Quality signal (did the stored content answer the QA probes?) comes entirely
from MemoryAgentLoop.run() → score_trajectory → AgentLoopOutput.reward.
Keeping these two signals separate avoids double-counting and makes each
one interpretable on its own in WandB.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter

from memory_fs import _token_f1, _normalize

_VALID_OPS            = {"STORE_FACT", "UPDATE", "SUPERSEDE", "COMPRESS", "ABSTAIN", "RESOLVE", "RETRIEVE"}
_CONTENT_OPTIONAL_OPS = {"ABSTAIN", "RETRIEVE"}  # RETRIEVE uses "query" not "content"

_OP_CODE = {"STORE_FACT": 1, "UPDATE": 2, "SUPERSEDE": 3, "COMPRESS": 4, "ABSTAIN": 5, "RESOLVE": 6, "INVALID": 0}


def compute_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> float:
    """Per-step format gate.

    Does NOT compute quality reward — that is handled by MemoryAgentLoop's
    terminal score_trajectory call so the signal is trajectory-level, not
    single-op level (which is too noisy for useful gradient).
    """
    valid   = _is_valid_memory_op(solution_str)
    content = _extract_content_field(solution_str) if valid else ""
    op_type = _extract_op_type(solution_str)

    if not valid:
        reward = -0.5
    elif not content:   # ABSTAIN
        reward = 0.0
    else:
        reward = 0.1    # format bonus

    _log_step(op_type, valid, reward)
    return reward


# ---------------------------------------------------------------------------
# WandB step logging
# ---------------------------------------------------------------------------

def _log_step(op_type: str, valid: bool, reward: float) -> None:
    try:
        import wandb
        if not wandb.run:
            return
        wandb.log({
            "step/format_valid": float(valid),
            "step/op_code":      _OP_CODE.get(op_type, 0),
            "step/reward":       reward,
        })
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_op_type(solution_str: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", solution_str, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return "INVALID"
    try:
        d = json.loads(m.group())
        return d.get("op", "INVALID")
    except json.JSONDecodeError:
        return "INVALID"


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
        # RESOLVE and RETRIEVE don't use a path — all write ops do
        if op not in ("RESOLVE", "RETRIEVE"):
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
