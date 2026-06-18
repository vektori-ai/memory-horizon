"""Reward function — loaded by verl at training time.

verl config:
    custom_reward_function.path=/root/reward.py
    custom_reward_function.name=compute_reward

Scoring: mean token-F1 of RESOLVE answers against QA probe gold answers,
plus a small format penalty for invalid ops.

Token-F1 is used instead of exact match because:
  - Short gold answers (e.g. "2019") get non-zero reward for close answers
    ("late 2019" → F1=0.67 vs EM=0.0) — preserves within-group variance for GRPO
  - Continuous signal across the full range [0, 1] per probe

Each probe is located in the transcript by its unique [Question] anchor.
This avoids reconstructing the full VirtualFilesystem state at every step,
which was the main cost in the previous cursor-based replay approach.
"""

from __future__ import annotations

import json
import re

from memory_fs import _token_f1, parse_op


# ---------------------------------------------------------------------------
# Entry point verl calls
# ---------------------------------------------------------------------------

def compute_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> float:
    try:
        reward = _score_transcript(solution_str, ground_truth, extra_info or {})
    except Exception as exc:
        # Neutral score — a large negative outlier inflates GRPO advantage
        # magnitudes and destabilizes the whole group.
        _log_step(reward=None, error=str(exc))
        return 0.0

    _log_step(reward=reward, error=None)
    return reward


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def _score_transcript(solution_str: str, ground_truth: str, extra_info: dict) -> float:
    """Score one decoded rollout against its QA probes.

    Each probe is found via its unique [Question] anchor in the transcript.
    The model's RESOLVE answer immediately follows the question block.
    No FS simulation — O(n_probes × len(transcript)) instead of O(n_turns × n_ledger).
    """
    try:
        qa_probes = json.loads(ground_truth) if isinstance(ground_truth, str) else (ground_truth or [])
    except (json.JSONDecodeError, TypeError):
        qa_probes = []

    # Same probe-window filtering as agent_loop.py
    window_sids = set(extra_info.get("window_session_ids", []))
    if window_sids:
        filtered = [
            p for p in qa_probes
            if not p.get("requires_sessions")
            or any(r in window_sids for r in p["requires_sessions"])
        ]
        if filtered:
            qa_probes = filtered

    probe_scores = []
    for probe in qa_probes:
        gold = str(probe.get("answer", "")).strip()
        if not gold or gold.lower() in ("n/a", "yes", "no", "abstain", ""):
            continue

        question = probe.get("question", "")
        answer   = _extract_resolve_answer(solution_str, question)
        if answer is None:
            continue  # probe not reached in this rollout — don't penalise, just skip

        probe_scores.append(_token_f1(answer, gold))

    if not probe_scores:
        return 0.0

    # Format penalty: -0.02 per INVALID op, capped at -0.10
    n_invalid   = solution_str.count('"op": "INVALID"')
    fmt_penalty = max(-0.10, -0.02 * n_invalid)

    reward = sum(probe_scores) / len(probe_scores) + fmt_penalty
    return float(max(-1.0, min(1.0, reward)))


def _extract_resolve_answer(transcript: str, question: str) -> str | None:
    """Find the model's RESOLVE answer for a given question.

    Anchor: `[Question]\\n{question}` — unique per probe since each question
    text is distinct. The model's response follows immediately after.
    """
    anchor = f"[Question]\n{question}"
    pos    = transcript.find(anchor)
    if pos == -1:
        return None

    resp_start = pos + len(anchor)
    # Response ends at the next [Memory state] block (next probe or next turn)
    next_block = transcript.find("\n[Memory state]", resp_start)
    resp_end   = next_block if next_block != -1 else len(transcript)

    raw = transcript[resp_start:resp_end].strip()
    if not raw:
        return None

    op = parse_op(raw)
    if op and op.get("op") == "RESOLVE":
        return op.get("content", raw)
    # Fallback: treat the raw model output as the answer
    return raw


# ---------------------------------------------------------------------------
# WandB step logging
# ---------------------------------------------------------------------------

def _log_step(reward: float | None, error: str | None) -> None:
    try:
        import wandb
        if not wandb.run:
            return
        if error is not None:
            wandb.log({"step/replay_error": 1.0})
        else:
            wandb.log({"step/reward": reward, "step/replay_error": 0.0})
    except Exception:
        pass
