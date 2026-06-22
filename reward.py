"""Reward function — loaded by verl at training time.

verl config:
    custom_reward_function.path=/root/reward.py
    custom_reward_function.name=compute_reward

Reward: harness.compute_reward() pre-computed in agent_loop.py and embedded as
a short ASCII prefix at the front of solution_str:

    "R:{bucket:03d};" where bucket = int(reward * 127 + 128)

This avoids fragile text extraction (JSON parsing, question anchors, token
round-trips) that caused rewards to stay at 0. The harness uses token-F1
against gold answers with a raw-text fallback for non-JSON model outputs.
"""

from __future__ import annotations

import re

_PREFIX_RE = re.compile(r'^R:(\d{3});')


def compute_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> float:
    try:
        m = _PREFIX_RE.match(solution_str.strip())
        if m:
            bucket = int(m.group(1))
            return float((bucket - 128) / 127.0)
        return 0.0
    except Exception:
        return 0.0
