"""MemoryHarnessState — harness-1's WorkingMemory analog for memory agents.

Single source of truth for episode state and reward computation.
Imported by agent_loop.py and replayed by reward.py.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from memory_fs import VirtualFilesystem, _token_f1

# ---------------------------------------------------------------------------
# Reward parameters
# ---------------------------------------------------------------------------

# Probe quality (token-F1 of RESOLVE answers) is the entire training signal.
# Everything else is noise — step-level ledger rewards and diversity bonuses
# had 0 gradient when all rollouts in a group scored them the same, and caused
# the reward range to exceed [-1, 1] before clipping.
FORMAT_PENALTY_PER_OP = float(os.environ.get("FORMAT_PENALTY_PER_OP", "-0.02"))
FORMAT_PENALTY_MAX    = float(os.environ.get("FORMAT_PENALTY_MAX",    "-0.10"))


def _score_resolve(answer: str, gold: str) -> float:
    """Token-F1 for all answer lengths.

    Binary exact match for short answers kills within-group variance in GRPO
    (all-correct or all-wrong groups → zero gradient). Token-F1 gives continuous
    signal even for single-word answers (e.g. "2019" vs "late 2019" → F1=0.67).
    """
    return _token_f1(answer, gold)


# ---------------------------------------------------------------------------
# MemoryHarnessState
# ---------------------------------------------------------------------------

@dataclass
class MemoryHarnessState:
    """Full episode state for one rollout window.

    Attributes:
        fs:              The VirtualFilesystem — agent's memory store.
        ledger:          Ground-truth fact entries (kept for compat; not used in reward).
        retrieval_log:   Each RETRIEVE call: {turn, query, results}.
        op_counts:       Counter of op types issued (including INVALID).
        step_rewards:    Per-op rewards (kept for compat; not used in final reward).
        probe_scores:    Token-F1 per RESOLVE call — the actual training signal.
        current_session: Session id being processed.
    """
    fs:              VirtualFilesystem    = field(default_factory=VirtualFilesystem)
    ledger:          list                 = field(default_factory=list)
    retrieval_log:   list[dict[str, Any]] = field(default_factory=list)
    op_counts:       Counter              = field(default_factory=Counter)
    step_rewards:    list[float]          = field(default_factory=list)
    probe_scores:    list[float]          = field(default_factory=list)
    probe_traces:    list[dict[str, Any]] = field(default_factory=list)  # {q, model_ans, gold, f1}
    op_sequence:     list[str]            = field(default_factory=list)   # op type per conv turn
    current_session: str | None           = None

    # ------------------------------------------------------------------
    # Auto-seeding
    # ------------------------------------------------------------------

    def seed_from_sessions(self, sessions: list[dict], seed_session_count: int = 1) -> None:
        """Pre-populate FS from the first N sessions before the episode starts."""
        for s_idx in range(min(seed_session_count, len(sessions))):
            self.fs.seed_from_session(sessions[s_idx], session_idx=s_idx)

    # ------------------------------------------------------------------
    # Apply an op
    # ------------------------------------------------------------------

    def apply_op(self, op: dict, session_idx: int, turn_idx: int) -> float:
        """Apply op to FS, track op counts. Returns small step signal."""
        op_type = op.get("op", "INVALID") if isinstance(op, dict) else "INVALID"
        self.op_counts[op_type] += 1
        self.op_sequence.append(op_type)

        if op_type == "INVALID" or not isinstance(op, dict):
            r = FORMAT_PENALTY_PER_OP
            self.step_rewards.append(r)
            return r

        self.fs.apply_op(op, session_idx, turn_idx)
        self.step_rewards.append(0.0)
        return 0.0

    # ------------------------------------------------------------------
    # RETRIEVE
    # ------------------------------------------------------------------

    def apply_retrieve(self, query: str, results: list[str], turn_idx: int) -> None:
        """Log a RETRIEVE call. No reward — retrieval quality is implicit
        in whether subsequent RESOLVE calls succeed."""
        self.op_counts["RETRIEVE"] += 1
        self.retrieval_log.append({
            "turn":    turn_idx,
            "query":   query,
            "results": results,
        })

    # ------------------------------------------------------------------
    # RESOLVE
    # ------------------------------------------------------------------

    def apply_resolve(self, answer: str, gold: str, question: str = "") -> float:
        """Score a RESOLVE answer with token-F1."""
        score = _score_resolve(answer, gold)
        self.probe_scores.append(score)
        self.probe_traces.append({
            "question":     question[:300],
            "model_answer": answer[:300],
            "gold":         gold[:300],
            "f1":           round(score, 4),
        })
        self.op_counts["RESOLVE"] += 1
        return score

    # ------------------------------------------------------------------
    # Episode reward
    # ------------------------------------------------------------------

    def compute_reward(self) -> float:
        """Final episode reward.

        reward = mean(probe_scores)   # token-F1 on QA probes, clamped to [0, 1]

        Format penalty removed: negative rewards caused large gradient steps that
        collapsed entropy (0.075 → 0.013) by step 3 in run_015. Rewards stay in
        [0, 1] — format compliance is learned implicitly since garbage output gives
        token-F1 ≈ 0 anyway.

        Returns 0.0 when no probes were answered (truncated/empty transcript).
        """
        if not self.probe_scores:
            return 0.0

        probe_score = sum(self.probe_scores) / len(self.probe_scores)
        return float(max(0.0, min(1.0, probe_score)))

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "probe_scores":  self.probe_scores,
            "probe_traces":  self.probe_traces,
            "op_sequence":   self.op_sequence,
            "mean_resolve":  sum(self.probe_scores) / max(len(self.probe_scores), 1),
            "op_counts":     dict(self.op_counts),
            "n_retrievals":  len(self.retrieval_log),
            "fs_paths":      list(self.fs.files.keys()),
            "fs_snapshot":   self.fs.render_for_prompt()[:800],
            "n_confirmed":   sum(1 for t in self.fs.importance.values() if t == "confirmed"),
            "n_tentative":   sum(1 for t in self.fs.importance.values() if t == "tentative"),
            "final_reward":  self.compute_reward(),
        }

    def render_context(self, max_lines: int = 30) -> str:
        """Render FS state for prompt injection."""
        fs_text  = self.fs.render_for_prompt()
        fs_lines = fs_text.split("\n")
        if len(fs_lines) > max_lines:
            return "\n".join(fs_lines[-max_lines:]) + "\n[...truncated]"
        return fs_text
