"""MemoryHarnessState — harness-1's WorkingMemory analog for memory agents.

harness-1 tracks: candidate_pool, curated_set, retrieval_history, budget.
We track:         filesystem (FS), retrieval_log, op_counts, probe_scores, step_rewards.

Single source of truth for episode state and reward computation.
Imported by agent_loop.py. Mirrors ultra_core.py's role in harness-1.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ledger import LedgerEntry, score_write_against_ledger, score_invalidate_against_ledger
from memory_fs import VirtualFilesystem, _token_f1

# ---------------------------------------------------------------------------
# Reward weights — mirror harness-1's OUTCOME_WEIGHT / TRAJECTORY_RECALL_WEIGHT
# ---------------------------------------------------------------------------

RESOLVE_WEIGHT      = float(os.environ.get("RESOLVE_WEIGHT",      "0.80"))  # terminal probe score
STEP_REWARD_WEIGHT  = float(os.environ.get("STEP_REWARD_WEIGHT",  "0.20"))  # ledger step signal
OP_PENALTY_MAX      = float(os.environ.get("OP_PENALTY_MAX",      "0.15"))  # max op-count penalty
OP_PENALTY_ONSET    = int(  os.environ.get("OP_PENALTY_ONSET",    "20"))    # ops before penalty starts
FORMAT_PENALTY      = float(os.environ.get("FORMAT_PENALTY",      "-0.5"))  # per invalid op
ABSTAIN_PENALTY     = float(os.environ.get("ABSTAIN_PENALTY",     "-0.3"))  # over-abstention on probe

_WRITE_OPS = {"STORE_FACT", "UPDATE", "SUPERSEDE", "COMPRESS"}


# ---------------------------------------------------------------------------
# MemoryHarnessState
# ---------------------------------------------------------------------------

@dataclass
class MemoryHarnessState:
    """Full episode state for one rollout window.

    Attributes:
        fs:              The VirtualFilesystem — agent's memory store.
        ledger:          Ground-truth fact entries for step-level reward.
        retrieval_log:   Each RETRIEVE call: {turn, query, results}.
        op_counts:       Counter of op types issued (including INVALID).
        step_rewards:    Ledger-aligned reward per write op.
        probe_scores:    Token-F1 per RESOLVE call.
        current_session: Session id being processed (for ledger validity check).
    """
    fs:              VirtualFilesystem          = field(default_factory=VirtualFilesystem)
    ledger:          list[LedgerEntry]          = field(default_factory=list)
    retrieval_log:   list[dict[str, Any]]       = field(default_factory=list)
    op_counts:       Counter                    = field(default_factory=Counter)
    step_rewards:    list[float]                = field(default_factory=list)
    probe_scores:    list[float]                = field(default_factory=list)
    current_session: str | None                 = None

    # ------------------------------------------------------------------
    # Apply an op and score it against the ledger
    # ------------------------------------------------------------------

    def apply_op(self, op: dict, session_idx: int, turn_idx: int) -> float:
        """Apply op to FS, compute step reward from ledger. Returns step reward.

        Called for every model output in the conversation turns phase.
        RETRIEVE and RESOLVE are handled separately (see apply_retrieve / apply_resolve).
        """
        op_type = op.get("op", "INVALID") if isinstance(op, dict) else "INVALID"
        self.op_counts[op_type] += 1

        if op_type == "INVALID" or not isinstance(op, dict):
            return FORMAT_PENALTY

        # Actually apply the op to the filesystem
        self.fs.apply_op(op, session_idx, turn_idx)

        # Step reward from ledger
        step_r = 0.0
        content = op.get("content", "").strip()

        if op_type in _WRITE_OPS and content:
            step_r = score_write_against_ledger(
                content, self.ledger, self.current_session
            )
            # Extra signal for correctly superseding stale content
            if op_type == "SUPERSEDE":
                step_r += score_invalidate_against_ledger(
                    content, self.ledger, self.current_session
                )

        self.step_rewards.append(step_r)
        return step_r

    # ------------------------------------------------------------------
    # RETRIEVE — agent calls this, harness executes Context-1
    # ------------------------------------------------------------------

    def apply_retrieve(self, query: str, results: list[str], turn_idx: int) -> None:
        """Log a RETRIEVE call. No reward signal — retrieval quality is implicit
        in whether subsequent RESOLVE calls succeed."""
        self.op_counts["RETRIEVE"] += 1
        self.retrieval_log.append({
            "turn":    turn_idx,
            "query":   query,
            "results": results,
        })

    # ------------------------------------------------------------------
    # RESOLVE — agent answers a probe
    # ------------------------------------------------------------------

    def apply_resolve(self, answer: str, gold: str) -> float:
        """Score a RESOLVE answer against the gold answer. Returns token-F1."""
        score = _token_f1(answer, gold)
        self.probe_scores.append(score)
        self.op_counts["RESOLVE"] += 1
        return score

    # ------------------------------------------------------------------
    # Episode reward
    # ------------------------------------------------------------------

    def compute_reward(self) -> float:
        """Compute final episode reward.

        reward = RESOLVE_WEIGHT  * mean(probe_scores)
               + STEP_WEIGHT     * mean(step_rewards)
               - op_penalty
               + format_penalty  (already baked into step_rewards as FORMAT_PENALTY)

        Mirrors harness-1's compute_reward in ultra_core.py:
          outcome_weight * curated_recall + trajectory_weight * pool_recall - turn_penalty
        """
        resolve_score = (
            sum(self.probe_scores) / len(self.probe_scores)
            if self.probe_scores else 0.0
        )
        step_score = (
            sum(r for r in self.step_rewards if r > 0) / max(len(self.step_rewards), 1)
        )

        # Op-count penalty — linear ramp (mirrors harness-1's turn penalty)
        total_ops = sum(self.op_counts[k] for k in self.op_counts if k != "RESOLVE")
        if total_ops > OP_PENALTY_ONSET:
            op_penalty = min(OP_PENALTY_MAX, 0.01 * (total_ops - OP_PENALTY_ONSET))
        else:
            op_penalty = 0.0

        # Over-abstention penalty: if there were probes but model never resolved any
        if self.op_counts.get("RESOLVE", 0) == 0 and self.op_counts.get("ABSTAIN", 0) > 0:
            abstain_penalty = ABSTAIN_PENALTY
        else:
            abstain_penalty = 0.0

        reward = (
            RESOLVE_WEIGHT     * resolve_score
            + STEP_REWARD_WEIGHT * step_score
            - op_penalty
            + abstain_penalty
        )
        return float(reward)

    # ------------------------------------------------------------------
    # Logging helpers (mirrors ultra_core.build_result_summary)
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "probe_scores":    self.probe_scores,
            "mean_resolve":    sum(self.probe_scores) / max(len(self.probe_scores), 1),
            "step_rewards":    self.step_rewards,
            "mean_step":       sum(self.step_rewards) / max(len(self.step_rewards), 1),
            "op_counts":       dict(self.op_counts),
            "n_retrievals":    len(self.retrieval_log),
            "fs_paths":        list(self.fs.files.keys()),
            "final_reward":    self.compute_reward(),
        }

    def render_context(self, max_lines: int = 30) -> str:
        """Render FS state for prompt injection — mirrors render_context_within_budget."""
        fs_text  = self.fs.render_for_prompt()
        fs_lines = fs_text.split("\n")
        if len(fs_lines) > max_lines:
            return "\n".join(fs_lines[-max_lines:]) + "\n[...truncated]"
        return fs_text
