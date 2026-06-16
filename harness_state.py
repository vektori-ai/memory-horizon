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

from ledger import LedgerEntry, _normalize, score_write_against_ledger, score_invalidate_against_ledger
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
DIVERSITY_BONUS     = float(os.environ.get("DIVERSITY_BONUS",     "0.10"))  # bonus for using ≥N distinct ops
DIVERSITY_TARGET    = int(  os.environ.get("DIVERSITY_TARGET",    "4"))     # distinct op types to earn bonus (4 forces real mixing)

_WRITE_OPS = {"STORE_FACT", "UPDATE", "SUPERSEDE", "COMPRESS"}
# Ops that count toward diversity (not RESOLVE since that's always injected)
_DIVERSITY_OPS = {"STORE_FACT", "UPDATE", "SUPERSEDE", "COMPRESS", "ABSTAIN", "RETRIEVE"}


def _score_resolve(answer: str, gold: str) -> float:
    """Route to exact match or token-F1 based on gold answer length."""
    gold_tokens = _normalize(gold)
    if len(gold_tokens) <= 3:
        return 1.0 if _normalize(answer) == gold_tokens else 0.0
    return _token_f1(answer, gold)


# ---------------------------------------------------------------------------
# MemoryHarnessState
# ---------------------------------------------------------------------------

@dataclass
class MemoryHarnessState:
    """Full episode state for one rollout window.

    Attributes:
        fs:              The VirtualFilesystem — agent's memory store (with importance tags).
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
    # Auto-seeding (adapted from Harness-1's warm-start idea, not the same
    # trigger — see docstring)
    # ------------------------------------------------------------------

    def seed_from_sessions(self, sessions: list[dict], seed_session_count: int = 1) -> None:
        """Pre-populate FS from the first N sessions before the episode starts.

        Avoids cold-start reward collapse: early GRPO rollouts start with warm
        prior state so reward variance exists from step 1. All seeded entries
        are tagged 'tentative' — the model must confirm or supersede them.

        Adapted from Harness-1's auto_populate_from_first_search, not a literal
        mirror: Harness-1 seeds only after the policy's own first successful
        search action (still earned by a real decision); this seeds
        unconditionally before the episode/window starts, from prior session
        content the policy never had to act to get. Different trigger by
        design — this task picks up mid-conversation where prior context
        plausibly already exists, unlike Harness-1's from-scratch search task.
        """
        for s_idx in range(min(seed_session_count, len(sessions))):
            self.fs.seed_from_session(sessions[s_idx], session_idx=s_idx)

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
            # Used to return here without recording anything in step_rewards at
            # all — meaning an invalid op had literally zero effect on the final
            # reward (not even diluting the average), despite FORMAT_PENALTY
            # existing. Now appended like every other step so compute_reward's
            # step_score (which now sums everything, not just r > 0) can
            # actually let this subtract.
            self.step_rewards.append(FORMAT_PENALTY)
            return FORMAT_PENALTY

        # SUPERSEDE archives whatever was at this path BEFORE the op is applied —
        # capture it now. score_invalidate_against_ledger wants the content that
        # just got archived (was it stale?), not the new content being written;
        # fs.apply_op below overwrites self.fs.files[path], so this is the last
        # point we can read the old value.
        path          = op.get("path", "").strip()
        prior_content = " ".join(self.fs.files.get(path, []))

        # Actually apply the op to the filesystem
        self.fs.apply_op(op, session_idx, turn_idx)

        # Step reward from ledger
        step_r = 0.0
        content = op.get("content", "").strip()

        if op_type in _WRITE_OPS and content:
            step_r = score_write_against_ledger(
                content, self.ledger, self.current_session
            )
            # Extra signal for correctly superseding stale content. Was passing
            # `content` (the NEW value just written) here — backwards; the
            # function scores whether the OLD value was stale, i.e. whether
            # archiving it was the right call.
            if op_type == "SUPERSEDE":
                step_r += score_invalidate_against_ledger(
                    prior_content, self.ledger, self.current_session
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
        """Score a RESOLVE answer. Routes by gold answer type.

        Short/categorical answers (≤3 content tokens after stop-word removal)
        use exact match. Token-F1 on a single-word answer or a letter option
        ("B", "Acupuncture") gives spurious partial credit when the model
        hallucinates a different word that shares no tokens at all — F1 = 0 in
        that case anyway — but partial overlap on 1-2 token answers can inflate
        score when the answer is wrong (e.g. gold="2019", pred="late 2019" → F1
        = 0.67 despite being wrong). Exact match is the right metric there.
        Free-form answers (>3 tokens) keep F1 because partial recall matters.
        """
        score = _score_resolve(answer, gold)
        self.probe_scores.append(score)
        self.op_counts["RESOLVE"] += 1
        return score

    # ------------------------------------------------------------------
    # Episode reward
    # ------------------------------------------------------------------

    def compute_reward(self) -> float:
        """Compute final episode reward.

        reward = RESOLVE_WEIGHT    * mean(probe_scores)
               + STEP_WEIGHT       * mean(step_rewards)        # all of them, not just r > 0
               + diversity_bonus   (smooth ramp toward DIVERSITY_TARGET distinct op types)
               - op_penalty        (linear ramp after OP_PENALTY_ONSET ops)
               + abstain_penalty   (negative, applied if probes exist but 0 RESOLVE)

        Adapted from harness-1's reward (arXiv 2606.02373):
          outcome_weight * curated_recall
          + trajectory_weight * pool_recall
          + tool_diversity_bonus
          - turn_penalty
        Not a literal mirror — see step_score and diversity_bonus comments below
        for where this deliberately differs from a straight copy.
        """
        resolve_score = (
            sum(self.probe_scores) / len(self.probe_scores)
            if self.probe_scores else 0.0
        )
        # Used to only sum step_rewards where r > 0 — meaning FORMAT_PENALTY
        # (-0.5, see apply_op) could never actually subtract from the episode
        # reward, only dilute the average via denominator inflation. Sum
        # everything now so invalid ops actually cost something, matching the
        # documented per-step penalty table instead of silently no-op'ing it.
        step_score = sum(self.step_rewards) / max(len(self.step_rewards), 1)

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

        # Tool-diversity bonus — incentivize mixing op types (Harness-1 Fig 5 lesson:
        # without this, RL collapses to a narrow search-only/single-op strategy).
        # Used to be a step function (0 below DIVERSITY_TARGET, flat DIVERSITY_BONUS
        # at/above it) — a hard cliff right at the threshold, which is exactly why
        # DIVERSITY_TARGET needed reactive retuning (3→4, see eval_local.py commit:
        # random policies earned the full bonus 90% of the time at 3). Harness-1's
        # actual formula (arXiv 2606.02373 reward section) is a smooth ramp,
        # w_div * min(ν/ν0, 1) — partial diversity earns proportional credit, no
        # single threshold to find and farm.
        distinct_ops    = sum(1 for op in _DIVERSITY_OPS if self.op_counts.get(op, 0) > 0)
        diversity_bonus = DIVERSITY_BONUS * min(distinct_ops / DIVERSITY_TARGET, 1.0)

        reward = (
            RESOLVE_WEIGHT       * resolve_score
            + STEP_REWARD_WEIGHT * step_score
            + diversity_bonus
            - op_penalty
            + abstain_penalty
        )
        return float(reward)

    # ------------------------------------------------------------------
    # Logging helpers (mirrors ultra_core.build_result_summary)
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        distinct_ops = sum(1 for op in _DIVERSITY_OPS if self.op_counts.get(op, 0) > 0)
        return {
            "probe_scores":    self.probe_scores,
            "mean_resolve":    sum(self.probe_scores) / max(len(self.probe_scores), 1),
            "step_rewards":    self.step_rewards,
            "mean_step":       sum(self.step_rewards) / max(len(self.step_rewards), 1),
            "op_counts":       dict(self.op_counts),
            "n_retrievals":    len(self.retrieval_log),
            "fs_paths":        list(self.fs.files.keys()),
            "n_confirmed":     sum(1 for t in self.fs.importance.values() if t == "confirmed"),
            "n_tentative":     sum(1 for t in self.fs.importance.values() if t == "tentative"),
            "distinct_ops":    distinct_ops,
            "diversity_bonus": DIVERSITY_BONUS * min(distinct_ops / DIVERSITY_TARGET, 1.0),
            "final_reward":    self.compute_reward(),
        }

    def render_context(self, max_lines: int = 30) -> str:
        """Render FS state for prompt injection — mirrors render_context_within_budget."""
        fs_text  = self.fs.render_for_prompt()
        fs_lines = fs_text.split("\n")
        if len(fs_lines) > max_lines:
            return "\n".join(fs_lines[-max_lines:]) + "\n[...truncated]"
        return fs_text
