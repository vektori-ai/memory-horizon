"""Core types for memory_horizon.

All training data, actions, and verifier results are expressed in terms of these types.
Design principle: everything JSON-serializable so trajectory datasets can be stored as JSONL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Memory operation taxonomy
# ---------------------------------------------------------------------------

class MemoryOp(str, Enum):
    """All operations a memory manager can perform on the memory store.

    Grouped into three families:
    - STORE family  (Base 1): new information enters memory
    - CONFLICT family (Base 3): resolve contradiction between old and new
    - UTILITY ops  (Base 4/5): housekeeping and refusal
    """
    # Store family
    STORE_FACT      = "STORE_FACT"       # atomic fact, retrievable by key
    CREATE_EPISODE  = "CREATE_EPISODE"   # episodic memory block (narrative unit)
    INFER_IMPLICIT  = "INFER_IMPLICIT"   # implicit inference not literally stated

    # Conflict-resolution family
    UPDATE          = "UPDATE"           # overwrite current value (value changed)
    SUPERSEDE       = "SUPERSEDE"        # old value archived in audit log (hard replace)
    DECAY           = "DECAY"            # reduce confidence without deleting (uncertain)
    KEEP_BOTH       = "KEEP_BOTH"        # store both versions with distinct keys+timestamps

    # Utility
    COMPRESS        = "COMPRESS"         # lossy summarization of existing memory block
    ABSTAIN         = "ABSTAIN"          # explicit refusal: memory lacks the answer

    # Harness-facing ops (agent calls these, harness executes them)
    RETRIEVE        = "RETRIEVE"         # search own memory via Context-1; harness returns slice
    RESOLVE         = "RESOLVE"          # emit answer for a probe question


STORE_OPS: frozenset[MemoryOp] = frozenset({
    MemoryOp.STORE_FACT,
    MemoryOp.CREATE_EPISODE,
    MemoryOp.INFER_IMPLICIT,
})

CONFLICT_OPS: frozenset[MemoryOp] = frozenset({
    MemoryOp.UPDATE,
    MemoryOp.SUPERSEDE,
    MemoryOp.DECAY,
    MemoryOp.KEEP_BOTH,
})


class MemoryLayer(str, Enum):
    """Three-layer SGMem-validated memory architecture."""
    FACT    = "fact"     # atomic, key-addressable facts
    INSIGHT = "insight"  # synthesized cross-fact insights
    RAW     = "raw"      # verbatim sentences from conversation


# ---------------------------------------------------------------------------
# Structured action the model emits
# ---------------------------------------------------------------------------

@dataclass
class MemoryOpAction:
    """Structured action emitted by the memory manager for one conversation turn.

    Fields:
        op: The operation to perform.
        content: The text content to store / the compressed text / the abstention reason.
        key: Normalized snake_case key for addressable retrieval.
        confidence: Model's confidence in this operation (0.0–1.0).
        decay_factor: Used only with DECAY op. How much to reduce confidence.
        supersedes_key: Used with SUPERSEDE — the key being archived.
        temporal_markers: ISO timestamps or relative markers ("3 days after first call").
        layer: Which memory layer to write to (defaults to FACT for STORE_FACT).
        raw_response: The unparsed model output before JSON extraction.
        metadata: Anything extra (e.g., source_session, turn_idx).
    """
    op: MemoryOp
    content: str
    key: str
    confidence: float = 1.0
    decay_factor: float | None = None
    supersedes_key: str | None = None
    temporal_markers: list[str] = field(default_factory=list)
    layer: MemoryLayer = MemoryLayer.FACT
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op.value,
            "content": self.content,
            "key": self.key,
            "confidence": self.confidence,
            "decay_factor": self.decay_factor,
            "supersedes_key": self.supersedes_key,
            "temporal_markers": self.temporal_markers,
            "layer": self.layer.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], raw_response: str = "") -> "MemoryOpAction":
        return cls(
            op=MemoryOp(d["op"]),
            content=d["content"],
            key=d["key"],
            confidence=float(d.get("confidence", 1.0)),
            decay_factor=d.get("decay_factor"),
            supersedes_key=d.get("supersedes_key"),
            temporal_markers=d.get("temporal_markers", []),
            layer=MemoryLayer(d.get("layer", MemoryLayer.FACT.value)),
            raw_response=raw_response,
            metadata=d.get("metadata", {}),
        )

    def validate_key(self) -> bool:
        """Keys must be normalized snake_case, no leading digits."""
        return bool(re.match(r"^[a-z][a-z0-9_]*$", self.key))


# ---------------------------------------------------------------------------
# Trajectory building blocks
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """One turn in a conversation session.

    Args:
        role: "user" or "assistant"
        content: The text of this turn.
        timestamp: Optional ISO-8601 timestamp. Used by Base 6 (Temporal).
        metadata: Optional per-turn extras (e.g. detected_entities).
    """
    role: Literal["user", "assistant"]
    content: str
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass
class Session:
    """One session (conversation) in a multi-session trajectory.

    Args:
        session_id: Unique within a trajectory.
        turns: Ordered list of conversation turns.
        memory_snapshot: Memory state *after* this session ends (oracle ground truth).
        metadata: Vertical-specific extras (e.g. customer_id, ticket_id).
    """
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    memory_snapshot: dict[str, Any] | None = None  # set by generator
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turns": [t.to_dict() for t in self.turns],
            "memory_snapshot": self.memory_snapshot,
            "metadata": self.metadata,
        }


@dataclass
class QAPair:
    """A question-answer probe across sessions.

    Args:
        question: Natural-language question.
        answer: Gold answer string.
        difficulty: "easy" | "medium" | "hard"
        requires_sessions: Which session_ids must have been processed to answer.
        answer_type: "free_form" | "multiple_choice" | "boolean"
        choices: Only for multiple_choice. List of options.
        answer_idx: Index into choices for MC.
    """
    question: str
    answer: str
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    requires_sessions: list[str] = field(default_factory=list)
    answer_type: Literal["free_form", "multiple_choice", "boolean"] = "free_form"
    choices: list[str] = field(default_factory=list)
    answer_idx: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "requires_sessions": self.requires_sessions,
            "answer_type": self.answer_type,
            "choices": self.choices,
            "answer_idx": self.answer_idx,
        }


@dataclass
class ConflictExample:
    """Labeled training example for Base 3 (Contradict).

    Args:
        old_memory: The existing memory entry (key + content).
        new_claim: The new information that contradicts it.
        correct_op: Ground-truth operation (UPDATE/SUPERSEDE/DECAY/KEEP_BOTH).
        why: Explanation — used for filtering ambiguous cases.
        conflict_type: Semantic label for the conflict pattern.
    """
    old_memory: dict[str, str]   # {"key": "...", "content": "..."}
    new_claim: str
    correct_op: MemoryOp
    why: str
    conflict_type: Literal[
        "value_updated",          # fact changed (old still accurate at time of statement)
        "factual_correction",     # old fact was simply wrong
        "temporal_shift",         # old was true then, new is true now
        "complementary_view",     # both valid from different perspectives
    ] = "value_updated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_memory": self.old_memory,
            "new_claim": self.new_claim,
            "correct_op": self.correct_op.value,
            "why": self.why,
            "conflict_type": self.conflict_type,
        }


# ---------------------------------------------------------------------------
# Full training trajectory
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """Complete multi-session training trajectory.

    This is the unit of data that flows through the RL training pipeline.
    A single Trajectory = one training example for GRPO.

    Args:
        trajectory_id: Unique identifier (uuid or hash).
        sessions: Ordered list of sessions. Minimum 3 for vertical envs.
        qa_probes: QA probes that span across sessions (the actual reward signal).
        oracle_memory: Ground-truth memory state after all sessions (for hallucination check).
        vertical: Which environment this came from ("base_store", "voice_customer", etc.).
        conflict_examples: Labeled conflict pairs, used for Base 3.
        metadata: Anything extra (generator version, seed id, etc.).
    """
    trajectory_id: str
    sessions: list[Session]
    qa_probes: list[QAPair]
    oracle_memory: dict[str, Any] = field(default_factory=dict)
    vertical: str = "base"
    conflict_examples: list[ConflictExample] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "sessions": [s.to_dict() for s in self.sessions],
            "qa_probes": [q.to_dict() for q in self.qa_probes],
            "oracle_memory": self.oracle_memory,
            "vertical": self.vertical,
            "conflict_examples": [c.to_dict() for c in self.conflict_examples],
            "metadata": self.metadata,
        }

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)

    @property
    def n_turns(self) -> int:
        return sum(len(s.turns) for s in self.sessions)

    @property
    def session_ids(self) -> list[str]:
        return [s.session_id for s in self.sessions]


# ---------------------------------------------------------------------------
# Verifier output
# ---------------------------------------------------------------------------

class VerifierTier(str, Enum):
    TIER1 = "tier1"   # deterministic
    TIER2 = "tier2"   # LLM judge (Base 5 only)
    TIER3 = "tier3"   # outcome-based (frozen QA model / token F1 / Kendall tau)


@dataclass
class VerifierResult:
    """Result from any verifier tier.

    Args:
        score: Scalar in [−2, +1]. See per-env reward tables in the plan.
        tier: Which tier produced this result.
        passed: True if score > 0.
        reasons: Human-readable list of what passed / failed.
        metadata: Tier-specific extras (e.g. tau value, F1 breakdown).
    """
    score: float
    tier: VerifierTier
    passed: bool
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "tier": self.tier.value,
            "passed": self.passed,
            "reasons": self.reasons,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# JSON Schema for model output validation (Tier 1 schema check)
# ---------------------------------------------------------------------------

MEMORY_OP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "op": {
            "type": "string",
            "enum": [op.value for op in MemoryOp],
            "description": "The memory operation to perform.",
        },
        "content": {
            "type": "string",
            "minLength": 1,
            "description": "The text to store, compress, or the abstention reason.",
        },
        "key": {
            "type": "string",
            "pattern": "^[a-z][a-z0-9_]*$",
            "description": "Normalized snake_case retrieval key.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Model confidence in this operation.",
        },
        "decay_factor": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Required for DECAY op. How much to reduce confidence.",
        },
        "supersedes_key": {
            "type": "string",
            "description": "Required for SUPERSEDE. Key being moved to audit log.",
        },
        "temporal_markers": {
            "type": "array",
            "items": {"type": "string"},
            "description": "ISO timestamps or relative markers for Base 6 (Temporal).",
        },
        "layer": {
            "type": "string",
            "enum": [layer.value for layer in MemoryLayer],
            "description": "Which memory layer to write to.",
        },
    },
    "required": ["op", "content", "key"],
    "additionalProperties": False,
}
