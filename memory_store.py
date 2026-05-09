"""Three-layer memory store for memory_horizon environments.

Architecture (SGMem-validated):
    Layer 1 — Facts:    atomic, key-addressable entries. Direct STORE/UPDATE/etc.
    Layer 2 — Insights: synthesized cross-fact entries. Written by INFER_IMPLICIT.
    Layer 3 — Raw:      verbatim sentence log per turn. Append-only.

The store is the *environment* side of the memory system, not the model.
The model decides *what* to store; the store *executes* it and tracks state.
"""

from __future__ import annotations

import copy
import datetime
from dataclasses import dataclass, field
from typing import Any

from memory_horizon.mh_types import MemoryLayer, MemoryOp, MemoryOpAction


# ---------------------------------------------------------------------------
# Layer-specific entry types
# ---------------------------------------------------------------------------

@dataclass
class FactEntry:
    """One entry in the FACT layer.

    Args:
        key: Normalized snake_case retrieval key.
        content: The stored text.
        confidence: 0.0–1.0. Reduced by DECAY; reset on UPDATE.
        timestamp: ISO-8601 creation time.
        source_session: Which session created this entry.
        audit_log: History of superseded values (populated by SUPERSEDE).
        decay_factor: Cumulative decay applied so far.
        is_active: False if soft-deleted (not returned by get_context).
    """
    key: str
    content: str
    confidence: float = 1.0
    timestamp: str = ""
    source_session: str = ""
    audit_log: list[dict[str, Any]] = field(default_factory=list)
    decay_factor: float = 1.0
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "content": self.content,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "source_session": self.source_session,
            "audit_log": self.audit_log,
            "decay_factor": self.decay_factor,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FactEntry":
        return cls(**d)


@dataclass
class InsightEntry:
    """One entry in the INSIGHT layer (cross-fact synthesis).

    Args:
        key: Snake_case key.
        content: Synthesized insight text.
        source_keys: Which FACT keys contributed to this insight.
        timestamp: ISO-8601 creation time.
        confidence: Model's confidence in the synthesis.
    """
    key: str
    content: str
    source_keys: list[str] = field(default_factory=list)
    timestamp: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "content": self.content,
            "source_keys": self.source_keys,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InsightEntry":
        return cls(**d)


@dataclass
class RawEntry:
    """One entry in the RAW layer — verbatim from the conversation.

    Append-only. Never updated or deleted. Used for hallucination detection.
    """
    content: str
    session_id: str
    turn_idx: int
    role: str = "user"
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "session_id": self.session_id,
            "turn_idx": self.turn_idx,
            "role": self.role,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RawEntry":
        return cls(**d)


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------

@dataclass
class StoreExecResult:
    """Result of executing a MemoryOpAction against the store.

    Args:
        success: Whether the operation applied cleanly.
        op: The operation that ran.
        affected_key: The primary key touched.
        error: Non-empty string if success=False.
        side_effects: Auxiliary info (e.g. old value moved to audit log).
    """
    success: bool
    op: MemoryOp
    affected_key: str
    error: str = ""
    side_effects: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

class MemoryStore:
    """Stateful three-layer memory store for one training episode.

    Usage::

        store = MemoryStore(persist_across_sessions=True)
        store.session_boundary("session_1")

        action = MemoryOpAction(op=MemoryOp.STORE_FACT, content="User is John",
                                key="user_name")
        result = store.execute(action)

        ctx = store.get_context()        # inject into next system prompt
        snap = store.snapshot()          # save after session ends
        store.session_boundary("session_2")   # session 2 begins

    Args:
        persist_across_sessions: If True (default), facts survive session boundaries.
            If False, the store is wiped at each new session (useful for ablations).
        max_facts: Hard cap on FACT layer size. Oldest entries displaced on overflow.
        max_raw: Hard cap on RAW layer size.
    """

    def __init__(
        self,
        persist_across_sessions: bool = True,
        max_facts: int = 500,
        max_raw: int = 2000,
    ) -> None:
        self.persist_across_sessions = persist_across_sessions
        self.max_facts = max_facts
        self.max_raw = max_raw

        self._facts: dict[str, FactEntry] = {}
        self._insights: dict[str, InsightEntry] = {}
        self._raw: list[RawEntry] = []

        self._current_session_id: str = ""
        self._session_history: list[str] = []   # ordered list of session ids seen
        self._op_log: list[dict[str, Any]] = []  # audit trail of every operation

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def session_boundary(self, new_session_id: str) -> None:
        """Transition to a new session.

        If persist_across_sessions=False, clears all layers.
        Always records the new session in history.
        """
        if self._current_session_id:
            self._session_history.append(self._current_session_id)
        self._current_session_id = new_session_id

        if not self.persist_across_sessions:
            self._facts.clear()
            self._insights.clear()
            self._raw.clear()

    def log_raw(self, content: str, turn_idx: int, role: str = "user") -> None:
        """Append a verbatim turn to the RAW layer. Called by the env, not the model."""
        entry = RawEntry(
            content=content,
            session_id=self._current_session_id,
            turn_idx=turn_idx,
            role=role,
            timestamp=_now(),
        )
        self._raw.append(entry)
        if len(self._raw) > self.max_raw:
            self._raw.pop(0)

    def execute(self, action: MemoryOpAction) -> StoreExecResult:
        """Apply a memory operation to the store.

        Dispatches to the per-op handler. All mutations go through here.
        Returns a StoreExecResult describing what happened.
        """
        handler = {
            MemoryOp.STORE_FACT:      self._op_store_fact,
            MemoryOp.CREATE_EPISODE:  self._op_create_episode,
            MemoryOp.INFER_IMPLICIT:  self._op_infer_implicit,
            MemoryOp.UPDATE:          self._op_update,
            MemoryOp.SUPERSEDE:       self._op_supersede,
            MemoryOp.DECAY:           self._op_decay,
            MemoryOp.KEEP_BOTH:       self._op_keep_both,
            MemoryOp.COMPRESS:        self._op_compress,
            MemoryOp.ABSTAIN:         self._op_abstain,
        }.get(action.op)

        if handler is None:
            return StoreExecResult(
                success=False,
                op=action.op,
                affected_key=action.key,
                error=f"Unknown op: {action.op}",
            )

        result = handler(action)
        self._op_log.append({
            "session_id": self._current_session_id,
            "op": action.op.value,
            "key": action.key,
            "success": result.success,
            "error": result.error,
            "timestamp": _now(),
        })
        return result

    def get_context(self, query: str | None = None, max_facts: int = 50) -> str:
        """Return formatted memory context for injection into the system prompt.

        Args:
            query: Optional query string. If provided, facts are sorted by
                rough keyword overlap (proxy for relevance — no embeddings needed).
            max_facts: Max number of FACT entries to include.

        Returns:
            A formatted string ready to inject into the system prompt.
        """
        active_facts = [f for f in self._facts.values() if f.is_active]

        if query and active_facts:
            query_words = set(query.lower().split())
            active_facts = sorted(
                active_facts,
                key=lambda f: -len(query_words & set(f.content.lower().split())),
            )

        active_facts = active_facts[:max_facts]
        insights = list(self._insights.values())

        parts: list[str] = []

        if active_facts:
            fact_lines = []
            for f in active_facts:
                conf_tag = f" (conf={f.confidence:.2f})" if f.confidence < 0.9 else ""
                fact_lines.append(f"  [{f.key}]{conf_tag} {f.content}")
            parts.append("## Stored Facts\n" + "\n".join(fact_lines))

        if insights:
            insight_lines = [f"  [{i.key}] {i.content}" for i in insights]
            parts.append("## Insights\n" + "\n".join(insight_lines))

        if not parts:
            return "<memory: empty>"

        return "\n\n".join(parts)

    def snapshot(self) -> dict[str, Any]:
        """Return a fully JSON-serializable snapshot of the store's current state.

        Used to:
        - Record oracle memory after a session (by the data generator)
        - Save/restore checkpoints during evaluation
        - Feed into hallucination verifier as the "oracle" set
        """
        return {
            "facts": {k: v.to_dict() for k, v in self._facts.items()},
            "insights": {k: v.to_dict() for k, v in self._insights.items()},
            "raw": [r.to_dict() for r in self._raw],
            "current_session_id": self._current_session_id,
            "session_history": list(self._session_history),
        }

    def load_snapshot(self, snap: dict[str, Any]) -> None:
        """Restore store state from a snapshot dict."""
        self._facts = {
            k: FactEntry.from_dict(v) for k, v in snap.get("facts", {}).items()
        }
        self._insights = {
            k: InsightEntry.from_dict(v) for k, v in snap.get("insights", {}).items()
        }
        self._raw = [RawEntry.from_dict(r) for r in snap.get("raw", [])]
        self._current_session_id = snap.get("current_session_id", "")
        self._session_history = snap.get("session_history", [])

    def get_fact(self, key: str) -> FactEntry | None:
        entry = self._facts.get(key)
        return entry if (entry and entry.is_active) else None

    def get_all_active_facts(self) -> dict[str, FactEntry]:
        return {k: v for k, v in self._facts.items() if v.is_active}

    def get_all_content(self) -> list[str]:
        """Flat list of all active content strings — used by hallucination verifier."""
        out: list[str] = []
        out.extend(f.content for f in self._facts.values() if f.is_active)
        out.extend(i.content for i in self._insights.values())
        return out

    def op_log(self) -> list[dict[str, Any]]:
        return list(self._op_log)

    def stats(self) -> dict[str, int]:
        return {
            "n_facts": sum(1 for f in self._facts.values() if f.is_active),
            "n_facts_inactive": sum(1 for f in self._facts.values() if not f.is_active),
            "n_insights": len(self._insights),
            "n_raw": len(self._raw),
            "n_sessions": len(self._session_history) + (1 if self._current_session_id else 0),
        }

    # -----------------------------------------------------------------------
    # Per-op handlers (private)
    # -----------------------------------------------------------------------

    def _op_store_fact(self, action: MemoryOpAction) -> StoreExecResult:
        if action.key in self._facts and self._facts[action.key].is_active:
            # Key collision — treat as implicit UPDATE rather than silent overwrite.
            return self._op_update(action)

        entry = FactEntry(
            key=action.key,
            content=action.content,
            confidence=action.confidence,
            timestamp=_now(),
            source_session=self._current_session_id,
        )
        if action.temporal_markers:
            entry.audit_log.append({"temporal_markers": action.temporal_markers})

        self._facts[action.key] = entry
        self._enforce_fact_cap()
        return StoreExecResult(success=True, op=action.op, affected_key=action.key)

    def _op_create_episode(self, action: MemoryOpAction) -> StoreExecResult:
        # Episodes go to FACT layer with layer=INSIGHT tagging in metadata.
        # Structurally identical to STORE_FACT but semantically distinct for verifiers.
        entry = FactEntry(
            key=action.key,
            content=action.content,
            confidence=action.confidence,
            timestamp=_now(),
            source_session=self._current_session_id,
            audit_log=[{"episode": True, "temporal_markers": action.temporal_markers}],
        )
        self._facts[action.key] = entry
        self._enforce_fact_cap()
        return StoreExecResult(success=True, op=action.op, affected_key=action.key)

    def _op_infer_implicit(self, action: MemoryOpAction) -> StoreExecResult:
        # Implicit inferences go to the INSIGHT layer.
        entry = InsightEntry(
            key=action.key,
            content=action.content,
            source_keys=action.metadata.get("source_keys", []),
            timestamp=_now(),
            confidence=action.confidence,
        )
        self._insights[action.key] = entry
        return StoreExecResult(success=True, op=action.op, affected_key=action.key)

    def _op_update(self, action: MemoryOpAction) -> StoreExecResult:
        existing = self._facts.get(action.key)
        if existing is None or not existing.is_active:
            # Key doesn't exist yet — fall back to store.
            return self._op_store_fact(action)

        if existing.content == action.content:
            # No-op UPDATE: Tier 1 penalizes this (-0.5).
            return StoreExecResult(
                success=True,
                op=action.op,
                affected_key=action.key,
                error="noop_update",
                side_effects={"old_content": existing.content},
            )

        old_content = existing.content
        existing.content = action.content
        existing.confidence = action.confidence
        existing.timestamp = _now()
        existing.audit_log.append({"op": "update", "old_content": old_content, "timestamp": _now()})
        return StoreExecResult(
            success=True,
            op=action.op,
            affected_key=action.key,
            side_effects={"old_content": old_content},
        )

    def _op_supersede(self, action: MemoryOpAction) -> StoreExecResult:
        supersedes_key = action.supersedes_key or action.key
        old = self._facts.get(supersedes_key)

        if old is None:
            return StoreExecResult(
                success=False,
                op=action.op,
                affected_key=action.key,
                error=f"supersedes_key '{supersedes_key}' not found in store",
            )

        archived = {
            "op": "supersede",
            "old_key": supersedes_key,
            "old_content": old.content,
            "old_confidence": old.confidence,
            "archived_at": _now(),
        }
        old.is_active = False
        old.audit_log.append(archived)

        new_entry = FactEntry(
            key=action.key,
            content=action.content,
            confidence=action.confidence,
            timestamp=_now(),
            source_session=self._current_session_id,
            audit_log=[archived],
        )
        self._facts[action.key] = new_entry
        self._enforce_fact_cap()
        return StoreExecResult(
            success=True,
            op=action.op,
            affected_key=action.key,
            side_effects={"archived_key": supersedes_key, "archived_content": old.content},
        )

    def _op_decay(self, action: MemoryOpAction) -> StoreExecResult:
        existing = self._facts.get(action.key)
        if existing is None or not existing.is_active:
            return StoreExecResult(
                success=False,
                op=action.op,
                affected_key=action.key,
                error=f"key '{action.key}' not found for DECAY",
            )

        if action.decay_factor is None:
            return StoreExecResult(
                success=False,
                op=action.op,
                affected_key=action.key,
                error="DECAY op requires decay_factor field",
            )

        old_conf = existing.confidence
        existing.confidence = round(existing.confidence * action.decay_factor, 4)
        existing.decay_factor = action.decay_factor
        existing.audit_log.append({
            "op": "decay",
            "old_confidence": old_conf,
            "new_confidence": existing.confidence,
            "decay_factor": action.decay_factor,
            "timestamp": _now(),
        })
        return StoreExecResult(
            success=True,
            op=action.op,
            affected_key=action.key,
            side_effects={"old_confidence": old_conf, "new_confidence": existing.confidence},
        )

    def _op_keep_both(self, action: MemoryOpAction) -> StoreExecResult:
        # Store the new value under action.key; old value stays under its own key.
        # Both entries get distinct timestamps.
        existing = self._facts.get(action.key)
        if existing is not None and existing.is_active:
            old_key = f"{action.key}_v{len(existing.audit_log) + 1}"
            archived = copy.deepcopy(existing)
            archived.key = old_key
            self._facts[old_key] = archived

        new_entry = FactEntry(
            key=action.key,
            content=action.content,
            confidence=action.confidence,
            timestamp=_now(),
            source_session=self._current_session_id,
            audit_log=[{"op": "keep_both", "note": "new version stored alongside old"}],
        )
        self._facts[action.key] = new_entry
        self._enforce_fact_cap()
        return StoreExecResult(
            success=True,
            op=action.op,
            affected_key=action.key,
            side_effects={"old_key_preserved": existing.key if existing else None},
        )

    def _op_compress(self, action: MemoryOpAction) -> StoreExecResult:
        existing = self._facts.get(action.key)
        if existing is None or not existing.is_active:
            return StoreExecResult(
                success=False,
                op=action.op,
                affected_key=action.key,
                error=f"key '{action.key}' not found for COMPRESS",
            )

        old_len = len(existing.content)
        old_content = existing.content
        existing.content = action.content
        existing.audit_log.append({
            "op": "compress",
            "old_len": old_len,
            "new_len": len(action.content),
            "ratio": round(len(action.content) / max(old_len, 1), 3),
            "timestamp": _now(),
        })
        return StoreExecResult(
            success=True,
            op=action.op,
            affected_key=action.key,
            side_effects={
                "old_len": old_len,
                "new_len": len(action.content),
                "old_content": old_content,
            },
        )

    def _op_abstain(self, action: MemoryOpAction) -> StoreExecResult:
        # ABSTAIN does not modify the store. It's recorded in the op_log only.
        return StoreExecResult(
            success=True,
            op=action.op,
            affected_key=action.key,
            side_effects={"reason": action.content},
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _enforce_fact_cap(self) -> None:
        active = [(k, v) for k, v in self._facts.items() if v.is_active]
        if len(active) <= self.max_facts:
            return
        # Evict oldest by timestamp (FIFO).
        active.sort(key=lambda kv: kv[1].timestamp)
        for key, _ in active[: len(active) - self.max_facts]:
            self._facts[key].is_active = False


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

