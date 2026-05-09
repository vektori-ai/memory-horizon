"""Tier 1 deterministic verifier for all memory_horizon base environments.

Zero LLM cost. Always runs before any other tier. Each check is a pure function
over the MemoryOpAction and (optionally) the current MemoryStore state.

Per-env check dispatch:
    Base 1 (Store)      → check_store_op
    Base 2 (Synthesize) → check_synthesize_response
    Base 3 (Contradict) → check_contradict_op
    Base 4 (Compress)   → check_compress_op
    Base 5 (Abstain)    → check_abstain_response
    Base 6 (Temporal)   → check_temporal_op
    Base 7 (Integration)→ runs all checks relevant to the op type

Reward conventions (match the plan exactly):
    +1.0  correct operation
    -0.5  no-op UPDATE (new value same as old)
    -2.0  SUPERSEDE without audit trail
    -2.0  KEEP_BOTH without distinct keys+timestamps
    -2.0  overwriting historical fact (UPDATE on a fact with audit trail = ok;
          SUPERSEDE with audit trail = ok; raw DELETE = -2)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from memory_horizon.memory_store import MemoryStore, StoreExecResult
from memory_horizon.mh_types import (
    MEMORY_OP_SCHEMA,
    CONFLICT_OPS,
    STORE_OPS,
    MemoryOp,
    MemoryOpAction,
    VerifierResult,
    VerifierTier,
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class Tier1Result:
    """Aggregated Tier 1 result for one step.

    Args:
        schema_valid: Did the action pass JSON schema validation?
        op_valid: Is the op type legal for this environment?
        content_valid: Is content non-empty and not a verbatim transcript copy?
        key_valid: Is the key normalized snake_case?
        op_specific_score: Score from the per-op check (can be negative).
        reasons: Human-readable list of what passed / failed.
        total_score: Sum of all sub-scores, clipped to [-2, +1].
        exec_result: The StoreExecResult from applying the action (may be None).
    """
    schema_valid: bool = False
    op_valid: bool = False
    content_valid: bool = False
    key_valid: bool = False
    op_specific_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    total_score: float = 0.0
    exec_result: StoreExecResult | None = None

    def to_verifier_result(self) -> VerifierResult:
        return VerifierResult(
            score=self.total_score,
            tier=VerifierTier.TIER1,
            passed=self.total_score > 0,
            reasons=self.reasons,
            metadata={
                "schema_valid": self.schema_valid,
                "op_valid": self.op_valid,
                "content_valid": self.content_valid,
                "key_valid": self.key_valid,
                "op_specific_score": self.op_specific_score,
            },
        )


# ---------------------------------------------------------------------------
# Main verifier class
# ---------------------------------------------------------------------------

class Tier1Verifier:
    """Stateless Tier 1 verifier. Instantiate once per env, call per step.

    Args:
        env_type: One of "store", "synthesize", "contradict", "compress",
                  "abstain", "temporal", "integration".
        allowed_ops: If set, restrict which ops are legal for this env.
    """

    def __init__(
        self,
        env_type: str = "store",
        allowed_ops: set[MemoryOp] | None = None,
    ) -> None:
        self.env_type = env_type
        self.allowed_ops = allowed_ops or _default_allowed_ops(env_type)

    def verify(
        self,
        action: MemoryOpAction,
        store: MemoryStore | None = None,
        exec_result: StoreExecResult | None = None,
        raw_transcript: str = "",
    ) -> Tier1Result:
        """Run all Tier 1 checks for one step.

        Args:
            action: The parsed MemoryOpAction from the model.
            store: Current memory store (before or after execute — used for
                   contradict checks that need the existing value).
            exec_result: Result of store.execute(action) — tells us if SUPERSEDE
                         moved the old value to audit log, etc.
            raw_transcript: The raw conversation text for this turn (used to
                            detect verbatim copy-paste in content).

        Returns:
            A Tier1Result with an aggregated score.
        """
        result = Tier1Result()

        # --- Schema ---
        schema_ok, schema_reasons = _check_schema(action)
        result.schema_valid = schema_ok
        result.reasons.extend(schema_reasons)

        if not schema_ok:
            result.total_score = -1.0
            return result

        # --- Op type ---
        op_ok, op_reasons = _check_op_type(action.op, self.allowed_ops)
        result.op_valid = op_ok
        result.reasons.extend(op_reasons)

        # --- Content ---
        content_ok, content_reasons = _check_content(action.content, raw_transcript)
        result.content_valid = content_ok
        result.reasons.extend(content_reasons)

        # --- Key ---
        key_ok, key_reasons = _check_key(action.key)
        result.key_valid = key_ok
        result.reasons.extend(key_reasons)

        # --- Op-specific ---
        op_score, op_reasons = self._check_op_specific(action, store, exec_result)
        result.op_specific_score = op_score
        result.reasons.extend(op_reasons)

        # Aggregate
        base = 1.0 if (schema_ok and op_ok and content_ok and key_ok) else 0.0
        result.total_score = max(-2.0, min(1.0, base + op_score))
        result.exec_result = exec_result
        return result

    def verify_synthesize_response(
        self,
        response: str,
        oracle_memory: dict[str, Any],
        gold_answer: str,
    ) -> Tier1Result:
        """Tier 1 for Base 2 (Synthesize): hallucination pre-check only.

        Full scoring (token F1) is Tier 3. Here we only flag responses that
        contain content provably absent from oracle memory.
        """
        result = Tier1Result()
        result.schema_valid = True
        result.op_valid = True
        result.key_valid = True
        result.content_valid = bool(response.strip())

        if not response.strip():
            result.reasons.append("FAIL: empty response")
            result.total_score = -1.0
            return result

        # Hallucination pre-check: any numeric token in response that's not in oracle?
        oracle_text = " ".join(str(v) for v in oracle_memory.values())
        hallucinated = _find_hallucinated_numbers(response, oracle_text)
        if hallucinated:
            result.reasons.append(f"FAIL: numeric hallucination detected: {hallucinated[:3]}")
            result.op_specific_score = -2.0
        else:
            result.reasons.append("PASS: no numeric hallucination detected")

        result.total_score = max(-2.0, 1.0 + result.op_specific_score)
        return result

    def verify_abstain_response(
        self,
        response: str,
        memory_has_answer: bool,
    ) -> Tier1Result:
        """Tier 1 for Base 5 (Abstain): presence of explicit abstention signal.

        Args:
            response: The model's full text response.
            memory_has_answer: Whether the memory store actually contains the answer.
        """
        result = Tier1Result()
        result.schema_valid = True
        result.op_valid = True
        result.key_valid = True
        result.content_valid = bool(response.strip())

        has_abstain_signal = _has_abstention_signal(response)

        if not memory_has_answer:
            # Model SHOULD abstain.
            if has_abstain_signal:
                result.reasons.append("PASS: correct abstention signal present")
                result.op_specific_score = 0.0  # +1 from base score
            else:
                result.reasons.append("FAIL: missing abstention signal when answer not in memory")
                result.op_specific_score = -1.0
        else:
            # Model SHOULD answer, not abstain.
            if has_abstain_signal:
                result.reasons.append("FAIL: over-abstained when memory has the answer (-1)")
                result.op_specific_score = -1.0
            else:
                result.reasons.append("PASS: answered when memory has the answer")
                result.op_specific_score = 0.0

        result.total_score = max(-2.0, min(1.0, 1.0 + result.op_specific_score))
        return result

    # -----------------------------------------------------------------------
    # Per-op dispatch (private)
    # -----------------------------------------------------------------------

    def _check_op_specific(
        self,
        action: MemoryOpAction,
        store: MemoryStore | None,
        exec_result: StoreExecResult | None,
    ) -> tuple[float, list[str]]:
        """Returns (score_delta, reasons)."""
        op = action.op

        if op in STORE_OPS:
            return _check_store_specifics(action)
        elif op == MemoryOp.UPDATE:
            return _check_update(action, exec_result)
        elif op == MemoryOp.SUPERSEDE:
            return _check_supersede(action, exec_result)
        elif op == MemoryOp.DECAY:
            return _check_decay(action)
        elif op == MemoryOp.KEEP_BOTH:
            return _check_keep_both(action, exec_result)
        elif op == MemoryOp.COMPRESS:
            return _check_compress(action, store)
        elif op == MemoryOp.ABSTAIN:
            return 0.0, ["INFO: ABSTAIN op — outcome scored in Tier 3"]
        return 0.0, []


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def _check_schema(action: MemoryOpAction) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    ok = True

    if not action.op:
        reasons.append("FAIL: missing 'op' field")
        ok = False
    if not action.content or not action.content.strip():
        reasons.append("FAIL: 'content' is empty")
        ok = False
    if not action.key:
        reasons.append("FAIL: missing 'key' field")
        ok = False
    if not (0.0 <= action.confidence <= 1.0):
        reasons.append(f"FAIL: confidence {action.confidence} outside [0, 1]")
        ok = False
    if action.decay_factor is not None and not (0.0 <= action.decay_factor <= 1.0):
        reasons.append(f"FAIL: decay_factor {action.decay_factor} outside [0, 1]")
        ok = False

    if ok:
        reasons.append("PASS: schema valid")
    return ok, reasons


# ---------------------------------------------------------------------------
# Op type check
# ---------------------------------------------------------------------------

def _check_op_type(op: MemoryOp, allowed: set[MemoryOp]) -> tuple[bool, list[str]]:
    if op in allowed:
        return True, [f"PASS: op {op.value} is allowed"]
    return False, [f"FAIL: op {op.value} not allowed in this env (allowed: {[o.value for o in allowed]})"]


# ---------------------------------------------------------------------------
# Content check
# ---------------------------------------------------------------------------

def _check_content(content: str, raw_transcript: str) -> tuple[bool, list[str]]:
    if not content.strip():
        return False, ["FAIL: content is empty"]

    # Reject verbatim copy-paste of the transcript (≥ 90% overlap by words).
    if raw_transcript:
        transcript_words = set(raw_transcript.lower().split())
        content_words = set(content.lower().split())
        if transcript_words and len(content_words) > 5:
            overlap = len(content_words & transcript_words) / len(content_words)
            if overlap >= 0.90:
                return False, [f"FAIL: content appears verbatim copy of transcript ({overlap:.0%} overlap)"]

    return True, ["PASS: content non-empty and not verbatim copy"]


# ---------------------------------------------------------------------------
# Key check
# ---------------------------------------------------------------------------

_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")

def _check_key(key: str) -> tuple[bool, list[str]]:
    if _KEY_RE.match(key):
        return True, [f"PASS: key '{key}' is valid snake_case"]
    return False, [f"FAIL: key '{key}' is not normalized snake_case (must match ^[a-z][a-z0-9_]*$)"]


# ---------------------------------------------------------------------------
# Per-op specific checks
# ---------------------------------------------------------------------------

def _check_store_specifics(action: MemoryOpAction) -> tuple[float, list[str]]:
    reasons: list[str] = []

    if action.op == MemoryOp.INFER_IMPLICIT:
        # Implicit inferences should have confidence < 1.0 (they're inferences, not facts).
        if action.confidence >= 1.0:
            reasons.append("WARN: INFER_IMPLICIT with confidence=1.0 — inferences rarely certain")
        else:
            reasons.append(f"PASS: INFER_IMPLICIT confidence={action.confidence:.2f}")

    return 0.0, reasons


def _check_update(
    action: MemoryOpAction,
    exec_result: StoreExecResult | None,
) -> tuple[float, list[str]]:
    if exec_result and exec_result.error == "noop_update":
        return -0.5, ["FAIL: no-op UPDATE — new value identical to existing (-0.5)"]
    return 0.0, ["PASS: UPDATE changed the stored value"]


def _check_supersede(
    action: MemoryOpAction,
    exec_result: StoreExecResult | None,
) -> tuple[float, list[str]]:
    # The plan: -2 for overwriting historical fact without audit trail.
    # Our store always puts the old value in audit_log on SUPERSEDE.
    # But if supersedes_key was missing, exec_result.success=False.
    if exec_result and not exec_result.success:
        return -2.0, [f"FAIL: SUPERSEDE failed — {exec_result.error} (-2.0)"]

    if not action.supersedes_key:
        return -1.0, ["WARN: SUPERSEDE without explicit supersedes_key field (-1.0)"]

    archived = exec_result.side_effects.get("archived_content") if exec_result else None
    if archived:
        return 0.0, [f"PASS: SUPERSEDE archived old value: '{archived[:60]}'"]
    return 0.0, ["PASS: SUPERSEDE executed"]


def _check_decay(action: MemoryOpAction) -> tuple[float, list[str]]:
    if action.decay_factor is None:
        return -2.0, ["FAIL: DECAY op missing required decay_factor field (-2.0)"]
    return 0.0, [f"PASS: DECAY decay_factor={action.decay_factor:.2f}"]


def _check_keep_both(
    action: MemoryOpAction,
    exec_result: StoreExecResult | None,
) -> tuple[float, list[str]]:
    if exec_result and not exec_result.success:
        return -2.0, ["FAIL: KEEP_BOTH failed to store both values (-2.0)"]

    old_key = exec_result.side_effects.get("old_key_preserved") if exec_result else None
    if old_key:
        return 0.0, [f"PASS: KEEP_BOTH preserved old value under '{old_key}'"]

    # First time writing this key — KEEP_BOTH without a prior value is suspicious.
    return 0.0, ["INFO: KEEP_BOTH on new key (no prior value to preserve)"]


def _check_compress(
    action: MemoryOpAction,
    store: MemoryStore | None,
) -> tuple[float, list[str]]:
    if store is None:
        return 0.0, ["INFO: COMPRESS — no store provided for length check"]

    existing = store.get_fact(action.key)
    if existing is None:
        return -1.0, [f"FAIL: COMPRESS target key '{action.key}' not in store (-1.0)"]

    original_len = len(existing.content)
    compressed_len = len(action.content)

    if original_len == 0:
        return 0.0, ["INFO: COMPRESS on empty content"]

    ratio = compressed_len / original_len
    if ratio >= 0.80:
        return -0.5, [
            f"FAIL: compression ratio {ratio:.1%} ≥ 80% — must compress by at least 20% (-0.5)"
        ]
    if ratio < 0.05:
        return -1.0, [
            f"FAIL: extreme compression {ratio:.1%} < 5% — likely reward hack (-1.0)"
        ]

    return 0.0, [f"PASS: COMPRESS ratio={ratio:.1%} within acceptable range [5%, 80%)"]


# ---------------------------------------------------------------------------
# Abstention signal detection (used by verify_abstain_response)
# ---------------------------------------------------------------------------

_ABSTAIN_SIGNALS = [
    "<abstain>",
    "i don't have that information",
    "i don't have information",
    "not in my records",
    "i cannot find",
    "no information available",
    "unable to find",
    "not stored in memory",
    "i lack",
    "i do not have",
]

def _has_abstention_signal(response: str) -> bool:
    r = response.lower()
    return any(sig in r for sig in _ABSTAIN_SIGNALS)


# ---------------------------------------------------------------------------
# Numeric hallucination pre-check (Base 2 Synthesize)
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\b\d[\d,\.]*\b")

def _find_hallucinated_numbers(response: str, oracle_text: str) -> list[str]:
    oracle_numbers = set(_NUMBER_RE.findall(oracle_text))
    response_numbers = _NUMBER_RE.findall(response)
    return [n for n in response_numbers if n not in oracle_numbers]


# ---------------------------------------------------------------------------
# Default allowed ops per env type
# ---------------------------------------------------------------------------

def _default_allowed_ops(env_type: str) -> set[MemoryOp]:
    mapping: dict[str, set[MemoryOp]] = {
        "store":      STORE_OPS,
        "synthesize": {MemoryOp.ABSTAIN},  # synthesize env: model answers, not stores
        "contradict": CONFLICT_OPS,
        "compress":   {MemoryOp.COMPRESS},
        "abstain":    {MemoryOp.ABSTAIN},
        "temporal":   STORE_OPS | {MemoryOp.UPDATE},
        "integration": set(MemoryOp),      # all ops allowed
    }
    return mapping.get(env_type, set(MemoryOp))


# ---------------------------------------------------------------------------
# Convenience: parse a model's raw JSON output into a MemoryOpAction
# ---------------------------------------------------------------------------

def parse_action(raw_output: str) -> tuple[MemoryOpAction | None, str]:
    """Extract and parse a MemoryOpAction from a model's raw output.

    Returns:
        (action, error_message) — action is None if parsing failed.
    """
    # Try to extract JSON from the output (model may wrap in prose).
    json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not json_match:
        return None, "No JSON object found in model output"

    try:
        d = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"

    required = {"op", "content", "key"}
    missing = required - d.keys()
    if missing:
        return None, f"Missing required fields: {missing}"

    try:
        action = MemoryOpAction.from_dict(d, raw_response=raw_output)
    except (ValueError, KeyError) as e:
        return None, f"Action construction failed: {e}"

    return action, ""
