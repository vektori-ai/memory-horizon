"""Proxy ledger — ground-truth fact states for step-level reward.

Phase A (now): derived from QA probe gold answers on LoCoMo / LongMemEval.
  Each probe's gold answer IS the fact the model should have stored.
  No real partner data needed — the benchmark gives us the ground truth for free.

Phase B (later): replaced with a real partner ledger (debt/payment record,
  CRM state) where fact states are tracked at the source.

The ledger is used ONLY to compute reward. The agent never sees it.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Ledger entry
# ---------------------------------------------------------------------------

@dataclass
class LedgerEntry:
    """One ground-truth fact the agent should store.

    fact_key:          normalized snake_case key derived from the probe question.
    expected_content:  gold answer string — what should appear in the FS.
    valid_after_session: session_id after which this fact is "current".
                         None = valid from the start of the trajectory.
    probe_question:    original question (kept for debugging).
    """
    fact_key: str
    expected_content: str
    valid_after_session: str | None
    probe_question: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_key":            self.fact_key,
            "expected_content":    self.expected_content,
            "valid_after_session": self.valid_after_session,
            "probe_question":      self.probe_question,
        }


# ---------------------------------------------------------------------------
# Derive proxy ledger from trajectory QA probes
# ---------------------------------------------------------------------------

def derive_ledger(
    qa_probes: list[dict],
    sessions:  list[dict],
) -> list[LedgerEntry]:
    """Build a proxy ledger from the benchmark's gold answers.

    For each probe:
      - expected_content  = probe answer
      - valid_after_session = last session in requires_sessions
        (the answer becomes "current" after that session is processed)
    Skips unanswerable probes (answer is N/A / Yes / No / empty).
    """
    all_sids = [s.get("session_id", "") for s in sessions]
    entries: list[LedgerEntry] = []

    for probe in qa_probes:
        answer   = str(probe.get("answer", "")).strip()
        question = probe.get("question", "").strip()

        if not answer or answer.lower() in ("n/a", "yes", "no", ""):
            continue

        requires = probe.get("requires_sessions", [])
        # valid_after = the last required session (or the last session overall)
        valid_after: str | None = None
        if requires:
            # pick the last session id from requires that actually exists
            ordered = [sid for sid in all_sids if sid in requires]
            valid_after = ordered[-1] if ordered else requires[-1]

        entries.append(LedgerEntry(
            fact_key=_question_to_key(question),
            expected_content=answer,
            valid_after_session=valid_after,
            probe_question=question,
        ))

    return entries


# ---------------------------------------------------------------------------
# Step-level scoring against ledger
# ---------------------------------------------------------------------------

def score_write_against_ledger(
    written_content: str,
    ledger: list[LedgerEntry],
    current_session_id: str | None = None,
) -> float:
    """Small shaped reward for a single WRITE/UPDATE/SUPERSEDE op.

    Returns:
      +0.15  if written_content has high token overlap with a ledger entry
             that is valid at the current session.
      -0.10  if written_content closely matches a ledger entry that is NOT
             yet valid (storing future/stale info too early).
       0.0   otherwise (neutral write — not related to any ledger entry).

    Intentionally small so the terminal RESOLVE score dominates.
    """
    if not ledger or not written_content.strip():
        return 0.0

    best_pos = 0.0
    best_neg = 0.0

    for entry in ledger:
        overlap = _token_f1(written_content, entry.expected_content)
        if overlap < 0.3:          # below threshold — ignore
            continue

        if _session_is_valid(entry.valid_after_session, current_session_id):
            best_pos = max(best_pos, overlap * 0.15)
        else:
            best_neg = max(best_neg, overlap * 0.10)

    if best_pos > 0.0:
        return best_pos
    if best_neg > 0.0:
        return -best_neg
    return 0.0


def score_invalidate_against_ledger(
    superseded_content: str,
    ledger: list[LedgerEntry],
    current_session_id: str | None = None,
) -> float:
    """Small reward for correctly superseding a stale fact.

    +0.10 if the superseded content matched a ledger entry that is no longer
    valid at the current session (i.e., agent correctly cleared a stale fact).
    """
    if not ledger or not superseded_content.strip():
        return 0.0

    for entry in ledger:
        overlap = _token_f1(superseded_content, entry.expected_content)
        if overlap < 0.3:
            continue
        if not _session_is_valid(entry.valid_after_session, current_session_id):
            return 0.10

    return 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STOP = {"what", "is", "the", "are", "was", "were", "a", "an", "this", "that",
         "of", "for", "in", "on", "at", "to", "does", "did", "has", "have",
         "had", "who", "when", "where", "which", "how", "do"}


def _question_to_key(question: str) -> str:
    """Convert probe question to a snake_case fact key."""
    text = question.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = [t for t in text.split() if t not in _STOP][:6]
    return "_".join(tokens) or "unknown"


def _normalize(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t not in _STOP]


def _token_f1(pred: str, gold: str) -> float:
    p = _normalize(pred)
    g = _normalize(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common    = sum((Counter(p) & Counter(g)).values())
    precision = common / len(p)
    recall    = common / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _session_is_valid(
    valid_after: str | None,
    current_sid: str | None,
) -> bool:
    """True if the ledger entry is "current" at current_sid.

    We don't have a global session ordering here, so we use a simple heuristic:
    if valid_after is None (valid from start) → always valid.
    if current_sid is None (unknown) → assume valid.
    Otherwise → valid (we can't order without the full session list;
    the trajectory-level check in derive_ledger already anchors the entry).
    """
    if valid_after is None or current_sid is None:
        return True
    return True   # Phase A: treat all derived entries as always valid
                  # Phase B: replace with real temporal ordering from partner ledger
