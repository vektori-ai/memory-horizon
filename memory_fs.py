"""VirtualFilesystem — append-only memory store with importance tags.

Each path (category/entity) maps to an ordered list of tagged content lines.
Each path also carries an importance tag: tentative / confirmed / superseded.

Importance ordering mirrors Harness-1:
  confirmed  — verified by multiple writes or explicit SUPERSEDE
  tentative  — initial STORE_FACT (auto-seed level, "fair" equivalent)
  superseded — explicitly overwritten; kept for audit, low weight

The model reads render_for_prompt() at each turn to see its current memory state
and writes ops (STORE_FACT / UPDATE / SUPERSEDE / COMPRESS / ABSTAIN) to update it.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

_STOP = {"what", "is", "the", "are", "was", "were", "a", "an", "this", "that",
         "of", "for", "in", "on", "at", "to", "does", "did", "has", "have", "had"}

_WRITE_OPS = {"STORE_FACT", "UPDATE", "SUPERSEDE", "COMPRESS"}

Importance = Literal["tentative", "confirmed", "superseded"]

# Importance rank for prompt ordering (lower = higher priority)
_IMPORTANCE_RANK: dict[str, int] = {
    "confirmed":  0,
    "tentative":  1,
    "superseded": 2,
}


@dataclass
class VirtualFilesystem:
    files:      dict[str, list[str]]  = field(default_factory=dict)
    importance: dict[str, Importance] = field(default_factory=dict)

    def apply_op(self, op: dict, session_idx: int, turn_idx: int) -> None:
        """Apply a parsed memory op to the filesystem.

        STORE_FACT: append line → tag becomes tentative (or stays confirmed).
        UPDATE:     append line → tag promoted to confirmed.
        SUPERSEDE:  clear path, append new line → tag set to confirmed; old path
                    archived as superseded under a _prior suffix.
        COMPRESS:   replace path with summary → tag becomes confirmed.
        ABSTAIN or missing/empty path: no-op.
        """
        if not isinstance(op, dict):
            return

        op_type = op.get("op", "")
        path    = op.get("path", "").strip()
        content = op.get("content", "").strip()

        if op_type not in _WRITE_OPS or not path or not content:
            return

        tagged = f"{content} [s={session_idx} t={turn_idx}]"

        if op_type == "STORE_FACT":
            self.files.setdefault(path, []).append(tagged)
            # Only set tentative if not already confirmed
            if self.importance.get(path) != "confirmed":
                self.importance[path] = "tentative"

        elif op_type == "UPDATE":
            self.files.setdefault(path, []).append(tagged)
            self.importance[path] = "confirmed"

        elif op_type == "SUPERSEDE":
            # Archive prior content under a _prior path for audit
            prior_path = f"{path}_prior"
            if path in self.files:
                self.files[prior_path] = self.files[path]
                self.importance[prior_path] = "superseded"
            self.files[path] = [tagged]
            self.importance[path] = "confirmed"

        elif op_type == "COMPRESS":
            self.files[path] = [tagged]
            self.importance[path] = "confirmed"

    def seed_from_session(self, session: dict, session_idx: int, max_turns: int = 5) -> None:
        """Auto-seed FS with key facts from a session (Harness-1 auto-populate analog).

        Called before the GRPO episode starts so the model has warm prior state
        rather than an empty FS — avoids cold-start reward collapse.
        All seeded entries are tagged tentative (model must confirm or supersede them).

        Only seeds substantive user turns (skips filler, greetings, short turns).
        Capped at max_turns to avoid polluting FS with noise.
        """
        _FILLER = {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay",
                   "sure", "yes", "no", "bye", "goodbye", "lol", "haha", "got it"}

        turns   = session.get("turns", [])
        seeded  = 0

        # Score turns by informativeness: length + proper noun count
        scored = []
        for t_idx, turn in enumerate(turns):
            content = turn.get("content", "").strip()
            role    = turn.get("role", "user")
            if not content or len(content) < 40:
                continue
            words_lower = set(content.lower().split())
            if words_lower & _FILLER and len(content) < 80:
                continue  # skip short filler turns
            # score = content length + 5 per proper noun
            proper_count = sum(1 for w in content.split() if w and w[0].isupper() and len(w) > 2)
            score = len(content) + 5 * proper_count
            scored.append((score, t_idx, content))

        # Take top-max_turns most informative turns
        scored.sort(reverse=True)
        for score, t_idx, content in scored[:max_turns]:
            path   = _infer_path(content)
            tagged = f"{content[:100]} [s={session_idx} t={t_idx} auto]"
            self.files.setdefault(path, []).append(tagged)
            if self.importance.get(path) != "confirmed":
                self.importance[path] = "tentative"
            seeded += 1

    def render_for_prompt(self) -> str:
        """Return a compact text block for prompt injection, ordered by importance."""
        if not self.files:
            return "(memory is empty)"

        # Group paths by importance (confirmed first, superseded last)
        groups: dict[str, list[str]] = {"confirmed": [], "tentative": [], "superseded": []}
        for path in sorted(self.files):
            tag = self.importance.get(path, "tentative")
            groups[tag].append(path)

        lines = []
        for tag in ("confirmed", "tentative", "superseded"):
            paths = groups[tag]
            if not paths:
                continue
            lines.append(f"[{tag.upper()}]")
            for path in paths:
                lines.append(f"  === {path} ===")
                lines.extend(f"  {l}" for l in self.files[path])
        return "\n".join(lines)

    def to_text(self) -> str:
        """Flat text of all content lines — used for QA greping."""
        return "\n".join(
            line for lines in self.files.values() for line in lines
        )


def parse_op(response_str: str) -> dict:
    """Extract the JSON op dict from a model response string.

    Returns {} on any parse failure (treated as ABSTAIN in apply_op).
    """
    cleaned = re.sub(r"<think>.*?</think>", "", response_str, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return {}


def retrieve_for_probe(
    query: str,
    fs: "VirtualFilesystem",
    context1_url: str | None = None,
) -> str:
    """Return FS entries relevant to query.

    Uses Context-1 retrieval service when context1_url is set; falls back to
    keyword grep otherwise (training without the service, or local testing).
    """
    all_lines = [line for lines in fs.files.values() for line in lines]
    if not all_lines:
        return ""

    if context1_url:
        try:
            import urllib.request, json as _json
            payload = _json.dumps({"query": query, "documents": all_lines}).encode()
            req = urllib.request.Request(
                f"{context1_url.rstrip('/')}/retrieve",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read())
            results = data.get("results", [])
            return " ".join(results) if results else _grep_relevant(fs.to_text(), query)
        except Exception:
            pass  # fall through to grep on any error

    return _grep_relevant(fs.to_text(), query)


def score_trajectory(fs: VirtualFilesystem, qa_probes: list, context1_url: str | None = None) -> float:
    """Terminal reward: average token-F1 of retrieved FS content vs each answer.

    Uses Context-1 for retrieval when context1_url is provided, else keyword grep.
    Returns 0.0 if no scoreable probes exist.
    """
    scores = []

    for probe in qa_probes:
        if isinstance(probe, dict):
            question = probe.get("question", "")
            answer   = str(probe.get("answer", ""))
        else:
            question = getattr(probe, "question", "")
            answer   = str(getattr(probe, "answer", ""))

        if not answer or answer in ("ABSTAIN", "Yes", "No", "N/A", ""):
            continue

        relevant = retrieve_for_probe(question, fs, context1_url)
        scores.append(_token_f1(relevant, answer))

    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_path(content: str) -> str:
    """Heuristic: map content to a VFS category/entity path for auto-seeding."""
    lower = content.lower()
    # Extract a clean proper noun (skip sentence-starting words like "Hi", "My", "I")
    _SKIP_WORDS = {"hi", "my", "i", "the", "a", "an", "so", "well", "ok", "oh", "yes", "no",
                   "she", "he", "they", "we", "you", "it", "this", "that"}
    proper = []
    for w in content.split():
        clean = w.strip(".,!?;:'\"").lower()
        if w and w[0].isupper() and len(clean) > 2 and clean not in _SKIP_WORDS:
            proper.append(clean)
    entity = proper[0] if proper else "general"

    if any(k in lower for k in ("born", "birthday", "age", "his name", "her name", "my name",
                                 "she is", "he is", "she was", "he was")):
        return f"people/{entity}"
    if any(k in lower for k in ("meeting", "appointment", "plan", "trip", "visit", "event",
                                 "tomorrow", "next week", "schedule")):
        return "events/general"
    if any(k in lower for k in ("live", "address", "city", "location", "place", "home",
                                 "moved to", "lives in")):
        return f"places/{entity}"
    if any(k in lower for k in ("like", "prefer", "love", "hate", "favorite", "enjoy",
                                 "dislike", "fond of")):
        return f"prefs/{entity}"
    if proper:
        return f"people/{entity}"
    return "facts/general"


def _grep_relevant(fs_text: str, question: str) -> str:
    """Return lines from fs_text whose content overlaps question keywords."""
    q_tokens = set(_normalize(question))
    if not q_tokens or not fs_text.strip():
        return fs_text
    lines    = fs_text.split("\n")
    relevant = [l for l in lines if any(t in l.lower() for t in q_tokens)]
    # Return "" (not full fs_text) when nothing matches — cleaner training signal;
    # falling back to the full FS would give noisy credit for irrelevant content.
    return " ".join(relevant)


def _token_f1(prediction: str, gold: str) -> float:
    pred = _normalize(prediction)
    ref  = _normalize(gold)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    common    = sum((Counter(pred) & Counter(ref)).values())
    precision = common / len(pred)
    recall    = common / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _normalize(text: str) -> list[str]:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t not in _STOP]
