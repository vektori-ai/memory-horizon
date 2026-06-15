"""VirtualFilesystem — append-only memory store for the memory agent.

Each path (category/entity) maps to an ordered list of tagged content lines.
The model reads render_for_prompt() at each turn to see its current memory state
and writes ops (STORE_FACT / UPDATE / SUPERSEDE / COMPRESS / ABSTAIN) to update it.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import dataclass, field

_STOP = {"what", "is", "the", "are", "was", "were", "a", "an", "this", "that",
         "of", "for", "in", "on", "at", "to", "does", "did", "has", "have", "had"}

_WRITE_OPS = {"STORE_FACT", "UPDATE", "SUPERSEDE", "COMPRESS"}


@dataclass
class VirtualFilesystem:
    files: dict[str, list[str]] = field(default_factory=dict)

    def apply_op(self, op: dict, session_idx: int, turn_idx: int) -> None:
        """Apply a parsed memory op to the filesystem.

        STORE_FACT / UPDATE: append a new tagged line to path.
        SUPERSEDE / COMPRESS: clear all prior lines for path, then append.
        ABSTAIN or missing/empty path: no-op.
        Parse failure (non-dict op): no-op.
        """
        if not isinstance(op, dict):
            return

        op_type = op.get("op", "")
        path    = op.get("path", "").strip()
        content = op.get("content", "").strip()

        if op_type not in _WRITE_OPS or not path or not content:
            return

        tagged = f"{content} [s={session_idx} t={turn_idx}]"

        if op_type in ("UPDATE", "SUPERSEDE", "COMPRESS"):
            self.files[path] = [tagged]
        else:
            self.files.setdefault(path, []).append(tagged)

    def render_for_prompt(self) -> str:
        """Return a compact, grep-friendly text block for prompt injection."""
        if not self.files:
            return "(memory is empty)"
        lines = []
        for path in sorted(self.files):
            lines.append(f"=== {path} ===")
            lines.extend(self.files[path])
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
