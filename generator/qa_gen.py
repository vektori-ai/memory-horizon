"""QA probe generator for memory_horizon training trajectories.

Programmatically creates QAPair objects at easy / medium / hard difficulty
across three answer types: free_form, boolean, multiple_choice.

Design principles:
- Easy:   single fact, single session — tests raw retrieval
- Medium: requires aggregating or paraphrasing across one session
- Hard:   multi-session synthesis or negation / absent-fact handling

Usage::

    gen = QAGenerator()
    probes = gen.generate_for_facts(
        facts={"customer_plan": "Premium Monthly", "ticket_id": "TKT-99281"},
        n=6,
    )

    # Or generate a fixed probe list from a SeedExample dict:
    probes = gen.generate_from_seed(seed_dict)
"""

from __future__ import annotations

import random
from typing import Any

from memory_horizon.mh_types import QAPair


# ---------------------------------------------------------------------------
# Template banks
# ---------------------------------------------------------------------------

# Each template is (question_template, answer_factory, difficulty, answer_type)
# question_template uses {key}, {readable_key}, {value}
# answer_factory is a callable(value) -> str

_FREE_FORM_EASY: list[tuple[str, str]] = [
    ("What is the {readable_key}?",            "{value}"),
    ("What {readable_key} is on record?",      "{value}"),
    ("Tell me the {readable_key}.",            "{value}"),
]

_FREE_FORM_MEDIUM: list[tuple[str, str]] = [
    ("Has a {readable_key} been mentioned in any session?",
     "Yes, {value}"),
    ("Summarize what you know about {readable_key}.",
     "The {readable_key} is {value}"),
]

_FREE_FORM_HARD: list[tuple[str, str]] = [
    ("Based on all sessions, what is the most recent {readable_key}?",
     "{value}"),
    ("Across all conversations, what can you infer about {readable_key}?",
     "From stored memory: {readable_key} is {value}"),
]

_BOOLEAN_TEMPLATES: list[tuple[str, str, str]] = [
    # (question, yes_answer, no_answer)
    ("Is there a recorded {readable_key}?",  "Yes", "No"),
    ("Do we have information on {readable_key}?", "Yes", "No"),
]

_MC_DISTRACTORS: list[str] = [
    "Unknown — not in memory",
    "Pending confirmation",
    "N/A",
    "Not recorded",
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class QAGenerator:
    """Generates QAPair objects from a fact dictionary.

    Args:
        rng: Optional random.Random for reproducibility.
        distractor_pool: Additional wrong-answer strings for multiple-choice.
    """

    def __init__(
        self,
        rng: random.Random | None = None,
        distractor_pool: list[str] | None = None,
    ) -> None:
        self._rng = rng or random.Random()
        self._distractors = list(distractor_pool or _MC_DISTRACTORS)

    def generate_for_facts(
        self,
        facts: dict[str, str],
        n: int = 4,
        session_ids: list[str] | None = None,
    ) -> list[QAPair]:
        """Generate up to n QA probes covering the given facts.

        Distributes difficulty roughly: 40% easy, 35% medium, 25% hard.
        """
        fact_items = list(facts.items())
        self._rng.shuffle(fact_items)

        probes: list[QAPair] = []
        difficulty_cycle = self._difficulty_schedule(n)

        for i, (key, value) in enumerate(fact_items):
            if len(probes) >= n:
                break
            difficulty = difficulty_cycle[i % len(difficulty_cycle)]
            answer_type = self._pick_answer_type(difficulty)
            probe = self._make_probe(
                key=key,
                value=value,
                difficulty=difficulty,
                answer_type=answer_type,
                session_ids=session_ids or [],
            )
            probes.append(probe)

        return probes[:n]

    def generate_cross_session(
        self,
        facts_by_session: dict[str, dict[str, str]],
        n: int = 3,
    ) -> list[QAPair]:
        """Generate hard multi-session probes that require synthesising across sessions.

        Args:
            facts_by_session: {session_id: {key: value}}
        """
        probes: list[QAPair] = []
        all_session_ids = list(facts_by_session.keys())

        for session_id, facts in facts_by_session.items():
            for key, value in facts.items():
                if len(probes) >= n:
                    return probes
                requires = [s for s in all_session_ids if s != session_id]
                requires = self._rng.sample(requires, k=min(1, len(requires)))
                q = f"Across all sessions, what is the {key.replace('_', ' ')}?"
                probes.append(QAPair(
                    question=q,
                    answer=value,
                    difficulty="hard",
                    requires_sessions=[session_id] + requires,
                    answer_type="free_form",
                ))

        return probes[:n]

    def generate_boolean(self, facts: dict[str, str], n: int = 2) -> list[QAPair]:
        """Generate boolean (yes/no) probes — always easy."""
        probes: list[QAPair] = []
        fact_items = list(facts.items())
        self._rng.shuffle(fact_items)

        for key, value in fact_items:
            if len(probes) >= n:
                break
            tmpl_q, tmpl_yes, _ = self._rng.choice(_BOOLEAN_TEMPLATES)
            readable_key = key.replace("_", " ")
            question = tmpl_q.format(readable_key=readable_key, key=key, value=value)
            probes.append(QAPair(
                question=question,
                answer=tmpl_yes,
                difficulty="easy",
                answer_type="boolean",
            ))

        return probes

    def generate_multiple_choice(
        self,
        facts: dict[str, str],
        n: int = 2,
        n_choices: int = 4,
    ) -> list[QAPair]:
        """Generate multiple-choice probes with one correct answer."""
        probes: list[QAPair] = []
        fact_items = list(facts.items())
        self._rng.shuffle(fact_items)

        for key, value in fact_items:
            if len(probes) >= n:
                break
            readable_key = key.replace("_", " ")
            question = f"What is the {readable_key}?"

            wrong = self._rng.sample(self._distractors, k=min(n_choices - 1, len(self._distractors)))
            choices = wrong + [value]
            self._rng.shuffle(choices)
            answer_idx = choices.index(value)

            probes.append(QAPair(
                question=question,
                answer=value,
                difficulty="medium",
                answer_type="multiple_choice",
                choices=choices,
                answer_idx=answer_idx,
            ))

        return probes

    def generate_from_seed(self, seed_dict: dict[str, Any]) -> list[QAPair]:
        """Build QAPairs from a raw seed dict (as used in SeedExample.qa_probes)."""
        probes = []
        for qa in seed_dict.get("qa_probes", []):
            probes.append(QAPair(
                question=qa["question"],
                answer=qa["answer"],
                difficulty=qa.get("difficulty", "easy"),
                requires_sessions=qa.get("requires_sessions", []),
                answer_type=qa.get("answer_type", "free_form"),
                choices=qa.get("choices", []),
                answer_idx=qa.get("answer_idx"),
            ))
        return probes

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _difficulty_schedule(self, n: int) -> list[str]:
        easy   = max(1, round(n * 0.40))
        medium = max(1, round(n * 0.35))
        hard   = max(0, n - easy - medium)
        schedule = ["easy"] * easy + ["medium"] * medium + ["hard"] * hard
        self._rng.shuffle(schedule)
        return schedule

    def _pick_answer_type(self, difficulty: str) -> str:
        if difficulty == "easy":
            return self._rng.choice(["free_form", "boolean"])
        if difficulty == "medium":
            return self._rng.choice(["free_form", "multiple_choice"])
        return "free_form"

    def _make_probe(
        self,
        key: str,
        value: str,
        difficulty: str,
        answer_type: str,
        session_ids: list[str],
    ) -> QAPair:
        readable_key = key.replace("_", " ")
        fmt = {"key": key, "readable_key": readable_key, "value": value}

        if answer_type == "boolean":
            tmpl_q, tmpl_yes, _ = self._rng.choice(_BOOLEAN_TEMPLATES)
            question = tmpl_q.format(**fmt)
            answer = tmpl_yes
            return QAPair(
                question=question,
                answer=answer,
                difficulty=difficulty,
                answer_type="boolean",
                requires_sessions=session_ids,
            )

        if answer_type == "multiple_choice":
            question = f"What is the {readable_key}?"
            wrong = self._rng.sample(
                self._distractors, k=min(3, len(self._distractors))
            )
            choices = wrong + [value]
            self._rng.shuffle(choices)
            answer_idx = choices.index(value)
            return QAPair(
                question=question,
                answer=value,
                difficulty=difficulty,
                answer_type="multiple_choice",
                choices=choices,
                answer_idx=answer_idx,
                requires_sessions=session_ids,
            )

        # free_form
        if difficulty == "easy":
            pool = _FREE_FORM_EASY
        elif difficulty == "medium":
            pool = _FREE_FORM_MEDIUM
        else:
            pool = _FREE_FORM_HARD

        tmpl_q, tmpl_a = self._rng.choice(pool)
        question = tmpl_q.format(**fmt)
        answer = tmpl_a.format(**fmt)
        return QAPair(
            question=question,
            answer=answer,
            difficulty=difficulty,
            answer_type="free_form",
            requires_sessions=session_ids,
        )
