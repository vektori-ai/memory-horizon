"""Conflict scenario generator for Base 3 (Contradict) environment.

Produces labeled ConflictExample objects with one of four conflict patterns:

    value_updated      -> correct op is UPDATE   (fact changed, old was accurate then)
    factual_correction -> correct op is SUPERSEDE (old was simply wrong)
    temporal_shift     -> correct op is DECAY     (old was true then, new is true now)
    complementary_view -> correct op is KEEP_BOTH (both valid from different angles)

Usage::

    gen = ConflictGenerator()
    example = gen.generate()   # one ConflictExample
    batch = gen.generate_batch(n=200)

    gen.generate_dataset(n=1000, output_path="data/conflicts.jsonl")
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from memory_horizon.types import ConflictExample, MemoryOp


# ---------------------------------------------------------------------------
# Conflict type → correct op mapping (canonical)
# ---------------------------------------------------------------------------

_CONFLICT_TYPE_TO_OP: dict[str, MemoryOp] = {
    "value_updated":       MemoryOp.UPDATE,
    "factual_correction":  MemoryOp.SUPERSEDE,
    "temporal_shift":      MemoryOp.DECAY,
    "complementary_view":  MemoryOp.KEEP_BOTH,
}

ConflictType = Literal[
    "value_updated",
    "factual_correction",
    "temporal_shift",
    "complementary_view",
]


# ---------------------------------------------------------------------------
# Seed templates — one per conflict pattern
# ---------------------------------------------------------------------------

@dataclass
class ConflictSeed:
    key: str
    old_content: str
    new_claim: str
    why: str
    conflict_type: ConflictType


_SEEDS: list[ConflictSeed] = [
    # ---------- value_updated → UPDATE ----------
    ConflictSeed(
        key="customer_plan",
        old_content="Customer is on the Premium Monthly plan",
        new_claim="The customer just upgraded to Premium Annual",
        why="Plan changed; old entry is now stale",
        conflict_type="value_updated",
    ),
    ConflictSeed(
        key="user_location",
        old_content="User lives in Berlin",
        new_claim="User relocated to Amsterdam last month",
        why="Location changed; old city is outdated",
        conflict_type="value_updated",
    ),
    ConflictSeed(
        key="ticket_status",
        old_content="Ticket TKT-9921 status: open",
        new_claim="Ticket TKT-9921 has been resolved",
        why="Status progressed; old status no longer current",
        conflict_type="value_updated",
    ),
    ConflictSeed(
        key="account_balance",
        old_content="Account balance is $450.00",
        new_claim="Customer reports their balance is now $320.00 after last payment",
        why="Balance changed due to transaction; overwrite with current value",
        conflict_type="value_updated",
    ),
    ConflictSeed(
        key="lead_engineer",
        old_content="Lead engineer on Project Helios is Priya Nair",
        new_claim="Priya handed off to Sam Chen who is now the lead",
        why="Personnel change; old name is incorrect going forward",
        conflict_type="value_updated",
    ),

    # ---------- factual_correction → SUPERSEDE ----------
    ConflictSeed(
        key="account_number",
        old_content="Customer account number is ACC-7821",
        new_claim="System confirms the correct account number is ACC-7812 — previous was transposed",
        why="Data entry error; old value was factually wrong from the start",
        conflict_type="factual_correction",
    ),
    ConflictSeed(
        key="device_model",
        old_content="Customer uses an iPhone 13 Pro",
        new_claim="Customer clarifies they have an iPhone 14 Pro, not 13",
        why="Agent misheard; old entry was never correct",
        conflict_type="factual_correction",
    ),
    ConflictSeed(
        key="contract_end_date",
        old_content="Contract ends on 2025-03-31",
        new_claim="Contracts team confirms the end date is 2025-06-30 — previous date was wrong",
        why="Source-of-truth correction; old date was inaccurate",
        conflict_type="factual_correction",
    ),
    ConflictSeed(
        key="budget",
        old_content="Project budget is $80,000",
        new_claim="Finance corrects the budget to $120,000 — $80k was an early draft figure",
        why="Approved budget differs from draft; archive the draft value",
        conflict_type="factual_correction",
    ),

    # ---------- temporal_shift → DECAY ----------
    ConflictSeed(
        key="on_call_status",
        old_content="Jordan is currently on call (week of 2025-01-06)",
        new_claim="The on-call rotation has moved on; Jordan may or may not be on call now",
        why="Time-bound fact; confidence should decay, not be deleted",
        conflict_type="temporal_shift",
    ),
    ConflictSeed(
        key="project_status",
        old_content="Project Helios status: on track (as of Q1 2025)",
        new_claim="No status update received since Q1; situation may have changed",
        why="Stale status without contradicting info; reduce confidence rather than overwrite",
        conflict_type="temporal_shift",
    ),
    ConflictSeed(
        key="issue_priority",
        old_content="Bug BUG-441 is marked P1 critical",
        new_claim="Three weeks have passed; P1 classification may have been downgraded",
        why="Priority classifications drift over time without explicit update",
        conflict_type="temporal_shift",
    ),

    # ---------- complementary_view → KEEP_BOTH ----------
    ConflictSeed(
        key="preferred_contact",
        old_content="Customer prefers contact by email for billing questions",
        new_claim="Customer prefers phone calls for urgent issues",
        why="Both preferences are valid — different channels for different contexts",
        conflict_type="complementary_view",
    ),
    ConflictSeed(
        key="team_assessment",
        old_content="Engineering team rates the API performance as acceptable",
        new_claim="Product team rates the same API performance as inadequate for their use case",
        why="Different stakeholders hold valid but contradictory views simultaneously",
        conflict_type="complementary_view",
    ),
    ConflictSeed(
        key="resolution_opinion",
        old_content="Agent 1 notes: customer seemed satisfied at call end",
        new_claim="Agent 2 notes: customer followed up unsatisfied 2 days later",
        why="Two agents saw different moments in time; both observations are accurate records",
        conflict_type="complementary_view",
    ),
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class ConflictGenerator:
    """Generates labeled ConflictExample objects for Base 3 training.

    Args:
        seeds: Conflict seed templates. Defaults to built-in set.
        weights: Probability weights for each conflict type (uniform by default).
        rng: Optional random.Random for reproducibility.
    """

    def __init__(
        self,
        seeds: list[ConflictSeed] | None = None,
        weights: dict[ConflictType, float] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._seeds = seeds if seeds is not None else list(_SEEDS)
        self._rng = rng or random.Random()
        self._weights = weights or {
            "value_updated":       0.35,
            "factual_correction":  0.25,
            "temporal_shift":      0.20,
            "complementary_view":  0.20,
        }
        self._seeds_by_type: dict[str, list[ConflictSeed]] = {}
        for s in self._seeds:
            self._seeds_by_type.setdefault(s.conflict_type, []).append(s)

    def generate(self) -> ConflictExample:
        """Sample one ConflictExample from the seed bank."""
        conflict_type = self._rng.choices(
            list(self._weights.keys()),
            weights=list(self._weights.values()),
            k=1,
        )[0]
        bucket = self._seeds_by_type.get(conflict_type)
        if not bucket:
            bucket = self._seeds
        seed = self._rng.choice(bucket)
        return self._from_seed(seed)

    def generate_batch(self, n: int) -> list[ConflictExample]:
        return [self.generate() for _ in range(n)]

    def generate_dataset(
        self,
        n: int,
        output_path: str | Path,
        shuffle: bool = True,
    ) -> None:
        """Write n ConflictExamples to a JSONL file."""
        examples = self.generate_batch(n)
        if shuffle:
            self._rng.shuffle(examples)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict()) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path, **kwargs: Any) -> "ConflictGenerator":
        """Load seeds from a JSONL file (one ConflictSeed per line)."""
        seeds: list[ConflictSeed] = []
        with Path(path).open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                seeds.append(ConflictSeed(
                    key=d["key"],
                    old_content=d["old_content"],
                    new_claim=d["new_claim"],
                    why=d["why"],
                    conflict_type=d["conflict_type"],
                ))
        return cls(seeds=seeds, **kwargs)

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    @staticmethod
    def _from_seed(seed: ConflictSeed) -> ConflictExample:
        return ConflictExample(
            old_memory={"key": seed.key, "content": seed.old_content},
            new_claim=seed.new_claim,
            correct_op=_CONFLICT_TYPE_TO_OP[seed.conflict_type],
            why=seed.why,
            conflict_type=seed.conflict_type,
        )
