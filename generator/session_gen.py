"""Multi-session trajectory generator for memory_horizon.

The generator is the data pipeline that creates training examples.
Design principles (from Loong/toolathlon patterns):
    - Start from a curated seed set (not fully procedural)
    - Generate multi-session trajectories (minimum 3 sessions for vertical envs)
    - Label QA probes at easy/medium/hard difficulty
    - Write oracle_memory ground truth alongside each trajectory
    - Output JSONL for streaming during training

Usage::

    gen = SessionGenerator.from_config("voice_customer")
    trajectory = gen.generate()
    # Returns a Trajectory with 3+ sessions, QA probes, oracle_memory

    # Batch generation:
    gen.generate_dataset(n=1000, output_path="data/voice_customer.jsonl")

Seed data is in memory_horizon/generator/seeds/<vertical>.jsonl.
Each seed is a minimal conversation outline that the generator expands.
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from memory_horizon.memory_store import MemoryStore
from memory_horizon.types import (
    ConflictExample,
    MemoryOp,
    QAPair,
    Session,
    Trajectory,
    Turn,
)


# ---------------------------------------------------------------------------
# Seed data format
# ---------------------------------------------------------------------------

@dataclass
class SeedExample:
    """One seed conversation that the generator will expand into a Trajectory.

    Fields:
        facts: Key facts that should appear in the conversation.
        conflict_pairs: (old_fact, new_fact, correct_op) triples for Base 3.
        n_sessions: Target number of sessions to generate.
        vertical: Domain label.
    """
    facts: dict[str, str]                          # key → canonical value
    qa_probes: list[dict[str, str]]                # [{question, answer, difficulty}]
    n_sessions: int = 3
    vertical: str = "base"
    conflict_pairs: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in seed banks (enough to bootstrap without external data)
# ---------------------------------------------------------------------------

_VOICE_CUSTOMER_SEEDS: list[SeedExample] = [
    SeedExample(
        facts={
            "customer_name": "Alex Johnson",
            "account_number": "ACC-7821",
            "plan": "Premium Monthly",
            "last_issue": "billing discrepancy on invoice 2024-11",
            "resolution_status": "pending refund of $23.50",
        },
        qa_probes=[
            {"question": "What plan is this customer on?",
             "answer": "Premium Monthly", "difficulty": "easy"},
            {"question": "Has this customer reported a billing issue before?",
             "answer": "Yes, a billing discrepancy on invoice 2024-11",
             "difficulty": "medium"},
            {"question": "What was the resolution status at the end of the last call?",
             "answer": "Pending refund of $23.50", "difficulty": "hard"},
        ],
        n_sessions=3,
        vertical="voice_customer",
    ),
    SeedExample(
        facts={
            "customer_name": "Maria Santos",
            "account_number": "ACC-4403",
            "plan": "Basic Annual",
            "device": "iPhone 14 Pro",
            "reported_issue": "dropped calls in home office area",
            "ticket_id": "TKT-99281",
        },
        qa_probes=[
            {"question": "What device does this customer use?",
             "answer": "iPhone 14 Pro", "difficulty": "easy"},
            {"question": "What is the customer's reported issue?",
             "answer": "dropped calls in home office area", "difficulty": "easy"},
            {"question": "What ticket ID was opened for this customer's issue?",
             "answer": "TKT-99281", "difficulty": "medium"},
        ],
        n_sessions=3,
        vertical="voice_customer",
    ),
]

_BASE_SEEDS: list[SeedExample] = [
    SeedExample(
        facts={
            "user_name": "Jordan",
            "user_role": "senior engineer",
            "user_location": "Berlin",
            "team": "Platform Infrastructure",
            "on_call_rotation": "every third week",
        },
        qa_probes=[
            {"question": "What is the user's name?",
             "answer": "Jordan", "difficulty": "easy"},
            {"question": "What team does the user work on?",
             "answer": "Platform Infrastructure", "difficulty": "easy"},
            {"question": "How often is the user on call?",
             "answer": "every third week", "difficulty": "medium"},
        ],
        n_sessions=2,
        vertical="base",
    ),
    SeedExample(
        facts={
            "project_name": "Helios",
            "deadline": "Q2 2025",
            "budget": "$120,000",
            "status": "at risk — backend delays",
            "lead_engineer": "Priya Nair",
        },
        qa_probes=[
            {"question": "What is the project deadline?",
             "answer": "Q2 2025", "difficulty": "easy"},
            {"question": "What is the current project status?",
             "answer": "at risk — backend delays", "difficulty": "medium"},
            {"question": "Who is the lead engineer and what is the budget?",
             "answer": "Priya Nair, $120,000", "difficulty": "hard"},
        ],
        n_sessions=3,
        vertical="base",
    ),
]

_SEED_BANKS: dict[str, list[SeedExample]] = {
    "base":           _BASE_SEEDS,
    "voice_customer": _VOICE_CUSTOMER_SEEDS,
}


# ---------------------------------------------------------------------------
# Session generator
# ---------------------------------------------------------------------------

class SessionGenerator:
    """Generates multi-session Trajectory objects from seed examples.

    Args:
        seeds: List of SeedExample objects to sample from.
        vertical: Domain label for generated trajectories.
        min_sessions: Minimum sessions per trajectory (plan requires ≥3 for verticals).
        max_turns_per_session: Max conversation turns generated per session.
        rng: Optional random.Random instance for reproducibility.
    """

    def __init__(
        self,
        seeds: list[SeedExample],
        vertical: str = "base",
        min_sessions: int = 2,
        max_turns_per_session: int = 6,
        rng: random.Random | None = None,
    ) -> None:
        if not seeds:
            raise ValueError("seeds list cannot be empty")
        self.seeds = seeds
        self.vertical = vertical
        self.min_sessions = min_sessions
        self.max_turns_per_session = max_turns_per_session
        self.rng = rng or random.Random()

    @classmethod
    def from_config(cls, vertical: str, **kwargs: Any) -> "SessionGenerator":
        """Load generator from built-in seed bank for a given vertical."""
        seeds = _SEED_BANKS.get(vertical) or _SEED_BANKS["base"]
        return cls(seeds=seeds, vertical=vertical, **kwargs)

    @classmethod
    def from_jsonl(cls, path: str | Path, vertical: str, **kwargs: Any) -> "SessionGenerator":
        """Load seeds from a JSONL file (one SeedExample per line)."""
        seeds: list[SeedExample] = []
        path = Path(path)
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                seeds.append(SeedExample(
                    facts=d.get("facts", {}),
                    qa_probes=d.get("qa_probes", []),
                    n_sessions=d.get("n_sessions", 3),
                    vertical=d.get("vertical", vertical),
                    conflict_pairs=d.get("conflict_pairs", []),
                    metadata=d.get("metadata", {}),
                ))
        return cls(seeds=seeds, vertical=vertical, **kwargs)

    def generate(self) -> Trajectory:
        """Generate one Trajectory from a randomly sampled seed."""
        seed = self.rng.choice(self.seeds)
        return self._expand_seed(seed)

    def stream(self) -> Iterator[Trajectory]:
        """Infinite stream of trajectories (random sample with replacement)."""
        while True:
            yield self.generate()

    def generate_dataset(
        self,
        n: int,
        output_path: str | Path,
        shuffle: bool = True,
    ) -> None:
        """Generate n trajectories and write as JSONL to output_path."""
        trajectories = [self.generate() for _ in range(n)]
        if shuffle:
            self.rng.shuffle(trajectories)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            for traj in trajectories:
                f.write(json.dumps(traj.to_dict()) + "\n")

    # -----------------------------------------------------------------------
    # Expansion logic
    # -----------------------------------------------------------------------

    def _expand_seed(self, seed: SeedExample) -> Trajectory:
        trajectory_id = str(uuid.uuid4())
        n_sessions = max(self.min_sessions, seed.n_sessions)

        # Determine which facts appear in which session.
        fact_items = list(seed.facts.items())
        self.rng.shuffle(fact_items)

        # Distribute facts roughly evenly across sessions, with some overlap.
        facts_per_session = _distribute_facts(fact_items, n_sessions, self.rng)

        sessions: list[Session] = []
        oracle_store = MemoryStore(persist_across_sessions=True)

        for i, session_facts in enumerate(facts_per_session):
            session_id = f"session_{i + 1}"
            oracle_store.session_boundary(session_id)

            turns = self._generate_turns(session_facts, session_id, oracle_store)
            snapshot = oracle_store.snapshot()

            sessions.append(Session(
                session_id=session_id,
                turns=turns,
                memory_snapshot=snapshot,
                metadata={"vertical": self.vertical, "seed_facts": dict(session_facts)},
            ))

        # Build oracle memory from final store state.
        oracle_memory: dict[str, Any] = {
            k: v.content for k, v in oracle_store.get_all_active_facts().items()
        }

        # Build QA probes from seed.
        qa_probes = [
            QAPair(
                question=qa["question"],
                answer=qa["answer"],
                difficulty=qa.get("difficulty", "easy"),
                requires_sessions=qa.get("requires_sessions", []),
                answer_type=qa.get("answer_type", "free_form"),
            )
            for qa in seed.qa_probes
        ]

        # Build conflict examples for Base 3.
        conflict_examples: list[ConflictExample] = []
        for cp in seed.conflict_pairs:
            try:
                conflict_examples.append(ConflictExample(
                    old_memory=cp["old_memory"],
                    new_claim=cp["new_claim"],
                    correct_op=MemoryOp(cp["correct_op"]),
                    why=cp.get("why", ""),
                    conflict_type=cp.get("conflict_type", "value_updated"),
                ))
            except (KeyError, ValueError):
                pass

        return Trajectory(
            trajectory_id=trajectory_id,
            sessions=sessions,
            qa_probes=qa_probes,
            oracle_memory=oracle_memory,
            vertical=self.vertical,
            conflict_examples=conflict_examples,
            metadata={"seed_vertical": seed.vertical, "generator": "SessionGenerator"},
        )

    def _generate_turns(
        self,
        session_facts: list[tuple[str, str]],
        session_id: str,
        oracle_store: MemoryStore,
    ) -> list[Turn]:
        """Generate realistic-ish conversation turns that surface the facts.

        Each fact gets revealed in a user turn. We add filler user/assistant turns
        around it to simulate realistic conversation flow.
        """
        turns: list[Turn] = []
        turn_idx = 0

        # Opening user turn.
        turns.append(Turn(
            role="user",
            content=_opening_line(session_id),
            metadata={"turn_type": "opening"},
        ))
        turn_idx += 1

        for key, value in session_facts:
            # User turn that reveals the fact.
            user_text = _fact_to_user_turn(key, value)
            turns.append(Turn(role="user", content=user_text))
            turn_idx += 1

            # Assistant acknowledgment.
            turns.append(Turn(
                role="assistant",
                content=f"Got it. I've noted that {value}.",
            ))
            turn_idx += 1

            # Simulate the oracle storing this fact.
            from memory_horizon.types import MemoryOpAction, MemoryLayer
            oracle_store.execute(MemoryOpAction(
                op=MemoryOp.STORE_FACT,
                content=value,
                key=_key_from_fact(key),
                confidence=1.0,
                layer=MemoryLayer.FACT,
                metadata={"session_id": session_id},
            ))

            if turn_idx >= self.max_turns_per_session * 2:
                break

        # Closing turn.
        turns.append(Turn(
            role="user",
            content="Thanks, that's all for now.",
            metadata={"turn_type": "closing"},
        ))

        return turns


# ---------------------------------------------------------------------------
# JSONL dataset loader (for training)
# ---------------------------------------------------------------------------

class TrajectoryDataset:
    """Loads pre-generated trajectories from a JSONL file.

    Provides a callable that returns one Trajectory per call (for use as
    trajectory_fn in MemoryHorizonEnv).

    Usage::

        dataset = TrajectoryDataset("data/voice_customer.jsonl", shuffle=True)
        env = MemoryHorizonEnv(env_type="store", trajectory_fn=dataset.next)
    """

    def __init__(self, path: str | Path, shuffle: bool = True, seed: int = 42) -> None:
        self.path = Path(path)
        self._rng = random.Random(seed)
        self._data: list[dict[str, Any]] = []
        self._idx = 0
        self._load(shuffle)

    def _load(self, shuffle: bool) -> None:
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    self._data.append(json.loads(line))
        if shuffle:
            self._rng.shuffle(self._data)

    def next(self) -> Trajectory:
        """Return next trajectory (wraps around on exhaustion)."""
        d = self._data[self._idx % len(self._data)]
        self._idx += 1
        return _dict_to_trajectory(d)

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _distribute_facts(
    facts: list[tuple[str, str]],
    n_sessions: int,
    rng: random.Random,
) -> list[list[tuple[str, str]]]:
    """Assign facts to sessions. Each fact appears in at least one session.
    Some facts appear in multiple sessions (for multi-session QA difficulty)."""
    per_session: list[list[tuple[str, str]]] = [[] for _ in range(n_sessions)]

    for i, fact in enumerate(facts):
        primary = i % n_sessions
        per_session[primary].append(fact)
        # 30% chance the fact also appears in the next session (cross-session test).
        if n_sessions > 1 and rng.random() < 0.3:
            next_session = (primary + 1) % n_sessions
            per_session[next_session].append(fact)

    return per_session


def _fact_to_user_turn(key: str, value: str) -> str:
    """Convert a fact key/value to a natural user turn."""
    readable_key = key.replace("_", " ")
    templates = [
        f"My {readable_key} is {value}.",
        f"Just so you know, {readable_key}: {value}.",
        f"Please note my {readable_key} — it's {value}.",
        f"For your records: {readable_key} is {value}.",
    ]
    return random.choice(templates)


def _opening_line(session_id: str) -> str:
    session_num = session_id.split("_")[-1]
    if session_num == "1":
        return "Hi, I'm calling to go over a few things."
    return "Hi again, I wanted to follow up from our last conversation."


_KEY_RE_CLEAN = __import__("re").compile(r"[^a-z0-9_]")

def _key_from_fact(key: str) -> str:
    k = key.lower().replace(" ", "_")
    return _KEY_RE_CLEAN.sub("", k) or "fact"


def _dict_to_trajectory(d: dict[str, Any]) -> Trajectory:
    """Reconstruct a Trajectory from its to_dict() representation."""
    sessions = []
    for sd in d.get("sessions", []):
        turns = [
            Turn(
                role=t["role"],
                content=t["content"],
                timestamp=t.get("timestamp"),
                metadata=t.get("metadata", {}),
            )
            for t in sd.get("turns", [])
        ]
        sessions.append(Session(
            session_id=sd["session_id"],
            turns=turns,
            memory_snapshot=sd.get("memory_snapshot"),
            metadata=sd.get("metadata", {}),
        ))

    qa_probes = [
        QAPair(
            question=q["question"],
            answer=q["answer"],
            difficulty=q.get("difficulty", "easy"),
            requires_sessions=q.get("requires_sessions", []),
            answer_type=q.get("answer_type", "free_form"),
            choices=q.get("choices", []),
            answer_idx=q.get("answer_idx"),
        )
        for q in d.get("qa_probes", [])
    ]

    return Trajectory(
        trajectory_id=d.get("trajectory_id", str(uuid.uuid4())),
        sessions=sessions,
        qa_probes=qa_probes,
        oracle_memory=d.get("oracle_memory", {}),
        vertical=d.get("vertical", "base"),
        metadata=d.get("metadata", {}),
    )
