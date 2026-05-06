"""Curriculum scheduler for memory_horizon training.

Implements the Batch A → B → Integration → Vertical sequencing from the plan,
with kill criteria enforcement and GEPA fallback.

Batch A (Weeks 1–6):  Base 6 (Temporal), Base 3 (Contradict), Base 4 (Compress)
Batch B (Weeks 7–9):  Base 1 (Store), Base 2 (Synthesize), Base 5 (Abstain)
Week 10:              Base 7 Integration Gate
Month 3+:             Vertical envs (V1 → V5+)

Kill criteria (from plan):
    Base 1: 8 runs without convergence
    Base 2: 8 runs without convergence
    Base 3: Always-UPDATE reward hack detected → pause
    Base 4: No-compression collapse → pause
    Base 5: Hallucination rate > 10% after 8 runs
    Base 6: Held-out eval < 30% after 8 runs → pause; switch to GEPA
    Base 7: Within 10pp of oracle → proceed to V1; gap > 15pp → iterate weakest

GEPA fallback: prompt-level evolutionary optimization, no weight updates.
Not a failure — a real Plan B when RL fails to converge.

Usage::

    scheduler = CurriculumScheduler()
    current_stage = scheduler.current_stage()
    # → CurriculumStage(name="batch_a", envs=["temporal", "contradict", "compress"])

    # After a training run:
    scheduler.record_run(env="temporal", eval_score=0.45, hallucination_rate=0.02)
    should_continue = scheduler.should_advance("temporal")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

class StagePhase(str, Enum):
    BATCH_A      = "batch_a"       # Base 6, 3, 4
    BATCH_B      = "batch_b"       # Base 1, 2, 5
    INTEGRATION  = "integration"   # Base 7
    VERTICAL_V1  = "vertical_v1"   # voice_customer, voice_debt
    VERTICAL_V2  = "vertical_v2"
    VERTICAL_V3  = "vertical_v3"
    VERTICAL_V4  = "vertical_v4"
    VERTICAL_V5  = "vertical_v5"


@dataclass
class CurriculumStage:
    phase: StagePhase
    envs: list[str]
    advance_threshold: float       # eval score needed to advance
    kill_threshold: float          # score below which kill criteria fires
    max_runs_before_kill: int      # number of consecutive runs before kill criteria
    notes: str = ""


STAGES: list[CurriculumStage] = [
    CurriculumStage(
        phase=StagePhase.BATCH_A,
        envs=["temporal", "contradict", "compress"],
        advance_threshold=0.70,
        kill_threshold=0.30,
        max_runs_before_kill=8,
        notes="Hardest reward functions. Start here because debug-intensive.",
    ),
    CurriculumStage(
        phase=StagePhase.BATCH_B,
        envs=["store", "synthesize", "abstain"],
        advance_threshold=0.70,
        kill_threshold=0.30,
        max_runs_before_kill=8,
        notes="Cleaner rewards. Benefits from Batch A infrastructure.",
    ),
    CurriculumStage(
        phase=StagePhase.INTEGRATION,
        envs=["integration"],
        advance_threshold=0.90,    # within 10pp of oracle (+1.0)
        kill_threshold=0.70,       # gap > 15pp → iterate weakest base env
        max_runs_before_kill=5,
        notes="Base 7 gate. Composite of all 6 skills.",
    ),
    CurriculumStage(
        phase=StagePhase.VERTICAL_V1,
        envs=["voice_customer", "voice_debt"],
        advance_threshold=0.75,
        kill_threshold=0.40,
        max_runs_before_kill=8,
        notes="V1: warmest BD pipeline, sub-500ms latency validates SLM thesis.",
    ),
    CurriculumStage(
        phase=StagePhase.VERTICAL_V2,
        envs=["support_technical", "support_saas"],
        advance_threshold=0.75,
        kill_threshold=0.40,
        max_runs_before_kill=8,
    ),
    CurriculumStage(
        phase=StagePhase.VERTICAL_V3,
        envs=["sales_saas", "sales_real_estate"],
        advance_threshold=0.75,
        kill_threshold=0.40,
        max_runs_before_kill=8,
    ),
    CurriculumStage(
        phase=StagePhase.VERTICAL_V4,
        envs=["coding_debugging", "coding_codebase"],
        advance_threshold=0.75,
        kill_threshold=0.40,
        max_runs_before_kill=8,
    ),
]


# ---------------------------------------------------------------------------
# Per-env kill criteria (env-specific rules beyond the generic threshold)
# ---------------------------------------------------------------------------

@dataclass
class EnvRunRecord:
    env: str
    run_id: int
    eval_score: float
    hallucination_rate: float = 0.0
    compression_ratio: float | None = None   # Base 4: detect no-compression collapse
    dominant_op: str | None = None           # Base 3: detect always-UPDATE hack
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KillDecision:
    should_kill: bool
    reason: str
    fallback: str = "gepa"   # "gepa" or "iterate_base_env"


# ---------------------------------------------------------------------------
# CurriculumScheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """Tracks training progress and decides when to advance or kill stages.

    Args:
        state_path: Optional path to persist scheduler state as JSON.
                    Useful for resuming after crashes.
    """

    def __init__(self, state_path: str | Path | None = None) -> None:
        self._stage_idx: int = 0
        self._run_records: dict[str, list[EnvRunRecord]] = {}
        self._gepa_active: set[str] = set()
        self._state_path = Path(state_path) if state_path else None

        if self._state_path and self._state_path.exists():
            self._load_state()

    def current_stage(self) -> CurriculumStage:
        return STAGES[min(self._stage_idx, len(STAGES) - 1)]

    def current_phase(self) -> StagePhase:
        return self.current_stage().phase

    def record_run(
        self,
        env: str,
        eval_score: float,
        hallucination_rate: float = 0.0,
        compression_ratio: float | None = None,
        dominant_op: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KillDecision:
        """Record an evaluation run and check kill criteria.

        Returns:
            KillDecision indicating whether to kill this env and what fallback to use.
        """
        run_id = len(self._run_records.get(env, [])) + 1
        record = EnvRunRecord(
            env=env,
            run_id=run_id,
            eval_score=eval_score,
            hallucination_rate=hallucination_rate,
            compression_ratio=compression_ratio,
            dominant_op=dominant_op,
            metadata=metadata or {},
        )
        self._run_records.setdefault(env, []).append(record)

        decision = self._check_kill_criteria(env, record)
        if self._state_path:
            self._save_state()
        return decision

    def should_advance(self, env: str) -> bool:
        """Return True if this env has met the advance threshold."""
        stage = self.current_stage()
        if env not in stage.envs:
            return False
        records = self._run_records.get(env, [])
        if not records:
            return False
        recent = records[-1].eval_score
        return recent >= stage.advance_threshold

    def try_advance_stage(self) -> bool:
        """Advance to next stage if ALL envs in current stage meet threshold.

        Returns True if advancement happened.
        """
        stage = self.current_stage()
        all_ready = all(self.should_advance(env) for env in stage.envs)

        if all_ready and self._stage_idx < len(STAGES) - 1:
            self._stage_idx += 1
            if self._state_path:
                self._save_state()
            return True
        return False

    def is_gepa_active(self, env: str) -> bool:
        return env in self._gepa_active

    def activate_gepa(self, env: str) -> None:
        self._gepa_active.add(env)

    def status(self) -> dict[str, Any]:
        stage = self.current_stage()
        env_status: dict[str, Any] = {}
        for env in stage.envs:
            records = self._run_records.get(env, [])
            env_status[env] = {
                "n_runs": len(records),
                "latest_score": records[-1].eval_score if records else None,
                "threshold": stage.advance_threshold,
                "ready": self.should_advance(env),
                "gepa_active": self.is_gepa_active(env),
            }
        return {
            "current_phase": stage.phase.value,
            "stage_idx": self._stage_idx,
            "envs": env_status,
        }

    # -----------------------------------------------------------------------
    # Kill criteria per env
    # -----------------------------------------------------------------------

    def _check_kill_criteria(self, env: str, latest: EnvRunRecord) -> KillDecision:
        stage = self.current_stage()
        records = self._run_records.get(env, [])

        # --- Base 5 (Abstain): hallucination rate > 10% after 8 runs ---
        if env == "abstain":
            if len(records) >= 8 and latest.hallucination_rate > 0.10:
                return KillDecision(
                    should_kill=True,
                    reason=f"Base 5 kill: hallucination rate {latest.hallucination_rate:.1%} > 10% after {len(records)} runs",
                    fallback="gepa",
                )

        # --- Base 4 (Compress): no-compression collapse ---
        if env == "compress":
            if latest.compression_ratio is not None and latest.compression_ratio > 0.95:
                return KillDecision(
                    should_kill=True,
                    reason=f"Base 4 kill: compression collapse (ratio={latest.compression_ratio:.2f} > 0.95)",
                    fallback="gepa",
                )

        # --- Base 3 (Contradict): always-UPDATE reward hack ---
        if env == "contradict":
            if latest.dominant_op == "UPDATE":
                recent_ops = [r.dominant_op for r in records[-3:]]
                if all(op == "UPDATE" for op in recent_ops):
                    return KillDecision(
                        should_kill=True,
                        reason="Base 3 kill: always-UPDATE reward hack detected (3 consecutive runs)",
                        fallback="gepa",
                    )

        # --- Base 6 (Temporal): held-out eval < 30% after 8 runs ---
        if env == "temporal":
            if len(records) >= 8 and latest.eval_score < 0.30:
                return KillDecision(
                    should_kill=True,
                    reason=f"Base 6 kill: held-out eval {latest.eval_score:.1%} < 30% after {len(records)} runs",
                    fallback="gepa",
                )

        # --- Generic: N runs without convergence ---
        if len(records) >= stage.max_runs_before_kill:
            recent_scores = [r.eval_score for r in records[-stage.max_runs_before_kill:]]
            avg_recent = sum(recent_scores) / len(recent_scores)
            if avg_recent < stage.kill_threshold:
                return KillDecision(
                    should_kill=True,
                    reason=(
                        f"{env} kill: avg score {avg_recent:.2f} < threshold {stage.kill_threshold} "
                        f"after {len(records)} runs"
                    ),
                    fallback="gepa",
                )

        # --- Base 7: identify weakest component ---
        if env == "integration" and latest.eval_score < stage.kill_threshold:
            return KillDecision(
                should_kill=True,
                reason=f"Base 7 kill: score {latest.eval_score:.2f} < {stage.kill_threshold}. "
                       "Iterate weakest base env.",
                fallback="iterate_base_env",
            )

        return KillDecision(should_kill=False, reason="within tolerance")

    # -----------------------------------------------------------------------
    # State persistence
    # -----------------------------------------------------------------------

    def _save_state(self) -> None:
        if not self._state_path:
            return
        state = {
            "stage_idx": self._stage_idx,
            "gepa_active": list(self._gepa_active),
            "run_records": {
                env: [
                    {
                        "run_id": r.run_id,
                        "eval_score": r.eval_score,
                        "hallucination_rate": r.hallucination_rate,
                        "compression_ratio": r.compression_ratio,
                        "dominant_op": r.dominant_op,
                    }
                    for r in records
                ]
                for env, records in self._run_records.items()
            },
        }
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(state, indent=2))

    def _load_state(self) -> None:
        data = json.loads(self._state_path.read_text())
        self._stage_idx = data.get("stage_idx", 0)
        self._gepa_active = set(data.get("gepa_active", []))
        for env, records in data.get("run_records", {}).items():
            self._run_records[env] = [
                EnvRunRecord(
                    env=env,
                    run_id=r["run_id"],
                    eval_score=r["eval_score"],
                    hallucination_rate=r.get("hallucination_rate", 0.0),
                    compression_ratio=r.get("compression_ratio"),
                    dominant_op=r.get("dominant_op"),
                )
                for r in records
            ]
