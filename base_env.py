"""MemoryHorizonEnv — base environment for all memory_horizon training envs.

Architecture:
    One episode = one Trajectory (N sessions + QA probes).
    The model (memory manager) processes each Turn, emitting a MemoryOpAction.
    After all sessions, a frozen QA model answers the probes using stored memory.
    Reward = Tier 1 (per-turn, immediate) + Tier 3 (episode-end, from QA probes).

Multi-session loop::

    reset() → obs(Session 1, Turn 0)
    step(action) → obs(Session 1, Turn 1), reward=tier1_score, done=False
    ...
    step(action) → obs(Session 2, Turn 0), reward=tier1_score, done=False  ← session boundary
    ...
    step(action) → obs(QA probe 0), reward=0, done=False                   ← qa phase
    step(action) → final reward, done=True                                 ← episode ends

Observation format (returned as a dict):
    {
        "phase":    "session" | "qa",
        "session_id":  str,
        "session_idx": int,
        "turn_idx":    int,
        "role":        "user" | "assistant",
        "content":     str,              ← the turn text or QA question
        "memory_context": str,           ← injected memory for this step
        "system_prompt": str,            ← ready-to-use system prompt
        "qa_idx":      int | None,       ← only in qa phase
        "total_turns": int,
    }

Action format (string — JSON or structured dict):
    For session phase:   MemoryOpAction JSON → {"op": "...", "content": "...", "key": "..."}
    For QA phase:        plain text answer   → {"answer": "..."}

This follows the stateful multi-turn env pattern from gecko/seta/toolathlon.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from memory_horizon.core import Env
from memory_horizon.spaces.text import Text

from memory_horizon.memory_store import MemoryStore
from memory_horizon.mh_types import (
    MemoryOp,
    MemoryOpAction,
    QAPair,
    Session,
    Trajectory,
    VerifierResult,
    VerifierTier,
)
from memory_horizon.verifiers.hallucination import HallucinationDetector
from memory_horizon.verifiers.tier1_deterministic import (
    Tier1Verifier,
    parse_action,
)
from memory_horizon.verifiers.tier3_outcome import (
    score_with_qa_model,
    score_synthesize,
    score_temporal,
    score_compress,
)


# ---------------------------------------------------------------------------
# Episode state
# ---------------------------------------------------------------------------

@dataclass
class EpisodeState:
    """Mutable state for one episode. Encapsulated to make reset() clean."""
    trajectory: Trajectory
    session_idx: int = 0
    turn_idx: int = 0
    qa_idx: int = 0
    phase: str = "session"             # "session" | "qa"
    tier1_scores: list[float] = field(default_factory=list)
    tier3_scores: list[float] = field(default_factory=list)
    tier1_results: list[VerifierResult] = field(default_factory=list)
    tier3_results: list[VerifierResult] = field(default_factory=list)
    qa_answers: list[str] = field(default_factory=list)
    step_count: int = 0

    @property
    def current_session(self) -> Session:
        return self.trajectory.sessions[self.session_idx]

    @property
    def current_turn(self):
        return self.current_session.turns[self.turn_idx]

    @property
    def current_qa(self) -> QAPair:
        return self.trajectory.qa_probes[self.qa_idx]

    @property
    def n_sessions(self) -> int:
        return len(self.trajectory.sessions)

    @property
    def n_qa(self) -> int:
        return len(self.trajectory.qa_probes)

    def total_turns(self) -> int:
        return sum(len(s.turns) for s in self.trajectory.sessions)


# ---------------------------------------------------------------------------
# MemoryHorizonEnv
# ---------------------------------------------------------------------------

class MemoryHorizonEnv(Env[dict[str, Any], dict[str, Any]]):
    """Base RL environment for memory manager training.

    Extend this class to create specific base envs (Store, Contradict, etc.)
    or vertical envs (voice_customer, support_technical, etc.).

    Args:
        env_type: "store" | "synthesize" | "contradict" | "compress" |
                  "abstain" | "temporal" | "integration"
        trajectory_fn: Callable[[], Trajectory] — called on reset() to get
                        the next training example. Typically a dataset iterator.
        store_config: Dict of kwargs forwarded to MemoryStore constructor.
        qa_fn: Optional callable(question, context) → answer string.
               If provided, used as the frozen QA model for Tier 3.
               If None, falls back to token-recall heuristic.
        tier1_weight: Weight of Tier 1 scores in final episode reward. Default 0.3.
        tier3_weight: Weight of Tier 3 scores in final episode reward. Default 0.7.
        reward_per_turn: If True, return Tier 1 score at each step (dense).
                         If False, return 0 until episode ends (sparse).
        system_prompt_prefix: Prepended to the generated system prompt.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        env_type: str,
        trajectory_fn: Callable[[], Trajectory],
        store_config: dict[str, Any] | None = None,
        qa_fn: Callable[[str, str], str] | None = None,
        tier1_weight: float = 0.3,
        tier3_weight: float = 0.7,
        reward_per_turn: bool = True,
        system_prompt_prefix: str = "",
    ) -> None:
        self.env_type = env_type
        self.trajectory_fn = trajectory_fn
        self.store_config = store_config or {}
        self.qa_fn = qa_fn
        self.tier1_weight = tier1_weight
        self.tier3_weight = tier3_weight
        self.reward_per_turn = reward_per_turn
        self.system_prompt_prefix = system_prompt_prefix

        self.action_space = Text(min_length=2, max_length=8192)
        self.observation_space = Text(min_length=0, max_length=65536)

        self._tier1 = Tier1Verifier(env_type=env_type)
        self._hallucination = HallucinationDetector()

        self._state: EpisodeState | None = None
        self._store: MemoryStore | None = None

    # -----------------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new episode with a fresh trajectory."""
        trajectory = self.trajectory_fn()
        self._store = MemoryStore(**self.store_config)
        self._state = EpisodeState(trajectory=trajectory)

        first_session = trajectory.sessions[0]
        self._store.session_boundary(first_session.session_id)

        obs = self._build_obs()
        info = {
            "trajectory_id": trajectory.trajectory_id,
            "n_sessions": self._state.n_sessions,
            "n_qa_probes": self._state.n_qa,
            "vertical": trajectory.vertical,
            "system_prompt": self._build_system_prompt(),
        }
        return obs, info

    def step(
        self, action: dict[str, Any] | str
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Process one model action.

        In session phase: action is a MemoryOpAction JSON.
        In QA phase: action is {"answer": "..."} or a plain string.

        Returns:
            obs: Next observation.
            reward: Per-turn Tier 1 score (if reward_per_turn) or 0.
            terminated: True only when the last QA probe is answered.
            truncated: False (handled by TimeLimit wrapper).
            info: Verifier results, step metadata.
        """
        if self._state is None or self._store is None:
            raise RuntimeError("reset() must be called before step()")

        s = self._state
        s.step_count += 1

        if s.phase == "session":
            return self._step_session(action)
        else:
            return self._step_qa(action)

    def render(self) -> str | None:
        if self._state is None:
            return None
        s = self._state
        lines = [
            f"Env: {self.env_type}  |  Trajectory: {s.trajectory.trajectory_id}",
            f"Phase: {s.phase}  |  Session {s.session_idx + 1}/{s.n_sessions}  "
            f"Turn {s.turn_idx}/{len(s.current_session.turns)}",
            f"Tier1 scores: {[round(x, 2) for x in s.tier1_scores]}",
            f"Tier3 scores: {[round(x, 2) for x in s.tier3_scores]}",
            "",
            "--- Memory Store ---",
            self._store.get_context() if self._store else "",
        ]
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Session phase
    # -----------------------------------------------------------------------

    def _step_session(
        self, action: dict[str, Any] | str
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        s = self._state
        store = self._store
        turn = s.current_turn

        # Log raw turn to RAW layer.
        store.log_raw(turn.content, s.turn_idx, role=turn.role)

        # Parse the action.
        raw_str = action if isinstance(action, str) else json.dumps(action)
        parsed_action, parse_error = parse_action(raw_str)

        reward: float = 0.0
        t1_result = None

        if parse_error or parsed_action is None:
            # Unparseable action — Tier 1 automatic fail.
            t1_result_obj = _make_parse_error_result(parse_error)
            s.tier1_scores.append(-1.0)
            s.tier1_results.append(t1_result_obj)
            reward = -1.0 if self.reward_per_turn else 0.0
        else:
            # Execute in store, then verify.
            exec_result = store.execute(parsed_action)
            t1_result_obj = self._tier1.verify(
                action=parsed_action,
                store=store,
                exec_result=exec_result,
                raw_transcript=turn.content,
            ).to_verifier_result()
            s.tier1_scores.append(t1_result_obj.score)
            s.tier1_results.append(t1_result_obj)
            reward = t1_result_obj.score if self.reward_per_turn else 0.0

        # Advance turn/session pointer.
        terminated, truncated = self._advance_session_pointer()

        obs = self._build_obs()
        info: dict[str, Any] = {
            "phase": "session",
            "tier1": t1_result_obj.to_dict(),
            "step": s.step_count,
            "parse_error": parse_error,
        }
        return obs, reward, terminated, truncated, info

    def _advance_session_pointer(self) -> tuple[bool, bool]:
        """Move to next turn or next session. Returns (terminated, truncated)."""
        s = self._state
        store = self._store
        session = s.current_session

        s.turn_idx += 1

        if s.turn_idx >= len(session.turns):
            # End of current session.
            s.turn_idx = 0
            s.session_idx += 1

            if s.session_idx >= s.n_sessions:
                # All sessions done — move to QA phase.
                s.phase = "qa"
                if s.n_qa == 0:
                    return True, False  # no QA probes → episode ends
                return False, False

            # Start next session.
            next_session = s.trajectory.sessions[s.session_idx]
            store.session_boundary(next_session.session_id)

        return False, False

    # -----------------------------------------------------------------------
    # QA phase
    # -----------------------------------------------------------------------

    def _step_qa(
        self, action: dict[str, Any] | str
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        s = self._state
        store = self._store
        qa = s.current_qa

        # Extract answer from action.
        if isinstance(action, str):
            try:
                d = json.loads(action)
                answer = str(d.get("answer", action))
            except (json.JSONDecodeError, AttributeError):
                answer = action
        elif isinstance(action, dict):
            answer = str(action.get("answer", ""))
        else:
            answer = str(action)

        s.qa_answers.append(answer)

        # Tier 3 scoring.
        memory_context = store.get_context(query=qa.question)
        t3_result = score_with_qa_model(
            question=qa.question,
            stored_memory_context=memory_context,
            gold_answer=qa.answer,
            qa_fn=self.qa_fn,
        )
        s.tier3_scores.append(t3_result.score)
        s.tier3_results.append(VerifierResult(
            score=t3_result.score,
            tier=VerifierTier.TIER3,
            passed=t3_result.passed,
            metadata=t3_result.details,
        ))

        s.qa_idx += 1
        terminated = s.qa_idx >= s.n_qa

        reward = 0.0
        if terminated:
            reward = self._compute_episode_reward()

        obs = self._build_obs()
        info: dict[str, Any] = {
            "phase": "qa",
            "qa_idx": s.qa_idx - 1,
            "tier3": t3_result.to_dict(),
            "step": s.step_count,
            "episode_reward": reward if terminated else None,
        }
        if terminated:
            info["episode_summary"] = self._episode_summary()

        return obs, reward, terminated, False, info

    # -----------------------------------------------------------------------
    # Reward computation
    # -----------------------------------------------------------------------

    def _compute_episode_reward(self) -> float:
        """Weighted combination of Tier 1 (per-turn avg) and Tier 3 (QA avg).

        Reward convention: clipped to [-2, +1].
        """
        s = self._state

        t1_avg = (sum(s.tier1_scores) / len(s.tier1_scores)) if s.tier1_scores else 0.0
        t3_avg = (sum(s.tier3_scores) / len(s.tier3_scores)) if s.tier3_scores else 0.0

        raw = self.tier1_weight * t1_avg + self.tier3_weight * t3_avg
        return max(-2.0, min(1.0, raw))

    def _episode_summary(self) -> dict[str, Any]:
        s = self._state
        return {
            "trajectory_id": s.trajectory.trajectory_id,
            "n_turns": s.step_count,
            "tier1_avg": round(sum(s.tier1_scores) / max(len(s.tier1_scores), 1), 3),
            "tier3_avg": round(sum(s.tier3_scores) / max(len(s.tier3_scores), 1), 3),
            "episode_reward": round(self._compute_episode_reward(), 3),
            "store_stats": self._store.stats(),
        }

    # -----------------------------------------------------------------------
    # Observation builder
    # -----------------------------------------------------------------------

    def _build_obs(self) -> dict[str, Any]:
        s = self._state
        store = self._store

        if s.phase == "qa":
            if s.qa_idx >= s.n_qa:
                return {"phase": "done", "content": "", "memory_context": "",
                        "system_prompt": "", "session_id": "", "session_idx": -1,
                        "turn_idx": -1, "qa_idx": s.qa_idx, "total_turns": s.total_turns()}

            qa = s.current_qa
            ctx = store.get_context(query=qa.question)
            return {
                "phase": "qa",
                "content": qa.question,
                "memory_context": ctx,
                "system_prompt": self._build_system_prompt(qa_mode=True),
                "session_id": "",
                "session_idx": s.session_idx,
                "turn_idx": -1,
                "qa_idx": s.qa_idx,
                "difficulty": qa.difficulty,
                "total_turns": s.total_turns(),
            }

        # Session phase.
        turn = s.current_turn
        ctx = store.get_context()
        return {
            "phase": "session",
            "session_id": s.current_session.session_id,
            "session_idx": s.session_idx,
            "turn_idx": s.turn_idx,
            "role": turn.role,
            "content": turn.content,
            "timestamp": turn.timestamp,
            "memory_context": ctx,
            "system_prompt": self._build_system_prompt(),
            "qa_idx": None,
            "total_turns": s.total_turns(),
        }

    def _build_system_prompt(self, qa_mode: bool = False) -> str:
        parts: list[str] = []

        if self.system_prompt_prefix:
            parts.append(self.system_prompt_prefix)

        parts.append(
            "You are a memory manager. For each conversation turn, decide which "
            "memory operation to perform and output a JSON action."
        )

        if self._store:
            ctx = self._store.get_context()
            parts.append(f"## Current Memory\n{ctx}")

        if qa_mode:
            parts.append(
                "## Task\nAnswer the following question using ONLY your stored memory. "
                "If the answer is not in memory, output: "
                '{"answer": "<abstain>I don\'t have that information"}'
            )
        else:
            parts.append(
                "## Task\nOutput a JSON memory operation for the current conversation turn.\n"
                'Format: {"op": "STORE_FACT|CREATE_EPISODE|INFER_IMPLICIT|UPDATE|'
                'SUPERSEDE|DECAY|KEEP_BOTH|COMPRESS|ABSTAIN", '
                '"content": "...", "key": "snake_case_key", "confidence": 0.0-1.0}'
            )

        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Convenience factory (used by environment modules)
# ---------------------------------------------------------------------------

def make_env(
    env_type: str,
    trajectory_fn: Callable[[], Trajectory],
    **kwargs: Any,
) -> MemoryHorizonEnv:
    """Factory for creating a typed MemoryHorizonEnv."""
    return MemoryHorizonEnv(env_type=env_type, trajectory_fn=trajectory_fn, **kwargs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_parse_error_result(error: str) -> VerifierResult:
    return VerifierResult(
        score=-1.0,
        tier=VerifierTier.TIER1,
        passed=False,
        reasons=[f"FAIL: parse error — {error}"],
    )
