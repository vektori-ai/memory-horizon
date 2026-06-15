"""Multi-turn memory agent loop for verl v0.5.0 GRPO training.

Two responsibilities:
  1. build_verl_batch()  — data-prep: slices trajectories into K-turn episode
                           windows and returns raw_chat format rows for verl.
  2. MemoryAgentLoop     — rollout: verl AgentLoopBase implementation.

Episode flow:
  Turn 0..K: model sees (system + harness context + turn) → emits op
    WRITE ops    → applied to FS, scored against ledger (step reward)
    RETRIEVE ops → harness calls Context-1, result injected as observation
    ABSTAIN      → no-op write, logged
  Probe phase: for each QA probe in window
    → harness injects [FS + retrieved slice + question]
    → model emits RESOLVE(answer)
    → scored against gold (terminal reward)
  Final reward = MemoryHarnessState.compute_reward()

Architecture mirrors harness-1:
  MemoryHarnessState ↔ WorkingMemory
  RETRIEVE action    ↔ search_corpus tool
  RESOLVE action     ↔ user_text (final answer)
  Ledger             ↔ answer documents ground truth
"""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING

from ledger import derive_ledger
from harness_state import MemoryHarnessState
from memory_fs import parse_op, retrieve_for_probe

if TYPE_CHECKING:
    from mh_types import Trajectory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPISODE_WINDOW  = 10   # turns per episode (20 OOMs at 32 concurrent seqs; 10 keeps KV ~9-12K tokens)
EPISODE_STRIDE  = 5    # stride between windows (overlap = WINDOW - STRIDE)

_SYSTEM_PROMPT = """\
You are a memory manager for a long-running conversational AI.
As each conversation turn arrives, decide what to store, update, retrieve, or compress.

Your memory is a filesystem of category paths:
  people/   — facts about individuals
  events/   — past and future events
  places/   — locations and addresses
  facts/    — general facts and context
  prefs/    — preferences and habits

Output a single JSON object (no explanation):

Write ops (modify the filesystem):
  {"op": "STORE_FACT",  "path": "category/entity", "content": "what to store"}
  {"op": "UPDATE",      "path": "category/entity", "content": "updated value"}
  {"op": "SUPERSEDE",   "path": "category/entity", "content": "new value replacing all prior"}
  {"op": "COMPRESS",    "path": "category/entity", "content": "summary of this path"}
  {"op": "ABSTAIN"}

Retrieval op (search your memory before writing or answering):
  {"op": "RETRIEVE", "query": "what you want to look up"}
  The harness will return relevant entries from your memory.

Answer op (only when asked a probe question):
  {"op": "RESOLVE", "content": "your answer based on retrieved memory"}
/nothink"""


# ---------------------------------------------------------------------------
# Data prep — pure Python, no verl dependency
# ---------------------------------------------------------------------------

def _resolve_probe_sessions(probes: list[dict], sessions: list[dict]) -> list[dict]:
    """Ensure every probe has requires_sessions that maps to real session IDs.

    LoCoMo probes already have correct session IDs — passed through unchanged.
    LongMemEval probes have answer-source IDs (e.g. 'answer_c63c0458') that don't
    match session IDs; for these we search session content for the answer text and
    tag the probe with the containing session(s).
    Probes whose answer can't be located are kept as-is (they'll match all windows).
    """
    all_session_ids = {s.get("session_id", "") for s in sessions}

    result = []
    for probe in probes:
        req = probe.get("requires_sessions", [])
        if req and any(r in all_session_ids for r in req):
            result.append(probe)
            continue

        # LongMemEval: locate the answer in session content
        answer = str(probe.get("answer", "")).lower().strip()
        if not answer or answer in ("n/a", "yes", "no", ""):
            continue  # unanswerable probe — skip entirely

        found = []
        for session in sessions:
            sid = session.get("session_id", "")
            for turn in session.get("turns", []):
                if answer in turn.get("content", "").lower():
                    found.append(sid)
                    break
        if found:
            probe = {**probe, "requires_sessions": list(set(found))}
        result.append(probe)

    return result


def build_verl_batch(
    trajectories: list[dict],
    window_size: int = EPISODE_WINDOW,
    stride: int = EPISODE_STRIDE,
) -> list[dict]:
    """Convert trajectory dicts → verl raw_chat format rows.

    Each row covers one K-turn window that has at least one scoreable QA probe.
    Windows with no probe-relevant sessions are skipped — they'd always score 0.0
    and contribute no gradient signal to GRPO.

    verl expects each row to have:
      data_source, prompt (raw_chat), ability, reward_model, extra_info
    """
    rows = []
    skipped_windows = 0
    total_windows = 0

    for traj in trajectories:
        traj_id  = traj.get("trajectory_id", "")
        vertical = traj.get("vertical", "locomo")
        sessions = traj.get("sessions", [])

        qa_probes = [
            {**p, "answer": str(p.get("answer", ""))}
            for p in traj.get("qa_probes", [])
        ]
        qa_probes = _resolve_probe_sessions(qa_probes, sessions)
        if not qa_probes:
            continue

        # Flatten all turns with session/turn indices and session_id string for filtering
        all_turns = []
        for s_idx, session in enumerate(sessions):
            sid = session.get("session_id", "")
            for t_idx, turn in enumerate(session.get("turns", [])):
                all_turns.append({
                    "role":        turn.get("role", "user"),
                    "content":     turn.get("content", "").strip(),
                    "session_idx": s_idx,
                    "session_id":  sid,
                    "turn_idx":    t_idx,
                })

        all_turns = [t for t in all_turns if t["content"]]
        if not all_turns:
            continue

        for win_start in range(0, max(1, len(all_turns) - window_size + 1), stride):
            window = all_turns[win_start : win_start + window_size]
            if not window:
                continue
            total_windows += 1

            window_session_ids = set(t["session_id"] for t in window)

            # Only include windows that have at least one scoreable probe
            scoreable = [
                p for p in qa_probes
                if not p.get("requires_sessions")
                or any(r in window_session_ids for r in p["requires_sessions"])
            ]
            if not scoreable:
                skipped_windows += 1
                continue

            first_turn = window[0]
            rest_turns = window[1:]

            prompt = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _format_turn(first_turn)},
            ]

            rows.append({
                "data_source":  vertical,
                "prompt":       prompt,
                "ability":      "memory_management",
                "reward_model": {"ground_truth": json.dumps(scoreable)},
                "extra_info": {
                    "trajectory_id":      traj_id,
                    "window_start":       win_start,
                    "first_turn":         first_turn,
                    "rest_turns":         rest_turns,
                    "qa_probes":          scoreable,
                    "window_session_ids": list(window_session_ids),
                    "sessions":           sessions,   # needed by derive_ledger
                    "context1_url":       "",  # filled in by train_modal.py at prep time
                },
            })

    kept = total_windows - skipped_windows
    print(
        f"Windows: {total_windows} total, {skipped_windows} skipped (0 probes), "
        f"{kept} kept ({100*kept/max(total_windows,1):.1f}%)"
    )
    return rows


# ---------------------------------------------------------------------------
# verl AgentLoop — runs inside Modal container where verl is installed
# ---------------------------------------------------------------------------

def _get_agent_loop_base():
    """Lazy import of verl's AgentLoopBase to keep this file importable locally."""
    try:
        from verl.trainer.ppo.agent_loop import AgentLoopBase, AgentLoopOutput
        return AgentLoopBase, AgentLoopOutput
    except ImportError:
        return None, None


class MemoryAgentLoop:
    """verl v0.5.0 AgentLoop implementation for multi-turn memory management.

    Registered with verl via train_modal.py config:
        actor_rollout_ref.rollout.agent_loop_cls=agent_loop.MemoryAgentLoop

    For each episode window:
      1. Model generates op for turn 0 (from initial prompt in the data row).
      2. Op applied to VirtualFilesystem.
      3. FS render + next turn injected as an observation (response_mask=0).
      4. Model generates op for turn 1. Repeat until window exhausted.
      5. Terminal reward = score_trajectory(FS, window_probes).
      6. Returns AgentLoopOutput: prompt_ids, response_ids, response_mask.
    """

    def __init_subclass_hook__(cls):
        AgentLoopBase, _ = _get_agent_loop_base()
        if AgentLoopBase is not None:
            cls.__bases__ = (AgentLoopBase,)

    async def run(self, messages: list[dict], sampling_params: dict):
        AgentLoopBase, AgentLoopOutput = _get_agent_loop_base()
        if AgentLoopBase is None:
            raise ImportError("verl not installed — MemoryAgentLoop requires verl v0.5.0")

        extra        = sampling_params.get("extra_info", {})
        rest_turns   = extra.get("rest_turns", [])
        qa_probes    = extra.get("qa_probes", [])
        window_sids  = set(extra.get("window_session_ids", []))
        context1_url = extra.get("context1_url") or None
        sessions     = extra.get("sessions", [])

        if window_sids:
            filtered = [
                p for p in qa_probes
                if not p.get("requires_sessions")
                or any(r in window_sids for r in p["requires_sessions"])
            ]
            if filtered:
                qa_probes = filtered

        # Build harness state — the WorkingMemory analog
        ledger  = derive_ledger(qa_probes, sessions)
        harness = MemoryHarnessState(ledger=ledger)

        all_prompt_ids   = []
        all_resp_ids     = []
        all_resp_mask    = []
        all_responses    = []
        session_idx      = extra.get("first_turn", {}).get("session_idx", 0)
        turn_offset      = extra.get("first_turn", {}).get("turn_idx", 0)
        current_messages = list(messages)

        # ── Conversation turns phase ───────────────────────────────────────
        for step_idx in range(1 + len(rest_turns)):
            step_prompt_ids = self.tokenize(current_messages)
            if step_idx == 0:
                all_prompt_ids = step_prompt_ids

            response_ids, response_text = await self.generate(step_prompt_ids, sampling_params)
            all_responses.append(response_text)
            all_resp_ids  += response_ids
            all_resp_mask += [1] * len(response_ids)

            op      = parse_op(response_text)
            op_type = op.get("op", "INVALID") if op else "INVALID"

            if op_type == "RETRIEVE":
                # Agent explicitly searches its own memory — harness executes Context-1
                query   = op.get("query", "").strip()
                results = retrieve_for_probe(query, harness.fs, context1_url)
                harness.apply_retrieve(query, [results] if results else [], turn_offset + step_idx)

                retrieve_obs = (
                    f"\n[Retrieved for: {query}]\n"
                    f"{results or '(no relevant entries found)'}"
                )
                ret_ids = self.tokenize_text(retrieve_obs)
                all_resp_ids  += ret_ids
                all_resp_mask += [0] * len(ret_ids)   # harness output, not trained on
                current_messages = current_messages + [
                    {"role": "assistant", "content": response_text},
                    {"role": "user",      "content": retrieve_obs},
                ]
            else:
                # All write ops (STORE_FACT, UPDATE, SUPERSEDE, COMPRESS, ABSTAIN)
                harness.apply_op(op, session_idx=session_idx, turn_idx=turn_offset + step_idx)

                if step_idx < len(rest_turns):
                    next_turn   = rest_turns[step_idx]
                    observation = (
                        f"\n[Memory state]\n{harness.render_context()}\n\n"
                        f"[Next turn]\n{_format_turn(next_turn)}"
                    )
                    obs_ids = self.tokenize_text(observation)
                    all_resp_ids  += obs_ids
                    all_resp_mask += [0] * len(obs_ids)
                    current_messages = current_messages + [
                        {"role": "assistant", "content": response_text},
                        {"role": "user",      "content": observation},
                    ]

        # ── Probe / RESOLVE phase ──────────────────────────────────────────
        for probe in qa_probes:
            question = probe.get("question", "") if isinstance(probe, dict) else getattr(probe, "question", "")
            answer   = str(probe.get("answer", "") if isinstance(probe, dict) else getattr(probe, "answer", ""))
            if not answer or answer.lower() in ("abstain", "yes", "no", "n/a", ""):
                continue

            retrieved = retrieve_for_probe(question, harness.fs, context1_url)
            probe_obs = (
                f"\n[Memory state]\n{harness.render_context()}\n\n"
                f"[Retrieved]\n{retrieved or '(no relevant entries found)'}\n\n"
                f"[Question]\n{question}"
            )
            probe_messages   = current_messages + [{"role": "user", "content": probe_obs}]
            probe_prompt_ids = self.tokenize(probe_messages)
            resolve_ids, resolve_text = await self.generate(probe_prompt_ids, sampling_params)

            all_resp_ids  += resolve_ids
            all_resp_mask += [1] * len(resolve_ids)
            all_responses.append(resolve_text)

            resolve_op      = parse_op(resolve_text)
            resolve_content = (
                resolve_op.get("content", "")
                if resolve_op and resolve_op.get("op") == "RESOLVE"
                else resolve_text.strip()   # fallback: treat raw text as answer
            )
            harness.apply_resolve(resolve_content, answer)

        reward = harness.compute_reward()
        _log_episode_stats(all_responses, harness.summary())

        return AgentLoopOutput(
            prompt_ids    = all_prompt_ids,
            response_ids  = all_resp_ids,
            response_mask = all_resp_mask,
            reward        = reward,
        )

    # -- stubs filled in by verl's AgentLoopWorker at runtime --

    async def generate(self, prompt_ids: list[int], sampling_params: dict):
        """Generate response tokens. Implemented by verl AgentLoopBase."""
        raise NotImplementedError

    def tokenize(self, messages: list[dict]) -> list[int]:
        """Tokenize a chat message list. Implemented by verl AgentLoopBase."""
        raise NotImplementedError

    def tokenize_text(self, text: str) -> list[int]:
        """Tokenize a raw string. Implemented by verl AgentLoopBase."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log_episode_stats(responses: list[str], summary: dict) -> None:
    """Log per-episode stats to WandB if a run is active."""
    try:
        import wandb
        if not wandb.run:
            return
        op_counts = summary.get("op_counts", {})
        n         = max(sum(op_counts.values()), 1)
        wandb.log({
            "episode/reward":          summary.get("final_reward", 0.0),
            "episode/mean_resolve":    summary.get("mean_resolve", 0.0),
            "episode/mean_step":       summary.get("mean_step", 0.0),
            "episode/n_retrievals":    summary.get("n_retrievals", 0),
            "episode/n_fs_paths":      len(summary.get("fs_paths", [])),
            "episode/abstain_rate":    op_counts.get("ABSTAIN", 0) / n,
            "episode/store_rate":      op_counts.get("STORE_FACT", 0) / n,
            "episode/update_rate":     (op_counts.get("UPDATE", 0) + op_counts.get("SUPERSEDE", 0)) / n,
            "episode/retrieve_rate":   op_counts.get("RETRIEVE", 0) / n,
            "episode/invalid_rate":    op_counts.get("INVALID", 0) / n,
        })
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_turn(turn: dict) -> str:
    role    = "Agent" if turn.get("role") == "assistant" else "User"
    content = turn.get("content", "").strip()
    return f"{role}: {content}"
