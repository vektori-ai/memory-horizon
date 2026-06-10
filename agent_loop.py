"""Multi-turn memory agent loop for verl v0.5.0 GRPO training.

Two responsibilities:
  1. build_verl_batch()     — data-prep: slices trajectories into K-turn episode
                              windows and returns raw_chat format rows for verl.
  2. MemoryAgentLoop        — rollout: implements verl's AgentLoopBase, runs K turns
                              of model inference while building the VirtualFilesystem,
                              returns AgentLoopOutput with proper response_mask.

Episode design (K=20 turns, stride=10):
  Each episode window = K consecutive turns from one trajectory.
  Turn 0: model sees (system_prompt + empty FS + turn_0) → outputs op_0
  Turn t: model sees (system_prompt + FS_after_t-1 + turn_t) → outputs op_t
  Terminal: score_trajectory(FS, window_probes) → reward R
  step-wise GRPO broadcasts R to all model-generated tokens in the episode.

Window filtering:
  Each window is only included if it covers at least one probe-relevant session.
  For LoCoMo, probe.requires_sessions are real session IDs.
  For LongMemEval, requires_sessions are answer-source IDs that don't match session IDs;
  we resolve these by searching session content for the answer text.
  Windows with 0 scoreable probes are skipped — they contribute zero gradient signal.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import TYPE_CHECKING

from memory_fs import VirtualFilesystem, parse_op, score_trajectory

if TYPE_CHECKING:
    from mh_types import Trajectory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPISODE_WINDOW  = 20   # turns per episode
EPISODE_STRIDE  = 10   # stride between windows (overlap = WINDOW - STRIDE)

_SYSTEM_PROMPT = """\
You are a memory manager for a long-running conversational AI.
As each conversation turn arrives, decide what to store, update, or compress in your memory filesystem.

Your memory is organised as files under category paths:
  people/   — facts about individuals
  events/   — past and future events
  places/   — locations and addresses
  facts/    — general facts and context
  prefs/    — preferences and habits

Output a single JSON object (no explanation):
{"op": "<STORE_FACT|UPDATE|SUPERSEDE|COMPRESS|ABSTAIN>", "path": "<category/entity>", "content": "<what to store>"}

Op rules:
  STORE_FACT  : new fact not yet stored; path = category/entity (e.g. people/alice)
  UPDATE      : existing fact changed or was wrong — replace current entry at path
  SUPERSEDE   : stronger overwrite — all prior entries at path replaced by this one
  COMPRESS    : collapse verbose path lines into one summary
  ABSTAIN     : no storable information in this turn
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
                    "trajectory_id":    traj_id,
                    "window_start":     win_start,
                    "first_turn":       first_turn,
                    "rest_turns":       rest_turns,
                    "qa_probes":        scoreable,
                    "window_session_ids": list(window_session_ids),
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

        extra = sampling_params.get("extra_info", {})
        rest_turns       = extra.get("rest_turns", [])
        qa_probes        = extra.get("qa_probes", [])
        window_sids      = set(extra.get("window_session_ids", []))

        # Filter probes to window-relevant ones (already filtered in build_verl_batch,
        # but re-filter here in case extra_info was serialized without window_session_ids)
        if window_sids:
            filtered = [
                p for p in qa_probes
                if not p.get("requires_sessions")
                or any(r in window_sids for r in p["requires_sessions"])
            ]
            if filtered:
                qa_probes = filtered

        fs             = VirtualFilesystem()
        all_prompt_ids = []
        all_resp_ids   = []
        all_resp_mask  = []
        all_responses  = []   # raw text per step, for logging
        session_idx    = extra.get("first_turn", {}).get("session_idx", 0)
        turn_offset    = extra.get("first_turn", {}).get("turn_idx", 0)

        current_messages = list(messages)

        for step_idx in range(1 + len(rest_turns)):
            step_prompt_ids = self.tokenize(current_messages)

            if step_idx == 0:
                all_prompt_ids = step_prompt_ids

            response_ids, response_text = await self.generate(
                step_prompt_ids, sampling_params
            )
            all_responses.append(response_text)

            op = parse_op(response_text)
            fs.apply_op(
                op,
                session_idx=op.get("_s", session_idx),
                turn_idx=turn_offset + step_idx,
            )

            all_resp_ids  += response_ids
            all_resp_mask += [1] * len(response_ids)

            if step_idx < len(rest_turns):
                next_turn = rest_turns[step_idx]

                observation = (
                    f"\n[Memory state]\n{fs.render_for_prompt()}\n\n"
                    f"[Next turn]\n{_format_turn(next_turn)}"
                )
                obs_ids = self.tokenize_text(observation)
                all_resp_ids  += obs_ids
                all_resp_mask += [0] * len(obs_ids)

                current_messages = current_messages + [
                    {"role": "assistant", "content": response_text},
                    {"role": "user",      "content": observation},
                ]

        reward = score_trajectory(fs, qa_probes)

        _log_episode_stats(all_responses, reward, len(qa_probes))

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

def _log_episode_stats(responses: list[str], terminal_reward: float, n_probes: int) -> None:
    """Log per-episode stats to WandB if a run is active."""
    try:
        import wandb
        if not wandb.run:
            return
        op_types = [parse_op(r).get("op", "INVALID") for r in responses]
        counts   = Counter(op_types)
        n        = max(len(op_types), 1)
        wandb.log({
            "episode/terminal_reward":  terminal_reward,
            "episode/n_probes":         n_probes,
            "episode/abstain_rate":     counts.get("ABSTAIN", 0) / n,
            "episode/store_rate":       counts.get("STORE_FACT", 0) / n,
            "episode/update_rate":      (counts.get("UPDATE", 0) + counts.get("SUPERSEDE", 0)) / n,
            "episode/compress_rate":    counts.get("COMPRESS", 0) / n,
            "episode/invalid_rate":     counts.get("INVALID", 0) / n,
            "episode/fs_paths":         0,   # placeholder; fs not accessible here
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
