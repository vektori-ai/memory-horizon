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
  Terminal: score_trajectory(FS, ALL qa_probes from trajectory) → reward R
  step-wise GRPO broadcasts R to all model-generated tokens in the episode.
"""

from __future__ import annotations

import json
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

def build_verl_batch(
    trajectories: list[dict],
    window_size: int = EPISODE_WINDOW,
    stride: int = EPISODE_STRIDE,
) -> list[dict]:
    """Convert trajectory dicts → verl raw_chat format rows.

    Each row covers one K-turn window from a trajectory.
    The remaining turns and qa_probes are stored in extra_info so
    MemoryAgentLoop can continue the episode during rollout.

    verl expects each row to have:
      data_source, prompt (raw_chat), ability, reward_model, extra_info
    """
    rows = []
    for traj in trajectories:
        traj_id  = traj.get("trajectory_id", "")
        vertical = traj.get("vertical", "locomo")
        sessions = traj.get("sessions", [])

        # Flatten all turns with session/turn indices for FS tagging
        all_turns = []
        for s_idx, session in enumerate(sessions):
            for t_idx, turn in enumerate(session.get("turns", [])):
                all_turns.append({
                    "role":        turn.get("role", "user"),
                    "content":     turn.get("content", "").strip(),
                    "session_idx": s_idx,
                    "turn_idx":    t_idx,
                })

        all_turns = [t for t in all_turns if t["content"]]
        if not all_turns:
            continue

        qa_probes = traj.get("qa_probes", [])

        # Sliding windows over the flattened turn list
        for win_start in range(0, max(1, len(all_turns) - window_size + 1), stride):
            window = all_turns[win_start : win_start + window_size]
            if not window:
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
                "reward_model": {"ground_truth": json.dumps(qa_probes)},
                "extra_info": {
                    "trajectory_id": traj_id,
                    "window_start":  win_start,
                    "first_turn":    first_turn,
                    "rest_turns":    rest_turns,
                    "qa_probes":     qa_probes,
                },
            })

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
      5. Terminal reward = score_trajectory(FS, qa_probes).
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

        # extra_info is injected by verl into sampling_params or messages metadata
        extra = sampling_params.get("extra_info", {})
        rest_turns  = extra.get("rest_turns", [])
        qa_probes   = extra.get("qa_probes", [])

        fs             = VirtualFilesystem()
        all_prompt_ids = []
        all_resp_ids   = []
        all_resp_mask  = []
        session_idx    = extra.get("first_turn", {}).get("session_idx", 0)
        turn_offset    = extra.get("first_turn", {}).get("turn_idx", 0)

        current_messages = list(messages)

        for step_idx in range(1 + len(rest_turns)):
            # Tokenize current messages → prompt_ids for this step
            step_prompt_ids = self.tokenize(current_messages)

            if step_idx == 0:
                all_prompt_ids = step_prompt_ids

            # Generate model response
            response_ids, response_text = await self.generate(
                step_prompt_ids, sampling_params
            )

            # Parse op and update FS
            op = parse_op(response_text)
            fs.apply_op(
                op,
                session_idx=op.get("_s", session_idx),
                turn_idx=turn_offset + step_idx,
            )

            # model-generated tokens → mask=1
            all_resp_ids  += response_ids
            all_resp_mask += [1] * len(response_ids)

            if step_idx < len(rest_turns):
                next_turn = rest_turns[step_idx]

                # Inject FS state + next turn as observation (mask=0)
                observation = (
                    f"\n[Memory state]\n{fs.render_for_prompt()}\n\n"
                    f"[Next turn]\n{_format_turn(next_turn)}"
                )
                obs_ids = self.tokenize_text(observation)
                all_resp_ids  += obs_ids
                all_resp_mask += [0] * len(obs_ids)

                # Append to messages for next generation context
                current_messages = current_messages + [
                    {"role": "assistant", "content": response_text},
                    {"role": "user",      "content": observation},
                ]

        # Terminal reward
        reward = score_trajectory(fs, qa_probes)

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
# Helpers
# ---------------------------------------------------------------------------

def _format_turn(turn: dict) -> str:
    role    = "Agent" if turn.get("role") == "assistant" else "User"
    content = turn.get("content", "").strip()
    return f"{role}: {content}"
