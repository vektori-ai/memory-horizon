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
import uuid
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

Your memory is a filesystem of category paths. Each path has an importance tag:
  [CONFIRMED]  — verified by multiple writes or an explicit UPDATE/SUPERSEDE
  [TENTATIVE]  — initial STORE_FACT (may be overwritten later)
  [SUPERSEDED] — replaced; kept for audit only

Categories:
  people/   — facts about individuals
  events/   — past and future events
  places/   — locations and addresses
  facts/    — general facts and context
  prefs/    — preferences and habits

Output a single JSON object (no explanation):

Write ops:
  {"op": "STORE_FACT",  "path": "category/entity", "content": "what to store"}       → tentative
  {"op": "UPDATE",      "path": "category/entity", "content": "updated value"}        → promotes to confirmed
  {"op": "SUPERSEDE",   "path": "category/entity", "content": "new value replacing all prior"} → confirmed
  {"op": "COMPRESS",    "path": "category/entity", "content": "summary of this path"} → confirmed
  {"op": "ABSTAIN"}

Retrieval op:
  {"op": "RETRIEVE", "query": "what you want to look up"}
  The harness returns relevant entries. Use this before answering probe questions.

Answer op (only when asked a probe question):
  {"op": "RESOLVE", "content": "your answer based on retrieved memory"}

Rules:
- Use UPDATE or SUPERSEDE (not STORE_FACT) when correcting something already stored.
- RETRIEVE before RESOLVE — check your memory before answering.
- Mix your op types; do not ABSTAIN on everything.
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
            # Truncate long turns to keep the gzip-compressed <__ep__> payload
            # small enough that all windows fit within max_prompt_length=4096.
            # Very long turns (articles, docs) tokenize to 400-4000 tokens; the
            # model can't meaningfully process >800 chars per turn in a 512-token
            # response anyway.
            _MAX_TURN_CHARS = 800
            rest_turns = [
                {**t, "content": t["content"][:_MAX_TURN_CHARS]}
                for t in window[1:]
            ]

            # Embed episode data in the system message so the agent loop can read it
            # even when verl's extra_info channel malfunctions (verl v0.5.0 calls
            # extra_info.strip() before json.loads(), which crashes on a Python dict).
            # We strip the marker before any generation call so the model never sees it.
            # gzip before base64: JSON compresses 3-5x, cutting prompt tokens from
            # ~4000-7500 down to ~600-1500 so all samples pass max_prompt_length=4096.
            import base64 as _b64, gzip as _gz
            _ep_json = json.dumps({
                "rest_turns":         rest_turns,
                "qa_probes":          scoreable,
                "window_session_ids": list(window_session_ids),
                "trajectory_id":      traj_id,
                "window_start":       win_start,
                "context1_url":       "",
            }).encode()
            _ep_payload = _b64.b64encode(_gz.compress(_ep_json, compresslevel=6)).decode()
            _system_with_ep = _SYSTEM_PROMPT + f"\n<__ep__>{_ep_payload}</__ep__>"

            prompt = [
                {"role": "system", "content": _system_with_ep},
                {"role": "user",   "content": _format_turn(first_turn)},
            ]

            rows.append({
                "data_source":  vertical,
                "prompt":       prompt,
                "ability":      "memory_management",
                # Top-level (not nested in extra_info) — verl reads this straight
                # into batch.non_tensor_batch["agent_name"] to route the row to
                # MemoryAgentLoop instead of the default single_turn_agent loop.
                "agent_name":   "memory_agent",
                "reward_model": {"ground_truth": json.dumps(scoreable)},
                "extra_info": {
                    "trajectory_id":      traj_id,
                    "window_start":       win_start,
                    "first_turn":         first_turn,
                    "rest_turns":         rest_turns,
                    "qa_probes":          scoreable,
                    "window_session_ids": list(window_session_ids),
                    "sessions":           sessions,
                    "context1_url":       "",
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
    """Lazy import of verl's AgentLoopBase to keep this file importable locally.

    Real module at verl v0.5.0 is verl.experimental.agent_loop.agent_loop —
    verl.trainer.ppo.agent_loop (the path this used to import from) does not
    exist at this pin (404 at the v0.5.0 tag); that import always failed
    silently and fell back to (None, None), even inside the Modal container
    where verl is installed.
    """
    try:
        from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
        return AgentLoopBase, AgentLoopOutput, register
    except ImportError:
        return None, None, None


_AgentLoopBase, _AgentLoopOutput, _register = _get_agent_loop_base()
_MemoryAgentLoopBase = _AgentLoopBase if _AgentLoopBase is not None else object


class MemoryAgentLoop(_MemoryAgentLoopBase):
    """verl v0.5.0 AgentLoop implementation for multi-turn memory management.

    Registered with verl via a YAML registry (verl v0.5.0 has no agent_loop_cls
    config key — it selects loops per-row via batch.non_tensor_batch["agent_name"],
    looked up in a registry populated from actor_rollout_ref.rollout.agent.agent_loop_config_path):
        # agent_loop_config.yaml
        - name: memory_agent
          _target_: agent_loop.MemoryAgentLoop
    Rows must carry a top-level "agent_name": "memory_agent" field (see
    build_verl_batch below) so verl routes them here instead of the default
    single_turn_agent loop.

    For each episode window:
      1. Model generates op for turn 0 (from initial prompt in the data row).
      2. Op applied to VirtualFilesystem.
      3. FS render + next turn injected as an observation (response_mask=0).
      4. Model generates op for turn 1. Repeat until window exhausted.
      5. Terminal reward = score_trajectory(FS, window_probes).
      6. Returns AgentLoopOutput: prompt_ids, response_ids, response_mask.
    """

    async def run(self, messages: list[dict], sampling_params: dict):
        if _AgentLoopBase is None:
            raise ImportError("verl not installed — MemoryAgentLoop requires verl v0.5.0")

        # verl tracks total tokens generated across all server_manager.generate() calls
        # and decrements sampling_params.max_new_tokens after each call. In a multi-turn
        # episode (10 conv turns + probes), this budget hits 0 and goes negative → ValueError.
        # Fix: create a per-call copy with a fresh max_new_tokens each time we generate.
        import copy as _copy
        def _sp_with_budget(budget: int):
            try:
                sp = _copy.copy(sampling_params)
                try:
                    sp["max_new_tokens"] = budget      # dict-like
                except TypeError:
                    sp.max_new_tokens = budget          # dataclass/object
                return sp
            except Exception:
                return sampling_params
        _CONV_BUDGET  = 512   # tokens per conv-turn response
        _PROBE_BUDGET = 512   # tokens per probe response

        # Extract episode data from the <__ep__> marker embedded in the system message.
        # verl v0.5.0 calls extra_info.strip() before json.loads() in generate_sequences(),
        # which crashes on a Python dict and silently returns {}. Embedding data in the
        # prompt bypasses this — messages IS correctly passed to run(). The marker is
        # stripped before any generate call so the model never sees it.
        import re as _re, base64 as _b64, gzip as _gz
        _ep_re = _re.compile(r'<__ep__>(.*?)</__ep__>', _re.DOTALL)
        extra = {}
        for _msg in messages:
            if _msg.get("role") == "system":
                _m = _ep_re.search(_msg.get("content", ""))
                if _m:
                    try:
                        _raw_bytes = _b64.b64decode(_m.group(1))
                        # Support both gzip-compressed (new) and plain (old) payloads
                        try:
                            extra = json.loads(_gz.decompress(_raw_bytes).decode())
                        except Exception:
                            extra = json.loads(_raw_bytes.decode())
                    except Exception:
                        pass
                break

        # Also check sampling_params["extra_info"] as fallback (in case verl fixes it).
        if not extra:
            _raw = sampling_params.get("extra_info", {})
            if isinstance(_raw, str):
                try:
                    extra = json.loads(_raw)
                except Exception:
                    extra = {}
            elif isinstance(_raw, dict):
                extra = _raw

        rest_turns   = extra.get("rest_turns", [])
        qa_probes    = extra.get("qa_probes", [])
        window_sids  = set(extra.get("window_session_ids", []))
        context1_url = extra.get("context1_url") or None
        sessions     = extra.get("sessions", [])

        # Strip the marker before any generate call.
        def _clean(msgs: list) -> list:
            return [
                {**m, "content": _ep_re.sub("", m["content"]).strip()}
                if m.get("role") == "system" else m
                for m in msgs
            ]

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

        # Auto-seeding removed: pre-populating VFS let the model answer probes from
        # seeded facts without ever writing valid ops (all INVALID, VFS pre-filled).
        # Now VFS starts empty — the model MUST write STORE_FACT/UPDATE/etc. during
        # conv turns to have anything to retrieve at probe time.

        all_prompt_ids   = []
        all_resp_ids     = []
        all_resp_mask    = []
        all_responses    = []
        turn_offset      = extra.get("first_turn", {}).get("turn_idx", 0)
        current_messages = _clean(list(messages))

        # ── Conversation turns phase ───────────────────────────────────────
        # Two guards prevent conv turns from consuming the entire response budget:
        #  1. _MAX_CONV_PROMPT: sglang computes max_new_tokens = (max_prompt_len +
        #     max_response_len) - len(prompt_ids); if prompt exceeds budget → negative.
        #  2. _MAX_CONV_RESP: probe phase needs room in all_resp_ids (probe markers +
        #     resolve ids) within the 4096 truncation limit. Capping conv tokens at ~1500
        #     guarantees probes fit even when model generates verbose conv responses.
        _MAX_CONV_PROMPT = 7680  # 8192 - 512 slack
        _MAX_CONV_RESP   = 1500  # lowered from 2048: ensures probe phase always has budget
        for step_idx in range(1 + len(rest_turns)):
            # AgentLoopBase doesn't expose a self.tokenize()/self.generate() convenience
            # API (verified against verl's own SingleTurnAgentLoop/ToolAgentLoop) — real
            # subclasses go through self.tokenizer / self.server_manager directly.
            step_prompt_ids = self.tokenizer.apply_chat_template(
                current_messages, add_generation_prompt=True, tokenize=True,
                enable_thinking=False,
            )
            if step_idx == 0:
                all_prompt_ids = step_prompt_ids
            elif len(step_prompt_ids) > _MAX_CONV_PROMPT or len(all_resp_ids) > _MAX_CONV_RESP:
                # Context window or response budget full — stop conv turns, preserve probe phase.
                break

            response_ids = await self.server_manager.generate(
                request_id=uuid.uuid4().hex,
                prompt_ids=step_prompt_ids,
                sampling_params=_sp_with_budget(_CONV_BUDGET),
            )
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
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
                ret_ids = self.tokenizer.encode(retrieve_obs, add_special_tokens=False)
                all_resp_ids  += ret_ids
                all_resp_mask += [0] * len(ret_ids)   # harness output, not trained on
                current_messages = current_messages + [
                    {"role": "assistant", "content": response_text},
                    {"role": "user",      "content": retrieve_obs},
                ]
            else:
                # All write ops (STORE_FACT, UPDATE, SUPERSEDE, COMPRESS, ABSTAIN)
                cur_turn = rest_turns[step_idx - 1] if step_idx > 0 else extra.get("first_turn", {})
                s_idx = cur_turn.get("session_idx", 0) if isinstance(cur_turn, dict) else 0
                harness.apply_op(op, session_idx=s_idx, turn_idx=turn_offset + step_idx)

                if step_idx < len(rest_turns):
                    next_turn   = rest_turns[step_idx]
                    observation = (
                        f"\n[Memory state]\n{harness.render_context()}\n\n"
                        f"[Next turn]\n{_format_turn(next_turn)}"
                    )
                    obs_ids = self.tokenizer.encode(observation, add_special_tokens=False)
                    all_resp_ids  += obs_ids
                    all_resp_mask += [0] * len(obs_ids)
                    current_messages = current_messages + [
                        {"role": "assistant", "content": response_text},
                        {"role": "user",      "content": observation},
                    ]

        # ── Probe / RESOLVE phase ──────────────────────────────────────────
        # Each probe adds ~80 (marker) + ~200 (resolve) = ~280 tokens. With up to
        # 23 probes per episode, that overflows the 4096 response budget, clips all
        # question markers out of solution_str, and makes rewards=0 for that batch.
        # Stop adding probes once we're within 400 tokens of the truncation limit.
        # Skip the guard for the FIRST probe to guarantee at least one probe score,
        # preventing probe_scores=[] → compute_reward()=0.0 for all rollouts → advantages=0.
        _MAX_PROBE_RESP = 3696   # _MAX_RESP(4096) - 400 slack; leave room for probe marker+resolve
        probe_count = 0
        for probe in qa_probes:
            if probe_count > 0 and len(all_resp_ids) >= _MAX_PROBE_RESP:
                break  # no room for another probe marker + resolve

            question = probe.get("question", "") if isinstance(probe, dict) else getattr(probe, "question", "")
            answer   = str(probe.get("answer", "") if isinstance(probe, dict) else getattr(probe, "answer", ""))
            if not answer or answer.lower() in ("abstain", "yes", "no", "n/a", ""):
                continue

            retrieved = retrieve_for_probe(question, harness.fs, context1_url)
            probe_obs = (
                f"\n[Memory state]\n{harness.render_context()}\n\n"
                f"[Retrieved]\n{retrieved or '(no relevant entries found)'}\n\n"
                f"[Question]\n{question}\n\n"
                f"[Instruction] Answer the question above using only what is in [Memory state] and [Retrieved]. "
                f"Output ONLY: {{\"op\": \"RESOLVE\", \"content\": \"<your answer>\"}}\n"
                f"Do NOT output STORE_FACT, UPDATE, RETRIEVE, or any other op. Just RESOLVE."
            )
            # Use a fresh 2-message context for probes (system + probe_obs only).
            # Including the full conversation history (current_messages) pushes
            # probe_prompt_ids > max_prompt_length + max_response_length, causing
            # verl to compute max_new_tokens = budget - len(prompt) < 0 → ValueError.
            # The memory FS state in probe_obs already captures all relevant info.
            clean_system = _ep_re.sub("", messages[0]["content"]).strip() if messages else _SYSTEM_PROMPT
            probe_messages   = [
                {"role": "system", "content": clean_system},
                {"role": "user",   "content": probe_obs},
            ]
            probe_prompt_ids = self.tokenizer.apply_chat_template(
                probe_messages, add_generation_prompt=True, tokenize=True,
                enable_thinking=False,
            )
            # reward.py extracts RESOLVE ops by position order (not by question anchor),
            # so no probe marker needed in all_resp_ids — just the model's resolve response.
            resolve_ids = await self.server_manager.generate(
                request_id=uuid.uuid4().hex,
                prompt_ids=probe_prompt_ids,
                sampling_params=_sp_with_budget(_PROBE_BUDGET),
            )
            resolve_text = self.tokenizer.decode(resolve_ids, skip_special_tokens=True)

            all_resp_ids  += resolve_ids
            all_resp_mask += [1] * len(resolve_ids)
            all_responses.append(resolve_text)

            resolve_op      = parse_op(resolve_text)
            resolve_content = (
                resolve_op.get("content", "")
                if resolve_op and resolve_op.get("op") == "RESOLVE"
                else resolve_text.strip()   # fallback: treat raw text as answer
            )
            harness.apply_resolve(resolve_content, answer, question=question)
            probe_count += 1

        # harness.summary() (below) computes harness.compute_reward() itself and logs
        # it to WandB as episode/reward — purely diagnostic. Confirmed against verl
        # v0.5.0 source (verl/experimental/agent_loop/agent_loop.py) and docs that
        # AgentLoopOutput has no reward field: reward for agent-loop rollouts is
        # always computed downstream by the reward manager calling
        # custom_reward_function (reward.py::compute_reward) on the decoded response
        # text, never returned from run(). reward.py independently replays this same
        summary = harness.summary()
        _log_episode_stats(all_responses, summary)

        # Embed the pre-computed reward as a short ASCII prefix at the FRONT of the
        # response so reward.py can read it without any text extraction or JSON parsing.
        # harness.compute_reward() uses probe_scores computed by apply_resolve() with
        # a raw-text fallback, so it's robust to non-JSON model outputs.
        # Format: "R:{bucket:03d};" where bucket = int(reward * 127 + 128), giving
        # range [-1.0, 1.0] mapped to [1, 255] (bucket 128 = reward 0.0).
        # Short fixed-width ASCII guarantees stable tokenization round-trips.
        episode_reward = harness.compute_reward()
        bucket = max(1, min(255, int(episode_reward * 127 + 128)))
        reward_prefix_str = f"R:{bucket:03d};"
        reward_prefix_ids = self.tokenizer.encode(reward_prefix_str, add_special_tokens=False)

        # Truncate episode to data.max_response_length (4096) — verl's _postprocess()
        # pads response_ids to exactly this length; any overage → shape mismatch.
        _MAX_RESP = 4096
        episode_ids  = all_resp_ids[:_MAX_RESP - len(reward_prefix_ids)]
        episode_mask = all_resp_mask[:_MAX_RESP - len(reward_prefix_ids)]
        all_resp_ids  = reward_prefix_ids + episode_ids
        all_resp_mask = [0] * len(reward_prefix_ids) + episode_mask

        # num_turns mirrors what verl's shipped SingleTurnAgentLoop/ToolAgentLoop both
        # populate (informational/logging — write+retrieve steps plus probe steps).
        num_turns = (1 + len(rest_turns)) + len(qa_probes)
        return _AgentLoopOutput(
            prompt_ids    = all_prompt_ids,
            response_ids  = all_resp_ids,
            response_mask = all_resp_mask,
            num_turns     = num_turns,
            metrics       = {
                "mean_resolve":  summary["mean_resolve"],
                "n_retrievals":  summary["n_retrievals"],
                "op_counts":     str(summary["op_counts"]),
            },
        )


# Register with verl's agent-loop registry so actor_rollout_ref.rollout.agent.agent_loop_config_path
# can resolve "memory_agent" -> this class. No-op (and harmless) when verl isn't installed locally.
if _register is not None:
    MemoryAgentLoop = _register("memory_agent")(MemoryAgentLoop)


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log_episode_stats(responses: list[str], summary: dict) -> None:
    """Log per-episode stats to WandB + stdout."""
    reward       = summary.get("final_reward", 0.0)
    probe_traces = summary.get("probe_traces", [])
    probe_scores = summary.get("probe_scores", [])
    op_counts    = summary.get("op_counts", {})
    op_sequence  = summary.get("op_sequence", [])
    fs_snapshot  = summary.get("fs_snapshot", "")
    n_ops        = max(sum(op_counts.values()), 1)

    # Always print a compact trace so Modal logs are inspectable
    _print_episode_trace(reward, probe_traces, op_sequence, fs_snapshot)

    try:
        import wandb
        if not wandb.run:
            return

        metrics: dict = {
            "episode/reward":        reward,
            "episode/mean_resolve":  summary.get("mean_resolve", 0.0),
            "episode/n_probes":      len(probe_scores),
            "episode/n_retrievals":  summary.get("n_retrievals", 0),
            "episode/n_fs_paths":    len(summary.get("fs_paths", [])),
            "episode/n_confirmed":   summary.get("n_confirmed", 0),
            "episode/n_tentative":   summary.get("n_tentative", 0),
            # op distribution
            "episode/store_rate":    op_counts.get("STORE_FACT", 0) / n_ops,
            "episode/update_rate":   (op_counts.get("UPDATE", 0) + op_counts.get("SUPERSEDE", 0)) / n_ops,
            "episode/abstain_rate":  op_counts.get("ABSTAIN", 0) / n_ops,
            "episode/retrieve_rate": op_counts.get("RETRIEVE", 0) / n_ops,
            "episode/invalid_rate":  op_counts.get("INVALID", 0) / n_ops,
        }
        # Individual probe scores p0..pN so we can see which probe positions are hard
        for i, s in enumerate(probe_scores):
            metrics[f"episode/probe_f1_p{i}"] = s

        wandb.log(metrics)

        # Log probe trace table for episodes with any non-zero score
        if probe_traces and any(t["f1"] > 0 for t in probe_traces):
            table = wandb.Table(columns=["question", "model_answer", "gold", "f1"])
            for t in probe_traces:
                table.add_data(t["question"], t["model_answer"], t["gold"], t["f1"])
            wandb.log({"episode/probe_table": table})

        # Log VFS snapshot for high-reward episodes
        if reward > 0.2 and fs_snapshot:
            wandb.log({"episode/vfs_snapshot": wandb.Html(f"<pre>{fs_snapshot}</pre>")})

    except Exception:
        pass


def _print_episode_trace(
    reward: float,
    probe_traces: list[dict],
    op_sequence: list[str],
    fs_snapshot: str,
) -> None:
    """Print a compact human-readable episode trace to stdout (visible in Modal logs)."""
    try:
        # Op sequence summary e.g. "STORE×3 ABSTAIN×2 INVALID×1"
        from collections import Counter as _Counter
        op_summary = "  ".join(
            f"{op}×{cnt}" for op, cnt in _Counter(op_sequence).most_common()
        ) or "(no ops)"

        lines = [
            f"[EP] reward={reward:.3f}  ops=[{op_summary}]  probes={len(probe_traces)}",
        ]
        for i, t in enumerate(probe_traces):
            lines.append(
                f"  P{i} f1={t['f1']:.3f} | Q: {t['question'][:80]} "
                f"| pred: {t['model_answer'][:60]} | gold: {t['gold'][:60]}"
            )
        if fs_snapshot and reward > 0.05:
            lines.append(f"  VFS: {fs_snapshot[:200].replace(chr(10), ' | ')}")
        print("\n".join(lines), flush=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_turn(turn: dict) -> str:
    role    = "Agent" if turn.get("role") == "assistant" else "User"
    content = turn.get("content", "").strip()
    return f"{role}: {content}"
