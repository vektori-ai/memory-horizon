"""Reward function — loaded by verl at training time.

verl config:
    custom_reward_function.path=/root/reward.py
    custom_reward_function.name=compute_reward

Why this file looks the way it does (re-derived from verl source + docs, not
assumed — see comments on replay_and_score below): verl v0.5.0's AgentLoopOutput
has no reward field at all (prompt_ids / response_ids / response_mask / num_turns
/ metrics only). For agent-loop rollouts, reward is ALWAYS computed downstream by
the reward manager, which decodes the rollout's response tokens into solution_str
and calls this function — MemoryAgentLoop.run() (agent_loop.py) cannot hand back
its own MemoryHarnessState.compute_reward() directly, no matter how it's wired.

So this file reconstructs the harness reward by replaying the same multi-turn
episode from the flat decoded transcript (replay_and_score), rather than acting
as a cheap separate format gate the way it used to when the (incorrect) plan was
for agent_loop.py to supply reward itself. The harness-level scoring logic
(MemoryHarnessState.compute_reward, score_write_against_ledger, etc.) is NOT
duplicated here — only the transcript-parsing/replay plumbing is new; the actual
scoring formula lives in exactly one place (harness_state.py), called from here.
"""

from __future__ import annotations

import json
import re

from agent_loop import _format_turn
from harness_state import MemoryHarnessState
from ledger import derive_ledger
from memory_fs import parse_op, retrieve_for_probe

# Literal markers agent_loop.py injects between model turns (see MemoryAgentLoop.run).
# These are used as anchors to recover where the model's own generated text starts/
# ends within the flat decoded solution_str — see replay_and_score for why a naive
# "just regex-find all {...} blobs" approach isn't reliable here (it silently
# misaligns whenever one step produces invalid/non-JSON output).
_RETRIEVE_MARKER_RE = re.compile(r"\[Retrieved for: (.*?)\]")


# ---------------------------------------------------------------------------
# Entry point verl calls
# ---------------------------------------------------------------------------

def compute_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> float:
    """Score one rollout. This IS the real training signal (see module docstring
    for why it can't come from MemoryAgentLoop.run() instead).

    ground_truth is the JSON-encoded list of QA probes for this window (set in
    agent_loop.build_verl_batch as reward_model.ground_truth). extra_info carries
    everything else replay_and_score needs (sessions, rest_turns, window_session_ids,
    context1_url, first_turn) — the same fields MemoryAgentLoop.run() reads.
    """
    try:
        reward = replay_and_score(solution_str, ground_truth, extra_info or {})
    except Exception as exc:
        # A malformed/unreplayable transcript should score as a bad rollout, not
        # crash the whole training step.
        _log_step(reward=None, error=str(exc))
        return -1.0

    _log_step(reward=reward, error=None)
    return reward


# ---------------------------------------------------------------------------
# Transcript replay — reconstruct MemoryHarnessState from the decoded rollout
# ---------------------------------------------------------------------------

def replay_and_score(solution_str: str, ground_truth: str, extra_info: dict) -> float:
    """Re-derive the same MemoryHarnessState.compute_reward() value MemoryAgentLoop.run()
    would have computed internally, from the flat decoded transcript alone.

    The hard part: solution_str is one long string containing the model's own
    generated JSON-op text interleaved with the harness's injected observation
    text ("[Memory state]...", "[Retrieved for: ...]...", "[Question]..."), with
    no delimiter marking where one ends and the other begins — the model's text
    just starts wherever the harness's f-string content happens to end.

    Naively scanning the whole string for `{...}`-shaped JSON blobs and assuming
    the i-th match corresponds to the i-th step breaks the moment any step
    produces zero matches (e.g. invalid/non-JSON output) — every later match
    silently shifts out of alignment with what step actually produced it.

    Instead: every harness-injected block is a pure function of state we already
    know (the exact conversation turns/questions are in extra_info; the FS render
    is whatever our own replay harness currently looks like, since we're rebuilding
    it in lockstep with run()'s own sequence). So at each step we recompute the
    EXACT text run() would have injected next, locate it in solution_str via
    str.find(), and whatever sits between our cursor and that position is exactly
    the model's generated text for this step — regardless of whether it parses as
    valid JSON or not.

    Returns whatever reward the harness reached if the transcript is truncated or
    a block can't be located (e.g. max_response_length cut the rollout short) —
    a truncated rollout naturally scores worse (fewer/no RESOLVE probes reached),
    which is the correct behavior, not a special case.
    """
    try:
        qa_probes = json.loads(ground_truth) if isinstance(ground_truth, str) else (ground_truth or [])
    except (json.JSONDecodeError, TypeError):
        qa_probes = []

    sessions     = extra_info.get("sessions", [])
    rest_turns   = extra_info.get("rest_turns", [])
    window_sids  = set(extra_info.get("window_session_ids", []))
    context1_url = extra_info.get("context1_url") or None

    # Same probe-filtering MemoryAgentLoop.run() applies before scoring.
    if window_sids:
        filtered = [
            p for p in qa_probes
            if not p.get("requires_sessions")
            or any(r in window_sids for r in p["requires_sessions"])
        ]
        if filtered:
            qa_probes = filtered

    ledger  = derive_ledger(qa_probes, sessions)
    harness = MemoryHarnessState(ledger=ledger)

    # Same auto-seed MemoryAgentLoop.run() applies before the turns phase.
    session_idx = extra_info.get("first_turn", {}).get("session_idx", 0)
    if session_idx > 0 and sessions:
        harness.seed_from_sessions(sessions, seed_session_count=session_idx)
    elif sessions:
        harness.seed_from_sessions(sessions, seed_session_count=1)

    turn_offset  = extra_info.get("first_turn", {}).get("turn_idx", 0)
    n_turn_steps = 1 + len(rest_turns)
    cursor       = 0

    # ── Turns phase — mirrors the for step_idx loop in MemoryAgentLoop.run() ──
    for step_idx in range(n_turn_steps):
        is_last_turn_step = step_idx == n_turn_steps - 1

        # Boundary-find using ONLY the constant marker prefixes ("\n[Retrieved
        # for: " / "\n[Memory state]\n"), never the full rendered block — the full
        # block (FS render text) depends on this step's op having already been
        # applied, which we don't know yet; that's exactly the bug this replaced
        # (caught by the round-trip test below, not by inspection: computing
        # render_context() *before* applying this step's own op silently produced
        # a block that never matched the transcript, since real run() renders
        # the observation *after* applying the op).
        #
        # "\n[Memory state]\n" is also the literal prefix of the probe phase's
        # block — within this turns-phase loop that's fine, since the probe phase
        # textually starts only once the loop is finished, EXCEPT on the very
        # last turn step: there, the first "\n[Memory state]\n" found is the
        # first probe's block, not a next-turn observation — handled below by
        # only treating it as a next-turn block when `not is_last_turn_step`.
        retrieve_pos  = solution_str.find("\n[Retrieved for: ", cursor)
        memory_pos    = solution_str.find("\n[Memory state]\n", cursor)
        candidates    = [p for p in (retrieve_pos, memory_pos) if p != -1]
        boundary      = min(candidates) if candidates else len(solution_str)

        model_chunk = solution_str[cursor:boundary].strip()
        op          = parse_op(model_chunk)
        op_type     = op.get("op", "INVALID") if op else "INVALID"

        if boundary == retrieve_pos and retrieve_pos != -1:
            m     = _RETRIEVE_MARKER_RE.match(solution_str[boundary + 1:])
            query = m.group(1) if m else op.get("query", "")
            # results is just as reconstructable as the next-turn block — it's a
            # pure function of (query, our own replayed harness.fs) — so compute
            # the exact block instead of guessing where it ends.
            results = retrieve_for_probe(query, harness.fs, context1_url)
            harness.apply_retrieve(query, [results] if results else [], turn_offset + step_idx)
            expected_retrieve_block = (
                f"\n[Retrieved for: {query}]\n{results or '(no relevant entries found)'}"
            )
            cursor = boundary + len(expected_retrieve_block)
        else:
            # Apply the op FIRST — render_context() below must reflect post-op
            # state to match what real run() actually injected.
            harness.apply_op(op, session_idx=session_idx, turn_idx=turn_offset + step_idx)

            if boundary == memory_pos and memory_pos != -1 and not is_last_turn_step:
                next_turn = rest_turns[step_idx]
                expected_next_turn_block = (
                    f"\n[Memory state]\n{harness.render_context()}\n\n"
                    f"[Next turn]\n{_format_turn(next_turn)}"
                )
                cursor = boundary + len(expected_next_turn_block)
            else:
                # Either no marker found (transcript ends here) or this
                # "\n[Memory state]\n" belongs to the probe phase, not us —
                # leave cursor at boundary so the probe loop picks up cleanly.
                cursor = boundary

        if cursor >= len(solution_str):
            break  # transcript truncated (e.g. max_response_length) — score what we have

    # ── Probe / RESOLVE phase — mirrors the for probe loop in run() ──────────
    for probe in qa_probes:
        question = probe.get("question", "")
        answer   = str(probe.get("answer", ""))
        if not answer or answer.lower() in ("abstain", "yes", "no", "n/a", ""):
            continue
        if cursor >= len(solution_str):
            break  # truncated before this probe was reached

        retrieved = retrieve_for_probe(question, harness.fs, context1_url)
        expected_probe_block = (
            f"\n[Memory state]\n{harness.render_context()}\n\n"
            f"[Retrieved]\n{retrieved or '(no relevant entries found)'}\n\n"
            f"[Question]\n{question}"
        )
        block_pos = solution_str.find(expected_probe_block, cursor)
        if block_pos == -1:
            break  # can't locate this probe's block — stop, score what's been built

        chunk_start    = block_pos + len(expected_probe_block)
        next_block_pos = solution_str.find("\n[Memory state]\n", chunk_start)
        chunk_end      = next_block_pos if next_block_pos != -1 else len(solution_str)
        resolve_text   = solution_str[chunk_start:chunk_end].strip()

        resolve_op      = parse_op(resolve_text)
        resolve_content = (
            resolve_op.get("content", "")
            if resolve_op and resolve_op.get("op") == "RESOLVE"
            else resolve_text   # fallback: treat raw text as the answer
        )
        harness.apply_resolve(resolve_content, answer)
        cursor = chunk_end

    return harness.compute_reward()


# ---------------------------------------------------------------------------
# WandB step logging
# ---------------------------------------------------------------------------

def _log_step(reward: float | None, error: str | None) -> None:
    try:
        import wandb
        if not wandb.run:
            return
        if error is not None:
            wandb.log({"step/replay_error": 1.0})
        else:
            wandb.log({"step/reward": reward, "step/replay_error": 0.0})
    except Exception:
        pass
