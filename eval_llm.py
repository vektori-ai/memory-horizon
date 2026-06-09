"""eval_llm.py — Baseline eval: run a frontier LLM as the memory manager agent.

Usage:
    python eval_llm.py                          # 20 trajectories, claude-opus-4-7
    python eval_llm.py --n 50 --model sonnet    # 50 trajectories, Sonnet
    python eval_llm.py --data data/custom.jsonl # custom dataset
    python eval_llm.py --n 0                    # all trajectories

What this measures:
    Tier 1 (format / op-selection compliance, per-turn avg)
    Tier 3 (QA recall: did the model store facts that answer the probes?)
    Episode reward (0.3 * T1 + 0.7 * T3, clipped [-2, +1])

Fine-tuning signal interpretation:
    Episode reward >= 0.85 → LLM already near ceiling; fine-tuning unlikely to help
    Episode reward 0.5-0.85 → real headroom; fine-tuning on this task likely helps
    Episode reward < 0.5   → model struggles with format/recall; strong fine-tune signal
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

# Adjust sys.path so we can import from the package regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent))

from memory_horizon.base_env import MemoryHorizonEnv
from memory_horizon.mh_types import (
    QAPair,
    Session,
    Trajectory,
    Turn,
)


# ---------------------------------------------------------------------------
# Model aliases
# ---------------------------------------------------------------------------

MODEL_ALIASES: dict[str, str] = {
    "opus":    "claude-opus-4-7",
    "sonnet":  "claude-sonnet-4-6",
    "haiku":   "claude-haiku-4-5-20251001",
    # Pass full model ID directly too.
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TrajResult:
    trajectory_id: str
    vertical: str
    n_turns: int
    n_qa_probes: int
    tier1_avg: float
    tier3_avg: float
    episode_reward: float
    parse_error_rate: float
    tier1_scores: list[float] = field(default_factory=list)
    tier3_scores: list[float] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    llm_calls: int = 0
    latency_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "vertical": self.vertical,
            "n_turns": self.n_turns,
            "n_qa_probes": self.n_qa_probes,
            "tier1_avg": round(self.tier1_avg, 3),
            "tier3_avg": round(self.tier3_avg, 3),
            "episode_reward": round(self.episode_reward, 3),
            "parse_error_rate": round(self.parse_error_rate, 3),
            "llm_calls": self.llm_calls,
            "latency_s": round(self.latency_s, 1),
        }


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------

def call_llm(
    client: anthropic.Anthropic,
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int = 256,
) -> str:
    """Call the LLM and return raw text response."""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# Trajectory loader
# ---------------------------------------------------------------------------

def load_trajectories(jsonl_path: str) -> list[Trajectory]:
    """Load Trajectory objects from a JSONL file."""
    trajectories = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            traj = _dict_to_trajectory(raw)
            if traj is not None:
                trajectories.append(traj)
    return trajectories


def _dict_to_trajectory(d: dict[str, Any]) -> Trajectory | None:
    try:
        sessions = []
        for s in d.get("sessions", []):
            turns = [
                Turn(
                    role=t["role"],
                    content=t["content"],
                    timestamp=t.get("timestamp"),
                    metadata=t.get("metadata", {}),
                )
                for t in s.get("turns", [])
            ]
            sessions.append(Session(
                session_id=s["session_id"],
                turns=turns,
                memory_snapshot=s.get("memory_snapshot"),
                metadata=s.get("metadata", {}),
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
            trajectory_id=d["trajectory_id"],
            sessions=sessions,
            qa_probes=qa_probes,
            oracle_memory=d.get("oracle_memory", {}),
            vertical=d.get("vertical", "unknown"),
            metadata=d.get("metadata", {}),
        )
    except (KeyError, TypeError) as e:
        print(f"[WARN] Failed to parse trajectory: {e}")
        return None


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    client: anthropic.Anthropic,
    model: str,
    trajectory: Trajectory,
    verbose: bool = False,
) -> TrajResult:
    """Run one trajectory through the env with the LLM as the agent."""
    traj_iter = iter([trajectory])

    def traj_fn() -> Trajectory:
        return next(traj_iter)

    env = MemoryHorizonEnv(
        env_type=trajectory.metadata.get("disposition", "store").lower(),
        trajectory_fn=traj_fn,
        reward_per_turn=True,
    )

    obs, info = env.reset()
    done = False
    llm_calls = 0
    parse_errors = 0
    t0 = time.time()
    actions_taken: list[str] = []

    while not done:
        system_prompt = obs.get("system_prompt", "")
        phase = obs.get("phase", "session")
        content = obs.get("content", "")

        if phase == "done":
            break

        if phase == "session":
            user_msg = (
                f"Conversation turn ({obs.get('role', 'user')}):\n{content}\n\n"
                "Output your memory operation as JSON:"
            )
            raw_action = call_llm(client, model, system_prompt, user_msg, max_tokens=200)
            llm_calls += 1
            actions_taken.append(raw_action[:80])

            if verbose:
                print(f"  [turn {obs.get('turn_idx')}] {content[:60]} → {raw_action[:60]}")

        else:  # qa phase
            # For QA phase: ask LLM to answer using the memory context in the system prompt.
            # Score is based on stored memory, not the answer — but we capture it for analysis.
            memory_ctx = obs.get("memory_context", "")
            user_msg = (
                f"Question: {content}\n\n"
                f"Using only the memory context provided, answer this question.\n"
                f"Format: {{\"answer\": \"your answer here\"}}"
            )
            raw_action = call_llm(client, model, system_prompt, user_msg, max_tokens=150)
            llm_calls += 1

        obs, reward, terminated, truncated, step_info = env.step(raw_action)

        if step_info.get("parse_error"):
            parse_errors += 1

        done = terminated or truncated

    latency = time.time() - t0

    # Extract final summary from env state.
    s = env._state
    t1_scores = s.tier1_scores if s else []
    t3_scores = s.tier3_scores if s else []
    t1_avg = sum(t1_scores) / max(len(t1_scores), 1)
    t3_avg = sum(t3_scores) / max(len(t3_scores), 1)
    episode_reward = max(-2.0, min(1.0, 0.3 * t1_avg + 0.7 * t3_avg))
    n_turns = len(t1_scores)

    return TrajResult(
        trajectory_id=trajectory.trajectory_id,
        vertical=trajectory.vertical,
        n_turns=n_turns,
        n_qa_probes=len(trajectory.qa_probes),
        tier1_avg=t1_avg,
        tier3_avg=t3_avg,
        episode_reward=episode_reward,
        parse_error_rate=parse_errors / max(n_turns, 1),
        tier1_scores=t1_scores,
        tier3_scores=t3_scores,
        actions_taken=actions_taken,
        llm_calls=llm_calls,
        latency_s=latency,
    )


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------

def aggregate(results: list[TrajResult]) -> dict[str, Any]:
    if not results:
        return {}

    def avg(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    t1_avgs = [r.tier1_avg for r in results]
    t3_avgs = [r.tier3_avg for r in results]
    rewards = [r.episode_reward for r in results]
    parse_err_rates = [r.parse_error_rate for r in results]

    reward_avg = avg(rewards)

    if reward_avg >= 0.85:
        finetuning_signal = "LOW — model near ceiling; fine-tuning unlikely to help much"
    elif reward_avg >= 0.5:
        finetuning_signal = "MEDIUM — real headroom; fine-tuning on this task should help"
    else:
        finetuning_signal = "HIGH — model struggles; strong case for fine-tuning"

    return {
        "n_trajectories": len(results),
        "tier1_avg": round(avg(t1_avgs), 3),
        "tier3_avg": round(avg(t3_avgs), 3),
        "episode_reward_avg": round(reward_avg, 3),
        "episode_reward_min": round(min(rewards), 3),
        "episode_reward_max": round(max(rewards), 3),
        "parse_error_rate_avg": round(avg(parse_err_rates), 3),
        "total_llm_calls": sum(r.llm_calls for r in results),
        "total_latency_s": round(sum(r.latency_s for r in results), 1),
        "finetuning_signal": finetuning_signal,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline LLM eval on memory_horizon envs")
    parser.add_argument("--data",    default="data/debt_collection.jsonl", help="JSONL trajectory file")
    parser.add_argument("--n",       type=int, default=20, help="Trajectories to eval (0 = all)")
    parser.add_argument("--model",   default="opus", help="opus | sonnet | haiku | full model ID")
    parser.add_argument("--out",     default=None, help="Write per-trajectory results to JSON")
    parser.add_argument("--verbose", action="store_true", help="Print per-turn actions")
    args = parser.parse_args()

    model = MODEL_ALIASES.get(args.model, args.model)
    print(f"\n{'='*60}")
    print(f"  memory_horizon LLM Baseline Eval")
    print(f"  Model:  {model}")
    print(f"  Data:   {args.data}")
    print(f"{'='*60}\n")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    trajectories = load_trajectories(args.data)
    print(f"Loaded {len(trajectories)} trajectories from {args.data}")

    if args.n > 0:
        trajectories = trajectories[:args.n]
    print(f"Running eval on {len(trajectories)} trajectories...\n")

    results: list[TrajResult] = []

    for i, traj in enumerate(trajectories, 1):
        try:
            result = run_episode(client, model, traj, verbose=args.verbose)
            results.append(result)
            print(
                f"[{i:3d}/{len(trajectories)}] "
                f"T1={result.tier1_avg:+.2f}  "
                f"T3={result.tier3_avg:.2f}  "
                f"R={result.episode_reward:+.2f}  "
                f"parse_err={result.parse_error_rate:.0%}  "
                f"turns={result.n_turns}  "
                f"qa={result.n_qa_probes}  "
                f"({result.latency_s:.1f}s)"
            )
        except Exception as e:
            print(f"[{i:3d}/{len(trajectories)}] ERROR: {e}")

    print(f"\n{'='*60}")
    print("  AGGREGATE RESULTS")
    print(f"{'='*60}")
    agg = aggregate(results)
    for k, v in agg.items():
        print(f"  {k:<30s} {v}")
    print(f"{'='*60}\n")

    if args.out:
        out_data = {
            "model": model,
            "data": args.data,
            "aggregate": agg,
            "trajectories": [r.to_dict() for r in results],
        }
        Path(args.out).write_text(json.dumps(out_data, indent=2))
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
