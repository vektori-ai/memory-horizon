"""Local reward-signal validator — runs BEFORE any GPU training.

Answers three questions on a small LoCoMo slice (no model needed):
  1. What fraction of random/greedy ops are valid JSON? (format floor)
  2. What is the reward distribution under a naive policy?
  3. Does the diversity bonus ever fire? Is step reward non-zero?

Also runs a rule-based oracle policy (always writes a gold-adjacent fact)
to show what the reward ceiling looks like.

Usage:
    python eval_local.py                         # uses data/locomo10.json
    python eval_local.py --n 5 --turns 3         # 5 trajectories, 3 turns each
    python eval_local.py --model qwen3           # run actual model (needs GPU)
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from agent_loop import build_verl_batch
from harness_state import MemoryHarnessState
from ledger import derive_ledger
from memory_fs import parse_op, retrieve_for_probe


# ---------------------------------------------------------------------------
# Stub policies (no GPU needed)
# ---------------------------------------------------------------------------

def _random_policy(turn_text: str, fs_render: str) -> str:
    """Emit a random valid op — simulates untrained model floor."""
    r = random.random()
    if r < 0.3:
        return '{"op": "ABSTAIN"}'
    elif r < 0.6:
        cats = ["people", "events", "facts", "prefs", "places"]
        return json.dumps({
            "op": "STORE_FACT",
            "path": f"{random.choice(cats)}/entity_{random.randint(1,5)}",
            "content": turn_text[:80],
        })
    elif r < 0.8:
        return json.dumps({
            "op": "UPDATE",
            "path": "facts/general",
            "content": turn_text[:80],
        })
    else:
        return '{"op": "RETRIEVE", "query": "recent facts"}'


def _invalid_policy(turn_text: str, fs_render: str) -> str:
    """Emit garbage — simulates totally broken output."""
    return "I don't know what to do here, the memory is " + turn_text[:20]


def _oracle_policy(turn_text: str, fs_render: str, gold_answers: list[str]) -> str:
    """Emit a STORE_FACT that contains a gold answer — reward ceiling."""
    if gold_answers:
        gold = random.choice(gold_answers)
        return json.dumps({
            "op": "STORE_FACT",
            "path": "facts/gold",
            "content": f"{turn_text[:40]} → {gold}",
        })
    return '{"op": "ABSTAIN"}'


def _resolve_policy(question: str, fs_render: str, gold_answers: list[str]) -> str:
    """Always RESOLVE with a gold answer — shows what perfect RESOLVE gives."""
    gold = gold_answers[0] if gold_answers else "unknown"
    return json.dumps({"op": "RESOLVE", "content": gold})


# ---------------------------------------------------------------------------
# Run one episode window through the harness (no verl, no GPU)
# ---------------------------------------------------------------------------

def run_episode(row: dict, policy_fn, resolve_fn=None) -> dict:
    extra      = row["extra_info"]
    rest_turns = extra["rest_turns"]
    qa_probes  = extra["qa_probes"]
    sessions   = extra.get("sessions", [])

    ledger  = derive_ledger(qa_probes, sessions)
    harness = MemoryHarnessState(ledger=ledger)

    # Auto-seed from prior sessions (same as agent_loop.py)
    session_idx = extra.get("first_turn", {}).get("session_idx", 0)
    if session_idx > 0 and sessions:
        harness.seed_from_sessions(sessions, seed_session_count=session_idx)
    elif sessions:
        harness.seed_from_sessions(sessions, seed_session_count=1)

    gold_answers = [str(p.get("answer", "")) for p in qa_probes if p.get("answer")]

    # Conversation turns
    first_turn = extra["first_turn"]
    all_turns  = [first_turn] + rest_turns
    for step, turn in enumerate(all_turns):
        content  = turn.get("content", "")
        fs_state = harness.render_context()
        raw      = policy_fn(content, fs_state)
        op       = parse_op(raw)
        if not op:
            op = {}
        op_type = op.get("op", "INVALID")
        if op_type == "RETRIEVE":
            query   = op.get("query", content[:40])
            results = retrieve_for_probe(query, harness.fs)
            harness.apply_retrieve(query, [results], step)
        else:
            harness.apply_op(op, session_idx=session_idx, turn_idx=step)

    # Probe / RESOLVE phase
    for probe in qa_probes:
        question = probe.get("question", "")
        answer   = str(probe.get("answer", ""))
        if not answer or answer.lower() in ("abstain", "yes", "no", "n/a", ""):
            continue

        retrieved = retrieve_for_probe(question, harness.fs)
        if resolve_fn:
            raw_resolve = resolve_fn(question, retrieved, gold_answers)
        else:
            raw_resolve = policy_fn(question, retrieved)

        resolve_op = parse_op(raw_resolve)
        if resolve_op and resolve_op.get("op") == "RESOLVE":
            resolve_content = resolve_op.get("content", "")
        else:
            resolve_content = raw_resolve.strip()[:200]
        harness.apply_resolve(resolve_content, answer)

    reward  = harness.compute_reward()
    summary = harness.summary()
    # mean_step/distinct_ops/diversity_bonus were dropped from compute_reward()
    # (harness_state.py: constant-per-group terms, zero GRPO gradient, just
    # noise) — current reward is mean(probe_scores) + format_penalty only.
    return {
        "reward":        reward,
        "mean_resolve":  summary["mean_resolve"],
        "n_confirmed":   summary["n_confirmed"],
        "n_tentative":   summary["n_tentative"],
        "op_counts":     summary["op_counts"],
        "n_probes":      len(harness.probe_scores),
        "n_paths":       len(harness.fs.files),
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _fmt(vals: list[float]) -> str:
    if not vals:
        return "  n/a"
    return f"  mean={mean(vals):.3f}  std={stdev(vals):.3f}  min={min(vals):.3f}  max={max(vals):.3f}"


def _report(label: str, results: list[dict]) -> None:
    rewards  = [r["reward"] for r in results]
    resolves = [r["mean_resolve"] for r in results]
    n_probes = mean(r["n_probes"] for r in results) if results else 0

    print(f"\n{'─'*55}")
    print(f"  Policy: {label}  ({len(results)} episodes)")
    print(f"{'─'*55}")
    print(f"  reward:        {_fmt(rewards)}")
    print(f"  resolve F1:    {_fmt(resolves)}")
    print(f"  mean probes/ep: {n_probes:.1f}")

    # Op type breakdown
    op_totals: dict[str, int] = {}
    for r in results:
        for k, v in r["op_counts"].items():
            op_totals[k] = op_totals.get(k, 0) + v
    total_ops = max(sum(op_totals.values()), 1)
    print(f"  op breakdown:  " + "  ".join(
        f"{k}={v/total_ops*100:.0f}%" for k, v in sorted(op_totals.items())
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",   default="data/locomo_train.jsonl")
    parser.add_argument("--n",     type=int, default=8,  help="trajectories to eval")
    parser.add_argument("--turns", type=int, default=5,  help="episode window size")
    args = parser.parse_args()

    # --src must be a converted Trajectory JSONL (sessions/qa_probes schema) —
    # raw locomo10.json (conversation/qa schema) is NOT what build_verl_batch
    # expects; run data/download_data.py first to produce the converted file.
    src = ROOT / args.src
    if src.exists():
        with open(src) as f:
            trajs = [json.loads(line) for line in f if line.strip()]
    else:
        print(f"[error] No data found at {src} — run: python3 data/download_data.py first")
        sys.exit(1)
    rows  = build_verl_batch(trajs, window_size=args.turns, stride=3)
    if not rows:
        print("[error] build_verl_batch returned 0 rows — check data format")
        sys.exit(1)

    sample = rows[:args.n]
    print(f"\nLoaded {len(rows)} episode windows from {src.name}, evaluating {len(sample)}")
    print(f"Window size={args.turns} turns, {sum(len(r['extra_info']['qa_probes']) for r in sample)} total probes")

    # --- Policy 1: invalid (garbage output) ---
    invalid_results = [run_episode(r, _invalid_policy) for r in sample]
    _report("INVALID (format failure floor)", invalid_results)

    # --- Policy 2: random valid ops ---
    random.seed(42)
    random_results = [run_episode(r, _random_policy) for r in sample]
    _report("RANDOM VALID OPS", random_results)

    # --- Policy 3: oracle writes (gold facts stored) + random RESOLVE ---
    def oracle_write(turn, fs):
        all_gold = []
        return _oracle_policy(turn, fs, all_gold)

    def make_oracle_write(row):
        probes = row["extra_info"]["qa_probes"]
        gold   = [str(p.get("answer","")) for p in probes if p.get("answer")]
        return lambda turn, fs: _oracle_policy(turn, fs, gold)

    oracle_results = [
        run_episode(r, make_oracle_write(r), resolve_fn=lambda q, fs, g: _resolve_policy(q, fs, g))
        for r in sample
    ]
    _report("ORACLE (gold facts written + perfect RESOLVE)", oracle_results)

    # --- Reward gap summary ---
    print(f"\n{'═'*55}")
    print(f"  REWARD GAP SUMMARY")
    print(f"{'═'*55}")
    r_invalid = mean(r["reward"] for r in invalid_results)
    r_random  = mean(r["reward"] for r in random_results)
    r_oracle  = mean(r["reward"] for r in oracle_results)
    print(f"  invalid policy:  {r_invalid:.3f}  (format floor)")
    print(f"  random policy:   {r_random:.3f}  (untrained model proxy)")
    print(f"  oracle policy:   {r_oracle:.3f}  (ceiling)")
    gap = r_oracle - r_random
    print(f"  learnable gap:   {gap:.3f}  {'✓ gradient signal exists' if gap > 0.05 else '✗ gap too small — reward needs tuning'}")
    print()

    # --- Auto-seed quality check ---
    print(f"  AUTO-SEED CHECK (importance tag distribution):")
    conf = mean(r["n_confirmed"] for r in random_results)
    tent = mean(r["n_tentative"] for r in random_results)
    print(f"    confirmed paths after episode: {conf:.1f}")
    print(f"    tentative paths after seed:    {tent:.1f}")
    if tent > 15:
        print(f"    ⚠ seeding too noisy — {tent:.0f} tentative paths (aim for ≤8)")
    else:
        print(f"    ✓ seed volume looks reasonable")


if __name__ == "__main__":
    main()
