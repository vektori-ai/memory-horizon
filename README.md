# memory_horizon

RL environment infrastructure for training a memory manager model via GRPO. The model learns to decide what to store, how to resolve conflicts, when to abstain, and how to answer questions using only its stored memory.

## Overview

One training episode = one `Trajectory` (multiple conversation sessions + QA probes).

The model (memory manager) reads each conversation turn and emits a `MemoryOpAction` JSON. After all sessions, a frozen QA model answers probes using only the stored memory. Reward comes from two verifier tiers:

- **Tier 1** (weight 0.3): Deterministic checks (schema, key format, op legality, no-op detection). Zero LLM cost.
- **Tier 3** (weight 0.7): Outcome-based scoring via frozen QA model (Memory-R1 pattern), token F1, or Kendall's tau depending on the environment.

## Repository Layout

```
memory_horizon/
  core.py              # Env / EnvSpec base classes (Gymnasium v26 API)
  base_env.py          # MemoryHorizonEnv: the main training environment
  memory_store.py      # Three-layer memory store (fact / insight / raw)
  types.py             # All shared dataclasses and enums
  spaces/              # Action and observation space types
  wrappers/            # TimeLimit, RecordEpisode, generic Wrapper
  verifiers/
    tier1_deterministic.py  # Fast deterministic verifier
    tier2_llm_judge.py      # LLM judge (Base 5 only)
    tier3_outcome.py        # Token F1, Kendall tau, QA model scoring
    hallucination.py        # Hallucination detector
  generator/
    session_gen.py     # SessionGenerator, TrajectoryDataset
  environments/
    mh_base_store/     # Base 1: store facts
    mh_base_synthesize/ # Base 2: cross-session synthesis
    mh_base_contradict/ # Base 3: conflict resolution
    mh_base_compress/   # Base 4: memory compression
    mh_base_abstain/    # Base 5: know when not to answer
    mh_base_temporal/   # Base 6: temporal ordering
    mh_base_integration/ # Base 7: integration gate (all skills)
  training/
    curriculum.py      # CurriculumScheduler: Batch A -> B -> Integration -> Vertical
    frozen_qa_model.py # Frozen QA model wrapper
```

## Memory Operations

The model emits one of nine ops per turn:

| Family | Op | Description |
|---|---|---|
| Store | `STORE_FACT` | Atomic fact, addressable by key |
| Store | `CREATE_EPISODE` | Episodic narrative block |
| Store | `INFER_IMPLICIT` | Inference not literally stated |
| Conflict | `UPDATE` | Overwrite current value |
| Conflict | `SUPERSEDE` | Archive old value, write new |
| Conflict | `DECAY` | Reduce confidence without deleting |
| Conflict | `KEEP_BOTH` | Store both versions under distinct keys |
| Utility | `COMPRESS` | Lossy summarization of existing block |
| Utility | `ABSTAIN` | Explicit refusal when answer not in memory |

Action JSON format:
```json
{
  "op": "STORE_FACT",
  "content": "Customer is on the Premium Monthly plan",
  "key": "customer_plan",
  "confidence": 0.95,
  "layer": "fact"
}
```

## Memory Store

Three-layer architecture:

- **Fact layer**: Key-addressable atomic facts. Supports UPDATE, SUPERSEDE, DECAY, KEEP_BOTH.
- **Insight layer**: Cross-fact synthesis. Written by `INFER_IMPLICIT`.
- **Raw layer**: Verbatim turn log. Append-only. Used by the hallucination verifier.

```python
from memory_horizon import MemoryStore, MemoryOpAction, MemoryOp

store = MemoryStore(persist_across_sessions=True)
store.session_boundary("session_1")

action = MemoryOpAction(op=MemoryOp.STORE_FACT, content="User is John", key="user_name")
result = store.execute(action)

ctx = store.get_context()       # inject into system prompt
snap = store.snapshot()         # serialize state
store.session_boundary("session_2")
```

## Using an Environment

```python
from memory_horizon.environments.mh_base_store import make_store_env

env = make_store_env()
obs, info = env.reset(seed=42)

while True:
    action = '{"op": "STORE_FACT", "content": "...", "key": "some_key"}'
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

**Episode loop:**

```
reset()  -> obs(Session 1, Turn 0)
step()   -> obs(Session 1, Turn 1), reward=tier1_score
...
step()   -> obs(Session 2, Turn 0)   <- session boundary
...
step()   -> obs(QA probe 0)          <- qa phase
step()   -> final reward, done=True
```

**Observation dict keys:** `phase`, `session_id`, `session_idx`, `turn_idx`, `role`, `content`, `memory_context`, `system_prompt`, `qa_idx`, `total_turns`.

## Reward

```
reward = clip(0.3 * tier1_avg + 0.7 * tier3_avg, -2, +1)
```

Tier 1 per-step penalties:
- `-1.0` unparseable action
- `-0.5` no-op UPDATE (value unchanged)
- `-2.0` SUPERSEDE without audit trail
- `-2.0` KEEP_BOTH without distinct keys
- `-1.0` COMPRESS by less than 20% or more than 95%

## Training Curriculum

`CurriculumScheduler` enforces the training sequence and kill criteria:

| Phase | Environments | Advance threshold |
|---|---|---|
| Batch A (weeks 1-6) | temporal, contradict, compress | 0.70 |
| Batch B (weeks 7-9) | store, synthesize, abstain | 0.70 |
| Integration (week 10) | integration | 0.90 |
| Vertical V1+ | voice_customer, support_technical, ... | 0.75 |

```python
from memory_horizon.training.curriculum import CurriculumScheduler

scheduler = CurriculumScheduler(state_path="runs/curriculum.json")
stage = scheduler.current_stage()

decision = scheduler.record_run(env="temporal", eval_score=0.45)
if decision.should_kill:
    print(decision.reason)   # switch to GEPA fallback

if scheduler.try_advance_stage():
    print("Advanced to next stage")
```

Kill criteria are env-specific:
- **contradict**: 3 consecutive runs where `UPDATE` is the dominant op
- **compress**: compression ratio > 0.95 (no-compression collapse)
- **abstain**: hallucination rate > 10% after 8 runs
- **temporal**: held-out eval < 30% after 8 runs
- **any env**: avg score < threshold after `max_runs_before_kill` runs

GEPA (prompt-level evolutionary optimization) is the fallback when RL fails to converge.

## Data Generation

```python
from memory_horizon.generator.session_gen import SessionGenerator, TrajectoryDataset

# Built-in seeds
gen = SessionGenerator.from_config("voice_customer")
trajectory = gen.generate()

# Generate a JSONL dataset
gen.generate_dataset(n=1000, output_path="data/voice_customer.jsonl")

# Load pre-generated data during training
dataset = TrajectoryDataset("data/voice_customer.jsonl", shuffle=True)
env = make_store_env(trajectory_fn=dataset.next)
```

Custom seeds can be loaded from JSONL via `SessionGenerator.from_jsonl()`. Each seed line specifies `facts`, `qa_probes`, `n_sessions`, `vertical`, and optional `conflict_pairs`.

## Verifier Tiers

**Tier 1** (`verifiers/tier1_deterministic.py`): Zero-cost checks run on every step. Validates JSON schema, op legality per environment, key format (`^[a-z][a-z0-9_]*$`), content not being a verbatim transcript copy.

**Tier 2** (`verifiers/tier2_llm_judge.py`): LLM judge, used only for Base 5 (Abstain).

**Tier 3** (`verifiers/tier3_outcome.py`):
- Base 1 / Base 7: Frozen QA model answers probes using stored context. `+1` if F1 >= 0.5.
- Base 2 (Synthesize): Token F1 against gold answer.
- Base 4 (Compress): QA accuracy delta before/after compression.
- Base 6 (Temporal): Kendall's tau on event ordering.
- Base 7 (Integration): Weighted composite across all six component scores.

## Installation

```bash
pip install -e .
```

Requires Python 3.10+. The frozen QA model (Tier 3) is loaded lazily and can be replaced with any `Callable[[str, str], str]` that takes `(question, context)` and returns an answer string.
