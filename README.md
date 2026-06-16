# memory_horizon

RL training infrastructure for a memory-manager model trained via GRPO (verl v0.5.0 +
SGLang) on multi-session conversations. The model reads conversation turns one at a
time and decides what to store, update, supersede, compress, or abstain from — then
answers held-out QA probes using only what it chose to keep in its own memory store.

Goal: a small model (Qwen3-8B base) that's fast enough to manage memory for live
voice/support conversations, not a general-purpose large LLM.

## Repository layout

```
mh_types.py          Trajectory/Session/Turn/QAPair dataclasses — JSONL-serializable
                      data model used by the converters. (Carries some legacy
                      MemoryOp/MemoryLayer members unused by the live pipeline —
                      see the LEGACY NOTE at the top of the file.)
memory_fs.py          VirtualFilesystem — the agent's memory store: category/entity
                      paths, each holding an ordered list of tagged content lines
                      with an importance tag (tentative/confirmed/superseded).
ledger.py             Proxy ledger — derives ground-truth fact states from QA probe
                      gold answers (Phase A; a real partner ledger is the Phase B
                      plan). Used only to compute reward, never seen by the agent.
harness_state.py      MemoryHarnessState — per-episode state (FS, op counts, step
                      rewards, probe scores) and compute_reward(): the actual reward
                      formula (resolve-F1 + step-ledger shaping + diversity bonus +
                      op penalty + abstain penalty).
agent_loop.py          build_verl_batch(): slices trajectories into K-turn episode
                      windows, raw_chat format for verl.
                       MemoryAgentLoop: verl AgentLoopBase rollout — runs the
                      conversation-turn + RETRIEVE + RESOLVE loop per episode window.
agent_loop_config.yaml verl's agent-loop registry file (name -> class), loaded via
                      actor_rollout_ref.rollout.agent.agent_loop_config_path.
reward.py             verl's custom_reward_function entrypoint. AgentLoopOutput has
                      no reward field at v0.5.0 — reward is always computed
                      downstream from the decoded rollout text, so this file
                      replays the transcript to recompute the same harness reward
                      MemoryAgentLoop.run() builds internally (see its docstring).
context1_service.py  Optional Modal-hosted retrieval service (chromadb/context-1)
                      used by RETRIEVE ops during training; falls back to keyword
                      grep when CONTEXT1_SERVICE_URL is unset.
patch_verl.py         Source patches applied to verl inside the Modal image build:
                      empty_cache before optimizer step, and step-wise GRPO
                      advantage broadcast (propagates terminal reward to earlier
                      memory-op turns, not just the final RESOLVE turn).
train_modal.py        Modal orchestration: data prep, SFT warm-up, GRPO training,
                      eval, baseline. See "Running" below.
eval_local.py         Local (no GPU) reward-signal validator — stub policies
                      (invalid/random/oracle) against the actual reward formula,
                      to sanity-check reward shape before spending on GPU time.
data/
  download_data.py   Fetches LoCoMo + LongMemEval from HuggingFace.
  converters/        LoCoMo / LongMemEval -> Trajectory JSONL converters.
  gen_sft.py          Generates SFT demonstration traces via GPT-OSS-120B.
```

## Memory operations

The model emits one JSON object per turn, one of:

| Op | Effect |
|---|---|
| `STORE_FACT` | append, tagged `tentative` |
| `UPDATE` | append, promotes path to `confirmed` |
| `SUPERSEDE` | archives prior content under `<path>_prior` (tag `superseded`), writes new value as `confirmed` |
| `COMPRESS` | replaces path with a summary, tagged `confirmed` |
| `ABSTAIN` | no-op, logged |
| `RETRIEVE` | harness-executed: searches the agent's own FS (Context-1 or grep fallback), result injected as an observation |
| `RESOLVE` | only during the probe phase — the agent's answer to a QA probe |

## Reward

Two layers, computed in different places (see `reward.py`'s module docstring for
why): a cheap per-step format signal folded into the replay, and the real signal —
`MemoryHarnessState.compute_reward()`:

```
reward = RESOLVE_WEIGHT * mean(probe_scores)        # token-F1 against gold answers
       + STEP_WEIGHT     * mean(step_rewards)        # ledger-shaped per-write reward
       + diversity_bonus                             # smooth ramp toward distinct op types used
       - op_penalty                                   # linear ramp after too many ops
       + abstain_penalty                              # if probes existed but none were resolved
```

Weights and thresholds are env vars in `harness_state.py` (`RESOLVE_WEIGHT`,
`STEP_REWARD_WEIGHT`, `DIVERSITY_BONUS`, `DIVERSITY_TARGET`, etc.).

## Running

```bash
pip install -e .                                    # base deps
python data/download_data.py                        # fetch LoCoMo + LongMemEval

# Optional SFT warm-up (teaches JSON-op format before GRPO)
export OPENAI_API_KEY=sk-...
modal run train_modal.py::run_sft
modal run train_modal.py::train_only --base-model-path <merged checkpoint from above>

# Or skip SFT and train straight from base Qwen3-8B
modal run train_modal.py                             # prep data + train
modal run train_modal.py::sanity                     # 20-step smoke test (~$7)
modal run train_modal.py::diagnose                    # local reward-coverage check, no GPU
python eval_local.py                                  # local reward-signal validator, no GPU
```

`context1_service.py` is optional — deploy it separately (`modal deploy
context1_service.py`) and set `CONTEXT1_SERVICE_URL` in `train_modal.py` to use it
for RETRIEVE; otherwise training falls back to keyword grep automatically.

## Requires

Python 3.10+. Training (`modal run train_modal.py::*`) needs the `training` extra
(`torch`, `transformers`, `trl`, `accelerate`, `datasets`) and runs on Modal, not
locally — verl + SGLang are installed inside the Modal image, not in this repo's
`pyproject.toml`.
