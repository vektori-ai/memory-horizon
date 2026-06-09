"""GRPO memory training on Modal — Qwen3-8B with verl + vLLM.

Run locally:
    modal run train_modal.py            # prep data + train
    modal run train_modal.py::prep      # data prep only
    modal run train_modal.py::train     # train only (after data is ready)
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT            = Path(__file__).parent
DATA_PATH_LOCAL      = REPO_ROOT / "data" / "train.jsonl"
TEST_PATHS_LOCAL     = {
    "locomo":       REPO_ROOT / "data" / "locomo_test.jsonl",
    "longmemeval":  REPO_ROOT / "data" / "longmemeval_test.jsonl",
}

MODEL_ID    = "Qwen/Qwen3-8B"
N_ROLLOUTS  = 4      # completions per prompt (GRPO group size)
N_STEPS     = 200

VERL_REPO_PATH = Path("/root/verl")
DATA_PATH      = Path("/data")
MODELS_PATH    = Path("/models")
MINUTES        = 60

# verl loads compute_reward from reward.py at runtime
PATH_TO_REWARD_FUNCTION = Path("/root/reward.py")
REWARD_FUNCTION_NAME    = "compute_reward"

# ---------------------------------------------------------------------------
# Modal image — verl base image with vLLM already baked in
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry("verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.1")
    .apt_install("git")
    .run_commands(f"git clone https://github.com/volcengine/verl {VERL_REPO_PATH}")
    .uv_pip_install("verl[vllm]==0.4.1", "pandas", "pyarrow")
    .add_local_file(Path(__file__).parent / "patch_verl.py", "/root/patch_verl.py", copy=True)
    .add_local_file(Path(__file__).parent / "reward.py",     "/root/reward.py",     copy=True)
    .run_commands("python /root/patch_verl.py")
    .env({
        "HF_HOME": "/hf-cache",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,garbage_collection_threshold:0.8",
    })
)

app = modal.App("memory-rlvr")

data_volume        = modal.Volume.from_name("memory-rlvr-data",        create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("memory-rlvr-checkpoints", create_if_missing=True)
hf_cache_vol       = modal.Volume.from_name("hf-model-cache",          create_if_missing=True)

# ---------------------------------------------------------------------------
# Dataset prep  —  converts JSONL → parquet in verl's expected format
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={DATA_PATH: data_volume, "/hf-cache": hf_cache_vol},
)
def prep(jsonl_data: str, test_jsonls: dict[str, str] | None = None) -> None:
    import pandas as pd
    import random

    trajectories = [
        json.loads(line) for line in jsonl_data.strip().split("\n") if line.strip()
    ]

    # Split at trajectory level to prevent conversation leakage into val.
    # Shuffle deterministically so reruns are stable.
    random.seed(42)
    random.shuffle(trajectories)
    cut = max(1, int(len(trajectories) * 0.9))
    train_trajs, val_trajs = trajectories[:cut], trajectories[cut:]

    # Edge case: if only one trajectory (e.g. LoCoMo-only run), keep it all in
    # train and mirror it as val — better than an empty val set.
    if not val_trajs:
        val_trajs = train_trajs

    train_examples = _build_verl_examples(train_trajs)
    val_examples   = _build_verl_examples(val_trajs)
    print(f"Trajectories — train: {len(train_trajs)}, val: {len(val_trajs)}")
    print(f"Examples     — train: {len(train_examples)}, val: {len(val_examples)}")

    DATA_PATH.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_examples).to_parquet(DATA_PATH / "train.parquet", index=False)
    pd.DataFrame(val_examples).to_parquet(DATA_PATH / "val.parquet",   index=False)

    # Store test JSONL files so eval() can load them from the volume.
    for name, content in (test_jsonls or {}).items():
        dest = DATA_PATH / f"{name}_test.jsonl"
        dest.write_text(content)
        n = content.count("\n")
        print(f"Stored test data: {name} ({n} trajectories) → {dest}")

    data_volume.commit()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-80GB:2",
    timeout=6 * 60 * MINUTES,
    volumes={
        MODELS_PATH: checkpoints_volume,
        DATA_PATH:   data_volume,
        "/hf-cache": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")],
)
def train(run_name: str = "locomo_lme_run_001", n_steps: int = N_STEPS) -> dict:
    data_volume.reload()

    cmd = [
        "python", "-m", "verl.trainer.main_ppo",
        # algorithm
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        # data
        f"data.train_files={DATA_PATH / 'train.parquet'}",
        f"data.val_files={DATA_PATH / 'val.parquet'}",
        "data.train_batch_size=8",       # prompts per step; total rollouts = 8 × N_ROLLOUTS
        "data.max_prompt_length=3072",
        "data.max_response_length=512",
        "data.filter_overlong_prompts=True",
        "data.truncation=right",
        # model + LoRA (cuts optimizer states from 32 GB → ~320 MB)
        f"actor_rollout_ref.model.path={MODEL_ID}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.model.lora_rank=32",
        "actor_rollout_ref.model.lora_alpha=32",
        "actor_rollout_ref.model.target_modules=all-linear",
        # actor
        f"actor_rollout_ref.actor.optim.lr=1e-4",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        # rollout (vLLM)
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        "actor_rollout_ref.rollout.load_format=safetensors",
        "actor_rollout_ref.rollout.layered_summon=True",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2",
        f"actor_rollout_ref.rollout.n={N_ROLLOUTS}",
        # ref model
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2",
        # trainer
        "trainer.critic_warmup=0",
        "trainer.logger=['console','wandb']",
        "trainer.project_name=memory-rlvr",
        f"trainer.experiment_name={run_name}",
        "trainer.n_gpus_per_node=2",
        "trainer.nnodes=1",
        f"trainer.test_freq={min(10, n_steps)}",
        f"trainer.save_freq={min(25, n_steps)}",
        f"trainer.total_training_steps={n_steps}",
        f"trainer.default_local_dir={MODELS_PATH / run_name}",
        "trainer.resume_mode=auto",
        # reward
        f"custom_reward_function.path={PATH_TO_REWARD_FUNCTION}",
        f"custom_reward_function.name={REWARD_FUNCTION_NAME}",
    ]

    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"verl exited with status {result.returncode}:\n{result.stderr[-4000:]}")
    return {"run_name": run_name, "steps": n_steps, "checkpoint": str(MODELS_PATH / run_name)}


# ---------------------------------------------------------------------------
# Eval — test the trained LoRA on a few example turns
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-80GB:1",
    timeout=90 * MINUTES,
    volumes={MODELS_PATH: checkpoints_volume, DATA_PATH: data_volume, "/hf-cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def eval(run_name: str = "locomo_lme_run_001", step: int = -1, n_examples: int = 0) -> dict:
    """Evaluate trained LoRA against LoCoMo + LongMemEval test sets.

    Args:
        run_name:   checkpoint directory under /models/
        step:       specific global step to load; -1 = latest
        n_examples: cap per dataset (0 = all)

    Returns dict of per-dataset metrics: mean_reward, valid_op_rate, abstain_rate.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch, glob
    from reward import compute_reward

    data_volume.reload()

    # ---- load checkpoint ----
    ckpt_root = MODELS_PATH / run_name
    if step == -1:
        dirs = sorted(glob.glob(str(ckpt_root / "global_step_*")))
        assert dirs, f"No checkpoints found under {ckpt_root}"
        ckpt_dir = dirs[-1]
    else:
        ckpt_dir = str(ckpt_root / f"global_step_{step}")
    print(f"Checkpoint: {ckpt_dir}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base  = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, ckpt_dir)
    model.eval()

    # ---- run eval per dataset ----
    all_results: dict[str, dict] = {}

    for dataset in ("locomo", "longmemeval"):
        test_path = DATA_PATH / f"{dataset}_test.jsonl"
        if not test_path.exists():
            print(f"[{dataset}] no test file at {test_path} — skipping")
            continue

        trajectories = [
            json.loads(line) for line in test_path.read_text().strip().split("\n") if line.strip()
        ]
        examples = _build_verl_examples(trajectories)
        if n_examples > 0:
            examples = examples[:n_examples]

        print(f"\n[{dataset}] {len(examples)} examples")

        rewards, abstains = [], 0

        for i, ex in enumerate(examples):
            prompt       = ex["prompt"]
            ground_truth = ex["reward_model"]["ground_truth"]

            ids = tokenizer.apply_chat_template(
                prompt, return_tensors="pt", add_generation_prompt=True
            ).to("cuda")

            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=256, do_sample=False)

            response = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            reward   = compute_reward(dataset, response, ground_truth, {})
            rewards.append(reward)

            if '"op": "ABSTAIN"' in response:
                abstains += 1

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(examples)} — running mean reward: {sum(rewards)/len(rewards):.3f}")

        n = len(rewards)
        mean_r     = sum(rewards) / n
        valid_rate = sum(1 for r in rewards if r > -0.5) / n
        abstain_rate = abstains / n

        print(f"\n[{dataset}] RESULTS ({n} examples)")
        print(f"  mean reward:   {mean_r:.3f}")
        print(f"  valid op rate: {valid_rate:.1%}")
        print(f"  abstain rate:  {abstain_rate:.1%}")
        print(f"  reward ≥ 0.5:  {sum(1 for r in rewards if r >= 0.5)/n:.1%}")
        print(f"  reward < 0:    {sum(1 for r in rewards if r < 0)/n:.1%}")

        all_results[dataset] = {
            "n":            n,
            "mean_reward":  round(mean_r, 4),
            "valid_op_rate": round(valid_rate, 4),
            "abstain_rate": round(abstain_rate, 4),
        }

    return all_results


# ---------------------------------------------------------------------------
# Local entrypoint — prep data then train
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def prep_data():
    """Rebuild the parquet dataset only. Run this after changing the system prompt."""
    if not DATA_PATH_LOCAL.exists():
        print(f"Data not found at {DATA_PATH_LOCAL}")
        print("Run: python3 data/converters/locomo_converter.py && python3 data/converters/longmemeval_converter.py")
        return
    jsonl_data  = DATA_PATH_LOCAL.read_text()
    test_jsonls = {k: p.read_text() for k, p in TEST_PATHS_LOCAL.items() if p.exists()}
    print(f"Loaded {jsonl_data.count(chr(10))} train trajectories, {len(test_jsonls)} test sets")
    prep.remote(jsonl_data=jsonl_data, test_jsonls=test_jsonls)
    print("Done. Run 'modal run train_modal.py::train_only' to start training.")


@app.local_entrypoint()
def sanity():
    """Sanity run — 20 steps on existing parquet. Confirms the full pipeline works (~$7).

    Run: modal run train_modal.py::sanity
    """
    print("Starting sanity run (20 steps)...")
    jsonl_data  = DATA_PATH_LOCAL.read_text()
    test_jsonls = {k: p.read_text() for k, p in TEST_PATHS_LOCAL.items() if p.exists()}
    prep.remote(jsonl_data=jsonl_data, test_jsonls=test_jsonls)
    result = train.remote(run_name="sanity_001", n_steps=20)
    print("Sanity run done:", result)


@app.local_entrypoint()
def train_only():
    """Train using existing parquet on the data volume (skips data prep)."""
    print("Starting GRPO training...")
    result = train.remote(run_name="locomo_lme_run_001")
    print("Done:", result)


@app.local_entrypoint()
def main():
    if not DATA_PATH_LOCAL.exists():
        print(f"Data not found at {DATA_PATH_LOCAL}")
        print("Run: python3 data/converters/locomo_converter.py && python3 data/converters/longmemeval_converter.py")
        return

    jsonl_data  = DATA_PATH_LOCAL.read_text()
    test_jsonls = {k: p.read_text() for k, p in TEST_PATHS_LOCAL.items() if p.exists()}
    print(f"Loaded {jsonl_data.count(chr(10))} train trajectories, {len(test_jsonls)} test sets")
    prep.remote(jsonl_data=jsonl_data, test_jsonls=test_jsonls)

    print("Starting GRPO training with verl + vLLM...")
    result = train.remote(run_name="locomo_lme_run_001")
    print("Done:", result)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a memory manager for a conversational AI agent. \
Given a conversation, extract and store the key facts that would help answer questions later.

Output a single JSON object:
{"op": "<STORE_FACT|UPDATE|SUPERSEDE|COMPRESS|ABSTAIN>", "content": "<what to remember>", "key": "<snake_case_key>", "confidence": <0.0-1.0>}

Rules:
- STORE_FACT: new fact not yet in memory
- UPDATE: a fact you already stored has changed
- SUPERSEDE: old fact was wrong, replace it permanently
- COMPRESS: summarise a long memory entry into a shorter one
- ABSTAIN: no storable information in this conversation
- Keep content concise — one sentence max
- key must be lowercase snake_case (e.g. user_name, last_location, job_title)
/nothink"""


def _build_verl_examples(trajectories: list[dict]) -> list[dict]:
    """Convert trajectory dicts → verl parquet rows.

    verl expects each row to have:
      - data_source: str
      - prompt: list of message dicts (chat format)
      - ability: str
      - reward_model: dict with 'ground_truth' key
      - extra_info: dict
    """
    examples = []
    for traj in trajectories:
        sessions = traj.get("sessions", [])
        if not sessions:
            continue

        # Use ALL sessions — not just the first one.
        # For long conversations (LoCoMo has 19-35 sessions) we concatenate
        # all turns so the model sees the full context when deciding what to store.
        all_turns = [t for s in sessions for t in s.get("turns", [])]
        conv_text = _format_conversation(all_turns)

        # Truncate if too long for context window (keep last N tokens worth)
        MAX_CONV_CHARS = 8_000
        if len(conv_text) > MAX_CONV_CHARS:
            conv_text = conv_text[-MAX_CONV_CHARS:]

        data_source = traj.get("vertical", "locomo")

        for probe in traj.get("qa_probes", []):
            question = probe.get("question", "")
            answer   = str(probe.get("answer", ""))
            if not question or not answer or answer in ("ABSTAIN", "Yes", "No", ""):
                continue

            prompt = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Conversation transcript:\n{conv_text}\n\n"
                        f"The following question will be asked from memory later:\n"
                        f"Q: {question}\n\n"
                        f"Store the relevant information now:"
                    ),
                },
            ]

            examples.append({
                "data_source":  data_source,
                "prompt":       prompt,
                "ability":      "memory_extraction",
                "reward_model": {"ground_truth": answer},
                "extra_info":   {"question": question},
            })

    return examples


def _format_conversation(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        role    = "Agent" if t.get("role") == "assistant" else "Customer"
        content = t.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


