"""GRPO memory training on Modal — Qwen3-8B with verl + SGLang (v0.5.0).

Run locally:
    modal run train_modal.py            # prep data + train
    modal run train_modal.py::prep      # data prep only
    modal run train_modal.py::train     # train only (after data is ready)
    modal run train_modal.py::sanity    # 20-step smoke test
"""

from __future__ import annotations

import json
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
N_ROLLOUTS  = 2      # completions per prompt (4 OOMs at 20-turn episodes; 2 halves KV pressure)
N_STEPS     = 500

# Context-1 retrieval service — deploy context1_service.py first, then paste the URL here.
# Leave empty to fall back to keyword grep during training.
CONTEXT1_SERVICE_URL = ""

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
    modal.Image.from_registry("verlai/verl:app-verl0.5-sglang0.4.8-mcore0.12.2-te2.2")
    .uv_pip_install("pandas", "pyarrow")
    # Reinstall verl Python sources at exact 0.5.0 without touching flash_attn/torch
    # (--no-deps keeps base image binaries intact, avoiding the flash_attn ABI conflict)
    .run_commands("pip install --no-deps verl==0.5.0")
    .add_local_file(Path(__file__).parent / "patch_verl.py",        "/root/patch_verl.py",        copy=True)
    .add_local_file(Path(__file__).parent / "reward.py",            "/root/reward.py",            copy=True)
    .add_local_file(Path(__file__).parent / "memory_fs.py",         "/root/memory_fs.py",         copy=True)
    .add_local_file(Path(__file__).parent / "ledger.py",            "/root/ledger.py",            copy=True)
    .add_local_file(Path(__file__).parent / "harness_state.py",     "/root/harness_state.py",     copy=True)
    .add_local_file(Path(__file__).parent / "agent_loop.py",        "/root/agent_loop.py",        copy=True)
    .add_local_file(Path(__file__).parent / "agent_loop_config.yaml", "/root/agent_loop_config.yaml", copy=True)
    .add_local_file(Path(__file__).parent / "context1_service.py",  "/root/context1_service.py",  copy=True)
    .run_commands("python /root/patch_verl.py")
    .env({
        "HF_HOME": "/hf-cache",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,garbage_collection_threshold:0.8",
        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "True",
        # Disable Ray's OOM killer so workers emit real CUDA OOM errors instead
        # of dying silently with SYSTEM_ERROR / connection code 2.
        "RAY_memory_monitor_refresh_ms": "0",
        "HYDRA_FULL_ERROR": "1",
    })
)

# Separate lightweight image for SFT — the verl base image has an apex build
# that conflicts with HuggingFace Trainer (apex.amp not importable). SFT only
# needs transformers + peft + torch; no verl/SGLang/apex required.
sft_image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime")
    .run_commands(
        "pip install transformers>=4.51.0 peft>=0.14.0 accelerate>=1.2.1 wandb datasets",
    )
    .add_local_file(Path(__file__).parent / "agent_loop.py",    "/root/agent_loop.py",    copy=True)
    .add_local_file(Path(__file__).parent / "harness_state.py", "/root/harness_state.py", copy=True)
    .add_local_file(Path(__file__).parent / "memory_fs.py",     "/root/memory_fs.py",     copy=True)
    .add_local_file(Path(__file__).parent / "ledger.py",        "/root/ledger.py",        copy=True)
    .env({"HF_HOME": "/hf-cache", "TOKENIZERS_PARALLELISM": "false"})
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
    import sys
    sys.path.insert(0, "/root")
    from agent_loop import build_verl_batch

    trajectories = [
        json.loads(line) for line in jsonl_data.strip().split("\n") if line.strip()
    ]

    # Split at trajectory level to prevent conversation leakage into val.
    # Shuffle deterministically so reruns are stable.
    random.seed(42)
    random.shuffle(trajectories)
    cut = max(1, int(len(trajectories) * 0.9))
    train_trajs, val_trajs = trajectories[:cut], trajectories[cut:]

    if not val_trajs:
        val_trajs = train_trajs

    train_examples = build_verl_batch(train_trajs)
    val_examples   = build_verl_batch(val_trajs)

    # Stamp Context-1 URL into every row so the agent loop can call it during rollouts
    if CONTEXT1_SERVICE_URL:
        for row in train_examples + val_examples:
            row["extra_info"]["context1_url"] = CONTEXT1_SERVICE_URL
    print(f"Trajectories — train: {len(train_trajs)}, val: {len(val_trajs)}")
    print(f"Episode windows — train: {len(train_examples)}, val: {len(val_examples)}")

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
# SFT — fine-tune Qwen3-8B on GPT-OSS-generated memory op demonstrations
# ---------------------------------------------------------------------------

@app.function(
    image=sft_image,
    gpu="A100-80GB:2",
    timeout=3 * 60 * MINUTES,
    volumes={
        MODELS_PATH: checkpoints_volume,
        DATA_PATH:   data_volume,
        "/hf-cache": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("wandb-secret")],
)
def sft(sft_jsonl: str, run_name: str = "sft_warmup", n_epochs: int = 1) -> dict:
    """SFT warm-up on GPT-OSS-generated traces before GRPO.

    Trains Qwen3-8B with LoRA for n_epochs on the demonstration traces.
    Saves checkpoint to /models/{run_name} so train() can resume from it.

    Args:
        sft_jsonl: raw JSONL string of SFT traces (from gen_sft.py)
        run_name:  checkpoint name
        n_epochs:  epochs to train (1 is usually enough for format learning)
    """
    import json as _json
    import sys
    import torch
    from pathlib import Path as _Path
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.utils.data import Dataset

    data_volume.reload()

    class SFTDataset(Dataset):
        def __init__(self, traces: list[dict], tokenizer, max_len: int = 1024):
            self.tokenizer = tokenizer
            self.max_len   = max_len
            self.examples  = []   # (full_text, prompt_len_in_tokens)
            for t in traces:
                msgs = t.get("messages", [])
                if not msgs:
                    continue
                full_text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                # Prompt-only portion (system + user), rendered the way it'd look
                # right before generation starts — used only to measure how many
                # leading tokens to mask out of the loss below. Tokenizing this
                # substring separately and trusting its length as the prefix
                # boundary is an approximation (BPE merges can occasionally
                # differ right at the boundary) but is the standard approach
                # used for this kind of completion-only masking.
                prompt_text = tokenizer.apply_chat_template(
                    msgs[:-1], tokenize=False, add_generation_prompt=True
                )
                prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
                self.examples.append((full_text, prompt_len))

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            full_text, prompt_len = self.examples[idx]
            enc = self.tokenizer(
                full_text,
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            ids    = enc["input_ids"].squeeze()
            labels = ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            # Mask the system+user prompt tokens too — previously the loss ran
            # over the ENTIRE sequence, including the rendered [Memory state]/
            # turn text the model is just being shown, not asked to produce.
            # Only the assistant's JSON completion should contribute to the
            # gradient; that's the one thing SFT is actually meant to teach here.
            mask_len = min(prompt_len, labels.shape[0])
            labels[:mask_len] = -100
            return {"input_ids": ids, "attention_mask": enc["attention_mask"].squeeze(), "labels": labels}

    traces = [_json.loads(l) for l in sft_jsonl.strip().split("\n") if l.strip()]
    print(f"[sft] {len(traces)} traces loaded")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32, lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset  = SFTDataset(traces, tokenizer)
    ckpt_dir = str(MODELS_PATH / run_name)

    args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb",
        run_name=run_name,
    )
    Trainer(model=model, args=args, train_dataset=dataset).train()

    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    # verl 0.5.0 has no LoRA-adapter resume path (lora_adapter_path landed in
    # PR #3523, merged 2025-10-28 — three months after the 0.5.0 release this
    # repo is pinned to; confirmed via PyPI's release date + the GitHub PR, not
    # assumed). The only way GRPO can actually start from this SFT checkpoint at
    # this verl version is from a full merged model, not the bare adapter dir —
    # train()'s base_model_path should point here, not at MODEL_ID, to use it.
    merged_dir = str(MODELS_PATH / f"{run_name}_merged")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    checkpoints_volume.commit()
    print(f"[sft] adapter checkpoint → {ckpt_dir}")
    print(f"[sft] merged checkpoint  → {merged_dir}  (pass this to train(base_model_path=...))")
    return {
        "run_name":          run_name,
        "n_traces":          len(traces),
        "checkpoint":        ckpt_dir,
        "merged_checkpoint": merged_dir,
    }


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
def train(run_name: str = "locomo_lme_run_001", n_steps: int = N_STEPS, base_model_path: str = MODEL_ID) -> dict:
    """base_model_path: defaults to the raw base model. Pass sft()'s returned
    "merged_checkpoint" path here to actually warm-start GRPO from the SFT run —
    previously this was always MODEL_ID regardless of whether sft() had been run,
    so run_sft()'s claimed "GRPO resumes from the SFT checkpoint" never happened.
    """
    data_volume.reload()

    import json as _json

    # Qwen3 tokenizer saved by transformers>=4.51 stores extra_special_tokens as a
    # list []; the older transformers in the verl image expects a dict and calls
    # .keys() on it, crashing Ray workers before training starts. Patch in-place.
    _tok_cfg = Path(base_model_path) / "tokenizer_config.json"
    if _tok_cfg.exists():
        _cfg = _json.loads(_tok_cfg.read_text())
        if isinstance(_cfg.get("extra_special_tokens"), list):
            _cfg["extra_special_tokens"] = {}
            _tok_cfg.write_text(_json.dumps(_cfg))
            print("[patch] tokenizer_config.json: extra_special_tokens list→dict")

    # transformers>=4.52 saves SFT checkpoints with use_cache=False and may write
    # rope_theta=10000.0 (wrong default) instead of 1000000. Fix both so the verl
    # image loads the checkpoint correctly.
    _model_cfg = Path(base_model_path) / "config.json"
    if _model_cfg.exists():
        _cfg = _json.loads(_model_cfg.read_text())
        changed = False
        if not _cfg.get("use_cache", True):
            _cfg["use_cache"] = True
            changed = True
        # Restore correct Qwen3-8B rope_theta if the checkpoint clobbered it
        if _cfg.get("rope_theta") == 10000.0 and _cfg.get("model_type") == "qwen3":
            _cfg["rope_theta"] = 1000000.0
            changed = True
        if changed:
            _model_cfg.write_text(_json.dumps(_cfg))
            print(f"[patch] config.json: use_cache={_cfg.get('use_cache')}, rope_theta={_cfg.get('rope_theta', 'N/A (uses rope_parameters)')}")

    cmd = [
        "python", "-m", "verl.trainer.main_ppo",
        # algorithm
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        # data
        f"data.train_files={DATA_PATH / 'train.parquet'}",
        f"data.val_files={DATA_PATH / 'val.parquet'}",
        "data.train_batch_size=4",       # prompts per step; total rollouts = 4 × N_ROLLOUTS (dense probe windows need fewer concurrent seqs)
        "data.max_prompt_length=2048",
        "data.max_response_length=256",
        "data.filter_overlong_prompts=True",
        "data.truncation=right",
        # model + LoRA (cuts optimizer states from 32 GB → ~320 MB)
        f"actor_rollout_ref.model.path={base_model_path}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.model.lora_rank=32",
        "actor_rollout_ref.model.lora_alpha=32",
        "actor_rollout_ref.model.target_modules=all-linear",
        # actor
        f"actor_rollout_ref.actor.optim.lr=1e-4",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.001",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        # rollout (SGLang — verl v0.5.0)
        # TP=2 splits model across both GPUs; free_cache_engine offloads KV between phases;
        # enforce_eager required with free_cache_engine; gpu_memory_utilization=0.2 leaves
        # headroom for FSDP allgathers (model=8GB + KV=8GB per GPU at 0.2×80GB=16GB)
        "actor_rollout_ref.rollout.name=sglang",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.30",  # 0.30×80=24GB KV; leaves ~50GB headroom so CUDA-graph capture + SGLang init don't race
        "actor_rollout_ref.rollout.free_cache_engine=True",
        "actor_rollout_ref.rollout.enforce_eager=True",
        # enforce_eager only disables runtime compilation; disable_cuda_graph
        # explicitly prevents the CUDA graph capture loop at init time.
        # Both are needed: enforce_eager alone does NOT stop "Capture cuda graph bs".
        "++actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=true",
        "actor_rollout_ref.rollout.multi_stage_wake_up=True",    # resume weights→state_dict→resume KV; prevents state_dict OOM when KV+weights+alloc all compete
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2",
        f"actor_rollout_ref.rollout.n={N_ROLLOUTS}",
        # multi-turn AgentLoop — mode=async is required: verl's AgentLoop registry/
        # agent_name routing is only consulted under async rollout (default is
        # "sync", which uses the plain SPMD path and ignores agent_name entirely —
        # confirmed against verl/trainer/config/rollout/rollout.yaml at the v0.5.0
        # tag). agent_loop_cls and max_turns/single_response_max_tokens below were
        # never-real config keys (none of the three appear anywhere in verl's
        # schema) — agent_name on each row (agent_loop.build_verl_batch) plus this
        # YAML registry path are the actual mechanism.
        "actor_rollout_ref.rollout.mode=async",
        # no + prefix — agent.agent_loop_config_path is already in verl's default
        # schema (rollout.yaml: agent.agent_loop_config_path, default null); this
        # repo already hit the "+ on an existing key" hydra error twice before
        # (985c35c, 27c4b42) for return_raw_chat and multi_stage_wake_up.
        "actor_rollout_ref.rollout.agent.agent_loop_config_path=/root/agent_loop_config.yaml",
        # raw chat format required for AgentLoop
        "data.return_raw_chat=True",
        # torch.compile / CUDA graph capture is disabled: capturing graphs for
        # batch_sizes=[1,2,4,...,160] creates huge GPU memory spikes that race
        # with SGLang's initial KV allocation, causing OOM even on A100-80GB.
        "actor_rollout_ref.actor.use_torch_compile=False",
        "actor_rollout_ref.ref.use_torch_compile=False",
        # ref model
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2",
        # trainer — val_before_train=False skips the initial _validate() that OOMs on
        # first SGLang wake_up (KV cache alloc) before FSDP has freed its GPU memory
        "trainer.critic_warmup=0",
        "trainer.val_before_train=False",
        "trainer.logger=['console','wandb']",
        "trainer.project_name=memory-rlvr",
        f"trainer.experiment_name={run_name}",
        "trainer.n_gpus_per_node=2",
        "trainer.nnodes=1",
        f"trainer.test_freq={n_steps + 1}",  # TODO: re-enable once SGLang KV alloc OOM is fixed (set to 50)
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
def eval(run_name: str = "locomo_lme_run_001", step: int = -1, n_trajectories: int = 0) -> dict:
    """Evaluate trained LoRA against LoCoMo + LongMemEval test sets.

    Runs a real multi-turn eval: model processes turns one by one, builds a
    VirtualFilesystem from its own ops, then QA probes are scored against the FS.
    This matches the training setup exactly (no question hint, real FS accumulation).

    Args:
        run_name:        checkpoint directory under /models/
        step:            specific global step to load; -1 = latest
        n_trajectories:  cap on trajectories per dataset (0 = all)

    Returns dict of per-dataset metrics.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch, glob, sys
    sys.path.insert(0, "/root")
    from memory_fs import VirtualFilesystem, parse_op, score_trajectory
    from agent_loop import _SYSTEM_PROMPT, _format_turn

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

    all_results: dict[str, dict] = {}

    for dataset in ("locomo", "longmemeval"):
        test_path = DATA_PATH / f"{dataset}_test.jsonl"
        if not test_path.exists():
            print(f"[{dataset}] no test file at {test_path} — skipping")
            continue

        trajectories = [
            json.loads(line) for line in test_path.read_text().strip().split("\n") if line.strip()
        ]
        if n_trajectories > 0:
            trajectories = trajectories[:n_trajectories]

        print(f"\n[{dataset}] {len(trajectories)} trajectories")

        traj_rewards, abstain_counts, op_counts = [], [], []

        for traj in trajectories:
            sessions    = traj.get("sessions", [])
            qa_probes   = traj.get("qa_probes", [])
            fs          = VirtualFilesystem()
            traj_abstains = 0
            traj_ops      = 0

            # Process every turn in order — model builds its own FS
            for s_idx, session in enumerate(sessions):
                for t_idx, turn in enumerate(session.get("turns", [])):
                    content = turn.get("content", "").strip()
                    if not content:
                        continue

                    messages = [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"[Memory state]\n{fs.render_for_prompt()}\n\n"
                                f"[Current turn]\n{_format_turn(turn)}"
                            ),
                        },
                    ]

                    ids = tokenizer.apply_chat_template(
                        messages, return_tensors="pt", add_generation_prompt=True
                    ).to("cuda")

                    with torch.no_grad():
                        out = model.generate(ids, max_new_tokens=256, do_sample=False)

                    response = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                    op = parse_op(response)
                    fs.apply_op(op, session_idx=s_idx, turn_idx=t_idx)
                    traj_ops += 1

                    if op.get("op") == "ABSTAIN" or not op:
                        traj_abstains += 1

            # Score final FS against all QA probes
            traj_reward = score_trajectory(fs, qa_probes)
            traj_rewards.append(traj_reward)
            abstain_counts.append(traj_abstains / max(traj_ops, 1))
            op_counts.append(traj_ops)

        n          = len(traj_rewards)
        mean_r     = sum(traj_rewards) / n
        abstain_r  = sum(abstain_counts) / n

        print(f"\n[{dataset}] RESULTS ({n} trajectories, avg {sum(op_counts)/n:.0f} turns/traj)")
        print(f"  mean FS-QA F1:  {mean_r:.3f}")
        print(f"  abstain rate:   {abstain_r:.1%}")
        print(f"  F1 ≥ 0.5:       {sum(1 for r in traj_rewards if r >= 0.5)/n:.1%}")
        print(f"  F1 < 0.1:       {sum(1 for r in traj_rewards if r < 0.1)/n:.1%}")

        all_results[dataset] = {
            "n":              n,
            "mean_fs_qa_f1":  round(mean_r, 4),
            "abstain_rate":   round(abstain_r, 4),
            "f1_ge_0.5":      round(sum(1 for r in traj_rewards if r >= 0.5) / n, 4),
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
    import time
    run_name = f"sanity_{int(time.time())}"   # unique name prevents resume from old checkpoint
    print(f"Starting sanity run (20 steps, name={run_name})...")
    jsonl_data  = DATA_PATH_LOCAL.read_text()
    test_jsonls = {k: p.read_text() for k, p in TEST_PATHS_LOCAL.items() if p.exists()}
    prep.remote(jsonl_data=jsonl_data, test_jsonls=test_jsonls)
    result = train.remote(run_name=run_name, n_steps=20)
    print("Sanity run done:", result)


@app.local_entrypoint()
def train_only(base_model_path: str = MODEL_ID):
    """Train using existing parquet on the data volume (skips data prep).

    Defaults to base Qwen3-8B (simplest path — use this to test whether the
    OOM is from the SFT checkpoint config). Override with:
        --base-model-path /models/sft_warmup_merged
    to warm-start from the SFT checkpoint once the base run is confirmed working.
    """
    print("Starting GRPO training...")
    kwargs = {"run_name": "locomo_lme_run_004", "base_model_path": base_model_path}
    print(f"Starting from: {base_model_path}")
    result = train.remote(**kwargs)
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
    result = train.remote(run_name="locomo_lme_run_002")
    print("Done:", result)


# ---------------------------------------------------------------------------
# Diagnose — local, no GPU needed
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def diagnose():
    """Inspect reward coverage and window quality before training.

    Runs locally (no GPU). Simulates build_verl_batch and reports:
      - How many windows survive the probe-coverage filter
      - Expected reward floor / ceiling per window
      - Op type distribution that a base model would need to produce to win

    Run: modal run train_modal.py::diagnose
    """
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from agent_loop import build_verl_batch, _resolve_probe_sessions
    from memory_fs import VirtualFilesystem, score_trajectory

    if not DATA_PATH_LOCAL.exists():
        print(f"No data at {DATA_PATH_LOCAL}")
        return

    trajectories = [
        json.loads(line) for line in DATA_PATH_LOCAL.read_text().strip().split("\n") if line.strip()
    ]
    print(f"\n=== Dataset: {len(trajectories)} trajectories ===")

    rows = build_verl_batch(trajectories)
    print(f"Training rows after window filter: {len(rows)}")

    # Reward floor: empty FS (model ABSTAINs everything)
    # Reward ceiling: model stores the answer verbatim into the FS
    floors, ceilings = [], []
    for row in rows[:200]:
        probes = row["extra_info"]["qa_probes"]
        if not probes:
            continue

        # Floor: empty FS
        fs_empty = VirtualFilesystem()
        floors.append(score_trajectory(fs_empty, probes))

        # Ceiling: oracle stores answer verbatim at a relevant path
        fs_oracle = VirtualFilesystem()
        for p in probes:
            fs_oracle.apply_op(
                {"op": "STORE_FACT", "path": "facts/oracle", "content": str(p.get("answer", ""))},
                session_idx=0, turn_idx=0,
            )
        ceilings.append(score_trajectory(fs_oracle, probes))

    def _stats(vals):
        n = len(vals)
        if not n:
            return "no data"
        return f"mean={sum(vals)/n:.3f}  min={min(vals):.3f}  max={max(vals):.3f}  >0: {sum(1 for v in vals if v>0)/n:.1%}"

    print(f"\nReward floor  (empty FS):  {_stats(floors)}")
    print(f"Reward ceiling (oracle FS): {_stats(ceilings)}")

    # Probe count distribution
    probe_counts = [len(r["extra_info"]["qa_probes"]) for r in rows]
    print(f"\nProbes per window: mean={sum(probe_counts)/len(probe_counts):.1f}  "
          f"max={max(probe_counts)}  1-probe: {sum(1 for c in probe_counts if c==1)/len(probe_counts):.1%}")

    # Vertical split
    from collections import Counter
    verticals = Counter(r["data_source"] for r in rows)
    print(f"\nVertical split: {dict(verticals)}")

    print("\nTo run baseline eval on base model (no RL): modal run train_modal.py::baseline")


# ---------------------------------------------------------------------------
# Local entrypoints for SFT
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def run_sft():
    """Generate SFT traces with GPT-OSS 120B, then fine-tune Qwen3-8B.

    Prerequisites:
        1. python data/download_data.py          (downloads LoCoMo)
        2. export OPENAI_API_KEY=sk-...
        3. modal run train_modal.py::run_sft

    After this completes, run GRPO warm-started from the SFT checkpoint (the
    exact command, with the merged checkpoint path filled in, is printed below):
        modal run train_modal.py::train_only --base-model-path /models/sft_warmup_merged
    """
    import subprocess, os, sys

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    locomo_src = REPO_ROOT / "data" / "locomo10.json"
    sft_out    = REPO_ROOT / "data" / "sft_traces.jsonl"

    if not locomo_src.exists():
        print(f"Error: {locomo_src} not found — run python data/download_data.py first")
        sys.exit(1)

    # Generate traces locally (cheap API calls, no GPU needed)
    if not sft_out.exists():
        print("Generating SFT traces with GPT-OSS 120B ...")
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "data" / "gen_sft.py"),
             "--src", str(locomo_src), "--out", str(sft_out), "--n", "200"],
            env={**os.environ, "OPENAI_API_KEY": api_key},
        )
        if result.returncode != 0:
            print("gen_sft.py failed")
            sys.exit(1)
    else:
        n = sft_out.read_text().count("\n")
        print(f"SFT traces already at {sft_out} ({n} lines), skipping generation")

    sft_jsonl = sft_out.read_text()
    n_traces  = sft_jsonl.count("\n")
    print(f"Uploading {n_traces} traces to Modal and starting SFT ...")
    result = sft.remote(sft_jsonl=sft_jsonl, run_name="sft_warmup", n_epochs=1)
    print("SFT done:", result)
    # train_only previously always trained from the raw base model regardless
    # of whether SFT had run — pass the merged checkpoint explicitly to
    # actually use it (verl 0.5.0 can't resume from a bare LoRA adapter dir,
    # see sft()'s comment on lora_adapter_path).
    merged = result.get("merged_checkpoint", "")
    print(f"\nNext: modal run train_modal.py::train_only --base-model-path {merged}")


# ---------------------------------------------------------------------------
# Baseline eval — runs base Qwen3-8B with no LoRA
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-80GB:1",
    timeout=90 * MINUTES,
    volumes={MODELS_PATH: checkpoints_volume, DATA_PATH: data_volume, "/hf-cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def _baseline_eval_remote(n_trajectories: int = 20) -> dict:
    """Run base Qwen3-8B (no RL) on test sets to establish the reward floor.

    This is the number the trained model must beat. Run once before the big run.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch, sys
    sys.path.insert(0, "/root")
    from memory_fs import VirtualFilesystem, parse_op, score_trajectory
    from agent_loop import _SYSTEM_PROMPT, _format_turn

    data_volume.reload()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    all_results = {}

    for dataset in ("locomo", "longmemeval"):
        test_path = DATA_PATH / f"{dataset}_test.jsonl"
        if not test_path.exists():
            print(f"[{dataset}] no test file — skipping")
            continue

        trajectories = [
            json.loads(line) for line in test_path.read_text().strip().split("\n") if line.strip()
        ]
        if n_trajectories > 0:
            trajectories = trajectories[:n_trajectories]

        print(f"\n[{dataset}] {len(trajectories)} trajectories (base model, no RL)")

        traj_rewards, abstain_counts = [], []

        for traj in trajectories:
            sessions    = traj.get("sessions", [])
            qa_probes   = traj.get("qa_probes", [])
            fs          = VirtualFilesystem()
            traj_ops    = 0
            traj_abstains = 0

            for s_idx, session in enumerate(sessions):
                for t_idx, turn in enumerate(session.get("turns", [])):
                    content = turn.get("content", "").strip()
                    if not content:
                        continue

                    messages = [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user",   "content": (
                            f"[Memory state]\n{fs.render_for_prompt()}\n\n"
                            f"[Current turn]\n{_format_turn(turn)}"
                        )},
                    ]

                    ids = tokenizer.apply_chat_template(
                        messages, return_tensors="pt", add_generation_prompt=True
                    ).to("cuda")

                    with torch.no_grad():
                        out = model.generate(ids, max_new_tokens=256, do_sample=False)

                    response = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                    op = parse_op(response)
                    fs.apply_op(op, session_idx=s_idx, turn_idx=t_idx)
                    traj_ops += 1
                    if op.get("op") == "ABSTAIN" or not op:
                        traj_abstains += 1

            traj_rewards.append(score_trajectory(fs, qa_probes))
            abstain_counts.append(traj_abstains / max(traj_ops, 1))

        n      = len(traj_rewards)
        mean_r = sum(traj_rewards) / n
        print(f"[{dataset}] BASE MODEL — mean FS-QA F1: {mean_r:.3f}  abstain: {sum(abstain_counts)/n:.1%}")

        all_results[dataset] = {
            "n":             n,
            "mean_fs_qa_f1": round(mean_r, 4),
            "abstain_rate":  round(sum(abstain_counts) / n, 4),
            "model":         "base (no RL)",
        }

    return all_results


@app.local_entrypoint()
def baseline():
    """Run base Qwen3-8B eval — establishes the floor the RL model must beat.

    Run once before the big training run.
    Run: modal run train_modal.py::baseline
    """
    print("Running baseline eval on base Qwen3-8B (no RL)...")
    results = _baseline_eval_remote.remote(n_trajectories=20)
    print("\n=== BASELINE RESULTS ===")
    for dataset, metrics in results.items():
        print(f"  {dataset}: FS-QA F1 = {metrics['mean_fs_qa_f1']:.3f}  "
              f"abstain = {metrics['abstain_rate']:.1%}")




