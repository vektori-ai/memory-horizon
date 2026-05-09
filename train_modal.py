"""GRPO memory training on Modal — Qwen3-8B with verl + vLLM.

Run locally:
    modal run train_modal.py            # prep data + train
    modal run train_modal.py::prep      # data prep only
    modal run train_modal.py::train     # train only (after data is ready)
"""

from __future__ import annotations

import json
import re
import string
import subprocess
from collections import Counter
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent
DATA_PATH_LOCAL = REPO_ROOT / "data" / "debt_collection.jsonl"

MODEL_ID    = "Qwen/Qwen3-8B"
N_ROLLOUTS  = 4      # completions per prompt (GRPO group size)
N_STEPS     = 30   # bump to 200 once rewards look healthy
LR          = 2e-6

VERL_REPO_PATH = Path("/root/verl")
DATA_PATH      = Path("/data")
MODELS_PATH    = Path("/models")
MINUTES        = 60

# Path where Modal mounts this script inside the container — verl loads
# compute_reward from it at runtime via custom_reward_function.path
PATH_TO_REWARD_FUNCTION = Path("/root/train_modal.py")
REWARD_FUNCTION_NAME    = "compute_reward"

# ---------------------------------------------------------------------------
# Modal image — verl base image with vLLM already baked in
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry("verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.1")
    .apt_install("git")
    .run_commands(f"git clone https://github.com/volcengine/verl {VERL_REPO_PATH}")
    .uv_pip_install("verl[vllm]==0.4.1", "pandas", "pyarrow")
    .run_commands(
        "sed -i 's/self\\.actor_optimizer\\.step()/torch.cuda.empty_cache(); self.actor_optimizer.step()/' "
        "/usr/local/lib/python3.10/dist-packages/verl/workers/actor/dp_actor.py"
    )
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
def prep(jsonl_data: str) -> None:
    import pandas as pd

    examples = _build_verl_examples(jsonl_data)
    print(f"Built {len(examples)} examples")

    split = int(len(examples) * 0.9)
    train_df = pd.DataFrame(examples[:split])
    val_df   = pd.DataFrame(examples[split:])

    DATA_PATH.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(DATA_PATH / "train.parquet", index=False)
    val_df.to_parquet(DATA_PATH / "val.parquet",   index=False)
    data_volume.commit()
    print(f"Saved {len(train_df)} train / {len(val_df)} val examples to volume")


# ---------------------------------------------------------------------------
# Reward function  —  called by verl at training time
# verl signature: compute_reward(data_source, solution_str, ground_truth, extra_info) -> float
# ---------------------------------------------------------------------------

def compute_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> float:
    content = _extract_content_field(solution_str)
    # Categorical enum labels (all-caps): score by keyword recall against core
    # concepts. F1 would penalise the model for storing a full sentence vs a
    # short label, so recall is the right signal here.
    if re.match(r"^[A-Z][A-Z0-9_]+$", ground_truth):
        return _categorical_score(content, ground_truth)
    # Free-form / natural language: token F1 with a lenient threshold.
    f1 = _token_f1(content, ground_truth)
    return 1.0 if f1 >= 0.5 else f1


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
def train(run_name: str = "debt_collection_run_001") -> dict:
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
        "data.max_prompt_length=1024",
        "data.max_response_length=768",
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
        "trainer.test_freq=20",
        "trainer.save_freq=50",
        f"trainer.total_training_steps={N_STEPS}",
        f"trainer.default_local_dir={MODELS_PATH / run_name}",
        "trainer.resume_mode=auto",
        # reward
        f"custom_reward_function.path={PATH_TO_REWARD_FUNCTION}",
        f"custom_reward_function.name={REWARD_FUNCTION_NAME}",
    ]

    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"verl exited with status {result.returncode}:\n{result.stderr[-4000:]}")
    return {"run_name": run_name, "steps": N_STEPS, "checkpoint": str(MODELS_PATH / run_name)}


# ---------------------------------------------------------------------------
# Eval — test the trained LoRA on a few example turns
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-80GB:1",
    timeout=10 * MINUTES,
    volumes={MODELS_PATH: checkpoints_volume, "/hf-cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def eval(run_name: str = "debt_collection_run_001", step: int = -1):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    import torch, glob, os

    # find latest checkpoint if step == -1
    ckpt_root = MODELS_PATH / run_name
    if step == -1:
        dirs = sorted(glob.glob(str(ckpt_root / "global_step_*")))
        assert dirs, f"No checkpoints found under {ckpt_root}"
        ckpt_dir = dirs[-1]
    else:
        ckpt_dir = str(ckpt_root / f"global_step_{step}")
    print(f"Loading checkpoint: {ckpt_dir}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, ckpt_dir)
    model.eval()

    test_turns = [
        ("Agent: Hello, calling about your DemoLender loan. Outstanding is 47,000 rupees. Settlement offer: 32,900.\n"
         "Customer: I already paid that loan off months ago.",
         "Customer disputes the outstanding balance, claims loan was already paid"),
        ("Agent: Can we schedule a callback?\nCustomer: Call me next Tuesday after 6pm.",
         "Customer requested callback on Tuesday after 6pm"),
        ("Agent: What's your current address?\nCustomer: I'm at 42 Park Street, Mumbai.",
         "Customer's address is 42 Park Street, Mumbai"),
    ]

    system = (
        "You are a memory manager for an AI calling agent. Extract key facts as JSON:\n"
        '{"op": "<STORE_FACT|UPDATE|SUPERSEDE|ABSTAIN>", "content": "<what to remember>", '
        '"key": "<snake_case_key>", "confidence": <0.0-1.0>}'
    )

    for transcript, expected in test_turns:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Conversation transcript:\n{transcript}\n\nStore the relevant information now:"},
        ]
        ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=256, temperature=0.1, do_sample=True)
        response = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        print(f"\nTranscript: {transcript[:80]}...")
        print(f"Expected:   {expected}")
        print(f"Model:      {response}")
        print("-" * 60)


# ---------------------------------------------------------------------------
# Local entrypoint — prep data then train
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    if not DATA_PATH_LOCAL.exists():
        print(f"Data not found at {DATA_PATH_LOCAL}")
        print("Run: python3 convert_calls.py")
        return

    jsonl_data = DATA_PATH_LOCAL.read_text()
    n_lines = jsonl_data.count("\n")
    print(f"Loaded {n_lines} trajectories — uploading and preparing dataset...")
    prep.remote(jsonl_data=jsonl_data)

    print("Starting GRPO training with verl + vLLM...")
    result = train.remote(run_name="debt_collection_run_001")
    print("Done:", result)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a memory manager for an AI calling agent. Your job is to extract and store \
the key facts from each conversation turn into a structured memory operation.

Output a single JSON object with this schema:
{"op": "<STORE_FACT|UPDATE|SUPERSEDE|ABSTAIN>", "content": "<what to remember>", "key": "<snake_case_key>", "confidence": <0.0-1.0>}

Rules:
- STORE_FACT: new information not yet in memory
- UPDATE: fact you already stored has changed (customer changed their mind)
- SUPERSEDE: old fact was wrong, new fact replaces it permanently
- ABSTAIN: this turn contains no storable memory (small talk, silence, wrong number)
- Keep content concise — one sentence max
- key must be lowercase snake_case (e.g. payment_amount, callback_date)
/nothink"""


def _build_verl_examples(jsonl_data: str) -> list[dict]:
    """Convert trajectory JSONL → verl parquet rows.

    verl expects each row to have:
      - data_source: str
      - prompt: list of message dicts (chat format)
      - ability: str
      - reward_model: dict with 'ground_truth' key
      - extra_info: dict
    """
    examples = []
    for line in jsonl_data.strip().split("\n"):
        if not line.strip():
            continue
        traj = json.loads(line)

        sessions = traj.get("sessions", [])
        if not sessions:
            continue
        turns = sessions[0].get("turns", [])
        conv_text = _format_conversation(turns)

        for probe in traj.get("qa_probes", []):
            question = probe.get("question", "")
            answer   = probe.get("answer", "")
            # Skip unanswerable probes: ABSTAIN labels carry no signal, and
            # Yes/No booleans can't be matched against a stored sentence via F1.
            if not question or not answer or answer in ("ABSTAIN", "Yes", "No"):
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
                "data_source":  "debt_collection",
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


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def _extract_content_field(completion: str) -> str:
    if isinstance(completion, list):
        # newer TRL/verl may pass message dicts — extract assistant turn
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                completion = msg.get("content", "")
                break
        else:
            completion = str(completion)

    completion = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL).strip()
    try:
        obj = json.loads(completion.strip())
        return str(obj.get("content", ""))
    except (json.JSONDecodeError, AttributeError):
        pass
    m = re.search(r'"content"\s*:\s*"([^"]+)"', completion)
    if m:
        return m.group(1)
    return completion


def _token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction)
    gold_tokens = _normalize(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common    = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    precision = common / len(pred_tokens)
    recall    = common / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


_STOP = {"what", "is", "the", "are", "was", "were", "a", "an", "this", "that",
         "of", "for", "in", "on", "at", "to", "does", "did", "has", "have", "had"}

# Canonical expansions for outcome/category enum labels.
# Include word variants (singular/plural, verb forms) so token F1 naturally fires
# when the model writes a descriptive sentence about that outcome.
# Underscore-separated ALL_CAPS labels not listed here get auto-expanded
# (e.g. NEW_LABEL → "new label").
_LABEL_EXPANSIONS: dict[str, str] = {
    "PTP":             "promise promises promised pay payment commit committed will pay",
    "STRONGEST_PTP":   "strong promise commit committed pay date specific confirmed",
    "CALLBACK":        "callback requested call back follow up later",
    "ALREADY_PAID":    "already paid payment made settled cleared loan paid",
    "DISPUTE":         "dispute disputes disputed customer disputes outstanding balance",
    "SCAM_ALLEGATION": "scam fraud alleged customer alleges cheating",
    "RTP":             "refuse refused refuse to pay not paying unwilling",
    "PARTIAL_PTP":     "partial payment part pay partial promise",
    "NO_CONTACT":      "no contact unreachable wrong number not reachable",
    "BROKEN_PTP":      "broken promise missed payment did not pay failed",
}


def _expand_label(label: str) -> str:
    """Expand enum-style labels to natural language for F1 scoring."""
    if label in _LABEL_EXPANSIONS:
        return _LABEL_EXPANSIONS[label]
    if re.match(r"^[A-Z][A-Z0-9_]+$", label):
        return label.replace("_", " ").lower()
    return label


# Core keyword sets for categorical labels — 2-3 words that MUST appear in
# stored content for the outcome to be correctly captured.  Scored by recall:
# what fraction of these keywords appear in the stored sentence.
# Core keyword PREFIXES for categorical labels.  Using prefixes (e.g. "disput"
# for dispute/disputes, "refus" for refuse/refused) handles common inflections
# without a full stemmer.  Each entry is checked with startswith() against the
# normalised content tokens, so 2-3 short prefixes per label are enough.
_LABEL_KEYWORDS: dict[str, list[str]] = {
    "PTP":             ["promis", "pay"],
    "STRONGEST_PTP":   ["promis", "pay", "date"],
    "CALLBACK":        ["callback"],
    "ALREADY_PAID":    ["paid", "alread"],
    "DISPUTE":         ["disput"],
    "SCAM_ALLEGATION": ["scam", "fraud"],
    "RTP":             ["refus", "pay"],
    "PARTIAL_PTP":     ["partial", "pay"],
    "NO_CONTACT":      ["contact", "unreach"],
    "BROKEN_PTP":      ["broken", "pay"],
}


def _categorical_score(content: str, label: str) -> float:
    """Score categorical outcome labels by keyword-prefix recall.

    For each core prefix in the label's keyword list, checks whether any token
    in the stored content starts with that prefix (handling inflections like
    dispute/disputes, refuse/refused without a full stemmer).

    Returns the fraction of prefixes matched (1.0 = all hit).
    Unknown labels fall back to token F1 against the auto-expanded phrase.
    """
    keywords = _LABEL_KEYWORDS.get(label)
    if not keywords:
        gold = _expand_label(label)
        f1 = _token_f1(content, gold)
        return 1.0 if f1 >= 0.5 else f1

    content_tokens = set(_normalize(content))
    hits = sum(
        1 for kw in keywords
        if any(t.startswith(kw) for t in content_tokens)
    )
    return hits / len(keywords)


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t not in _STOP]
