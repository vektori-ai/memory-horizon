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
DATA_PATH_LOCAL = REPO_ROOT / "data" / "train.jsonl"

MODEL_ID    = "Qwen/Qwen3-8B"
N_ROLLOUTS  = 4      # completions per prompt (GRPO group size)
N_STEPS     = 200

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
    .add_local_file(Path(__file__).parent / "patch_verl.py", "/root/patch_verl.py", copy=True)
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
def prep(jsonl_data: str) -> None:
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
    data_volume.commit()


# ---------------------------------------------------------------------------
# Reward function  —  called by verl at training time
# ---------------------------------------------------------------------------

_VALID_OPS = {"STORE_FACT", "UPDATE", "SUPERSEDE", "COMPRESS", "ABSTAIN"}
_CONTENT_OPTIONAL_OPS = {"ABSTAIN"}


def compute_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> float:
    """Vektori reward: R = 0.6 * token_F1 + 0.4 * ROUGE1_recall - volume_penalty.

    Format gate: invalid/unparseable op → -0.5 (penalised, not neutral).
    ABSTAIN / DECAY with no content → 0.0 (model chose not to store; no signal).
    """
    if not _is_valid_memory_op(solution_str):
        return -0.5

    content = _extract_content_field(solution_str)
    if not content:
        return 0.0

    r_task   = _token_f1(content, ground_truth)
    r_memory = _rouge1_recall(content, ground_truth)
    p_volume = min(0.002 * len(content.split()), 0.3)

    return max(-1.0, min(1.0, 0.6 * r_task + 0.4 * r_memory - p_volume))


def _is_valid_memory_op(solution_str: str) -> bool:
    """Gate: does the output contain a parseable JSON with a valid op and content?

    Returns False (→ -0.5 penalty) if:
      - no JSON found
      - JSON unparseable
      - op missing or not in _VALID_OPS
      - content missing/empty for ops that require it
    """
    cleaned = re.sub(r"<think>.*?</think>", "", solution_str, flags=re.DOTALL).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return False
    try:
        d = json.loads(m.group())
    except json.JSONDecodeError:
        return False

    op = d.get("op", "")
    if op not in _VALID_OPS:
        return False

    if op not in _CONTENT_OPTIONAL_OPS and not d.get("content"):
        return False

    return True


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
    timeout=10 * MINUTES,
    volumes={MODELS_PATH: checkpoints_volume, "/hf-cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def eval(run_name: str = "locomo_lme_run_001", step: int = -1):
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
def prep_data():
    """Rebuild the parquet dataset only. Run this after changing the system prompt."""
    if not DATA_PATH_LOCAL.exists():
        print(f"Data not found at {DATA_PATH_LOCAL}")
        print("Run: python3 convert_calls.py")
        return
    jsonl_data = DATA_PATH_LOCAL.read_text()
    print(f"Loaded {jsonl_data.count(chr(10))} trajectories — rebuilding parquet...")
    prep.remote(jsonl_data=jsonl_data)
    print("Done. Run 'modal run train_modal.py::train_only' to start training.")


@app.local_entrypoint()
def sanity():
    """Sanity run — 20 steps on existing parquet. Confirms the full pipeline works (~$7).

    Run: python3.10 -m modal run train_modal.py::sanity
    """
    print("Starting sanity run (20 steps)...")
    jsonl_data = DATA_PATH_LOCAL.read_text()
    prep.remote(jsonl_data=jsonl_data)
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
        print("Run: python3 convert_calls.py")
        return

    jsonl_data = DATA_PATH_LOCAL.read_text()
    n_lines = jsonl_data.count("\n")
    print(f"Loaded {n_lines} trajectories — uploading and preparing dataset...")
    prep.remote(jsonl_data=jsonl_data)

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
                "extra_info":   {"question": question, "conv_text": conv_text[:2000]},
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


def _rouge1_recall(hypothesis: str, reference: str) -> float:
    """ROUGE-1 recall: fraction of reference tokens found in hypothesis."""
    hyp = _normalize(hypothesis)
    ref = _normalize(reference)
    if not ref:
        return 1.0
    if not hyp:
        return 0.0
    hyp_c = Counter(hyp)
    ref_c = Counter(ref)
    common = sum((hyp_c & ref_c).values())
    return common / len(ref)


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

    Returns the fraction of prefixes matched (1.0 = all hit), minus a penalty
    for keyword-stuffing (hitting prefixes from other outcome categories).
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
    recall = hits / len(keywords)

    # Anti-stuffing penalty: count how many OTHER outcome categories also fire.
    # A legitimate response should only match the correct category's keywords.
    other_categories_hit = sum(
        1 for other_label, other_kws in _LABEL_KEYWORDS.items()
        if other_label != label
        and all(any(t.startswith(kw) for t in content_tokens) for kw in other_kws)
    )
    penalty = 0.2 * other_categories_hit

    return max(0.0, recall - penalty)


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t not in _STOP]
