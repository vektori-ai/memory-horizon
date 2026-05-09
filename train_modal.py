"""GRPO memory training on Modal — Qwen3-8B.

Run locally to kick off a Modal job:
    modal run train_modal.py

What this does:
    1. Uploads the trajectory JSONL from data/debt_collection.jsonl
    2. Trains Qwen3-8B with GRPO on a single A100-80GB
    3. Reward = token F1 between model's stored memory content and gold QA answer
    4. Saves checkpoint to a Modal Volume

Reward design:
    - Input to model: conversation transcript + QA question
    - Model outputs: memory op JSON {"op": ..., "content": ..., "key": ...}
    - Reward: token_f1(content_field, gold_answer) — fully verifiable, no LLM judge
    - Episode passes if F1 >= 0.7 → reward = +1, else proportional

This is the minimal RLVR formulation: verifiable reward, no shaped signal.
"""

from __future__ import annotations

import json
import os
import re
import string
from collections import Counter
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal config
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent
DATA_PATH = REPO_ROOT / "data" / "debt_collection.jsonl"

MODEL_ID    = "Qwen/Qwen3-8B"
BATCH_SIZE  = 4       # GRPO group size (completions per prompt)
N_EPOCHS    = 1
MAX_STEPS   = 200     # cap for a quick first run — bump to 1000+ for real training
LR          = 2e-6
MAX_SEQ_LEN = 2048

volume = modal.Volume.from_name("memory-rlvr-checkpoints", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("packaging")  # flash-attn setup.py needs this before torch
    .pip_install(              # flash-attn setup.py also needs torch present at metadata time
        "torch==2.4.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.45.0",
        "trl>=0.12.0",
        "accelerate>=0.34.0",
        "peft>=0.13.0",
        "datasets>=2.20.0",
        "vllm==0.6.3",
        "flash-attn>=2.6.3",
        "sentencepiece",
        "bitsandbytes",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({"HF_HOME": "/hf-cache", "TOKENIZERS_PARALLELISM": "false"})
)

app = modal.App("memory-rlvr")


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=60 * 60 * 6,
    volumes={
        "/checkpoints": volume,
        "/hf-cache": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],  # HF_TOKEN for gated models
)
def train(jsonl_data: str, run_name: str = "run_001") -> dict:
    """Main training function. jsonl_data is the raw JSONL string."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Load tokenizer + model ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # --- Build dataset ---
    examples = _build_grpo_examples(jsonl_data)
    print(f"Dataset size: {len(examples)} examples")
    dataset = Dataset.from_list(examples)

    # --- Reward function ---
    def reward_fn(completions: list[str], gold_answers: list[str], **_) -> list[float]:
        rewards = []
        for completion, gold in zip(completions, gold_answers):
            content = _extract_content_field(completion)
            f1 = _token_f1(content, gold)
            # Scale: F1 >= 0.7 → +1.0, else proportional in [0, 1)
            rewards.append(1.0 if f1 >= 0.7 else f1)
        return rewards

    # --- GRPO config ---
    training_args = GRPOConfig(
        output_dir=f"/checkpoints/{run_name}",
        num_train_epochs=N_EPOCHS,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=BATCH_SIZE,         # rollouts per prompt (GRPO group size)
        learning_rate=LR,
        bf16=True,
        logging_steps=10,
        save_steps=50,
        max_completion_length=256,
        max_prompt_length=MAX_SEQ_LEN,
        temperature=0.9,
        report_to="none",                   # swap to "wandb" when ready
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
    )

    print(f"Starting GRPO training — {MAX_STEPS} steps, group size {BATCH_SIZE}")
    trainer.train()

    save_path = f"/checkpoints/{run_name}/final"
    trainer.save_model(save_path)
    volume.commit()
    print(f"Checkpoint saved to {save_path}")

    return {"run_name": run_name, "steps": MAX_STEPS, "checkpoint": save_path}


# ---------------------------------------------------------------------------
# Local entrypoint — runs when you do `modal run train_modal.py`
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    if not DATA_PATH.exists():
        print(f"Data not found at {DATA_PATH}")
        print("Run the converter first:  python3 convert_calls.py")
        return

    jsonl_data = DATA_PATH.read_text()
    examples = _build_grpo_examples(jsonl_data)
    print(f"Loaded {jsonl_data.count(chr(10))} trajectories → {len(examples)} GRPO examples")

    if len(examples) < 10:
        print("WARNING: very few examples — add more call data before a real run")

    result = train.remote(jsonl_data=jsonl_data, run_name="debt_collection_run_001")
    print("Training complete:", result)


# ---------------------------------------------------------------------------
# Data formatting helpers (run locally before upload)
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
"""


def _build_grpo_examples(jsonl_data: str) -> list[dict]:
    """Convert trajectory JSONL → list of GRPO training examples.

    Each QA probe in each trajectory becomes one training example:
      - prompt: system + conversation transcript + question
      - gold_answer: the expected answer (for reward computation)
    """
    examples = []
    for line in jsonl_data.strip().split("\n"):
        if not line.strip():
            continue
        traj = json.loads(line)

        # Build the conversation text from the single session.
        sessions = traj.get("sessions", [])
        if not sessions:
            continue
        turns = sessions[0].get("turns", [])
        conv_text = _format_conversation(turns)

        # One training example per QA probe.
        for probe in traj.get("qa_probes", []):
            question = probe.get("question", "")
            answer = probe.get("answer", "")
            if not question or not answer or answer == "ABSTAIN":
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
                "prompt": prompt,
                "gold_answer": answer,
            })

    return examples


def _format_conversation(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        role = "Agent" if t.get("role") == "assistant" else "Customer"
        content = t.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def _extract_content_field(completion: str) -> str:
    """Extract the 'content' value from a memory op JSON string."""
    # Try JSON parse first.
    try:
        obj = json.loads(completion.strip())
        return str(obj.get("content", ""))
    except (json.JSONDecodeError, AttributeError):
        pass
    # Fallback: regex extract.
    m = re.search(r'"content"\s*:\s*"([^"]+)"', completion)
    if m:
        return m.group(1)
    return completion  # treat whole completion as content if JSON fails


def _token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold answer."""
    pred_tokens = _normalize(prediction)
    gold_tokens = _normalize(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


_STOP = {"what", "is", "the", "are", "was", "were", "a", "an", "this", "that",
         "of", "for", "in", "on", "at", "to", "does", "did", "has", "have", "had"}


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t not in _STOP]
