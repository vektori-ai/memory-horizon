"""Frozen QA model — Memory-R1 pattern for Tier 3 verification.

The frozen QA model is a separate Qwen-7B instance (frozen weights, eval mode)
used to answer QA probes using only the memory manager's stored memory as context.
Its correctness is the primary reward signal for Base 1 (Store) and Base 7 (Integration).

The model is deterministic relative to the training run — frozen weights + greedy decode
means the same (question, context) always produces the same answer, making it a valid
verifier without introducing gradient noise.

Mock mode:
    FrozenQAModel(mock=True) uses token-overlap heuristics instead of a real model.
    Use this during development, CI, and when no GPU is available.
    The mock is intentionally conservative (under-rewards) to avoid inflating metrics.

Usage::

    # Production (requires GPU + transformers):
    model = FrozenQAModel(model_name="Qwen/Qwen2.5-7B-Instruct")
    answer = model.answer("What is the customer's name?", context="[user_name] Alice")
    # → "Alice"

    # Development / CI:
    model = FrozenQAModel(mock=True)
    answer = model.answer("What is the customer's name?", context="[user_name] Alice")
"""

from __future__ import annotations

import os
import re
import string
from collections import Counter
from typing import Any

_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
_MAX_NEW_TOKENS = 128
_QA_SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer using ONLY the information in the provided memory context. "
    "If the answer is not in the context, say: 'I don't know'. "
    "Be concise — one sentence or less."
)


class FrozenQAModel:
    """Frozen Qwen-7B instance used as a Tier 3 verifier.

    Args:
        model_name: HuggingFace model ID. Default: Qwen/Qwen2.5-7B-Instruct.
        device: "cuda", "cpu", or "auto". Default: "auto".
        mock: If True, skip model loading and use token-overlap heuristic.
              Useful for CI and development without GPU.
        cache_dir: Optional directory for HF model cache.
        max_new_tokens: Max tokens to generate per answer.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "auto",
        mock: bool = False,
        cache_dir: str | None = None,
        max_new_tokens: int = _MAX_NEW_TOKENS,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.mock = mock
        self.cache_dir = cache_dir or os.environ.get("HF_HOME")
        self.max_new_tokens = max_new_tokens

        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False

        if not mock:
            self._load()

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device,
                trust_remote_code=True,
            )
            # Freeze all parameters — this model is a verifier, never trained.
            for param in self._model.parameters():
                param.requires_grad_(False)
            self._model.eval()
            self._loaded = True
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for FrozenQAModel. "
                "Install with: pip install 'memory-horizon[training]'\n"
                f"Original error: {e}"
            )

    def answer(self, question: str, context: str) -> str:
        """Answer a question using only the provided memory context.

        Args:
            question: The question to answer.
            context: Memory context (output of MemoryStore.get_context()).

        Returns:
            A concise answer string, or "I don't know" if not in context.
        """
        if self.mock:
            return self._mock_answer(question, context)

        if not self._loaded:
            raise RuntimeError("Model not loaded. Use mock=True for testing.")

        import torch

        prompt = self._build_prompt(question, context)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,       # greedy — deterministic verifier
                temperature=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the new tokens.
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return raw.strip()

    def answer_batch(self, qa_pairs: list[tuple[str, str]]) -> list[str]:
        """Answer multiple (question, context) pairs in one forward pass.

        Args:
            qa_pairs: List of (question, context) tuples.

        Returns:
            List of answer strings, one per input pair.
        """
        if self.mock:
            return [self._mock_answer(q, c) for q, c in qa_pairs]

        if not self._loaded:
            raise RuntimeError("Model not loaded.")

        import torch

        prompts = [self._build_prompt(q, c) for q, c in qa_pairs]
        encodings = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self._model.device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **encodings,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        answers: list[str] = []
        for i, out in enumerate(output_ids):
            new_tokens = out[encodings["input_ids"].shape[1]:]
            raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            answers.append(raw.strip())
        return answers

    def as_callable(self):
        """Return self.answer as a plain callable for Tier 3 score_with_qa_model."""
        return self.answer

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _build_prompt(self, question: str, context: str) -> str:
        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": _QA_SYSTEM_PROMPT},
                {"role": "user", "content": f"Memory context:\n{context}\n\nQuestion: {question}"},
            ]
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Fallback plain-text prompt.
        return (
            f"System: {_QA_SYSTEM_PROMPT}\n\n"
            f"Memory context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )

    def _mock_answer(self, question: str, context: str) -> str:
        """Heuristic mock: find the context sentence with most keyword overlap with question.

        Conservative: returns "I don't know" if overlap is below threshold.
        """
        q_tokens = _normalize_tokens(question)
        if not q_tokens:
            return "I don't know"

        # Extract fact lines from context.
        lines = [ln.strip() for ln in context.split("\n") if ln.strip() and not ln.startswith("#")]
        if not lines:
            return "I don't know"

        best_line = ""
        best_overlap = 0

        for line in lines:
            line_tokens = _normalize_tokens(line)
            overlap = len(q_tokens & line_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_line = line

        if best_overlap < 2:
            return "I don't know"

        # Extract the value part after ']' if it's a fact line like "[key] value".
        value_match = re.search(r"\]\s*(.+)", best_line)
        if value_match:
            return value_match.group(1).strip()

        return best_line


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_tokens(text: str) -> set[str]:
    # Split on hyphens and underscores so "user_name" → "user name" before stop removal.
    text = text.lower().replace("-", " ").replace("_", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    _STOP = {"what", "is", "the", "are", "was", "were", "a", "an", "this",
             "that", "of", "for", "in", "on", "at", "to", "does", "did",
             "has", "have", "had", "agent"}
    return set(text.split()) - _STOP
