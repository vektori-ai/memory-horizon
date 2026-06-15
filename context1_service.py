"""Context-1 retrieval service on Modal.

Deploys chromadb/context-1 (20B MoE) via vLLM and exposes a simple retrieve endpoint.
Called from MemoryAgentLoop during rollouts to find relevant FS entries for probe questions.

Note: Context-1's full agent harness is not yet public. We implement a simplified
single-pass search: given a query + list of FS entries, Context-1 selects the relevant ones.

Deploy:
    modal deploy context1_service.py

The deployed URL is passed to train_modal.py via CONTEXT1_SERVICE_URL env var.
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# Image — vLLM with Context-1 weights baked in
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.6.0", "fastapi", "uvicorn", "huggingface_hub")
    .env({"HF_HOME": "/hf-cache", "TOKENIZERS_PARALLELISM": "false"})
)

app        = modal.App("context1-retrieval")
hf_cache   = modal.Volume.from_name("hf-model-cache", create_if_missing=True)

MODEL_ID   = "chromadb/context-1"
TOP_K      = 5
MAX_DOCS   = 40   # cap to keep prompt within context window

# ---------------------------------------------------------------------------
# Retrieval logic
# ---------------------------------------------------------------------------

_RETRIEVE_SYSTEM = """\
You are a search agent. Given a question and a numbered list of memory entries, \
identify which entries are relevant to answering the question.
Reply with ONLY a comma-separated list of entry numbers (e.g. "1,3,7").
If none are relevant, reply "none"."""


def _build_prompt(query: str, docs: list[str]) -> str:
    numbered = "\n".join(f"{i+1}. {d}" for i, d in enumerate(docs[:MAX_DOCS]))
    return f"Question: {query}\n\nMemory entries:\n{numbered}\n\nRelevant entry numbers:"


def _parse_selected(response: str, n_docs: int) -> list[int]:
    """Parse '1,3,7' → [0, 2, 6] (0-indexed). Clamp to valid range."""
    indices = []
    for tok in response.replace(" ", "").split(","):
        try:
            idx = int(tok) - 1
            if 0 <= idx < n_docs:
                indices.append(idx)
        except ValueError:
            continue
    return indices


# ---------------------------------------------------------------------------
# Modal class — keeps vLLM engine warm between calls
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    gpu="A100-80GB",
    volumes={"/hf-cache": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=10 * 60,
    scaledown_window=300,
)
class Context1Retriever:

    @modal.enter()
    def load(self):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=MODEL_ID,
            tensor_parallel_size=1,
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.85,
        )
        self.sampling = SamplingParams(
            temperature=0.0,
            max_tokens=64,
        )
        print(f"[context1] loaded {MODEL_ID}")

    @modal.method()
    def retrieve(self, query: str, documents: list[str], top_k: int = TOP_K) -> list[str]:
        """Return up to top_k relevant documents from the provided list."""
        if not documents:
            return []

        docs = documents[:MAX_DOCS]
        prompt = _build_prompt(query, docs)
        messages = [
            {"role": "system", "content": _RETRIEVE_SYSTEM},
            {"role": "user",   "content": prompt},
        ]

        from vllm import SamplingParams
        # Use tokenizer to apply chat template
        tokenizer = self.llm.get_tokenizer()
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = self.llm.generate([text], self.sampling)
        response = outputs[0].outputs[0].text.strip()

        indices = _parse_selected(response, len(docs))

        # Fall back to first top_k if parsing fails
        if not indices:
            indices = list(range(min(top_k, len(docs))))

        return [docs[i] for i in indices[:top_k]]


# ---------------------------------------------------------------------------
# FastAPI wrapper for HTTP calls from the training cluster
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=None,
    timeout=60,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web() -> None:
    from fastapi import FastAPI
    from pydantic import BaseModel

    api = FastAPI()
    retriever = Context1Retriever()

    class RetrieveRequest(BaseModel):
        query: str
        documents: list[str]
        top_k: int = TOP_K

    @api.post("/retrieve")
    def retrieve(req: RetrieveRequest) -> dict:
        results = retriever.retrieve.remote(req.query, req.documents, req.top_k)
        return {"results": results}

    @api.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return api
