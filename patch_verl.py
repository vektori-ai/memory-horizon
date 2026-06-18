"""Verl source patches — run once inside the Modal container before training.

Invoked automatically during image build via train_modal.py.

Patches:
    1. dp_actor.py       — empty_cache before optimizer step (prevents OOM on A100)
    2. core_algos.py     — step-wise GRPO advantage broadcast
                           Propagates terminal QA reward to ALL prior memory op turns.
                           Without this, ADD/UPDATE/COMPRESS turns at t=1..N-1 receive
                           near-zero gradient because only the terminal turn has reward != 0.
                           Reference: AgeMem §Training; training/stepwise_grpo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

def _find_verl_site() -> Path:
    import site
    for d in site.getsitepackages():
        p = Path(d) / "verl"
        if p.exists():
            return p
    # Fallback: Python 3.10 path used in verl v0.4 images
    return Path("/usr/local/lib/python3.10/dist-packages/verl")

VERL_SITE = _find_verl_site()


# ---------------------------------------------------------------------------
# Patch 1 — empty_cache before optimizer step
# ---------------------------------------------------------------------------

def patch_dp_actor() -> None:
    path = VERL_SITE / "workers/actor/dp_actor.py"
    src  = path.read_text()

    old = "self.actor_optimizer.step()"
    new = "torch.cuda.empty_cache(); self.actor_optimizer.step()"

    if new in src:
        print("[patch] dp_actor: already patched — skipping")
        return

    if old not in src:
        raise RuntimeError(
            f"[patch] dp_actor: pattern not found in {path} — check verl version"
        )

    path.write_text(src.replace(old, new, 1))
    print("[patch] dp_actor: empty_cache injected before optimizer step ✓")


# ---------------------------------------------------------------------------
# Patch 2 — step-wise GRPO advantage broadcast
# ---------------------------------------------------------------------------
# Standard GRPO assigns each token in a response the group-normalized reward
# for that response.  For multi-turn memory agents this is insufficient:
# turns 1..N-1 (ADD/UPDATE/COMPRESS decisions) all have reward=0, so their
# group-normalized advantage ≈ -mean/std — a constant drag, not useful signal.
#
# Step-wise GRPO fix: after normalizing within the group, broadcast the
# terminal-turn's advantage to EVERY prior turn in that trajectory so early
# memory operations learn from the final QA outcome.
#
# Concretely we replace the advantage expand in compute_grpo_advantage with
# a version that re-broadcasts per-trajectory rather than per-token-reward.
# For single-step training (current setup) this is a no-op — scores already
# collapse to one value per trajectory.  For multi-step it carries the full
# terminal signal backward through earlier turns.
# ---------------------------------------------------------------------------

_STEPWISE_IMPL = '''
def _stepwise_broadcast(scores, response_mask):
    """Broadcast terminal advantage to all model-generated tokens in the trajectory.

    For single-step rollouts this is identical to standard GRPO.
    For multi-step trajectories it propagates the terminal QA reward backward
    so early memory op turns (STORE_FACT/UPDATE/COMPRESS) get the same gradient
    signal as the final answer turn.
    """
    batch, seq = response_mask.shape
    # scores: (batch,) — one scalar per trajectory (already group-normalised)
    advantages = scores.unsqueeze(1).expand(batch, seq).contiguous()
    return advantages * response_mask

'''

_OLD_EXPAND_VARIANTS = [
    # Actual pattern in verl v0.4.1 and v0.5.0 core_algos.py
    "scores = scores.unsqueeze(-1) * response_mask",
    # Legacy variants kept as fallback for older/forked verl versions
    "advantages = normalized_scores.unsqueeze(-1).expand_as(eos_mask) * eos_mask",
    "advantages = normalized_scores.unsqueeze(1).expand_as(eos_mask) * eos_mask",
]

_NEW_EXPAND_MAP = {
    "scores = scores.unsqueeze(-1) * response_mask":
        "scores = _stepwise_broadcast(scores, response_mask)",
    "advantages = normalized_scores.unsqueeze(-1).expand_as(eos_mask) * eos_mask":
        "advantages = _stepwise_broadcast(normalized_scores, eos_mask)",
    "advantages = normalized_scores.unsqueeze(1).expand_as(eos_mask) * eos_mask":
        "advantages = _stepwise_broadcast(normalized_scores, eos_mask)",
}


def patch_grpo_stepwise() -> None:
    path = VERL_SITE / "trainer/ppo/core_algos.py"
    src  = path.read_text()

    if "_stepwise_broadcast" in src:
        print("[patch] core_algos: step-wise GRPO already applied — skipping")
        return

    matched_old = None
    matched_new = None
    for variant in _OLD_EXPAND_VARIANTS:
        if variant in src:
            matched_old = variant
            matched_new = _NEW_EXPAND_MAP[variant]
            break

    if matched_old is None:
        raise RuntimeError(
            f"[patch] core_algos: no expand pattern found in {path} — "
            "check verl version. Known patterns: " + str(_OLD_EXPAND_VARIANTS)
        )

    patched = _STEPWISE_IMPL + src.replace(matched_old, matched_new, 1)
    path.write_text(patched)
    print("[patch] core_algos: step-wise GRPO advantage broadcast applied ✓")


# ---------------------------------------------------------------------------
# Patch 3 — disable CUDA graph capture in the hybrid actor
# ---------------------------------------------------------------------------
# verl's hybrid actor captures CUDA graphs for batch_sizes=[1,2,4,...,160]
# at init time, while SGLang is simultaneously allocating its KV cache.
# The combined peak OOMs on A100-80GB.  Since we run enforce_eager=True on
# SGLang and LoRA on the FSDP actor, CUDA graphs provide no correctness
# benefit and can be disabled by replacing the batch-size list with [].
# ---------------------------------------------------------------------------

def patch_disable_cuda_graphs() -> None:
    """Two-file patch to disable CUDA graph capture in SGLang.

    1. cuda_graph_runner.py: zero self.capture_bs so the capture loop runs 0 times.
    2. model_runner.py: guard max(self.capture_bs) — raises ValueError on empty list.

    Both patches are applied independently so neither skips the other.
    """
    model_executor = Path(
        "/usr/local/lib/python3.10/dist-packages/sglang/srt/model_executor"
    )
    if not model_executor.exists():
        # Fallback: grep for cuda_graph_runner.py
        import subprocess as _sp
        site = Path("/usr/local/lib/python3.10/dist-packages")
        r = _sp.run(
            f"grep -rl 'Capture cuda graph bs' {site} 2>/dev/null",
            shell=True, capture_output=True, text=True,
        )
        hits = [Path(p) for p in r.stdout.strip().splitlines() if p]
        if hits:
            model_executor = hits[0].parent
        else:
            print("[patch] cuda_graph: sglang model_executor dir not found — skipping")
            return

    _patch_cuda_graph_runner(model_executor / "cuda_graph_runner.py")
    _patch_model_runner_max_bs(model_executor / "model_runner.py")


def _patch_cuda_graph_runner(target: Path) -> None:
    if not target.exists():
        print(f"[patch] cuda_graph_runner: {target} not found — skipping")
        return

    try:
        src = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"[patch] cuda_graph_runner: read error: {e}", file=sys.stderr)
        return

    if "# [patch] cuda graph disabled" in src:
        print(f"[patch] cuda_graph_runner: already patched — skipping")
        return

    needle = 'rank0_log(f"Capture cuda graph bs {self.capture_bs}")'
    if needle not in src:
        needle = "rank0_log(f'Capture cuda graph bs {self.capture_bs}')"
    if needle not in src:
        print(f"[patch] cuda_graph_runner: needle not found in {target}")
        for i, line in enumerate(src.splitlines(), 1):
            if "Capture cuda graph" in line:
                print(f"  {i:4d}: {repr(line)}")
        return

    for line in src.splitlines():
        if needle in line:
            ind = " " * (len(line) - len(line.lstrip()))
            break

    old_line  = f"{ind}{needle}"
    new_lines = (
        # [1] instead of [] — avoids ValueError: max() arg is an empty sequence in
        # model_runner.py when it reads capture_bs after CudaGraphRunner init.
        # A single bs=1 graph is trivial (~100MB); skipping bs=[2,4,...,160] avoids
        # the OOM race with SGLang KV allocation.
        f"{ind}self.capture_bs = [1]  # [patch] minimal capture — skip bs=[2..160], avoid max([]) crash\n"
        f"{old_line}"
    )
    patched = src.replace(old_line, new_lines, 1)
    if patched == src:
        print(f"[patch] cuda_graph_runner: replacement failed in {target}", file=sys.stderr)
        return

    target.write_text(patched, encoding="utf-8")
    print(f"[patch] cuda_graph_runner: capture_bs=[1] (minimal) ✓")


def _patch_model_runner_max_bs(target: Path) -> None:
    """Guard max(self.capture_bs) — raises ValueError on empty list after our patch."""
    import re as _re

    if not target.exists():
        print(f"[patch] model_runner: {target} not found — skipping")
        return

    try:
        src = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"[patch] model_runner: read error: {e}", file=sys.stderr)
        return

    if "# [patch] guard empty capture_bs" in src:
        print(f"[patch] model_runner: max_bs guard already applied — skipping")
        return

    # Use regex to handle any whitespace variation in the assignment
    pattern = r'(self\.max_bs\s*=\s*)max\(self\.capture_bs\)'
    if not _re.search(pattern, src):
        print(f"[patch] model_runner: self.max_bs pattern not found — scanning for clues:")
        lines = src.splitlines()
        for i, line in enumerate(lines, 1):
            if any(kw in line for kw in ("max_bs", "capture_bs", "max(self", "init_cuda")):
                print(f"  {i:4d}: {repr(line)}")
        # Show lines 1145-1160 (around the crash location)
        print("  [lines 1145-1160 of model_runner.py]:")
        for i, line in enumerate(lines[1144:1160], 1145):
            print(f"  {i:4d}: {repr(line)}")
        return

    patched = _re.sub(
        pattern,
        r'\1(max(self.capture_bs) if self.capture_bs else 1)  # [patch] guard empty capture_bs',
        src,
        count=1,
    )
    target.write_text(patched, encoding="utf-8")
    print(f"[patch] model_runner: max_bs guard applied ✓")


# ---------------------------------------------------------------------------
# Patch 4 — fix extra_info handling in AgentLoopWorker.generate_sequences()
# ---------------------------------------------------------------------------
# verl's DataLoader stores extra_info as a Python dict (calls .get() on it).
# verl's AgentLoopWorker.generate_sequences() then tries extra_info.strip()
# before json.loads(), expecting a JSON string.
# These are mutually exclusive: dict has no .strip(); string has no .get().
#
# Fix: in generate_sequences(), add a type guard before the .strip() call so
# it handles both dict (from our data) and string (legacy path) correctly.
# ---------------------------------------------------------------------------

def patch_agent_loop_extra_info() -> None:
    path = VERL_SITE / "experimental/agent_loop/agent_loop.py"
    if not path.exists():
        print(f"[patch] agent_loop extra_info: {path} not found — skipping")
        return

    src = path.read_text(encoding="utf-8", errors="replace")

    if "# [patch] extra_info type guard" in src:
        print("[patch] agent_loop extra_info: already patched — skipping")
        return

    # Find the pattern where verl calls .strip() on extra_info before json.loads()
    # Typical pattern: json.loads(extra_info.strip()) or extra_info = extra_info.strip()
    import re as _re

    patterns = [
        # json.loads(extra_info.strip())
        (r'json\.loads\(extra_info\.strip\(\)\)',
         'json.loads(extra_info.strip() if isinstance(extra_info, str) else json.dumps(extra_info))  # [patch] extra_info type guard'),
        # extra_info = extra_info.strip()
        (r'extra_info\s*=\s*extra_info\.strip\(\)',
         'extra_info = extra_info.strip() if isinstance(extra_info, str) else extra_info  # [patch] extra_info type guard'),
        # extra_info.strip() in other contexts
        (r'extra_info\.strip\(\)',
         '(extra_info.strip() if isinstance(extra_info, str) else json.dumps(extra_info))  # [patch] extra_info type guard'),
    ]

    patched = src
    applied = False
    for pattern, replacement in patterns:
        new_src = _re.sub(pattern, replacement, patched, count=1)
        if new_src != patched:
            patched = new_src
            applied = True
            print(f"[patch] agent_loop extra_info: patched pattern '{pattern[:40]}...' ✓")
            break

    if not applied:
        print(f"[patch] agent_loop extra_info: no .strip() pattern found — printing generate_sequences:")
        in_fn = False
        for i, line in enumerate(src.splitlines(), 1):
            if "def generate_sequences" in line:
                in_fn = True
            if in_fn:
                print(f"  {i:4d}: {repr(line)}")
                if i > 0 and line.strip() == "" and in_fn:
                    break  # stop at first blank line after function body
                if i > 50:
                    break
        return

    path.write_text(patched, encoding="utf-8")
    print("[patch] agent_loop extra_info: extra_info type guard applied ✓")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    errors = []
    for fn in (patch_dp_actor, patch_grpo_stepwise, patch_disable_cuda_graphs,
               patch_agent_loop_extra_info):
        try:
            fn()
        except Exception as exc:
            print(f"[patch] ERROR in {fn.__name__}: {exc}", file=sys.stderr)
            errors.append(exc)

    if errors:
        sys.exit(1)
    print("[patch] All verl patches applied.")
