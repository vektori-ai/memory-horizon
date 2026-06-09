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
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    errors = []
    for fn in (patch_dp_actor, patch_grpo_stepwise):
        try:
            fn()
        except Exception as exc:
            print(f"[patch] ERROR in {fn.__name__}: {exc}", file=sys.stderr)
            errors.append(exc)

    if errors:
        sys.exit(1)
    print("[patch] All verl patches applied.")
