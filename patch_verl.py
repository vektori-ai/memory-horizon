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

VERL_SITE = Path("/usr/local/lib/python3.10/dist-packages/verl")


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
def _stepwise_broadcast(normalized_scores, eos_mask):
    """Broadcast terminal advantage to all steps in each trajectory.

    For single-step rollouts this is identical to standard GRPO.
    For multi-step trajectories it propagates the terminal reward backward
    so early memory operations (ADD/UPDATE/COMPRESS) get the same gradient
    signal as the final QA answer turn.
    """
    batch, seq = eos_mask.shape
    # normalized_scores: (batch,) — one scalar per trajectory (already group-normalised)
    advantages = normalized_scores.unsqueeze(1).expand(batch, seq).contiguous()
    return advantages * eos_mask

'''

_OLD_EXPAND_VARIANTS = [
    "advantages = normalized_scores.unsqueeze(-1).expand_as(eos_mask) * eos_mask",
    "advantages = normalized_scores.unsqueeze(1).expand_as(eos_mask) * eos_mask",
]

_NEW_EXPAND = "advantages = _stepwise_broadcast(normalized_scores, eos_mask)"


def patch_grpo_stepwise() -> None:
    path = VERL_SITE / "trainer/ppo/core_algos.py"
    src  = path.read_text()

    if "_stepwise_broadcast" in src:
        print("[patch] core_algos: step-wise GRPO already applied — skipping")
        return

    old = None
    for variant in _OLD_EXPAND_VARIANTS:
        if variant in src:
            old = variant
            break

    if old is None:
        print(
            "[patch] core_algos: expand pattern not found — "
            "skipping step-wise GRPO (check verl version; may already broadcast correctly)"
        )
        return

    patched = _STEPWISE_IMPL + src.replace(old, _NEW_EXPAND, 1)
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
