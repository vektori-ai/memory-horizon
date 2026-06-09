"""Step-wise GRPO advantage broadcast — AgeMem credit assignment trick.

Problem with standard GRPO for memory agents:
    A memory decision at turn 3 only gets rewarded at turn 40 (QA phase).
    Standard GRPO assigns advantage = 0 to all non-terminal steps.
    Gradient signal for early memory ops is nearly zero.

Solution (AgeMem §Training):
    Compute terminal advantage normally (group-relative normalization).
    Then broadcast it to EVERY prior step in the same trajectory.
    Every ADD/UPDATE/COMPRESS decision gets the final QA reward as its signal.

This is a ~10 line patch to verl's advantage computation.

Usage (in train_modal.py, passed to verl config):
    "algorithm.adv_estimator=grpo"  ← keep standard GRPO
    Then post-process advantages with broadcast_terminal_advantage()
    before the policy gradient update.

Standalone usage for testing:
    from training.stepwise_grpo import broadcast_terminal_advantage
    advantages = broadcast_terminal_advantage(rewards, terminals)
"""

from __future__ import annotations

import torch


def broadcast_terminal_advantage(
    rewards: torch.Tensor,
    terminals: torch.Tensor,
    group_size: int = 8,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Broadcast terminal advantage to all steps in each trajectory.

    Args:
        rewards:    Shape (batch, seq_len). Reward at each step.
                    Non-terminal steps have reward=0; terminal step has episode reward.
        terminals:  Shape (batch, seq_len). 1.0 at the terminal step, 0.0 elsewhere.
        group_size: Number of rollouts per query (G in GRPO). Used for within-group norm.
        eps:        Stability epsilon for normalization.

    Returns:
        advantages: Shape (batch, seq_len).
                    All steps in a trajectory share the terminal advantage.
    """
    batch_size, seq_len = rewards.shape

    # Extract terminal rewards — one per trajectory
    terminal_rewards = (rewards * terminals).sum(dim=1)  # (batch,)

    # Within-group normalization (standard GRPO)
    # Reshape to (n_queries, group_size) for group-relative advantage
    n_queries = batch_size // group_size
    grouped = terminal_rewards.view(n_queries, group_size)

    group_mean = grouped.mean(dim=1, keepdim=True)   # (n_queries, 1)
    group_std  = grouped.std(dim=1, keepdim=True)    # (n_queries, 1)

    normalized = (grouped - group_mean) / (group_std + eps)  # (n_queries, group_size)
    terminal_advantages = normalized.view(batch_size)         # (batch,)

    # Broadcast to all steps — every step in trajectory i gets terminal_advantages[i]
    advantages = terminal_advantages.unsqueeze(1).expand(-1, seq_len)  # (batch, seq_len)

    return advantages.contiguous()


def patch_verl_grpo(trainer) -> None:
    """Monkey-patch a verl PPOTrainer to use step-wise advantage broadcast.

    Call this after constructing the trainer, before training starts:
        trainer = PPOTrainer(...)
        patch_verl_grpo(trainer)
        trainer.fit()

    The patch replaces the advantage computation in the actor's loss function
    with broadcast_terminal_advantage.
    """
    original_compute_advantages = getattr(
        trainer, "_compute_advantages", None
    ) or getattr(trainer, "compute_advantages", None)

    if original_compute_advantages is None:
        raise RuntimeError(
            "Could not find advantage computation method on trainer. "
            "Check verl version — expected _compute_advantages or compute_advantages."
        )

    def _stepwise_advantages(rewards, terminals, *args, **kwargs):
        group_size = getattr(trainer, "config", {}).get(
            "algorithm.n_samples_per_prompt", 8
        )
        return broadcast_terminal_advantage(rewards, terminals, group_size=group_size)

    # Replace the method
    method_name = (
        "_compute_advantages"
        if hasattr(trainer, "_compute_advantages")
        else "compute_advantages"
    )
    setattr(trainer, method_name, _stepwise_advantages)
    print(f"[stepwise_grpo] Patched {method_name} on {type(trainer).__name__}")
