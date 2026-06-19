#!/bin/bash
# Shared GRPO launch logic for gpt-oss-20b, parametrized by env vars set in
# each task YAML's `envs:` block. Mirrors train_modal.py::train's verl CLI
# args (same flags, same values) except: MODEL_ID, gpu_memory_utilization,
# n_gpus_per_node, train_batch_size scale with N_GPUS -- see the plan's
# "GPU instance choice" / "Maximizing GPU utilization" sections for why.
set -euo pipefail

N_GPUS="${N_GPUS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
N_ROLLOUTS="${N_ROLLOUTS:-2}"
N_STEPS="${N_STEPS:-20}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.25}"
RUN_NAME="${RUN_NAME:-gpt_oss_20b_aws_001}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-openai/gpt-oss-20b}"
DATA_DIR="${DATA_DIR:-/mnt/mh-data}"
MODELS_DIR="${MODELS_DIR:-/mnt/mh-models}"

cd /root/repo

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/val.parquet" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length=2048 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation=right \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="${BASE_MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-4 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enforce_eager=True \
  ++actor_rollout_ref.rollout.engine_kwargs.sglang.disable_cuda_graph=true \
  actor_rollout_ref.rollout.multi_stage_wake_up=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.n="${N_ROLLOUTS}" \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=/root/repo/agent_loop_config.yaml \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  trainer.critic_warmup=0 \
  trainer.val_before_train=False \
  trainer.logger="['console','wandb']" \
  trainer.project_name=memory-rlvr-aws \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes=1 \
  trainer.test_freq=$((N_STEPS + 1)) \
  trainer.save_freq=$((N_STEPS < 25 ? N_STEPS : 25)) \
  trainer.total_training_steps="${N_STEPS}" \
  trainer.default_local_dir="${MODELS_DIR}/${RUN_NAME}" \
  trainer.resume_mode=auto \
  custom_reward_function.path=/root/repo/reward.py \
  custom_reward_function.name=compute_reward
