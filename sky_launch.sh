#!/bin/bash
# Thin wrapper around `sky launch`, mirroring train_modal.py's
# @app.local_entrypoint() functions for the AWS/SkyPilot path.
#
# Usage:
#   ./sky_launch.sh prep
#   ./sky_launch.sh compat-gate
#   ./sky_launch.sh sanity
#   ./sky_launch.sh probe          # 8-GPU utilization probe, ~10-20 steps
#   ./sky_launch.sh train [N_STEPS] # 8-GPU real run, default 500 steps
#
# Requires HF_TOKEN and WANDB_API_KEY set in your local shell env --
# SkyPilot reads them via --secret at launch time (see plan's "Resolved"
# section on secrets handling).
set -euo pipefail

cmd="${1:-}"

case "$cmd" in
  prep)
    sky launch -c mh-data sky/prep.yaml
    ;;
  compat-gate)
    sky launch -c mh-compat-gate sky/compat_gate.yaml --secret HF_TOKEN
    ;;
  sanity)
    sky launch -c mh-sanity sky/gpt_oss_sanity.yaml --secret HF_TOKEN --secret WANDB_API_KEY
    ;;
  probe)
    sky launch -c mh-train sky/gpt_oss_train.yaml --secret HF_TOKEN --secret WANDB_API_KEY
    ;;
  train)
    n_steps="${2:-500}"
    sky launch -c mh-train sky/gpt_oss_train.yaml --env N_STEPS="${n_steps}" \
      --secret HF_TOKEN --secret WANDB_API_KEY
    ;;
  stop)
    sky stop mh-sanity mh-train mh-compat-gate 2>/dev/null || true
    ;;
  *)
    echo "Usage: $0 {prep|compat-gate|sanity|probe|train [n_steps]|stop}"
    exit 1
    ;;
esac
