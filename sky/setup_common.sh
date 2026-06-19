#!/bin/bash
# Shared image setup for all gpt-oss-20b SkyPilot tasks.
#
# Starts from the same base image train_modal.py used on Modal
# (verlai/verl:app-verl0.5-sglang0.4.8-mcore0.12.2-te2.2) and upgrades ONLY
# SGLang in place to >=0.5.1.post3 (gpt-oss support landed at 0.4.10, 0.5.1.post3+
# recommended) -- verl stays pinned at 0.5.0 on purpose, see the plan's
# "Software compatibility" section for why patch_verl.py is fragile against a
# verl version bump (patch #2/#3 string-match exact verl 0.5.0 / older-SGLang
# source). This script is the actual compat-gate test of whether that
# in-place SGLang bump works without breaking the rest of the image.
set -euo pipefail

pip install --no-deps verl==0.5.0
pip install -U "sglang[all]>=0.5.1.post3"

cd /root/repo
python3 patch_verl.py
echo "[setup] patch_verl.py exit code: $?"
