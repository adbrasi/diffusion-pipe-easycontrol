#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
export KREA2_CAMPAIGN_ARTIFACTS=/workspace/k2ab/artifacts/fullbudget_lr1e4_20261001
export KREA2_CAMPAIGN_OUTPUT=/workspace/k2ab/checkpoints/A_native_fullbudget_lr1e4_fromscratch_5000
export KREA2_CAMPAIGN_LR=0.0001
export KREA2_CAMPAIGN_JOB_PREFIX=fullbudget_lr1e4
cd /workspace/diffusion-pipe-easycontrol
exec /venv/main/bin/python -u tools/krea2_fullbudget_campaign.py
