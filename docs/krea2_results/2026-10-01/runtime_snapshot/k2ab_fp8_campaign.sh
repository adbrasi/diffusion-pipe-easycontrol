#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/diffusion-pipe-easycontrol
exec /venv/main/bin/python -u tools/k2ab_fp8_campaign.py --micro-batch 2
