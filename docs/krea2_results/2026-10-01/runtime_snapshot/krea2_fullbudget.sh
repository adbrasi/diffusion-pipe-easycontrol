#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/diffusion-pipe-easycontrol
exec /venv/main/bin/python -u tools/krea2_fullbudget_campaign.py
