#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
exec /venv/main/bin/python -u /workspace/diffusion-pipe-easycontrol/tools/k2ab_sync.py
