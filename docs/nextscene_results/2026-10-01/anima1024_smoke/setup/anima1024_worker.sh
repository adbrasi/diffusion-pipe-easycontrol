#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/diffusion-pipe-easycontrol
exec /venv/main/bin/python -u /workspace/nextscene_artifacts/anima1024_20261001/worker.py
