#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/k2ab/ComfyUI_legacy
exec /workspace/k2ab/stock_venv/bin/python -u main.py --listen 127.0.0.1 --port 18820 --disable-all-custom-nodes --whitelist-custom-nodes ctxrush_legacy --disable-api-nodes --models-directory /workspace/models/krea2 --extra-model-paths-config /workspace/k2ab/legacy_models.yaml --input-directory /workspace/k2ab/heldout/control --output-directory /workspace/k2ab/artifacts/legacy_outputs --user-directory /workspace/k2ab/legacy_user --preview-method none
