#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
cd /workspace/ComfyUI_stock
exec /workspace/k2ab/stock_venv/bin/python -u main.py --listen 127.0.0.1 --port 18819 --disable-all-custom-nodes --disable-api-nodes --models-directory /workspace/models/krea2 --input-directory /workspace/k2ab/heldout/control --output-directory /workspace/k2ab/artifacts/stock_outputs --user-directory /workspace/k2ab/stock_user --preview-method none
