#!/bin/bash
until grep -q OSTRIS_DONE /workspace/krea2/run_ostris.log; do sleep 20; done
source /workspace/venv/bin/activate; cd /workspace/ai-toolkit
rm -f /workspace/evals/engine/engine.json
python run.py /workspace/evals/engine.yaml > /workspace/evals/engine_ostris.log 2>&1 &
EPID=$!
until [ -f /workspace/evals/engine/engine.json ]; do sleep 5; kill -0 $EPID || { echo ENGINE_DIED; exit 1; }; done; sleep 5
python /workspace/evals/run_tests.py /workspace/evals/ostris_prefix_test.json /workspace/output/krea2_ostris_edit_cap1000/krea2_ostris_edit_cap1000.safetensors /workspace/evals/ostris_prefix
kill $EPID; sleep 10; kill $(pgrep -f "python ru[n].py /workspace/evals/engine.yaml") 2>/dev/null
python /workspace/evals/grid_prefix.py && echo PREFIX_TEST_DONE
python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = "AdwolfCzar/moss-image-nextscene-lora"
api.upload_file(repo_id=repo, path_or_fileobj="/workspace/outputs_for_user/KREA2_OSTRIS_com_vs_sem_prefixo.jpg", path_in_repo="krea2_ostris_edit_cap1000/KREA2_OSTRIS_com_vs_sem_prefixo.jpg")
api.upload_folder(repo_id=repo, folder_path="/workspace/evals/ostris_prefix", path_in_repo="krea2_ostris_edit_cap1000/teste_prefixo")
print("HF_UPLOAD_OK")
PY
