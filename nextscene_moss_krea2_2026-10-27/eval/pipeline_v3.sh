#!/bin/bash
source /workspace/venv/bin/activate
while pgrep -f "run_v3.sh" >/dev/null; do sleep 15; done
LORA=/workspace/output/moss_nextscene_v3_cap1000/moss_nextscene_v3_cap1000.safetensors
ls -la $LORA || exit 1
rm -f /workspace/evals/engine/engine.json
cd /workspace/ai-toolkit; python run.py /workspace/evals/engine.yaml > /workspace/evals/engine_v3.log 2>&1 &
EPID=$!
until [ -f /workspace/evals/engine/engine.json ]; do sleep 5; kill -0 $EPID || { echo ENGINE_DIED; exit 1; }; done; sleep 5
python /workspace/evals/gen_refs.py
python /workspace/evals/run_tests.py /workspace/evals/v3_tests.json $LORA /workspace/evals/v3_final
kill $EPID; sleep 10; kill $(pgrep -f "^python run.py /workspace/evals/engine.yaml") 2>/dev/null
python /workspace/evals/summary_v3.py /workspace/evals/v3_final "MOSS v3 · 1000 pares, legendas delta (Muse 1.3) · 768px · 1000 passos — vs v1" /workspace/outputs_for_user/RESUMO_TESTES_v3.jpg
python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = "AdwolfCzar/moss-image-nextscene-lora"
api.upload_file(repo_id=repo, path_or_fileobj="/workspace/outputs_for_user/RESUMO_TESTES_v3.jpg", path_in_repo="v3_cap1000/RESUMO_TESTES_v3.jpg")
api.upload_folder(repo_id=repo, folder_path="/workspace/evals/v3_final", path_in_repo="v3_cap1000/eval_imgs")
api.upload_folder(repo_id=repo, folder_path="/workspace/test_refs", path_in_repo="test_refs")
for f in ["v3_tests.json", "gen_refs.py", "summary_v3.py", "run_tests.py", "pipeline_v3.sh"]:
    api.upload_file(repo_id=repo, path_or_fileobj=f"/workspace/evals/{f}", path_in_repo=f"scripts/eval/{f}")
print("HF_UPLOAD_OK")
PY
echo PIPELINE_DONE
