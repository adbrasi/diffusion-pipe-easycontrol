#!/bin/bash
source /workspace/venv/bin/activate
cd /workspace/ai-toolkit
echo "[$(date +%H:%M)] START v4"
python run.py /workspace/scripts/moss_nextscene_v4_vlm576.yaml > /workspace/train_v4.log 2>&1; echo "[$(date +%H:%M)] END v4 exit=$?"
python /workspace/scripts/grids.py moss_nextscene_v4_vlm576 | tail -1
python /workspace/scripts/upload_run.py moss_nextscene_v4_vlm576 v4_vlm576 /workspace/scripts/moss_nextscene_v4_vlm576.yaml
LORA=/workspace/output/moss_nextscene_v4_vlm576/moss_nextscene_v4_vlm576.safetensors
rm -f /workspace/evals/engine/engine.json
python run.py /workspace/evals/engine.yaml > /workspace/evals/engine_v4.log 2>&1 &
EPID=$!
until [ -f /workspace/evals/engine/engine.json ]; do sleep 5; kill -0 $EPID || { echo ENGINE_DIED; exit 1; }; done; sleep 5
python /workspace/evals/run_tests.py /workspace/evals/v4_tests.json $LORA /workspace/evals/v4_final
kill $EPID; sleep 10; kill $(pgrep -f "^python run.py /workspace/evals/engine.yaml") 2>/dev/null
CMP_DIR=/workspace/evals/v3_final CMP_LABEL="v3 · 144 tokens visão (1.0)" NEW_LABEL="v4 · 576 tokens visão" python /workspace/evals/summary_v3.py /workspace/evals/v4_final "MOSS v4 (Qwen vê a ref com 576 tokens) vs v3 (144) — mesmo dataset, mesmos prompts, seed 42" /workspace/outputs_for_user/RESUMO_TESTES_v4_vs_v3.jpg
python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = "AdwolfCzar/moss-image-nextscene-lora"
api.upload_file(repo_id=repo, path_or_fileobj="/workspace/outputs_for_user/RESUMO_TESTES_v4_vs_v3.jpg", path_in_repo="v4_vlm576/RESUMO_TESTES_v4_vs_v3.jpg")
api.upload_folder(repo_id=repo, folder_path="/workspace/evals/v4_final", path_in_repo="v4_vlm576/eval_imgs")
for f in ["v4_tests.json", "pipeline_v4.sh", "summary_v3.py"]:
    api.upload_file(repo_id=repo, path_or_fileobj=f"/workspace/evals/{f}", path_in_repo=f"scripts/eval/{f}")
print("HF_UPLOAD_OK")
PY
echo PIPELINE_DONE
