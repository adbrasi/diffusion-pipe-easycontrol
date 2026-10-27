#!/bin/bash
while pgrep -f "^/workspace/dpk/venv/bin/python -u train.py" >/dev/null; do sleep 20; done
OUT=/workspace/checkpoints/krea2_A1000recipe_cap1000
ADAPTER=$(for d in $OUT/*/step*; do echo "${d##*step} $d"; done | sort -n | tail -1 | cut -d" " -f2)/adapter_model.safetensors
echo "ADAPTER=$ADAPTER"; [ -f "$ADAPTER" ] || { echo NO_ADAPTER; exit 1; }
source /workspace/dpk/venv/bin/activate; cd /workspace/dpk/repo
python tools/krea2_v3_eval.py --cases /workspace/evals/krea/cases.json --out /workspace/evals/krea_a1000/out --seeds 42 \
  --comfy /workspace/comfy/ComfyUI --python /workspace/dpk/venv/bin/python --models /workspace/models/krea2 \
  --col "label=a1000,adapter=$ADAPTER,strength=1.0" 2>&1 | grep -vE "^\s*$" | tail -2
echo EVAL_DONE
/workspace/venv/bin/python /workspace/evals/summary_krea.py /workspace/evals/krea/cases.json /workspace/evals/krea_a1000/out rawv3,a1000 "Krea 2 receita A1000 (Turbo) vs raw-v3 vs MOSS v4 — 1000 pares delta, mesmos prompts, seed 42" /workspace/outputs_for_user/RESUMO_KREA2_A1000.jpg
/workspace/venv/bin/python - <<'PY'
import os, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = "AdwolfCzar/moss-image-nextscene-lora"; sub = "krea2_A1000recipe_cap1000"
api.upload_folder(repo_id=repo, folder_path="/workspace/checkpoints/krea2_A1000recipe_cap1000", path_in_repo=f"{sub}/checkpoints", allow_patterns=["**/adapter_model.safetensors", "**/adapter_config.json", "**/*.toml"])
for f in ["/workspace/dpk/configs/krea2_A1000_cap1000.toml", "/workspace/dpk/configs/dataset_cap1000_768_anative.toml", "/workspace/dpk/configs/val_cap1000_768_anative.toml", "/workspace/outputs_for_user/RESUMO_KREA2_A1000.jpg", "/workspace/outputs_for_user/KREA2_A1000_testes.jpg", "/workspace/train_krea_A1000.log", "/workspace/evals/pipeline_a1000.sh"]:
    api.upload_file(repo_id=repo, path_or_fileobj=f, path_in_repo=f"{sub}/{os.path.basename(f)}")
api.upload_folder(repo_id=repo, folder_path="/workspace/evals/krea_a1000/out", path_in_repo=f"{sub}/eval", allow_patterns=["*_a1000.png"])
api.upload_folder(repo_id=repo, folder_path="/workspace/evals/krea_a1000/steps", path_in_repo=f"{sub}/testes_a_cada_250", allow_patterns=["*_s250.png", "*_s500.png", "*_s750.png", "*_s1000.png"])
print("HF_UPLOAD_OK")
PY
echo PIPELINE_DONE
