#!/bin/bash
while pgrep -f "train.py --deepspeed --config /workspace/dpk/configs/krea2_nextscene_cap1000.toml" >/dev/null; do sleep 20; done
OUT=/workspace/checkpoints/krea2_nextscene_cap1000
ADAPTER=$(for d in $OUT/*/step*; do echo "${d##*step} $d"; done | sort -n | tail -1 | cut -d" " -f2)/adapter_model.safetensors
[ -f "$ADAPTER" ] || ADAPTER=$(find $OUT -name adapter_model.safetensors | sort | tail -1)
echo "ADAPTER=$ADAPTER"; [ -f "$ADAPTER" ] || { echo NO_ADAPTER; exit 1; }
source /workspace/dpk/venv/bin/activate; cd /workspace/dpk/repo
python tools/krea2_v3_eval.py --cases /workspace/evals/krea/cases.json --out /workspace/evals/krea/out --seeds 42 \
  --comfy /workspace/comfy/ComfyUI --python /workspace/dpk/venv/bin/python --models /workspace/models/krea2 \
  --col "label=lora_turbo,adapter=$ADAPTER,strength=1.0" 2>&1 | grep -vE "^\s*$" | tail -3
echo EVAL_DONE
python /workspace/evals/summary_krea.py /workspace/evals/krea/cases.json /workspace/evals/krea/out lora_turbo "Krea 2 passo 1000 (seu método raw-v3, Turbo 8 passos) vs MOSS v4 — mesmo dataset (1000 pares delta), mesmos prompts, seed 42" /workspace/outputs_for_user/RESUMO_KREA2_vs_MOSS.jpg
/workspace/venv/bin/python - <<'PY'
import os, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = "AdwolfCzar/moss-image-nextscene-lora"; sub = "krea2_rawv3_cap1000"
out = "/workspace/checkpoints/krea2_nextscene_cap1000"
api.upload_folder(repo_id=repo, folder_path=out, path_in_repo=f"{sub}/checkpoints", allow_patterns=["**/adapter_model.safetensors", "**/adapter_config.json", "**/*.toml"])
for f in glob.glob("/workspace/dpk/configs/*.toml"): api.upload_file(repo_id=repo, path_or_fileobj=f, path_in_repo=f"{sub}/configs/{os.path.basename(f)}")
api.upload_file(repo_id=repo, path_or_fileobj="/workspace/outputs_for_user/RESUMO_KREA2_vs_MOSS.jpg", path_in_repo=f"{sub}/RESUMO_KREA2_vs_MOSS.jpg")
api.upload_folder(repo_id=repo, folder_path="/workspace/evals/krea/out", path_in_repo=f"{sub}/eval")
for f in ["/workspace/evals/krea/cases.json", "/workspace/evals/summary_krea.py", "/workspace/evals/pipeline_krea.sh", "/workspace/train_krea.log"]:
    api.upload_file(repo_id=repo, path_or_fileobj=f, path_in_repo=f"{sub}/{os.path.basename(f)}")
print("HF_UPLOAD_OK")
PY
echo PIPELINE_DONE
