#!/bin/bash
source /workspace/venv/bin/activate
while pgrep -f "recaption.py pick1000" >/dev/null || pgrep -f "pipeline_v2.sh" >/dev/null; do sleep 15; done
python -I /workspace/recap/../scripts/recaption.py /workspace/recap/pick1000.txt /workspace/recap/cap1000.jsonl 16 | tail -1   # retry any failures
python -I /workspace/scripts/build_cap1000.py
echo "[$(date +%H:%M)] START v3"
cd /workspace/ai-toolkit && python run.py /workspace/scripts/moss_nextscene_v3_cap1000.yaml > /workspace/train_v3.log 2>&1; echo "[$(date +%H:%M)] END v3 exit=$?"
python /workspace/scripts/grids.py moss_nextscene_v3_cap1000
python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = "AdwolfCzar/moss-image-nextscene-lora"
api.upload_folder(repo_id=repo, folder_path="/workspace/data/ns_cap1000", path_in_repo="v3_cap1000/dataset_captions", allow_patterns=["**/*.txt"])
for f in ["/workspace/recap/cap1000.jsonl", "/workspace/recap/val6.jsonl", "/workspace/recap/pick1000.txt", "/workspace/scripts/recaption.py", "/workspace/scripts/build_cap1000.py"]:
    api.upload_file(repo_id=repo, path_or_fileobj=f, path_in_repo=f"v3_cap1000/captioning/{os.path.basename(f)}")
PY
python /workspace/scripts/upload_run.py moss_nextscene_v3_cap1000 v3_cap1000 /workspace/scripts/moss_nextscene_v3_cap1000.yaml
echo V3_ALL_DONE
