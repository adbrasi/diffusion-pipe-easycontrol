"""Continuously back up completed adapters, evaluation artifacts and the run log."""
import json
import os
from pathlib import Path
import time
import traceback

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
from huggingface_hub import HfApi

ROOT = Path('/workspace/k2ab')
REPO = 'AdwolfCzar/krea2-ab-runs'
api = HfApi()
state_file = ROOT / 'ops/sync_state.json'
state = json.loads(state_file.read_text()) if state_file.exists() else {}
while True:
    try:
        for path in sorted((ROOT / 'checkpoints').glob('*/*/*/adapter_model.safetensors')):
            if str(path) in state or time.time() - path.stat().st_mtime < 20:
                continue
            dest = 'checkpoints/' + str(path.parent.relative_to(ROOT / 'checkpoints'))
            api.upload_folder(repo_id=REPO, folder_path=path.parent, path_in_repo=dest)
            state[str(path)] = dest
            state_file.write_text(json.dumps(state, indent=2))
            print('Uploaded', dest, flush=True)
        api.upload_folder(repo_id=REPO, folder_path=ROOT / 'artifacts', path_in_repo='artifacts',
                          ignore_patterns=['*.pt', '*.tmp'])
        api.upload_file(repo_id=REPO,
                        path_or_fileobj='/workspace/diffusion-pipe-easycontrol/docs/KREA2_AB_RUN_LOG.md',
                        path_in_repo='KREA2_AB_RUN_LOG.md')
    except Exception:
        traceback.print_exc()
    time.sleep(60)
