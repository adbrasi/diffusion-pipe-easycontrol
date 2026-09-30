import os,time,json,traceback
from pathlib import Path
from huggingface_hub import HfApi
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
api=HfApi();repo='AdwolfCzar/anima-nextscene-runs';statep=Path('/workspace/nextscene_ops/sync_state.json')
state=json.loads(statep.read_text()) if statep.exists() else {}
while True:
 try:
  for p in sorted(Path('/workspace/checkpoints/anima_nextscene').glob('*/*/*/adapter_model.safetensors')):
   if time.time()-p.stat().st_mtime<15 or str(p) in state:continue
   from safetensors import safe_open
   with safe_open(p,framework='pt') as f:
    if not f.metadata().get('nextscene_contract'):continue
   dest='checkpoints/'+str(p.parent.relative_to('/workspace/checkpoints/anima_nextscene'))
   api.upload_folder(repo_id=repo,folder_path=p.parent,path_in_repo=dest)
   state[str(p)]=dest;statep.write_text(json.dumps(state,indent=2));print('Uploaded',dest,flush=True)
  api.upload_folder(repo_id=repo,folder_path='/workspace/nextscene_artifacts',path_in_repo='artifacts',ignore_patterns=['*.tmp'])
  api.upload_file(repo_id=repo,path_or_fileobj='/workspace/diffusion-pipe-easycontrol/docs/NEXTSCENE_RUN_LOG.md',path_in_repo='NEXTSCENE_RUN_LOG.md')
 except Exception:traceback.print_exc()
 time.sleep(60)
