import sys, json, shutil, time
from pathlib import Path
sys.path.insert(0, '/workspace/diffusion-pipe-easycontrol/tools')
from anima1024_campaign import verified_upload, atomic_json
from huggingface_hub import HfApi
import torch
from safetensors import safe_open
from safetensors.torch import save_file

root = Path('/workspace/checkpoints/anima_nextscene')
run = root / 'A_aligned_1024_scratch_20261001/20261001_09-58-41'
export = run / 'step1830'
export.mkdir(exist_ok=True)
with safe_open(str(run/'step1500/adapter_model.safetensors'), framework='pt') as f:
    expected = {k: f.get_slice(k).get_shape() for k in f.keys()}
    metadata = f.metadata()
weights = {}
for index in range(28):
    state = torch.load(run/f'global_step1830/layer_{index+2:02}-model_states.pt',
                       map_location='cpu', weights_only=False)
    for k, v in state.items():
        key = 'diffusion_model.blocks.'+str(index)+'.'+k.removeprefix('block.').replace('.default.', '.')
        if key not in expected or list(v.shape) != expected[key]:
            raise RuntimeError(f'Unexpected export: {key}')
        weights[key] = v.contiguous()
if set(weights) != set(expected):
    raise RuntimeError('Incomplete final adapter export')
metadata['source_step'] = '1830'
save_file(weights, str(export/'adapter_model.safetensors'), metadata=metadata)
for name in ('adapter_config.json', 'nextscene_contract.json', 'adapter_key_audit.json'):
    source = run/'step1500'/name
    if source.exists(): shutil.copy2(source, export/name)
preserved = Path('/workspace/qwen21/anima_preserved')
preserved.mkdir(exist_ok=True)
shutil.copytree(export, preserved/'last_step1830', dirs_exist_ok=True)
winner = root/'E2_A/20260930_08-15-31/epoch1'
shutil.copytree(winner, preserved/'E2_A_winner', dirs_exist_ok=True)
api = HfApi()
repo = 'AdwolfCzar/anima-nextscene-archive'
api.create_repo(repo, private=False, exist_ok=True)
if api.repo_info(repo).private: raise RuntimeError('Archive must be public')
receipts = Path('/workspace/qwen21/anima_archive_receipts')
receipts.mkdir(exist_ok=True)
for child in sorted(root.iterdir()):
    if not child.is_dir(): continue
    receipt = receipts / (child.name + '.json')
    files = verified_upload(api, repo, child, 'checkpoints/'+child.name)
    atomic_json(receipt, dict(repo=repo, files=files, verified_utc=time.time()))
    print('VERIFIED', child.name, len(files), flush=True)
artifacts = Path('/workspace/nextscene_artifacts')
files = verified_upload(api, repo, artifacts, 'artifacts')
atomic_json(receipts/'artifacts.json', dict(repo=repo, files=files, verified_utc=time.time()))
card = '# Anima NextScene — archived experiments\n\nAnima experiments and samples, stopped at step 1830. Checkpoints were backed up with verified checksums before local cleanup.\n'
api.upload_file(repo_id=repo, path_or_fileobj=card.encode(), path_in_repo='README.md')
atomic_json(receipts/'complete.json', dict(repo=repo, complete=True))
print('ARCHIVE_COMPLETE', flush=True)
