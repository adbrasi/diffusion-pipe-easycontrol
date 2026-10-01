import json
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi

repo = 'Qwen/Qwen-Image-2.1'
revision = 'd26bb61231c349cf6b7896fa83353113880e1ba3'
directory = Path('/workspace/models/qwen_image_21')
snapshot_download(repo, revision=revision, local_dir=directory,
                  allow_patterns=['transformer/*', 'text_encoder/*', 'vae/*',
                                  'processor/*', 'scheduler/*', 'model_index.json'],
                  max_workers=2)
info = HfApi().model_info(repo, revision=revision, files_metadata=True)
records = []
for f in info.siblings:
    p = directory / f.rfilename
    if not p.is_file():
        continue
    if p.stat().st_size != f.size:
        raise RuntimeError(f'Wrong size: {p}')
    records.append(dict(path=f.rfilename, bytes=f.size,
                        sha256=f.lfs.sha256 if f.lfs else None))
Path('/workspace/qwen21/model_provenance.json').write_text(json.dumps(
    dict(repo=repo, revision=revision, files=records), indent=2))
print('DOWNLOAD_COMPLETE', flush=True)
