#!/usr/bin/env python3
"""Uploader incremental dos treinos full5k: varre step* novos a cada 3 min,
confirma estabilidade (2 checagens) e sobe para o HF. Nunca reenvia."""
import glob
import json
import os
import time

from huggingface_hub import HfApi

REPO = 'AdwolfCzar/groundedsecret'
ROOTS = {
    'v3_full5k': '/workspace/checkpoints/anima_v3_full5k',
    'omini_broad_full5k': '/workspace/checkpoints/anima_omini_broad_full5k',
}
STATE = '/workspace/state/full5k_uploaded.txt'
api = HfApi()
os.makedirs(os.path.dirname(STATE), exist_ok=True)


def uploaded():
    return set(open(STATE).read().split()) if os.path.exists(STATE) else set()


def size_of(d):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)


sizes = {}
while True:
    done = uploaded()
    for arm, root in ROOTS.items():
        for d in sorted(glob.glob(f'{root}/*/step*')):
            name = f'{arm}_{os.path.basename(d)}'
            if name in done or not os.path.exists(os.path.join(d, 'adapter_model.safetensors')):
                continue
            s = size_of(d)
            if sizes.get(name) != s:
                sizes[name] = s
                continue  # espera estabilizar
            try:
                api.upload_folder(repo_id=REPO, repo_type='model', folder_path=d,
                                  path_in_repo=f'anima_full5k/{name}',
                                  commit_message=f'full5k {name}')
                with open(STATE, 'a') as f:
                    f.write(name + '\n')
                print(f'UPLOAD OK {name}', flush=True)
            except Exception as e:
                print(f'upload {name} falhou: {e!r}', flush=True)
    time.sleep(180)
