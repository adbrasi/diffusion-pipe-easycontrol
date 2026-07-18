#!/usr/bin/env python3
"""Uploader incremental do treino raiz_iclora."""
import glob
import os
import time

from huggingface_hub import HfApi

REPO = 'AdwolfCzar/groundedsecret'
ROOT = '/workspace/checkpoints/anima_raiz_iclora'
STATE = '/workspace/state/raiz_uploaded.txt'
api = HfApi()
os.makedirs(os.path.dirname(STATE), exist_ok=True)


def uploaded():
    return set(open(STATE).read().split()) if os.path.exists(STATE) else set()


def size_of(d):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)


sizes = {}
while True:
    done = uploaded()
    for d in sorted(glob.glob(f'{ROOT}/*/step*')):
        name = f'raiz_{os.path.basename(d)}'
        if name in done or not os.path.exists(os.path.join(d, 'adapter_model.safetensors')):
            continue
        s = size_of(d)
        if sizes.get(name) != s:
            sizes[name] = s
            continue
        try:
            api.upload_folder(repo_id=REPO, repo_type='model', folder_path=d,
                              path_in_repo=f'anima_raiz/{name}',
                              commit_message=f'raiz_iclora {name}')
            with open(STATE, 'a') as f:
                f.write(name + '\n')
            print(f'UPLOAD OK {name}', flush=True)
        except Exception as e:
            print(f'upload {name} falhou: {e!r}', flush=True)
    time.sleep(180)
