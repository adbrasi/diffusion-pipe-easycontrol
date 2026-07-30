#!/usr/bin/env python3
"""Uploader live da saga KREA2 edit (CLAUDE.md §9).

- No arranque: cria o repo (público) se preciso e sobe README.md, train.toml,
  dataset.toml, NOTES e sampling_inputs (os 3 exemplos exatos usados no
  sampling, imagem + prompt).
- Loop: varre step<N> novos no run dir (estabilidade em 2 leituras), sobe
  adapter + metadata; sobe também os samples gerados pelo supervisor.
- Nunca reenvia (estado em /workspace/state/krea2_saga_uploaded.txt).
- Token: HF_TOKEN do ambiente. Nunca escrito em lugar nenhum.

Uso:
  python tools/krea2_saga_uploader.py --repo <user/nome> \
      --run-base /workspace/checkpoints/krea2_edit_saga \
      --samples-out /workspace/outputs/krea2_edit_saga_samples \
      --extra README_HF.md examples/krea2_edit_saga/train.toml ...
"""
import argparse
import glob
import os
import time

from huggingface_hub import HfApi

STATE = '/workspace/state/krea2_saga_uploaded.txt'


def uploaded():
    return set(open(STATE).read().split()) if os.path.exists(STATE) else set()


def mark(name):
    with open(STATE, 'a') as f:
        f.write(name + '\n')


def size_of(d):
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(d) for f in fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--run-base', required=True)
    ap.add_argument('--samples-out', required=True)
    ap.add_argument('--sampling-inputs', default=None)
    ap.add_argument('--extra', nargs='*', default=[])
    ap.add_argument('--poll', type=int, default=180)
    args = ap.parse_args()

    api = HfApi()
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    api.create_repo(args.repo, repo_type='model', exist_ok=True, private=False)
    print(f'repo pronto: {args.repo}', flush=True)

    for path in args.extra:
        if os.path.exists(path):
            try:
                api.upload_file(path_or_fileobj=path, repo_id=args.repo,
                                path_in_repo=os.path.basename(path),
                                commit_message=f'config: {os.path.basename(path)}')
                print(f'UPLOAD OK {path}', flush=True)
            except Exception as e:
                print(f'upload {path} falhou: {e!r}', flush=True)
    if args.sampling_inputs and os.path.isdir(args.sampling_inputs):
        try:
            api.upload_folder(repo_id=args.repo, folder_path=args.sampling_inputs,
                              path_in_repo='sampling_inputs',
                              commit_message='sampling inputs (exact)')
            print('UPLOAD OK sampling_inputs', flush=True)
        except Exception as e:
            print(f'upload sampling_inputs falhou: {e!r}', flush=True)

    sizes = {}
    while True:
        done = uploaded()
        for d in sorted(glob.glob(f'{args.run_base}/*/step*')):
            name = os.path.basename(d)
            if not os.path.isdir(d) or name in done:
                continue
            if not os.path.exists(os.path.join(d, 'adapter_model.safetensors')):
                continue
            s = size_of(d)
            if sizes.get(name) != s:
                sizes[name] = s
                continue  # espera estabilizar entre polls
            try:
                api.upload_folder(repo_id=args.repo, repo_type='model',
                                  folder_path=d, path_in_repo=f'checkpoints/{name}',
                                  commit_message=f'checkpoint {name}')
                mark(name)
                print(f'UPLOAD OK {name}', flush=True)
            except Exception as e:
                print(f'upload {name} falhou: {e!r}', flush=True)

        for d in sorted(glob.glob(f'{args.samples_out}/step*')):
            name = 'samples_' + os.path.basename(d)
            if not os.path.isdir(d) or name in done:
                continue
            s = size_of(d)
            if sizes.get(name) != s:
                sizes[name] = s
                continue
            try:
                api.upload_folder(repo_id=args.repo, repo_type='model',
                                  folder_path=d,
                                  path_in_repo=f'samples/{os.path.basename(d)}',
                                  commit_message=f'{name}')
                mark(name)
                print(f'UPLOAD OK {name}', flush=True)
            except Exception as e:
                print(f'upload {name} falhou: {e!r}', flush=True)

        time.sleep(args.poll)


if __name__ == '__main__':
    main()
