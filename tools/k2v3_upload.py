#!/usr/bin/env python3
"""Upload files/folders to a PUBLIC HF repo and verify every file (size + sha256 vs the remote LFS/oid metadata).

    python tools/k2v3_upload.py --repo AdwolfCzar/krea2-nextscene-raw-v3 LOCAL[:REMOTE_PREFIX] ...
"""
import argparse
import hashlib
from pathlib import Path
import sys

from huggingface_hub import HfApi


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='AdwolfCzar/krea2-nextscene-raw-v3')
    ap.add_argument('--message', default='upload')
    ap.add_argument('items', nargs='+')
    args = ap.parse_args()
    api = HfApi()
    api.create_repo(args.repo, private=False, exist_ok=True)
    expected = {}
    for item in args.items:
        local, _, prefix = item.partition(':')
        local = Path(local)
        if local.is_dir():
            remote = prefix or local.name
            api.upload_folder(repo_id=args.repo, folder_path=str(local), path_in_repo=remote, commit_message=args.message)
            for f in local.rglob('*'):
                if f.is_file():
                    expected[f'{remote}/{f.relative_to(local)}'] = f
        else:
            remote = prefix or local.name
            api.upload_file(repo_id=args.repo, path_or_fileobj=str(local), path_in_repo=remote, commit_message=args.message)
            expected[remote] = local
    info = {s.rfilename: s for s in api.model_info(args.repo, files_metadata=True).siblings}
    bad = []
    for remote, local in expected.items():
        s = info.get(remote)
        if s is None or s.size != local.stat().st_size:
            bad.append((remote, 'missing or size mismatch'))
            continue
        if s.lfs is not None:
            if s.lfs.sha256 != sha256(local):
                bad.append((remote, 'sha256 mismatch'))
        elif s.blob_id and s.blob_id != git_blob_sha1(local):
            bad.append((remote, 'blob sha1 mismatch'))
    print(f'verified {len(expected) - len(bad)}/{len(expected)} files in https://huggingface.co/{args.repo}')
    if bad:
        print('FAILED:', bad)
        sys.exit(1)


if __name__ == '__main__':
    main()
