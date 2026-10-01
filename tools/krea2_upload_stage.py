#!/usr/bin/env python3
"""Back up and verify one complete native-training milestone in private HF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--status-only', action='store_true')
    args = parser.parse_args()
    repo = 'AdwolfCzar/krea2-ab-runs'
    with tempfile.TemporaryDirectory(prefix='krea-upload-', dir='/workspace/.tmp') as tmp:
        os.environ['HF_XET_CACHE'] = tmp
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        from huggingface_hub import HfApi
        api = HfApi()
        if not api.repo_info(repo).private:
            raise RuntimeError('Refusing to upload to a public model repository')
        destinations = []
        folders = [] if args.status_only else [args.run / f'step{args.step}', args.run / f'global_step{args.step}']
        for folder in folders:
            destination = 'checkpoints/' + str(folder.relative_to('/workspace/k2ab/checkpoints'))
            api.upload_folder(repo_id=repo, folder_path=folder, path_in_repo=destination)
            files = list(folder.rglob('*'))
            remote_paths = [destination + '/' + str(p.relative_to(folder)) for p in files if p.is_file()]
            infos = {p.path: p for p in api.get_paths_info(repo, paths=remote_paths)}
            for local in files:
                if not local.is_file():
                    continue
                remote = destination + '/' + str(local.relative_to(folder))
                info = infos[remote]
                if info.size != local.stat().st_size:
                    raise RuntimeError(f'Backup size mismatch: {local}')
                if info.lfs and info.lfs.sha256 != digest(local):
                    raise RuntimeError(f'Backup SHA256 mismatch: {local}')
            destinations.append(destination)
        base = 'checkpoints/' + str(args.run.relative_to('/workspace/k2ab/checkpoints'))
        api.upload_file(repo_id=repo, path_or_fileobj=args.run / 'latest', path_in_repo=base + '/latest')
        api.upload_folder(repo_id=repo, folder_path=args.artifacts,
                          path_in_repo='artifacts/' + args.artifacts.name,
                          ignore_patterns=['*.tmp', 'fullbudget_backup*.log', 'fullbudget_backup*.gpu.csv',
                                           'campaign_state.json'])
        state = args.artifacts / 'campaign_state.json'
        if state.exists():
            api.upload_file(repo_id=repo, path_or_fileobj=state.read_bytes(),
                            path_in_repo='artifacts/' + args.artifacts.name + '/campaign_state.json')
        if args.status_only:
            return
        marker = args.artifacts / f'backup_step{args.step}.json'
        marker.write_text(json.dumps(dict(verified=True, step=args.step, destinations=destinations), indent=2))
        api.upload_file(repo_id=repo, path_or_fileobj=marker.read_bytes(),
                        path_in_repo='artifacts/' + args.artifacts.name + '/' + marker.name)
        print(f'Private HF backup verified: step{args.step}', flush=True)


if __name__ == '__main__':
    main()
