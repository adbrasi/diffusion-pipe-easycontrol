#!/usr/bin/env python3
"""Back up and verify one complete native-training milestone in HF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


def check_visibility(private, public):
    if private == public:
        raise RuntimeError('Repository visibility differs from the explicitly requested upload mode')


def artifact_upload_options(public):
    if public:
        # Publish training evidence; source caption manifests and account audits remain local.
        return dict(allow_patterns=['configs/**', 'eval/**', 'smoke_eval/**',
                    'smoke_report.json', 'RELATORIO_SMOKE.md', 'cache_measurement.json',
                    'restart_request.json', 'sampling_manifest.json'])
    return dict(ignore_patterns=['*.tmp', '*backup*.log', '*backup*.gpu.csv', 'campaign_state.json'])


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
    parser.add_argument('--repo', default='AdwolfCzar/krea2-ab-runs')
    parser.add_argument('--public', action='store_true', help='Explicitly upload to a public repository')
    parser.add_argument('--status-only', action='store_true')
    parser.add_argument('--checkpoint-only', action='store_true',
                        help='Back up a save_quit state without requiring an adapter export')
    args = parser.parse_args()
    repo = args.repo
    with tempfile.TemporaryDirectory(prefix='krea-upload-', dir='/workspace/.tmp') as tmp:
        os.environ['HF_XET_CACHE'] = tmp
        os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
        from huggingface_hub import HfApi
        api = HfApi()
        check_visibility(api.repo_info(repo).private, args.public)
        destinations = []
        folders = [] if args.status_only else [args.run / f'step{args.step}', args.run / f'global_step{args.step}']
        if args.checkpoint_only and not args.status_only:
            folders = [args.run / f'global_step{args.step}']
        for folder in folders:
            if not folder.is_dir():
                raise RuntimeError(f'Missing checkpoint folder: {folder}')
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
                          **artifact_upload_options(args.public))
        state = args.artifacts / 'campaign_state.json'
        if state.exists():
            api.upload_file(repo_id=repo, path_or_fileobj=state.read_bytes(),
                            path_in_repo='artifacts/' + args.artifacts.name + '/campaign_state.json')
        if args.status_only:
            return
        marker = args.artifacts / f'backup_step{args.step}.json'
        marker.write_text(json.dumps(dict(verified=True, step=args.step, repo_id=repo,
                          private=not args.public, destinations=destinations), indent=2))
        api.upload_file(repo_id=repo, path_or_fileobj=marker.read_bytes(),
                        path_in_repo='artifacts/' + args.artifacts.name + '/' + marker.name)
        print(f'HF backup verified ({"public" if args.public else "private"}): {repo}, step{args.step}', flush=True)


if __name__ == '__main__':
    main()
