#!/usr/bin/env python3
"""One complete fresh Anima epoch, with serial sampling and verified backups.

Every 500 steps the trainer finishes a stage with its full optimizer/dataloader
state, sampling runs on the freed GPU, then training resumes from that state.
Only older resume states with a verified public backup may be pruned.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import toml
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def file_hash(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    if algorithm == 'sha1':
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verified_upload(api, repo, folder, prefix):
    files = sorted(p for p in folder.rglob('*') if p.is_file())
    paths = [f'{prefix}/{p.relative_to(folder).as_posix()}' for p in files]
    api.upload_folder(repo_id=repo, folder_path=str(folder), path_in_repo=prefix)
    remote = {}
    for start in range(0, len(paths), 100):
        for item in api.get_paths_info(repo, paths[start:start + 100]):
            remote[item.path] = item
    manifest = []
    for p, name in zip(files, paths):
        info = remote.get(name)
        if info is None or info.size != p.stat().st_size:
            raise RuntimeError(f'Remote file missing or wrong size: {name}')
        digest = file_hash(p)
        if info.lfs:
            same = info.lfs.sha256 == digest
        else:
            same = info.blob_id == file_hash(p, 'sha1')
        if not same:
            raise RuntimeError(f'Remote checksum mismatch: {name}')
        manifest.append(dict(path=name, bytes=p.stat().st_size, sha256=digest))
    return manifest


def run(argv, log, env, stop_file, train_output=None):
    if stop_file.exists():
        return False
    with log.open('a') as f:
        proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        with log.with_suffix('.gpu.csv').open('a') as metrics:
            metrics.write('utc,memory_MiB,util_percent,power_W\n')
            while proc.poll() is None:
                if stop_file.exists() and train_output is not None:
                    # Saver watches the timestamped run directory, not its parent.
                    for run_dir in train_output.glob('*'):
                        if run_dir.is_dir():
                            (run_dir / 'save_quit').touch()
                reading = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,power.draw',
                     '--format=csv,noheader,nounits'], capture_output=True, text=True)
                metrics.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ',' + reading.stdout.strip() + '\n')
                metrics.flush()
                time.sleep(2)
        if proc.returncode:
            raise RuntimeError(f'Job failed ({proc.returncode}); see {log}')
    return not stop_file.exists()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--interval', type=int, default=500)
    args = parser.parse_args()
    cfg = toml.load(args.config)
    if cfg['model']['dtype'] != 'bfloat16' or cfg['adapter']['dtype'] != 'bfloat16':
        raise RuntimeError('This campaign must retain the E2 BF16 policy')
    if 'init_from_existing' in cfg['adapter'] or cfg.get('resume_from_checkpoint'):
        raise RuntimeError('First stage must start from a fresh LoRA')
    if cfg['nextscene']['rope_layout'] != 'aligned' or cfg['gradient_accumulation_steps'] != 1:
        raise RuntimeError('Expected aligned geometry and real batch without accumulation')
    if cfg['epochs'] != 1 or args.interval < 1:
        raise RuntimeError('Expected one complete initial epoch and a positive sample interval')
    art = args.artifacts
    art.mkdir(parents=True, exist_ok=True)
    stop_file = art / 'stop_campaign'
    status_path = art / 'campaign_state.json'
    output = Path(cfg['output_dir'])
    env = os.environ.copy()
    env.update(NCCL_P2P_DISABLE='1', OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false',
               PATH='/venv/main/bin:' + env.get('PATH', ''), HF_HUB_DISABLE_PROGRESS_BARS='1')
    api = HfApi()
    if api.repo_info(args.repo).private:
        raise RuntimeError('Public backups were requested; supplied repository is private')
    state = json.loads(status_path.read_text()) if status_path.exists() else dict(status='new', target_step=args.interval, verified_states=[])
    if state['status'] in ('complete', 'stopped') or stop_file.exists():
        return
    if state['status'] == 'new' and output.exists() and any(output.iterdir()):
        raise RuntimeError('Refusing to initialize a fresh campaign in an existing output')
    configs = art / 'stage_configs'
    configs.mkdir(exist_ok=True)
    model = cfg['model']
    while True:
        if shutil.disk_usage(art).free < 6 * 1024**3:
            raise RuntimeError('Less than 6 GiB free; no dataset truncation or unverified deletion is allowed')
        target = state['target_step']
        stage = dict(cfg, max_steps=target)
        config = configs / f'until_{target:06}.toml'
        config.write_text(toml.dumps(stage))
        checkpoints = sorted(output.glob('*/latest'))
        resume = checkpoints[-1].parent.name if checkpoints else None
        state.update(status='training', resumed_run=resume)
        atomic_json(status_path, state)
        argv = ['/venv/main/bin/deepspeed', '--num_gpus=1', '--master_port=29605',
                'train.py', '--deepspeed', '--config', str(config)]
        if resume:
            argv.extend(['--resume_from_checkpoint', resume, '--trust_cache'])
        stage_saved = False
        if checkpoints:
            existing_run = checkpoints[-1].parent
            existing_tag = (existing_run / 'latest').read_text().strip()
            stage_saved = ((existing_run / 'epoch1/adapter_model.safetensors').exists()
                           or (int(existing_tag.removeprefix('global_step')) >= target
                               and (existing_run / f'step{target}/adapter_model.safetensors').exists()))
        print(f'Train through step {target}, resume={resume}, stage_saved={stage_saved}', flush=True)
        continuing = (not stop_file.exists() if stage_saved else
                      run(argv, art / f'train_until_{target:06}.log', env, stop_file, output))
        run_dir = sorted(output.glob('*/latest'))[-1].parent
        latest = (run_dir / 'latest').read_text().strip()
        completed_epoch = (run_dir / 'epoch1/adapter_model.safetensors').exists()
        adapter_dir = run_dir / ('epoch1' if completed_epoch else f'step{target}')
        state.update(status='backup', run_dir=str(run_dir), latest=latest)
        atomic_json(status_path, state)
        # Never upload or remove a still-changing trainer state.
        verified = art / 'verified_backups'
        verified.mkdir(exist_ok=True)
        to_backup = [p for p in sorted(run_dir.glob('global_step*')) if p.is_dir()]
        to_backup += [p for p in sorted(run_dir.glob('step*')) if p.is_dir()]
        if completed_epoch:
            to_backup.append(run_dir / 'epoch1')
        for folder in to_backup:
            receipt = verified / f'{folder.name}.json'
            if receipt.exists():
                continue
            manifest = verified_upload(api, args.repo, folder, f'checkpoints/{run_dir.name}/{folder.name}')
            atomic_json(receipt, dict(repo=args.repo, files=manifest, verified_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        api.upload_file(repo_id=args.repo, path_or_fileobj=str(run_dir / 'latest'), path_in_repo=f'checkpoints/{run_dir.name}/latest')
        # Keep every adapter and the latest complete resume state locally.
        for folder in run_dir.glob('global_step*'):
            if folder.name != latest and (verified / f'{folder.name}.json').exists():
                shutil.rmtree(folder)
        if continuing:
            state['status'] = 'sampling'
            atomic_json(status_path, state)
            print(f'Sample {adapter_dir.name}: 8 held-out pairs, true/shuffled/null at 1024', flush=True)
            evaluation = ['/venv/main/bin/python', 'tools/nextscene_eval.py',
                          '--dit', model['transformer_path'], '--vae', model['vae_path'], '--llm', model['llm_path'],
                          '--pairs', '/workspace/heldout_short', '--ckpt', str(adapter_dir),
                          '--out', str(art / 'eval'), '--limit', '8', '--width', '1024', '--height', '1024',
                          '--match-target-ar', '--steps', '30', '--cfg', '4', '--flow_shift', '3', '--ref_cfg', '1', '--seed', '76']
            if not run(evaluation, art / f'eval_{adapter_dir.name}.log', env, stop_file):
                continuing = False
        # Evidence upload excludes the source captions and private chat transcript.
        evidence = art / 'public_evidence'
        evidence.mkdir(exist_ok=True)
        for pattern in ('*.log', '*.gpu.csv', '*summary.json', '*report.json', '*provenance.json'):
            for source in art.glob(pattern):
                shutil.copy2(source, evidence / source.name)
        shutil.copy2(config, evidence / config.name)
        if (art / 'eval').exists():
            shutil.copytree(art / 'eval', evidence / 'eval', dirs_exist_ok=True)
        api.upload_folder(repo_id=args.repo, folder_path=str(evidence), path_in_repo='evidence')
        api.upload_folder(repo_id=args.repo, folder_path=str(verified), path_in_repo='verification')
        if completed_epoch or not continuing:
            state['status'] = 'complete' if continuing else 'stopped'
            atomic_json(status_path, state)
            api.upload_file(repo_id=args.repo, path_or_fileobj=str(status_path), path_in_repo='campaign_state.json')
            print(f'Campaign {state["status"]}', flush=True)
            break
        state.update(status='ready', target_step=target + args.interval)
        atomic_json(status_path, state)


if __name__ == '__main__':
    main()
