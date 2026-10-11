#!/usr/bin/env python3
"""Krea 2 campaign: train in 500-step stages from a separate code worktree, verified HF
backup, stock-ComfyUI Turbo held-out eval on the freed GPU, then resume.

Same state machine as tools/anima1024_campaign.py (stop file, status json, prune only
after remote checksums). Training runs with cwd=--train-root (e.g. the A1000 code
worktree); evaluation runs tools/krea2_v2_eval.py from this repo.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import toml
from huggingface_hub import HfApi

from anima1024_campaign import atomic_json, stage_is_saved, verified_upload

REPO = Path(__file__).resolve().parents[1]


def run(argv, log, env, stop_file, cwd, train_output=None):
    if stop_file.exists():
        return False
    with log.open('a') as f:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
        with log.with_suffix('.gpu.csv').open('a') as metrics:
            metrics.write('utc,memory_MiB,util_percent,power_W\n')
            while proc.poll() is None:
                if stop_file.exists() and train_output is not None:
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
    parser.add_argument('--train-root', type=Path, required=True)
    parser.add_argument('--heldout', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=500)
    parser.add_argument('--eval-limit', type=int, default=3)
    args = parser.parse_args()
    cfg = toml.load(args.config)
    if 'init_from_existing' in cfg['adapter'] or cfg.get('resume_from_checkpoint'):
        raise RuntimeError('First stage must start from a fresh LoRA')
    if cfg['gradient_accumulation_steps'] != 1:
        raise RuntimeError('Expected real batch without accumulation')
    total_steps = cfg.get('max_steps')
    if not isinstance(total_steps, int) or total_steps < 1:
        raise RuntimeError('Expected a positive max_steps')
    art = args.artifacts
    art.mkdir(parents=True, exist_ok=True)
    stop_file = art / 'stop_campaign'
    status_path = art / 'campaign_state.json'
    output = Path(cfg['output_dir'])
    env = os.environ.copy()
    env.update(NCCL_P2P_DISABLE='1', OMP_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               PATH='/venv/main/bin:' + env.get('PATH', ''), HF_HUB_DISABLE_PROGRESS_BARS='1')
    api = HfApi()
    if api.repo_info(args.repo).private:
        raise RuntimeError('Public backups were requested; supplied repository is private')
    state = json.loads(status_path.read_text()) if status_path.exists() else dict(
        status='new', target_step=min(args.interval, total_steps), verified_states=[])
    state['total_steps'] = total_steps
    if state['status'] in ('complete', 'stopped') or stop_file.exists():
        return
    if state['status'] == 'new' and output.exists() and any(output.iterdir()):
        raise RuntimeError('Refusing to initialize a fresh campaign in an existing output')
    configs = art / 'stage_configs'
    configs.mkdir(exist_ok=True)
    while True:
        if shutil.disk_usage(art).free < 8 * 1024**3:
            raise RuntimeError('Less than 8 GiB free; no unverified deletion is allowed')
        target = min(state['target_step'], total_steps)
        config = configs / f'until_{target:06}.toml'
        config.write_text(toml.dumps(dict(cfg, max_steps=target)))
        checkpoints = sorted(output.glob('*/latest'))
        resume = checkpoints[-1].parent.name if checkpoints else None
        state.update(status='training', resumed_run=resume)
        atomic_json(status_path, state)
        argv = ['/venv/main/bin/deepspeed', '--num_gpus=1', '--master_port=29606',
                'train.py', '--deepspeed', '--config', str(config)]
        if resume:
            argv.extend(['--resume_from_checkpoint', resume, '--trust_cache'])
        stage_saved = bool(checkpoints and stage_is_saved(checkpoints[-1].parent, target))
        print(f'Train through step {target}, resume={resume}, stage_saved={stage_saved}', flush=True)
        continuing = (not stop_file.exists() if stage_saved else
                      run(argv, art / f'train_until_{target:06}.log', env, stop_file, args.train_root, output))
        run_dir = sorted(output.glob('*/latest'))[-1].parent
        latest = (run_dir / 'latest').read_text().strip()
        actual_step = int(latest.removeprefix('global_step'))
        adapter_dir = run_dir / f'step{target}'
        if continuing and actual_step < target:
            raise RuntimeError(f'Trainer ended at {actual_step}, before requested stage {target}')
        for adapter in run_dir.glob('step*/adapter_model.safetensors'):
            adapter.chmod(0o644)
        state.update(status='backup', run_dir=str(run_dir), latest=latest)
        atomic_json(status_path, state)
        verified = art / 'verified_backups'
        verified.mkdir(exist_ok=True)
        to_backup = [p for p in sorted(run_dir.glob('global_step*')) if p.is_dir()]
        to_backup += [p for p in sorted(run_dir.glob('step*')) if p.is_dir()]
        to_backup += [p for p in sorted(run_dir.glob('epoch*')) if p.is_dir()]
        for folder in to_backup:
            receipt = verified / f'{folder.name}.json'
            if receipt.exists():
                continue
            manifest = verified_upload(api, args.repo, folder, f'checkpoints/{run_dir.name}/{folder.name}')
            atomic_json(receipt, dict(repo=args.repo, files=manifest,
                                      verified_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        api.upload_file(repo_id=args.repo, path_or_fileobj=str(run_dir / 'latest'),
                        path_in_repo=f'checkpoints/{run_dir.name}/latest')
        for folder in run_dir.glob('global_step*'):
            if folder.name != latest and (verified / f'{folder.name}.json').exists():
                shutil.rmtree(folder)
        if continuing:
            state['status'] = 'sampling'
            atomic_json(status_path, state)
            print(f'Sample {adapter_dir.name}: {args.eval_limit} held-out pairs, Turbo, stock ComfyUI', flush=True)
            evaluation = ['/venv/main/bin/python', 'tools/krea2_v2_eval.py', '--adapter', str(adapter_dir),
                          '--pairs', str(args.heldout), '--out', str(art / 'eval'), '--limit', str(args.eval_limit)]
            if not run(evaluation, art / f'eval_{adapter_dir.name}.log', env, stop_file, REPO):
                continuing = False
        evidence = art / 'public_evidence'
        evidence.mkdir(exist_ok=True)
        for pattern in ('*.log', '*.gpu.csv', '*summary.json'):
            for source in art.glob(pattern):
                shutil.copy2(source, evidence / source.name)
        shutil.copy2(config, evidence / config.name)
        if (art / 'eval').exists():
            shutil.copytree(art / 'eval', evidence / 'eval', dirs_exist_ok=True)
        api.upload_folder(repo_id=args.repo, folder_path=str(evidence), path_in_repo='evidence')
        api.upload_folder(repo_id=args.repo, folder_path=str(verified), path_in_repo='verification')
        if actual_step >= total_steps or not continuing:
            state['status'] = 'complete' if continuing else 'stopped'
            atomic_json(status_path, state)
            api.upload_file(repo_id=args.repo, path_or_fileobj=str(status_path), path_in_repo='campaign_state.json')
            print(f'Campaign {state["status"]}', flush=True)
            break
        state.update(status='ready', target_step=target + args.interval)
        atomic_json(status_path, state)


if __name__ == '__main__':
    main()
