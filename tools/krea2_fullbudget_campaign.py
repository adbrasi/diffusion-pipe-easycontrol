#!/usr/bin/env python3
"""Run the approved scratch A-native campaign serially under supervisor."""
import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import time

import requests
import toml

ROOT = Path('/workspace/k2ab')
REPO = Path('/workspace/diffusion-pipe-easycontrol')
ART = ROOT / 'artifacts/fullbudget_20261001'
OUTPUT = ROOT / 'checkpoints/A_native_fullbudget_fromscratch_5000'
QUEUE = ROOT / 'ops/jobs'
STATE = ART / 'campaign_state.json'
PYTHON = '/venv/main/bin/python'


def atomic(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def validate_recipe(recipe):
    if recipe['output_dir'] != str(OUTPUT):
        raise RuntimeError('Wrong training output: baseline/smoke states must not be reused')
    if recipe['model']['type'] != 'krea2_native' or recipe['model']['base_quant'] != 'fp8_scaled':
        raise RuntimeError('Wrong model contract or base precision')
    if recipe['micro_batch_size_per_gpu'] != 2 or recipe['gradient_accumulation_steps'] != 1:
        raise RuntimeError('Batch contract mismatch')
    if not math.isclose(recipe['optimizer']['lr'], .0004):
        raise RuntimeError('LR differs from the approved value')
    if recipe.get('resume_from_checkpoint') or recipe['adapter'].get('init_from_existing'):
        raise RuntimeError('Fresh training recipe must not initialize from an existing run')


class Stopped(Exception):
    pass


def job(name, argv, state, train=False, env=None):
    path = QUEUE / (name + '.job.json')
    if not path.exists():
        value = dict(argv=argv, cwd=str(ROOT / 'native_worktree') if train else str(REPO),
                     log=str(ART / (name + '.log')), env=env or {})
        if train:
            value['train_output'] = str(OUTPUT)
        atomic(path, value)
    else:
        existing = json.loads(path.read_text())
        if existing['argv'] != argv:
            raise RuntimeError('Queued command differs from current recipe')
    status_path = path.with_suffix('.state.json')
    while True:
        result = json.loads(status_path.read_text()) if status_path.exists() else {}
        state.update(current_job=name, job_status=result)
        atomic(STATE, state)
        if (ART / 'stop_training').exists():
            if train and result.get('status') == 'running':
                runs = sorted(p for p in OUTPUT.glob('*') if p.is_dir())
                if runs:
                    (runs[-1] / 'save_quit').touch()
            if result.get('status') != 'running':
                raise Stopped('Requested scheduling stop')
        if result.get('status') == 'done':
            return Path(json.loads(path.read_text())['log'])
        if result.get('status') == 'failed':
            raise RuntimeError(f'{name} failed; inspect {ART / (name + ".log")}')
        time.sleep(5)


def cache_count(directory):
    total = 0
    for db in directory.glob('ar_frames_*/text_embeddings_1/metadata.db'):
        with sqlite3.connect(f'file:{db}?mode=ro', uri=True) as con:
            total += con.execute('SELECT count(*) FROM items').fetchone()[0]
    return total


def prune(run):
    if run.resolve().parent != OUTPUT.resolve():
        raise RuntimeError('Refusing to prune outside the new training run')
    latest = (run / 'latest').read_text().strip()
    for prefix in ('global_step', 'step'):
        folders = sorted(run.glob(prefix + '[0-9]*'), key=lambda p: int(p.name[len(prefix):]))
        keep = {p.name for p in folders[-2:]} | {latest}
        for folder in folders:
            number = int(folder.name[len(prefix):])
            marker = ART / f'backup_step{number}.json'
            if folder.name in keep or not marker.exists():
                continue
            if not json.loads(marker.read_text()).get('verified'):
                continue
            shutil.rmtree(folder)
            print('Pruned verified older new-run checkpoint', folder, flush=True)
    for link in (ROOT / '../models/krea2/loras').glob('A_native_fullbudget_fromscratch_5000_step*.safetensors'):
        if link.is_symlink() and not link.exists():
            link.unlink()


def publish_milestone(step, evaluation, state):
    from PIL import Image
    destination = REPO / 'docs/krea2_results/2026-10-01/native_fullbudget'
    destination.mkdir(exist_ok=True)
    metrics = evaluation / 'Turbo/metrics.json'
    grid = evaluation / 'Turbo/grid.png'
    paths = [REPO / 'docs/KREA2_AB_RUN_LOG.md']
    for source in (metrics,):
        if source.exists():
            target = destination / f'step{step}_{source.name}'
            shutil.copy2(source, target)
            paths.append(target)
    if grid.exists():
        target = destination / f'step{step}.jpg'
        with Image.open(grid) as image:
            image.convert('RGB').save(target, quality=92)
        paths.append(target)
    target = destination / 'campaign_state.json'
    atomic(target, state)
    paths.append(target)
    names = [str(p.relative_to(REPO)) for p in paths]
    env = os.environ.copy()
    env.update(GIT_ASKPASS='/workspace/nextscene_ops/git-askpass.sh', GIT_TERMINAL_PROMPT='0')
    try:
        subprocess.run(['git', 'add', '--', *names], cwd=REPO, env=env, check=True, timeout=30)
        subprocess.run(['git', 'commit', '--only', '-m', f'Record scratch A-native step{step} and evaluation',
                        '--', *names], cwd=REPO, env=env, check=True, timeout=30)
        subprocess.run(['git', 'push', 'origin', 'HEAD:claude/elegant-ptolemy-8kywqy'],
                       cwd=REPO, env=env, check=True, timeout=45)
    except subprocess.SubprocessError as error:
        state['git_publish_pending'] = str(error)
        print('Git publication pending; private HF backup already verified:', error, flush=True)


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else dict(milestones=[])
    if state.get('status') in ('failed', 'stopped', 'complete'):
        print('Campaign terminal state:', state['status'], flush=True)
        return
    if not json.loads((ART / 'smoke_report.json').read_text()).get('passed'):
        raise RuntimeError('Smoke gate has not passed')
    recipe = toml.load(ART / 'configs/train_from_scratch_5000.toml')
    validate_recipe(recipe)
    state.update(status='caching', target_steps=5000, lr=.0004, initialized_from='random LoRA')
    atomic(STATE, state)
    args = ['/venv/main/bin/deepspeed', '--num_gpus=1', '--master_port=29601', 'train.py',
            '--deepspeed', '--config', str(ART / 'configs/train_from_scratch_5000.toml'), '--cache_only']
    # Cache-only still imports the native model; use the validated native checkout.
    cache_path = QUEUE / 'fullbudget_010_cache.job.json'
    if not cache_path.exists():
        atomic(cache_path, dict(argv=args, cwd=str(ROOT / 'native_worktree'),
                               log=str(ART / 'fullbudget_010_cache.log'),
                               env={'KREA2_CACHE_MIN_FREE_BYTES': str(8 * 1024 ** 3)}))
    job('fullbudget_010_cache', args, state)
    plan = json.loads((ART / 'data_plan_final.json').read_text())
    cache_root = Path(plan['destination']) / 'target/cache/krea2_native'
    if cache_count(cache_root) != plan['selected_pairs']:
        raise RuntimeError('Incomplete text cache; refusing to train a partial dataset')
    if shutil.disk_usage(ROOT).free < 8 * 1024 ** 3:
        raise RuntimeError('Cache consumed the checkpoint reserve')
    state.update(status='training', cache_pairs=plan['selected_pairs'])
    atomic(STATE, state)
    for step in range(250, 5001, 250):
        config_path = ART / 'configs' / f'A_native_through{step}.toml'
        config = toml.load(config_path)
        validate_recipe(config)
        args = ['/venv/main/bin/deepspeed', '--num_gpus=1', '--master_port=29601', 'train.py',
                '--deepspeed', '--config', str(config_path)]
        if step > 250:
            runs = sorted(OUTPUT.glob('*/latest'))
            if len(runs) != 1:
                raise RuntimeError('Expected exactly one NEW run to resume')
            args += ['--resume_from_checkpoint', runs[0].parent.name]
        elif not (QUEUE / 'fullbudget_train000250.job.json').exists() and list(OUTPUT.glob('*/latest')):
            raise RuntimeError('First segment already has a checkpoint; refusing accidental restart')
        log = job(f'fullbudget_train{step:06d}', args, state, train=True)
        logs = log.read_text(errors='replace')
        if f'steps: {step} loss:' not in logs or '[krea2_native] adapter audit OK: 512 keys' not in logs:
            raise RuntimeError('Training endpoint or adapter audit missing')
        losses = [float(x) for x in re.findall(r'loss: (\S+)', logs)]
        if not losses or not all(math.isfinite(x) for x in losses):
            raise RuntimeError('Nonfinite loss')
        lrs = re.findall(r'lr=\[([^\]]+)\]', logs)
        if not lrs or not math.isclose(float(lrs[-1]), .0004):
            raise RuntimeError('Unexpected LR after warmup/resume')
        adapters = list(OUTPUT.glob(f'*/step{step}/adapter_model.safetensors'))
        if len(adapters) != 1:
            raise RuntimeError('Missing or ambiguous new adapter')
        adapter = adapters[0]
        run = adapter.parent.parent
        state.update(status='evaluating', step=step, samples_seen=step * 2)
        atomic(STATE, state)
        subprocess.run(['supervisorctl', 'start', 'k2ab_stock'], check=True)
        try:
            for _ in range(60):
                try:
                    if requests.get('http://127.0.0.1:18819/object_info', timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError('Stock ComfyUI did not become ready')
            evaluation = ART / 'eval' / f'step{step}'
            args = [PYTHON, 'tools/k2ab_eval_stock.py', '--adapter', str(adapter), '--out', str(evaluation),
                    '--manifest', str(ART / 'sampling_manifest.json'), '--base-model',
                    'krea2_raw_fp8_scaled.safetensors', '--reference-pixels', 'target',
                    '--limit', '4', '--variant', 'Turbo', '--adapter-only']
            job(f'fullbudget_eval{step:06d}', args, state)
        finally:
            subprocess.run(['supervisorctl', 'stop', 'k2ab_stock'], check=True)
        job(f'fullbudget_metrics{step:06d}', [PYTHON, 'tools/k2ab_metrics.py', '--pairs',
            str(ROOT / 'heldout'), '--dir', str(evaluation / 'Turbo'), '--adapter-only'], state)
        state['status'] = 'backing_up'
        atomic(STATE, state)
        job(f'fullbudget_backup{step:06d}', [PYTHON, 'tools/krea2_upload_stage.py',
            '--artifacts', str(ART), '--run', str(run), '--step', str(step)], state)
        prune(run)
        link = Path('/workspace/comfy/ComfyUI/models/loras/A_native_fullbudget_latest.safetensors')
        if link.exists() and not link.is_symlink():
            raise RuntimeError('Latest adapter link would overwrite an existing regular file')
        if link.is_symlink():
            link.unlink()
        link.symlink_to(adapter)
        milestone = dict(step=step, samples_seen=step * 2, adapter=str(adapter),
                         utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), last_loss=losses[-1])
        if step not in [m['step'] for m in state['milestones']]:
            state['milestones'].append(milestone)
            with (REPO / 'docs/KREA2_AB_RUN_LOG.md').open('a') as stream:
                stream.write(f'\n\n## {milestone["utc"]} — A native novo step{step}\n\n'
                             f'Do zero, dataset completo por orçamento de disco, LR0,0004 confirmado; '
                             f'{step * 2}amostras vistas, lossfinal{losses[-1]}. '
                             '4Turbo512+grid/métricas e adapter/estado completos enviados ao HF privado; '
                             'backup verificado antes da poda. Não é retomada do A1000 antigo.\n')
        state['status'] = 'training'
        atomic(STATE, state)
        publish_milestone(step, evaluation, state)
        print('Milestone complete', step, flush=True)
    state.update(status='complete', current_job=None)
    atomic(STATE, state)
    subprocess.run([PYTHON, 'tools/krea2_upload_stage.py', '--artifacts', str(ART), '--run', str(run),
                    '--step', '5000', '--status-only'], cwd=REPO, check=True)
    print('A native fresh 5000 steps complete', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        state = json.loads(STATE.read_text()) if STATE.exists() else {}
        state.update(status='stopped' if isinstance(error, Stopped) else 'failed', error=str(error))
        atomic(STATE, state)
        print('Campaign stopped:', error, flush=True)
        # Normal exit prevents replaying failed GPU jobs on supervisor restart.
