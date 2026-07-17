#!/usr/bin/env python3
"""Supervisor do treino GROUNDEDSECRET (krea2_omini_grounded, 7000 steps).

Responsabilidades (CLAUDE.md §9):
- inicia o treinamento (deepspeed) e registra stdout/stderr em train.log + PID;
- monitora processo, GPU (gpu.csv) e progresso (steps no log);
- detecta OOM e recupera (max 3 tentativas: blocks_to_swap 8->16->24->32,
  resume do último checkpoint DeepSpeed);
- descobre checkpoints concluídos (step<N> estável) e sobe para o HF
  AdwolfCzar/groundedsecret com retry/backoff; nunca reenvia confirmados;
- (se autorizado via flag --prune-resume-states) mantém apenas os 2 estados de
  resume global_step* mais recentes, NUNCA tocando nos step<N> (adapters);
- para tudo com um resumo em supervisor.log.
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RUN_ID = 'groundedsecret_7k'
REPO_ROOT = Path('/workspace/diffusion-pipe-easycontrol')
CONFIG = Path('/workspace/configs/groundedsecret_7000.toml')
OUTPUT_BASE = Path('/workspace/checkpoints/groundedsecret_7k')
LOG_DIR = Path(f'/workspace/logs/{RUN_ID}')
STATE_DIR = Path(f'/workspace/state/{RUN_ID}')
HF_REPO = 'AdwolfCzar/groundedsecret'
PYTHON = str(REPO_ROOT / '.venv/bin/python')
DEEPSPEED = str(REPO_ROOT / '.venv/bin/deepspeed')
MAX_RECOVERIES = 3
GPU_POLL_S = 60
PROGRESS_POLL_S = 120
CKPT_POLL_S = 180
STALL_WARN_S = 45 * 60
UPLOAD_BACKOFF_S = [60, 300, 900, 1800]
OOM_PATTERNS = [
    'CUDA out of memory', 'torch.OutOfMemoryError',
    'CUBLAS_STATUS_ALLOC_FAILED', 'CUDA error: out of memory',
]
STEP_RE = re.compile(r'steps?[=:\s]+(\d+)', re.IGNORECASE)


def now():
    return datetime.now(timezone.utc).isoformat()


def log(msg):
    line = f'{now()} {msg}'
    print(line, flush=True)
    with open(LOG_DIR / 'supervisor.log', 'a') as f:
        f.write(line + '\n')


def read_swap_from_config():
    for line in CONFIG.read_text().splitlines():
        m = re.match(r'\s*blocks_to_swap\s*=\s*(\d+)', line)
        if m:
            return int(m.group(1))
    return 8


def write_swap_to_config(new_swap):
    text = CONFIG.read_text()
    text = re.sub(r'(?m)^blocks_to_swap\s*=\s*\d+', f'blocks_to_swap = {new_swap}', text)
    CONFIG.write_text(text)


def newest_run_dir():
    dirs = [d for d in OUTPUT_BASE.glob('*') if d.is_dir()]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def has_resume_checkpoint(run_dir):
    return run_dir is not None and (run_dir / 'latest').exists() and list(run_dir.glob('global_step*'))


def launch_training(resume):
    env = dict(os.environ)
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    env['NCCL_P2P_DISABLE'] = '1'
    env['NCCL_IB_DISABLE'] = '1'
    cmd = [DEEPSPEED, '--num_gpus=1', 'train.py', '--deepspeed', '--config', str(CONFIG)]
    if resume:
        cmd.append('--resume_from_checkpoint')
    train_log = open(LOG_DIR / 'train.log', 'ab')
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env,
                            stdout=train_log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    (STATE_DIR / 'training.pid').write_text(str(proc.pid))
    log(f'treino lançado pid={proc.pid} resume={resume} blocks_to_swap={read_swap_from_config()}')
    return proc


def gpu_row():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20).stdout.strip()
        with open(LOG_DIR / 'gpu.csv', 'a') as f:
            f.write(f'{now()},{out}\n')
        return out
    except Exception as e:
        log(f'gpu.csv falhou: {e!r}')
        return ''


def tail_train_log(nbytes=200_000):
    p = LOG_DIR / 'train.log'
    if not p.exists():
        return ''
    with open(p, 'rb') as f:
        f.seek(max(0, p.stat().st_size - nbytes))
        return f.read().decode('utf-8', errors='replace')


def detect_oom(text):
    return any(pat in text for pat in OOM_PATTERNS)


def last_step_seen(text):
    steps = STEP_RE.findall(text)
    return int(steps[-1]) if steps else None


def checkpoint_ready(step_dir, sizes):
    """step<N> é concluído: adapter presente, sem tmp/, tamanho estável em 2 checagens."""
    if not (step_dir / 'adapter_model.safetensors').exists():
        return False
    if (step_dir / 'tmp').exists():
        return False
    total = sum(f.stat().st_size for f in step_dir.rglob('*') if f.is_file())
    prev = sizes.get(step_dir.name)
    sizes[step_dir.name] = total
    return prev is not None and prev == total


def upload_checkpoint(step_dir):
    from huggingface_hub import HfApi
    step_num = int(step_dir.name.replace('step', ''))
    remote = f'checkpoints/step-{step_num:05d}'
    api = HfApi()
    info = api.upload_folder(
        repo_id=HF_REPO, repo_type='model',
        folder_path=str(step_dir), path_in_repo=remote,
        commit_message=f'Upload {step_dir.name} ({RUN_ID})',
    )
    return getattr(info, 'oid', None) or str(info)


def uploaded_set():
    p = STATE_DIR / 'uploaded-checkpoints.txt'
    return set(p.read_text().split()) if p.exists() else set()


def mark_uploaded(name):
    with open(STATE_DIR / 'uploaded-checkpoints.txt', 'a') as f:
        f.write(name + '\n')


def record_recovery(entry):
    with open(LOG_DIR / 'recovery-history.jsonl', 'a') as f:
        f.write(json.dumps(entry) + '\n')


def prune_resume_states(run_dir):
    """Política autorizada pelo usuário: sobe o estado de resume antigo para o HF
    (resume_states/) e SÓ ENTÃO deleta local, mantendo sempre os 2 mais recentes
    locais para retomada. step<N> (adapters) NUNCA são tocados."""
    from huggingface_hub import HfApi
    states = sorted(run_dir.glob('global_step*'), key=lambda d: int(d.name.replace('global_step', '')))
    api = HfApi()
    for old in states[:-2]:
        try:
            api.upload_folder(repo_id=HF_REPO, repo_type='model',
                              folder_path=str(old), path_in_repo=f'resume_states/{old.name}',
                              commit_message=f'Resume state {old.name} ({RUN_ID})')
            log(f'resume state {old.name} enviado ao HF; removendo local (mantendo os 2 mais recentes)')
            shutil.rmtree(old)
        except Exception as e:
            log(f'upload do resume state {old.name} falhou: {e!r} — NAO deletei; tento na proxima rodada')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prune-resume-states', action='store_true')
    ap.add_argument('--no-launch', action='store_true', help='apenas supervisiona processo existente')
    args = ap.parse_args()

    for d in (LOG_DIR, STATE_DIR, OUTPUT_BASE, Path(f'/workspace/uploads/{RUN_ID}')):
        d.mkdir(parents=True, exist_ok=True)

    shutil.copy(CONFIG, STATE_DIR / 'original-config.toml')
    (STATE_DIR / 'run.json').write_text(json.dumps({
        'run_id': RUN_ID, 'config': str(CONFIG), 'hf_repo': HF_REPO,
        'max_steps': 7000, 'save_every': 500, 'started': now(),
    }, indent=2))

    recoveries = 0
    upload_queue = {}      # name -> {'dir': Path, 'fails': int, 'next_try': ts}
    sizes = {}
    proc = None if args.no_launch else launch_training(resume=False)
    last_gpu = last_prog = last_ckpt = 0.0
    last_step = None
    last_step_time = time.time()
    stall_warned = False

    while True:
        time.sleep(10)
        t = time.time()

        if t - last_gpu >= GPU_POLL_S:
            last_gpu = t
            gpu_row()

        if t - last_prog >= PROGRESS_POLL_S:
            last_prog = t
            text = tail_train_log()
            step = last_step_seen(text)
            if step is not None and step != last_step:
                last_step = step
                last_step_time = t
                stall_warned = False
            elif t - last_step_time > STALL_WARN_S and not stall_warned:
                log(f'AVISO: sem novo step ha {int((t - last_step_time)/60)} min (ultimo step={last_step}). Preservando processo.')
                stall_warned = True
            free_gb = shutil.disk_usage('/workspace').free / 1e9
            if free_gb < 8:
                log(f'AVISO: disco baixo ({free_gb:.1f} GB livres)')

        # processo terminou?
        if proc is not None and proc.poll() is not None:
            rc = proc.returncode
            text = tail_train_log()
            if rc == 0:
                log('treinamento terminou com exit code 0')
                break
            if detect_oom(text):
                if recoveries >= MAX_RECOVERIES:
                    log(f'OOM apos {MAX_RECOVERIES} recuperacoes — parando. Ultimo step={last_step}.')
                    break
                recoveries += 1
                old_swap = read_swap_from_config()
                new_swap = min(old_swap + 8, 32)
                write_swap_to_config(new_swap)
                run_dir = newest_run_dir()
                resume = bool(has_resume_checkpoint(run_dir))
                record_recovery({'time': now(), 'attempt': recoveries, 'reason': 'oom',
                                 'change': f'blocks_to_swap {old_swap}->{new_swap}',
                                 'resume': resume, 'last_step': last_step})
                log(f'OOM detectado. Recuperacao {recoveries}/{MAX_RECOVERIES}: blocks_to_swap {old_swap}->{new_swap}, resume={resume}')
                time.sleep(30)
                proc = launch_training(resume=resume)
            else:
                log(f'treinamento morreu com exit code {rc} SEM padrao de OOM — nao vou reiniciar automaticamente. Veja train.log.')
                break

        # checkpoints novos -> fila de upload
        if t - last_ckpt >= CKPT_POLL_S:
            last_ckpt = t
            run_dir = newest_run_dir()
            if run_dir:
                done = uploaded_set()
                for step_dir in sorted(run_dir.glob('step*'), key=lambda d: int(d.name.replace('step', ''))):
                    if step_dir.name in done or step_dir.name in upload_queue:
                        continue
                    if checkpoint_ready(step_dir, sizes):
                        upload_queue[step_dir.name] = {'dir': step_dir, 'fails': 0, 'next_try': t}
                        log(f'checkpoint concluido: {step_dir.name} -> fila de upload')
                if args.prune_resume_states:
                    try:
                        prune_resume_states(run_dir)
                    except Exception as e:
                        log(f'prune falhou: {e!r}')

        # processa fila de upload
        for name in list(upload_queue):
            item = upload_queue[name]
            if t < item['next_try']:
                continue
            try:
                rev = upload_checkpoint(item['dir'])
                mark_uploaded(name)
                del upload_queue[name]
                log(f'UPLOAD OK: {name} -> {HF_REPO} (rev {rev})')
            except Exception as e:
                item['fails'] += 1
                delay = UPLOAD_BACKOFF_S[min(item['fails'] - 1, len(UPLOAD_BACKOFF_S) - 1)]
                item['next_try'] = t + delay
                log(f'upload de {name} falhou ({item["fails"]}x): {e!r} — retry em {delay}s')

    # drena a fila de upload no final
    log('drenando fila de upload final...')
    deadline = time.time() + 4 * 3600
    while upload_queue and time.time() < deadline:
        t = time.time()
        for name in list(upload_queue):
            item = upload_queue[name]
            if t < item['next_try']:
                continue
            try:
                rev = upload_checkpoint(item['dir'])
                mark_uploaded(name)
                del upload_queue[name]
                log(f'UPLOAD OK (final): {name} (rev {rev})')
            except Exception as e:
                item['fails'] += 1
                delay = UPLOAD_BACKOFF_S[min(item['fails'] - 1, len(UPLOAD_BACKOFF_S) - 1)]
                item['next_try'] = t + delay
                log(f'upload final de {name} falhou: {e!r} — retry em {delay}s')
        time.sleep(15)

    log(f'supervisor encerrado. uploads pendentes: {sorted(upload_queue)} | recoveries: {recoveries} | ultimo step: {last_step}')


if __name__ == '__main__':
    main()
