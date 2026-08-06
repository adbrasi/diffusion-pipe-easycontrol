#!/usr/bin/env python3
"""Supervisor do treino K2-PROXIMACENA-GROUNDED-V2 (krea2_omini_grounded).

Fusão dos dois supervisores validados do projeto:
- krea2_saga_supervisor.py — pausa via save_quit + sampling com o runner
  (não existe sampling in-process para pipelines de referência; não dá para
  treinar e inferir junto em 32 GB);
- groundedsecret_supervisor.py — upload contínuo ao HF, OOM ladder de
  blocks_to_swap com resume, prune de resume states.

Responsabilidades:
- lança o treinamento (deepspeed) com log + PID;
- a cada checkpoint step<N> novo com N % sample_every == 0: pausa o treino
  (save_quit), gera os samples fixos com tools/infer_reference_adapter.py
  (turbo, 8 steps), retoma com --resume_from_checkpoint;
- sobe cada step<N> (adapter) e seus samples para o HF assim que estáveis;
- detecta OOM e recupera (blocks_to_swap 0->8->16->24->32, resume);
- guarda de disco via df (nunca du);
- mantém apenas os 2 global_step* mais recentes (sobe o antigo pro HF antes).

Uso:
  python tools/k2_proximacena_supervisor.py \
      --config /workspace/configs/k2_proximacena_v2.toml \
      --samples /workspace/configs/proximacena_samples.json \
      [--sample-every 250] [--min-free-gb 20] [--resume] [--no-launch]
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

import os

RUN_ID = os.environ.get('K2SUP_RUN_ID', 'k2_proximacena_v2')
REPO_ROOT = Path('/workspace/projects/diffusion-pipe-easycontrol')
PYTHON = '/venv/main/bin/python'
DEEPSPEED = '/venv/main/bin/deepspeed'
TURBO_LORA = '/workspace/models/krea2/loras/krea2_turbo_lora_rank_64_bf16.safetensors'
HF_REPO = os.environ.get('K2SUP_HF_REPO', 'AdwolfCzar/k2-proximacena-grounded-v2-full')
LOG_DIR = Path(f'/workspace/logs/{RUN_ID}')
STATE_DIR = Path(f'/workspace/state/{RUN_ID}')
SAMPLES_OUT = Path(f'/workspace/outputs/{RUN_ID}_samples')
OOM_PATTERNS = ['CUDA out of memory', 'torch.OutOfMemoryError',
                'CUBLAS_STATUS_ALLOC_FAILED', 'CUDA error: out of memory']
MAX_RECOVERIES = 4
SWAP_LADDER = [8, 16, 24, 32]
UPLOAD_BACKOFF_S = [60, 300, 900, 1800]


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def log(msg):
    line = f'{now()} {msg}'
    print(line, flush=True)
    with open(LOG_DIR / 'supervisor.log', 'a') as f:
        f.write(line + '\n')


def read_toml_value(path, key, cast=str):
    for line in Path(path).read_text().splitlines():
        m = re.match(rf"\s*{key}\s*=\s*'?([^'#\n]+)'?", line)
        if m:
            return cast(m.group(1).strip())
    return None


def write_swap(config_path, new_swap):
    text = Path(config_path).read_text()
    if re.search(r'(?m)^blocks_to_swap\s*=', text):
        text = re.sub(r'(?m)^blocks_to_swap\s*=\s*\d+', f'blocks_to_swap = {new_swap}', text)
    else:
        text = text.replace('[model]', f'blocks_to_swap = {new_swap}\n\n[model]', 1)
    Path(config_path).write_text(text)


def free_gb(path='/workspace'):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2**30


def newest_run_dir(output_base: Path):
    dirs = [d for d in output_base.glob('*') if d.is_dir()]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def step_checkpoints(run_dir):
    if run_dir is None:
        return {}
    out = {}
    for d in run_dir.glob('step*'):
        m = re.fullmatch(r'step(\d+)', d.name)
        if m and (d / 'adapter_model.safetensors').exists() and not (d / 'tmp').exists():
            out[int(m.group(1))] = d
    return out


def stable(path: Path, wait=5):
    def size():
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    s1 = size()
    time.sleep(wait)
    return size() == s1


class Trainer:
    def __init__(self, config):
        self.config = config
        self.proc = None
        self.log_path = None

    def launch(self, resume):
        cmd = [DEEPSPEED, '--num_gpus=1', 'train.py', '--deepspeed',
               '--config', self.config]
        if resume:
            cmd.append('--resume_from_checkpoint')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        self.log_path = LOG_DIR / f'train_{stamp}.log'
        env = dict(os.environ,
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                   NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1')
        logf = open(self.log_path, 'ab')
        self.proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=logf,
                                     stderr=subprocess.STDOUT,
                                     start_new_session=True, env=env)
        (STATE_DIR / 'training.pid').write_text(str(self.proc.pid))
        log(f'treino lançado pid={self.proc.pid} resume={resume} '
            f'blocks_to_swap={read_toml_value(self.config, "blocks_to_swap", int)} log={self.log_path}')

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self, run_dir=None):
        if not self.alive():
            return
        # Parada limpa: o trainer observa <run_dir>/save_quit e salva o
        # checkpoint DeepSpeed COMPLETO antes de sair. SIGTERM destrói o resume.
        if run_dir is not None:
            sig = Path(run_dir) / 'save_quit'
            log(f'parando treino pid={self.proc.pid} via {sig}')
            sig.touch()
            try:
                self.proc.wait(timeout=900)
                time.sleep(10)
                return
            except subprocess.TimeoutExpired:
                log('save_quit não surtiu efeito em 15 min; caindo para SIGTERM')
        pgid = os.getpgid(self.proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            self.proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            self.proc.wait(timeout=60)
        time.sleep(10)

    def tail(self, nbytes=200_000):
        if self.log_path is None or not self.log_path.exists():
            return ''
        data = self.log_path.read_bytes()
        return data[-nbytes:].decode('utf-8', 'replace')

    def tail_has_oom(self):
        return any(p in self.tail(20_000) for p in OOM_PATTERNS)


def run_samples(config, step_dir: Path, samples, seed=76):
    out_dir = SAMPLES_OUT / step_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for i, s in enumerate(samples):
        out = out_dir / f'sample{i}_seed{seed}.png'
        if out.exists():
            ok += 1
            continue
        # Turbo (decisão do usuário 2026-08-05): destilado para inferência; na
        # bisseção o turbo 8 steps rendeu micro-textura nítida onde o raw 28
        # steps/guidance 5.5 saiu liso/plástico neste runner.
        cmd = [PYTHON, 'tools/infer_reference_adapter.py',
               '--config', config, '--adapter', str(step_dir),
               '--reference', s['reference'], '--prompt', s['prompt'],
               '--width', str(s.get('width', 1024)), '--height', str(s.get('height', 1024)),
               '--seed', str(seed), '--turbo-lora', TURBO_LORA,
               '--output', str(out)]
        r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                           timeout=1800)
        if r.returncode == 0 and out.exists():
            ok += 1
        else:
            (out_dir / f'sample{i}_error.log').write_text(
                (r.stdout or '') + '\n' + (r.stderr or ''))
    (out_dir / 'sampling_inputs.json').write_text(json.dumps(
        {'seed': seed, 'variant': 'raw', 'steps': 28, 'text_guidance': 5.5,
         'samples': samples}, indent=2))
    log(f'samples {step_dir.name}: {ok}/{len(samples)} ok -> {out_dir}')
    return ok


def upload_folder(folder: Path, remote: str, message: str):
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.upload_folder(repo_id=HF_REPO, repo_type='model',
                             folder_path=str(folder), path_in_repo=remote,
                             commit_message=message)
    return getattr(info, 'oid', None) or str(info)


def uploaded_set():
    p = STATE_DIR / 'uploaded.txt'
    return set(p.read_text().split()) if p.exists() else set()


def mark_uploaded(name):
    with open(STATE_DIR / 'uploaded.txt', 'a') as f:
        f.write(name + '\n')


def prune_local_checkpoints(run_dir, uploaded, sampled, keep=4):
    """Deleta step<N> locais que já foram ENVIADOS ao HF e SAMPLEADOS,
    mantendo os `keep` mais recentes. Sem isso, ~60 checkpoints de 0,46 GB
    consomem a margem de disco do cache."""
    ckpts = step_checkpoints(run_dir)
    for n in sorted(ckpts)[:-keep]:
        d = ckpts[n]
        if d.name in uploaded and n in sampled:
            shutil.rmtree(d)
            log(f'checkpoint local {d.name} removido (já no HF e sampleado)')


def prune_resume_states(run_dir):
    """Sobe o global_step* antigo pro HF e só então deleta, mantendo os 2 mais
    recentes locais. step<N> (adapters) NUNCA são tocados."""
    states = sorted(run_dir.glob('global_step*'),
                    key=lambda d: int(d.name.replace('global_step', '')))
    for old in states[:-2]:
        try:
            upload_folder(old, f'resume_states/{old.name}', f'Resume state {old.name}')
            log(f'resume state {old.name} no HF; removendo local')
            shutil.rmtree(old)
        except Exception as e:
            log(f'upload do resume state {old.name} falhou: {e!r} — mantido; retry depois')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--samples', required=True)
    ap.add_argument('--sample-every', type=int, default=250)
    ap.add_argument('--min-free-gb', type=float, default=20)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--no-launch', action='store_true')
    args = ap.parse_args()

    for d in (LOG_DIR, STATE_DIR, SAMPLES_OUT):
        d.mkdir(parents=True, exist_ok=True)

    samples = json.loads(Path(args.samples).read_text())['samples']
    output_base = Path(read_toml_value(args.config, 'output_dir'))
    max_steps = read_toml_value(args.config, 'max_steps', int)
    output_base.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, STATE_DIR / 'config-as-launched.toml')
    log(f'supervisor start: config={args.config} max_steps={max_steps} '
        f'sample_every={args.sample_every} exemplos={len(samples)} hf={HF_REPO}')

    # checkpoints pré-existentes contam como já sampleados/enviados (resume)
    sampled = set()
    run_dir = newest_run_dir(output_base)
    if args.resume and run_dir is not None:
        for n, d in step_checkpoints(run_dir).items():
            if (SAMPLES_OUT / d.name).exists():
                sampled.add(n)
        if sampled:
            log(f'{len(sampled)} checkpoints pré-existentes já sampleados')

    trainer = Trainer(args.config)
    if not args.no_launch:
        run_dir = newest_run_dir(output_base)
        trainer.launch(resume=args.resume and bool(
            run_dir and (run_dir / 'latest').exists()))

    recoveries = 0
    upload_queue = {}   # name -> {'dir': Path, 'remote': str, 'fails': int, 'next_try': ts}
    done_final = False
    step_re = re.compile(r'step[=:\s]+(\d+)', re.IGNORECASE)
    last_step = None

    while True:
        time.sleep(30)
        t = time.time()
        run_dir = newest_run_dir(output_base)

        if free_gb() < args.min_free_gb:
            log(f'DISCO BAIXO ({free_gb():.1f} GB livres) — parando tudo')
            trainer.stop(run_dir=run_dir)
            sys.exit(2)

        text = trainer.tail(50_000)
        steps_seen = step_re.findall(text)
        if steps_seen:
            last_step = int(steps_seen[-1])

        ckpts = step_checkpoints(run_dir)
        done_final = bool(ckpts) and max_steps in ckpts

        # processo morreu?
        if not args.no_launch and not trainer.alive():
            rc = trainer.proc.returncode if trainer.proc else None
            if done_final:
                log(f'treino terminou rc={rc} com checkpoint final step{max_steps}')
            elif trainer.tail_has_oom() and recoveries < MAX_RECOVERIES:
                recoveries += 1
                cur = read_toml_value(args.config, 'blocks_to_swap', int) or 0
                nxt = next((s for s in SWAP_LADDER if s > cur), 32)
                write_swap(args.config, nxt)
                log(f'OOM (recovery {recoveries}/{MAX_RECOVERIES}): '
                    f'blocks_to_swap {cur}->{nxt}, resume')
                trainer.launch(resume=bool(run_dir and (run_dir / 'latest').exists()))
                time.sleep(30)
                continue
            else:
                recoveries += 1
                if recoveries > MAX_RECOVERIES:
                    log(f'treino MORREU rc={rc} {recoveries}x seguidas — desistindo; '
                        f'supervisor segue só para drenar uploads. Ver {trainer.log_path}')
                else:
                    log(f'treino MORREU rc={rc} sem OOM reconhecido (ultimo step={last_step}) '
                        f'— relançando com resume ({recoveries}/{MAX_RECOVERIES}). '
                        f'Ver {trainer.log_path}')
                    time.sleep(30)
                    trainer.launch(resume=bool(run_dir and (run_dir / 'latest').exists()))
                    time.sleep(30)
                    continue

        # sampling nos checkpoints múltiplos de sample_every (e no final)
        pending = [n for n in sorted(ckpts) if n not in sampled
                   and (n % args.sample_every == 0 or n == max_steps)]
        for n in pending:
            if not stable(ckpts[n]):
                continue
            was_alive = trainer.alive()
            if was_alive:
                log(f'checkpoint step{n}: pausando treino para samples')
                trainer.stop(run_dir=run_dir)
            try:
                run_samples(args.config, ckpts[n], samples)
            except Exception as e:
                log(f'sampling step{n} falhou: {e!r}')
            sampled.add(n)
            sdir = SAMPLES_OUT / ckpts[n].name
            if sdir.exists() and f'samples/{sdir.name}' not in upload_queue:
                upload_queue[f'samples/{sdir.name}'] = {
                    'dir': sdir, 'remote': f'samples/{sdir.name}', 'fails': 0, 'next_try': t}
            if was_alive and n != max_steps:
                trainer.launch(resume=True)
                time.sleep(30)
            break  # um por ciclo

        # checkpoints novos -> fila de upload
        done = uploaded_set()
        for n in sorted(ckpts):
            d = ckpts[n]
            key = f'checkpoints/{d.name}'
            if d.name in done or key in upload_queue:
                continue
            if stable(d, wait=3):
                upload_queue[key] = {'dir': d, 'remote': f'checkpoints/step-{n:05d}',
                                     'fails': 0, 'next_try': t}
                log(f'checkpoint concluído: {d.name} -> fila de upload')

        if run_dir is not None:
            prune_resume_states(run_dir)
            prune_local_checkpoints(run_dir, uploaded_set(), sampled)

        # processa fila de upload
        for key in list(upload_queue):
            item = upload_queue[key]
            if t < item['next_try']:
                continue
            try:
                rev = upload_folder(item['dir'], item['remote'],
                                    f'Upload {item["remote"]} ({RUN_ID})')
                if key.startswith('checkpoints/'):
                    mark_uploaded(item['dir'].name)
                del upload_queue[key]
                log(f'UPLOAD OK: {key} -> {HF_REPO} (rev {rev})')
            except Exception as e:
                item['fails'] += 1
                delay = UPLOAD_BACKOFF_S[min(item['fails'] - 1, len(UPLOAD_BACKOFF_S) - 1)]
                item['next_try'] = t + delay
                log(f'upload {key} falhou ({item["fails"]}x): {e!r} — retry em {delay}s')

        if done_final and max_steps in sampled and not trainer.alive() and not upload_queue:
            log('RUN COMPLETO: checkpoint final salvo, sampleado e enviado.')
            sys.exit(0)


if __name__ == '__main__':
    main()
