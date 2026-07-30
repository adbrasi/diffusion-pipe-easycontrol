#!/usr/bin/env python3
"""Supervisor do treino KREA2 edit saga.

Responsabilidades (CLAUDE.md §7/§9):
- inicia o treinamento (deepspeed) com log em arquivo + PID registrado;
- a cada checkpoint step<N> novo (N % sample_every == 0): PAUSA o treino
  (o train.py não tem sampling in-process para pipelines de referência;
  docs/KREA2_MULTIREF_RECEITA.md §7.3 — não dá para treinar e inferir junto),
  gera samples dos 3 exemplos fixos com tools/infer_reference_adapter.py
  (turbo, 8 steps), e RETOMA com --resume_from_checkpoint;
- detecta OOM e recupera (blocks_to_swap 8->16->24->32, resume);
- guarda de disco: aborta com resumo se df < min_free_gb (df, nunca du);
- para com resumo em supervisor.log.

O upload para o HF é de um processo separado (krea2_saga_uploader.py).

Uso:
  python tools/krea2_saga_supervisor.py --config examples/krea2_edit_saga/train.toml \
      --samples /workspace/outputs/sampling_inputs/samples.json [--max-steps-check]
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path('/workspace/projects/diffusion-pipe')
PYTHON = '/venv/main/bin/python'
DEEPSPEED = '/venv/main/bin/deepspeed'
TURBO_LORA = '/workspace/models/krea2/loras/krea2_turbo_lora_rank_64_bf16.safetensors'
LOG_DIR = Path('/workspace/logs/krea2_edit_saga')
SAMPLES_OUT = Path('/workspace/outputs/krea2_edit_saga_samples')
OOM_PATTERNS = ['CUDA out of memory', 'torch.OutOfMemoryError',
                'CUBLAS_STATUS_ALLOC_FAILED', 'CUDA error: out of memory']
MAX_RECOVERIES = 4
SWAP_LADDER = [8, 16, 24, 32]


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
    text = re.sub(r'(?m)^blocks_to_swap\s*=\s*\d+', f'blocks_to_swap = {new_swap}', text)
    Path(config_path).write_text(text)


def free_gb(path='/workspace'):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2**30


def newest_run_dir(output_base: Path):
    dirs = [d for d in output_base.glob('*') if d.is_dir()]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def step_checkpoints(run_dir: Path):
    out = {}
    if run_dir is None:
        return out
    for d in run_dir.glob('step*'):
        m = re.fullmatch(r'step(\d+)', d.name)
        if m and (d / 'adapter_model.safetensors').exists():
            out[int(m.group(1))] = d
    return out


def stable(path: Path, wait=5):
    def size():
        return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
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
        log(f'treino lançado pid={self.proc.pid} resume={resume} log={self.log_path}')

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self, run_dir=None):
        if not self.alive():
            return
        # Parada limpa: o trainer observa <run_dir>/save_quit a cada step e,
        # ao vê-lo, salva o checkpoint DeepSpeed COMPLETO (global_step* +
        # latest) e sai. SIGTERM mataria sem estado de resume — foi
        # exatamente o bug do step 500 (2026-07-29T17:19).
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
        log(f'parando treino pid={self.proc.pid} (SIGTERM no grupo {pgid})')
        os.killpg(pgid, signal.SIGTERM)
        try:
            self.proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            log('SIGTERM não bastou; SIGKILL no grupo')
            os.killpg(pgid, signal.SIGKILL)
            self.proc.wait(timeout=60)
        time.sleep(10)  # deixa a VRAM ser liberada

    def tail_has_oom(self):
        if self.log_path is None or not self.log_path.exists():
            return False
        tail = self.log_path.read_bytes()[-20000:].decode('utf-8', 'replace')
        return any(p in tail for p in OOM_PATTERNS)


def run_samples(config, step_dir: Path, samples, seed=76):
    out_dir = SAMPLES_OUT / step_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for i, s in enumerate(samples):
        out = out_dir / f'sample{i}_seed{seed}.png'
        cmd = [PYTHON, 'tools/infer_reference_adapter.py',
               '--config', config, '--adapter', str(step_dir),
               '--reference', s['reference'], '--prompt', s['prompt'],
               '--width', str(s.get('width', 512)), '--height', str(s.get('height', 512)),
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
        {'seed': seed, 'turbo_lora': TURBO_LORA, 'samples': samples}, indent=2))
    log(f'samples {step_dir.name}: {ok}/{len(samples)} ok -> {out_dir}')
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--samples', required=True, help='json com os 3 exemplos fixos')
    ap.add_argument('--sample-every', type=int, default=500)
    ap.add_argument('--min-free-gb', type=float, default=30.0)
    ap.add_argument('--resume', action='store_true',
                    help='Retomar de um checkpoint existente em vez de comecar do zero. '
                         'OBRIGATORIO ao continuar um treino (ex.: mudanca de resolucao); '
                         'sem isto a primeira execucao ignora o global_step* e recomeca.')
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_OUT.mkdir(parents=True, exist_ok=True)
    samples = json.loads(Path(args.samples).read_text())['samples']
    output_base = Path(read_toml_value(args.config, 'output_dir'))
    max_steps = read_toml_value(args.config, 'max_steps', int)
    log(f'supervisor start config={args.config} max_steps={max_steps} '
        f'sample_every={args.sample_every} exemplos={len(samples)}')

    trainer = Trainer(args.config)
    if args.resume:
        run_dir_now = newest_run_dir(output_base)
        latest = (run_dir_now / 'latest') if run_dir_now else None
        if latest is None or not latest.exists():
            log('ERRO: --resume pedido mas nao ha checkpoint (latest) em ' + str(output_base))
            sys.exit(3)
        log(f'retomando de {latest.read_text().strip()} em {run_dir_now}')
    trainer.launch(resume=args.resume)
    time.sleep(30)  # não competir com a largada do launcher

    sampled = set()
    recoveries = 0
    while True:
        time.sleep(60)
        run_dir = newest_run_dir(output_base)
        if free_gb() < args.min_free_gb:
            log(f'DISCO BAIXO ({free_gb():.1f} GB livres) — parando tudo')
            trainer.stop(run_dir=run_dir)
            sys.exit(2)

        ckpts = step_checkpoints(run_dir)
        done_final = ckpts and max_steps in ckpts

        if not trainer.alive():
            rc = trainer.proc.returncode
            if done_final:
                log(f'treino terminou rc={rc} com checkpoint final step{max_steps}')
            else:
                if trainer.tail_has_oom() and recoveries < MAX_RECOVERIES:
                    recoveries += 1
                    cur = read_toml_value(args.config, 'blocks_to_swap', int) or 8
                    nxt = next((s for s in SWAP_LADDER if s > cur), 32)
                    write_swap(args.config, nxt)
                    log(f'OOM detectado (recovery {recoveries}/{MAX_RECOVERIES}): '
                        f'blocks_to_swap {cur}->{nxt}, resume')
                    trainer.launch(resume=bool(ckpts))
                    time.sleep(30)
                    continue
                log(f'treino MORREU rc={rc} sem OOM reconhecido — ver {trainer.log_path}')
                sys.exit(1)

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
            if was_alive or not done_final:
                if n != max_steps:
                    trainer.launch(resume=True)
                    time.sleep(30)
            break  # um por ciclo

        if done_final and max_steps in sampled and not trainer.alive():
            log('SAGA COMPLETA: checkpoint final salvo e sampleado')
            sys.exit(0)


if __name__ == '__main__':
    main()
