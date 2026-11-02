#!/usr/bin/env python3
"""Supervisor do BETA1-R (derivado de tools/krea2_saga_supervisor.py, o que rodou o beta1).

- lança o treino (deepspeed) com log em arquivo;
- a cada adapter step<N> novo com N % sample_every == 0: PAUSA o treino via
  <run_dir>/save_quit (salva estado DeepSpeed completo; nunca SIGTERM), roda
  tools/beta1r_sample.py (3 exemplos fixos do beta1 + pares held-out de cópia,
  turbo 8 steps seed 76) e RETOMA com --resume_from_checkpoint;
- sobe adapter + samples + metrics para o HF (thread em background, sem README);
- poda: mantém 1 estado DeepSpeed (global_step*) e os 2 adapters locais mais
  recentes (os mais antigos só são apagados depois de confirmados no HF);
- guarda de disco: para limpo se df < min_free_gb.

Uso:
  python tools/beta1r_supervisor.py --config examples/beta1r/train.toml \
      --cases /workspace/runs/beta1r_samples_inputs/cases.json [--resume]
Parar: touch /workspace/runs/beta1r_supervisor.stop (para o treino com save_quit e sai).
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV = Path('/workspace/dpe-venv/bin')
TURBO_LORA = '/dev/shm/models/krea2/loras/krea2_turbo_lora_rank_64_bf16.safetensors'
LOG_DIR = Path('/workspace/runs/logs')
SAMPLES_OUT = Path(os.environ.get('BETA1R_SAMPLES_OUT', '/workspace/runs/beta1r_samples'))
STOP_FILE = Path('/workspace/runs/beta1r_supervisor.stop')
HF_REPO = 'AdwolfCzar/krea2-beta1r'


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def log(msg):
    line = f'{now()} {msg}'
    print(line, flush=True)
    with open(LOG_DIR / 'beta1r_supervisor.log', 'a') as f:
        f.write(line + '\n')


def read_toml_value(path, key, cast=str):
    for line in Path(path).read_text().splitlines():
        m = re.match(rf"\s*{key}\s*=\s*'?([^'#\n]+)'?", line)
        if m:
            return cast(m.group(1).strip())
    return None


def free_gb(path='/workspace'):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2**30


def newest_run_dir(base: Path):
    dirs = [d for d in base.glob('*') if d.is_dir()]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def step_checkpoints(run_dir):
    out = {}
    if run_dir is None:
        return out
    for d in run_dir.glob('step*'):
        m = re.fullmatch(r'step(\d+)', d.name)
        if m and (d / 'adapter_model.safetensors').exists():
            out[int(m.group(1))] = d
    return out


def stable(path: Path, wait=5):
    size = lambda: sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
    s1 = size()
    time.sleep(wait)
    return size() == s1


class Uploader:
    """Fila de uploads em background; registra o que já subiu."""

    def __init__(self, config):
        from huggingface_hub import HfApi
        self.api = HfApi(token=os.environ['HF_TOKEN'])
        self.config = config
        self.done = set()
        self.lock = threading.Lock()
        self.threads = []
        self.upload_file(config, 'configs/train.toml')
        self.upload_file(read_toml_value(config, 'dataset'), 'configs/dataset.toml')

    def upload_file(self, local, remote):
        for i in range(5):
            try:
                self.api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                                     repo_id=HF_REPO, commit_message=f'upload {remote}')
                return True
            except Exception as e:
                log(f'upload {remote} falhou ({i + 1}/5): {e!r}')
                time.sleep(30)
        return False

    def upload_folder(self, local, remote):
        for i in range(5):
            try:
                self.api.upload_folder(folder_path=str(local), path_in_repo=remote, repo_id=HF_REPO,
                                       commit_message=f'upload {remote}')
                return True
            except Exception as e:
                log(f'upload {remote} falhou ({i + 1}/5): {e!r}')
                time.sleep(30)
        return False

    def submit(self, step, ckpt_dir, sample_dir):
        def job():
            ok = self.upload_folder(ckpt_dir, f'checkpoints/step{step}')
            if sample_dir is not None and sample_dir.exists():
                ok = self.upload_folder(sample_dir, f'samples/step{step}') and ok
            if ok:
                with self.lock:
                    self.done.add(step)
                log(f'HF: step{step} enviado ({HF_REPO})')
        t = threading.Thread(target=job, daemon=True)
        t.start()
        self.threads.append(t)

    def wait(self):
        for t in self.threads:
            t.join()


class NoUploader:
    def __init__(self):
        self.done, self.lock = set(), threading.Lock()

    def submit(self, step, ckpt_dir, sample_dir):
        log(f'(no-upload) step{step}')

    def wait(self):
        pass


def prune(run_dir, uploader, keep_states=1, keep_adapters=2):
    if run_dir is None:
        return
    states = sorted(run_dir.glob('global_step*'), key=lambda d: d.stat().st_mtime, reverse=True)
    for old in states[keep_states:]:
        shutil.rmtree(old, ignore_errors=True)
        log(f'poda: {old.name} removido ({free_gb():.1f} GB livres)')
    ckpts = step_checkpoints(run_dir)
    for n in sorted(ckpts)[:-keep_adapters]:
        with uploader.lock:
            uploaded = n in uploader.done
        if uploaded:
            shutil.rmtree(ckpts[n], ignore_errors=True)
            log(f'poda: adapter step{n} local removido (já no HF)')


class Trainer:
    def __init__(self, config):
        self.config = config
        self.proc = None
        self.log_path = None

    def launch(self, resume):
        cmd = [str(VENV / 'deepspeed'), '--num_gpus=1', 'train.py', '--deepspeed', '--config', self.config]
        if resume:
            cmd.append('--resume_from_checkpoint')
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        self.log_path = LOG_DIR / f'beta1r_train_{stamp}.log'
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                   NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', PATH=f"{VENV}:{os.environ['PATH']}")
        self.proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=open(self.log_path, 'ab'),
                                     stderr=subprocess.STDOUT, start_new_session=True, env=env)
        (LOG_DIR / 'beta1r_train_latest.log').unlink(missing_ok=True)
        (LOG_DIR / 'beta1r_train_latest.log').symlink_to(self.log_path)
        log(f'treino lançado pid={self.proc.pid} resume={resume} log={self.log_path}')

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self, run_dir):
        if not self.alive():
            return
        sig = Path(run_dir) / 'save_quit'
        log(f'parando treino pid={self.proc.pid} via {sig}')
        sig.touch()
        try:
            self.proc.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            log('ERRO: save_quit sem efeito em 20 min; SIGTERM no grupo')
            os.killpg(os.getpgid(self.proc.pid), 15)
            self.proc.wait(timeout=120)
        sig.unlink(missing_ok=True)
        time.sleep(10)


def run_samples(config, step, ckpt_dir, cases):
    out = SAMPLES_OUT / f'step{step}'
    cmd = [str(VENV / 'python'), '-u', 'tools/beta1r_sample.py', '--config', config, '--adapter', str(ckpt_dir),
           '--cases', str(cases), '--out', str(out), '--turbo-lora', TURBO_LORA]
    t0 = time.time()
    with open(LOG_DIR / f'beta1r_sample_step{step}.log', 'w') as lf:
        r = subprocess.run(cmd, cwd=REPO_ROOT, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    tail = (LOG_DIR / f'beta1r_sample_step{step}.log').read_text()[-400:]
    m = re.search(r'COPY_RATE (\S+) mean_dhash_gen_ref (\S+)', tail)
    log(f'samples step{step}: rc={r.returncode} {time.time() - t0:.0f}s '
        f'copy_rate={m.group(1) if m else "?"} mean_dhash_gen_ref={m.group(2) if m else "?"} -> {out}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--sample-every', type=int, default=250)
    ap.add_argument('--min-free-gb', type=float, default=3.0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--no-upload', action='store_true', help='smoke: não sobe nada ao HF')
    args = ap.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_OUT.mkdir(parents=True, exist_ok=True)
    output_base = Path(read_toml_value(args.config, 'output_dir'))
    max_steps = read_toml_value(args.config, 'max_steps', int)
    log(f'supervisor start config={args.config} max_steps={max_steps} sample_every={args.sample_every}')
    uploader = NoUploader() if args.no_upload else Uploader(args.config)
    trainer = Trainer(args.config)
    if args.resume:
        rd = newest_run_dir(output_base)
        if rd is None or not (rd / 'latest').exists():
            log('ERRO: --resume sem checkpoint latest')
            sys.exit(3)
        log(f'retomando de {(rd / "latest").read_text().strip()} em {rd}')
    sampled = set()
    for p in SAMPLES_OUT.glob('step*'):
        if (p / 'metrics.json').exists():
            sampled.add(int(p.name[4:]))
    trainer.launch(resume=args.resume)
    time.sleep(60)
    while True:
        time.sleep(30)
        run_dir = newest_run_dir(output_base)
        if STOP_FILE.exists():
            log('stop file: parando treino e saindo')
            trainer.stop(run_dir)
            STOP_FILE.unlink(missing_ok=True)
            uploader.wait()
            sys.exit(0)
        if free_gb() < args.min_free_gb:
            log(f'DISCO BAIXO ({free_gb():.1f} GB) — parando')
            trainer.stop(run_dir)
            uploader.wait()
            sys.exit(2)
        ckpts = step_checkpoints(run_dir)
        pending = [n for n in sorted(ckpts) if n not in sampled
                   and (n % args.sample_every == 0 or n == max_steps)]
        if pending:
            n = pending[0]
            if stable(ckpts[n]):
                was_alive = trainer.alive()
                if was_alive:
                    log(f'step{n}: pausando treino para samples')
                    trainer.stop(run_dir)
                out = None
                try:
                    out = run_samples(args.config, n, ckpts[n], args.cases)
                except Exception as e:
                    log(f'sampling step{n} falhou: {e!r}')
                sampled.add(n)
                uploader.submit(n, ckpts[n], out)
                if n != max_steps:
                    trainer.launch(resume=True)
                    time.sleep(60)
                prune(run_dir, uploader)
                continue
        if not trainer.alive():
            rc = trainer.proc.returncode
            if max_steps in ckpts and max_steps in sampled:
                log(f'treino concluído rc={rc}; aguardando uploads')
                uploader.wait()
                prune(run_dir, uploader)
                sys.exit(0)
            if max_steps in ckpts:
                continue
            log(f'treino MORREU rc={rc} — ver {trainer.log_path}')
            uploader.wait()
            sys.exit(1)


if __name__ == '__main__':
    main()
