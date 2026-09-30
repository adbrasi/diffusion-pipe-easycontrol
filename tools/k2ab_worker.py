"""Run explicitly queued GPU jobs serially under supervisor."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

QUEUE = Path('/workspace/k2ab/ops/jobs')
QUEUE.mkdir(parents=True, exist_ok=True)


def write_state(path, state):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(path)


def monitor(process, path):
    with path.open('a') as stream:
        stream.write('utc,memory_MiB,util_percent,power_W\n')
        while process.poll() is None:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,power.draw',
                '--format=csv,noheader,nounits'], capture_output=True, text=True)
            stream.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ',' + result.stdout.strip() + '\n')
            stream.flush()
            time.sleep(2)


def execute(path, job):
    state_path = path.with_suffix('.state.json')
    old = json.loads(state_path.read_text()) if state_path.exists() else {}
    if old.get('status') in ('done', 'failed'):
        return
    command = job['argv'][:]
    if job.get('train_output') and (old.get('status') == 'running' or job.get('resume')):
        runs = sorted(Path(job['train_output']).glob('*/latest'))
        if runs and '--resume_from_checkpoint' not in command:
            command += ['--resume_from_checkpoint', runs[-1].parent.name]
    env = os.environ.copy()
    env.update(NCCL_P2P_DISABLE='1', OMP_NUM_THREADS='8', HF_HUB_DISABLE_PROGRESS_BARS='1',
               TOKENIZERS_PARALLELISM='false', PATH='/venv/main/bin:' + env.get('PATH', ''))
    env.update(job.get('env', {}))
    log = Path(job['log'])
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log.open('a') as stream:
        process = subprocess.Popen(command, cwd=job['cwd'], env=env, stdout=stream, stderr=subprocess.STDOUT)
        write_state(state_path, dict(status='running', start=start, argv=command, pid=process.pid))
        thread = threading.Thread(target=monitor, args=(process, log.with_suffix('.gpu.csv')), daemon=True)
        thread.start()
        code = process.wait()
        thread.join(timeout=3)
    write_state(state_path, dict(status='done' if code == 0 else 'failed', start=start,
        end=time.time(), seconds=time.time()-start, returncode=code, argv=command))
    print(path.stem, 'completed', code, flush=True)


while True:
    for path in sorted(QUEUE.glob('*.job.json')):
        try:
            execute(path, json.loads(path.read_text()))
        except Exception:
            traceback.print_exc()
            write_state(path.with_suffix('.state.json'), dict(status='failed', error=traceback.format_exc()))
    time.sleep(3)
