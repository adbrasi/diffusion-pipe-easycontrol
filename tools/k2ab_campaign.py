"""Finish the gated A/B serially, preserve recovery states, then render and score."""
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/workspace/k2ab')
REPO = Path('/workspace/diffusion-pipe-easycontrol')
QUEUE = ROOT / 'ops/jobs'
STATE = ROOT / 'artifacts/campaign_state.json'
PYTHON = '/venv/main/bin/python'


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def note(message):
    now = dt.datetime.now(dt.timezone.utc)
    start = dt.datetime(2026, 9, 30, 16, 9, tzinfo=dt.timezone.utc)
    cost = (now-start).total_seconds()/3600*.62
    with (REPO / 'docs/KREA2_AB_RUN_LOG.md').open('a') as stream:
        stream.write(f'\n## {now:%Y-%m-%d %H:%M UTC} — Campanha\n\n{message}\n\n'
                     f'Custo Krea acumulado estimado desde16:09UTC:US${cost:.2f}, '
                     'inclui setup/cache/ociosidade, não é extrato.\n')
    print(message, flush=True)
    env = os.environ.copy()
    env.update(GIT_ASKPASS='/workspace/nextscene_ops/git-askpass.sh', GIT_TERMINAL_PROMPT='0')
    try:
        subprocess.run(['git', 'commit', '--only', '-m', 'Record Krea 2 campaign milestone',
            'docs/KREA2_AB_RUN_LOG.md'], cwd=REPO, env=env, check=True, timeout=30)
        subprocess.run(['git', 'push', 'origin', 'claude/elegant-ptolemy-8kywqy'],
            cwd=REPO, env=env, check=True, timeout=45)
    except subprocess.SubprocessError as error:
        print('Git sync pending:', error, flush=True)


def enqueue(name, argv, cwd=REPO, train_output=None):
    path = QUEUE / f'{name}.job.json'
    if not path.exists():
        job = dict(argv=argv, cwd=str(cwd), log=str(ROOT / 'artifacts' / f'{name}.log'))
        if train_output:
            job['train_output'] = str(train_output)
        atomic(path, job)
    return path


def prune_old_states():
    for arm in (ROOT / 'checkpoints').glob('*_probe'):
        for run in arm.glob('*'):
            latest = run / 'latest'
            if not latest.exists():
                continue
            states = sorted(run.glob('global_step*'), key=lambda path: int(path.name[11:]))
            keep = {path.name for path in states[-2:]} | {latest.read_text().strip()}
            for path in states:
                if path.name not in keep and time.time()-path.stat().st_mtime > 60:
                    shutil.rmtree(path)
                    print('Pruned obsolete optimizer state', path, flush=True)


def wait_job(path, campaign):
    state_path = path.with_suffix('.state.json')
    while True:
        if (ROOT / 'ops/stop_campaign').exists():
            campaign['status'] = 'stopped'
            atomic(STATE, campaign)
            raise SystemExit('Campaign stopped before scheduling dependent work')
        prune_old_states()
        status = json.loads(state_path.read_text()) if state_path.exists() else {}
        campaign['current_job'] = path.stem
        campaign['job_status'] = status
        for arm in campaign.get('checkpoint_arms', ('A_native_probe', 'B_beta1_fixed_probe')):
            for adapter in (ROOT / 'checkpoints' / arm).glob('*/step*/adapter_model.safetensors'):
                milestone = str(adapter.relative_to(ROOT))
                if milestone not in campaign['milestones'] and time.time()-adapter.stat().st_mtime > 15:
                    campaign['milestones'].append(milestone)
                    note(f'{arm} salvou {adapter.parent.name}; adapter local: {adapter}. Sync HF contínuo ativo.')
        atomic(STATE, campaign)
        if status.get('status') == 'done':
            return
        if status.get('status') == 'failed':
            raise RuntimeError(f'{path.stem} failed; inspect its log')
        time.sleep(10)


def adapter_for(arm, step):
    files = sorted((ROOT / 'checkpoints' / arm).glob(f'*/step{step}/adapter_model.safetensors'))
    if not files:
        raise RuntimeError(f'Missing {arm} step{step}')
    return files[-1]


def start_service(name):
    status = subprocess.run(['supervisorctl', 'status', name], capture_output=True, text=True)
    if 'RUNNING' not in status.stdout:
        subprocess.run(['supervisorctl', 'start', name], check=True)


def main():
    state = json.loads(STATE.read_text()) if STATE.exists() else dict(status='running', milestones=[])
    state['status'] = 'running'
    atomic(STATE, state)
    wait_job(QUEUE / '010_A_probe.job.json', state)
    a = adapter_for('A_native_probe', 500)
    note(f'A/nativo500 completo: {a}. Iniciando B do zero com a mesma receita/dados comuns.')
    available = shutil.disk_usage(ROOT).free
    cache = ROOT / 'data/target/cache/krea2_native'
    cache_size = sum(path.stat().st_size for path in cache.rglob('*') if path.is_file())
    if available < cache_size + 10*1024**3:
        # Reproducible embeddings, not weights/data/results. Keep the common
        # data and native recipe so they can be regenerated if A wins.
        shutil.rmtree(cache)
        note(f'Cache reproduzível A removido após500 para caber cache B; '
             f'{cache_size/1024**3:.2f}GiB liberados. Adapters/dados/resultados preservados.')
    config = ROOT / 'artifacts/configs/B_beta1_fixed_probe.toml'
    job = enqueue('020_B_probe', ['/venv/main/bin/deepspeed', '--num_gpus=1', '--master_port=29601',
        'train.py', '--deepspeed', '--config', str(config)],
        train_output=ROOT / 'checkpoints/B_beta1_fixed_probe')
    wait_job(job, state)
    note(f'B/beta1_fixed500 completo: {adapter_for("B_beta1_fixed_probe",500)}. '
         'Avaliando250/500,13heldout,seed76,Turbo/Raw,certa/trocada.')
    start_service('k2ab_stock')
    try:
        for step in (250, 500):
            job = enqueue(f'030_A{step}_eval', [PYTHON, 'tools/k2ab_eval_stock.py', '--adapter',
                str(adapter_for('A_native_probe', step)), '--out',
                str(ROOT / 'artifacts/eval' / f'A_native_step{step}')])
            wait_job(job, state)
        job = enqueue('040_T2I_base', [PYTHON, 'tools/k2ab_eval_stock.py', '--t2i-base', '--out',
            str(ROOT / 'artifacts/eval/T2I_base')])
        wait_job(job, state)
    finally:
        subprocess.run(['supervisorctl', 'stop', 'k2ab_stock'], check=True)
    for step in (250, 500):
        for variant in ('Turbo', 'Raw'):
            job = enqueue(f'050_B{step}_{variant}_eval', [PYTHON, 'tools/k2ab_eval_runner.py',
                '--config', str(config), '--adapter', str(adapter_for('B_beta1_fixed_probe', step).parent),
                '--out', str(ROOT / 'artifacts/eval' / f'B_beta1_fixed_step{step}'), '--variant', variant])
            wait_job(job, state)
    start_service('k2ab_legacy')
    try:
        wait_job(enqueue('055_beta1_original', [PYTHON, 'tools/k2ab_eval_legacy.py', '--out',
            str(ROOT / 'artifacts/eval/beta1_original_step13250')]), state)
    finally:
        subprocess.run(['supervisorctl', 'stop', 'k2ab_legacy'], check=True)
    command = [PYTHON, 'tools/k2ab_metrics.py', '--pairs', str(ROOT / 'heldout')]
    for arm in ('A_native', 'B_beta1_fixed'):
        for step in (250, 500):
            for variant in ('Turbo', 'Raw'):
                command += ['--dir', str(ROOT / 'artifacts/eval' / f'{arm}_step{step}' / variant)]
    for variant in ('Turbo', 'Raw'):
        command += ['--dir', str(ROOT / 'artifacts/eval/beta1_original_step13250' / variant)]
    wait_job(enqueue('060_metrics', command), state)
    state.update(status='awaiting_visual_review', current_job=None)
    atomic(STATE, state)
    note('A/B500 e avaliações250/500 concluídos; métricas/grids em /workspace/k2ab/artifacts/eval. '
         'Nenhum vencedor ou braço C decidido automaticamente; revisão visual pendente.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        state = json.loads(STATE.read_text()) if STATE.exists() else {}
        state.update(status='failed', error=str(error))
        atomic(STATE, state)
        note(f'Campanha parou em erro: {error}; etapas dependentes não executadas.')
        raise
