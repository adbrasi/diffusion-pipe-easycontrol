#!/usr/bin/env python3
"""Durable serial E2 training/evaluation campaign, managed by nextscene_worker.

Train both one-epoch arms before sampling. Recover interrupted training from
that arm's latest DeepSpeed state; never resume an E1 adapter. A failed stage
stops the campaign instead of executing dependent stages.
"""
import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import toml

ROOT = Path('/workspace/diffusion-pipe-easycontrol')
ART = Path('/workspace/nextscene_artifacts/E2')
STATE = ART / 'campaign_state.json'
LOG = ROOT / 'docs/NEXTSCENE_RUN_LOG.md'
PYTHON = '/venv/main/bin/python'
RATE = 0.62
ACCOUNTING_START = dt.datetime(2026, 9, 30, 6, 5, tzinfo=dt.timezone.utc)


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def note(message, push=True):
    now = dt.datetime.now(dt.timezone.utc)
    elapsed = (now - ACCOUNTING_START).total_seconds() / 3600
    with LOG.open('a') as f:
        f.write(f'\n### E2 — {now:%Y-%m-%d %H:%M UTC}\n\n{message}\n\n'
                f'Custo estimado acumulado desde06:05 UTC: US${elapsed * RATE:.2f} '
                f'({elapsed:.2f}h × US${RATE}/h; inclui setup, cache e ociosidade).\n')
    print(message, flush=True)
    if push:
        env = os.environ.copy()
        env.update(GIT_ASKPASS='/workspace/nextscene_ops/git-askpass.sh', GIT_TERMINAL_PROMPT='0')
        try:
            subprocess.run(['git', 'commit', '--only', '-m', 'docs: record E2 GPU campaign milestone',
                            'docs/NEXTSCENE_RUN_LOG.md'], cwd=ROOT, env=env, check=True, timeout=30)
            subprocess.run(['git', 'push', 'origin', 'claude/elegant-ptolemy-8kywqy'],
                           cwd=ROOT, env=env, check=True, timeout=60)
        except (subprocess.SubprocessError, OSError) as exc:
            # A transient Git outage must not kill an active trainer.
            print(f'Git sync pending: {exc}', flush=True)


def stage(name, argv, state, train_output=None):
    old = state['stages'].get(name, {})
    if old.get('status') == 'done':
        return
    log = ART / f'{name}.log'
    if train_output:
        # Recover the small gap between successful trainer exit and recording
        # completion if the service was restarted in that interval.
        completed = list(Path(train_output).glob('*/epoch1/adapter_model.safetensors'))
        if completed and log.exists() and 'TRAINING COMPLETE!' in log.read_text():
            state['stages'][name] = dict(old, status='done', recovered=True)
            atomic_json(STATE, state)
            return
        latest = sorted(Path(train_output).glob('*/latest'))
        if latest:
            argv = argv + ['--resume_from_checkpoint', latest[-1].parent.name]
    started = time.time()
    state['stages'][name] = dict(status='running', start=started, argv=argv)
    atomic_json(STATE, state)
    note(f'Iniciando `{name}`. Log local: `{log}`.')
    milestones = set(old.get('milestones', []))
    with log.open('a') as f:
        f.write(f'\nStage {name} {dt.datetime.now(dt.timezone.utc).isoformat()}\n')
        f.flush()
        proc = subprocess.Popen(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        state['stages'][name]['pid'] = proc.pid
        atomic_json(STATE, state)
        while proc.poll() is None:
            time.sleep(10)
            if train_output:
                ready = sorted(Path(train_output).glob('*/step*/adapter_model.safetensors'))
                for adapter in ready:
                    step = int(adapter.parent.name.removeprefix('step'))
                    if step not in milestones:
                        milestones.add(step)
                        state['stages'][name]['milestones'] = sorted(milestones)
                        atomic_json(STATE, state)
                        note(f'`{name}` salvou step{step} ({step * 4:,} amostras vistas). '
                             f'Adapter: `{adapter.parent}`. Treino segue até o fim da época.')
        rc = proc.returncode
    text = log.read_text()
    complete = not train_output or 'TRAINING COMPLETE!' in text
    entry = state['stages'][name]
    entry.update(status='done' if rc == 0 and complete else 'failed',
                 end=time.time(), seconds=time.time() - started, returncode=rc)
    if train_output:
        steps = re.findall(r'^steps: (\d+) loss:', text, re.M)
        entry['final_step'] = int(steps[-1]) if steps else None
        schedule = re.findall(r'Training schedule: (\d+) steps/epoch', text)
        entry['steps_per_epoch'] = int(schedule[-1]) if schedule else None
        times = [float(t) for t in re.findall(r'iter time \(s\): ([\d.]+)', text)]
        entry['logged_compute_seconds'] = sum(times)
    atomic_json(STATE, state)
    if entry['status'] != 'done':
        note(f'Falha em `{name}` (rc={rc}, treino completo={complete}). '
             'A campanha parou; etapas dependentes não foram executadas.')
        raise RuntimeError(f'{name} failed; inspect {log}')
    note(f'Concluído `{name}` em {entry["seconds"] / 60:.1f}min. '
         + (f'Último step: {entry["final_step"]}; época completa.' if train_output else ''))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', nargs='+', type=int, default=[76, 142])
    args = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    configs = {arm: ROOT / f'examples/anima_nextscene/gpu_20260930/E2_{arm}.toml'
               for arm in ['A', 'B']}
    parsed = {arm: toml.load(path) for arm, path in configs.items()}
    for arm in ['A', 'B']:
        c = parsed[arm]
        assert c['epochs'] == 1 and 'max_steps' not in c
        assert c['nextscene']['diff_weight'] is False and c['optimizer']['lr'] == 1e-4
        assert c['activation_checkpointing'] is True
        assert c['nextscene']['ref_dropout'] == 0.1 and c['nextscene']['high_noise_prob'] == 0.2
    a, b = json.loads(json.dumps(parsed['A'])), json.loads(json.dumps(parsed['B']))
    for c in [a, b]:
        c.pop('output_dir')
        c['nextscene'].pop('rope_layout')
    assert a == b, 'Arms differ beyond output_dir and rope_layout'
    state = json.loads(STATE.read_text()) if STATE.exists() else dict(stages={})
    state.update(status='running', seeds=args.seeds,
                 source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    atomic_json(STATE, state)
    shutil.copytree(ROOT / 'examples/anima_nextscene/gpu_20260930', ART / 'configs', dirs_exist_ok=True)
    # Keep the GPU doing training across both arms; sampling starts afterward.
    for arm in ['A', 'B']:
        stage(f'train_{arm}_epoch1', ['/venv/main/bin/deepspeed', '--num_gpus=1', 'train.py',
                                    '--deepspeed', '--config', str(configs[arm])], state,
              train_output=parsed[arm]['output_dir'])
    summaries = []
    for seed in args.seeds:
        for arm, layout in [('A', 'aligned'), ('B', 'disjoint_w')]:
            c = parsed[arm]
            runs = sorted(Path(c['output_dir']).glob('*/epoch1'))
            assert runs, f'{arm}: missing completed epoch1'
            run = runs[-1].parent
            for checkpoint in ['step1000', 'step2000', 'step3000', 'step5000', 'epoch1']:
                ckpt = run / checkpoint
                assert (ckpt / 'adapter_model.safetensors').exists(), ckpt
                out = ART / f'eval_seed{seed}' / f'{arm}{checkpoint.removeprefix("step")}'
                argv = [PYTHON, 'tools/nextscene_eval.py', '--dit', c['model']['transformer_path'],
                        '--vae', c['model']['vae_path'], '--llm', c['model']['llm_path'],
                        '--pairs', '/workspace/heldout_short', '--ckpt', str(ckpt), '--out', str(out),
                        '--limit', '24', '--steps', '20', '--width', '512', '--height', '512',
                        '--match-target-ar', '--seed', str(seed), '--ref_cfg', '1.0']
                stage(f'eval_{arm}_{checkpoint}_seed{seed}', argv, state)
                paths = list(out.glob('*/metrics.json'))
                assert len(paths) == 1
                metrics = json.loads(paths[0].read_text())
                summaries.append(dict(layout=layout, checkpoint=checkpoint, seed=seed,
                                      **metrics['summary']))
                with (ART / 'summary_all_seeds.csv').open('w') as f:
                    writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
                    writer.writeheader()
                    writer.writerows(summaries)
    state['status'] = 'complete'
    state['completed_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_json(STATE, state)
    note('E2 concluído: uma época em cada braço; checkpoints1000/2000/3000/5000 e '
         'epoch1 avaliados em24pares, seeds76/142, com buckets de aspecto. '
         'Métricas em `/workspace/nextscene_artifacts/E2/summary_all_seeds.csv`; '
         'outputs e grids sob `E2/eval_seed*/`. A escolha visual de checkpoint '
         'continua necessária; a campanha não declara identidade resolvida automaticamente.')


if __name__ == '__main__':
    main()
