"""Fresh scaled-FP8 512 A/B, serial training/evaluation at each saved adapter."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import toml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import k2ab_campaign as ops

ROOT = ops.ROOT
ART = ROOT / 'artifacts/fp8_512'
MANIFEST = ART / 'heldout_manifest.json'
ops.STATE = ART / 'campaign_state.json'
RUN_SUFFIX = 'fp8_512'
JOB_PREFIX = ''
MICRO_BATCH = 4


def run_job(name, argv, state, cwd=ops.REPO, train_output=None):
    path = ops.enqueue(name, argv, cwd=cwd, train_output=train_output)
    ops.wait_job(path, state)


def stock_args(out, adapter=None, limit=13):
    args = [ops.PYTHON, 'tools/k2ab_eval_stock.py', '--out', str(out),
            '--manifest', str(MANIFEST), '--base-model', 'krea2_raw_fp8_scaled.safetensors',
            '--reference-pixels', 'target', '--limit', str(limit)]
    return args + (['--adapter', str(adapter)] if adapter else ['--t2i-base'])


def main():
    state = json.loads(ops.STATE.read_text()) if ops.STATE.exists() else dict(milestones=[])
    # Terminal failure is actionable state, not permission to replay failed
    # jobs forever under supervisor's autorestart=unexpected policy.
    if state.get('status') in ('failed', 'awaiting_user_visual_verdict'):
        print(f'Campaign already {state["status"]}; explicit recovery required.', flush=True)
        return
    state.setdefault('reported_stages', [])
    state.update(status='running', recipe=f'fp8_scaled_512_micro{MICRO_BATCH}_accum1', fresh=True,
                 checkpoint_arms=[f'{arm}_{RUN_SUFFIX}_probe' for arm in ('A_native', 'B_beta1_fixed')])
    ops.atomic(ops.STATE, state)
    for arm_index, arm in enumerate(('A_native', 'B_beta1_fixed')):
        output = ROOT / 'checkpoints' / f'{arm}_{RUN_SUFFIX}_probe'
        config = ART / 'configs' / f'{arm}_probe.toml'
        cwd = Path('/workspace/k2ab/native_worktree') if arm_index == 0 else ops.REPO
        for stage_index, step in enumerate((125, 250, 375, 500)):
            name = f'{JOB_PREFIX}{200+arm_index*100+stage_index*10}_{arm}_{step}'
            recipe = toml.load(config)
            if recipe['micro_batch_size_per_gpu'] != MICRO_BATCH or recipe['gradient_accumulation_steps'] != 1:
                raise RuntimeError('Campaign/config batch mismatch')
            if recipe['output_dir'] != str(output):
                raise RuntimeError('Campaign/config output mismatch; refusing to mix old runs')
            recipe['max_steps'] = step
            stage_config = config.with_name(f'{arm}_through{step}.toml')
            toml.dump(recipe, stage_config.open('w'))
            args = ['/venv/main/bin/deepspeed', '--num_gpus=1', '--master_port=29601',
                    'train.py', '--deepspeed', '--config', str(stage_config)]
            if stage_index:
                runs = sorted(output.glob('*/latest'))
                if not runs:
                    raise RuntimeError(f'No recovery state for the new {arm} run')
                args += ['--resume_from_checkpoint', runs[-1].parent.name]
            run_job(name, args, state, cwd=cwd, train_output=output)
            adapters = sorted(output.glob(f'*/step{step}/adapter_model.safetensors'))
            if not adapters:
                raise RuntimeError(f'Missing {arm} step{step}')
            adapter = adapters[-1]
            state.update(arm=arm, step=step, samples_seen=step*recipe['micro_batch_size_per_gpu'])
            ops.atomic(ops.STATE, state)
            if name not in state['reported_stages']:
                state['reported_stages'].append(name)
                ops.atomic(ops.STATE, state)
                ops.note(f'Novo {arm}/FP8/512/micro{MICRO_BATCH} salvou step{step}, '
                         f'{state["samples_seen"]} amostras; nenhum peso do A74 utilizado. '
                         'Avaliando antes do próximo segmento.')
            out = ART / 'eval' / f'{arm}_step{step}'
            if arm_index == 0:
                ops.start_service('k2ab_stock')
                try:
                    run_job(name+'_eval', stock_args(out, adapter), state)
                finally:
                    subprocess.run(['supervisorctl', 'stop', 'k2ab_stock'], check=True)
            else:
                for variant in ('Turbo', 'Raw'):
                    run_job(name+'_eval_'+variant, [ops.PYTHON, 'tools/k2ab_eval_runner.py',
                        '--config', str(config), '--adapter', str(adapter.parent), '--out', str(out),
                        '--variant', variant, '--manifest', str(MANIFEST)], state)
            metrics = [ops.PYTHON, 'tools/k2ab_metrics.py', '--pairs', str(ROOT / 'heldout')]
            for variant in ('Turbo', 'Raw'):
                metrics += ['--dir', str(out / variant)]
            run_job(name+'_metrics', metrics, state)
            if arm_index == 0 and step == 125:
                # Smoke12 showed a checkerboard in the native zero-timestep
                # contract, including stock without an adapter. Require our
                # visual inspection of the first trained save before spending
                # on the rest of A. This is an operator gate, not a user
                # permission request; the user's A/B authorization stands.
                gate = ART / 'A125_sampling_review.ok'
                state.update(status='awaiting_A125_sampling_review')
                ops.atomic(ops.STATE, state)
                while not gate.exists():
                    if (ROOT / 'ops/stop_campaign').exists():
                        raise SystemExit('Campaign stopped at the sampling gate')
                    time.sleep(10)
                state.update(status='running')
                ops.atomic(ops.STATE, state)
    ops.start_service('k2ab_stock')
    try:
        run_job(JOB_PREFIX+'410_fp8_T2I_base', stock_args(ART / 'eval/T2I_base'), state)
    finally:
        subprocess.run(['supervisorctl', 'stop', 'k2ab_stock'], check=True)
    ops.start_service('k2ab_legacy')
    try:
        run_job(JOB_PREFIX+'420_beta1_original_512', [ops.PYTHON, 'tools/k2ab_eval_legacy.py',
            '--out', str(ART / 'eval/beta1_original_step13250'), '--manifest', str(MANIFEST)], state)
    finally:
        subprocess.run(['supervisorctl', 'stop', 'k2ab_legacy'], check=True)
    metrics = [ops.PYTHON, 'tools/k2ab_metrics.py', '--pairs', str(ROOT / 'heldout')]
    for variant in ('Turbo', 'Raw'):
        metrics += ['--dir', str(ART / 'eval/beta1_original_step13250' / variant)]
    run_job(JOB_PREFIX+'430_beta1_original_metrics', metrics, state)
    state.update(status='awaiting_user_visual_verdict', current_job=None)
    ops.atomic(ops.STATE, state)
    ops.note(f'Novo A/B FP8/512500/micro{MICRO_BATCH} completo:{500*MICRO_BATCH}amostras por braço, '
             f'grids/metrics125/250/375/500 em {ART}/eval. '
             'Decisão visual do usuário pendente; sem iniciar variante target ou W8A8 automaticamente.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--micro-batch', type=int, choices=(2, 4), default=4)
    cli = parser.parse_args()
    MICRO_BATCH = cli.micro_batch
    if MICRO_BATCH == 2:
        RUN_SUFFIX = 'fp8_512_micro2'
        JOB_PREFIX = 'm2_'
        ART = ROOT / 'artifacts' / RUN_SUFFIX
        MANIFEST = ART / 'heldout_manifest.json'
        ops.STATE = ART / 'campaign_state.json'
    try:
        main()
    except Exception as error:
        state = json.loads(ops.STATE.read_text()) if ops.STATE.exists() else {}
        state.update(status='failed', error=str(error))
        ops.atomic(ops.STATE, state)
        ops.note(f'Campanha FP8 parou em erro: {error}; investigar antes de seguir.')
        # The JSON records failure, while a normal process exit prevents
        # supervisor from repeatedly scheduling the already-failed job.
        sys.exit(0)
