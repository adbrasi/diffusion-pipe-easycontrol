"""Fresh scaled-FP8 512 A/B, serial training with Turbo evaluation every 250 steps."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import toml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import k2ab_campaign as ops

ROOT = ops.ROOT
ART = ROOT / 'artifacts/fp8_512'
MANIFEST = ART / 'heldout_manifest_quick.json'
MANIFEST_1024 = ART / 'heldout_manifest_quick_1024.json'
ops.STATE = ART / 'campaign_state.json'
RUN_SUFFIX = 'fp8_512'
JOB_PREFIX = ''
MICRO_BATCH = 4


def run_job(name, argv, state, cwd=ops.REPO, train_output=None):
    path = ops.enqueue(name, argv, cwd=cwd, train_output=train_output)
    ops.wait_job(path, state)


def stock_args(out, adapter=None, limit=2, manifest=None):
    args = [ops.PYTHON, 'tools/k2ab_eval_stock.py', '--out', str(out),
            '--manifest', str(manifest or MANIFEST), '--base-model', 'krea2_raw_fp8_scaled.safetensors',
            '--reference-pixels', 'target', '--limit', str(limit), '--variant', 'Turbo', '--adapter-only']
    return args + (['--adapter', str(adapter)] if adapter else ['--t2i-base'])


def main():
    state = json.loads(ops.STATE.read_text()) if ops.STATE.exists() else dict(milestones=[])
    # Terminal failure is actionable state, not permission to replay failed
    # jobs forever under supervisor's autorestart=unexpected policy.
    if state.get('status') in ('failed', 'stopped', 'awaiting_user_visual_verdict'):
        print(f'Campaign already {state["status"]}; explicit recovery required.', flush=True)
        return
    full_manifest = json.loads((ART / 'heldout_manifest.json').read_text())
    selected = ['03_ds4_imagem000268', '09_ds2_000248']
    by_stem = {row['stem']: row for row in full_manifest}
    review_rows = [by_stem[stem] for stem in selected]
    MANIFEST.write_text(json.dumps(review_rows, indent=2))
    high_rows = [dict(row, width=row['width'] * 2, height=row['height'] * 2) for row in review_rows]
    MANIFEST_1024.write_text(json.dumps(high_rows, indent=2))
    state.setdefault('reported_stages', [])
    state.update(status='running', recipe=f'fp8_scaled_512_micro{MICRO_BATCH}_accum1', fresh=True,
                 checkpoint_arms=[f'{arm}_{RUN_SUFFIX}_probe' for arm in ('A_native', 'B_beta1_fixed')],
                 sampling_variants=['Turbo'], sampling_steps=[250, 500, 750, 1000], target_steps=1000,
                 sampling_resolutions=[512, 1024],
                 sampling_cases=selected, images_per_checkpoint=4, extra_baselines=False, sampling_comparison='trained_adapter_only')
    ops.atomic(ops.STATE, state)
    for arm_index, arm in enumerate(('A_native', 'B_beta1_fixed')):
        output = ROOT / 'checkpoints' / f'{arm}_{RUN_SUFFIX}_probe'
        config = ART / 'configs' / f'{arm}_probe.toml'
        cwd = Path('/workspace/k2ab/native_worktree') if arm_index == 0 else ops.REPO
        for stage_index, step in enumerate((250, 500, 750, 1000)):
            # Keep existing queue names so an active 250-step job is reused.
            stage_id = 210 + (step // 250 - 1) * 20
            name = f'{JOB_PREFIX}{stage_id+arm_index*100}_{arm}_{step}'
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
            runs = sorted(output.glob('*/latest'))
            if stage_index or runs:
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
                         'Avaliando apenas Turbo antes do próximo segmento.')
            if arm_index == 1:
                # Reuse completed images, and load/fuse B's large base only
                # once for both requested resolutions at each checkpoint.
                out = ART / 'eval' / f'{arm}_step{step}'
                run_job(name+'_eval_Turbo_both_resolutions', [ops.PYTHON, 'tools/k2ab_eval_runner.py',
                    '--config', str(config), '--adapter', str(adapter.parent), '--out', str(out),
                    '--variant', 'Turbo', '--manifest', str(MANIFEST),
                    '--manifest-1024', str(MANIFEST_1024), '--limit', '2', '--adapter-only'], state)
            for resolution, manifest in ((512, MANIFEST), (1024, MANIFEST_1024)):
                # Preserve the existing 512 directories and completed queue jobs.
                out = ART / 'eval' / f'{arm}_step{step}'
                if resolution == 1024:
                    out = out / 'resolution_1024'
                suffix = '' if resolution == 512 else '_1024'
                if arm_index == 0:
                    ops.start_service('k2ab_stock')
                    try:
                        run_job(name+'_eval'+suffix, stock_args(out, adapter, manifest=manifest), state)
                    finally:
                        subprocess.run(['supervisorctl', 'stop', 'k2ab_stock'], check=True)
                metrics = [ops.PYTHON, 'tools/k2ab_metrics.py', '--pairs', str(ROOT / 'heldout'),
                           '--adapter-only', '--dir', str(out / 'Turbo')]
                run_job(name+'_metrics_adapter_only'+suffix, metrics, state)
    state.update(status='awaiting_user_visual_verdict', current_job=None)
    ops.atomic(ops.STATE, state)
    ops.note(f'A/B FP8/5121000/micro{MICRO_BATCH} completo:{1000*MICRO_BATCH}amostras por braço, '
             f'grids/metrics Turbo250/500/750/1000,512 e1024 em {ART}/eval. '
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
        MANIFEST = ART / 'heldout_manifest_quick.json'
        MANIFEST_1024 = ART / 'heldout_manifest_quick_1024.json'
        ops.STATE = ART / 'campaign_state.json'
    try:
        main()
    except SystemExit:
        # wait_job already records stopped. An intentional scheduling hold
        # must not trigger supervisor's unexpected-exit restart policy.
        sys.exit(0)
    except Exception as error:
        state = json.loads(ops.STATE.read_text()) if ops.STATE.exists() else {}
        state.update(status='failed', error=str(error))
        ops.atomic(ops.STATE, state)
        ops.note(f'Campanha FP8 parou em erro: {error}; investigar antes de seguir.')
        # The JSON records failure, while a normal process exit prevents
        # supervisor from repeatedly scheduling the already-failed job.
        sys.exit(0)
