"""Sample a fixed 6-example set (saga512_samples.json format) with a given
adapter checkpoint via infer_reference_adapter, one subprocess per sample.

Usage: arms_sampler.py --config <arm toml> --adapter <stepN dir> --out <dir>
       [--scale 1.0] [--samples /workspace/configs/saga512_samples.json]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

TURBO = '/workspace/models/krea2/loras/krea2_turbo_lora_rank_64_bf16.safetensors'
PY = '/venv/main/bin/python'
REPO = '/workspace/projects/diffusion-pipe-easycontrol'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scale', default='1.0')
    ap.add_argument('--seed', default='76')
    ap.add_argument('--samples', default='/workspace/configs/saga512_samples.json')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    for s in json.load(open(args.samples))['samples']:
        dest = out / f"{s['name'] if 'name' in s else Path(s['reference']).stem}_s{args.scale}.png"
        if dest.exists():
            ok += 1
            continue
        cmd = [PY, 'tools/infer_reference_adapter.py', '--config', args.config,
               '--adapter', args.adapter, '--reference', s['reference'],
               '--prompt', s['prompt'], '--width', str(s['width']), '--height', str(s['height']),
               '--seed', args.seed, '--adapter-scale', args.scale,
               '--turbo-lora', TURBO, '--output', str(dest)]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        if r.returncode == 0 and dest.exists():
            ok += 1
        else:
            fail += 1
            (out / f"{dest.stem}_error.log").write_text(r.stdout[-3000:] + '\n' + r.stderr[-3000:])
    print(f'[arms_sampler] {ok} ok, {fail} fail -> {out}')
    sys.exit(1 if fail else 0)


if __name__ == '__main__':
    main()
