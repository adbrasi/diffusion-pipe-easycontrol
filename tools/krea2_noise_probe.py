#!/usr/bin/env python3
"""Controlled Turbo-only visual diagnosis; never starts training."""
import argparse
import json
from pathlib import Path
import sys

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.k2ab_eval_stock import graph, execute, URL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--mask-adapter', type=Path, required=True)
    parser.add_argument('--turbo-adapter', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=16)
    args = parser.parse_args()
    row = json.loads(args.manifest.read_text())[0]
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        for variant, path in [('base', None), ('mask_only', args.mask_adapter), ('mask_turbo', args.turbo_adapter)]:
            name = None
            if path:
                name = path.parent.parent.parent.name + '_' + path.parent.name + '.safetensors'
                link = Path('/workspace/models/krea2/loras') / name
                if not link.exists():
                    link.symlink_to(path.resolve())
                elif link.resolve() != path.resolve():
                    raise RuntimeError(f'Adapter name collision: {link}')
            dest = args.out / f'{variant}_turbo{args.steps}.png'
            if dest.exists():
                continue
            prompt = graph(row, row['reference'], 'Turbo', name, f'noise_probe/{dest.stem}',
                           base_model='krea2_raw_fp8_scaled.safetensors', reference_pixels='target',
                           steps=args.steps, cfg=1., mu=1.15)
            dest.with_suffix('.json').write_text(json.dumps(prompt, indent=2))
            execute(prompt, dest)
            print(f'Saved {dest}', flush=True)
    finally:
        requests.post(URL + '/free', json={'unload_models': True, 'free_memory': True}, timeout=30)


if __name__ == '__main__':
    main()
