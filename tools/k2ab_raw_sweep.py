"""Small controlled stock Raw settings comparison, with exact API workflows."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.k2ab_eval_stock import graph, execute, URL
from tools.krea2_sampling import build_krea2_timesteps

ROOT = Path('/workspace/k2ab')
PROFILES = [
    dict(name='Raw28_CFG1', steps=28, cfg=1.0, mu=None),
    dict(name='Raw28_CFG3', steps=28, cfg=3.0, mu=None),
    dict(name='Raw28_CFG4p5', steps=28, cfg=4.5, mu=None),
    dict(name='Raw52_CFG4p5', steps=52, cfg=4.5, mu=None),
    dict(name='Raw52_CFG4p5_mu1p15', steps=52, cfg=4.5, mu=1.15),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.manifest.read_text())[:args.limit]
    name = args.adapter.parent.parent.parent.name + '_' + args.adapter.parent.name + '.safetensors'
    link = Path('/workspace/models/krea2/loras') / name
    if not link.exists():
        link.symlink_to(args.adapter.resolve())
    subprocess.run(['supervisorctl', 'start', 'k2ab_stock'], capture_output=True)
    recorded = []
    try:
        for profile in PROFILES:
            out = args.out / profile['name']
            out.mkdir(exist_ok=True)
            for row in rows:
                dest = out / f'{row["stem"]}_true.png'
                sigmas, mu = build_krea2_timesteps(row['width']//16 * (row['height']//16),
                                                  profile['steps'], mu=profile['mu'])
                recorded.append(dict(stem=row['stem'], profile=profile, resolved_mu=mu,
                                     sigmas=sigmas, prompt=row['prompt'], seed=row['seed']))
                if dest.exists():
                    continue
                prompt = graph(row, row['reference'], 'Raw', name,
                               f'raw_sweep/{profile["name"]}/{dest.stem}',
                               base_model='krea2_raw_fp8_scaled.safetensors', reference_pixels='target',
                               steps=profile['steps'], cfg=profile['cfg'], mu=profile['mu'])
                dest.with_suffix('.json').write_text(json.dumps(prompt, indent=2))
                execute(prompt, dest)
                print('Saved', dest, flush=True)
    finally:
        requests.post(URL + '/free', json={'unload_models': True, 'free_memory': True}, timeout=30)
        subprocess.run(['supervisorctl', 'stop', 'k2ab_stock'], check=True)
    (args.out/'manifest.json').write_text(json.dumps(dict(adapter=str(args.adapter),
                                                       comparisons=recorded), indent=2))
    columns = [('A ref', ROOT/'heldout/control'), ('B alvo', ROOT/'heldout/target'),
               ('Turbo8 CFG1', args.baseline/'Turbo'), ('Raw28 CFG5.5 anterior', args.baseline/'Raw')]
    columns += [(p['name'], args.out/p['name']) for p in PROFILES]
    cell, height, header, label = 320, 180, 64, 52
    canvas = Image.new('RGB', (len(columns)*cell, header+len(rows)*(height+label)), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('DejaVuSans.ttf', 15)
    draw.text((8, 6), 'Krea2 A125 micro2 | same prompt/seed76/reference | Euler | CFG convention: ComfyUI', font=font, fill='black')
    for i,(title,_) in enumerate(columns):
        draw.text((i*cell+8, 35), title, font=font, fill='black')
    for r,row in enumerate(rows):
        y = header+r*(height+label)
        draw.text((8,y+3),row['stem']+' | '+row['prompt'],font=font,fill='black')
        for c,(_,folder) in enumerate(columns):
            name = row['reference'] if c < 2 else f'{row["stem"]}_true.png'
            tile = ImageOps.contain(Image.open(folder/name).convert('RGB'),(cell,height))
            canvas.paste(tile,(c*cell+(cell-tile.width)//2,y+label+(height-tile.height)//2))
    canvas.save(args.out/'grid.jpg', quality=95)
    print('Grid', args.out/'grid.jpg', flush=True)


if __name__ == '__main__':
    main()
