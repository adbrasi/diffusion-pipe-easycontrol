#!/usr/bin/env python3
"""Monta o grid de avaliação de um checkpoint da bateria 2026-07-25.
Uso: battery_grid_assemble.py <label> <out_dir>
Colunas: referencia | alvo real | sem ref | ref forca 1.0 | ref forca 1.5
Linhas: ex1, ex2, ex3 (os 3 pares fixos)
"""
import sys
import os
from PIL import Image, ImageDraw

label, out_dir = sys.argv[1], sys.argv[2]
DS = '/workspace/dataset_raw/extracted'

EXAMPLES = {
    'ex1': dict(ref=f'{DS}/input_A/imagem000180.jpg', target=f'{DS}/input_B/imagem000180.jpg'),
    'ex2': dict(ref=f'{DS}/input_A/imagem001129.jpg', target=f'{DS}/input_B/imagem001129.jpg'),
    'ex3': dict(ref=f'{DS}/input_A/imagem001549.jpg', target=f'{DS}/input_B/imagem001549.jpg'),
}
COLS = ['referencia', 'alvo real', 'sem ref', 'ref forca 1.0', 'ref forca 1.5']

W, H, LH = 260, 146, 20
grid = Image.new('RGB', (W * len(COLS), (H + LH) * len(EXAMPLES) + 30), 'white')
d = ImageDraw.Draw(grid)
d.text((8, 6), f'{label}', fill='black')

for row, (name, ex) in enumerate(EXAMPLES.items()):
    paths = [
        ex['ref'], ex['target'],
        f'{out_dir}/{label}_{name}_noref.png',
        f'{out_dir}/{label}_{name}_ref1.0.png',
        f'{out_dir}/{label}_{name}_ref1.5.png',
    ]
    y = 30 + row * (H + LH)
    for cix, p in enumerate(paths):
        d.text((cix * W + 4, y + 2), f'{name} - {COLS[cix]}', fill='black')
        if not os.path.exists(p):
            d.rectangle([cix * W, y + LH, (cix + 1) * W - 1, y + LH + H - 1], outline='red')
            continue
        img = Image.open(p).convert('RGB')
        img.thumbnail((W, H))
        # centraliza
        ox = cix * W + (W - img.width) // 2
        oy = y + LH + (H - img.height) // 2
        grid.paste(img, (ox, oy))

out_path = f'{out_dir}/GRID_{label}.png'
grid.save(out_path)
print(f'saved {out_path}')
