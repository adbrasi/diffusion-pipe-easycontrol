#!/usr/bin/env python3
"""Monta o grid de avaliação de um checkpoint da bateria 2026-07-25 (v2).
Uso: battery_grid_assemble.py <label> <out_dir>
Colunas: referencia | alvo real | sem ref | ref forca 1.0 | ref forca 1.5 | ref EMBARALHADA
Linhas: ex1 (Demon Slayer, dataset), ex2 (elf/dungeon, usuário),
        ex3 (floresta noturna, usuário — CUIDADO: caption descreve 2 garotas
        que não aparecem na imagem de referência, checar com o usuário se é
        intencional antes de tirar conclusões deste exemplo).
Thumbnail bem maior que a v1 (era 260x146, ilegível pra detalhe de rosto).
"""
import sys
import os
from PIL import Image, ImageDraw

label, out_dir = sys.argv[1], sys.argv[2]
DS = '/workspace/dataset_raw/extracted'
OUTS = '/workspace/outputs'

EXAMPLES = {
    'ex1': dict(ref=f'{DS}/input_A/imagem000180.jpg', target=f'{DS}/input_B/imagem000180.jpg', shuffled_from='ex2'),
    'ex2': dict(ref=f'{OUTS}/image1.webp', target=None, shuffled_from='ex3'),
    'ex3': dict(ref=f'{OUTS}/image2.png', target=None, shuffled_from='ex1'),
}
COLS = ['referencia', 'alvo real', 'sem ref', 'ref forca 1.0', 'ref forca 1.5', 'ref EMBARALHADA']

W, H, LH = 460, 258, 22
grid = Image.new('RGB', (W * len(COLS), (H + LH) * len(EXAMPLES) + 34), 'white')
d = ImageDraw.Draw(grid)
d.text((8, 8), f'{label}', fill='black')

for row, (name, ex) in enumerate(EXAMPLES.items()):
    paths = [
        ex['ref'], ex['target'],
        f'{out_dir}/{label}_{name}_noref.png',
        f'{out_dir}/{label}_{name}_ref1.0.png',
        f'{out_dir}/{label}_{name}_ref1.5.png',
        f'{out_dir}/{label}_{name}_refshuffle.png',
    ]
    y = 34 + row * (H + LH)
    for cix, p in enumerate(paths):
        label_text = f"{name} - {COLS[cix]}"
        if cix == 5:
            label_text += f" (ref de {ex['shuffled_from']})"
        d.text((cix * W + 4, y + 2), label_text, fill='black')
        if p is None or not os.path.exists(p):
            d.rectangle([cix * W, y + LH, (cix + 1) * W - 1, y + LH + H - 1], outline='red')
            if p is None:
                d.text((cix * W + 4, y + LH + H // 2), 'n/a (sem alvo real)', fill='red')
            continue
        img = Image.open(p).convert('RGB')
        img.thumbnail((W, H))
        ox = cix * W + (W - img.width) // 2
        oy = y + LH + (H - img.height) // 2
        grid.paste(img, (ox, oy))

out_path = f'{out_dir}/GRID_{label}.png'
grid.save(out_path)
print(f'saved {out_path}')
