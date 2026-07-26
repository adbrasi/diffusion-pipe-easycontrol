#!/usr/bin/env python3
"""Grid da avaliação held-out do multi-referência do Krea 2.

Colunas: ref 1 | ref 2 | ordem correta | ordem TROCADA.
Não há "alvo real": os prompts foram reescritos, então não existe imagem
alvo — o que se julga é se a saída obedece ao prompt E usa as referências
nos papéis certos.

Uso: macro_eval_grid.py <dir> [rotulo_do_step]
"""
import sys
import os
import textwrap

from PIL import Image, ImageDraw

out_dir = sys.argv[1]
step = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(os.path.normpath(out_dir))

A = '/workspace/dataset_raw/extracted/input_A'
PROMPTS = {
    'an1': 'the man with glasses from image 1 holding the small blue creature from image 2 in his arms',
    'an2': 'the woman from image 1 standing outdoors next to the lion from image 2',
    'an3': 'the small blue creature from image 1 sitting on a desk inside the room from image 2',
    'an4': 'the young man from image 1 walking alone through the street scene from image 2 at night',
    'an5': 'the character from image 1 standing in the hallway from image 2, seen from behind',
    'an6': 'the two characters from image 1 on the LEFT and the man with glasses from image 2 on the RIGHT',
}
REFS = {
    'an1': (f'{A}/imagem000297.jpg', f'{A}/imagem000409.jpg'),
    'an2': (f'{A}/imagem001395.jpg', f'{A}/imagem000878.jpg'),
    'an3': (f'{A}/imagem000409.jpg', f'{A}/imagem000105.jpg'),
    'an4': (f'{A}/imagem001549.jpg', f'{A}/imagem001063.jpg'),
    'an5': (f'{A}/imagem000180.jpg', f'{A}/imagem001395.jpg'),
    'an6': (f'{A}/imagem000105.jpg', f'{A}/imagem000297.jpg'),
}

COLS = ['ref 1  (image 1)', 'ref 2  (image 2)', 'GERADO']
W, H, LH, CAP, HEAD = 470, 470, 18, 34, 66

nomes = [n for n in PROMPTS if os.path.exists(f'{out_dir}/{n}_A_correta.png')]
if not nomes:
    raise SystemExit(f'nenhuma saida em {out_dir}')

ROW = LH + H + CAP
grid = Image.new('RGB', (W * len(COLS), ROW * len(nomes) + HEAD), 'white')
d = ImageDraw.Draw(grid)
d.text((12, 8), f'KREA 2 MULTI-REF — avaliacao HELD-OUT   |   checkpoint {step}', fill='black')
d.text((12, 26), 'Referencias do dataset do ANIMA — dominio totalmente FORA da distribuicao de '
                 'treino (o Macro e quase todo fotografico). Nenhuma vista em nenhum step.',
       fill=(80, 80, 80))
d.text((12, 44), 'seed 76 | 512x512 | prompts no registro das captions de treino '
                 '("Generate an image of the X from image 1 ... the Y from image 2 ...")',
       fill=(110, 110, 110))
d.line([(0, HEAD - 4), (W * len(COLS), HEAD - 4)], fill=(170, 170, 170))

for row, nome in enumerate(nomes):
    y = HEAD + row * ROW
    paths = [REFS[nome][0], REFS[nome][1], f'{out_dir}/{nome}_A_correta.png']
    for cix, p in enumerate(paths):
        d.text((cix * W + 5, y + 2), f'{nome}  -  {COLS[cix]}', fill='black')
        if not os.path.exists(p):
            d.rectangle([cix * W + 1, y + LH, (cix + 1) * W - 2, y + LH + H - 1], outline=(210, 210, 210))
            continue
        img = Image.open(p).convert('RGB')
        img.thumbnail((W - 4, H - 4))
        grid.paste(img, (cix * W + (W - img.width) // 2, y + LH + (H - img.height) // 2))
    for i, line in enumerate(textwrap.wrap(PROMPTS[nome], 190)[:2]):
        d.text((6, y + LH + H + 3 + i * 13), line, fill=(60, 60, 60))
    d.line([(0, y + ROW - 2), (W * len(COLS), y + ROW - 2)], fill=(215, 215, 215))

path = f'{out_dir}/GRID_{step}.png'
grid.save(path)
print(f'saved {path}  ({grid.width}x{grid.height})')
