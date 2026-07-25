#!/usr/bin/env python3
"""Grid do teste de ordem multi-referência.

Uma linha por exemplo: ref 1 | ref 2 | alvo real | ordem CORRETA | ordem
TROCADA. As duas últimas colunas usam a mesma seed e o mesmo prompt — se
forem iguais, o modelo não endereça as referências.

Uso: multiref_grid.py <dir_das_saidas> <dir_do_dataset>
"""
import sys
import os
from PIL import Image, ImageDraw

out_dir, ds_dir = sys.argv[1], sys.argv[2]

W, H, LH = 420, 420, 20
HEAD = 52
COLS = ['ref 1 (<image 1>)', 'ref 2 (<image 2>)', 'alvo real',
        'ordem CORRETA', 'ordem TROCADA']

examples = sorted({
    f.split('_')[0] for f in os.listdir(out_dir) if f.endswith('_A_ordem_correta.png')
})
if not examples:
    raise SystemExit(f'nenhuma saida _A_ordem_correta.png em {out_dir}')

grid = Image.new('RGB', (W * len(COLS), (H + LH) * len(examples) + HEAD), 'white')
d = ImageDraw.Draw(grid)
d.text((10, 6), 'KREA 2 MULTI-REF — teste de ordem   |   mesma seed, mesmo prompt, '
                'so muda a ORDEM das referencias', fill='black')
d.text((10, 22), 'Se as duas ultimas colunas forem IGUAIS, o modelo nao endereca: '
                 'ele mistura as referencias e "<image 1>" nao significa nada.',
       fill=(90, 90, 90))
d.text((10, 37), f'adapter: {os.path.basename(os.path.normpath(out_dir))}', fill=(120, 120, 120))
d.line([(0, HEAD - 3), (W * len(COLS), HEAD - 3)], fill=(180, 180, 180))

for row, ex in enumerate(examples):
    paths = [
        f'{ds_dir}/refs/{ex}_1.jpg',
        f'{ds_dir}/refs/{ex}_2.jpg',
        f'{ds_dir}/target/{ex}.jpg',
        f'{out_dir}/{ex}_A_ordem_correta.png',
        f'{out_dir}/{ex}_B_ordem_trocada.png',
    ]
    y = HEAD + row * (H + LH)
    for cix, p in enumerate(paths):
        d.text((cix * W + 4, y + 2), f'{ex} - {COLS[cix]}', fill='black')
        if not os.path.exists(p):
            d.rectangle([cix * W, y + LH, (cix + 1) * W - 1, y + LH + H - 1], outline='red')
            continue
        img = Image.open(p).convert('RGB')
        img.thumbnail((W, H))
        grid.paste(img, (cix * W + (W - img.width) // 2, y + LH + (H - img.height) // 2))

path = f'{out_dir}/GRID_ordem.png'
grid.save(path)
print(f'saved {path}')

# distância entre as duas ordens — triagem, não critério
try:
    import numpy as np

    print(f"\n{'exemplo':<12} {'dist(A,B)':>10}   (0 = identicas = NAO endereca)")
    for ex in examples:
        a = f'{out_dir}/{ex}_A_ordem_correta.png'
        b = f'{out_dir}/{ex}_B_ordem_trocada.png'
        if not (os.path.exists(a) and os.path.exists(b)):
            continue
        ia = np.asarray(Image.open(a).convert('L').resize((64, 64)), dtype=np.float32) / 255
        ib = np.asarray(Image.open(b).convert('L').resize((64, 64)), dtype=np.float32) / 255
        print(f'{ex:<12} {float(np.abs(ia - ib).mean()):>10.4f}')
except ImportError:
    pass
