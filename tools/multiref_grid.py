#!/usr/bin/env python3
"""Grid do teste de ordem multi-referência do Krea 2.

Uma linha por exemplo: ref 1 | ref 2 | alvo real | ordem CORRETA | ordem
TROCADA — as duas últimas com a MESMA seed e o MESMO prompt, mudando só a
ordem em que as referências entram.

Se as duas últimas colunas forem iguais, o modelo não endereça: ele mistura
as referências e '<image 1>' não significa nada para ele. É o teste de
referência embaralhada da bateria do Anima adaptado — lá se trocava a
referência por outra, aqui se troca a ORDEM entre duas válidas.

Uso: multiref_grid.py <dir_das_saidas> <dir_do_dataset> [rotulo]
"""
import sys
import os
import textwrap

from PIL import Image, ImageDraw

out_dir = sys.argv[1]
ds_dir = sys.argv[2]
label = sys.argv[3] if len(sys.argv) > 3 else 'val50 (50 steps)'

W, H = 460, 460
LH = 18          # rótulo da coluna
CAP = 52         # faixa da caption embaixo de cada linha
HEAD = 78
COLS = ['ref 1  (<image 1>)', 'ref 2  (<image 2>)', 'alvo real do dataset',
        'GERADO — ordem correta', 'GERADO — ordem TROCADA']

ANIME = {
    'ANIME': dict(
        refs=['/workspace/dataset_raw/extracted/input_A/imagem000180.jpg',
              '/workspace/dataset_raw/extracted/input_A/imagem000409.jpg'],
        target=None,
        caption='[FORA DA DISTRIBUICAO — anime do dataset do Anima] Generate an image '
                'featuring the character from <image 1> standing next to the small '
                'creature from <image 2> in an outdoor scene.',
    ),
}


def rows():
    found = sorted({
        f.split('_')[1] for f in os.listdir(out_dir)
        if f.startswith('val50_') and f.endswith('_A_correta.png')
    })
    for ex in found:
        caption_file = f'{ds_dir}/target/{ex}.txt'
        caption = open(caption_file).read().strip() if os.path.exists(caption_file) else ''
        yield (ex, [f'{ds_dir}/refs/{ex}_1.jpg', f'{ds_dir}/refs/{ex}_2.jpg'],
               f'{ds_dir}/target/{ex}.jpg', caption,
               f'{out_dir}/val50_{ex}_A_correta.png', f'{out_dir}/val50_{ex}_B_trocada.png')
    for name, spec in ANIME.items():
        a = f'{out_dir}/{name}_A_correta.png'
        if os.path.exists(a):
            yield (name, spec['refs'], spec['target'], spec['caption'],
                   a, f'{out_dir}/{name}_B_trocada.png')


data = list(rows())
if not data:
    raise SystemExit(f'nenhuma saida em {out_dir}')

ROW = LH + H + CAP
grid = Image.new('RGB', (W * len(COLS), ROW * len(data) + HEAD), 'white')
d = ImageDraw.Draw(grid)
d.text((12, 8), 'KREA 2 MULTI-REFERENCIA  —  teste de ordem das referencias', fill='black')
d.text((12, 26), f'adapter: {label}   |   seed 76 fixa   |   28 steps   |   512x512   |   '
                 f'caption verbatim do Macro (com <image N>), blocos de visao rotulados "Picture N:"',
       fill=(80, 80, 80))
d.text((12, 44), 'As duas ultimas colunas so diferem na ORDEM em que as referencias entram. '
                 'Se forem IGUAIS, o modelo nao endereca — mistura.', fill=(150, 40, 40))
d.text((12, 60), 'ATENCAO: exemplos do Macro estao no conjunto de TREINO (50 amostras, 4 epocas). '
                 'A linha ANIME e a unica fora da distribuicao.', fill=(120, 120, 120))
d.line([(0, HEAD - 4), (W * len(COLS), HEAD - 4)], fill=(170, 170, 170))

for row, (ex, refs, target, caption, gen_a, gen_b) in enumerate(data):
    y = HEAD + row * ROW
    paths = [refs[0], refs[1] if len(refs) > 1 else None, target, gen_a, gen_b]
    for cix, p in enumerate(paths):
        d.text((cix * W + 5, y + 2), f'{ex}  -  {COLS[cix]}', fill='black')
        box = [cix * W + 1, y + LH, (cix + 1) * W - 2, y + LH + H - 1]
        if p is None or not os.path.exists(p):
            d.rectangle(box, outline=(200, 200, 200))
            d.text((cix * W + 10, y + LH + H // 2), 'n/a', fill=(170, 170, 170))
            continue
        img = Image.open(p).convert('RGB')
        img.thumbnail((W - 4, H - 4))
        grid.paste(img, (cix * W + (W - img.width) // 2, y + LH + (H - img.height) // 2))
    for i, line in enumerate(textwrap.wrap(caption, 210)[:3]):
        d.text((6, y + LH + H + 3 + i * 13), line, fill=(60, 60, 60))
    d.line([(0, y + ROW - 2), (W * len(COLS), y + ROW - 2)], fill=(210, 210, 210))

path = f'{out_dir}/GRID_teste_ordem.png'
grid.save(path)
print(f'saved {path}  ({grid.width}x{grid.height})')

try:
    import numpy as np

    print(f"\n{'exemplo':<12} {'dist(correta, trocada)':>24}   0 = identicas = NAO endereca")
    for ex, _, _, _, a, b in data:
        if not (os.path.exists(a) and os.path.exists(b)):
            continue
        ia = np.asarray(Image.open(a).convert('L').resize((64, 64)), dtype=np.float32) / 255
        ib = np.asarray(Image.open(b).convert('L').resize((64, 64)), dtype=np.float32) / 255
        print(f'{ex:<12} {float(np.abs(ia - ib).mean()):>24.4f}')
except ImportError:
    pass
