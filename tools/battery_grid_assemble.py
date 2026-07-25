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

# Configuração de cada braço, para estampar DENTRO do grid. Sem isso o grid
# solto (fora da pasta) vira informação órfã.
ARM_CONFIG = {
    'arm1_broad_llm_frozen': ('IC-LoRA v3 escopo largo',
        'llm_adapter CONGELADO · dropout 0.0 · adaln fora · target-first · rank 32'),
    'arm3_broad_llm_full': ('IC-LoRA v3 escopo largo',
        'llm_adapter TREINAVEL lr=1e-4 · dropout 0.0 · adaln fora · target-first · rank 32'),
    'armA_adaln_in': ('IC-LoRA v3 escopo largo',
        'llm_adapter congelado · dropout 0.0 · adaln DENTRO (skip_adaln na inferencia) · target-first · rank 32'),
    'armB_dropout': ('IC-LoRA v3 escopo largo',
        'llm_adapter congelado · dropout 0.1 · adaln fora · target-first · rank 32'),
    'armC_routed': ('IC-LoRA DUAL (routing condition-only)',
        'llm_adapter congelado · dropout 0.1 · adaln fora · aparencia routada na ref · rank 32'),
    'arm1_broad_llm_frozen_s500': ('IC-LoRA v3 escopo largo (pico historico do arm1)',
        'llm_adapter CONGELADO · dropout 0.0 · adaln fora · target-first · rank 32 · parado em 500 steps'),
    'armD_dropout_2000': ('IC-LoRA v3 escopo largo',
        'llm_adapter congelado · dropout 0.1 · adaln fora · target-first · rank 32 · treino 2000 steps'),
}


def arm_from_outdir(out_dir):
    """Nome do braço a partir da pasta. As pastas `_eval10_<arm>` são o mesmo
    braço avaliado no conjunto de 10 exemplos — mapeiam para a mesma config."""
    import os as _os
    name = _os.path.basename(_os.path.normpath(out_dir))
    if name.startswith('_eval10_'):
        name = name[len('_eval10_'):]
    return name

DS = '/workspace/dataset_raw/extracted'
OUTS = '/workspace/outputs'

EXAMPLES = {
    'ex1':  dict(ref=f'{DS}/input_A/imagem000180.jpg', target=f'{DS}/input_B/imagem000180.jpg', shuffled_from='ex2'),
    'ex2':  dict(ref=f'{OUTS}/image1.webp', target=None, shuffled_from='ex3'),
    'ex3':  dict(ref=f'{OUTS}/image2.png', target=None, shuffled_from='ex4'),
    'ex4':  dict(ref=f'{DS}/input_A/imagem000297.jpg', target=None, shuffled_from='ex5'),
    'ex5':  dict(ref=f'{DS}/input_A/imagem001395.jpg', target=None, shuffled_from='ex6'),
    'ex6':  dict(ref=f'{DS}/input_A/imagem001549.jpg', target=None, shuffled_from='ex7'),
    'ex7':  dict(ref=f'{DS}/input_A/imagem000409.jpg', target=None, shuffled_from='ex8'),
    'ex8':  dict(ref=f'{DS}/input_A/imagem000878.jpg', target=None, shuffled_from='ex9'),
    'ex9':  dict(ref=f'{DS}/input_A/imagem000105.jpg', target=None, shuffled_from='ex10'),
    'ex10': dict(ref=f'{DS}/input_A/imagem001063.jpg', target=None, shuffled_from='ex1'),
}
COLS = ['referencia', 'alvo real', 'sem ref', 'lora 1.0 (CRITERIO)', 'lora 1.0 + ref_cfg 1.75', 'ref EMBARALHADA']

W, H, LH = 460, 258, 22
HEAD = 56  # faixa de identificacao no topo
_arm = arm_from_outdir(out_dir)
_title, _cfg = ARM_CONFIG.get(_arm, (_arm, ''))
_step = label.replace('_eval10', '').lstrip('s')
_set = 'conjunto de 10 exemplos held-out' if '_eval10' in label else 'conjunto de 3 exemplos'

grid = Image.new('RGB', (W * len(COLS), (H + LH) * len(EXAMPLES) + HEAD), 'white')
d = ImageDraw.Draw(grid)
# linha 1: qual braço + step (o que identifica o experimento)
d.text((10, 7), f'{_arm}   |   checkpoint step {_step}   |   {_set}   |   seed 76', fill='black')
# linha 2: qual método
d.text((10, 23), _title, fill=(70, 70, 70))
# linha 3: a configuração que difere dos outros braços
d.text((10, 38), _cfg, fill=(110, 110, 110))
# régua separando o cabeçalho das imagens
d.line([(0, HEAD - 3), (W * len(COLS), HEAD - 3)], fill=(180, 180, 180))

for row, (name, ex) in enumerate(EXAMPLES.items()):
    paths = [
        ex['ref'], ex['target'],
        f'{out_dir}/{label}_{name}_noref.png',
        f'{out_dir}/{label}_{name}_ref1.0.png',
        f'{out_dir}/{label}_{name}_refcfg1.75.png',
        f'{out_dir}/{label}_{name}_refshuffle.png',
    ]
    y = HEAD + row * (H + LH)
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
