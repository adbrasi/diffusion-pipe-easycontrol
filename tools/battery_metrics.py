#!/usr/bin/env python3
"""Métricas objetivas para a bateria Anima — para não julgar só "no olho".

O critério central da bateria é: o adapter REALMENTE usa a referência, ou
está satisfazendo a loss só com o caption? Duas métricas complementares:

1. SENSIBILIDADE (ref_sensitivity): distância entre a geração com a
   referência CORRETA e com a referência EMBARALHADA (mesmo caption, mesma
   seed). MAIOR = melhor — significa que trocar a referência muda a saída,
   ou seja, o modelo está lendo a referência.
   Sozinha é enganosa: ruído/instabilidade também produz distância grande.

2. FIDELIDADE (ref_fidelity): quão mais parecida com a REFERÊNCIA a
   geração-com-ref-correta é, comparada à geração-sem-ref.

   *** NÃO USAR COMO CRITÉRIO DE DECISÃO. *** Medido em 2026-07-25: como
   ela compara paleta+estrutura contra a referência, gerações ESCURAS e
   DEGRADADAS pontuam alto quando a referência é escura. No armD ela deu
   pico (0.226) exatamente no checkpoint com artefato de painel duplicado e
   imagens ilegíveis, e valor menor (0.156) no checkpoint visualmente bom.
   Ela recompensa o defeito. Serve no máximo como triagem grosseira.

Um bom adapter tem AS DUAS altas. Sensibilidade alta com fidelidade ~0 é
instabilidade, não uso de referência.

Ambas usam duas visões da imagem, combinadas:
- paleta: histograma de cor no espaço HSV (o que "clima/iluminação" captura)
- estrutura: imagem reduzida a 32x32 em luminância (composição grosseira)

Uso: battery_metrics.py <dir_do_braco> [<dir2> ...]
     (cada dir deve conter os PNGs sNNN_exN_{noref,ref1.0,refshuffle}.png)
"""
import sys
import os
import glob
import numpy as np
from PIL import Image

DS = '/workspace/dataset_raw/extracted'
OUTS = '/workspace/outputs'

# referência real de cada exemplo (a imagem que foi dada como condição)
REF_IMAGE = {
    'ex1': f'{DS}/input_A/imagem000180.jpg',
    'ex2': f'{OUTS}/image1.webp',
    'ex3': f'{OUTS}/image2.png',
}


def _load(path, size=(256, 256)):
    return Image.open(path).convert('RGB').resize(size, Image.BILINEAR)


def palette_vec(img):
    """Histograma HSV normalizado — captura paleta/clima/iluminação."""
    hsv = np.asarray(img.convert('HSV'), dtype=np.float32)
    h = np.histogram(hsv[..., 0], bins=24, range=(0, 256), density=True)[0]
    s = np.histogram(hsv[..., 1], bins=16, range=(0, 256), density=True)[0]
    v = np.histogram(hsv[..., 2], bins=16, range=(0, 256), density=True)[0]
    return np.concatenate([h, s, v])


def struct_vec(img):
    """Luminância 32x32 normalizada — composição grosseira."""
    g = np.asarray(img.convert('L').resize((32, 32), Image.BILINEAR), dtype=np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    return g.ravel()


def dist(a_img, b_img):
    """Distância combinada paleta+estrutura, escala ~0-1 (empírica)."""
    dp = np.linalg.norm(palette_vec(a_img) - palette_vec(b_img))
    ds = np.linalg.norm(struct_vec(a_img) - struct_vec(b_img)) / 32.0
    return float(dp + ds)


def analyse(arm_dir):
    steps = sorted({os.path.basename(p).split('_')[0]
                    for p in glob.glob(f'{arm_dir}/s*_ex*_ref1.0.png')},
                   key=lambda s: int(s[1:]))
    rows = []
    for step in steps:
        sens, fid = [], []
        for ex in ('ex1', 'ex2', 'ex3'):
            p_ok = f'{arm_dir}/{step}_{ex}_ref1.0.png'
            p_sh = f'{arm_dir}/{step}_{ex}_refshuffle.png'
            p_no = f'{arm_dir}/{step}_{ex}_noref.png'
            if not (os.path.exists(p_ok) and os.path.exists(p_sh)):
                continue
            g_ok, g_sh = _load(p_ok), _load(p_sh)
            sens.append(dist(g_ok, g_sh))
            if os.path.exists(p_no) and os.path.exists(REF_IMAGE[ex]):
                ref = _load(REF_IMAGE[ex])
                g_no = _load(p_no)
                # quanto a ref APROXIMA a geração da própria referência,
                # comparado à geração sem referência nenhuma
                fid.append(dist(g_no, ref) - dist(g_ok, ref))
        if sens:
            rows.append((step, float(np.mean(sens)),
                         float(np.mean(fid)) if fid else float('nan')))
    return rows


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print(__doc__)
        return
    print(f"{'braço':<34} {'step':>6} {'sensibilidade':>14} {'fidelidade':>12}")
    print('-' * 70)
    for d in dirs:
        name = os.path.basename(d.rstrip('/'))
        for step, s, f in analyse(d):
            print(f'{name:<34} {step:>6} {s:>14.3f} {f:>12.3f}')
    print('\nsensibilidade: MAIOR = troca de referência muda mais a saída (usa a ref)')
    print('fidelidade:    POSITIVO = a referência puxa a geração na direção dela')
    print('(sensibilidade alta com fidelidade ~0 = instabilidade, não uso de ref)')


if __name__ == '__main__':
    main()
