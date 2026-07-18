#!/usr/bin/env python3
"""Tagger WD14 (SmilingWolf swinv2-v3) para as imagens B do ctx_part_2.
Formato de saída: tags danbooru com underscore, separadas por ', '
(mesmo estilo das captions A do dataset parents de abril).
general >= 0.35, character >= 0.85; rating excluída."""
import csv
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL = '/workspace/models/wd14/model.onnx'
TAGS = '/workspace/models/wd14/selected_tags.csv'
GEN_T, CHAR_T = 0.35, 0.85

rows = list(csv.DictReader(open(TAGS)))
names = [r['name'] for r in rows]
cats = [int(r['category']) for r in rows]  # 0=general 4=character 9=rating

sess = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0]
size = inp.shape[1]  # NHWC


def prep(path):
    im = Image.open(path).convert('RGB')
    w, h = im.size
    side = max(w, h)
    canvas = Image.new('RGB', (side, side), (255, 255, 255))
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    canvas = canvas.resize((size, size), Image.BICUBIC)
    arr = np.asarray(canvas, dtype=np.float32)[:, :, ::-1]  # RGB->BGR
    return arr[None]


def tag(path):
    probs = sess.run(None, {inp.name: prep(path)})[0][0]
    out = []
    for name, cat, p in zip(names, cats, probs):
        if cat == 0 and p >= GEN_T:
            out.append((p, name))
        elif cat == 4 and p >= CHAR_T:
            out.append((p + 1.0, name))  # personagens primeiro
    out.sort(reverse=True)
    return ', '.join(n for _, n in out)


def main():
    folder = Path(sys.argv[1])
    imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.jpg', '.png', '.jpeg', '.webp'))
    done = 0
    for p in imgs:
        txt = folder / (p.stem + '.txt')
        if txt.exists():
            continue
        txt.write_text(tag(p), encoding='utf-8')
        done += 1
        if done % 100 == 0:
            print(f'{done}/{len(imgs)}', flush=True)
    print(f'CONCLUIDO: {done} taggeadas de {len(imgs)}', flush=True)


if __name__ == '__main__':
    main()
