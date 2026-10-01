#!/usr/bin/env python3
"""Build contact sheets for the October 1 native campaigns and correction smokes."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


CASES = [
    ('03_ds4_imagem000268', 'Espectadores'),
    ('08_ds1_40zjrzfdij_image_0013', 'Mulher'),
    ('09_ds2_000248', 'Robô'),
    ('23_ds4_imagem000762', 'Castelo'),
]
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def sheet(output, name, title, tests, heldout, cases=CASES):
    width, image_h, row_h, top = 360, 208, 244, 142
    canvas = Image.new('RGB', (width * (len(tests) + 2), top + row_h * len(cases)), '#121820')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 18)
    small = ImageFont.truetype(FONT, 15)
    draw.text((16, 12), title, font=ImageFont.truetype(FONT, 24), fill='white')
    draw.text((16, 46), 'A native · geração Turbo, 8 passos, CFG 1 · A = referência; B = alvo real', font=font, fill='#c4d1df')
    columns = [('Referência A', None), ('Alvo B', None)] + [(t['label'], t) for t in tests]
    for c, (label, test) in enumerate(columns):
        x = c * width
        draw.multiline_text((x + 12, 82), label, font=font, fill='#94d8ff', spacing=4)
        for r, (stem, case) in enumerate(cases):
            y = top + r * row_h
            draw.rectangle((x + 4, y + 4, x + width - 5, y + image_h + 5), fill='#202a36')
            path = heldout / ('control' if c == 0 else 'target') / (stem + '.jpg') if c < 2 else test['images'].get(stem)
            if path:
                with Image.open(path) as source:
                    img = ImageOps.contain(source.convert('RGB'), (width - 12, image_h - 8), Image.Resampling.LANCZOS)
                canvas.paste(img, (x + (width - img.width) // 2, y + (image_h - img.height) // 2 + 4))
            else:
                draw.text((x + 100, y + 90), 'Não gerado', font=font, fill='#8e9dad')
            draw.text((x + 12, y + image_h + 10), case + ' · ' + stem.split('_')[0], font=small, fill='#c4d1df')
    canvas.save(output / f'{name}.png')
    canvas.save(output / f'{name}.jpg', quality=91, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('/workspace/k2ab/artifacts'))
    parser.add_argument('--heldout', type=Path, default=Path('/workspace/k2ab/heldout'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tests = []
    for artifact, group, lr, steps in [
        ('fullbudget_20261001', 'lr0004', '0,0004', [250, 500, 750]),
        ('fullbudget_lr1e4_20261001', 'lr0001', '0,0001', [250, 500, 750, 1000, 1250, 1500, 1750]),
    ]:
        tests.append(dict(id=group + '_smoke10', group=group, label=f'LR {lr}\nSmoke · passo 10', folder=args.artifacts / artifact / 'smoke_eval/Turbo'))
        for step in steps:
            tests.append(dict(id=f'{group}_step{step}', group=group, label=f'LR {lr}\nPasso {step}', folder=args.artifacts / artifact / f'eval/step{step}/Turbo'))
    for variant, label in [('mask_only', 'Máscara corrigida'), ('mask_turbo', 'Máscara + Turbo no treino')]:
        tests.append(dict(id=variant + '_smoke10', group='corrected', label=label + '\nLR 0,0001 · passo 10', folder=args.artifacts / f'corrected_native_{variant}_20261001/smoke_eval/Turbo'))

    inventory = []
    for test in tests:
        if not test['folder'].is_dir():
            raise FileNotFoundError(test['folder'])
        test['images'] = {p.name.removesuffix('_with_lora.png'): p for p in sorted(test['folder'].glob('*_with_lora.png'))}
        unexpected = set(test['images']) - {stem for stem, _ in CASES}
        if unexpected:
            raise ValueError(f'Unrecognized cases: {unexpected}')
        for stem, path in test['images'].items():
            relative = Path('originais') / test['id'] / path.name
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            inventory.append(dict(test=test['id'], label=test['label'], case=stem, source=str(path), file=relative.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    sheet(args.output, 'grid_todos_testes', f'Todos os testes atuais · {len(inventory)} imagens · {len(tests)} avaliações', tests, args.heldout)
    for group, title in [('lr0004', 'Treino do zero · LR 0,0004'), ('lr0001', 'Treino do zero · LR 0,0001'), ('corrected', 'Smokes com as correções · passo 10')]:
        sheet(args.output, 'grid_' + group, title, [t for t in tests if t['group'] == group], args.heldout, cases=CASES[:1] if group == 'corrected' else CASES)
    for stem, label in CASES:
        sheet(args.output, 'grid_caso_' + stem.split('_')[0], label + ' · todos os testes atuais', tests, args.heldout, cases=[(stem, label)])
    manifest = dict(scope='Campanhas de 2026-10-01 e seus smokes; não inclui o histórico anterior A/B.', count=len(inventory), evaluations=len(tests), generation=dict(mode='Turbo', steps=8, cfg=1), images=inventory)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    content = '# Imagens dos testes A native\n\n'
    content += f'{len(inventory)} imagens das duas execuções novas e dos quatro smokes, em {len(tests)} avaliações. Cada linha mostra a mesma cena. As duas primeiras colunas são a referência A e o alvo B. “Não gerado” significa que aquele smoke não avaliou a cena.\n\n'
    content += 'Todas as gerações usam Turbo, 8 passos e CFG 1. Os dois smokes corrigidos foram gerados no passo 10; o teste de resume chegou ao passo 12 sem gerar novas imagens.\n\n'
    content += '[Abrir o grid geral em resolução completa](grid_todos_testes.png)\n\n![Grid geral](grid_todos_testes.jpg)\n\n'
    for group, title in [('lr0001', 'LR 0,0001 — até o passo 1750'), ('lr0004', 'LR 0,0004 — até o passo 750'), ('corrected', 'Máscara corrigida e Turbo congelado no treino — passo 10')]:
        content += f'## {title}\n\n[Abrir PNG](grid_{group}.png)\n\n![{title}](grid_{group}.jpg)\n\n'
    content += '## Imagens originais\n\n'
    for test in tests:
        content += '**' + test['label'].replace('\n', ' · ') + '**\n\n'
        for item in inventory:
            if item['test'] == test['id']:
                content += f'- [{item["case"]}]({item["file"]})\n'
        content += '\n'
    (args.output / 'README.md').write_text(content)
    print(json.dumps(dict(images=len(inventory), evaluations=len(tests), output=str(args.output))))


if __name__ == '__main__':
    main()
