#!/usr/bin/env python3
"""One three-column review of saved checkpoints, grouped by held-out pair.

Uses saved outputs only; does not run inference. Shuffled rows display the
reference actually supplied to that generation, not the original reference.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)


def find_image(root, stem):
    matches = [p for p in root.glob(stem + '.*')
               if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one image for {root / stem}: {matches}')
    return matches[0]


def read_image(path, width, height):
    with Image.open(path) as im:
        return ImageOps.fit(im.convert('RGB'), (width, height))


def render(rows, cell, title):
    top, band, section = 80, 56 if cell < 512 else 72, 42
    pairs = list(dict.fromkeys(r['stem'] for r in rows))
    canvas = Image.new('RGB', (3 * cell, top + len(rows) * (cell + band)
                              + len(pairs) * section), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, font=font(15 if cell < 512 else 23), fill='#222222')
    for col, label in enumerate(('A / referencia usada', 'B / proxima cena real', 'Resultado do teste')):
        draw.text((col * cell + 10, 42), label,
                  font=font(13 if cell < 512 else 21), fill='#222222')
    y = top
    previous = None
    for row in rows:
        if row['stem'] != previous:
            draw.rectangle((0, y, canvas.width, y + section), fill='#e5e7eb')
            draw.text((10, y + 10), f"PAR {row['pair_number']:02d} / {row['stem']}",
                      font=font(13 if cell < 512 else 20), fill='#111827')
            y += section
            previous = row['stem']
        background = '#dbeafe' if row['layout'] == 'aligned' else '#dcfce7'
        draw.rectangle((0, y, canvas.width, y + band), fill=background)
        condition = 'REF ORIGINAL' if row['condition'] == 'true' else 'SHUFFLE / REF TROCADA'
        label = f"{row['run']} | {row['layout']} | step {row['step']} | {condition}"
        draw.text((10, y + 5), label, font=font(13 if cell < 512 else 21), fill='#111827')
        draw.text((10, y + (29 if cell < 512 else 39)),
                  f"A usada: {row['reference_stem']} | seed {row['seed']}",
                  font=font(11 if cell < 512 else 17), fill='#374151')
        y += band
        for col, key in enumerate(('reference', 'target', 'output')):
            im = read_image(row[key], row['width'], row['height'])
            im.thumbnail((cell, cell), Image.Resampling.LANCZOS)
            canvas.paste(im, (col * cell + (cell - im.width) // 2,
                              y + (cell - im.height) // 2))
        y += cell
    return canvas


def save_png(im, path):
    tmp = path.with_suffix('.tmp')
    im.save(tmp, format='PNG')
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evaluations', type=Path, default=Path('/workspace/nextscene_artifacts/E1'))
    ap.add_argument('--out', type=Path, default=Path('/workspace/nextscene_artifacts/comparacoes'))
    ap.add_argument('--runs', nargs='+', default=['A250', 'B250', 'A500', 'B500'])
    args = ap.parse_args()
    evaluations = []
    for name in args.runs:
        paths = list((args.evaluations / name).glob('*/metrics.json'))
        if len(paths) != 1:
            raise ValueError(f'{name}: expected one completed evaluation, got {len(paths)}')
        path = paths[0]
        metrics = json.loads(path.read_text())
        config = json.loads((path.parent / 'eval_config.json').read_text())
        evaluations.append((name, path.parent, metrics['pairs'], config))
    stems = [r['stem'] for r in evaluations[0][2]]
    for name, _, pairs, _ in evaluations:
        if [r['stem'] for r in pairs] != stems:
            raise ValueError(f'{name}: pair order differs; cannot compare these evaluations')
    rows = []
    for i, stem in enumerate(stems):
        for name, directory, pairs, config in evaluations:
            for condition in ('true', 'shuffled'):
                reference_stem = stem if condition == 'true' else stems[(i + 1) % len(stems)]
                root = Path(config['pairs'])
                output = directory / f'{stem}_{condition}.png'
                if not output.exists():
                    raise FileNotFoundError(output)
                rows.append(dict(run=name, step=int(name[1:]),
                                 layout='aligned' if name.startswith('A') else 'disjoint_w',
                                 stem=stem, pair_number=i + 1, condition=condition,
                                 reference_stem=reference_stem,
                                 reference=str(find_image(root / 'control', reference_stem)),
                                 target=str(find_image(root / 'target', stem)), output=str(output),
                                 width=pairs[i].get('width', config['width']),
                                 height=pairs[i].get('height', config['height']), seed=config['seed']))
    args.out.mkdir(parents=True, exist_ok=True)
    prefix = 'REVISAO_250_500_3_COLUNAS'
    save_png(render(rows, 256, '250 + 500 steps | originais + shuffles | agrupados por par'),
             args.out / f'{prefix}.png')
    for step in sorted(set(r['step'] for r in rows)):
        selected = [r for r in rows if r['step'] == step]
        save_png(render(selected, 384, f'Todos de {step} steps | originais + shuffles'),
                 args.out / f'REVISAO_{step}_3_COLUNAS.png')
    # A single file with one held-out pair per page avoids an enormous scroll.
    pages = [render([r for r in rows if r['stem'] == stem], 512,
                    f'Par {i + 1:02d}/{len(stems)} | 250 + 500 | original + shuffle')
             for i, stem in enumerate(stems)]
    pdf = args.out / f'{prefix}.pdf'
    tmp = pdf.with_suffix('.tmp')
    pages[0].save(tmp, format='PDF', save_all=True, append_images=pages[1:],
                  resolution=150, quality=95, title='Anima nextscene: 250 e 500 steps')
    tmp.replace(pdf)
    save_png(pages[0], args.out / f'{prefix}_pagina_01.png')
    (args.out / f'{prefix}_manifest.json').write_text(json.dumps(rows, indent=2))
    print(f'{len(rows)} rows; {len(stems)} pairs; {len(evaluations)} checkpoints; 3 columns. {pdf}')


if __name__ == '__main__':
    main()
