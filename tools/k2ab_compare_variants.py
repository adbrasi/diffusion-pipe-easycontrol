"""Build a labeled reference/target/Turbo/Raw grid from existing renders."""
import argparse
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps


def build(rows, pairs, renders, destination, first=1):
    cell, header = 512, 96
    font = ImageFont.truetype('DejaVuSans.ttf', 20)
    small = ImageFont.truetype('DejaVuSans.ttf', 17)
    prepared = []
    for index, row in enumerate(rows, first):
        paths = [pairs / 'control' / row['reference'], pairs / 'target' / row['reference']]
        paths += [renders / variant / f'{row["stem"]}_with_lora.png' for variant in ('Turbo', 'Raw')]
        tiles = []
        for path in paths:
            with Image.open(path) as image:
                tiles.append(ImageOps.contain(image.convert('RGB'), (cell, 384)))
        prompt_lines = textwrap.wrap(row['prompt'], width=180)
        label_height = 32 + 23 * len(prompt_lines)
        prepared.append((index, row, tiles, prompt_lines, label_height, max(t.height for t in tiles)))
    canvas = Image.new('RGB', (cell * 4, header + sum(p[4] + p[5] + 12 for p in prepared)), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), 'Krea2 | Adapter A/native step1000 | sampling512 (buckets) | seed76 | LoRA strength1', fill='black', font=font)
    for column, label in enumerate(('Imagem A - referencia', 'Imagem B - alvo real', 'Turbo | 8 steps | CFG1', 'Raw | 28 steps | CFG5.5')):
        draw.text((column * cell + 10, 55), label, fill='black', font=font)
    y = header
    for index, row, tiles, lines, label_height, height in prepared:
        draw.text((10, y + 3), f'{index:02d} | {row["stem"]} | {row["width"]}x{row["height"]}', fill='black', font=small)
        for line_index, line in enumerate(lines):
            draw.text((10, y + 28 + line_index * 23), line, fill='black', font=small)
        for column, tile in enumerate(tiles):
            canvas.paste(tile, (column * cell + (cell - tile.width) // 2,
                                y + label_height + (height - tile.height) // 2))
        y += label_height + height + 12
        draw.line((0, y - 4, cell * 4, y - 4), fill='#aaaaaa')
    canvas.save(destination, quality=95)
    print(destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--renders', type=Path, required=True)
    parser.add_argument('--pairs', type=Path, default=Path('/workspace/k2ab/heldout'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    build(rows, args.pairs, args.renders, args.out / 'A1000_Turbo_Raw_512_all10.jpg')
    for start in range(0, len(rows), 5):
        build(rows[start:start + 5], args.pairs, args.renders,
              args.out / f'A1000_Turbo_Raw_512_cases{start + 1:02d}-{min(start + 5, len(rows)):02d}.jpg', first=start + 1)


if __name__ == '__main__':
    main()
