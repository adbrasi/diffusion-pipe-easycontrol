#!/usr/bin/env python3
"""Build a stitched-pair dataset for the classic IC-LoRA fallback.

Composes each reference/target pair into one side-by-side image::

    [reference | target]

and writes the target caption through a template. The result trains with the
stock upstream ``type = 'ideogram4'`` pipeline — no custom packing code — which
makes it the fallback method when the reference-token pipeline misbehaves:
it proves whether the model can learn the pair relation at all.

Usage:

    python tools/make_ideogram4_stitched_pairs.py \
        --target_dir /workspace/dataset/target_images \
        --reference_dir /workspace/dataset/reference_images \
        --output_dir /workspace/dataset/stitched_pairs
"""

import argparse
from pathlib import Path
import sys

from PIL import Image


MEDIA_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}

DEFAULT_TEMPLATE = (
    'Two-panel image showing the same scene and characters. '
    'Left panel: the reference frame. Right panel: {caption}'
)


def _media_by_stem(directory):
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    }


def stitch_pair(reference_path, target_path):
    target = Image.open(target_path).convert('RGB')
    reference = Image.open(reference_path).convert('RGB')
    if reference.size != target.size:
        reference = reference.resize(target.size, Image.LANCZOS)
    combined = Image.new('RGB', (target.width * 2, target.height))
    combined.paste(reference, (0, 0))
    combined.paste(target, (target.width, 0))
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target_dir', required=True, type=Path)
    parser.add_argument('--reference_dir', required=True, type=Path)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument(
        '--caption_template',
        default=DEFAULT_TEMPLATE,
        help='Template applied to each target caption; use {caption} as placeholder.',
    )
    args = parser.parse_args()

    if '{caption}' not in args.caption_template:
        sys.exit('--caption_template must contain the {caption} placeholder')
    for directory in (args.target_dir, args.reference_dir):
        if not directory.is_dir():
            sys.exit(f'Not a directory: {directory}')

    targets = _media_by_stem(args.target_dir)
    references = _media_by_stem(args.reference_dir)
    missing_references = sorted(set(targets) - set(references))
    if missing_references:
        preview = ', '.join(missing_references[:10])
        sys.exit(f'{len(missing_references)} targets have no reference (first: {preview})')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_no_caption = []
    for stem, target_path in targets.items():
        caption_path = target_path.with_suffix('.txt')
        if not caption_path.is_file():
            skipped_no_caption.append(stem)
            continue
        caption = caption_path.read_text(encoding='utf-8').strip()
        stitched = stitch_pair(references[stem], target_path)
        stitched.save(args.output_dir / f'{stem}.png')
        (args.output_dir / f'{stem}.txt').write_text(
            args.caption_template.format(caption=caption) + '\n',
            encoding='utf-8',
        )
        written += 1

    if skipped_no_caption:
        preview = ', '.join(skipped_no_caption[:10])
        print(f'WARNING: skipped {len(skipped_no_caption)} pairs without captions (first: {preview})')
    print(f'Wrote {written} stitched pairs to {args.output_dir}')
    if written == 0:
        sys.exit('No pairs were written')


if __name__ == '__main__':
    main()
