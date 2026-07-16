#!/usr/bin/env python3
"""Validate a Krea 2 Edit (dual conditioning) run before allocating GPU time.

Beyond the config/dataset checks shared with the IC-LoRA preflight, this
verifies that the Qwen3-VL text encoder checkpoint ships the vision tower
(``visual.*`` weights). Without it the image-grounded branch silently breaks.
Torch-free: the safetensors key list is read straight from the file header.
"""

import argparse
import json
from pathlib import Path
import sys
import tomllib


MEDIA_EXTENSIONS = {
    '.bmp',
    '.jpeg',
    '.jpg',
    '.png',
    '.tif',
    '.tiff',
    '.webp',
}

# 100 MB is far beyond any real safetensors JSON header; catches corrupt files.
MAX_HEADER_BYTES = 100 * 1024 * 1024


def _resolve_path(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    # Match train.py: relative paths are resolved from the launch directory.
    return Path.cwd() / path


def _media_by_stem(directory):
    return {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    }


def read_safetensors_keys(path):
    with open(path, 'rb') as handle:
        header_length = int.from_bytes(handle.read(8), 'little')
        if not 0 < header_length <= MAX_HEADER_BYTES:
            raise ValueError(f'Implausible safetensors header length: {header_length}')
        header = json.loads(handle.read(header_length))
    return [key for key in header if key != '__metadata__']


def check_vision_tower(te_path):
    """Return (ok, detail) for the Qwen3-VL vision tower in the TE checkpoint."""
    try:
        keys = read_safetensors_keys(te_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return False, f'Could not read safetensors header: {error}'
    vision_keys = [key for key in keys if 'visual' in key]
    if not vision_keys:
        return False, (
            'No visual.* weights found. krea2_edit grounds the reference image '
            'through the Qwen3-VL vision tower; use a full Qwen3-VL 4B text '
            'encoder export (not a text-only one).'
        )
    return True, f'{len(vision_keys)} vision-tower tensors found'


def validate(config_path):
    config_path = Path(config_path).expanduser().resolve()
    errors = []
    warnings = []
    summaries = []

    if not config_path.is_file():
        return [f'Config does not exist: {config_path}'], warnings, summaries

    with config_path.open('rb') as handle:
        config = tomllib.load(handle)

    model = config.get('model', {})
    if model.get('type') != 'krea2_edit':
        errors.append("[model].type must be 'krea2_edit'")

    model_paths = {
        'diffusion model': model.get('diffusion_model'),
        'VAE': model.get('vae'),
    }
    text_encoders = model.get('text_encoders', [])
    if len(text_encoders) != 1 or not text_encoders[0].get('path'):
        errors.append('[model].text_encoders must contain exactly one path')
    else:
        model_paths['text encoder'] = text_encoders[0]['path']
        if text_encoders[0].get('type') != 'krea2':
            errors.append("Krea 2 text encoder type must be 'krea2'")

    for label, value in model_paths.items():
        if not value:
            errors.append(f'Missing {label} path in [model]')
            continue
        path = _resolve_path(value)
        if not path.is_file():
            errors.append(f'{label.title()} does not exist: {path}')
        else:
            summaries.append(f'{label}: {path}')
            if label == 'text encoder':
                ok, detail = check_vision_tower(path)
                if ok:
                    summaries.append(f'text encoder vision tower: {detail}')
                else:
                    errors.append(f'Text encoder vision tower check failed: {detail}')

    diffusion_path = str(model.get('diffusion_model', '')).lower()
    if 'turbo' in diffusion_path:
        warnings.append('Krea 2 Edit should train on Krea 2 Raw, not Turbo')

    section = config.get('krea2_edit', {})
    dropout = float(section.get('condition_dropout', 0.0))
    if dropout != 0.0:
        errors.append(
            'krea2_edit requires condition_dropout = 0.0: the public Krea Edit '
            'training never drops references, and dropping only the VAE branch '
            'while the Qwen3-VL grounding remains would be inconsistent'
        )
    position_mode = section.get('position_mode', 'subject')
    if position_mode not in ('subject', 'spatial'):
        errors.append("position_mode must be 'subject' or 'spatial'")
    if position_mode == 'subject' and float(
        section.get('reference_position_offset', 1.0)
    ) != 1.0:
        warnings.append('subject reference_position_offset differs from the public +1 contract')
    if int(section.get('condition_token_stride', 1)) != 1:
        errors.append('krea2_edit requires condition_token_stride = 1 (canonical Edit contract)')
    vl_max_pixels = int(section.get('vl_image_max_pixels', 384 * 384))
    if vl_max_pixels < 28 * 28:
        errors.append('vl_image_max_pixels must be at least 28*28')
    if vl_max_pixels != 384 * 384:
        warnings.append(
            f'vl_image_max_pixels = {vl_max_pixels} differs from the public 384*384 budget'
        )
    vl_tokens = vl_max_pixels // (32 * 32)
    summaries.append(
        f'VL grounding: <= ~{vl_tokens} vision tokens per reference in the text embeddings'
    )

    dataset_value = config.get('dataset')
    if not dataset_value:
        errors.append('Missing top-level dataset path')
        return errors, warnings, summaries
    dataset_path = _resolve_path(dataset_value)
    if not dataset_path.is_file():
        errors.append(f'Dataset config does not exist: {dataset_path}')
        return errors, warnings, summaries

    with dataset_path.open('rb') as handle:
        dataset = tomllib.load(handle)

    frame_buckets = dataset.get('frame_buckets', [1])
    if frame_buckets != [1]:
        errors.append('Krea 2 Edit currently requires frame_buckets = [1]')

    for resolution in dataset.get('resolutions', []):
        if not isinstance(resolution, int):
            continue
        if resolution % 16:
            warnings.append(f'{resolution}px is not divisible by Krea VAE f8 * patch 2')
        image_tokens = (resolution // 16) ** 2
        summaries.append(
            f'{resolution}px square: {image_tokens} target + '
            f'{image_tokens} reference image tokens'
        )

    directories = dataset.get('directory', [])
    if not directories:
        errors.append('Dataset config has no [[directory]] entries')

    total_pairs = 0
    for index, directory_config in enumerate(directories):
        target_value = directory_config.get('path')
        reference_value = directory_config.get('control_path')
        if not target_value or not reference_value:
            errors.append(f'Dataset directory {index} requires both path and control_path')
            continue

        target_dir = _resolve_path(target_value)
        reference_dir = _resolve_path(reference_value)
        if not target_dir.is_dir():
            errors.append(f'Target directory does not exist: {target_dir}')
            continue
        if not reference_dir.is_dir():
            errors.append(f'Reference directory does not exist: {reference_dir}')
            continue

        targets = _media_by_stem(target_dir)
        references = _media_by_stem(reference_dir)
        missing_references = sorted(targets.keys() - references.keys())
        extra_references = sorted(references.keys() - targets.keys())
        if not targets:
            errors.append(f'No target images found in {target_dir}')
        if missing_references:
            errors.append(
                f'{len(missing_references)} target images have no reference with the same '
                f'stem in {reference_dir}: {", ".join(missing_references[:10])}'
            )
        if extra_references:
            warnings.append(f'{len(extra_references)} unpaired reference images in {reference_dir}')

        paired = len(targets.keys() & references.keys())
        total_pairs += paired
        missing_captions = sum(
            1 for stem in targets if not (target_dir / f'{stem}.txt').is_file()
        )
        if missing_captions:
            warnings.append(
                f'{missing_captions}/{len(targets)} target images have no .txt caption '
                f'in {target_dir}'
            )
        summaries.append(f'dataset[{index}]: {paired} paired control/target images')

    if total_pairs == 0:
        errors.append('No valid control/target pairs were found')
    else:
        summaries.append(f'total pairs: {total_pairs}')
        # (caption + ~vl_tokens vision tokens + suffix) x 12 layers x 2560 dims x 2 bytes
        approx_mb = (vl_tokens + 80) * 12 * 2560 * 2 / (1024 * 1024)
        summaries.append(
            f'text-embedding cache estimate: ~{approx_mb:.0f} MB per pair '
            f'(~{approx_mb * total_pairs / 1024:.1f} GB total)'
        )

    return errors, warnings, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Training TOML path')
    args = parser.parse_args()

    errors, warnings, summaries = validate(args.config)
    for message in summaries:
        print(f'[OK] {message}')
    for message in warnings:
        print(f'[WARN] {message}')
    for message in errors:
        print(f'[ERROR] {message}')

    if errors:
        print(f'Preflight failed with {len(errors)} error(s).')
        return 1
    print('Preflight passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
