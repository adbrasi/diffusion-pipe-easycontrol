#!/usr/bin/env python3
"""Validate an Ideogram 4 IC-LoRA config before allocating GPU time."""

import argparse
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


def _resolve_path(value, _config_path):
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
    if model.get('type') != 'ideogram4_ic_lora':
        errors.append("[model].type must be 'ideogram4_ic_lora'")

    model_paths = {
        'diffusion model': model.get('diffusion_model'),
        'VAE': model.get('vae'),
    }
    text_encoders = model.get('text_encoders', [])
    if len(text_encoders) != 1 or not text_encoders[0].get('path'):
        errors.append('[model].text_encoders must contain exactly one path')
    else:
        model_paths['text encoder'] = text_encoders[0]['path']

    for label, value in model_paths.items():
        if not value:
            errors.append(f'Missing {label} path in [model]')
            continue
        path = _resolve_path(value, config_path)
        if not path.is_file():
            errors.append(f'{label.title()} does not exist: {path}')
        else:
            summaries.append(f'{label}: {path}')

    reference_config = config.get('ideogram4_ic_lora', {})
    dropout = float(reference_config.get('condition_dropout', 0.1))
    if not 0.0 <= dropout <= 1.0:
        errors.append('condition_dropout must be between 0.0 and 1.0')
    if int(reference_config.get('reference_position_offset', 1)) != 1:
        warnings.append('reference_position_offset differs from the proven +1 contract')
    if float(reference_config.get('reference_model_timestep', 1.0)) != 1.0:
        warnings.append('reference_model_timestep differs from the clean 1.0 contract')

    dataset_value = config.get('dataset')
    if not dataset_value:
        errors.append('Missing top-level dataset path')
        return errors, warnings, summaries
    dataset_path = _resolve_path(dataset_value, config_path)
    if not dataset_path.is_file():
        errors.append(f'Dataset config does not exist: {dataset_path}')
        return errors, warnings, summaries

    with dataset_path.open('rb') as handle:
        dataset = tomllib.load(handle)

    resolutions = dataset.get('resolutions', [])
    if resolutions:
        for resolution in resolutions:
            if isinstance(resolution, int):
                side = resolution
                image_tokens = (side // 16) ** 2
                summaries.append(
                    f'{side}px square: {image_tokens} target + '
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
            errors.append(
                f'Dataset directory {index} requires both path and control_path'
            )
            continue

        target_dir = _resolve_path(target_value, dataset_path)
        reference_dir = _resolve_path(reference_value, dataset_path)
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
            preview = ', '.join(missing_references[:10])
            errors.append(
                f'{len(missing_references)} target images have no reference '
                f'with the same stem in {reference_dir}: {preview}'
            )
        if extra_references:
            warnings.append(
                f'{len(extra_references)} unpaired reference images in {reference_dir}'
            )

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
        summaries.append(
            f'dataset[{index}]: {paired} paired target/reference images'
        )

    if total_pairs == 0:
        errors.append('No valid target/reference pairs were found')
    else:
        summaries.append(f'total pairs: {total_pairs}')

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
