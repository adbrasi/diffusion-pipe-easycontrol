"""Export a Krea single-stage LoRA checkpoint without resuming GPU training.

DeepSpeed save_quit saves recovery states only. Use a compatible saved adapter
as the key/shape/contract template, then export the actual checkpoint tensors.
"""
import argparse
import json
from pathlib import Path
import shutil

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--training-commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    weights = {}
    for path in sorted(args.checkpoint.glob('layer_*-model_states.pt')):
        layer = int(path.name.split('_')[1].split('-')[0])
        for name, value in torch.load(path, map_location='cpu', weights_only=True).items():
            if '.lora_A.' not in name and '.lora_B.' not in name:
                raise ValueError(f'Unexpected non-LoRA tensor: {path.name}:{name}')
            name = name.replace('.default', '')
            if layer == 0:
                key = 'diffusion_model.' + name
            elif 1 <= layer <= 28 and name.startswith('layer.'):
                key = f'diffusion_model.blocks.{layer-1}.' + name.removeprefix('layer.')
            else:
                raise ValueError(f'Unsupported layer mapping: {path.name}:{name}')
            if key in weights:
                raise ValueError(f'Duplicate tensor: {key}')
            weights[key] = value.to(torch.bfloat16).contiguous()
    expected = load_file(args.template)
    if weights.keys() != expected.keys():
        raise ValueError('Exported keys differ from the compatible adapter template')
    for key, value in weights.items():
        if value.shape != expected[key].shape or not torch.isfinite(value).all().item():
            raise ValueError(f'Invalid tensor: {key}')
    with safe_open(args.template, framework='pt') as handle:
        metadata = handle.metadata()
    metadata.update(diffusion_pipe_commit=args.training_commit,
                    exported_from_checkpoint=args.checkpoint.name,
                    training_step=args.checkpoint.name.removeprefix('global_step'))
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(args.out / 'adapter_model.safetensors'), metadata=metadata)
    shutil.copyfile(args.template.parent / 'adapter_config.json', args.out / 'adapter_config.json')
    for config in args.checkpoint.parent.glob('*.toml'):
        shutil.copyfile(config, args.out / config.name)
    report = dict(checkpoint=str(args.checkpoint), adapter=str(args.out / 'adapter_model.safetensors'),
                  keys=len(weights), dtype='bfloat16', template_shape_check=True, finite=True,
                  training_commit=args.training_commit)
    (args.out / 'export_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
