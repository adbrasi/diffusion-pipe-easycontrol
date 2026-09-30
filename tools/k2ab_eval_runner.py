"""Batch the historical runner contract, loading the BF16 DiT only once."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import infer_reference_adapter as runner
import torch

ROOT = Path('/workspace/k2ab')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variant', choices=('Turbo', 'Raw'), required=True)
    parser.add_argument('--adapter-only', action='store_true')
    parser.add_argument('--compare-adapter', action='store_true')
    parser.add_argument('--limit', type=int, default=13)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'artifacts/heldout_manifest.json')
    args = parser.parse_args()
    config = runner.load_raw_config(args.config)
    adapter = runner.find_adapter_file(args.adapter)
    runner.validate_contract(config, runner.read_metadata(adapter), False)
    runner.normalize_runtime_config(config)
    pipeline = runner.create_pipeline(config)
    turbo = args.variant == 'Turbo'
    rows = json.loads(args.manifest.read_text())[:args.limit]
    prepared = []
    for row in rows:
        conditions = ([('with_lora', row['reference']), ('without_lora', row['reference'])]
                      if args.compare_adapter else [('true', row['reference']), ('shuffled', row['shuffled_reference'])])
        if args.adapter_only:
            conditions = [('with_lora', row['reference'])]
        for kind, name in conditions:
            output = args.out / args.variant / f'{row["stem"]}_{kind}.png'
            if output.exists():
                continue
            reference_path = ROOT / 'heldout/control' / name
            pipeline.prepare_sample_test(row['prompt'], negative_prompt='', cfg=2,
                                         control_files=[str(reference_path)])
            conds = tuple(value.cpu() for value in pipeline.conds)
            unconds = tuple(value.cpu() for value in pipeline.unconds)
            reference = runner.encode_reference(pipeline, reference_path, row['width'], row['height'], 'crop')
            prepared.append((row, output, reference, conds, unconds, kind))
    if not prepared:
        return
    official = Path('/workspace/models/krea2/loras/krea2_turbo_lora_rank_64_bf16.safetensors')
    swap = 0 if config['model'].get('base_quant') == 'fp8_scaled' else 8
    sequential, blocks = runner.setup_diffusion_pipeline(pipeline, adapter, config, swap,
                                                        turbo_lora=official if turbo else None)
    for row, output, reference, conds, unconds, kind in prepared:
        # Decoding offloads the previous DiT. Restore its non-block modules and
        # the inference block layout without loading weights again from disk.
        if blocks:
            block_list = pipeline.diffusion_model.blocks
            pipeline.diffusion_model.blocks = None
            pipeline.diffusion_model.to('cuda')
            pipeline.diffusion_model.blocks = block_list
            pipeline.prepare_block_swap_inference()
        else:
            pipeline.diffusion_model.to('cuda')
        pipeline.conds, pipeline.unconds = conds, unconds
        shape = (1, pipeline.channels, 1, row['height']//8, row['width']//8)
        reference = pipeline.prepare_reference_latents(reference, torch.zeros(shape), timestep_quantile=.5)
        # PEFT context disables all trained DiT/text-fusion adapter layers,
        # while the official Turbo LoRA remains fused in the frozen base.
        context = pipeline.lora_model.disable_adapter() if kind == 'without_lora' else nullcontext()
        with context:
            latent = runner.denoise(pipeline, sequential, reference, shape,
                8 if turbo else 28, row['seed'], 1. if turbo else 5.5,
                1., None, row['width'], row['height'], 1.15 if turbo else None,
                256, 1280, .5, 1.15, noise_device='cpu')
        runner.offload_diffusion(pipeline, sequential)
        runner.decode_and_save(pipeline, latent, output)
        print('Saved', output, flush=True)


if __name__ == '__main__':
    main()
