"""Historical beta1 fidelity ruler via CtxRush v2 + Training Base, inference only.

This intentionally reproduces the beta1's unscaled FP8 training grid and its
historical Turbo fusion (including ignored diff_b). A/B training stays BF16.
"""
import argparse
import json
from pathlib import Path
import sys

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.k2ab_eval_stock import ROOT, execute
from tools.krea2_sampling import build_krea2_timesteps

URL = 'http://127.0.0.1:18820'


def graph(row, reference, variant, prefix):
    turbo = variant == 'Turbo'
    sigmas, _ = build_krea2_timesteps((row['width']//16)*(row['height']//16),
                                    8 if turbo else 28, mu=1.15 if turbo else None)
    nodes = {}

    def add(name, node_type, **inputs):
        nodes[name] = dict(class_type=node_type, inputs=inputs)
        return [name, 0]

    model = add('1', 'UNETLoader', unet_name='krea2_raw_fp8_scaled.safetensors', weight_dtype='default')
    clip = add('2', 'CLIPLoader', clip_name='qwen3vl_4b_bf16.safetensors', type='krea2', device='default')
    vae = add('3', 'VAELoader', vae_name='qwen_image_vae.safetensors')
    image = add('4', 'K2LoadReferencePIL', image=reference)
    model = add('5', 'K2TrainingBase', model=model, fp8_training_grid=True,
                turbo_lora='krea2_turbo_lora_rank_64_bf16.safetensors' if turbo else 'nenhuma',
                turbo_strength=1., verbose=True, weight_storage='fp8_1byte')
    model = add('6', 'CtxRushKrea2MultiRefApply', model=model, clip=clip, vae=vae, image_1=image,
                positive_prompt=row['prompt'], negative_prompt='',
                lora_name='beta1_original_step13250.safetensors', block_strength=1., fusion_strength=1.,
                model_variant=variant.lower(), width=row['width'], height=row['height'], batch_size=1,
                training_vl_contract=True, training_base_quant=True, debug=True)
    guider = add('11', 'CFGGuider', model=model, positive=['6', 1], negative=['6', 2],
                 cfg=1. if turbo else 5.5)
    noise = add('13', 'RandomNoise', noise_seed=row['seed'])
    sampler = add('14', 'KSamplerSelect', sampler_name='euler')
    sigma = add('15', 'ManualSigmas', sigmas=', '.join(f'{value:.12f}' for value in sigmas))
    samples = add('16', 'SamplerCustomAdvanced', noise=noise, guider=guider,
                  sampler=sampler, sigmas=sigma, latent_image=['6', 3])
    pixels = add('17', 'VAEDecode', samples=samples, vae=vae)
    add('18', 'SaveImage', images=pixels, filename_prefix=prefix)
    return nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=13)
    parser.add_argument('--variant', choices=('Turbo', 'Raw'), action='append')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'artifacts/heldout_manifest.json')
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())[:args.limit]
    try:
        for variant in args.variant or ('Turbo', 'Raw'):
            out = args.out / variant
            out.mkdir(parents=True, exist_ok=True)
            for row in rows:
                for kind, reference in [('true', row['reference']), ('shuffled', row['shuffled_reference'])]:
                    dest = out / f'{row["stem"]}_{kind}.png'
                    if dest.exists():
                        continue
                    prompt = graph(row, reference, variant, f'{args.out.name}/{variant}/{dest.stem}')
                    dest.with_suffix('.json').write_text(json.dumps(prompt, indent=2))
                    execute(prompt, dest, url=URL, output_root=ROOT / 'artifacts/legacy_outputs')
                    print('Saved', dest, flush=True)
            # Training Base mutates its loaded weights. Raw must reload pristine
            # weights after Turbo instead of reusing the converted cached model.
            requests.post(URL + '/free', json={'unload_models': True, 'free_memory': True}, timeout=30)
    finally:
        requests.post(URL + '/free', json={'unload_models': True, 'free_memory': True}, timeout=30)


if __name__ == '__main__':
    main()
