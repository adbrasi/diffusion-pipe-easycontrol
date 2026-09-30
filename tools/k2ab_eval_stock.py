"""Evaluate native adapters through the unmodified, local stock ComfyUI API."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.krea2_sampling import build_krea2_timesteps

URL = 'http://127.0.0.1:18819'
ROOT = Path('/workspace/k2ab')


def graph(row, reference, variant, adapter_name, prefix, reference_method=True, t2i=False,
          base_model='krea2_raw_bf16.safetensors', reference_pixels='node_1mp'):
    turbo = variant == 'Turbo'
    sigmas, _ = build_krea2_timesteps((row['width']//16)*(row['height']//16),
                                    8 if turbo else 28, mu=1.15 if turbo else None)
    nodes = {}

    def add(name, node_type, **inputs):
        nodes[name] = dict(class_type=node_type, inputs=inputs)
        return [name, 0]

    model = add('1', 'UNETLoader', unet_name=base_model, weight_dtype='default')
    clip = add('2', 'CLIPLoader', clip_name='qwen3vl_4b_bf16.safetensors', type='krea2', device='default')
    vae = add('3', 'VAELoader', vae_name='qwen_image_vae.safetensors')
    image = add('4', 'LoadImage', image=reference)
    if turbo:
        model = add('5', 'LoraLoaderModelOnly', model=model,
                    lora_name='krea2_turbo_lora_rank_64_bf16.safetensors', strength_model=1.)
    if adapter_name:
        model = add('6', 'LoraLoaderModelOnly', model=model, lora_name=adapter_name, strength_model=1.)
    if t2i:
        positive = add('7', 'CLIPTextEncode', clip=clip, text=row['prompt'])
        negative = add('8', 'CLIPTextEncode', clip=clip, text='')
    else:
        if reference_pixels == 'target':
            # Core nodes only. Keep VL grounding on the original image, while
            # the guiding latent uses the 512 probe's target-sized contract.
            positive = add('7', 'TextEncodeQwenImageEditPlus', clip=clip, prompt=row['prompt'], image1=image)
            negative = add('8', 'TextEncodeQwenImageEditPlus', clip=clip, prompt='', image1=image)
            resized = add('19', 'ImageScale', image=image, upscale_method='bicubic',
                          width=row['width'], height=row['height'], crop='center')
            reference_latent = add('20', 'VAEEncode', pixels=resized, vae=vae)
            positive = add('21', 'ReferenceLatent', conditioning=positive, latent=reference_latent)
            negative = add('22', 'ReferenceLatent', conditioning=negative, latent=reference_latent)
        else:
            positive = add('7', 'TextEncodeQwenImageEditPlus', clip=clip, prompt=row['prompt'], vae=vae, image1=image)
            negative = add('8', 'TextEncodeQwenImageEditPlus', clip=clip, prompt='', vae=vae, image1=image)
    if reference_method and not t2i:
        positive = add('9', 'FluxKontextMultiReferenceLatentMethod', conditioning=positive,
                       reference_latents_method='index_timestep_zero')
        negative = add('10', 'FluxKontextMultiReferenceLatentMethod', conditioning=negative,
                       reference_latents_method='index_timestep_zero')
    guider = add('11', 'CFGGuider', model=model, positive=positive, negative=negative, cfg=1. if turbo else 5.5)
    latent = add('12', 'EmptyLatentImage', width=row['width'], height=row['height'], batch_size=1)
    noise = add('13', 'RandomNoise', noise_seed=row['seed'])
    sampler = add('14', 'KSamplerSelect', sampler_name='euler')
    sigma = add('15', 'ManualSigmas', sigmas=', '.join(f'{value:.12f}' for value in sigmas))
    samples = add('16', 'SamplerCustomAdvanced', noise=noise, guider=guider,
                  sampler=sampler, sigmas=sigma, latent_image=latent)
    pixels = add('17', 'VAEDecode', samples=samples, vae=vae)
    add('18', 'SaveImage', images=pixels, filename_prefix=prefix)
    return nodes


def execute(prompt, destination, url=URL, output_root=None):
    # Supervisor reports RUNNING before ComfyUI finishes importing nodes.
    ready_deadline = time.time() + 120
    while True:
        try:
            requests.get(url + '/system_stats', timeout=5).raise_for_status()
            break
        except requests.RequestException:
            if time.time() >= ready_deadline:
                raise
            time.sleep(2)
    response = requests.post(url + '/prompt', json={'prompt': prompt}, timeout=30)
    response.raise_for_status()
    result = response.json()
    if result.get('node_errors'):
        raise RuntimeError(json.dumps(result))
    prompt_id = result['prompt_id']
    start = time.time()
    while time.time() - start < 1200:
        history = requests.get(url + '/history/' + prompt_id, timeout=30).json()
        if prompt_id in history:
            entry = history[prompt_id]
            if entry.get('status', {}).get('status_str') == 'error':
                raise RuntimeError(json.dumps(entry['status']))
            images = entry.get('outputs', {}).get('18', {}).get('images', [])
            if images:
                row = images[0]
                source = (output_root or ROOT / 'artifacts/stock_outputs') / row['subfolder'] / row['filename']
                shutil.copyfile(source, destination)
                return
        time.sleep(2)
    raise TimeoutError(prompt_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=13)
    parser.add_argument('--variant', choices=('Turbo', 'Raw'), action='append')
    parser.add_argument('--disable-reference-method', action='store_true')
    parser.add_argument('--t2i-base', action='store_true')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'artifacts/heldout_manifest.json')
    parser.add_argument('--base-model', default='krea2_raw_bf16.safetensors')
    parser.add_argument('--reference-pixels', choices=('node_1mp', 'target'), default='node_1mp')
    args = parser.parse_args()
    name = None
    if args.adapter:
        name = args.adapter.parent.parent.parent.name + '_' + args.adapter.parent.name + '.safetensors'
        link = Path('/workspace/models/krea2/loras') / name
        if not link.exists():
            link.symlink_to(args.adapter.resolve())
    rows = json.loads(args.manifest.read_text())[:args.limit]
    try:
        for variant in args.variant or ('Turbo', 'Raw'):
            out = args.out / variant
            out.mkdir(parents=True, exist_ok=True)
            for row in rows:
                conditions = [('true', row['reference'])]
                if not args.t2i_base:
                    conditions.append(('shuffled', row['shuffled_reference']))
                for kind, reference in conditions:
                    dest = out / f'{row["stem"]}_{kind}.png'
                    if dest.exists():
                        continue
                    prompt = graph(row, reference, variant, name, f'{args.out.name}/{variant}/{dest.stem}',
                                   reference_method=not args.disable_reference_method, t2i=args.t2i_base,
                                   base_model=args.base_model, reference_pixels=args.reference_pixels)
                    dest.with_suffix('.json').write_text(json.dumps(prompt, indent=2))
                    execute(prompt, dest)
                    print('Saved', dest, flush=True)
    finally:
        requests.post(URL + '/free', json={'unload_models': True, 'free_memory': True}, timeout=30)


if __name__ == '__main__':
    main()
