"""Base-native control (Codex veto #3): stock Ideogram 4 T2I through the SAME
weights/sampler/decoder as the reference runner, but with the original packing —
no reference tokens, no adapter. One process, N prompts.

Usage: infer_base_native.py --config <pilot toml> --cases cases.json --width W --height H
Cases: [{"prompt", "output", "seed": 76}, ...]
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import infer_reference_adapter as ira  # noqa: E402


def ideogram4_sigmas(num_steps, width, height, mu=0.5, std=1.75):
    """Official ComfyUI Ideogram 4 schedule: logit-normal quantile sigmas."""
    import math
    mean = mu + 0.5 * math.log((width * height) / (512 * 512))
    u = torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float64)
    t = 1.0 - torch.special.expit(torch.as_tensor(mean) + std * torch.special.ndtri(u))
    t_min = 1.0 / (1.0 + math.exp(0.5 * 18.0))
    t_max = 1.0 / (1.0 + math.exp(0.5 * -15.0))
    sigmas = (1.0 - t.clamp(t_min, t_max)).flip(0)
    sigmas[0] = 1.0
    sigmas[-1] = 0.0
    return sigmas.to(torch.float32)



@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--text-guidance', type=float, default=7.0)
    args = parser.parse_args()

    from models import ideogram4

    cases = json.loads(args.cases.read_text())
    config = ira.load_raw_config(args.config)
    config = copy.deepcopy(config)
    config['model']['type'] = 'ideogram4'
    shift = float(config['model'].get('shift', 3.0))
    ira.normalize_runtime_config(config)

    pipeline = ideogram4.Ideogram4Pipeline(config)

    # FASE A: codificar todos os textos com só o TE na GPU.
    encoded = []
    for case in cases:
        pipeline.prepare_sample_test(case['prompt'], negative_prompt='', cfg=1)
        encoded.append(tuple(v.cpu() for v in pipeline.conds))
    from comfy import model_management
    model_management.unload_all_models()
    pipeline.text_encoders = []
    import gc; gc.collect(); torch.cuda.empty_cache()

    # FASE B: DiT + uncond dedicado.
    pipeline.load_diffusion_model()
    pipeline.diffusion_model.eval()
    pipeline.diffusion_model.to('cuda')
    sequential = torch.nn.Sequential(*pipeline.to_layers())
    sequential.eval()
    from ideogram4_uncond import Ideogram4Uncond
    uncond = Ideogram4Uncond()

    target_shape = (
        1, pipeline.channels,
        args.height // pipeline.spatial_compression,
        args.width // pipeline.spatial_compression,
    )

    for i, case in enumerate(cases):
        out = Path(case['output'])
        if out.exists():
            print(f'[{i+1}/{len(cases)}] pulando (existe): {out.name}')
            continue
        conds = tuple(value.to('cuda') for value in encoded[i])
        generator = torch.Generator(device='cuda').manual_seed(int(case.get('seed', 76)))
        latent = torch.randn(target_shape, generator=generator, device='cuda')
        sigmas = ideogram4_sigmas(args.steps, args.width, args.height)
        guidance = args.text_guidance
        n_steps = len(sigmas) - 1
        for si, (current, next_value) in enumerate(tqdm(
            list(zip(sigmas[:-1].tolist(), sigmas[1:].tolist())), desc='Base-native sampling'
        )):
            timestep = latent.new_full((1,), current)
            full = sequential((latent, timestep, *conds)).float()
            g = guidance if si < 0.7 * n_steps else min(guidance, 3.0)
            if g != 1.0:
                # Paridade de deployment: negativo = transformer incondicional
                # dedicado, passe image-only (DualModelGuider oficial).
                negative = uncond.velocity(latent, current)
                velocity = negative + g * (full - negative)
            else:
                velocity = full
            latent = latent + (next_value - current) * velocity
        ira.decode_and_save(pipeline, latent, out)
        print(f'[{i+1}/{len(cases)}] OK: {out.name}', flush=True)

    print('BASE-NATIVE COMPLETO')


if __name__ == '__main__':
    main()
