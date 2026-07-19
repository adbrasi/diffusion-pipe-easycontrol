"""Batch driver for infer_reference_adapter: load the pipeline once, run N cases.

Cases manifest (JSON list): [{"prompt", "reference", "output",
"adapter_scale": 1.0, "disable_vae_reference": false, "seed": 76}, ...]
Text guidance is fixed at 1.0 (first-judgment protocol); use the single-case
tool for guidance sweeps.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

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



def set_adapter_scale_absolute(pipeline, scale, registry={}):
    """scale_adapter() multiplies in place; store originals once and set absolutely."""
    count = 0
    for name, module in pipeline.diffusion_model.named_modules():
        scaling = getattr(module, 'scaling', None)
        if not isinstance(scaling, dict):
            continue
        for adapter_name in list(scaling):
            key = (name, adapter_name)
            if key not in registry:
                registry[key] = scaling[adapter_name]
            scaling[adapter_name] = registry[key] * scale
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--adapter', required=True, type=Path)
    parser.add_argument('--cases', required=True, type=Path)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--text-guidance', type=float, default=7.0)
    parser.add_argument('--cfg-tail', type=float, default=3.0)
    parser.add_argument('--triple-cfg', action='store_true',
                        help='v = vU + sR(vR-vU) + sT(vRT-vR): separa força da referência e do texto')
    parser.add_argument('--sr', type=float, default=5.0)
    parser.add_argument('--st', type=float, default=9.0)
    parser.add_argument('--blocks-to-swap', type=int, default=None)
    parser.add_argument('--reference-fit', default='crop')
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text())
    config = ira.load_raw_config(args.config)
    adapter_file = ira.find_adapter_file(args.adapter)
    metadata = ira.read_metadata(adapter_file)
    ira.validate_contract(config, metadata, allow_mismatch=False)
    ira.normalize_runtime_config(config)
    pipeline = ira.create_pipeline(config)

    if args.width % pipeline.pixels_round_to_multiple or args.height % pipeline.pixels_round_to_multiple:
        raise ValueError(f'width/height must be multiples of {pipeline.pixels_round_to_multiple}')

    grounded = config['model']['type'] in ('krea2_edit', 'krea2_omini_grounded', 'ideogram4_omini_grounded')

    # FASE A: codificar todos os textos com só o TE na GPU.
    T0_NULL = ('{"high_level_description": "A coherent image conditioned only by the '
               'supplied reference.", "style_description": "", "compositional_deconstruction": ""}')
    encoded = []
    encoded_null = []
    for case in cases:
        sample_kwargs = {}
        if grounded and not case.get('disable_vl_reference', False):
            sample_kwargs['control_files'] = [str(case['reference'])]
        pipeline.prepare_sample_test(case['prompt'], negative_prompt='', cfg=1, **sample_kwargs)
        encoded.append(tuple(v.cpu() for v in pipeline.conds))
        if args.triple_cfg:
            pipeline.prepare_sample_test(T0_NULL, negative_prompt='', cfg=1, **sample_kwargs)
            encoded_null.append(tuple(v.cpu() for v in pipeline.conds))
    from comfy import model_management
    model_management.unload_all_models()
    pipeline.text_encoders = []
    import gc; gc.collect(); torch.cuda.empty_cache()

    # FASE B: DiT (bf16 + swap) + uncond dedicado fp8.
    sequential, blocks = ira.setup_diffusion_pipeline(pipeline, adapter_file, config, args.blocks_to_swap)
    from ideogram4_uncond import Ideogram4Uncond
    uncond = Ideogram4Uncond()
    print(f'Pipeline loaded once; block swap={blocks}; {len(cases)} cases')

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
        set_adapter_scale_absolute(pipeline, float(case.get('adapter_scale', 1.0)))
        guidance = float(case.get('text_guidance', args.text_guidance))
        full_reference = ira.encode_reference(
            pipeline, Path(case['reference']), args.width, args.height, args.reference_fit
        )
        if case.get('disable_vae_reference', False):
            full_reference = torch.zeros_like(full_reference)
        reference = pipeline.prepare_reference_latents(
            full_reference,
            torch.zeros(target_shape, dtype=full_reference.dtype),
            timestep_quantile=0.5,
        ).to('cuda')
        # Schedule oficial + CFG dual estagiado (uncond = modelo incondicional
        # dedicado, image-only => referência só no positivo, paridade BitPoet).
        with torch.no_grad():
            conds = tuple(v.to('cuda') for v in encoded[i])
            generator = torch.Generator(device='cuda').manual_seed(int(case.get('seed', 76)))
            latent = torch.randn(target_shape, generator=generator, device='cuda')
            sigmas = ideogram4_sigmas(args.steps, args.width, args.height)
            n_steps = len(sigmas) - 1
            conds_null = tuple(v.to('cuda') for v in encoded_null[i]) if args.triple_cfg else None
            for si, (current, next_value) in enumerate(zip(sigmas[:-1].tolist(), sigmas[1:].tolist())):
                timestep = latent.new_full((1,), current)
                full = ira.call_model(sequential, latent, timestep, conds, reference)
                tail = 1.0 if si < 0.7 * n_steps else (args.cfg_tail / max(guidance, 1e-6))
                if args.triple_cfg:
                    u = uncond.velocity(latent, current)
                    r = ira.call_model(sequential, latent, timestep, conds_null, reference)
                    sr = args.sr * tail if args.sr * tail > 1 else args.sr
                    st = args.st * tail if args.st * tail > 1 else args.st
                    velocity = u + sr * (r - u) + st * (full - r)
                else:
                    g = guidance if si < 0.7 * n_steps else min(guidance, args.cfg_tail)
                    if g != 1.0:
                        negative = uncond.velocity(latent, current)
                        velocity = negative + g * (full - negative)
                    else:
                        velocity = full
                latent = latent + (next_value - current) * velocity
        ira.decode_and_save(pipeline, latent, out)
        print(f'[{i+1}/{len(cases)}] OK: {out.name}', flush=True)

    print('BATCH COMPLETO')


if __name__ == '__main__':
    main()
