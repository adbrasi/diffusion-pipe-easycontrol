#!/usr/bin/env python3
"""Reference-conditioned inference for diffusion-pipe adapters.

This runner intentionally builds the model from the pipeline's ``to_layers()``
implementation. Training and inference therefore share the exact sequence
packing, role masks, MRoPE coordinates, clean-reference timestep, compact-token
transform, and target-only output slicing.

Supported model types:

* ideogram4_ic_lora
* ideogram4_ominicontrol
* ideogram4_ominicontrol2
* krea2_ic_lora
* krea2_edit
* krea2_ominicontrol
* krea2_ominicontrol2

For ``krea2_edit`` the reference image also grounds the Qwen3-VL text
embeddings (conditional and unconditional), matching the public Krea Edit
contract. ``--reference-guidance`` only zeroes the VAE branch for that model;
the VL grounding stays in place.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import toml
import torch
from PIL import Image, ImageOps
from safetensors import safe_open
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO_ROOT / 'submodules' / 'ComfyUI'
sys.path.insert(0, str(REPO_ROOT))
# The repo's `utils` is a namespace package (no __init__.py) while ComfyUI
# ships a regular `utils` package, which wins resolution regardless of
# sys.path order. Import the repo package first so it lands in sys.modules
# before COMFY_ROOT is visible (train.py relies on the same ordering).
import utils.common  # noqa: E402,F401
sys.path.insert(0, str(COMFY_ROOT))

from tools.krea2_sampling import (
    build_krea2_timesteps,
    resolve_krea2_inference_defaults,
    resolve_krea2_inference_mu,
)


MODEL_CLASSES = {
    'ideogram4_ic_lora': ('models.ideogram4_ic_lora', 'Ideogram4ICLoRAPipeline'),
    'ideogram4_ominicontrol': ('models.ideogram4_ominicontrol', 'Ideogram4OminiControlPipeline'),
    'ideogram4_ominicontrol2': ('models.ideogram4_ominicontrol2', 'Ideogram4OminiControl2Pipeline'),
    'ideogram4_omini_grounded': ('models.ideogram4_omini_grounded', 'Ideogram4OminiGroundedPipeline'),
    'krea2_ic_lora': ('models.krea2_ic_lora', 'Krea2ICLoRAPipeline'),
    'krea2_edit': ('models.krea2_edit', 'Krea2EditPipeline'),
    'krea2_ominicontrol': ('models.krea2_ominicontrol', 'Krea2OminiControlPipeline'),
    'krea2_ominicontrol2': ('models.krea2_ominicontrol2', 'Krea2OminiControl2Pipeline'),
    'krea2_omini_grounded': ('models.krea2_omini_grounded', 'Krea2OminiGroundedPipeline'),
    'krea2_multiref_grounded': ('models.krea2_multiref', 'Krea2MultiRefGroundedPipeline'),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a diffusion-pipe clean-reference adapter without ComfyUI.'
    )
    parser.add_argument('--config', type=Path, required=True, help='Training TOML used for the adapter.')
    parser.add_argument('--adapter', type=Path, required=True, help='Checkpoint directory containing one safetensors file.')
    parser.add_argument(
        '--reference', type=Path, action='append', dest='references',
        help='Reference/control image. Repita para N referencias: a ORDEM das '
             'flags e a ordem dos slots, e e o que <image 1>/<image 2> enderecam.',
    )
    parser.add_argument('--prompt', default='', help='Target prompt (Ideogram structured JSON is accepted).')
    parser.add_argument('--negative-prompt', default='', help='Negative/unconditional prompt.')
    parser.add_argument('--output', type=Path, default=Path('reference_sample.png'))
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument(
        '--steps', type=int, default=None,
        help='Denoising steps. Defaults to Raw=28, Turbo=8, and non-Krea=20.',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--text-guidance', type=float, default=None,
        help=(
            'Standard CFG scale used by this runner. Defaults to Raw=5.5 '
            '(equivalent to Krea guidance 4.5), Turbo=1.0, and non-Krea=1.0.'
        ),
    )
    parser.add_argument('--reference-guidance', type=float, default=1.0)
    parser.add_argument('--adapter-scale', type=float, default=1.0)
    parser.add_argument(
        '--conrad-contract', action='store_true',
        help=(
            'Infer under the conradlocke identity-edit contract instead of the '
            'ostris/fork one: reference tokens modulated with the TARGET '
            'timestep, bare vision block (no "Picture 1:" prefix), grounding '
            'capped by longest side 768 (area resample), and reference encoded '
            'at native resolution with the latent resized to the target grid.'
        ),
    )
    parser.add_argument(
        '--vl-longest-side', type=int, default=None,
        help='Cap the Qwen3-VL copy by longest side (conradlocke grounding_px).',
    )
    parser.add_argument(
        '--disable-vl-reference', action='store_true',
        help=(
            'Ablation: do not show the reference to Qwen3-VL. The clean VAE '
            'reference branch remains active.'
        ),
    )
    parser.add_argument(
        '--disable-vae-reference', action='store_true',
        help=(
            'Ablation: zero the image signal in the clean VAE reference branch. '
            'Qwen3-VL visual grounding remains active for krea2_edit.'
        ),
    )
    parser.add_argument(
        '--shift', type=float, default=None,
        help='FlowMatchEuler shift for non-Krea models; defaults to model config or 3.',
    )
    parser.add_argument(
        '--mu', type=float, default=None,
        help='Explicit Krea 2 timestep-shift mu. By default it is derived from output resolution.',
    )
    parser.add_argument('--krea-min-res', type=int, default=256)
    parser.add_argument('--krea-max-res', type=int, default=1280)
    parser.add_argument('--krea-y1', type=float, default=0.5)
    parser.add_argument('--krea-y2', type=float, default=1.15)
    parser.add_argument(
        '--krea-variant', choices=('auto', 'raw', 'turbo'), default='auto',
        help=(
            'Select Krea inference defaults. Auto recognizes "turbo" in the '
            'diffusion checkpoint path and otherwise uses Raw.'
        ),
    )
    parser.add_argument('--blocks-to-swap', type=int, default=None, help='Override block swapping from the TOML.')
    parser.add_argument(
        '--reference-fit', choices=('crop', 'stretch', 'exact', 'native_latent'), default='crop',
        help='How to map the reference to the requested output size.',
    )
    parser.add_argument('--allow-contract-mismatch', action='store_true')
    parser.add_argument(
        '--validate-only', action='store_true',
        help='Validate config/checkpoint metadata without loading CUDA models.',
    )
    return parser.parse_args()


def find_adapter_file(path: Path) -> Path:
    if path.is_file():
        if path.suffix != '.safetensors':
            raise ValueError(f'Adapter file must be safetensors: {path}')
        return path
    files = sorted(path.glob('*.safetensors'))
    if not files:
        raise FileNotFoundError(f'No safetensors adapter found in {path}')
    if len(files) != 1:
        raise RuntimeError(f'Expected one safetensors adapter in {path}, found {len(files)}')
    return files[0]


def read_metadata(adapter_file: Path) -> dict[str, str]:
    with safe_open(adapter_file, framework='pt', device='cpu') as handle:
        return dict(handle.metadata() or {})


def load_raw_config(path: Path) -> dict:
    with path.open() as handle:
        return json.loads(json.dumps(toml.load(handle)))


def resolve_sampling_args(config: dict, args) -> None:
    """Fill model-specific defaults without overriding explicit CLI values."""
    model_type = config['model']['type']
    if model_type.startswith('krea2_'):
        variant, args.steps, args.text_guidance = resolve_krea2_inference_defaults(
            config['model'].get('diffusion_model', ''),
            variant=args.krea_variant,
            steps=args.steps,
            text_guidance=args.text_guidance,
        )
        args.mu = resolve_krea2_inference_mu(variant, args.mu)
        mu_label = 'resolution-derived' if args.mu is None else str(args.mu)
        print(
            f'Krea 2 inference profile: variant={variant}, steps={args.steps}, '
            f'text_guidance={args.text_guidance}, mu={mu_label} '
            f'(standard CFG convention)'
        )
    else:
        args.steps = 20 if args.steps is None else args.steps
        args.text_guidance = 1.0 if args.text_guidance is None else args.text_guidance


def expected_contract(config: dict) -> dict[str, str]:
    model_type = config['model']['type']
    expected = {'model_type': model_type, 'sequence_layout': 'text,target,reference'}
    if model_type.startswith('ideogram4_'):
        section = config.get('ideogram4_ic_lora', {})
        control = config.get('ominicontrol', {})
        expected.update({
            'reference_model_timestep': str(float(control.get(
                'reference_model_timestep', section.get('reference_model_timestep', 1.0)
            ))),
            # Packing geometry must match training: an adapter trained with
            # e.g. offset -1 silently produces wrong RoPE placement under the
            # default config otherwise.
            'reference_position_offset': str(int(control.get(
                'reference_position_offset', section.get('reference_position_offset', 1)
            ))),
            'condition_dropout': str(float(control.get(
                'condition_dropout', section.get('condition_dropout', 0.1)
            ))),
        })
        if 'ominicontrol' in model_type:
            expected['condition_only_lora'] = str(bool(control.get('condition_only_lora', True))).lower()
    elif model_type in ('krea2_edit', 'krea2_omini_grounded', 'krea2_multiref_grounded'):
        section = config.get(model_type, {})
        expected.update({
            'reference_model_timestep': (
                'target' if section.get('reference_timestep', 'zero') == 'target' else '0.0'
            ),
            'position_mode': str(section.get('position_mode', 'subject')),
            'condition_token_stride': '1',
            'control_family': (
                'krea2_omini_grounded' if model_type == 'krea2_omini_grounded'
                else 'krea2_edit_dual'
            ),
            'vl_conditioning': 'qwen3vl_image_grounded',
            'vl_image_max_pixels': str(int(section.get('vl_image_max_pixels', 384 * 384))),
            'lora_targets': 'blocks+txtfusion',
        })
        if model_type in ('krea2_omini_grounded', 'krea2_multiref_grounded'):
            expected['condition_only_lora'] = str(bool(section.get('condition_only_lora', True))).lower()
    else:
        section_name = 'krea2_ic_lora' if model_type == 'krea2_ic_lora' else 'ominicontrol'
        section = config.get(section_name, {})
        expected.update({
            'reference_model_timestep': (
                'target' if section.get('reference_timestep', 'zero') == 'target' else '0.0'
            ),
            'position_mode': str(section.get('position_mode', 'subject')),
            'condition_token_stride': str(int(section.get('condition_token_stride', 1))),
        })
        if 'ominicontrol' in model_type:
            expected['condition_only_lora'] = str(bool(section.get('condition_only_lora', True))).lower()
    if model_type.endswith('ominicontrol2'):
        control = config.get('ominicontrol', {})
        expected.update({
            'control_family': 'ominicontrol_v2',
            'condition_token_stride': str(int(control.get('condition_token_stride', 2))),
            'independent_condition': str(bool(control.get('independent_condition', True))).lower(),
            'condition_encode': 'pixel_bilinear',
        })
    return expected


def validate_contract(config: dict, metadata: dict[str, str], allow_mismatch: bool) -> None:
    model_type = config['model']['type']
    if model_type not in MODEL_CLASSES:
        raise ValueError(f'Unsupported reference model type: {model_type}')
    if not metadata:
        message = 'Adapter has no reference-contract metadata; exact compatibility cannot be verified.'
        if allow_mismatch:
            print(f'WARNING: {message}')
            return
        raise RuntimeError(message + ' Use --allow-contract-mismatch only for a known legacy checkpoint.')

    # Soft check: warn (never fail) when the adapter was trained on a different
    # base checkpoint, e.g. a Raw-trained LoRA applied to Turbo. That can be
    # intentional, but should never happen silently.
    trained_base = metadata.get('base_model_file')
    inference_base = Path(config['model']['diffusion_model']).name
    if trained_base and trained_base != inference_base:
        print(
            f'WARNING: adapter was trained on base model {trained_base!r} but inference '
            f'is using {inference_base!r}. Cross-applying is out of the trained distribution.'
        )

    mismatches = []
    missing = []
    for key, expected in expected_contract(config).items():
        actual = metadata.get(key)
        if actual is None:
            missing.append(key)
        elif actual != expected:
            mismatches.append(f'{key}: checkpoint={actual!r}, config={expected!r}')
    if missing or mismatches:
        details = []
        if missing:
            details.append('missing metadata: ' + ', '.join(missing))
        details.extend(mismatches)
        message = 'Reference adapter contract mismatch:\n  ' + '\n  '.join(details)
        if allow_mismatch:
            print('WARNING: ' + message)
        else:
            raise RuntimeError(message)


def normalize_runtime_config(config: dict) -> None:
    from utils import common

    model = config['model']
    dtype_name = model['dtype']
    model['dtype'] = common.DTYPE_MAP[dtype_name]
    for key in ('transformer_dtype', 'diffusion_model_dtype'):
        if key in model:
            model[key] = common.DTYPE_MAP[model[key]]
    adapter = config.get('adapter')
    if adapter is None:
        raise ValueError('Inference requires the original [adapter] section in the TOML')
    adapter['alpha'] = adapter.get('alpha', adapter['rank'])
    adapter['dropout'] = 0.0
    adapter['dtype'] = common.DTYPE_MAP[adapter.get('dtype', dtype_name)]
    config.setdefault('reentrant_activation_checkpointing', False)
    common.AUTOCAST_DTYPE = model['dtype']


def create_pipeline(config: dict):
    model_type = config['model']['type']
    module_name, class_name = MODEL_CLASSES[model_type]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(config)


def load_reference_pixels(path: Path, width: int, height: int, fit: str) -> torch.Tensor:
    image = Image.open(path)
    if image.mode == 'RGBA' or ('transparency' in image.info and image.mode != 'RGB'):
        rgba = image.convert('RGBA')
        canvas = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
        canvas.alpha_composite(rgba)
        image = canvas.convert('RGB')
    else:
        image = image.convert('RGB')
    requested = (width, height)
    if fit == 'exact':
        if image.size != requested:
            raise ValueError(f'Reference is {image.size}, expected exactly {requested}')
    elif fit == 'stretch':
        image = image.resize(requested, Image.Resampling.LANCZOS)
    elif fit == 'native_latent':
        # conradlocke node semantics: the source is VAE-encoded at its own
        # resolution and the LATENT is bilinear-resized to the target grid.
        # Here we only snap the pixels to /16; encode_reference resizes the
        # latent afterwards.
        new_w = max(round(image.width / 16) * 16, 16)
        new_h = max(round(image.height / 16) * 16, 16)
        if (new_w, new_h) != image.size:
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    else:
        image = ImageOps.fit(image, requested, method=Image.Resampling.LANCZOS)

    import torchvision.transforms.functional as TF
    pixels = TF.pil_to_tensor(image).to(torch.float32) / 127.5 - 1.0
    return pixels.unsqueeze(0)


@torch.no_grad()
def encode_reference(pipeline, path: Path, width: int, height: int, fit: str) -> torch.Tensor:
    from comfy import model_management

    pixels = load_reference_pixels(path, width, height, fit)
    if pipeline.is_video_vae:
        pixels = pixels.unsqueeze(2)
    pixels = pipeline.prepare_reference_media(pixels)
    vae = pipeline.get_vae()
    vae.load_model_if_needed()
    latents = pipeline.vae_encode(pixels.to('cuda', pipeline.dtype)).float().cpu()
    if fit == 'native_latent':
        target_h = height // pipeline.spatial_compression
        target_w = width // pipeline.spatial_compression
        if latents.shape[-2:] != (target_h, target_w):
            squeeze_frames = latents.ndim == 5
            work = latents[:, :, 0] if squeeze_frames else latents
            work = torch.nn.functional.interpolate(
                work, size=(target_h, target_w), mode='bilinear'
            )
            latents = work.unsqueeze(2) if squeeze_frames else work
    model_management.unload_all_models()
    torch.cuda.empty_cache()
    return latents


def scale_adapter(pipeline, scale: float) -> int:
    count = 0
    for module in pipeline.diffusion_model.modules():
        scaling = getattr(module, 'scaling', None)
        if not isinstance(scaling, dict):
            continue
        for adapter_name in list(scaling):
            scaling[adapter_name] *= scale
            count += 1
    return count


def setup_diffusion_pipeline(pipeline, adapter_path: Path, config: dict, blocks_to_swap: int | None):
    pipeline.load_diffusion_model()
    pipeline.configure_adapter(config['adapter'])
    pipeline.load_adapter_weights(adapter_path)
    pipeline.diffusion_model.eval()

    blocks = config.get('blocks_to_swap', 0) if blocks_to_swap is None else blocks_to_swap
    if blocks:
        pipeline.enable_block_swap(blocks)
        pipeline.prepare_block_swap_inference()
    else:
        pipeline.diffusion_model.to('cuda')

    sequential = torch.nn.Sequential(*pipeline.to_layers())
    sequential.eval()
    return sequential, blocks


def call_model(model, latent, timestep, conds, reference):
    return model((latent, timestep, *conds, reference)).float()


@torch.no_grad()
def denoise(
    pipeline,
    model,
    reference,
    target_shape,
    steps,
    seed,
    text_guidance,
    reference_guidance,
    shift,
    width,
    height,
    krea_mu,
    krea_min_res,
    krea_max_res,
    krea_y1,
    krea_y2,
):
    from diffusers import FlowMatchEulerDiscreteScheduler

    generator = torch.Generator(device='cuda').manual_seed(seed)
    latent = torch.randn(target_shape, generator=generator, device='cuda')
    reference = reference.to('cuda')
    zero_reference = torch.zeros_like(reference)
    conds = tuple(value.to('cuda') for value in pipeline.conds)
    unconds = tuple(value.to('cuda') for value in getattr(pipeline, 'unconds', ()))

    def predict_velocity(timestep):
        if text_guidance == 1.0 and reference_guidance == 1.0:
            return call_model(model, latent, timestep, conds, reference)
        elif text_guidance != 1.0 and reference_guidance == 1.0:
            negative = call_model(model, latent, timestep, unconds, reference)
            full = call_model(model, latent, timestep, conds, reference)
            return negative + text_guidance * (full - negative)
        elif text_guidance == 1.0:
            no_reference = call_model(model, latent, timestep, conds, zero_reference)
            full = call_model(model, latent, timestep, conds, reference)
            return no_reference + reference_guidance * (full - no_reference)
        else:
            unconditional = call_model(model, latent, timestep, unconds, zero_reference)
            with_reference = call_model(model, latent, timestep, unconds, reference)
            full = call_model(model, latent, timestep, conds, reference)
            return (
                unconditional
                + reference_guidance * (with_reference - unconditional)
                + text_guidance * (full - with_reference)
            )

    if pipeline.name.startswith('krea2_'):
        patch = int(pipeline.diffusion_model.patch)
        sequence_length = (target_shape[-2] // patch) * (target_shape[-1] // patch)
        schedule, resolved_mu = build_krea2_timesteps(
            sequence_length,
            steps,
            min_resolution=krea_min_res,
            max_resolution=krea_max_res,
            spatial_compression=pipeline.spatial_compression,
            patch_size=patch,
            y1=krea_y1,
            y2=krea_y2,
            mu=krea_mu,
        )
        print(
            f'Krea 2 official Euler schedule: {width}x{height}, '
            f'image_tokens={sequence_length}, mu={resolved_mu:.6f}'
        )
        pairs = zip(schedule[:-1], schedule[1:])
        for current, next_value in tqdm(
            pairs, total=steps, desc='Reference sampling (Krea 2)'
        ):
            timestep = latent.new_full((latent.shape[0],), current)
            velocity = predict_velocity(timestep)
            latent = latent + (next_value - current) * velocity
    else:
        scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
        sigmas = torch.linspace(1.0, 1.0 / steps, steps)
        scheduler.set_timesteps(sigmas=sigmas, device='cuda')
        for step in tqdm(scheduler.timesteps, desc='Reference sampling'):
            timestep = (step / 1000).float().view(1)
            velocity = predict_velocity(timestep)
            latent = scheduler.step(velocity, step, latent, return_dict=False)[0]
    return latent


def offload_diffusion(pipeline, sequential) -> None:
    del sequential
    pipeline.diffusion_model.to('cpu')
    torch.cuda.empty_cache()


@torch.no_grad()
def decode_and_save(pipeline, latent: torch.Tensor, output: Path) -> None:
    from comfy import model_management
    import torchvision.utils

    vae = pipeline.get_vae()
    vae.load_model_if_needed()
    image = pipeline.vae_decode(latent)
    model_management.unload_all_models()

    # Comfy VAE output is channel-last; Krea's image VAE retains a frame axis.
    if image.ndim == 5:
        image = image[:, 0]
    if image.ndim != 4:
        raise RuntimeError(f'Unexpected decoded image shape: {tuple(image.shape)}')
    if image.shape[-1] in (3, 4):
        image = image.movedim(-1, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(image[:, :3].float().clamp(0, 1).cpu(), output)


def main():
    args = parse_args()
    config = load_raw_config(args.config)
    resolve_sampling_args(config, args)
    adapter_file = find_adapter_file(args.adapter)
    metadata = read_metadata(adapter_file)
    validate_contract(config, metadata, args.allow_contract_mismatch)
    print(json.dumps({
        'model_type': config['model']['type'],
        'adapter': str(adapter_file),
        'contract': metadata.get('reference_contract', 'unknown'),
        'control_family': metadata.get('control_family', 'ic_lora'),
    }, indent=2))
    if args.validate_only:
        return
    if not args.references:
        raise ValueError('--reference is required unless --validate-only is used')
    if args.width <= 0 or args.height <= 0 or args.steps <= 0:
        raise ValueError('width, height, and steps must be positive')

    normalize_runtime_config(config)
    pipeline = create_pipeline(config)
    if args.conrad_contract:
        if config['model']['type'] != 'krea2_edit':
            raise ValueError('--conrad-contract requires model type krea2_edit')
        pipeline.reference_timestep_mode = 'target'
        pipeline.vl_prompt_style = 'plain'
        pipeline.vl_longest_side = args.vl_longest_side or 768
        args.reference_fit = 'native_latent'
        print(
            'Conrad contract: reference_timestep=target, vl_prompt_style=plain, '
            f'vl_longest_side={pipeline.vl_longest_side}, reference_fit=native_latent'
        )
    elif args.vl_longest_side:
        pipeline.vl_longest_side = args.vl_longest_side
    if args.width % pipeline.pixels_round_to_multiple or args.height % pipeline.pixels_round_to_multiple:
        raise ValueError(
            f'width/height must be multiples of {pipeline.pixels_round_to_multiple} for {pipeline.name}'
        )
    need_unconditional = args.text_guidance != 1.0
    sample_kwargs = {}
    if config['model']['type'] in ('krea2_edit', 'krea2_omini_grounded', 'krea2_multiref_grounded', 'ideogram4_omini_grounded') and not args.disable_vl_reference:
        # Dual conditioning: the reference grounds the Qwen3-VL embeddings of
        # both the conditional and the unconditional prompt.
        # lista na MESMA ordem dos slots VAE — se os dois canais divergirem,
        # o binding '<image N>' -> slot N aprendido no treino nao vale mais
        sample_kwargs['control_files'] = [[str(p) for p in args.references]]
    pipeline.prepare_sample_test(
        args.prompt,
        negative_prompt=args.negative_prompt,
        cfg=2 if need_unconditional else 1,
        **sample_kwargs,
    )
    encoded = [
        encode_reference(pipeline, path, args.width, args.height, args.reference_fit)
        for path in args.references
    ]
    if len(encoded) == 1:
        full_reference = encoded[0]
    else:
        # (1, C, 1, h, w) cada -> (1, C, N, h, w): o mesmo layout que o
        # treino monta a partir do cache do VAE
        squeezed = [e.squeeze(2) if e.ndim == 5 else e for e in encoded]
        full_reference = torch.stack(squeezed, dim=2)
    if args.disable_vae_reference:
        full_reference = torch.zeros_like(full_reference)
    print(
        'Reference branches: '
        f'Qwen3-VL={"off" if args.disable_vl_reference else "on"}, '
        f'VAE-image={"off" if args.disable_vae_reference else "on"}'
    )
    target_shape = (
        (1, pipeline.channels, args.height // pipeline.spatial_compression, args.width // pipeline.spatial_compression)
        if not pipeline.is_video_vae else
        (1, pipeline.channels, 1, args.height // pipeline.spatial_compression, args.width // pipeline.spatial_compression)
    )

    sequential, blocks = setup_diffusion_pipeline(
        pipeline,
        adapter_file,
        config,
        args.blocks_to_swap,
    )
    scaled = scale_adapter(pipeline, args.adapter_scale)
    print(f'Adapter scale applied to {scaled} PEFT modules; block swap={blocks}')

    target_template = torch.zeros(target_shape, dtype=full_reference.dtype)
    reference = pipeline.prepare_reference_latents(
        full_reference,
        target_template,
        timestep_quantile=0.5,
    )
    is_krea = pipeline.name.startswith('krea2_')
    if is_krea and args.shift is not None:
        raise ValueError('Krea 2 uses --mu (or resolution-derived mu), not --shift')
    shift = args.shift
    if not is_krea and shift is None:
        # Priority: config -> adapter's recorded training shift -> legacy 3.
        # ComfyUI's canonical Ideogram 4 sampling is shift=1.0; the adapter
        # metadata makes the training-time schedule reproducible here.
        metadata_shift = metadata.get('training_shift')
        if 'shift' in config['model']:
            shift = float(config['model']['shift'])
        elif metadata_shift not in (None, 'none'):
            shift = float(metadata_shift)
        else:
            shift = 3.0
    krea_mu = args.mu
    if is_krea and krea_mu is None and 'mu' in config['model']:
        krea_mu = float(config['model']['mu'])
    latent = denoise(
        pipeline,
        sequential,
        reference,
        target_shape,
        args.steps,
        args.seed,
        args.text_guidance,
        args.reference_guidance,
        shift,
        args.width,
        args.height,
        krea_mu,
        args.krea_min_res,
        args.krea_max_res,
        args.krea_y1,
        args.krea_y2,
    )
    offload_diffusion(pipeline, sequential)
    decode_and_save(pipeline, latent, args.output)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
