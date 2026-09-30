"""Krea 2 edit trained on the STOCK ComfyUI contract (arm A of the Krea 2 A/B).

Goal: a LoRA that runs in an unmodified ComfyUI (>= commit c9602625, 2026-07-18)
with only core nodes, and no fp8 patch:

    UNETLoader -> LoraLoaderModelOnly -> KSampler
    TextEncodeQwenImageEditPlus(clip, prompt, vae, image1)
        -> FluxKontextMultiReferenceLatentMethod('index_timestep_zero')

Everything below mirrors that path; see docs/KREA2_ANALISE_METODO_2026-09.md.

  * Reference latent  : image resized with comfy.utils.common_upscale(..., "area")
                        to ~1024*1024 px, AR kept, sides rounded to /8
                        (TextEncodeQwenImageEditPlus), encoded on its OWN grid.
  * Reference position: frame 1, h/w grid from 0 at the reference's own size
                        (Krea2 process_img(ref, index=1)) == fork position_mode
                        'subject' with offset 1.0.
  * Reference timestep: 0 per token (index_timestep_zero); 'target' reproduces
                        the stock 'index' method instead.
  * Text              : Qwen-Image-Edit system template +
                        "Picture 1: <|vision_start|><|image_pad|><|vision_end|>" + prompt,
                        VL copy resized with "area" to ~384*384 px (upscaling allowed).
                        vl_grounding=false gives the CLIPTextEncode+ReferenceLatent
                        variant (no image in the text encoder).
  * LoRA              : global (blocks + txtfusion), loadable by LoraLoaderModelOnly.

Base numerics: NEVER train with diffusion_model_dtype='float8' through
models/base.py (it re-quantizes WITHOUT the fp8_scaled weight_scale and zeroes
up to ~27% of block weights). Use 'bfloat16' (+ blocks_to_swap).

Text-encoder parity: the Qwen3-VL implementation must be the one of the ComfyUI
the user runs (DeepStack/MRoPE changed after 2026-06-23). Validate with
tools/krea2_native_parity.py before caching.
"""

import math

import torch
from PIL import Image, ImageOps

import comfy.utils

from models.krea2_edit import Krea2EditPipeline, VISION_BLOCK
from models.krea2_reference import Krea2ReferencePipeline

# comfy_extras/nodes_qwen.py::TextEncodeQwenImageEditPlus (identical upstream @ fb2315f1)
QWEN_EDIT_PLUS_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should alter or modify the "
    "image. Generate a new image that meets the user's requirements while maintaining consistency with "
    "the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
VL_TOTAL_PIXELS = 384 * 384
REF_TOTAL_PIXELS = 1024 * 1024


def native_vl_size(width, height, total=VL_TOTAL_PIXELS):
    scale = math.sqrt(total / (width * height))
    return round(width * scale), round(height * scale)


def native_ref_size(width, height, total=REF_TOTAL_PIXELS):
    scale = math.sqrt(total / (width * height))
    return round(width * scale / 8.0) * 8, round(height * scale / 8.0) * 8


def load_comfy_image(path):
    """LoadImage (PIL branch): exif transpose, RGB, float [0,1], (1, H, W, 3).

    Stock LoadImage decodes through PyAV first; JPEG decoders can differ by a
    few LSB. Prefer PNG references when measuring parity."""
    image = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return (pixels.reshape(image.height, image.width, 3).to(torch.float32) / 255.0).unsqueeze(0)


def area_resize_bhwc(pixels, width, height):
    samples = pixels.movedim(-1, 1)
    return comfy.utils.common_upscale(samples, width, height, 'area', 'disabled').movedim(1, -1)


def native_vl_image(path):
    pixels = load_comfy_image(path)
    w, h = native_vl_size(pixels.shape[2], pixels.shape[1])
    return area_resize_bhwc(pixels, w, h)


def native_reference_pixels(path):
    """Reference pixels exactly as the node feeds vae.encode, returned in the
    trainer's layout (C, 1, H, W) and range [-1, 1]."""
    pixels = load_comfy_image(path)
    w, h = native_ref_size(pixels.shape[2], pixels.shape[1])
    pixels = area_resize_bhwc(pixels, w, h)[0].clamp(0, 1)
    return (pixels.permute(2, 0, 1) * 2.0 - 1.0).unsqueeze(1)


def native_text(caption, num_images=1, grounded=True):
    if not grounded or num_images == 0:
        return caption
    return ''.join(f'Picture {i + 1}: {VISION_BLOCK}' for i in range(num_images)) + caption


def native_tokenize(text_encoder, caption, control_files=None, grounded=True):
    """Tokens exactly as TextEncodeQwenImageEditPlus (grounded) or CLIPTextEncode
    (grounded=False) build them for a Krea 2 CLIP."""
    images = []
    if control_files and grounded:
        files = control_files if isinstance(control_files, (list, tuple)) else [control_files]
        images = [native_vl_image(f) for f in files]
    text = native_text(caption, len(images), grounded=grounded)
    if images:
        return text_encoder.tokenize(text, images=images, llama_template=QWEN_EDIT_PLUS_TEMPLATE)
    return text_encoder.tokenize(text)


class PreprocessNativeControlFile:
    """CONTROL-file preprocessing with the node's own ~1MP rule (ignores the
    target bucket on purpose: the stock node does too)."""

    def __call__(self, spec, mask_filepath=None, size_bucket=None):
        path = spec[1] if isinstance(spec, (list, tuple)) else spec
        return [(native_reference_pixels(path), None)]


class Krea2NativePipeline(Krea2EditPipeline):
    name = 'krea2_native'
    config_section = 'krea2_native'
    adapter_allowed_key_substrings = ('.blocks.', '.txtfusion.')

    def __init__(self, config):
        section = config.setdefault(self.config_section, {})
        # Geometry is the stock contract, not a knob.
        for key, value in (('position_mode', 'subject'), ('reference_position_offset', 1.0),
                           ('reference_position_scale', 1.0), ('independent_condition', False),
                           ('condition_only_lora', False)):
            if key in section and section[key] != value:
                raise ValueError(f'krea2_native fixes {key}={value!r} (stock ComfyUI contract)')
            section[key] = value
        if config['model'].get('diffusion_model_dtype') == 'float8':
            raise ValueError(
                "krea2_native: diffusion_model_dtype='float8' re-quantizes the fp8_scaled base without "
                "its weight_scale (see docs/KREA2_ANALISE_METODO_2026-09.md §1.1). Use 'bfloat16' "
                "with blocks_to_swap."
            )
        super().__init__(config)
        self.vl_grounding = bool(section.get('vl_grounding', True))
        self.vl_prompt_style = 'qwen_edit_plus'

    def get_preprocess_control_file_fn(self):
        return PreprocessNativeControlFile()

    def prepare_reference_latents(self, reference, noisy_target, timestep_quantile=None):
        # The stock node encodes the reference on its own ~1MP grid, so its
        # latent size is independent of the target bucket. Only batch/channel/
        # frame must agree (micro_batch_size_per_gpu = 1 keeps shapes simple).
        if reference.shape[:3] != noisy_target.shape[:3]:
            raise ValueError(
                f'Krea2 reference and target batch/channel/frame shapes must match: '
                f'{tuple(reference.shape[:3])} != {tuple(noisy_target.shape[:3])}'
            )
        return reference

    def encode_caption_for_cache(self, text_encoder, caption, control_file):
        return native_tokenize(text_encoder, caption, control_file, grounded=self.vl_grounding)

    def get_call_text_encoder_fn(self, text_encoder):
        te_idx = next((i for i, te in enumerate(self.text_encoders) if te == text_encoder), None)
        if te_idx is None:
            raise RuntimeError('Unknown text encoder')

        @torch.inference_mode()
        def fn(captions, is_video, control_files=None):
            assert not any(is_video)
            control_files = control_files if control_files is not None else [None] * len(captions)
            embeds_list, mask_list = [], []
            for caption, control_file in zip(captions, control_files):
                tokens = self.encode_caption_for_cache(text_encoder, caption, control_file)
                o = text_encoder.encode_from_tokens_scheduled(tokens)
                text_embeds = o[0][0].to(self.dtype)
                extra = o[0][1]
                if 'attention_mask' in extra:
                    attention_mask = extra['attention_mask'].to(torch.int64)
                else:
                    attention_mask = torch.ones(text_embeds.shape[:2], dtype=torch.int64, device=text_embeds.device)
                embeds_list.append(text_embeds[0])
                mask_list.append(attention_mask[0])
            return {f'text_embeds_{te_idx}': embeds_list, f'attention_mask_{te_idx}': mask_list}

        return fn

    def get_reference_metadata(self):
        meta = super().get_reference_metadata()
        meta.update({
            'control_family': 'krea2_native_comfy',
            'comfy_workflow': (
                'TextEncodeQwenImageEditPlus(vae,image1) + FluxKontextMultiReferenceLatentMethod('
                + ('index_timestep_zero' if self.reference_timestep_mode == 'zero' else 'index') + ')'
                if self.vl_grounding else
                'CLIPTextEncode + ReferenceLatent + FluxKontextMultiReferenceLatentMethod('
                + ('index_timestep_zero' if self.reference_timestep_mode == 'zero' else 'index') + ')'
            ),
            'vl_prompt_layout': 'qwen_edit_plus_picture_n' if self.vl_grounding else 'none',
            'vl_image_max_pixels': str(VL_TOTAL_PIXELS) if self.vl_grounding else '0',
            'reference_pixels': 'area_1mp_round8_own_grid',
        })
        return meta


# Keep the base-class check honest: this pipeline must never fall back to the
# target-sized reference assertion of the generic reference pipeline.
assert Krea2NativePipeline.prepare_reference_latents is not Krea2ReferencePipeline.prepare_reference_latents
