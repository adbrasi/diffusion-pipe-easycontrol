"""Krea 2 Edit: dual-conditioning (VAE + Qwen3-VL visual) reference training.

Implements the public Krea Edit contract (Krea2OstrisEdit / ai-toolkit
``edit=true`` / ComfyUI grounded encode, audited 2026-07-15) on top of the
fork's clean-reference infrastructure. Every reference conditions the model
through two paths at once::

    reference → VAE → DiT                          (clean latents, t=0, RoPE frame 1)
    reference + caption → Qwen3-VL (visual) → DiT  (image-grounded text embeddings)

Compared to ``krea2_ic_lora`` (VAE-only), the reference image is additionally
serialized into the Qwen3-VL prompt as a named vision block::

    Picture 1: <|vision_start|><|image_pad|><|vision_end|>

inside the standard Krea 2 conditioning template, so the text embeddings carry
image-grounded semantics. Sizing follows the public contract: the VL copy is
downscaled (aspect preserved, never upscaled) to fit ``vl_image_max_pixels``
(default 384*384); high-resolution detail flows through the VAE branch.

CFG semantics at inference match the public pipeline: the unconditional
embedding keeps the reference grounding and guidance contrasts the prompt only
(``v = uncond + scale * (cond - uncond)``).
"""

import math

import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

from models.krea2_reference import Krea2ReferencePipeline
from comfy.text_encoders.krea2 import KREA2_TEMPLATE
from comfy import model_management
from utils.common import AUTOCAST_DTYPE

VISION_BLOCK = '<|vision_start|><|image_pad|><|vision_end|>'


def build_vl_image_prompt(num_images):
    """Named vision placeholders, one per reference (EditPlus / ostris layout)."""
    return ''.join(f'Picture {i + 1}: {VISION_BLOCK}' for i in range(num_images))


def prepare_vl_image(image_path, max_pixels):
    """Load a reference image for the Qwen3-VL pass.

    White background under transparency, aspect-preserving downscale-only to
    fit ``max_pixels`` total area (never upscaled, 28px floor per side),
    returned in the ComfyUI image layout ``(1, H, W, C)`` in ``[0, 1]``.
    """
    image = Image.open(image_path)
    if image.mode == 'RGBA' or ('transparency' in image.info and image.mode != 'RGB'):
        rgba = image.convert('RGBA')
        canvas = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
        canvas.alpha_composite(rgba)
        image = canvas.convert('RGB')
    else:
        image = image.convert('RGB')

    pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    pixels = pixels.reshape(image.height, image.width, 3).to(torch.float32) / 255.0

    scale = min(1.0, math.sqrt(max_pixels / (image.height * image.width)))
    new_h = max(round(image.height * scale), 28)
    new_w = max(round(image.width * scale), 28)
    if (new_h, new_w) != (image.height, image.width):
        pixels = F.interpolate(
            pixels.permute(2, 0, 1).unsqueeze(0),
            size=(new_h, new_w),
            mode='bicubic',
            antialias=True,
        ).squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)
    return pixels.unsqueeze(0)


class Krea2EditPipeline(Krea2ReferencePipeline):
    name = 'krea2_edit'
    config_section = 'krea2_edit'

    def __init__(self, config):
        super().__init__(config)
        section = config.get(self.config_section, {})
        self.vl_image_max_pixels = int(section.get('vl_image_max_pixels', 384 * 384))
        if self.vl_image_max_pixels < 28 * 28:
            raise ValueError('vl_image_max_pixels must be at least 28*28')
        if self.condition_token_stride != 1:
            raise ValueError(
                'krea2_edit follows the canonical Krea Edit contract; condition_token_stride must be 1'
            )
        # The public edit training never drops references; nonzero dropout on
        # the VAE branch stays available but is a local extension.
        if 'condition_dropout' not in section:
            self.condition_dropout = 0.0

    def get_call_text_encoder_fn(self, text_encoder):
        te_idx = None
        for i, te in enumerate(self.text_encoders):
            if text_encoder == te:
                te_idx = i
                break
        if te_idx is None:
            raise RuntimeError('Unknown text encoder')

        @torch.inference_mode()
        def fn(captions: list[str], is_video: list[bool], control_files):
            assert not any(is_video)
            if control_files is None:
                control_files = [None] * len(captions)
            if len(control_files) != len(captions):
                raise ValueError(
                    f'Got {len(captions)} captions but {len(control_files)} control files'
                )

            embeds_list = []
            mask_list = []
            for caption, control_file in zip(captions, control_files):
                images = []
                text = caption
                if control_file is not None:
                    files = control_file if isinstance(control_file, (list, tuple)) else [control_file]
                    images = [prepare_vl_image(file, self.vl_image_max_pixels) for file in files]
                    text = build_vl_image_prompt(len(images)) + caption

                # llama_template must be passed explicitly: with images present
                # the Qwen3-VL tokenizer would otherwise switch to its
                # user-only image template instead of the Krea system template.
                tokens = text_encoder.tokenize(text, images=images, llama_template=KREA2_TEMPLATE)
                o = text_encoder.encode_from_tokens_scheduled(tokens)

                text_embeds = o[0][0].to(self.dtype)
                extra = o[0][1]
                if 'attention_mask' in extra:
                    attention_mask = extra['attention_mask'].to(torch.int64)
                else:
                    # Krea2 TE removes the attention_mask when it is all 1s.
                    attention_mask = torch.ones(
                        text_embeds.shape[:2], dtype=torch.int64, device=text_embeds.device
                    )
                embeds_list.append(text_embeds[0])
                mask_list.append(attention_mask[0])

            # Per-row natural-length tensors: get_conds pads to the batch max.
            return {
                f'text_embeds_{te_idx}': embeds_list,
                f'attention_mask_{te_idx}': mask_list,
            }

        return fn

    @torch.no_grad()
    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def prepare_sample_test(self, prompt, negative_prompt='', cfg=1, control_files=None):
        inputs = {}
        inputs_uncond = {}
        for te in self.get_text_encoders():
            if isinstance(te, nn.Module):
                te = te.to('cuda')
            else:
                te.load_model_if_needed()
            call_text_encoder_fn = self.get_call_text_encoder_fn(te)
            inputs.update(call_text_encoder_fn([prompt], [False], [control_files]))
            if cfg > 1:
                # Canonical Krea Edit CFG: the unconditional embedding keeps
                # the reference grounding; guidance contrasts the prompt only.
                inputs_uncond.update(call_text_encoder_fn([negative_prompt], [False], [control_files]))
            if isinstance(te, nn.Module):
                te = te.to('cpu')
            else:
                model_management.unload_all_models()
        self.conds = tuple(tensor.cuda() for tensor in self.get_conds(inputs))
        if cfg > 1:
            self.unconds = tuple(tensor.cuda() for tensor in self.get_conds(inputs_uncond))
        self.sample_cfg = cfg

    def get_reference_metadata(self):
        return {
            'control_family': 'krea2_edit_dual',
            'vl_conditioning': 'qwen3vl_image_grounded',
            'vl_image_max_pixels': str(self.vl_image_max_pixels),
            'vl_prompt_layout': 'picture_n_vision_blocks',
            'vl_reference_in_uncond': 'true',
        }
