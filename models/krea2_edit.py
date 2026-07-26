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
import peft
from PIL import Image

from models.krea2_reference import Krea2ReferencePipeline
from comfy.text_encoders.krea2 import KREA2_TEMPLATE
from comfy import model_management
from utils.common import AUTOCAST_DTYPE, is_main_process

VISION_BLOCK = '<|vision_start|><|image_pad|><|vision_end|>'


def build_vl_image_prompt(num_images, label='Picture'):
    """Named vision placeholders, one per reference (EditPlus / ostris layout).

    ``label`` exists for the multi-reference path: when the captions address
    the references as "image 1"/"image 2", labelling the vision blocks
    "Picture 1:" forces the model to bridge two vocabularies for the same
    thing. Passing ``label='image'`` makes both channels speak one.
    """
    return ''.join(f'{label} {i + 1}: {VISION_BLOCK}' for i in range(num_images))


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


def prepare_vl_image_longest_side(image_path, longest_side):
    """conradlocke grounding_px semantics: cap the LONGEST side (area
    resample, downscale-only), instead of the ostris area budget."""
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
    if longest_side and max(image.height, image.width) > longest_side:
        scale = longest_side / max(image.height, image.width)
        new_h = max(round(image.height * scale), 28)
        new_w = max(round(image.width * scale), 28)
        pixels = F.interpolate(
            pixels.permute(2, 0, 1).unsqueeze(0),
            size=(new_h, new_w),
            mode='area',
        ).squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0)
    return pixels.unsqueeze(0)


class Krea2EditPipeline(Krea2ReferencePipeline):
    name = 'krea2_edit'
    config_section = 'krea2_edit'
    adapter_allowed_key_substrings = ('.blocks.', '.txtfusion.')

    def __init__(self, config):
        super().__init__(config)
        section = config.get(self.config_section, {})
        self.vl_image_max_pixels = int(section.get('vl_image_max_pixels', 384 * 384))
        if self.vl_image_max_pixels < 28 * 28:
            raise ValueError('vl_image_max_pixels must be at least 28*28')
        # 'picture_n' = ostris layout ("Picture 1: <vision>"); 'plain' =
        # conradlocke layout (bare vision block before the caption).
        self.vl_prompt_style = section.get('vl_prompt_style', 'picture_n')
        if self.vl_prompt_style not in ('picture_n', 'plain'):
            raise ValueError("vl_prompt_style must be 'picture_n' or 'plain'")
        # Rótulo dos blocos de visão. Default 'Picture' = contrato ostris.
        # O caminho multi-ref usa 'image' para casar com as captions, que
        # endereçam as referências como "image 1"/"image 2".
        self.vl_image_label = section.get('vl_image_label', 'Picture')
        # When set, the VL copy is capped by LONGEST SIDE (area resample),
        # conradlocke's grounding_px semantics, instead of the ostris area
        # budget above.
        self.vl_longest_side = section.get('vl_longest_side', None)
        if self.vl_longest_side is not None:
            self.vl_longest_side = int(self.vl_longest_side)
        # [min, max]: per-sample grounding-resolution jitter (conradlocke v1.1
        # trained with 384-768). Text embeddings are cached, so the jitter is
        # sampled ONCE per file (deterministic hash of the path), giving the
        # dataset a spread of grounding resolutions rather than per-step noise.
        self.vl_grounding_jitter = section.get('vl_grounding_jitter', None)
        if self.vl_grounding_jitter is not None:
            lo, hi = (int(v) for v in self.vl_grounding_jitter)
            if not 28 <= lo <= hi:
                raise ValueError('vl_grounding_jitter must be [min, max] with 28 <= min <= max')
            self.vl_grounding_jitter = (lo, hi)
        # Fraction of samples trained with an EMPTY caption while keeping the
        # reference grounded in BOTH branches (vision block + VAE tokens) —
        # exactly the unconditional used by CFG at inference. Distinct from
        # condition_dropout (which would drop the reference itself and stays
        # forbidden).
        self.caption_dropout = float(section.get('caption_dropout', 0.0))
        if not 0.0 <= self.caption_dropout <= 0.5:
            raise ValueError('caption_dropout must be between 0.0 and 0.5')
        if self.condition_token_stride != 1:
            raise ValueError(
                'krea2_edit follows the canonical Krea Edit contract; condition_token_stride must be 1'
            )
        # The public edit training never drops references, and a nonzero value
        # here would only zero the VAE branch while Qwen3-VL keeps seeing the
        # reference — an incoherent partial dropout. Reject it outright.
        self.condition_dropout = 0.0
        if float(section.get('condition_dropout', 0.0)) != 0.0:
            raise ValueError(
                'krea2_edit does not support condition_dropout: the public Krea Edit '
                'training never drops references, and dropping only the VAE branch '
                'while the Qwen3-VL grounding remains would be inconsistent'
            )

    def configure_adapter(self, adapter_config):
        """Match the public Krea Edit LoRA coverage: DiT blocks + text fusion.

        Public dual-conditioning LoRAs (Krea2OstrisEdit family; verified from
        the ostris style-reference and conradlocke identity-edit safetensors
        headers, 448 + 64 tensors) adapt every linear in the 28
        SingleStreamBlocks plus the 4 TextFusionBlocks (layerwise + refiner),
        excluding the layer projector and txtmlp. The text fusion is where the
        stacked Qwen3-VL hidden states — including the reference's vision
        tokens — are collapsed, so it must be trainable for the visual
        grounding to adapt.
        """
        target_model = self.diffusion_model
        target_linear_modules = set()
        for name, module in target_model.named_modules():
            if module.__class__.__name__ not in ('SingleStreamBlock', 'TextFusionTransformer'):
                continue
            for full_name, submodule in module.named_modules(prefix=name):
                if not isinstance(submodule, nn.Linear):
                    continue
                if full_name.endswith('projector'):
                    continue
                target_linear_modules.add(full_name)
        targets = sorted(target_linear_modules)
        if not targets:
            raise RuntimeError('No Krea2 DiT linear modules found for the edit adapter')

        adapter_type = adapter_config['type']
        if adapter_type == 'lora':
            peft_config = peft.LoraConfig(
                r=adapter_config['rank'],
                lora_alpha=adapter_config['alpha'],
                lora_dropout=adapter_config['dropout'],
                bias='none',
                target_modules=targets,
            )
        elif adapter_type == 'lokr':
            peft_config = peft.LoKrConfig(
                r=adapter_config['rank'],
                decompose_factor=adapter_config['decompose_factor'],
                alpha=adapter_config['alpha'],
                rank_dropout=adapter_config['rank_dropout'],
                target_modules=targets,
            )
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
        self.peft_config = peft_config
        self.lora_model = peft.get_peft_model(target_model, peft_config)
        if is_main_process():
            fusion = sum(1 for name in targets if 'txtfusion' in name)
            print(
                f'[{self.name}] dual-conditioning LoRA targets: {len(targets)} linears '
                f'({len(targets) - fusion} in DiT blocks, {fusion} in text fusion)'
            )
            self.lora_model.print_trainable_parameters()
        for name, parameter in target_model.named_parameters():
            parameter.original_name = name
            if parameter.requires_grad:
                parameter.data = parameter.data.to(adapter_config['dtype'])

    def _jittered_grounding_side(self, file):
        import hashlib
        lo, hi = self.vl_grounding_jitter
        digest = int(hashlib.sha256(str(file).encode()).hexdigest(), 16)
        return lo + digest % (hi - lo + 1)

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
                    if self.vl_grounding_jitter:
                        images = [
                            prepare_vl_image_longest_side(
                                file, self._jittered_grounding_side(file)
                            )
                            for file in files
                        ]
                    elif self.vl_longest_side:
                        images = [
                            prepare_vl_image_longest_side(file, self.vl_longest_side)
                            for file in files
                        ]
                    else:
                        images = [prepare_vl_image(file, self.vl_image_max_pixels) for file in files]
                    if self.vl_prompt_style == 'plain':
                        text = VISION_BLOCK * len(images) + caption
                    else:
                        text = build_vl_image_prompt(len(images), self.vl_image_label) + caption

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
            'vl_prompt_layout': (
                'plain_vision_blocks' if self.vl_prompt_style == 'plain'
                else 'picture_n_vision_blocks'
            ),
            'vl_longest_side': str(self.vl_longest_side or 0),
            'vl_grounding_jitter': (
                f'{self.vl_grounding_jitter[0]}-{self.vl_grounding_jitter[1]}'
                if self.vl_grounding_jitter else 'none'
            ),
            'caption_dropout': str(self.caption_dropout),
            'vl_reference_in_uncond': 'true',
            'lora_targets': 'blocks+txtfusion',
        }
