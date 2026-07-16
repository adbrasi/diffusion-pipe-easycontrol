"""One-reference Krea 2 Edit inference nodes for ComfyUI.

The implementation mirrors diffusion-pipe's ``krea2_edit`` training contract:

* the reference image is encoded by the VAE and appended after the noisy
  target as clean tokens at model timestep zero and RoPE frame one;
* the same reference image is shown to Qwen3-VL for both positive and negative
  conditioning, so CFG changes the instruction rather than the reference;
* only target tokens are returned by the patched diffusion model.

The recommended setup node keeps the image, both prompts, VAE latent and model
patch in one graph operation. The modular nodes expose the same pieces for
advanced workflows without encoding the VAE reference twice.
"""

from dataclasses import dataclass
import math

import torch
from einops import rearrange

import comfy.conds
import comfy.ldm.common_dit
import comfy.model_management
import comfy.model_sampling
import comfy.utils
import node_helpers
from comfy.ldm.flux.layers import timestep_embedding
from comfy.text_encoders.krea2 import KREA2_TEMPLATE


REFERENCE_TYPE = "CTXRUSH_KREA2_REFERENCE"
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
DEFAULT_VL_MAX_PIXELS = 384 * 384
PUBLIC_REFERENCE_MAX_PIXELS = 1024 * 1024
REFERENCE_SNAP = 16

# Krea 2 Raw official mu interpolation in image-token space: 256px -> mu 0.5,
# 1280px -> mu 1.15 (same constants as tools/krea2_sampling.py). ComfyUI's
# stock Krea 2 config uses a fixed mu of 1.15, which matches Turbo but roughly
# doubles the effective shift for Raw at 672x384.
KREA2_MU_TOKENS_MIN = 256
KREA2_MU_TOKENS_MAX = 6400
KREA2_MU_MIN = 0.5
KREA2_MU_MAX = 1.15
KREA2_TURBO_MU = 1.15


def _krea2_raw_mu(width, height):
    tokens = (width // 16) * (height // 16)
    slope = (KREA2_MU_MAX - KREA2_MU_MIN) / (KREA2_MU_TOKENS_MAX - KREA2_MU_TOKENS_MIN)
    return slope * tokens + (KREA2_MU_MIN - slope * KREA2_MU_TOKENS_MIN)


def _apply_krea2_sampling(patched_model, mu):
    """Set the model's flow shift to ``mu``, mirroring the validated runner."""

    class _Krea2ModelSampling(
        comfy.model_sampling.ModelSamplingFlux, comfy.model_sampling.CONST
    ):
        pass

    model_sampling = _Krea2ModelSampling(patched_model.model.model_config)
    model_sampling.set_parameters(shift=mu)
    patched_model.add_object_patch("model_sampling", model_sampling)
    return patched_model


@dataclass(frozen=True)
class Krea2Reference:
    """A paired visual/Qwen reference encoded once for both CFG branches."""

    vl_image: torch.Tensor
    latent: torch.Tensor
    fit_mode: str
    target_width: int
    target_height: int


def _require_single_image(image):
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError(
            "CtxRush Krea 2 Edit expects one ComfyUI IMAGE in BHWC layout."
        )
    if image.shape[0] != 1:
        raise ValueError(
            "CtxRush Krea 2 Edit supports exactly one reference image. "
            "Use a batch size of one on Load Image."
        )
    return image[..., :3]


def _fit_area(image, max_pixels, snap=1):
    """Downscale only, preserving aspect ratio and optionally snapping size."""
    samples = image.movedim(-1, 1)
    height, width = samples.shape[-2:]
    scale = min(1.0, math.sqrt(max_pixels / (height * width)))
    new_width = max(round(width * scale / snap) * snap, snap)
    new_height = max(round(height * scale / snap) * snap, snap)
    if (new_height, new_width) == (height, width):
        return image
    samples = comfy.utils.common_upscale(
        samples, new_width, new_height, "area", "disabled"
    )
    return samples.movedim(1, -1)


def _fit_vl(image, max_pixels):
    """Resize the Qwen3-VL copy exactly like training's prepare_vl_image:
    aspect-preserving downscale-only, bicubic with antialias, 28px floor per
    side (models/krea2_edit.py). The generic _fit_area (area kernel, 1px
    floor) stays for the VAE branch only."""
    samples = image.movedim(-1, 1)
    height, width = samples.shape[-2:]
    scale = min(1.0, math.sqrt(max_pixels / (height * width)))
    new_height = max(round(height * scale), 28)
    new_width = max(round(width * scale), 28)
    if (new_height, new_width) == (height, width):
        return image
    samples = torch.nn.functional.interpolate(
        samples.float(),
        size=(new_height, new_width),
        mode="bicubic",
        antialias=True,
    ).clamp(0.0, 1.0).to(image.dtype)
    return samples.movedim(1, -1)


def _crop_fit(image, width, height):
    """Match diffusion-pipe's same-bucket center-crop reference contract."""
    samples = image.movedim(-1, 1)
    source_height, source_width = samples.shape[-2:]
    scale = max(width / source_width, height / source_height)
    resized_width = max(round(source_width * scale), width)
    resized_height = max(round(source_height * scale), height)
    samples = comfy.utils.common_upscale(
        samples, resized_width, resized_height, "lanczos", "disabled"
    )
    top = (resized_height - height) // 2
    left = (resized_width - width) // 2
    return samples[:, :, top : top + height, left : left + width].movedim(1, -1)


def _build_reference(
    vae,
    image,
    width,
    height,
    fit_mode="training_crop",
    vl_image_max_pixels=DEFAULT_VL_MAX_PIXELS,
):
    image = _require_single_image(image)
    if width % REFERENCE_SNAP or height % REFERENCE_SNAP:
        raise ValueError("Target width and height must be multiples of 16.")

    vl_image = _fit_vl(image, vl_image_max_pixels)
    if fit_mode == "training_crop":
        vae_image = _crop_fit(image, width, height)
    elif fit_mode == "preserve_aspect_1mp":
        vae_image = _fit_area(
            image, PUBLIC_REFERENCE_MAX_PIXELS, snap=REFERENCE_SNAP
        )
    else:
        raise ValueError(f"Unknown reference fit mode: {fit_mode}")

    latent = vae.encode(vae_image)
    return Krea2Reference(
        vl_image=vl_image,
        latent=latent,
        fit_mode=fit_mode,
        target_width=width,
        target_height=height,
    )


def _encode_conditioning(clip, prompt, reference):
    text = f"Picture 1: {VISION_BLOCK}{prompt}"
    try:
        tokens = clip.tokenize(
            text,
            images=[reference.vl_image],
            llama_template=KREA2_TEMPLATE,
        )
        conditioning = clip.encode_from_tokens_scheduled(tokens)
    except Exception as error:
        raise RuntimeError(
            "Krea 2 visual conditioning failed. Use a Krea 2 CLIP/text encoder "
            "checkpoint that includes the Qwen3-VL visual.* weights."
        ) from error

    return node_helpers.conditioning_set_values(
        conditioning,
        {"reference_latents": [reference.latent]},
        append=True,
    )


def _empty_krea_latent(width, height, batch_size):
    latent = torch.zeros(
        [batch_size, 16, height // 8, width // 8],
        device=comfy.model_management.intermediate_device(),
        dtype=comfy.model_management.intermediate_dtype(),
    )
    return {"samples": latent, "downscale_ratio_spacial": 8}


def _pack_reference(dit, reference, batch_size, device, dtype):
    """Patchify one processed reference and assign fixed RoPE frame one."""
    if reference.ndim == 5:
        ref_batch, channels, frames, height, width = reference.shape
        if frames != 1:
            raise ValueError(
                "CtxRush Krea 2 Edit supports one reference frame, "
                f"but received {frames}."
            )
        reference = reference.reshape(
            ref_batch * frames, channels, height, width
        )
    if reference.ndim != 4:
        raise ValueError(
            "The processed Krea 2 reference latent must be BCHW or BCTHW."
        )

    reference = comfy.ldm.common_dit.pad_to_patch_size(
        reference.to(device=device, dtype=dtype), (dit.patch, dit.patch)
    )
    reference = comfy.utils.repeat_to_batch_size(reference, batch_size)
    grid_height = reference.shape[-2] // dit.patch
    grid_width = reference.shape[-1] // dit.patch
    tokens = rearrange(
        reference,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=dit.patch,
        pw=dit.patch,
    )

    positions = torch.zeros(
        grid_height,
        grid_width,
        3,
        device=device,
        dtype=torch.float32,
    )
    positions[..., 0] = 1.0
    positions[..., 1] = torch.arange(
        grid_height, device=device, dtype=torch.float32
    )[:, None]
    positions[..., 2] = torch.arange(
        grid_width, device=device, dtype=torch.float32
    )[None, :]
    positions = positions.reshape(1, grid_height * grid_width, 3).repeat(
        batch_size, 1, 1
    )
    return tokens, positions


def _block_with_clean_reference(
    block,
    hidden_states,
    target_timestep,
    clean_timestep,
    reference_start,
    frequencies,
    transformer_options,
):
    """Apply Krea modulation at sampled t to target/text and t=0 to reference."""
    target_mod = block.mod(target_timestep)
    reference_mod = block.mod(clean_timestep)

    def modulate(states, scale_index, shift_index):
        return torch.cat(
            (
                (1 + target_mod[scale_index]) * states[:, :reference_start]
                + target_mod[shift_index],
                (1 + reference_mod[scale_index]) * states[:, reference_start:]
                + reference_mod[shift_index],
            ),
            dim=1,
        )

    def gate(states, gate_index):
        return torch.cat(
            (
                target_mod[gate_index] * states[:, :reference_start],
                reference_mod[gate_index] * states[:, reference_start:],
            ),
            dim=1,
        )

    attention_input = modulate(block.prenorm(hidden_states), 0, 1)
    attention_output = block.attn(
        attention_input,
        frequencies,
        None,
        transformer_options=transformer_options,
    )
    hidden_states = hidden_states + gate(attention_output, 2)
    mlp_input = modulate(block.postnorm(hidden_states), 3, 4)
    hidden_states = hidden_states + gate(block.mlp(mlp_input), 5)
    return hidden_states


def _forward_with_reference(
    dit,
    x,
    timesteps,
    context,
    reference_latents,
    transformer_options,
):
    if len(reference_latents) != 1:
        raise ValueError(
            "This CtxRush adapter was trained with exactly one reference image."
        )

    temporal = x.ndim == 5
    if temporal:
        batch_5d, channels_5d, frames_5d, height_5d, width_5d = x.shape
        x = x.reshape(batch_5d * frames_5d, channels_5d, height_5d, width_5d)

    batch_size, _, original_height, original_width = x.shape
    patch = dit.patch
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    grid_height = x.shape[-2] // patch
    grid_width = x.shape[-1] // patch

    context = dit._unpack_context(context)
    target_tokens = rearrange(
        x,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=patch,
        pw=patch,
    )
    reference_tokens, reference_positions = _pack_reference(
        dit,
        reference_latents[0],
        batch_size,
        x.device,
        x.dtype,
    )
    image_tokens = dit.first(torch.cat((target_tokens, reference_tokens), dim=1))

    target_features = dit.tmlp(
        timestep_embedding(timesteps, dit.tdim).unsqueeze(1).to(image_tokens.dtype)
    )
    clean_features = dit.tmlp(
        timestep_embedding(torch.zeros_like(timesteps), dit.tdim)
        .unsqueeze(1)
        .to(image_tokens.dtype)
    )
    target_timestep = dit.tproj(target_features)
    clean_timestep = dit.tproj(clean_features)

    context = dit.txtfusion(
        context, mask=None, transformer_options=transformer_options
    )
    context = dit.txtmlp(context)
    text_length = context.shape[1]
    target_length = target_tokens.shape[1]
    reference_start = text_length + target_length
    hidden_states = torch.cat((context, image_tokens), dim=1)

    text_positions = torch.zeros(
        batch_size,
        text_length,
        3,
        device=x.device,
        dtype=torch.float32,
    )
    target_positions = torch.zeros(
        grid_height,
        grid_width,
        3,
        device=x.device,
        dtype=torch.float32,
    )
    target_positions[..., 1] = torch.arange(
        grid_height, device=x.device, dtype=torch.float32
    )[:, None]
    target_positions[..., 2] = torch.arange(
        grid_width, device=x.device, dtype=torch.float32
    )[None, :]
    target_positions = target_positions.reshape(
        1, grid_height * grid_width, 3
    ).repeat(batch_size, 1, 1)
    frequencies = dit.pe_embedder(
        torch.cat((text_positions, target_positions, reference_positions), dim=1)
    )

    for block in dit.blocks:
        hidden_states = _block_with_clean_reference(
            block,
            hidden_states,
            target_timestep,
            clean_timestep,
            reference_start,
            frequencies,
            transformer_options,
        )

    output = dit.last(hidden_states, target_features)
    output = output[:, text_length:reference_start]
    output = rearrange(
        output,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=grid_height,
        w=grid_width,
        ph=patch,
        pw=patch,
        c=dit.channels,
    )
    output = output[:, :, :original_height, :original_width]
    if temporal:
        output = output.reshape(
            batch_5d, frames_5d, dit.channels, original_height, original_width
        ).movedim(1, 2)
    return output


def _patch_model(model):
    patched = model.clone()
    base_model = patched.model
    dit = patched.get_model_object("diffusion_model")
    required = (
        "patch",
        "channels",
        "blocks",
        "txtfusion",
        "txtmlp",
        "pe_embedder",
        "_unpack_context",
    )
    missing = [name for name in required if not hasattr(dit, name)]
    if dit.__class__.__name__ != "SingleStreamDiT" or missing:
        detail = f"; missing attributes: {', '.join(missing)}" if missing else ""
        raise ValueError(
            "CtxRush Krea 2 Edit Model Patch requires the ComfyUI Krea 2 "
            f"SingleStreamDiT, got {dit.__class__.__name__}{detail}."
        )

    original_extra_conds = base_model.extra_conds
    original_extra_conds_shapes = base_model.extra_conds_shapes
    original_forward = dit.forward

    def extra_conds(**kwargs):
        output = original_extra_conds(**kwargs)
        references = kwargs.get("reference_latents")
        if references:
            output["ctxrush_reference_latents"] = comfy.conds.CONDList(
                [base_model.process_latent_in(latent) for latent in references]
            )
        return output

    def extra_conds_shapes(**kwargs):
        output = original_extra_conds_shapes(**kwargs)
        references = kwargs.get("reference_latents")
        if references:
            total_elements = sum(math.prod(reference.size()) for reference in references)
            output["ctxrush_reference_latents"] = [1, 16, total_elements // 16]
        return output

    def forward(
        x,
        timesteps,
        context,
        attention_mask=None,
        transformer_options=None,
        ctxrush_reference_latents=None,
        **kwargs,
    ):
        options = {} if transformer_options is None else transformer_options
        if not ctxrush_reference_latents:
            return original_forward(
                x,
                timesteps,
                context,
                attention_mask=attention_mask,
                transformer_options=options,
                **kwargs,
            )
        return _forward_with_reference(
            dit,
            x,
            timesteps,
            context,
            ctxrush_reference_latents,
            options,
        )

    patched.add_object_patch("extra_conds", extra_conds)
    patched.add_object_patch("extra_conds_shapes", extra_conds_shapes)
    patched.add_object_patch("diffusion_model.forward", forward)
    return patched


REFERENCE_INPUTS = {
    "vae": ("VAE", {"tooltip": "Qwen Image VAE used by Krea 2."}),
    "reference": (
        "IMAGE",
        {
            "tooltip": (
                "The single source/control image. It is used by both the VAE "
                "detail path and the Qwen3-VL semantic path."
            )
        },
    ),
    "width": (
        "INT",
        {
            "default": 672,
            "min": 64,
            "max": 8192,
            "step": 16,
            "tooltip": "Target generation width. Use the evaluated training bucket first.",
        },
    ),
    "height": (
        "INT",
        {
            "default": 384,
            "min": 64,
            "max": 8192,
            "step": 16,
            "tooltip": "Target generation height. Use the evaluated training bucket first.",
        },
    ),
    "reference_fit": (
        ["training_crop", "preserve_aspect_1mp"],
        {
            "default": "training_crop",
            "tooltip": (
                "training_crop center-crops the reference to the target bucket and "
                "matches this diffusion-pipe LoRA. preserve_aspect_1mp matches public "
                "Ostris/ai-toolkit edit LoRAs but is out of distribution for this run."
            ),
        },
    ),
    "vl_image_max_pixels": (
        "INT",
        {
            "default": DEFAULT_VL_MAX_PIXELS,
            "min": 784,
            "max": 1048576,
            "step": 784,
            "tooltip": (
                "Pixel budget shown to Qwen3-VL. 147456 (384x384 area) is the "
                "training value; high-resolution detail comes from the VAE path."
            ),
        },
    ),
}


class CtxRushKrea2ReferenceEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": dict(REFERENCE_INPUTS)}

    RETURN_TYPES = (REFERENCE_TYPE,)
    RETURN_NAMES = ("reference",)
    FUNCTION = "encode"
    CATEGORY = "CtxRush/Krea 2 Edit"
    DESCRIPTION = (
        "Encode one Krea 2 Edit reference once for both CFG branches. The "
        "output contains the VAE latent and the downscaled Qwen3-VL image."
    )

    def encode(
        self,
        vae,
        reference,
        width,
        height,
        reference_fit,
        vl_image_max_pixels,
    ):
        return (
            _build_reference(
                vae,
                reference,
                width,
                height,
                reference_fit,
                vl_image_max_pixels,
            ),
        )


class CtxRushKrea2EditCFGEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": (
                    "CLIP",
                    {
                        "tooltip": (
                            "Krea 2 Qwen3-VL text encoder with visual.* weights."
                        )
                    },
                ),
                "reference": (
                    REFERENCE_TYPE,
                    {
                        "tooltip": (
                            "Reference produced by CtxRush Krea 2 Reference Encode."
                        )
                    },
                ),
                "positive_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": (
                            "Describe the target/next scene, not the source image."
                        ),
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": (
                            "Negative instruction. The reference remains grounded in "
                            "this branch so CFG contrasts text only."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "CtxRush/Krea 2 Edit"
    DESCRIPTION = (
        "Encode positive and negative Krea 2 conditioning with the same visual "
        "reference attached to both branches."
    )

    def encode(self, clip, reference, positive_prompt, negative_prompt=""):
        return (
            _encode_conditioning(clip, positive_prompt, reference),
            _encode_conditioning(clip, negative_prompt, reference),
        )


class CtxRushKrea2EditModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "Krea 2 model after loading the edit LoRA. The patch is a "
                            "no-op when conditioning contains no reference."
                        )
                    },
                ),
                "model_variant": (
                    ["raw", "turbo"],
                    {
                        "default": "raw",
                        "tooltip": (
                            "Raw derives the flow shift (mu) from the output "
                            "resolution like the official Krea sampler; Turbo "
                            "keeps the fixed mu 1.15."
                        ),
                    },
                ),
                "width": (
                    "INT",
                    {"default": 672, "min": 64, "max": 4096, "step": 16,
                     "tooltip": "Output width, used to derive the Raw flow shift."},
                ),
                "height": (
                    "INT",
                    {"default": 384, "min": 64, "max": 4096, "step": 16,
                     "tooltip": "Output height, used to derive the Raw flow shift."},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "CtxRush/Krea 2 Edit"
    DESCRIPTION = (
        "Enable the clean-reference sequence used by diffusion-pipe krea2_edit: "
        "text, noisy target, clean reference at t=0/RoPE frame 1. Also sets the "
        "resolution-dependent Raw flow shift (ComfyUI's stock Krea 2 config uses "
        "Turbo's fixed mu 1.15, which over-shifts Raw)."
    )

    def patch(self, model, model_variant="raw", width=672, height=384):
        mu = _krea2_raw_mu(width, height) if model_variant == "raw" else KREA2_TURBO_MU
        return (_apply_krea2_sampling(_patch_model(model), mu),)


class CtxRushKrea2EditSetup:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "Krea 2 model after loading the edit LoRA."},
                ),
                "clip": (
                    "CLIP",
                    {
                        "tooltip": (
                            "Krea 2 Qwen3-VL text encoder with visual.* weights."
                        )
                    },
                ),
                **REFERENCE_INPUTS,
                "positive_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Describe the desired target/next scene.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": (
                            "Negative instruction; the visual reference is retained."
                        ),
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 64,
                        "tooltip": "Number of targets generated from the same reference.",
                    },
                ),
                "model_variant": (
                    ["raw", "turbo"],
                    {
                        "default": "raw",
                        "tooltip": (
                            "Returns safe sampler defaults: Raw=28 steps/CFG 5.5; "
                            "Turbo=8 steps/CFG 1.0."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "MODEL",
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
        "INT",
        "FLOAT",
    )
    RETURN_NAMES = ("model", "positive", "negative", "latent", "steps", "cfg")
    FUNCTION = "setup"
    CATEGORY = "CtxRush/Krea 2 Edit"
    DESCRIPTION = (
        "Recommended all-in-one Krea 2 Edit setup. Encodes one reference once, "
        "grounds both CFG branches, patches the model and creates the target latent."
    )

    def setup(
        self,
        model,
        clip,
        vae,
        reference,
        width,
        height,
        reference_fit,
        vl_image_max_pixels,
        positive_prompt,
        negative_prompt,
        batch_size,
        model_variant,
    ):
        encoded_reference = _build_reference(
            vae,
            reference,
            width,
            height,
            reference_fit,
            vl_image_max_pixels,
        )
        positive = _encode_conditioning(
            clip, positive_prompt, encoded_reference
        )
        negative = _encode_conditioning(
            clip, negative_prompt, encoded_reference
        )
        steps, cfg = (28, 5.5) if model_variant == "raw" else (8, 1.0)
        mu = _krea2_raw_mu(width, height) if model_variant == "raw" else KREA2_TURBO_MU
        patched_model = _apply_krea2_sampling(_patch_model(model), mu)
        return (
            patched_model,
            positive,
            negative,
            _empty_krea_latent(width, height, batch_size),
            steps,
            cfg,
        )


NODE_CLASS_MAPPINGS = {
    "CtxRushKrea2EditSetup": CtxRushKrea2EditSetup,
    "CtxRushKrea2ReferenceEncode": CtxRushKrea2ReferenceEncode,
    "CtxRushKrea2EditCFGEncode": CtxRushKrea2EditCFGEncode,
    "CtxRushKrea2EditModelPatch": CtxRushKrea2EditModelPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CtxRushKrea2EditSetup": "CtxRush - Krea 2 Edit Setup",
    "CtxRushKrea2ReferenceEncode": "CtxRush - Krea 2 Reference Encode",
    "CtxRushKrea2EditCFGEncode": "CtxRush - Krea 2 Edit CFG Encode",
    "CtxRushKrea2EditModelPatch": "CtxRush - Krea 2 Edit Model Patch",
}
