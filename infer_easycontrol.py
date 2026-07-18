#!/usr/bin/env python3
"""
EasyControl Anima inference — generate images with spatial control conditioning.

Requires: run from the diffusion-pipe directory (or set PYTHONPATH).

Usage:
  python infer_easycontrol.py \
    --dit anima-preview2.safetensors \
    --vae qwen_image_vae.safetensors \
    --llm qwen_3_06b_base.safetensors \
    --lora adapter_model.safetensors \
    --control_image canny.png \
    --prompt "1girl, standing" \
    --save_path ./outputs
"""

import argparse
import datetime
import os
import sys
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from einops import rearrange

# Ensure diffusion-pipe modules are importable
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# Pre-import namespace packages that lack __init__.py
import utils  # noqa: F401

from models.cosmos_predict2 import get_dit_config, WanVAE, vae_encode, _compute_text_embeddings
from models.cosmos_predict2_modeling import MiniTrainDIT
from models.easycontrol import (
    AnimaControlSelfAttn,
    generate_condition_rope,
    compute_condition_t_embedding,
)
from utils.common import load_state_dict, iterate_safetensors
from safetensors.torch import load_file as load_safetensors

import transformers
from transformers import AutoTokenizer, T5TokenizerFast
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

KEEP_IN_HIGH_PRECISION = ['x_embedder', 't_embedder', 't_embedding_norm', 'final_layer']


# ============================================================================
# Args
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="EasyControl Anima inference")
    p.add_argument("--dit", required=True, help="Anima DiT safetensors")
    p.add_argument("--vae", required=True, help="Qwen-Image VAE safetensors")
    p.add_argument("--llm", required=True, help="Qwen3-0.6B (dir or safetensors)")
    p.add_argument("--lora", default=None, help="LoRA safetensors (EasyControl or levzzz style)")
    p.add_argument("--control_image", default=None, help="Control image (required if --lora is set)")
    p.add_argument("--mode", default="auto", choices=["auto", "easycontrol", "levzzz", "ic_lora", "ic_lora_full", "ic_lora_routed", "routed_targetfirst", "ominicontrol", "ominicontrol_subject"],
                   help="Control mode: easycontrol, levzzz, ic_lora, ic_lora_full (ref_first), ominicontrol (spatial), ominicontrol_subject, auto (detect)")
    p.add_argument("--skip_adaln", action="store_true", default=False,
                   help="Skip LoRA on adaln_modulation layers (smoother strength control, matches LTX-2)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--width", type=int, default=512, help="Width (divisible by 16)")
    p.add_argument("--height", type=int, default=512, help="Height (divisible by 16)")
    p.add_argument("--steps", type=int, default=50, help="Sampling steps (Anima official: 50)")
    p.add_argument("--cfg", type=float, default=3.5, help="CFG scale (Anima official: 3.5)")
    p.add_argument("--flow_shift", type=float, default=5.0, help="Flow shift (Anima official: 5.0, community: 2.5-3.0)")
    p.add_argument("--control_strength", type=float, default=1.0, help="Control strength (0.0 = no control, 1.0 = full)")
    p.add_argument("--lora_strength", type=float, default=1.0,
                   help="PEFT LoRA merge scale for ic_lora/ominicontrol modes (0.0 = base model + ref, honest baseline)")
    p.add_argument("--ref_cfg", type=float, default=1.0,
                   help="Guidance da REFERENCIA (3 branches, independente do --cfg do texto). 0 = sem guidance de ref.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_path", default="./outputs", help="Output directory")
    p.add_argument("--vae_chunk_size", type=int, default=None)
    return p.parse_args()


# ============================================================================
# Timestep schedule (from sd-scripts hunyuan_image_utils)
# ============================================================================

def get_timesteps_sigmas(sampling_steps: int, shift: float, device: torch.device):
    sigmas = torch.linspace(1, 0, sampling_steps + 1)
    sigmas = (shift * sigmas) / (1 + (shift - 1) * sigmas)
    sigmas = sigmas.to(torch.float32)
    timesteps = (sigmas[:-1] * 1000).to(dtype=torch.float32, device=device)
    return timesteps, sigmas


def euler_step(latents, noise_pred, sigmas, step_i):
    return latents.float() - (sigmas[step_i] - sigmas[step_i + 1]) * noise_pred.float()


# ============================================================================
# Model loading
# ============================================================================

def load_dit(dit_path, device, dtype):
    print(f"Loading DiT: {dit_path}")
    state_dict = load_state_dict(dit_path)
    state_dict = {(k[4:] if k.startswith('net.') else k): v for k, v in state_dict.items()}
    dit_config = get_dit_config(state_dict)
    dit_config['use_llm_adapter'] = 'llm_adapter.out_proj.weight' in state_dict

    with init_empty_weights():
        dit = MiniTrainDIT(**dit_config)
    for name, p in dit.named_parameters():
        if name in state_dict:
            d = dtype if (p.ndim > 1 and not any(kw in name for kw in KEEP_IN_HIGH_PRECISION)) else torch.float32
            set_module_tensor_to_device(dit, name, device='cpu', dtype=d, value=state_dict[name])
    del state_dict
    dit.to(device).eval().requires_grad_(False)
    print(f"  DiT loaded: {dit.num_blocks} blocks, {dit.model_channels} dim")
    return dit


def load_vae(vae_path, device, dtype, chunk_size=None):
    print(f"Loading VAE: {vae_path}")
    vae = WanVAE(vae_pth=vae_path, device='cpu', dtype=dtype)
    vae.model.to(device)
    vae.mean = vae.mean.to(device)
    vae.std = vae.std.to(device)
    vae.scale = [vae.mean, 1.0 / vae.std]
    return vae


def load_text_encoder(llm_path, device, dtype):
    print(f"Loading text encoder: {llm_path}")
    if os.path.isdir(llm_path):
        tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
        text_encoder = transformers.AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=dtype, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained('configs/qwen3_06b', local_files_only=True)
        llm_config = transformers.Qwen3Config.from_pretrained('configs/qwen3_06b', local_files_only=True)
        with init_empty_weights():
            text_encoder = transformers.Qwen3ForCausalLM(llm_config)
        for key, tensor in iterate_safetensors(llm_path):
            set_module_tensor_to_device(text_encoder, key, device='cpu', dtype=dtype, value=tensor)

    text_encoder = text_encoder.model
    text_encoder.config.use_cache = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_encoder.to(device).eval().requires_grad_(False)

    t5_tokenizer = T5TokenizerFast(
        vocab_file='configs/t5_old/spiece.model',
        tokenizer_file='configs/t5_old/tokenizer.json',
    )
    return text_encoder, tokenizer, t5_tokenizer


def load_control_lora(dit, lora_path, device, dtype):
    print(f"Loading EasyControl LoRA: {lora_path}")
    sd = load_safetensors(lora_path, device='cpu')

    # Infer rank
    rank = 128
    for k, v in sd.items():
        if '.down.weight' in k:
            rank = v.shape[0]
            break

    # Read metadata
    alpha = float(rank)
    n_loras = 1
    try:
        from safetensors import safe_open
        with safe_open(lora_path, framework="pt") as f:
            meta = f.metadata() or {}
            rank = int(meta.get('rank', rank))
            alpha = float(meta.get('network_alpha', meta.get('alpha', alpha)))
            n_loras = int(meta.get('n_loras', n_loras))
    except Exception:
        pass

    dim = dit.model_channels
    # cond_size is dynamic, use default for init (will be overridden at runtime)
    cond_size = 1024
    processors = nn.ModuleList([
        AnimaControlSelfAttn(dim=dim, rank=rank, network_alpha=alpha, cond_size=cond_size, n_loras=n_loras)
        for _ in range(dit.num_blocks)
    ])

    clean_sd = {(k.replace('control_processors.', '') if k.startswith('control_processors.') else k): v for k, v in sd.items()}
    info = processors.load_state_dict(clean_sd, strict=False)
    if info.missing_keys:
        print(f"  Warning: {len(info.missing_keys)} missing keys")
    if info.unexpected_keys:
        print(f"  Warning: {len(info.unexpected_keys)} unexpected keys")

    processors.to(device, dtype).eval()
    n_params = sum(p.numel() for p in processors.parameters())
    print(f"  LoRA loaded: rank={rank}, alpha={alpha}, params={n_params:,}")
    return processors


# ============================================================================
# Encoding
# ============================================================================

def encode_prompt(text_encoder, tokenizer, t5_tokenizer, prompt, device, dtype):
    batch = tokenizer(
        [prompt], return_tensors="pt", truncation=True, padding="max_length", max_length=512,
    )
    t5_batch = t5_tokenizer(
        [prompt], return_tensors="pt", truncation=True, padding="max_length", max_length=512,
    )
    with torch.no_grad():
        embeds = _compute_text_embeddings(text_encoder, batch.input_ids, batch.attention_mask)
    return embeds.to(device, dtype), batch.attention_mask.to(device), t5_batch.input_ids.to(device), t5_batch.attention_mask.to(device)


def preprocess_control_image(img, width, height):
    """EXATAMENTE o preprocessing do treino (models/base.py):
    ImageOps.fit (crop central preservando aspect ratio) + Normalize p/ [-1,1].
    Retorna (1, 3, 1, H, W). Função pura para teste numérico de contrato."""
    from PIL import ImageOps
    img = ImageOps.fit(img.convert("RGB"), (width, height))
    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    return to_tensor(img).unsqueeze(0).unsqueeze(2)


def encode_control(vae, image_path, height, width, device, dtype):
    img = Image.open(image_path)
    tensor = preprocess_control_image(img, width, height).to(device, dtype)
    with torch.no_grad():
        latents = vae_encode(tensor, vae)  # (1, 16, 1, H/8, W/8)
    return latents


# ============================================================================
# Normal Anima sampling (no control — identical to sd-scripts generate_body)
# ============================================================================

@torch.no_grad()
def sample_normal(dit, pos_context, neg_context, height, width, steps, cfg, flow_shift, seed, device, dtype):
    """Standard Anima Euler sampling without any control conditioning."""
    latent_h = height // 8
    latent_w = width // 8

    gen = torch.Generator(device="cpu").manual_seed(seed)
    from diffusers.utils.torch_utils import randn_tensor
    latents = randn_tensor((1, 16, 1, latent_h, latent_w), generator=gen, device=device, dtype=torch.bfloat16)

    padding_mask = torch.zeros(1, 1, latent_h, latent_w, dtype=torch.bfloat16, device=device)

    timesteps, sigmas = get_timesteps_sigmas(steps, flow_shift, device)
    timesteps /= 1000
    timesteps = timesteps.to(device, dtype=torch.bfloat16)

    do_cfg = cfg > 1.0 and neg_context is not None

    for step_i in tqdm(range(steps), desc="Sampling"):
        t_expand = timesteps[step_i].expand(latents.shape[0])

        with torch.autocast('cuda', dtype=torch.bfloat16):
            noise_pred = dit(latents, t_expand, pos_context, padding_mask=padding_mask)

        if do_cfg:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                uncond_pred = dit(latents, t_expand, neg_context, padding_mask=padding_mask)
            noise_pred = uncond_pred + cfg * (noise_pred - uncond_pred)

        latents = euler_step(latents, noise_pred, sigmas, step_i).to(latents.dtype)

    return latents


# ============================================================================
# IC-LoRA sampling (temporal concat with per-token timestep)
# ============================================================================

@torch.no_grad()
def sample_ic_lora(
    dit, pos_context, neg_context,
    control_latents, height, width, steps, cfg, flow_shift, seed, device, dtype,
):
    """IC-LoRA sampling: temporal concat with per-token timestep.

    Like levzzz but with per-token timestep:
    - Target frame (T=0): timestep = sigma (being denoised)
    - Reference frame (T=1): timestep = 0 (clean, context)

    The model sees (B, 2) timesteps, applying different AdaLN modulation
    per temporal position — matching how it was trained.
    """
    latent_h = height // 8
    latent_w = width // 8

    gen = torch.Generator(device="cpu").manual_seed(seed)
    from diffusers.utils.torch_utils import randn_tensor
    latents = randn_tensor((1, 16, 1, latent_h, latent_w), generator=gen, device=device, dtype=torch.bfloat16)

    padding_mask = torch.zeros(1, 1, latent_h, latent_w, dtype=torch.bfloat16, device=device)

    timesteps, sigmas = get_timesteps_sigmas(steps, flow_shift, device)
    timesteps /= 1000
    timesteps = timesteps.to(device, dtype=torch.bfloat16)

    do_cfg = cfg > 1.0 and neg_context is not None
    ctrl = control_latents.to(torch.bfloat16)

    for step_i in tqdm(range(steps), desc="Sampling (IC-LoRA)"):
        # Per-token timestep: [sigma_target, 0_reference]
        t_target = timesteps[step_i].unsqueeze(0)  # (1,)
        t_ref = torch.zeros(1, device=device, dtype=torch.bfloat16)
        t_per_token = torch.stack([t_target, t_ref], dim=1)  # (1, 2)

        # Temporal concat: [noise_frame, clean_reference_frame]
        x_input = torch.cat([latents, ctrl], dim=2)  # (1, 16, 2, H/8, W/8)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = dit(x_input, t_per_token, pos_context, padding_mask=padding_mask)

        noise_pred = output[:, :, :1, :, :]

        if do_cfg:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output_neg = dit(x_input, t_per_token, neg_context, padding_mask=padding_mask)
            uncond_pred = output_neg[:, :, :1, :, :]
            noise_pred = uncond_pred + cfg * (noise_pred - uncond_pred)

        latents = euler_step(latents, noise_pred, sigmas, step_i).to(latents.dtype)

    return latents


# ============================================================================
# IC-LoRA Full sampling (ref_first, corrected from LTX-2)
# ============================================================================

@torch.no_grad()
def combine_three_branch(u, t, c, text_cfg, ref_cfg):
    """Guidance InstructPix2Pix-style com escalas independentes.

    u = f(texto negativo, ref zerada)
    t = f(texto positivo, ref zerada)
    c = f(texto positivo, ref real)
    pred = u + text_cfg*(t - u) + ref_cfg*(c - t)

    ref_cfg NAO é multiplicado por text_cfg — as duas forças são independentes
    (o CFG de 2 branches com ref só no positivo amplificava a ref por text_cfg).
    """
    return u + text_cfg * (t - u) + ref_cfg * (c - t)


def sample_ic_lora_full(
    dit, pos_context, neg_context,
    control_latents, height, width, steps, cfg, flow_shift, seed, device, dtype,
    ref_cfg=1.0,
):
    """IC-LoRA Full sampling: reference FIRST, then target (LTX-2 style).

    Concat order: [clean_ref T=0, noisy_target T=1]
    Timestep:     [0_ref,         sigma_target   ]
    Output slice: last frame (T=1) = target prediction
    Guidance: 3 branches (combine_three_branch) — text_cfg = cfg, ref_cfg separado.
    """
    latent_h = height // 8
    latent_w = width // 8

    gen = torch.Generator(device="cpu").manual_seed(seed)
    from diffusers.utils.torch_utils import randn_tensor
    latents = randn_tensor((1, 16, 1, latent_h, latent_w), generator=gen, device=device, dtype=torch.bfloat16)

    padding_mask = torch.zeros(1, 1, latent_h, latent_w, dtype=torch.bfloat16, device=device)

    timesteps, sigmas = get_timesteps_sigmas(steps, flow_shift, device)
    timesteps /= 1000
    timesteps = timesteps.to(device, dtype=torch.bfloat16)

    do_text_cfg = cfg > 1.0 and neg_context is not None
    ctrl = control_latents.to(torch.bfloat16)
    ctrl_zero = torch.zeros_like(ctrl)

    def _fwd(x_frames, t_per_token, context):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = dit(x_frames, t_per_token, context, padding_mask=padding_mask)
        return out[:, :, -1:, :, :]  # target = último frame

    for step_i in tqdm(range(steps), desc="Sampling (IC-LoRA Full, 3-branch)"):
        t_target = timesteps[step_i].unsqueeze(0)  # (1,)
        t_ref = torch.zeros(1, device=device, dtype=torch.bfloat16)
        t_per_token = torch.stack([t_ref, t_target], dim=1)  # (1, 2)

        c = _fwd(torch.cat([ctrl, latents], dim=2), t_per_token, pos_context)
        t_branch = _fwd(torch.cat([ctrl_zero, latents], dim=2), t_per_token, pos_context)
        if do_text_cfg:
            u = _fwd(torch.cat([ctrl_zero, latents], dim=2), t_per_token, neg_context)
        else:
            u = t_branch
        noise_pred = combine_three_branch(u, t_branch, c, cfg if do_text_cfg else 1.0, ref_cfg)

        latents = euler_step(latents, noise_pred, sigmas, step_i).to(latents.dtype)

    return latents


# ============================================================================
# OminiControl sampling (temporal concat + position manipulation)
# ============================================================================

@torch.no_grad()
def sample_ominicontrol(
    dit, pos_context, neg_context,
    control_latents, height, width, steps, cfg, flow_shift, seed, device, dtype,
    position_mode='spatial', ref_cfg=1.0,
):
    """OminiControl sampling: temporal concat + per-token timestep + RoPE manipulation.

    Like IC-LoRA but with position-aware RoPE:
    - Spatial mode: condition shares noise positions (pixel correspondence)
    - Subject mode: condition keeps T=1 position (semantic correspondence)
    """
    latent_h = height // 8
    latent_w = width // 8

    gen = torch.Generator(device="cpu").manual_seed(seed)
    from diffusers.utils.torch_utils import randn_tensor
    latents = randn_tensor((1, 16, 1, latent_h, latent_w), generator=gen, device=device, dtype=torch.bfloat16)

    padding_mask = torch.zeros(1, 1, latent_h, latent_w, dtype=torch.bfloat16, device=device)

    timesteps, sigmas = get_timesteps_sigmas(steps, flow_shift, device)
    timesteps /= 1000
    timesteps = timesteps.to(device, dtype=torch.bfloat16)

    do_text_cfg = cfg > 1.0 and neg_context is not None
    ctrl = control_latents.to(torch.bfloat16)
    ctrl_zero = torch.zeros_like(ctrl)

    # For spatial mode, we need to hook into the model to override RoPE positions.
    # We use a forward hook on prepare_embedded_sequence result.
    rope_hook_handle = None
    if position_mode == 'spatial':
        rope_hook_handle = _install_spatial_rope_hook(dit)

    def _fwd(x_frames, t_per_token, context):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = dit(x_frames, t_per_token, context, padding_mask=padding_mask)
        return out[:, :, :1, :, :]  # target = primeiro frame

    try:
        for step_i in tqdm(range(steps), desc=f"Sampling (OminiControl-{position_mode}, 3-branch)"):
            # Per-token timestep: [sigma_target, 0_condition]
            t_target = timesteps[step_i].unsqueeze(0)
            t_ref = torch.zeros(1, device=device, dtype=torch.bfloat16)
            t_per_token = torch.stack([t_target, t_ref], dim=1)  # (1, 2)

            c = _fwd(torch.cat([latents, ctrl], dim=2), t_per_token, pos_context)
            t_branch = _fwd(torch.cat([latents, ctrl_zero], dim=2), t_per_token, pos_context)
            if do_text_cfg:
                u = _fwd(torch.cat([latents, ctrl_zero], dim=2), t_per_token, neg_context)
            else:
                u = t_branch
            noise_pred = combine_three_branch(u, t_branch, c, cfg if do_text_cfg else 1.0, ref_cfg)

            latents = euler_step(latents, noise_pred, sigmas, step_i).to(latents.dtype)
    finally:
        if rope_hook_handle is not None:
            rope_hook_handle.remove()

    return latents


def _install_spatial_rope_hook(dit):
    """Install a hook that duplicates noise RoPE positions for condition tokens.

    In spatial mode, condition should share the SAME RoPE positions as noise,
    creating pixel-to-pixel spatial correspondence.
    """
    original_prepare = dit.prepare_embedded_sequence

    def hooked_prepare(x_B_C_T_H_W, fps=None, padding_mask=None):
        x_B_T_H_W_D, rope_emb, extra_pos_emb = original_prepare(x_B_C_T_H_W, fps=fps, padding_mask=padding_mask)
        T = x_B_C_T_H_W.shape[2]
        if T > 1 and rope_emb is not None:
            total_seq = rope_emb.shape[0]
            noise_seq_len = total_seq // T
            noise_rope = rope_emb[:noise_seq_len]
            rope_emb = noise_rope.repeat(T, 1, 1, 1)
        return x_B_T_H_W_D, rope_emb, extra_pos_emb

    dit.prepare_embedded_sequence = hooked_prepare
    # Return a handle-like object for cleanup
    class HookHandle:
        def remove(self):
            dit.prepare_embedded_sequence = original_prepare
    return HookHandle()


# ============================================================================
# levzzz sampling (temporal concatenation — standard PEFT LoRA)
# ============================================================================

@torch.no_grad()
def sample_levzzz(
    dit, pos_context, neg_context,
    control_latents, height, width, steps, cfg, flow_shift, seed, device, dtype,
):
    """levzzz-style sampling: concatenate control as temporal frame T=1.

    Dead simple: cat([noise_T0, control_T1], dim=2), forward, take output[:,:,:1].
    The model's 3D RoPE naturally encodes the temporal position difference.
    """
    latent_h = height // 8
    latent_w = width // 8

    gen = torch.Generator(device="cpu").manual_seed(seed)
    from diffusers.utils.torch_utils import randn_tensor
    latents = randn_tensor((1, 16, 1, latent_h, latent_w), generator=gen, device=device, dtype=torch.bfloat16)

    padding_mask = torch.zeros(1, 1, latent_h, latent_w, dtype=torch.bfloat16, device=device)

    timesteps, sigmas = get_timesteps_sigmas(steps, flow_shift, device)
    timesteps /= 1000
    timesteps = timesteps.to(device, dtype=torch.bfloat16)

    do_cfg = cfg > 1.0 and neg_context is not None
    ctrl = control_latents.to(torch.bfloat16)

    for step_i in tqdm(range(steps), desc="Sampling (levzzz)"):
        t_expand = timesteps[step_i].expand(latents.shape[0])

        # Temporal concat: [noise_frame, control_frame]
        x_input = torch.cat([latents, ctrl], dim=2)  # (1, 16, 2, H/8, W/8)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = dit(x_input, t_expand, pos_context, padding_mask=padding_mask)

        # Take only the noise frame prediction
        noise_pred = output[:, :, :1, :, :]

        if do_cfg:
            x_input_neg = torch.cat([latents, ctrl], dim=2)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output_neg = dit(x_input_neg, t_expand, neg_context, padding_mask=padding_mask)
            uncond_pred = output_neg[:, :, :1, :, :]
            noise_pred = uncond_pred + cfg * (noise_pred - uncond_pred)

        latents = euler_step(latents, noise_pred, sigmas, step_i).to(latents.dtype)

    return latents


def load_routed_lora_entries(dit, lora_path, device, dtype):
    """Map ic_lora_routed LoRA pairs onto live DiT modules WITHOUT merging.

    Returns [(module, A, B, scale)] for _RoutedLoraScope. Reuses the PEFT key
    parsing of load_peft_lora but keeps base weights untouched.
    """
    print(f"Loading ROUTED LoRA (masked runtime application): {lora_path}")
    lora_dir = os.path.dirname(lora_path)
    sd = load_safetensors(lora_path, device='cpu')
    sd = {k.replace('diffusion_model.', ''): v for k, v in sd.items()}
    lora_pairs = {}
    for k in sd.keys():
        if 'lora_A' in k:
            base = k.replace('lora_A.default.weight', '').replace('lora_A.weight', '')
            base = base.replace('base_model.model.', '').rstrip('.')
            key_b = k.replace('lora_A', 'lora_B')
            if key_b in sd:
                lora_pairs[base] = (sd[k], sd[key_b])
    scale = 1.0
    cfg_path = os.path.join(lora_dir, 'adapter_config.json')
    if os.path.exists(cfg_path):
        import json
        with open(cfg_path) as f:
            cfg = json.load(f)
        if cfg.get('lora_alpha') and cfg.get('r'):
            scale = cfg['lora_alpha'] / cfg['r']
    named = dict(dit.named_modules())
    entries, missing = [], []
    for path, (a, b) in lora_pairs.items():
        module = named.get(path)
        if module is None:
            missing.append(path)
        else:
            entries.append((module, a.to(device, dtype), b.to(device, dtype), scale))
    if not entries:
        raise ValueError(f'No routed LoRA modules matched the DiT (examples: {list(lora_pairs)[:3]})')
    if missing:
        print(f"  WARNING: {len(missing)} keys not matched (e.g. {missing[:3]})")
    print(f"  {len(entries)} linears wrapped (masked to the reference frame, scale={scale:.2f})")
    return entries


class _RoutedLoraScope:
    """Wraps linears so the LoRA delta hits ONLY the reference-frame rows.

    Inference layout is fixed: T=2, ref_first. Inside Anima blocks:
    - (B, 2*HW, D) self_attn tokens -> mask first HW rows
    - (B, 2, H, W, D) mlp tokens    -> mask frame 0
    Anything else (e.g. a vanilla single-frame pass) gets no delta.
    """

    def __init__(self, entries, strength=1.0, masked='first'):
        self.entries = entries
        self.strength = strength
        assert masked in ('first', 'last')
        self.masked = masked
        self._originals = []

    def __enter__(self):
        import torch.nn.functional as _F
        ref_first = self.masked == 'first'
        for module, a, b, scale in self.entries:
            orig = module.forward

            def wrapped(x, *args, _orig=orig, _a=a, _b=b, _s=scale * self.strength, **kwargs):
                result = _orig(x, *args, **kwargs)
                if _s == 0:
                    return result
                if result.ndim == 3 and result.shape[1] % 2 == 0:
                    hw = result.shape[1] // 2
                    mask = result.new_zeros(1, result.shape[1], 1)
                    if ref_first:
                        mask[:, :hw] = 1
                    else:
                        mask[:, hw:] = 1
                elif result.ndim == 5 and result.shape[1] == 2:
                    mask = result.new_zeros(1, 2, 1, 1, 1)
                    mask[:, 0 if ref_first else 1] = 1
                else:
                    return result
                delta = _F.linear(_F.linear(x.to(_a.dtype), _a), _b) * _s
                return result + delta.to(result.dtype) * mask

            module.forward = wrapped
            self._originals.append((module, orig))
        return self

    def __exit__(self, *exc):
        for module, orig in self._originals:
            module.forward = orig
        self._originals = []
        return False


def load_peft_lora(dit, lora_path, device, dtype, skip_adaln=False, lora_strength=1.0):
    """Load a PEFT LoRA into the DiT and merge weights.

    Handles both:
    - PEFT format (adapter_config.json + adapter_model.safetensors)
    - diffusion-pipe format (keys with 'diffusion_model.' prefix + lora_A/lora_B)
    """
    print(f"Loading PEFT LoRA: {lora_path}")
    from peft import PeftModel

    # Resolve path
    if os.path.isdir(lora_path):
        lora_dir = lora_path
        lora_file = os.path.join(lora_path, 'adapter_model.safetensors')
    else:
        lora_dir = os.path.dirname(lora_path)
        lora_file = lora_path

    sd = load_safetensors(lora_file, device='cpu')

    # Check if adapter_config.json exists (PEFT format)
    has_peft_config = os.path.exists(os.path.join(lora_dir, 'adapter_config.json'))

    # Check key format
    has_lora_A = any('lora_A' in k for k in sd.keys())
    has_diffusion_prefix = any(k.startswith('diffusion_model.') for k in sd.keys())

    # Always use manual merge — most reliable, works regardless of key format
    # Strip prefix if present
    if has_diffusion_prefix:
        sd = {k.replace('diffusion_model.', ''): v for k, v in sd.items()}

    if has_lora_A or any('lora_down' in k for k in sd.keys()):
        # Manual merge — most reliable approach
        # Build a map: base_param_name → (lora_A_tensor, lora_B_tensor)
        lora_pairs = {}
        for k in sd.keys():
            if 'lora_A' in k:
                # Strip PEFT prefix patterns: base_model.model. and .default
                base = k.replace('lora_A.default.weight', '').replace('lora_A.weight', '')
                base = base.replace('base_model.model.', '').rstrip('.')
                key_b = k.replace('lora_A', 'lora_B')
                if key_b in sd:
                    lora_pairs[base] = (sd[k], sd[key_b])
            elif 'lora_down' in k:
                base = k.replace('.lora_down.weight', '')
                base = base.replace('base_model.model.', '')
                key_b = k.replace('lora_down', 'lora_up')
                if key_b in sd:
                    lora_pairs[base] = (sd[k], sd[key_b])

        # Read alpha from config if available
        lora_alpha = None
        lora_rank = None
        if has_peft_config:
            import json
            with open(os.path.join(lora_dir, 'adapter_config.json')) as f:
                cfg = json.load(f)
                lora_alpha = cfg.get('lora_alpha', None)
                lora_rank = cfg.get('r', None)

        merged = 0
        for name, param in dit.named_parameters():
            base_name = name.replace('.weight', '')
            if base_name in lora_pairs:
                if skip_adaln and 'adaln_modulation' in base_name:
                    continue
                lora_a, lora_b = lora_pairs[base_name]
                lora_a = lora_a.to(device, dtype)
                lora_b = lora_b.to(device, dtype)
                # Apply scaling: delta_W = strength * (alpha / rank) * B @ A
                scale = 1.0
                if lora_alpha is not None and lora_rank is not None and lora_rank > 0:
                    scale = lora_alpha / lora_rank
                scale *= lora_strength
                param.data += (scale * (lora_b @ lora_a)).to(param.dtype)
                merged += 1
        print(f"  LoRA manually merged ({merged} weight matrices, scale={scale:.2f})")
        if merged == 0:
            print(f"  WARNING: No weights merged! LoRA keys: {list(lora_pairs.keys())[:5]}")
            print(f"  Model keys: {[n for n, _ in dit.named_parameters()][:5]}")
    else:
        print(f"  WARNING: Unrecognized LoRA format. Keys: {list(sd.keys())[:5]}")

    return dit


# ============================================================================
# EasyControl sampling
# ============================================================================

@torch.no_grad()
def sample_easycontrol(
    dit, control_processors, pos_context, neg_context,
    control_latents, height, width, steps, cfg, flow_shift, seed, device, dtype,
    control_strength=1.0,
):
    """EasyControl sampling — entire function runs under autocast to avoid dtype mismatches.

    Many model components (x_embedder, t_embedder, final_layer) have float32 weights
    while inputs are bf16. During training DeepSpeed handles this; during inference
    we need autocast globally.
    """
    latent_h = height // 8
    latent_w = width // 8

    # Noise (generated in float32, cast to bf16)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    from diffusers.utils.torch_utils import randn_tensor
    latents = randn_tensor((1, 16, 1, latent_h, latent_w), generator=gen, device=device, dtype=torch.bfloat16)

    # Padding mask
    padding_mask = torch.zeros(1, 1, latent_h, latent_w, dtype=torch.bfloat16, device=device)

    # Timestep schedule
    timesteps, sigmas = get_timesteps_sigmas(steps, flow_shift, device)
    timesteps /= 1000  # scale to [0,1]
    timesteps = timesteps.to(device, dtype=torch.bfloat16)

    do_cfg = cfg > 1.0 and neg_context is not None

    # GLOBAL AUTOCAST: many model components (x_embedder, t_embedder, adaln, final_layer)
    # have float32 weights. During training DeepSpeed handles dtype mixing automatically.
    # For inference we need autocast around ALL model calls.
    _autocast = torch.autocast('cuda', dtype=torch.bfloat16)
    _autocast.__enter__()

    # Embed control through shared PatchEmbed
    pad_cond = torch.zeros(1, 1, 1, control_latents.shape[3], control_latents.shape[4],
                           device=device, dtype=torch.bfloat16)
    cond_with_mask = torch.cat([control_latents.to(torch.bfloat16), pad_cond], dim=1)
    cond_embedded = dit.x_embedder(cond_with_mask)
    _, Tc, Hc, Wc, D = cond_embedded.shape

    # Condition t=0 embedding
    cond_t_emb, cond_adaln_lora = compute_condition_t_embedding(
        dit.t_embedder, dit.t_embedding_norm, 1, device, torch.bfloat16
    )

    def run_forward(x, context_emb):
        """One EasyControl forward pass through all 28 blocks."""
        # Prepare noise
        x_5d, noise_rope, _ = dit.prepare_embedded_sequence(x, fps=None, padding_mask=padding_mask)
        B, T, H, W, D = x_5d.shape

        # Noise timestep
        t_1d = timesteps[step_i].unsqueeze(0).unsqueeze(0)
        t_emb, adaln_lora = dit.t_embedder(t_1d)
        t_emb = dit.t_embedding_norm(t_emb)

        # Condition RoPE
        cond_rope = generate_condition_rope(dit.pos_embedder, (H, W), (Hc, Wc), device)

        cond_state = cond_embedded.clone()

        for bi, block in enumerate(dit.blocks):
            # AdaLN
            if block.use_adaln_lora:
                sh_sa, sc_sa, g_sa = (block.adaln_modulation_self_attn(t_emb) + adaln_lora).chunk(3, -1)
                sh_ca, sc_ca, g_ca = (block.adaln_modulation_cross_attn(t_emb) + adaln_lora).chunk(3, -1)
                sh_ml, sc_ml, g_ml = (block.adaln_modulation_mlp(t_emb) + adaln_lora).chunk(3, -1)
                c_sh_sa, c_sc_sa, c_g_sa = (block.adaln_modulation_self_attn(cond_t_emb) + cond_adaln_lora).chunk(3, -1)
                c_sh_ml, c_sc_ml, c_g_ml = (block.adaln_modulation_mlp(cond_t_emb) + cond_adaln_lora).chunk(3, -1)
            else:
                sh_sa, sc_sa, g_sa = block.adaln_modulation_self_attn(t_emb).chunk(3, -1)
                sh_ca, sc_ca, g_ca = block.adaln_modulation_cross_attn(t_emb).chunk(3, -1)
                sh_ml, sc_ml, g_ml = block.adaln_modulation_mlp(t_emb).chunk(3, -1)
                c_sh_sa, c_sc_sa, c_g_sa = block.adaln_modulation_self_attn(cond_t_emb).chunk(3, -1)
                c_sh_ml, c_sc_ml, c_g_ml = block.adaln_modulation_mlp(cond_t_emb).chunk(3, -1)

            def _r(t):
                return rearrange(t, "b t d -> b t 1 1 d")

            # 1. Self-attention with condition
            norm_noise = block.layer_norm_self_attn(x_5d) * (1 + _r(sc_sa)) + _r(sh_sa)
            norm_cond = block.layer_norm_self_attn(cond_state) * (1 + _r(c_sc_sa)) + _r(c_sh_sa)

            noise_flat = rearrange(norm_noise, "b t h w d -> b (t h w) d")
            cond_flat = rearrange(norm_cond, "b t hc wc d -> b (t hc wc) d")

            noise_out, cond_out = control_processors[bi](
                block.self_attn, noise_flat, cond_flat, noise_rope, cond_rope,
                lora_weights=[control_strength],
            )

            noise_out = rearrange(noise_out, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
            cond_out = rearrange(cond_out, "b (t hc wc) d -> b t hc wc d", t=Tc, hc=Hc, wc=Wc)
            x_5d = x_5d + _r(g_sa) * noise_out
            cond_state = cond_state + _r(c_g_sa) * cond_out

            # 2. Cross-attention (noise only)
            norm_ca = block.layer_norm_cross_attn(x_5d) * (1 + _r(sc_ca)) + _r(sh_ca)
            ca_out = rearrange(
                block.cross_attn(
                    rearrange(norm_ca, "b t h w d -> b (t h w) d"),
                    context_emb,
                    rope_emb=noise_rope,
                ),
                "b (t h w) d -> b t h w d", t=T, h=H, w=W,
            )
            x_5d = ca_out * _r(g_ca) + x_5d

            # 3. MLP (both)
            norm_ml = block.layer_norm_mlp(x_5d) * (1 + _r(sc_ml)) + _r(sh_ml)
            x_5d = x_5d + _r(g_ml) * block.mlp(norm_ml)
            c_norm_ml = block.layer_norm_mlp(cond_state) * (1 + _r(c_sc_ml)) + _r(c_sh_ml)
            cond_state = cond_state + _r(c_g_ml) * block.mlp(c_norm_ml)

        # Final layer
        x_out = dit.final_layer(x_5d, t_emb, adaln_lora_B_T_3D=adaln_lora)
        return dit.unpatchify(x_out)

    # Sampling loop (already inside global autocast context)
    for step_i in tqdm(range(steps), desc="Sampling"):
        pos_out = run_forward(latents, pos_context).float()
        if do_cfg:
            neg_out = run_forward(latents, neg_context).float()
            noise_pred = neg_out + cfg * (pos_out - neg_out)
        else:
            noise_pred = pos_out

        latents = euler_step(latents, noise_pred, sigmas, step_i).to(latents.dtype)

    _autocast.__exit__(None, None, None)
    return latents


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Load
    dit = load_dit(args.dit, device, dtype)
    vae = load_vae(args.vae, device, dtype, args.vae_chunk_size)
    text_encoder, tokenizer, t5_tokenizer = load_text_encoder(args.llm, device, dtype)

    # Detect mode
    mode = args.mode
    control_processors = None

    if args.lora:
        if mode == "auto":
            # Auto-detect: check if LoRA has EasyControl-specific keys
            sd_check = load_safetensors(args.lora, device='cpu')
            has_ec_keys = any('q_loras' in k or 'k_loras' in k for k in sd_check.keys())
            mode = "easycontrol" if has_ec_keys else "levzzz"
            del sd_check
            print(f"Auto-detected mode: {mode}")

        if mode == "easycontrol":
            control_processors = load_control_lora(dit, args.lora, device, dtype)
        elif mode in ("ic_lora_routed", "routed_targetfirst"):
            # Masked runtime application — merging would apply the delta to ALL
            # rows, breaking the condition-only training contract.
            routed_entries = load_routed_lora_entries(dit, args.lora, device, dtype)
        elif mode in ("levzzz", "ic_lora", "ic_lora_full", "ominicontrol", "ominicontrol_subject"):
            dit = load_peft_lora(dit, args.lora, device, dtype, skip_adaln=args.skip_adaln,
                                 lora_strength=args.lora_strength)
    else:
        mode = "normal"
        print("No LoRA specified — running normal Anima generation (no control)")

    # Encode prompts
    print("Encoding prompts...")
    pos_emb, pos_mask, pos_t5, pos_t5_mask = encode_prompt(text_encoder, tokenizer, t5_tokenizer, args.prompt, device, dtype)

    neg_context = None
    if args.cfg > 1.0:
        neg_emb, neg_mask, neg_t5, neg_t5_mask = encode_prompt(
            text_encoder, tokenizer, t5_tokenizer, args.negative_prompt or "", device, dtype
        )

    # Run LLM adapter (autocast needed: nn.Embedding returns float32, weights are bf16)
    print("Running LLM adapter...")
    if dit.use_llm_adapter and hasattr(dit, 'llm_adapter'):
        with torch.autocast('cuda', dtype=dtype):
            pos_context = dit.llm_adapter(
                source_hidden_states=pos_emb.to(dtype),
                target_input_ids=pos_t5,
                target_attention_mask=pos_t5_mask,
                source_attention_mask=pos_mask,
            )
        pos_context[~pos_t5_mask.bool()] = 0
        if args.cfg > 1.0:
            with torch.autocast('cuda', dtype=dtype):
                neg_context = dit.llm_adapter(
                    source_hidden_states=neg_emb.to(dtype),
                    target_input_ids=neg_t5,
                    target_attention_mask=neg_t5_mask,
                    source_attention_mask=neg_mask,
                )
            neg_context[~neg_t5_mask.bool()] = 0
    else:
        pos_context = pos_emb
        neg_context = neg_emb if args.cfg > 1.0 else None

    # Free text encoder
    text_encoder.to("cpu")
    torch.cuda.empty_cache()

    # Encode control (if LoRA + control image provided)
    control_latents = None
    if args.control_image and args.lora:
        print(f"Encoding control: {args.control_image}")
        control_latents = encode_control(vae, args.control_image, args.height, args.width, device, dtype)
        print(f"  Control latents: {control_latents.shape}")

    # Sample
    print(f"Generating {args.width}x{args.height}, {args.steps} steps, CFG={args.cfg}, shift={args.flow_shift}, seed={args.seed}")
    if mode == "easycontrol" and control_processors is not None and control_latents is not None:
        print(f"  Mode: EasyControl, strength: {args.control_strength}")
        latents = sample_easycontrol(
            dit, control_processors, pos_context, neg_context,
            control_latents, args.height, args.width, args.steps, args.cfg, args.flow_shift, args.seed,
            device, dtype,
            control_strength=args.control_strength,
        )
    elif mode == "ic_lora" and control_latents is not None:
        print(f"  Mode: IC-LoRA (per-token timestep)")
        latents = sample_ic_lora(
            dit, pos_context, neg_context,
            control_latents, args.height, args.width, args.steps, args.cfg, args.flow_shift, args.seed,
            device, dtype,
        )
    elif mode == "ic_lora_full" and control_latents is not None:
        print(f"  Mode: IC-LoRA Full (ref_first, LTX-2 style, ref_cfg={args.ref_cfg})")
        latents = sample_ic_lora_full(
            dit, pos_context, neg_context,
            control_latents, args.height, args.width, args.steps, args.cfg, args.flow_shift, args.seed,
            device, dtype, ref_cfg=args.ref_cfg,
        )
    elif mode == "ic_lora_routed" and control_latents is not None:
        print(f"  Mode: IC-LoRA Routed (ref_first, masked runtime LoRA, strength={args.lora_strength}, ref_cfg={args.ref_cfg})")
        with _RoutedLoraScope(routed_entries, args.lora_strength, masked='first'):
            latents = sample_ic_lora_full(
                dit, pos_context, neg_context,
                control_latents, args.height, args.width, args.steps, args.cfg, args.flow_shift, args.seed,
                device, dtype, ref_cfg=args.ref_cfg,
            )
    elif mode == "routed_targetfirst" and control_latents is not None:
        print(f"  Mode: Routed TARGET-FIRST (alvo T=0, ref T=1, LoRA mascarado, strength={args.lora_strength}, ref_cfg={args.ref_cfg})")
        with _RoutedLoraScope(routed_entries, args.lora_strength, masked='last'):
            latents = sample_ominicontrol(
                dit, pos_context, neg_context,
                control_latents, args.height, args.width, args.steps, args.cfg, args.flow_shift, args.seed,
                device, dtype, position_mode='subject', ref_cfg=args.ref_cfg,
            )
    elif mode in ("ominicontrol", "ominicontrol_subject") and control_latents is not None:
        pos_mode = "subject" if mode == "ominicontrol_subject" else "spatial"
        print(f"  Mode: OminiControl ({pos_mode}, ref_cfg={args.ref_cfg})")
        latents = sample_ominicontrol(
            dit, pos_context, neg_context,
            control_latents, args.height, args.width, args.steps, args.cfg, args.flow_shift, args.seed,
            device, dtype,
            position_mode=pos_mode, ref_cfg=args.ref_cfg,
        )
    elif mode == "levzzz" and control_latents is not None:
        print(f"  Mode: levzzz (temporal concat)")
        latents = sample_levzzz(
            dit, pos_context, neg_context,
            control_latents, args.height, args.width, args.steps, args.cfg, args.flow_shift, args.seed,
            device, dtype,
        )
    else:
        print(f"  Mode: normal (no control)")
        latents = sample_normal(
            dit, pos_context, neg_context,
            args.height, args.width, args.steps, args.cfg, args.flow_shift, args.seed,
            device, dtype,
        )

    # Decode — WanVAE_.decode(z, scale) handles denormalization internally
    print("Decoding...")
    with torch.no_grad():
        # latents shape: (1, 16, 1, H/8, W/8) — already 5D
        pixels = vae.model.decode(latents.to(dtype), vae.scale)

    if pixels.ndim == 5:
        pixels = pixels.squeeze(2)
    pixels = pixels.float().clamp(-1, 1)

    # Save
    os.makedirs(args.save_path, exist_ok=True)
    time_flag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{time_flag}_{args.seed}.png"
    save_file = os.path.join(args.save_path, filename)

    x = ((pixels[0] + 1) * 127.5).to(torch.uint8).cpu().numpy()
    x = x.transpose(1, 2, 0)  # CHW -> HWC
    Image.fromarray(x).save(save_file)
    print(f"Saved: {save_file}")


if __name__ == "__main__":
    main()
