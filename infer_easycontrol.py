#!/usr/bin/env python3
"""
EasyControl Anima inference — generate images with spatial control conditioning.

Uses the EasyControl approach: condition tokens injected into self-attention
with binary-masked LoRA and causal attention mask.

Usage:
  python infer_easycontrol.py \
    --dit /path/to/anima-preview2.safetensors \
    --vae /path/to/qwen_image_vae.safetensors \
    --llm /path/to/qwen_3_06b_base.safetensors \
    --lora /path/to/adapter_model.safetensors \
    --control_image /path/to/canny.png \
    --prompt "1girl, standing" \
    --output output.png

  # With negative prompt and CFG:
  python infer_easycontrol.py \
    --dit ... --vae ... --llm ... --lora ... \
    --control_image canny.png \
    --prompt "1girl, anime style, high quality" \
    --negative_prompt "low quality, blurry" \
    --cfg 5.0 \
    --output output.png
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add diffusion-pipe root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from models.cosmos_predict2 import get_dit_config, WanVAE, vae_encode, _tokenize, _compute_text_embeddings
from models.cosmos_predict2_modeling import MiniTrainDIT
from models.easycontrol import (
    AnimaControlSelfAttn,
    generate_condition_rope,
    compute_condition_t_embedding,
    build_causal_attn_mask,
)
from utils.common import load_state_dict
from safetensors.torch import load_file as load_safetensors
from einops import rearrange

import transformers
from transformers import AutoTokenizer, T5TokenizerFast
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device


def parse_args():
    p = argparse.ArgumentParser(description="EasyControl Anima inference")
    p.add_argument("--dit", required=True, help="Anima DiT .safetensors")
    p.add_argument("--vae", required=True, help="Qwen-Image VAE .safetensors")
    p.add_argument("--llm", required=True, help="Qwen3-0.6B (dir or .safetensors)")
    p.add_argument("--lora", required=True, help="EasyControl LoRA .safetensors")
    p.add_argument("--control_image", required=True, help="Control image (canny, depth, etc.)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="output.png")
    return p.parse_args()


def load_dit(dit_path, device, dtype):
    """Load Anima DiT model."""
    state_dict = load_state_dict(dit_path)
    state_dict = {(k[4:] if k.startswith('net.') else k): v for k, v in state_dict.items()}
    dit_config = get_dit_config(state_dict)
    dit_config['use_llm_adapter'] = 'llm_adapter.out_proj.weight' in state_dict

    with init_empty_weights():
        dit = MiniTrainDIT(**dit_config)
    for name, p in dit.named_parameters():
        if name in state_dict:
            d = dtype if p.ndim > 1 else torch.float32
            set_module_tensor_to_device(dit, name, device='cpu', dtype=d, value=state_dict[name])
    del state_dict
    dit.to(device).eval().requires_grad_(False)
    return dit


def load_text_encoder(llm_path, device, dtype):
    """Load Qwen3-0.6B text encoder + tokenizers."""
    if os.path.isdir(llm_path):
        tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
        text_encoder = transformers.AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=dtype, local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained('configs/qwen3_06b', local_files_only=True)
        llm_config = transformers.Qwen3Config.from_pretrained('configs/qwen3_06b', local_files_only=True)
        with init_empty_weights():
            text_encoder = transformers.Qwen3ForCausalLM(llm_config)
        from utils.common import iterate_safetensors
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


def load_control_lora(dit, lora_path, rank=128, alpha=128.0, n_loras=1):
    """Load EasyControl LoRA processors and attach to DiT."""
    sd = load_safetensors(lora_path, device='cpu')

    # Infer rank from first down weight
    inferred_rank = None
    for k, v in sd.items():
        if '.down.weight' in k:
            inferred_rank = v.shape[0]
            break
    if inferred_rank is not None:
        rank = inferred_rank
        alpha = float(rank)  # default: alpha = rank

    # Read metadata if available
    try:
        from safetensors import safe_open
        with safe_open(lora_path, framework="pt") as f:
            meta = f.metadata()
            if meta:
                rank = int(meta.get('rank', rank))
                alpha = float(meta.get('network_alpha', meta.get('alpha', alpha)))
                n_loras = int(meta.get('n_loras', n_loras))
    except Exception:
        pass

    dim = dit.model_channels
    num_blocks = dit.num_blocks
    # cond_size from metadata or compute from first block's LoRA
    cond_size = None
    for k, v in sd.items():
        if '.cond_size' in k:
            cond_size = int(v.item())
            break

    # Fallback: infer from parameter naming
    if cond_size is None:
        for k in sd.keys():
            if 'q_loras.0.down.weight' in k:
                # cond_size was stored in the module but we compute dynamically now
                break
        # Default to 1024 (512px / 8 / 2 = 32, 32*32=1024)
        cond_size = 1024

    processors = nn.ModuleList([
        AnimaControlSelfAttn(dim=dim, rank=rank, network_alpha=alpha, cond_size=cond_size, n_loras=n_loras)
        for _ in range(num_blocks)
    ])

    # Load weights
    clean_sd = {}
    for k, v in sd.items():
        clean_k = k.replace('control_processors.', '') if k.startswith('control_processors.') else k
        clean_sd[clean_k] = v
    info = processors.load_state_dict(clean_sd, strict=False)
    if info.missing_keys:
        print(f"Warning: {len(info.missing_keys)} missing keys in LoRA checkpoint")
    if info.unexpected_keys:
        print(f"Warning: {len(info.unexpected_keys)} unexpected keys in LoRA checkpoint")

    print(f"Loaded EasyControl LoRA: rank={rank}, alpha={alpha}, blocks={num_blocks}")
    return processors


def encode_prompt(text_encoder, tokenizer, t5_tokenizer, prompt, device):
    """Encode prompt through Qwen3 + T5 tokenizer."""
    batch = tokenizer(
        [prompt], return_tensors="pt", truncation=True, padding="max_length", max_length=512,
    )
    t5_batch = _tokenize(t5_tokenizer, [prompt])

    with torch.no_grad():
        embeds = _compute_text_embeddings(text_encoder, batch.input_ids, batch.attention_mask)
    return (
        embeds.to(device),
        batch.attention_mask.to(device),
        t5_batch.input_ids.to(device),
        t5_batch.attention_mask.to(device),
    )


def encode_control(vae, image_path, height, width, device, dtype):
    """Encode control image through VAE."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((width, height), Image.LANCZOS)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device, dtype)
    with torch.no_grad():
        latents = vae_encode(tensor, vae)
    if latents.dim() == 4:
        latents = latents.unsqueeze(2)
    return latents  # (1, 16, 1, H/8, W/8)


@torch.no_grad()
def sample_easycontrol(
    dit, control_processors,
    crossattn_emb, source_attn_mask, t5_ids, t5_mask,
    neg_crossattn_emb, neg_source_attn_mask, neg_t5_ids, neg_t5_mask,
    control_latents_5d,
    height, width, steps, cfg, flow_shift, seed,
    device, dtype,
):
    """Euler discrete sampling with EasyControl condition injection."""
    latent_h = height // 8
    latent_w = width // 8

    # Start from pure noise
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, 16, 1, latent_h, latent_w, dtype=torch.float32, generator=gen).to(device, dtype)

    # Sigma schedule
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    if flow_shift != 1.0:
        sigmas = (sigmas * flow_shift) / (1 + (flow_shift - 1) * sigmas)

    padding_mask = torch.zeros(1, 1, latent_h, latent_w, dtype=dtype, device=device)
    use_cfg = cfg > 1.0 and neg_crossattn_emb is not None

    # --- Precompute condition path (constant across all steps) ---
    # Run LLM adapter on text embeddings
    if dit.use_llm_adapter and hasattr(dit, 'llm_adapter'):
        pos_context = dit.llm_adapter(
            source_hidden_states=crossattn_emb,
            target_input_ids=t5_ids,
            target_attention_mask=t5_mask,
            source_attention_mask=source_attn_mask,
        )
        pos_context[~t5_mask.bool()] = 0
        if use_cfg:
            neg_context = dit.llm_adapter(
                source_hidden_states=neg_crossattn_emb,
                target_input_ids=neg_t5_ids,
                target_attention_mask=neg_t5_mask,
                source_attention_mask=neg_source_attn_mask,
            )
            neg_context[~neg_t5_mask.bool()] = 0
        else:
            neg_context = None
    else:
        pos_context = crossattn_emb
        neg_context = neg_crossattn_emb

    # Embed condition through PatchEmbed
    pad_cond = torch.zeros(1, 1, 1, control_latents_5d.shape[3], control_latents_5d.shape[4],
                           device=device, dtype=dtype)
    cond_with_mask = torch.cat([control_latents_5d, pad_cond], dim=1)  # (1, 17, 1, Hc, Wc)
    cond_embedded = dit.x_embedder(cond_with_mask)  # (1, 1, Hc_patch, Wc_patch, 2048)
    _, Tc, Hc, Wc, D = cond_embedded.shape

    # Condition timestep embedding (t=0)
    cond_t_emb, cond_adaln_lora = compute_condition_t_embedding(
        dit.t_embedder, dit.t_embedding_norm, 1, device, dtype
    )

    for step_i in tqdm(range(steps), desc="Sampling"):
        sigma = sigmas[step_i]
        t = sigma.unsqueeze(0)

        def run_forward(noisy_x, context_emb):
            """Run one EasyControl forward pass."""
            # Prepare noise embedded sequence
            x_5d, noise_rope, extra_pos_emb = dit.prepare_embedded_sequence(
                noisy_x, fps=None, padding_mask=padding_mask,
            )
            B, T, H, W, D = x_5d.shape

            # Noise timestep embedding
            t_1d = t.unsqueeze(1) if t.ndim == 1 else t
            t_emb, adaln_lora = dit.t_embedder(t_1d)
            t_emb = dit.t_embedding_norm(t_emb)

            # Condition RoPE
            cond_rope = generate_condition_rope(dit.pos_embedder, (H, W), (Hc, Wc), device)

            # Run through all blocks
            cond_state = cond_embedded.clone()

            for block_idx, block in enumerate(dit.blocks):
                # AdaLN modulation
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

                noise_out, cond_out = control_processors[block_idx](
                    block.self_attn, noise_flat, cond_flat, noise_rope, cond_rope,
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

            # Final layer (noise only)
            x_out = dit.final_layer(x_5d, t_emb, adaln_lora_B_T_3D=adaln_lora)
            return dit.unpatchify(x_out)

        # Positive pass
        pos_out = run_forward(x, pos_context).float()

        if use_cfg:
            neg_out = run_forward(x, neg_context).float()
            model_output = neg_out + cfg * (pos_out - neg_out)
        else:
            model_output = pos_out

        # Euler step
        dt = sigmas[step_i + 1] - sigma
        x = x + model_output * dt
        x = x.to(dtype)

    return x


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Load models
    print("Loading models...")
    dit = load_dit(args.dit, device, dtype)
    vae = WanVAE(vae_pth=args.vae, device='cpu', dtype=dtype)
    text_encoder, tokenizer, t5_tokenizer = load_text_encoder(args.llm, device, dtype)

    # Load EasyControl LoRA
    control_processors = load_control_lora(dit, args.lora)
    control_processors.to(device, dtype).eval()

    # Encode prompts
    print("Encoding prompts...")
    pos_emb, pos_mask, pos_t5, pos_t5_mask = encode_prompt(text_encoder, tokenizer, t5_tokenizer, args.prompt, device)
    neg_emb, neg_mask, neg_t5, neg_t5_mask = None, None, None, None
    if args.cfg > 1.0:
        neg_emb, neg_mask, neg_t5, neg_t5_mask = encode_prompt(
            text_encoder, tokenizer, t5_tokenizer, args.negative_prompt or "", device
        )

    # Free text encoder
    text_encoder.to("cpu")
    torch.cuda.empty_cache()

    # Encode control image
    print(f"Encoding control: {args.control_image}")
    vae.model.to(device)
    vae.mean = vae.mean.to(device)
    vae.std = vae.std.to(device)
    vae.scale = [vae.mean, 1.0 / vae.std]
    control_latents = encode_control(vae, args.control_image, args.height, args.width, device, dtype)
    print(f"  Control latents: {control_latents.shape}")

    # Sample
    print(f"Sampling {args.width}x{args.height}, {args.steps} steps, CFG={args.cfg}")
    latents = sample_easycontrol(
        dit, control_processors,
        pos_emb, pos_mask, pos_t5, pos_t5_mask,
        neg_emb, neg_mask, neg_t5, neg_t5_mask,
        control_latents,
        args.height, args.width, args.steps, args.cfg, args.flow_shift, args.seed,
        device, dtype,
    )

    # Decode
    print("Decoding...")
    latents_4d = latents.squeeze(2)
    mean = vae.mean.view(1, -1, 1, 1)
    std_inv = (1.0 / vae.std).view(1, -1, 1, 1)
    latents_denorm = latents_4d / std_inv + mean
    latents_denorm = latents_denorm.unsqueeze(2)

    with torch.no_grad():
        pixels = vae.model.decode(latents_denorm.to(dtype))
    pixels = pixels.squeeze(2).float().clamp(-1, 1)
    pixels = ((pixels + 1) / 2 * 255).byte()
    pixels = pixels[0].permute(1, 2, 0).cpu().numpy()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    Image.fromarray(pixels).save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
