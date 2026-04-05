#!/usr/bin/env python3
"""
Anima inference with optional control conditioning (canny, depth, etc.)

Supports both:
  - Normal Anima generation (no control)
  - Control LoRA generation (levzzz temporal concat approach)

Usage:
  # Normal generation
  python infer_control.py \
    --dit /path/to/anima-preview2.safetensors \
    --vae /path/to/qwen_image_vae.safetensors \
    --llm /path/to/qwen_3_06b_base.safetensors \
    --prompt "1girl, standing in a field" \
    --output output.png

  # With control LoRA (canny)
  python infer_control.py \
    --dit /path/to/anima-preview2.safetensors \
    --vae /path/to/qwen_image_vae.safetensors \
    --llm /path/to/qwen_3_06b_base.safetensors \
    --lora /path/to/adapter_model.safetensors \
    --control_image /path/to/canny_edge.png \
    --prompt "1girl, standing in a field" \
    --output output.png
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add diffusion-pipe root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from models.cosmos_predict2 import (
    CosmosPredict2Pipeline,
    get_dit_config,
    vae_encode,
    WanVAE,
    _tokenize,
    _compute_text_embeddings,
)
from models.cosmos_predict2_modeling import MiniTrainDIT
from utils.common import load_state_dict

import transformers
from transformers import AutoTokenizer, T5TokenizerFast
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device


def parse_args():
    p = argparse.ArgumentParser(description="Anima inference with optional control")

    # Model paths
    p.add_argument("--dit", required=True, help="Path to Anima DiT .safetensors")
    p.add_argument("--vae", required=True, help="Path to Qwen-Image VAE .safetensors")
    p.add_argument("--llm", required=True, help="Path to Qwen3-0.6B (dir or .safetensors)")

    # LoRA / Control
    p.add_argument("--lora", default=None, help="Path to LoRA adapter .safetensors (for control)")
    p.add_argument("--lora_weight", type=float, default=1.0, help="LoRA weight multiplier")
    p.add_argument("--control_image", default=None, help="Path to control image (canny, depth, etc.)")

    # Generation settings
    p.add_argument("--prompt", required=True, help="Positive prompt")
    p.add_argument("--negative_prompt", default="", help="Negative prompt")
    p.add_argument("--width", type=int, default=512, help="Output width (divisible by 32)")
    p.add_argument("--height", type=int, default=512, help="Output height (divisible by 32)")
    p.add_argument("--steps", type=int, default=30, help="Number of sampling steps")
    p.add_argument("--cfg", type=float, default=5.0, help="CFG guidance scale (1.0 = no CFG)")
    p.add_argument("--flow_shift", type=float, default=3.0, help="Flow shift for sigma schedule")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output
    p.add_argument("--output", default="output.png", help="Output image path")

    return p.parse_args()


def load_models(args, device, dtype):
    """Load all models to device."""

    # --- VAE ---
    print("Loading VAE...")
    vae = WanVAE(vae_pth=args.vae, device='cpu', dtype=dtype)
    vae.model.to(device)

    # --- Text encoder (Qwen3-0.6B) ---
    print("Loading Qwen3 text encoder...")
    llm_path = args.llm
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
    text_encoder = text_encoder.model  # Use the inner model, not the LM head
    text_encoder.config.use_cache = False
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_encoder.to(device).eval().requires_grad_(False)

    t5_tokenizer = T5TokenizerFast(
        vocab_file='configs/t5_old/spiece.model',
        tokenizer_file='configs/t5_old/tokenizer.json',
    )

    # --- DiT ---
    print("Loading Anima DiT...")
    state_dict = load_state_dict(args.dit)
    # Remove 'net.' prefix if present
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

    # --- LoRA (optional) ---
    if args.lora:
        print(f"Loading LoRA: {args.lora}")
        import peft
        adapter_config = peft.LoraConfig(
            r=32,  # will be overridden by checkpoint
            lora_alpha=32,
            target_modules=['Block', 'TransformerBlock'],
            lora_dropout=0.0,
        )
        dit = peft.get_peft_model(dit, adapter_config)
        # Load the actual weights
        lora_sd = load_state_dict(args.lora)
        # Handle diffusion_model. prefix from ComfyUI format
        clean_sd = {}
        for k, v in lora_sd.items():
            k = k.replace('diffusion_model.', '')
            clean_sd[k] = v
        dit.load_state_dict(clean_sd, strict=False)
        if args.lora_weight != 1.0:
            for name, module in dit.named_modules():
                if hasattr(module, 'scaling'):
                    module.scaling = {k: v * args.lora_weight for k, v in module.scaling.items()}
        dit.eval()
        print(f"  LoRA loaded (weight={args.lora_weight})")

    return dit, vae, text_encoder, tokenizer, t5_tokenizer


def encode_prompt(text_encoder, tokenizer, t5_tokenizer, prompt, device):
    """Encode a prompt through Qwen3 + T5 tokenizer."""
    batch_encoding = _tokenize(tokenizer, [prompt])
    t5_encoding = _tokenize(t5_tokenizer, [prompt])

    with torch.no_grad():
        embeds = _compute_text_embeddings(text_encoder, batch_encoding.input_ids, batch_encoding.attention_mask)
    embeds = embeds.to(device)
    attn_mask = batch_encoding.attention_mask.to(device)
    t5_ids = t5_encoding.input_ids.to(device)
    t5_mask = t5_encoding.attention_mask.to(device)

    return embeds, attn_mask, t5_ids, t5_mask


def encode_control_image(vae, image_path, target_h, target_w, device, dtype):
    """Encode a control image through VAE and return latents."""
    img = Image.open(image_path).convert("RGB")
    # Resize to match target latent dimensions
    img = img.resize((target_w, target_h), Image.LANCZOS)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device, dtype)  # (1, 3, H, W)
    with torch.no_grad():
        latents = vae_encode(tensor, vae)  # (1, 16, 1, H/8, W/8) or (1, 16, H/8, W/8)
    if latents.dim() == 4:
        latents = latents.unsqueeze(2)  # Ensure 5D: (1, 16, 1, H/8, W/8)
    return latents


@torch.no_grad()
def sample(
    dit, crossattn_emb, neg_crossattn_emb,
    height, width, steps, cfg, flow_shift, seed,
    device, dtype,
    control_latents=None,
    source_attention_mask=None, neg_source_attention_mask=None,
    t5_input_ids=None, t5_attn_mask=None,
    neg_t5_input_ids=None, neg_t5_attn_mask=None,
):
    """Euler discrete sampling with optional control conditioning."""
    latent_h = height // 8
    latent_w = width // 8

    # Start from pure noise
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(1, 16, 1, latent_h, latent_w, dtype=torch.float32, generator=generator)
    x = noise.to(device, dtype)

    # Sigma schedule with flow shift
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
    if flow_shift != 1.0:
        sigmas = (sigmas * flow_shift) / (1 + (flow_shift - 1) * sigmas)

    # Padding mask
    padding_mask = torch.zeros(1, 1, latent_h, latent_w, dtype=dtype, device=device)

    use_cfg = cfg > 1.0 and neg_crossattn_emb is not None

    for i in tqdm(range(steps), desc="Sampling"):
        sigma = sigmas[i]
        t = sigma.unsqueeze(0)

        # Current noisy latents — if control, concat as frame 1
        if control_latents is not None:
            x_input = torch.cat([x, control_latents.to(x.dtype)], dim=2)  # (1, 16, 2, H/8, W/8)
        else:
            x_input = x

        # Positive pass
        pos_out = dit(
            x_input, t, crossattn_emb, padding_mask=padding_mask,
            source_attention_mask=source_attention_mask,
            target_input_ids=t5_input_ids,
            target_attention_mask=t5_attn_mask,
        )
        # Extract frame 0 only if control was used
        if control_latents is not None and pos_out.shape[2] > 1:
            pos_out = pos_out[:, :, :1, :, :]
        pos_out = pos_out.float()

        if use_cfg:
            # Negative pass (same control if present)
            neg_out = dit(
                x_input, t, neg_crossattn_emb, padding_mask=padding_mask,
                source_attention_mask=neg_source_attention_mask,
                target_input_ids=neg_t5_input_ids,
                target_attention_mask=neg_t5_attn_mask,
            )
            if control_latents is not None and neg_out.shape[2] > 1:
                neg_out = neg_out[:, :, :1, :, :]
            neg_out = neg_out.float()

            model_output = neg_out + cfg * (pos_out - neg_out)
        else:
            model_output = pos_out

        # Euler step
        dt = sigmas[i + 1] - sigma
        x = x + model_output * dt
        x = x.to(dtype)

    return x


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Load models
    dit, vae, text_encoder, tokenizer, t5_tokenizer = load_models(args, device, dtype)

    # Encode prompts
    print("Encoding prompt...")
    pos_embeds, pos_mask, pos_t5_ids, pos_t5_mask = encode_prompt(
        text_encoder, tokenizer, t5_tokenizer, args.prompt, device
    )

    neg_embeds, neg_mask, neg_t5_ids, neg_t5_mask = None, None, None, None
    if args.cfg > 1.0:
        neg_embeds, neg_mask, neg_t5_ids, neg_t5_mask = encode_prompt(
            text_encoder, tokenizer, t5_tokenizer, args.negative_prompt or "", device
        )

    # Free text encoder
    text_encoder.to("cpu")
    torch.cuda.empty_cache()

    # Encode control image if provided
    control_latents = None
    if args.control_image:
        print(f"Encoding control image: {args.control_image}")
        control_latents = encode_control_image(vae, args.control_image, args.height, args.width, device, dtype)
        print(f"  Control latents shape: {control_latents.shape}")

    # Run sampling
    print(f"Sampling {args.width}x{args.height}, {args.steps} steps, CFG={args.cfg}, shift={args.flow_shift}, seed={args.seed}")
    latents = sample(
        dit, pos_embeds, neg_embeds,
        args.height, args.width, args.steps, args.cfg, args.flow_shift, args.seed,
        device, dtype,
        control_latents=control_latents,
        source_attention_mask=pos_mask, neg_source_attention_mask=neg_mask,
        t5_input_ids=pos_t5_ids, t5_attn_mask=pos_t5_mask,
        neg_t5_input_ids=neg_t5_ids, neg_t5_attn_mask=neg_t5_mask,
    )

    # Decode to image
    print("Decoding latents...")
    # Move VAE to GPU for decode
    vae.model.to(device)
    vae.mean = vae.mean.to(device)
    vae.std = vae.std.to(device)
    vae.scale = [vae.mean, 1.0 / vae.std]

    # Denormalize latents
    latents_for_decode = latents.squeeze(2)  # (1, 16, H/8, W/8)
    mean = vae.mean.view(1, -1, 1, 1)
    std_inv = (1.0 / vae.std).view(1, -1, 1, 1)
    latents_denorm = latents_for_decode / std_inv + mean
    latents_denorm = latents_denorm.unsqueeze(2)  # (1, 16, 1, H/8, W/8)

    with torch.no_grad():
        pixels = vae.model.decode(latents_denorm.to(dtype))  # (1, 3, 1, H, W)
    pixels = pixels.squeeze(2).float().clamp(-1, 1)  # (1, 3, H, W)

    # Save image
    pixels = ((pixels + 1) / 2 * 255).byte()
    pixels = pixels[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    img = Image.fromarray(pixels)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    img.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
