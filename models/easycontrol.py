# EasyControl spatial control for Anima — diffusion-pipe plugin.
#
# Injects trainable LoRA adapters into the frozen DiT self-attention blocks
# so that a condition image (canny, depth, etc.) guides generation through
# joint noise+condition attention with causal masking.

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
from einops import rearrange, repeat

from models.cosmos_predict2 import (
    CosmosPredict2Pipeline,
    InitialLayer,
    LLMAdapterLayer,
    TransformerLayer,
    FinalLayer,
    vae_encode,
    _compute_text_embeddings,
    _tokenize,
    time_shift,
    get_lin_function,
)
from models.cosmos_predict2_modeling import (
    Attention,
    apply_rotary_pos_emb,
    VideoRopePosition3DEmb,
)
from models.base import make_contiguous
from utils.common import AUTOCAST_DTYPE, is_main_process


# ---------------------------------------------------------------------------
# 1. AnimaLoRALinearLayer
# ---------------------------------------------------------------------------

class AnimaLoRALinearLayer(nn.Module):
    """LoRA layer with binary masking that only affects condition tokens."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        cond_size: int = 1024,
        number: int = 0,
        n_loras: int = 1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.network_alpha = network_alpha
        self.cond_size = cond_size
        self.number = number
        self.n_loras = n_loras

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        nn.init.normal_(self.down.weight, std=1.0 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor, cond_size: Optional[int] = None) -> torch.Tensor:
        B, seq_len, _ = hidden_states.shape
        # Dynamic cond_size: use runtime value if provided, else fall back to init value
        cs = cond_size if cond_size is not None else self.cond_size
        noise_len = seq_len - cs * self.n_loras

        mask = torch.zeros(B, seq_len, 1, device=hidden_states.device, dtype=hidden_states.dtype)
        cond_start = noise_len + self.number * cs
        cond_end = cond_start + cs
        mask[:, cond_start:cond_end, :] = 1.0

        hidden_states = hidden_states * mask
        out = self.up(self.down(hidden_states))
        if self.network_alpha is not None:
            out = out * (self.network_alpha / self.rank)
        return out


# ---------------------------------------------------------------------------
# 2. build_causal_attn_mask
# ---------------------------------------------------------------------------

def build_causal_attn_mask(
    noise_len: int,
    cond_size: int,
    n_conds: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a causal attention mask for joint noise+condition SDPA.

    Noise rows attend to ALL tokens.  Cond_i rows attend ONLY to their own block.
    Returns (1, 1, L_total, L_total).
    """
    total_len = noise_len + cond_size * n_conds
    mask = torch.full((total_len, total_len), -1e20, device=device, dtype=dtype)
    mask[:noise_len, :] = 0.0
    for i in range(n_conds):
        row_start = noise_len + i * cond_size
        row_end = row_start + cond_size
        mask[row_start:row_end, row_start:row_end] = 0.0
    return mask.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# 3. AnimaControlSelfAttn
# ---------------------------------------------------------------------------

class AnimaControlSelfAttn(nn.Module):
    """Joint noise+condition self-attention with LoRA adapters on Q/K/V/O."""

    def __init__(
        self,
        dim: int = 2048,
        rank: int = 128,
        network_alpha: float = 128.0,
        cond_size: int = 1024,
        n_loras: int = 1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.network_alpha = network_alpha
        self.cond_size = cond_size
        self.n_loras = n_loras

        self.q_loras = nn.ModuleList([
            AnimaLoRALinearLayer(dim, dim, rank, network_alpha, cond_size, i, n_loras)
            for i in range(n_loras)
        ])
        self.k_loras = nn.ModuleList([
            AnimaLoRALinearLayer(dim, dim, rank, network_alpha, cond_size, i, n_loras)
            for i in range(n_loras)
        ])
        self.v_loras = nn.ModuleList([
            AnimaLoRALinearLayer(dim, dim, rank, network_alpha, cond_size, i, n_loras)
            for i in range(n_loras)
        ])
        self.proj_loras = nn.ModuleList([
            AnimaLoRALinearLayer(dim, dim, rank, network_alpha, cond_size, i, n_loras)
            for i in range(n_loras)
        ])

    def forward(
        self,
        base_self_attn: Attention,
        noise_flat: torch.Tensor,
        cond_flat: torch.Tensor,
        noise_rope: torch.Tensor,
        cond_rope: torch.Tensor,
        lora_weights: Optional[list] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if lora_weights is None:
            lora_weights = [1.0] * self.n_loras

        B, L_noise, D = noise_flat.shape
        L_cond = cond_flat.shape[1]

        joint = torch.cat([noise_flat, cond_flat], dim=1)
        joint_rope = torch.cat([noise_rope, cond_rope], dim=0)

        q = base_self_attn.q_proj(joint)
        k = base_self_attn.k_proj(joint)
        v = base_self_attn.v_proj(joint)

        # Pass actual L_cond as cond_size so masking works with dynamic resolutions
        for i in range(self.n_loras):
            w = lora_weights[i]
            if w != 0.0:
                q = q + w * self.q_loras[i](joint, cond_size=L_cond)
                k = k + w * self.k_loras[i](joint, cond_size=L_cond)
                v = v + w * self.v_loras[i](joint, cond_size=L_cond)

        n_heads = base_self_attn.n_heads
        head_dim = base_self_attn.head_dim

        q, k, v = map(
            lambda t: rearrange(t, "b l (h d) -> b l h d", h=n_heads, d=head_dim),
            (q, k, v),
        )

        q = base_self_attn.q_norm(q)
        k = base_self_attn.k_norm(k)
        v = base_self_attn.v_norm(v)

        q = apply_rotary_pos_emb(q, joint_rope, tensor_format="bshd", fused=False)
        k = apply_rotary_pos_emb(k, joint_rope, tensor_format="bshd", fused=False)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if q.dtype != v.dtype:
            target_dtype = v.dtype
            q = q.to(target_dtype)
            k = k.to(target_dtype)

        causal_mask = build_causal_attn_mask(
            noise_len=L_noise,
            cond_size=L_cond,  # dynamic: actual condition token count
            n_conds=self.n_loras,
            device=q.device,
            dtype=q.dtype,
        )

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)
        del q, k, v, causal_mask

        x = x.transpose(1, 2).reshape(B, -1, n_heads * head_dim)

        out = base_self_attn.output_proj(x)
        out = base_self_attn.output_dropout(out)

        for i in range(self.n_loras):
            w = lora_weights[i]
            if w != 0.0:
                out = out + w * self.proj_loras[i](x, cond_size=L_cond)

        del x

        noise_attn_out = out[:, :L_noise, :]
        cond_attn_out = out[:, L_noise:, :]
        return noise_attn_out, cond_attn_out


# ---------------------------------------------------------------------------
# 4. generate_condition_rope
# ---------------------------------------------------------------------------

def generate_condition_rope(
    pos_embedder: VideoRopePosition3DEmb,
    noise_hw: Tuple[int, int],
    cond_hw: Tuple[int, int],
    device: torch.device,
    fps: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate RoPE embeddings for condition tokens with interpolated positions.

    Condition positions are scaled to match the noise coordinate space.
    Returns (T*H_cond*W_cond, 1, 1, head_dim).
    """
    H_noise, W_noise = noise_hw
    H_cond, W_cond = cond_hw
    T = 1

    h_theta = 10000.0 * pos_embedder.h_ntk_factor
    w_theta = 10000.0 * pos_embedder.w_ntk_factor
    t_theta = 10000.0 * pos_embedder.t_ntk_factor

    h_spatial_freqs = 1.0 / (h_theta ** pos_embedder.dim_spatial_range.to(device))
    w_spatial_freqs = 1.0 / (w_theta ** pos_embedder.dim_spatial_range.to(device))
    temporal_freqs = 1.0 / (t_theta ** pos_embedder.dim_temporal_range.to(device))

    if H_cond == 1:
        frac_pos_h = torch.zeros(1, device=device)
    else:
        frac_pos_h = torch.linspace(0, H_noise - 1, H_cond, device=device)

    if W_cond == 1:
        frac_pos_w = torch.zeros(1, device=device)
    else:
        frac_pos_w = torch.linspace(0, W_noise - 1, W_cond, device=device)

    frac_pos_t = torch.zeros(T, device=device)

    half_emb_h = torch.outer(frac_pos_h, h_spatial_freqs)
    half_emb_w = torch.outer(frac_pos_w, w_spatial_freqs)
    half_emb_t = torch.outer(frac_pos_t, temporal_freqs)

    if pos_embedder.enable_fps_modulation and fps is not None:
        half_emb_t = torch.outer(
            frac_pos_t / fps[:1].to(device) * pos_embedder.base_fps,
            temporal_freqs,
        )

    em_T_H_W_D = torch.cat(
        [
            repeat(half_emb_t, "t d -> t h w d", h=H_cond, w=W_cond),
            repeat(half_emb_h, "h d -> t h w d", t=T, w=W_cond),
            repeat(half_emb_w, "w d -> t h w d", t=T, h=H_cond),
        ]
        * 2,
        dim=-1,
    )

    return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()


# ---------------------------------------------------------------------------
# 5. compute_condition_t_embedding
# ---------------------------------------------------------------------------

def compute_condition_t_embedding(
    t_embedder: nn.Module,
    t_embedding_norm: nn.Module,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute timestep embedding for the condition branch at t=0."""
    timesteps_B_T = torch.zeros(batch_size, 1, device=device, dtype=dtype)
    t_embedding_B_T_D, adaln_lora_B_T_3D = t_embedder(timesteps_B_T)
    t_embedding_B_T_D = t_embedding_norm(t_embedding_B_T_D)
    return t_embedding_B_T_D, adaln_lora_B_T_3D


# ---------------------------------------------------------------------------
# 6. ControlInitialLayer
# ---------------------------------------------------------------------------

class ControlInitialLayer(nn.Module):
    """Extends InitialLayer to also embed condition latents and generate condition RoPE."""

    def __init__(self, model, text_encoder, is_generic_llm):
        super().__init__()
        self.x_embedder = model.x_embedder
        self.pos_embedder = model.pos_embedder
        if model.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder = model.extra_pos_embedder
        self.t_embedder = model.t_embedder
        self.t_embedding_norm = model.t_embedding_norm
        self.text_encoder = text_encoder
        self.model = [model]
        self.is_generic_llm = is_generic_llm

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_C_T_H_W, timesteps_B_T, *rest = inputs
        # Last tensor in the tuple is the control latents
        prompt_embeds_or_batch_encoding = rest[:-1]
        control_latents_B_C_T_Hc_Wc = rest[-1]

        if torch.is_floating_point(prompt_embeds_or_batch_encoding[0]):
            crossattn_emb, attn_mask, t5_input_ids, t5_attn_mask = prompt_embeds_or_batch_encoding
        else:
            with torch.no_grad():
                input_ids, attn_mask, t5_input_ids, t5_attn_mask = prompt_embeds_or_batch_encoding
                crossattn_emb = _compute_text_embeddings(self.text_encoder[0], input_ids, attn_mask, is_generic_llm=self.is_generic_llm)

        # --- Noise path (same as InitialLayer) ---
        padding_mask = torch.zeros(x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[3], x_B_C_T_H_W.shape[4], dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.model[0].prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=None,
            padding_mask=padding_mask,
        )
        assert extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is None
        assert rope_emb_L_1_1_D is not None

        B, T, H, W, D = x_B_T_H_W_D.shape

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # --- Condition path ---
        # Embed condition latents through the shared PatchEmbed
        # control_latents already have the padding-mask channel from VAE encoding
        cond_padding_mask = torch.zeros(
            control_latents_B_C_T_Hc_Wc.shape[0], 1,
            control_latents_B_C_T_Hc_Wc.shape[3], control_latents_B_C_T_Hc_Wc.shape[4],
            dtype=control_latents_B_C_T_Hc_Wc.dtype, device=control_latents_B_C_T_Hc_Wc.device,
        )
        # Prepare embedded sequence for condition (using the same model)
        cond_B_T_Hc_Wc_D, _, _ = self.model[0].prepare_embedded_sequence(
            control_latents_B_C_T_Hc_Wc,
            fps=None,
            padding_mask=cond_padding_mask,
        )
        _, Tc, Hc, Wc, _ = cond_B_T_Hc_Wc_D.shape

        # Generate condition RoPE (interpolated to noise coordinate space)
        cond_rope = generate_condition_rope(
            self.pos_embedder, (H, W), (Hc, Wc), x_B_T_H_W_D.device, fps=None,
        )

        # Condition timestep embedding (t=0 for clean condition image)
        cond_t_emb, cond_adaln_lora = compute_condition_t_embedding(
            self.t_embedder, self.t_embedding_norm, B, x_B_T_H_W_D.device, x_B_T_H_W_D.dtype,
        )

        outputs = make_contiguous(
            x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, t5_input_ids, attn_mask, t5_attn_mask,
            rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T,
            cond_B_T_Hc_Wc_D, cond_t_emb, cond_adaln_lora, cond_rope,
        )
        for tensor in outputs:
            if torch.is_floating_point(tensor):
                tensor.requires_grad_(True)
        return outputs


# ---------------------------------------------------------------------------
# 7. ControlLLMAdapterLayer
# ---------------------------------------------------------------------------

class ControlLLMAdapterLayer(nn.Module):
    """LLMAdapterLayer that passes through the extra condition tensors."""

    def __init__(self, llm_adapter):
        super().__init__()
        self.llm_adapter = llm_adapter

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        (x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, t5_input_ids, attn_mask, t5_attn_mask,
         rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T,
         cond_B_T_Hc_Wc_D, cond_t_emb, cond_adaln_lora, cond_rope) = inputs

        if self.llm_adapter is not None:
            crossattn_emb = self.llm_adapter(
                source_hidden_states=crossattn_emb,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=attn_mask,
            )
            crossattn_emb[~t5_attn_mask.bool()] = 0

        return make_contiguous(
            x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb,
            rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T,
            cond_B_T_Hc_Wc_D, cond_t_emb, cond_adaln_lora, cond_rope,
        )


# ---------------------------------------------------------------------------
# 8. ControlTransformerLayer
# ---------------------------------------------------------------------------

class ControlTransformerLayer(nn.Module):
    """Replaces TransformerLayer with EasyControl block logic.

    Runs joint noise+condition self-attention via LoRA, cross-attention on
    noise only, and MLP on both with separate AdaLN modulation.
    """

    def __init__(self, block, block_idx, offloader, control_processor):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader
        self.control_processor = control_processor

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        (x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb,
         rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T,
         cond_B_T_Hc_Wc_D, cond_t_emb, cond_adaln_lora, cond_rope) = inputs

        self.offloader.wait_for_block(self.block_idx)

        block = self.block
        B, T, H, W, D = x_B_T_H_W_D.shape
        _, Tc, Hc, Wc, _ = cond_B_T_Hc_Wc_D.shape

        # --- AdaLN modulation ---
        if block.use_adaln_lora:
            shift_sa, scale_sa, gate_sa = (
                block.adaln_modulation_self_attn(t_embedding_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_ca, scale_ca, gate_ca = (
                block.adaln_modulation_cross_attn(t_embedding_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = (
                block.adaln_modulation_mlp(t_embedding_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            cond_shift_sa, cond_scale_sa, cond_gate_sa = (
                block.adaln_modulation_self_attn(cond_t_emb) + cond_adaln_lora
            ).chunk(3, dim=-1)
            cond_shift_mlp, cond_scale_mlp, cond_gate_mlp = (
                block.adaln_modulation_mlp(cond_t_emb) + cond_adaln_lora
            ).chunk(3, dim=-1)
        else:
            shift_sa, scale_sa, gate_sa = block.adaln_modulation_self_attn(
                t_embedding_B_T_D
            ).chunk(3, dim=-1)
            shift_ca, scale_ca, gate_ca = block.adaln_modulation_cross_attn(
                t_embedding_B_T_D
            ).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = block.adaln_modulation_mlp(
                t_embedding_B_T_D
            ).chunk(3, dim=-1)
            cond_shift_sa, cond_scale_sa, cond_gate_sa = block.adaln_modulation_self_attn(
                cond_t_emb
            ).chunk(3, dim=-1)
            cond_shift_mlp, cond_scale_mlp, cond_gate_mlp = block.adaln_modulation_mlp(
                cond_t_emb
            ).chunk(3, dim=-1)

        # Reshape for broadcasting: (B, T, D) -> (B, T, 1, 1, D)
        shift_sa_5d = rearrange(shift_sa, "b t d -> b t 1 1 d")
        scale_sa_5d = rearrange(scale_sa, "b t d -> b t 1 1 d")
        gate_sa_5d = rearrange(gate_sa, "b t d -> b t 1 1 d")

        shift_ca_5d = rearrange(shift_ca, "b t d -> b t 1 1 d")
        scale_ca_5d = rearrange(scale_ca, "b t d -> b t 1 1 d")
        gate_ca_5d = rearrange(gate_ca, "b t d -> b t 1 1 d")

        shift_mlp_5d = rearrange(shift_mlp, "b t d -> b t 1 1 d")
        scale_mlp_5d = rearrange(scale_mlp, "b t d -> b t 1 1 d")
        gate_mlp_5d = rearrange(gate_mlp, "b t d -> b t 1 1 d")

        cond_shift_sa_5d = rearrange(cond_shift_sa, "b t d -> b t 1 1 d")
        cond_scale_sa_5d = rearrange(cond_scale_sa, "b t d -> b t 1 1 d")
        cond_gate_sa_5d = rearrange(cond_gate_sa, "b t d -> b t 1 1 d")

        cond_shift_mlp_5d = rearrange(cond_shift_mlp, "b t d -> b t 1 1 d")
        cond_scale_mlp_5d = rearrange(cond_scale_mlp, "b t d -> b t 1 1 d")
        cond_gate_mlp_5d = rearrange(cond_gate_mlp, "b t d -> b t 1 1 d")

        # --- 1. Self-attention with condition injection ---
        norm_noise = block.layer_norm_self_attn(x_B_T_H_W_D) * (1 + scale_sa_5d) + shift_sa_5d
        norm_cond = block.layer_norm_self_attn(cond_B_T_Hc_Wc_D) * (1 + cond_scale_sa_5d) + cond_shift_sa_5d

        noise_flat = rearrange(norm_noise, "b t h w d -> b (t h w) d")
        cond_flat = rearrange(norm_cond, "b t hc wc d -> b (t hc wc) d")

        noise_attn_out, cond_attn_out = self.control_processor(
            block.self_attn, noise_flat, cond_flat, rope_emb_L_1_1_D, cond_rope,
        )

        noise_attn_out = rearrange(noise_attn_out, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        cond_attn_out = rearrange(cond_attn_out, "b (t hc wc) d -> b t hc wc d", t=Tc, hc=Hc, wc=Wc)

        x_B_T_H_W_D = x_B_T_H_W_D + gate_sa_5d * noise_attn_out
        cond_B_T_Hc_Wc_D = cond_B_T_Hc_Wc_D + cond_gate_sa_5d * cond_attn_out

        # --- 2. Cross-attention (noise only) ---
        norm_x_ca = block.layer_norm_cross_attn(x_B_T_H_W_D) * (1 + scale_ca_5d) + shift_ca_5d
        ca_result = rearrange(
            block.cross_attn(
                rearrange(norm_x_ca, "b t h w d -> b (t h w) d"),
                crossattn_emb,
                rope_emb=rope_emb_L_1_1_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T, h=H, w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_ca_5d * ca_result

        # --- 3. MLP (both noise and cond, with separate AdaLN) ---
        norm_x_mlp = block.layer_norm_mlp(x_B_T_H_W_D) * (1 + scale_mlp_5d) + shift_mlp_5d
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_5d * block.mlp(norm_x_mlp)

        norm_cond_mlp = block.layer_norm_mlp(cond_B_T_Hc_Wc_D) * (1 + cond_scale_mlp_5d) + cond_shift_mlp_5d
        cond_B_T_Hc_Wc_D = cond_B_T_Hc_Wc_D + cond_gate_mlp_5d * block.mlp(norm_cond_mlp)

        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(
            x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb,
            rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T,
            cond_B_T_Hc_Wc_D, cond_t_emb, cond_adaln_lora, cond_rope,
        )


# ---------------------------------------------------------------------------
# 9. ControlFinalLayer
# ---------------------------------------------------------------------------

class ControlFinalLayer(nn.Module):
    """FinalLayer that ignores the extra condition tensors."""

    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        (x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb,
         rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T,
         cond_B_T_Hc_Wc_D, cond_t_emb, cond_adaln_lora, cond_rope) = inputs
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        net_output_B_C_T_H_W = self.unpatchify(x_B_T_H_W_O)
        return net_output_B_C_T_H_W


# ---------------------------------------------------------------------------
# 10. EasyControlPipeline
# ---------------------------------------------------------------------------

class EasyControlPipeline(CosmosPredict2Pipeline):
    """EasyControl pipeline for spatial condition injection via LoRA."""

    name = 'easycontrol'
    checkpointable_layers = ['ControlTransformerLayer']

    def __init__(self, config):
        super().__init__(config)
        self.control_config = config.get('control', {})
        self.control_rank = self.control_config.get('rank', 128)
        self.control_alpha = self.control_config.get('alpha', 128.0)
        self.control_n_loras = self.control_config.get('n_loras', 1)
        self.control_cond_size = self.control_config.get('cond_size', 512)

    def load_diffusion_model(self):
        super().load_diffusion_model()

        # Freeze the entire base model
        self.transformer.requires_grad_(False)

        # Compute condition token count: cond_size -> VAE (//8) -> PatchEmbed (//2)
        cond_patch = self.control_cond_size // 8 // self.transformer.patch_spatial
        cond_token_count = cond_patch * cond_patch

        dim = self.transformer.model_channels
        num_blocks = self.transformer.num_blocks

        self.control_processors = nn.ModuleList([
            AnimaControlSelfAttn(
                dim=dim,
                rank=self.control_rank,
                network_alpha=self.control_alpha,
                cond_size=cond_token_count,
                n_loras=self.control_n_loras,
            )
            for _ in range(num_blocks)
        ])

        # Tag parameters with original_name for get_param_groups
        for name, p in self.control_processors.named_parameters():
            p.original_name = f'control_processors.{name}'

        trainable_params = sum(p.numel() for p in self.control_processors.parameters())
        if is_main_process():
            print(f'EasyControl: rank={self.control_rank}, alpha={self.control_alpha}, '
                  f'n_loras={self.control_n_loras}, cond_size={self.control_cond_size}, '
                  f'cond_tokens={cond_token_count}, blocks={num_blocks}')
            print(f'EasyControl: trainable LoRA parameters: {trainable_params:,}')

    def load_adapter_weights(self, path):
        """Load control LoRA weights from a previous checkpoint."""
        sd = safetensors.torch.load_file(str(path), device='cpu')
        # Strip 'control_processors.' prefix if present
        clean_sd = {}
        for k, v in sd.items():
            clean_k = k.replace('control_processors.', '') if k.startswith('control_processors.') else k
            clean_sd[clean_k] = v
        info = self.control_processors.load_state_dict(clean_sd, strict=False)
        if is_main_process():
            if info.missing_keys:
                print(f'EasyControl load_adapter_weights: {len(info.missing_keys)} missing keys')
            if info.unexpected_keys:
                print(f'EasyControl load_adapter_weights: {len(info.unexpected_keys)} unexpected keys')
            print(f'Loaded {len(sd)} control LoRA tensors from {path}')

    def get_call_vae_fn(self, vae):
        """Encode both target and control images through the VAE (levzzz pattern)."""
        def fn(*args):
            p = next(vae.parameters())
            if len(args) == 1:
                tensor = args[0]
                tensor = tensor.to(p.device, p.dtype)
                latents = vae_encode(tensor, self.vae)
                return {'latents': latents}
            elif len(args) == 2:
                tensor, control_tensor = args
                tensor = tensor.to(p.device, p.dtype)
                control_tensor = control_tensor.to(p.device, p.dtype)
                latents = vae_encode(tensor, self.vae)
                control_latents = vae_encode(control_tensor, self.vae)
                return {'latents': latents, 'control_latents': control_latents}
            else:
                raise RuntimeError(f'Unexpected number of args: {len(args)}')
        return fn

    def prepare_inputs(self, inputs, timestep_quantile=None):
        """Same as parent but appends control_latents to model input tuple."""
        latents = inputs['latents'].float()
        mask = inputs['mask']

        if self.cache_text_embeddings:
            prompt_embeds_or_batch_encoding = (inputs['prompt_embeds'], inputs['attn_mask'], inputs['t5_input_ids'], inputs['t5_attn_mask'])
        else:
            captions = inputs['caption']
            batch_encoding = _tokenize(self.tokenizer, captions)
            t5_batch_encoding = _tokenize(self.t5_tokenizer, captions)
            prompt_embeds_or_batch_encoding = (batch_encoding.input_ids, batch_encoding.attention_mask, t5_batch_encoding.input_ids, t5_batch_encoding.attention_mask)

        bs, channels, num_frames, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')
            mask = mask.unsqueeze(2)

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')

        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=latents.device))
        else:
            t = dist.sample((bs,)).to(latents.device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents
        t = t.view(-1, 1)

        # Append control latents as a separate tensor (NOT temporal concat)
        control_latents = inputs['control_latents'].float()

        return (noisy_latents, t, *prompt_embeds_or_batch_encoding, control_latents), (target, mask)

    def to_layers(self):
        transformer = self.transformer
        text_encoder = None if self.cache_text_embeddings else self.text_encoder
        layers = [
            ControlInitialLayer(transformer, text_encoder, self.is_generic_llm),
            ControlLLMAdapterLayer(transformer.llm_adapter if self.use_llm_adapter else None),
        ]
        for i, block in enumerate(transformer.blocks):
            layers.append(ControlTransformerLayer(
                block, i, self.offloader, self.control_processors[i],
            ))
        layers.append(ControlFinalLayer(transformer))
        return layers

    def save_adapter(self, save_dir, peft_state_dict):
        """Save the control LoRA weights."""
        state_dict = {}
        for name, param in self.control_processors.named_parameters():
            state_dict[f'control_processors.{name}'] = param.detach().cpu().contiguous()

        metadata = {
            'format': 'pt',
            'rank': str(self.control_rank),
            'network_alpha': str(self.control_alpha),
            'cond_size': str(self.control_cond_size),
            'n_loras': str(self.control_n_loras),
        }
        safetensors.torch.save_file(state_dict, save_dir / 'adapter_model.safetensors', metadata=metadata)

    def save_model(self, save_dir, state_dict):
        """Save the full model state (base + control LoRA)."""
        # Save base model
        base_state_dict = {'net.' + k: v for k, v in state_dict.items()}
        safetensors.torch.save_file(base_state_dict, save_dir / 'model.safetensors', metadata={'format': 'pt'})
        # Also save the control LoRA separately
        self.save_adapter(save_dir, None)

    def get_param_groups(self, parameters):
        """All trainable params are control LoRA params — single group."""
        base_lr = self.config['optimizer'].get('lr', None)
        control_lr = self.control_config.get('lr', base_lr)

        param_groups = []
        params = [p for p in parameters if p.requires_grad]
        if params:
            param_groups.append({'params': params, 'lr': control_lr})

        if is_main_process():
            print(f'EasyControl param groups: lr={control_lr}, num_params={len(params)}')

        return param_groups
