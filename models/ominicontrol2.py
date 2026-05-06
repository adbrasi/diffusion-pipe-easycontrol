"""
OminiControl2 training pipeline for Anima (Cosmos-Predict2 DiT).

Extends OminiControl v1 with two key improvements:
1. Independent condition: asymmetric attention mask where condition tokens
   do NOT attend to noise/text tokens. This makes condition K/V deterministic
   across denoising steps, enabling KV cache at inference.
2. Compact token representation: condition image encoded at lower resolution
   with position_scale to remap RoPE coordinates. The RoPE is recomputed
   from scaled coordinate grids (not by scaling precomputed tensors).

Training differences from v1:
- Attention mask: condition→noise BLOCKED (condition only sees itself)
- Position scale: condition RoPE computed from coords * position_scale
- Condition can be at lower resolution than target

Based on: OminiControl v2 (arXiv:2503.08280)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from models.ominicontrol import OminiControlPipeline, OminiControlInitialLayer
from models.cosmos_predict2 import (
    InitialLayer,
    TransformerLayer,
    FinalLayer,
    LLMAdapterLayer,
    _tokenize,
    _compute_text_embeddings,
)
from models.base import make_contiguous
from utils.common import AUTOCAST_DTYPE


class OminiControl2Pipeline(OminiControlPipeline):
    """OminiControl2 training pipeline for Anima.

    Extends v1 with independent_condition and position_scale.

    Config options under [ominicontrol]:
        position_mode: 'spatial' | 'subject' (default: 'spatial')
        condition_dropout: float (default: 0.1)
        condition_timestep: float (default: 0.0)
        independent_condition: bool (default: true)
            Asymmetric attention mask: condition cannot attend to noise.
            Required for KV cache at inference.
        position_scale: float (default: 1.0)
            Scale factor for condition RoPE positions.
            Use 2.0 with condition at half resolution for compact tokens.
    """

    def __init__(self, config):
        super().__init__(config)
        oc_config = config.get('ominicontrol', {})
        self.independent_condition = oc_config.get('independent_condition', True)
        self.position_scale = oc_config.get('position_scale', 1.0)

    def to_layers(self):
        transformer = self.transformer
        text_encoder = None if self.cache_text_embeddings else self.text_encoder
        layers = [
            OminiControl2InitialLayer(
                transformer, text_encoder, self.is_generic_llm,
                position_mode=self.position_mode,
                position_scale=self.position_scale,
                independent_condition=self.independent_condition,
            ),
            LLMAdapterLayer(transformer.llm_adapter if self.use_llm_adapter else None),
        ]
        for i, block in enumerate(transformer.blocks):
            layers.append(OminiControl2TransformerLayer(
                block, i, self.offloader, self.independent_condition,
            ))
        layers.append(FinalLayer(transformer))
        return layers


class OminiControl2InitialLayer(OminiControlInitialLayer):
    """InitialLayer with OminiControl2 position scale + independent condition flag."""

    def __init__(self, model, text_encoder, is_generic_llm,
                 position_mode='spatial', position_scale=1.0,
                 independent_condition=True):
        super().__init__(model, text_encoder, is_generic_llm, position_mode=position_mode)
        self.position_scale = position_scale
        self.independent_condition = independent_condition

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_C_T_H_W, timesteps_B_T, *prompt_embeds_or_batch_encoding = inputs

        if torch.is_floating_point(prompt_embeds_or_batch_encoding[0]):
            crossattn_emb, attn_mask, t5_input_ids, t5_attn_mask = prompt_embeds_or_batch_encoding
        else:
            with torch.no_grad():
                input_ids, attn_mask, t5_input_ids, t5_attn_mask = prompt_embeds_or_batch_encoding
                crossattn_emb = _compute_text_embeddings(
                    self.text_encoder, input_ids, attn_mask, is_generic_llm=self.is_generic_llm,
                )

        padding_mask = torch.zeros(
            x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[3], x_B_C_T_H_W.shape[4],
            dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device,
        )
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = self.model[0].prepare_embedded_sequence(
            x_B_C_T_H_W, fps=None, padding_mask=padding_mask,
        )
        assert extra_pos_emb is None
        assert rope_emb_L_1_1_D is not None

        T = x_B_C_T_H_W.shape[2]
        if T > 1 and rope_emb_L_1_1_D is not None:
            total_seq = rope_emb_L_1_1_D.shape[0]
            noise_seq_len = total_seq // T

            if self.position_scale != 1.0:
                # Recompute condition RoPE from scaled coordinates.
                # RoPE is e^{i * pos * freq}, so we must scale pos BEFORE
                # computing the outer product, not multiply the final tensor.
                cond_rope = self._compute_scaled_cond_rope(
                    x_B_T_H_W_D, rope_emb_L_1_1_D[:noise_seq_len],
                )
            elif self.position_mode == 'spatial':
                # Spatial: condition shares exact noise positions
                cond_rope = rope_emb_L_1_1_D[:noise_seq_len].clone()
            else:
                # Subject: condition keeps T=1 position (natural separation)
                cond_rope = rope_emb_L_1_1_D[noise_seq_len:]

            rope_emb_L_1_1_D = torch.cat(
                [rope_emb_L_1_1_D[:noise_seq_len], cond_rope], dim=0
            )

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        outputs = make_contiguous(
            x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb,
            t5_input_ids, attn_mask, t5_attn_mask,
            rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T,
        )
        for tensor in outputs:
            if torch.is_floating_point(tensor):
                tensor.requires_grad_(True)
        return outputs

    def _compute_scaled_cond_rope(self, x_B_T_H_W_D, noise_rope_ref):
        """Compute condition RoPE from scaled coordinate grid.

        For position_scale=2.0 and condition at half resolution:
        - Condition has H_c, W_c tokens (e.g., 16x16)
        - Target has H_n, W_n tokens (e.g., 32x32)
        - Each condition coord (h, w) maps to (h * scale, w * scale)
          in target coordinate space

        RoPE formula: emb[i] = pos[i] * freq[d]
        We compute outer(scaled_coords, freqs) directly.
        """
        pos_embedder = self.model[0].pos_embedder
        B, T, H_total, W_total, D = x_B_T_H_W_D.shape

        # Condition is at T=1, so its spatial dims = noise spatial dims
        # (but if cond is at lower resolution, H_c < H_n after patchify)
        # For now, H_c = W_c = H_total = W_total (same resolution after patchify)
        # The position_scale maps these smaller coords to the noise coord space
        H_c = H_total  # condition H in patch space
        W_c = W_total  # condition W in patch space

        device = noise_rope_ref.device

        # Get frequency vectors from the pos_embedder
        h_ntk = getattr(pos_embedder, 'h_ntk_factor', 1.0)
        w_ntk = getattr(pos_embedder, 'w_ntk_factor', 1.0)
        t_ntk = getattr(pos_embedder, 't_ntk_factor', 1.0)

        h_theta = 10000.0 * h_ntk
        w_theta = 10000.0 * w_ntk
        t_theta = 10000.0 * t_ntk

        dim_spatial = pos_embedder.dim_spatial_range.to(device)
        dim_temporal = pos_embedder.dim_temporal_range.to(device)

        h_freqs = 1.0 / (h_theta ** dim_spatial)
        w_freqs = 1.0 / (w_theta ** dim_spatial)
        t_freqs = 1.0 / (t_theta ** dim_temporal)

        # Scaled coordinate grids for condition
        scale = self.position_scale
        # Center the scaled coords: shift by (scale-1)/2 to center them
        # OminiControl2 paper: ids[:, 1:] *= scale; ids[:, 1:] += (scale-1)/2
        scale_bias = (scale - 1.0) / 2.0

        # Condition spatial coords: [0, 1, ..., H_c-1] * scale + bias
        coords_h = (torch.arange(H_c, device=device, dtype=torch.float32) * scale) + scale_bias
        coords_w = (torch.arange(W_c, device=device, dtype=torch.float32) * scale) + scale_bias

        # Temporal coord: 0 for spatial mode, 1 for subject mode
        if self.position_mode == 'spatial':
            coords_t = torch.zeros(1, device=device, dtype=torch.float32)
        else:
            coords_t = torch.ones(1, device=device, dtype=torch.float32)

        # Compute outer products: coords * freqs
        half_emb_h = torch.outer(coords_h, h_freqs)   # (H_c, dim_h/2)
        half_emb_w = torch.outer(coords_w, w_freqs)   # (W_c, dim_w/2)
        half_emb_t = torch.outer(coords_t, t_freqs)   # (1, dim_t/2)

        T_c = 1  # condition is always 1 frame

        # Assemble: same structure as VideoRopePosition3DEmb.generate_embeddings
        em = torch.cat([
            repeat(half_emb_t, "t d -> t h w d", h=H_c, w=W_c),
            repeat(half_emb_h, "h d -> t h w d", t=T_c, w=W_c),
            repeat(half_emb_w, "w d -> t h w d", t=T_c, h=H_c),
        ] * 2, dim=-1)

        # Reshape to match rope_emb format: (T_c*H_c*W_c, 1, 1, head_dim)
        cond_rope = rearrange(em, "t h w d -> (t h w) 1 1 d").float()

        return cond_rope


class OminiControl2TransformerLayer(TransformerLayer):
    """TransformerLayer with independent_condition attention mask.

    When independent_condition=True, monkey-patches the block's self_attn
    to inject an asymmetric attention mask that prevents condition tokens
    from attending to noise tokens. This makes condition features
    deterministic across denoising steps → enables KV cache at inference.

    Mask structure (for T=2, noise=T0, cond=T1):
        noise→noise: ALLOW  |  noise→cond: ALLOW
        cond→noise: BLOCK   |  cond→cond: ALLOW
    """

    def __init__(self, block, block_idx, offloader, independent_condition=True):
        super().__init__(block, block_idx, offloader)
        self.independent_condition = independent_condition

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T = inputs

        self.offloader.wait_for_block(self.block_idx)

        T = x_B_T_H_W_D.shape[1]
        original_compute_attn = None

        if T > 1 and self.independent_condition:
            B, T_dim, H, W, D = x_B_T_H_W_D.shape
            noise_len = H * W
            total_len = T_dim * H * W

            neg_inf = -65504.0 if t_embedding_B_T_D.dtype == torch.float16 else -1e9
            attn_mask = torch.zeros(1, 1, total_len, total_len,
                                    device=x_B_T_H_W_D.device, dtype=t_embedding_B_T_D.dtype)
            attn_mask[:, :, noise_len:, :noise_len] = neg_inf

            # Monkey-patch self_attn.compute_attention to inject the mask
            sa = self.block.self_attn
            original_compute_attn = sa.compute_attention

            def masked_compute_attention(q, k, v):
                # q, k, v are (B, S, H, D) from compute_qkv
                from einops import rearrange as _rearrange
                q_sdpa = _rearrange(q, "b s h d -> b h s d")
                k_sdpa = _rearrange(k, "b s h d -> b h s d")
                v_sdpa = _rearrange(v, "b s h d -> b h s d")
                mask = attn_mask.to(device=q_sdpa.device, dtype=q_sdpa.dtype)
                out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=mask)
                result = _rearrange(out, "b h s d -> b s (h d)")
                # MUST apply output_proj + dropout (original compute_attention does this)
                return sa.output_dropout(sa.output_proj(result))

            sa.compute_attention = masked_compute_attention

        try:
            x_B_T_H_W_D = self.block(
                x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
            )
        finally:
            if original_compute_attn is not None:
                self.block.self_attn.compute_attention = original_compute_attn

        self.offloader.submit_move_blocks_forward(self.block_idx)

        return make_contiguous(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T)
