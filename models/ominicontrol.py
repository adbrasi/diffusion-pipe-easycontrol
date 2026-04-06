"""
OminiControl training pipeline for Anima (Cosmos-Predict2 DiT).

Adapts OminiControl (https://github.com/Yuanshi9815/OminiControl) to Anima:
- Token concat in temporal dimension (condition as T=1)
- Position manipulation via 3D RoPE override:
  - Spatial-aligned (canny, depth, pose): condition shares noise positions → pixel correspondence
  - Subject-driven (identity, style): condition has separate positions → semantic correspondence
- Per-token timestep: target=sigma, condition=0
- Asymmetric noise: condition stays clean
- Loss only on target frame
- Standard PEFT LoRA (same as IC-LoRA, but position-aware)

Key difference from IC-LoRA:
  IC-LoRA always uses T=1 temporal separation (condition at different temporal position).
  OminiControl spatial mode overrides this so condition and noise share EXACT positions,
  creating pixel-to-pixel spatial correspondence through RoPE — critical for control tasks.

Based on: OminiControl v1 (arXiv:2411.15098) and OminiControl v2 (arXiv:2503.08280).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cosmos_predict2 import (
    CosmosPredict2Pipeline,
    InitialLayer,
    TransformerLayer,
    FinalLayer,
    LLMAdapterLayer,
    get_lin_function,
    time_shift,
    _tokenize,
    _compute_text_embeddings,
)
from models.base import make_contiguous
from utils.common import AUTOCAST_DTYPE


class OminiControlPipeline(CosmosPredict2Pipeline):
    """OminiControl training pipeline for Anima.

    Config options under [ominicontrol]:
        position_mode: 'spatial' | 'subject' (default: 'spatial')
            - 'spatial': condition shares noise positions (for canny, depth, pose, etc.)
            - 'subject': condition has separate temporal position (for identity, style)
        condition_dropout: float (default: 0.1)
            - Probability of dropping condition during training (for classifier-free guidance)
        condition_timestep: float (default: 0.0)
            - Timestep assigned to condition tokens (0 = clean)
    """

    def __init__(self, config):
        super().__init__(config)
        oc_config = config.get('ominicontrol', {})
        self.position_mode = oc_config.get('position_mode', 'spatial')
        self.condition_dropout = oc_config.get('condition_dropout', 0.1)
        self.condition_timestep = oc_config.get('condition_timestep', 0.0)

        assert self.position_mode in ('spatial', 'subject'), \
            f"position_mode must be 'spatial' or 'subject', got '{self.position_mode}'"

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']

        if self.cache_text_embeddings:
            prompt_embeds_or_batch_encoding = (
                inputs['prompt_embeds'], inputs['attn_mask'],
                inputs['t5_input_ids'], inputs['t5_attn_mask'],
            )
        else:
            captions = inputs['caption']
            batch_encoding = _tokenize(self.tokenizer, captions)
            t5_batch_encoding = _tokenize(self.t5_tokenizer, captions)
            prompt_embeds_or_batch_encoding = (
                batch_encoding.input_ids, batch_encoding.attention_mask,
                t5_batch_encoding.input_ids, t5_batch_encoding.attention_mask,
            )

        bs, channels, num_frames, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')
            mask = mask.unsqueeze(2)

        # Timestep sampling (standard flow matching)
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

        # Asymmetric noise: only target gets noise, condition stays clean
        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        noisy_latents = (1 - t_expanded) * latents + t_expanded * noise
        target = noise - latents

        if 'control_latents' in inputs:
            control_latents = inputs['control_latents'].float()

            # Condition dropout for classifier-free guidance training
            # prepare_inputs is only called during training, so always apply
            if self.condition_dropout > 0:
                drop_mask = torch.rand(bs, device=latents.device) < self.condition_dropout
                if drop_mask.any():
                    control_latents = control_latents.clone()
                    control_latents[drop_mask] = 0.0

            # Temporal concat: [noisy_target_T0, clean_condition_T1]
            noisy_latents = torch.cat([noisy_latents, control_latents], dim=2)

            # Per-token timestep: target=sigma, condition=condition_timestep (default 0)
            cond_t = torch.full_like(t, self.condition_timestep)
            t = torch.stack([t, cond_t], dim=1)  # (B, 2)
        else:
            # No condition — standard single-frame training (fallback)
            t = t.view(-1, 1)  # (B, 1)

        return (noisy_latents, t, *prompt_embeds_or_batch_encoding), (target, mask)

    def to_layers(self):
        transformer = self.transformer
        text_encoder = None if self.cache_text_embeddings else self.text_encoder
        layers = [
            OminiControlInitialLayer(
                transformer, text_encoder, self.is_generic_llm,
                position_mode=self.position_mode,
            ),
            LLMAdapterLayer(transformer.llm_adapter if self.use_llm_adapter else None),
        ]
        for i, block in enumerate(transformer.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(transformer))
        return layers

    def get_loss_fn(self):
        def loss_fn(output, label):
            target, mask = label
            with torch.autocast('cuda', enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)

                # OminiControl: output has T=2 (target + condition prediction)
                # Loss ONLY on the target frame (T=0), exclude condition
                if output.shape[2] != target.shape[2]:
                    output = output[:, :, :target.shape[2], :, :]

                if 'pseudo_huber_c' in self.config:
                    c = self.config['pseudo_huber_c']
                    loss = torch.sqrt((output - target) ** 2 + c ** 2) - c
                else:
                    loss = F.mse_loss(output, target, reduction='none')

                if mask is not None and mask.numel() > 0:
                    mask = mask.to(output.device, torch.float32)
                    loss *= mask
                loss = loss.mean()

                if weight := self.multiscale_loss_weight:
                    assert output.ndim == 5 and target.ndim == 5
                    # Squeeze temporal dim (T=1 after slicing) for 2D interpolation
                    output_2d = output.squeeze(2)  # (B, C, H, W)
                    target_2d = target.squeeze(2)
                    for factor in [2, 4]:
                        output_2d = F.avg_pool2d(output_2d, 2)
                        target_2d = F.avg_pool2d(target_2d, 2)
                        if 'pseudo_huber_c' in self.config:
                            c = self.config['pseudo_huber_c']
                            ds_loss = torch.sqrt(
                                (output_2d - target_2d) ** 2 + c ** 2
                            ) - c
                        else:
                            ds_loss = F.mse_loss(
                                output_2d, target_2d, reduction='none'
                            )
                        loss = loss + weight * ds_loss.mean()

            return loss
        return loss_fn


class OminiControlInitialLayer(InitialLayer):
    """InitialLayer with OminiControl position manipulation.

    In spatial mode: condition tokens get the SAME RoPE positions as noise tokens,
    creating pixel-to-pixel spatial correspondence. The model distinguishes them
    via per-token timestep (condition=0, noise=sigma).

    In subject mode: condition tokens keep their natural T=1 position,
    providing temporal separation that forces semantic (not spatial) attention.
    """

    def __init__(self, model, text_encoder, is_generic_llm, position_mode='spatial'):
        super().__init__(model, text_encoder, is_generic_llm)
        self.position_mode = position_mode

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_C_T_H_W, timesteps_B_T, *prompt_embeds_or_batch_encoding = inputs

        if torch.is_floating_point(prompt_embeds_or_batch_encoding[0]):
            crossattn_emb, attn_mask, t5_input_ids, t5_attn_mask = prompt_embeds_or_batch_encoding
        else:
            with torch.no_grad():
                input_ids, attn_mask, t5_input_ids, t5_attn_mask = prompt_embeds_or_batch_encoding
                crossattn_emb = _compute_text_embeddings(
                    self.text_encoder[0], input_ids, attn_mask, is_generic_llm=self.is_generic_llm,
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

        # === OminiControl Position Manipulation ===
        T = x_B_C_T_H_W.shape[2]
        if T > 1 and self.position_mode == 'spatial' and rope_emb_L_1_1_D is not None:
            # Spatial mode: condition tokens should share EXACT positions with noise tokens.
            # rope_emb shape: (T*H_pat*W_pat, 1, 1, D)
            # First HW positions are noise (t=0), next HW are condition (t=1).
            # Override: replace condition positions with noise positions.
            total_seq = rope_emb_L_1_1_D.shape[0]
            noise_seq_len = total_seq // T  # H_patched * W_patched for one frame
            noise_rope = rope_emb_L_1_1_D[:noise_seq_len]
            # Duplicate noise positions for all frames (condition gets same positions)
            rope_emb_L_1_1_D = noise_rope.repeat(T, 1, 1, 1)
        # Subject mode: keep default RoPE (T=1 gives natural temporal separation)

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
