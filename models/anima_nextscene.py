"""Anima Next-Scene — reference-conditioned "next scene" / edit training for Anima.

Why this pipeline exists (full reasoning in docs/ANIMA_NEXTSCENE_PESQUISA_2026-09.md):

1. Anima is a T2I-only Cosmos-Predict2 DiT (x_embedder = 16 latent + 1 padding
   channel; no Video2World condition-mask channel). Temporal RoPE positions > 0
   and cross-frame attention were never trained.
2. The old IC-LoRA packing puts the reference at (t=1, h, w) — the SAME spatial
   grid as the target. With Anima's RoPE (h/w/t = 42/42/44 dims, t theta 1e4)
   the mean positional kernel between target(0,h,w) and ref(1,h,w) is 0.987,
   the same as a target token and its immediate spatial neighbour (0.988).
   Attention therefore treats the reference as a pixel-aligned overlay of the
   target: the natural minimum is "copy the reference" — exactly the recurring
   failure ("it gives me back the same image").
3. Independent evidence: AnimaRefLora (crazysheep924, 500K steps on Anima) moved
   reference frames to disjoint spatial tiles for this reason; UNO/OminiControl
   offset non-aligned references for the same reason; Qwen-Image-Edit-2511,
   Z-Image-Edit, ChronoEdit (Cosmos-2B) and the Krea 2 edit trainers all keep the
   reference clean with its own timestep 0.

Contract (target-first, per-frame timestep):

    x   = [ noisy_target (T=0) | clean_ref (T=1) ]          (B, 16, 2, H, W)
    t   = [ sigma              | ref_t (default 0) ]        (B, 2)
    RoPE: target at (t=0, h, w); ref at (t=ref_temporal_index, h+dh, w+dw)
          rope_layout: aligned (dh=dw=0, old IC-LoRA) | disjoint_w (dw=W)
                       | disjoint_h (dh=H) | disjoint_diag (dh=H, dw=W)
    loss: velocity MSE on the target frame only, optionally difference-weighted.

Anti-copy / anti-shortcut knobs (all under [nextscene]):
    rope_layout, ref_temporal_index      geometry (the main A/B)
    ref_dropout                          blank the ref (zeros) -> trained null for ref-CFG
    diff_weight*                         changed regions get more loss than copied ones
    high_noise_prob/min/max              extra mass at sigma in [0.8, 1] (composition band)
    ref_hflip_prob                       latent-space horizontal flip of the reference only
    ref_noise_prob / ref_noise_logmean   LTX/SVD-style small noise on the reference; the
                                         ref frame timestep is then set to that sigma
"""

import json
import types

import peft
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cosmos_predict2 import (
    CosmosPredict2Pipeline,
    get_lin_function,
    time_shift,
    _tokenize,
)
from utils.common import is_main_process


ROPE_LAYOUTS = ('aligned', 'disjoint_w', 'disjoint_h', 'disjoint_diag')
CONTRACT_VERSION = '1'


# ---------------------------------------------------------------------------
# RoPE layout (shared by training and inference)
# ---------------------------------------------------------------------------

def frame_positions(layout, ref_temporal_index, num_frames, height, width):
    """Per-frame (t_index, h_offset, w_offset). Frame 0 is the target, frames
    1.. are references. With num_frames == 1 this is the stock T2I layout."""
    if layout not in ROPE_LAYOUTS:
        raise ValueError(f'Unknown rope_layout {layout!r}; expected one of {ROPE_LAYOUTS}')
    positions = [(0, 0, 0)]
    for i in range(1, num_frames):
        t_index = int(ref_temporal_index) + (i - 1)
        if layout == 'aligned':
            dh, dw = 0, 0
        elif layout == 'disjoint_w':
            dh, dw = 0, width * i
        elif layout == 'disjoint_h':
            dh, dw = height * i, 0
        else:  # disjoint_diag
            dh, dw = height * i, width * i
        positions.append((t_index, dh, dw))
    return positions


def _make_generate_embeddings(layout, ref_temporal_index):
    def generate_embeddings(self, B_T_H_W_C, fps=None, h_ntk_factor=None, w_ntk_factor=None, t_ntk_factor=None):
        if getattr(self, 'enable_fps_modulation', False) and fps is not None:
            raise NotImplementedError('nextscene RoPE layout supports image mode (fps=None) only')
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_freqs = 1.0 / ((10000.0 * h_ntk_factor) ** self.dim_spatial_range)
        w_freqs = 1.0 / ((10000.0 * w_ntk_factor) ** self.dim_spatial_range)
        t_freqs = 1.0 / ((10000.0 * t_ntk_factor) ** self.dim_temporal_range)

        _, T, H, W, _ = B_T_H_W_C
        device = self.dim_spatial_range.device
        frames = []
        for t_index, dh, dw in frame_positions(layout, ref_temporal_index, T, H, W):
            # Positions are plain float aranges exactly like the stock
            # self.seq[:N] — frame 0 reproduces the T2I embedding bit for bit.
            pos_t = torch.full((1,), float(t_index), device=device)
            pos_h = torch.arange(H, device=device, dtype=torch.float) + float(dh)
            pos_w = torch.arange(W, device=device, dtype=torch.float) + float(dw)
            et = torch.outer(pos_t, t_freqs).view(1, 1, -1).expand(H, W, -1)
            eh = torch.outer(pos_h, h_freqs).view(H, 1, -1).expand(H, W, -1)
            ew = torch.outer(pos_w, w_freqs).view(1, W, -1).expand(H, W, -1)
            half = torch.cat([et, eh, ew], dim=-1)
            frames.append(torch.cat([half, half], dim=-1))
        emb = torch.stack(frames, dim=0)  # (T, H, W, D)
        return emb.reshape(T * H * W, 1, 1, emb.shape[-1]).float()

    return generate_embeddings


def install_nextscene_rope(pos_embedder, layout, ref_temporal_index):
    """Patch a VideoRopePosition3DEmb in place. Returns a restore() callable."""
    if hasattr(pos_embedder, '_nextscene_restore'):
        pos_embedder._nextscene_restore()
    original = pos_embedder.generate_embeddings
    pos_embedder.generate_embeddings = types.MethodType(
        _make_generate_embeddings(layout, ref_temporal_index), pos_embedder
    )

    def restore():
        pos_embedder.generate_embeddings = original
        del pos_embedder._nextscene_restore

    pos_embedder._nextscene_restore = restore
    return restore


def contract_from_metadata(metadata):
    """Read the geometry contract a nextscene adapter was trained with."""
    if not metadata or metadata.get('nextscene_contract') is None:
        return None
    return json.loads(metadata['nextscene_contract'])


# ---------------------------------------------------------------------------
# Loss weighting
# ---------------------------------------------------------------------------

def diff_weight_map(target_latents, ref_latents, floor=0.2, slope_hi=0.5, max_weight=3.0):
    """Difference-weighted flow matching (AnimaRefLora recipe).

    d = channel-mean |z_tgt - z_ref|, normalised to mean 1 per sample.
    w = floor + (1-floor)*d        for d < 1   (regions the ref already solves)
        1 + slope_hi*(d-1)         for d >= 1  (regions that actually changed)
    clamped to max_weight and renormalised to mean 1, so the effective LR is
    unchanged — gradient is only redistributed away from copyable regions.
    Shapes: (B, C, 1, H, W) -> (B, 1, 1, H, W).
    """
    d = (target_latents - ref_latents).abs().mean(dim=1, keepdim=True)
    d = d / d.mean(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    w = torch.where(d < 1, floor + (1 - floor) * d, 1 + slope_hi * (d - 1))
    w = w.clamp(max=max_weight)
    return w / w.mean(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class AnimaNextScenePipeline(CosmosPredict2Pipeline):
    adapter_log_tag = 'Anima NextScene'

    def __init__(self, config):
        super().__init__(config)
        self._parse_nextscene_config(config)

    def _parse_nextscene_config(self, config):
        s = config.get('nextscene', {})
        self.rope_layout = s.get('rope_layout', 'disjoint_w')
        if self.rope_layout not in ROPE_LAYOUTS:
            raise ValueError(f'[nextscene] rope_layout must be one of {ROPE_LAYOUTS}')
        self.ref_temporal_index = int(s.get('ref_temporal_index', 1))
        self.ref_dropout = float(s.get('ref_dropout', 0.1))
        self.lora_cross_attn = bool(s.get('lora_cross_attn', True))
        self.diff_weight = bool(s.get('diff_weight', True))
        self.diff_weight_floor = float(s.get('diff_weight_floor', 0.2))
        self.diff_weight_slope_hi = float(s.get('diff_weight_slope_hi', 0.5))
        self.diff_weight_max = float(s.get('diff_weight_max', 3.0))
        self.high_noise_prob = float(s.get('high_noise_prob', 0.2))
        self.high_noise_min = float(s.get('high_noise_min', 0.8))
        self.high_noise_max = float(s.get('high_noise_max', 1.0))
        self.ref_hflip_prob = float(s.get('ref_hflip_prob', 0.0))
        self.ref_noise_prob = float(s.get('ref_noise_prob', 0.0))
        self.ref_noise_logmean = float(s.get('ref_noise_logmean', -3.0))
        self.ref_noise_logstd = float(s.get('ref_noise_logstd', 0.5))
        # adaln_modulation: Anima has an internal adaln LoRA (double-LoRA was the
        # April meltdown). llm_adapter: official advice + Round 1 A/B say frozen.
        forbidden = ['adaln_modulation', 'llm_adapter']
        if not self.lora_cross_attn:
            forbidden.append('cross_attn')
        self.forbidden_adapter_key_patterns = tuple(forbidden)

    # -- model -------------------------------------------------------------

    def load_diffusion_model(self):
        super().load_diffusion_model()
        install_nextscene_rope(self.transformer.pos_embedder, self.rope_layout, self.ref_temporal_index)
        if is_main_process():
            print(f'[{self.adapter_log_tag}] RoPE layout={self.rope_layout} '
                  f'ref_temporal_index={self.ref_temporal_index}')

    def configure_adapter(self, adapter_config):
        if adapter_config['type'] != 'lora':
            raise NotImplementedError(f"Adapter type {adapter_config['type']} is not implemented")
        targets = set()
        for name, module in self.transformer.named_modules():
            if module.__class__.__name__ not in self.adapter_target_modules or name.startswith('llm_adapter'):
                continue
            for sub_name, sub in module.named_modules(prefix=name):
                if not isinstance(sub, nn.Linear):
                    continue
                parts = sub_name.split('.')
                if any(p.startswith('adaln_modulation') for p in parts) or parts[0] == 'llm_adapter':
                    continue
                if not self.lora_cross_attn and 'cross_attn' in parts:
                    continue
                targets.add(sub_name)
        targets = sorted(targets)
        for t in targets:
            assert not any(p in t for p in self.forbidden_adapter_key_patterns), t
        if is_main_process():
            n_cross = sum('cross_attn' in t for t in targets)
            print(f'[{self.adapter_log_tag}] LoRA targets: {len(targets)} linears '
                  f'({n_cross} cross_attn; adaln/llm_adapter excluded)')
        self.peft_config = peft.LoraConfig(
            r=adapter_config['rank'],
            lora_alpha=adapter_config['alpha'],
            lora_dropout=adapter_config['dropout'],
            bias='none',
            target_modules=targets,
        )
        self.lora_model = peft.get_peft_model(self.transformer, self.peft_config)
        if is_main_process():
            self.lora_model.print_trainable_parameters()
        for name, p in self.transformer.named_parameters():
            p.original_name = name
            if p.requires_grad:
                p.data = p.data.to(adapter_config['dtype'])

    def contract(self):
        return {
            'version': CONTRACT_VERSION,
            'layout': 'target_first',
            'rope_layout': self.rope_layout,
            'ref_temporal_index': self.ref_temporal_index,
            'ref_timestep': 0.0,
            'null_ref': 'zeros',
        }

    def save_adapter(self, save_dir, peft_state_dict):
        import safetensors.torch
        from utils.common import get_git_commit
        self.peft_config.save_pretrained(save_dir)
        peft_state_dict = {'diffusion_model.' + k: v for k, v in peft_state_dict.items()}
        self._audit_adapter_keys(peft_state_dict.keys(), save_dir)
        metadata = {
            'format': 'pt',
            'diffusion_pipe_commit': get_git_commit(),
            'model_type': str(self.model_config.get('type', 'anima_nextscene')),
            'nextscene_contract': json.dumps(self.contract()),
        }
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata=metadata)
        with open(save_dir / 'nextscene_contract.json', 'w') as f:
            json.dump(self.contract(), f, indent=2)

    # -- training step -----------------------------------------------------

    def _sample_sigma(self, bs, h, w, device, timestep_quantile):
        method = self.model_config.get('timestep_sample_method', 'logit_normal')
        if method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0.0, 1.0)
        elif method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0.0, 1.0)
        else:
            raise NotImplementedError(method)
        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=device))
        else:
            t = dist.sample((bs,)).to(device)
        if method == 'logit_normal':
            t = torch.sigmoid(t * self.model_config.get('sigmoid_scale', 1.0))
        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)
        # Extra mass in the composition band: at high sigma almost nothing of the
        # target survives, so the model must use ref + text there.
        if timestep_quantile is None and self.high_noise_prob > 0:
            use_high = torch.rand(bs, device=device) < self.high_noise_prob
            high = torch.empty(bs, device=device).uniform_(self.high_noise_min, self.high_noise_max)
            t = torch.where(use_high, high, t)
        return t.clamp(1e-3, 1.0)

    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']

        if self.cache_text_embeddings:
            text = (inputs['prompt_embeds'], inputs['attn_mask'], inputs['t5_input_ids'], inputs['t5_attn_mask'])
        else:
            captions = inputs['caption']
            be = _tokenize(self.tokenizer, captions)
            t5 = _tokenize(self.t5_tokenizer, captions)
            text = (be.input_ids, be.attention_mask, t5.input_ids, t5.attention_mask)

        bs, _, num_frames, h, w = latents.shape
        assert num_frames == 1, 'nextscene trains on still images (target T=1)'
        device = latents.device

        if mask is not None:
            mask = F.interpolate(mask.unsqueeze(1), size=(h, w), mode='nearest-exact').unsqueeze(2)

        t = self._sample_sigma(bs, h, w, device, timestep_quantile)
        noise = torch.randn_like(latents)
        noisy = (1 - t.view(-1, 1, 1, 1, 1)) * latents + t.view(-1, 1, 1, 1, 1) * noise
        target = noise - latents

        if 'control_latents' not in inputs:
            # Plain T2I sample (no reference in this batch).
            weight = torch.ones(bs, 1, 1, h, w, device=device)
            return (noisy, t.view(-1, 1), *text), (target, mask, weight)

        ref = inputs['control_latents'].float().to(device)
        assert ref.shape == latents.shape, (
            f'reference latents {tuple(ref.shape)} must match target {tuple(latents.shape)} '
            '(same size bucket); check the dataset control_path'
        )
        is_train = timestep_quantile is None

        if is_train and self.ref_hflip_prob > 0:
            flip = torch.rand(bs, device=device) < self.ref_hflip_prob
            if flip.any():
                ref = torch.where(flip.view(-1, 1, 1, 1, 1), ref.flip(-1), ref)

        if self.diff_weight:
            weight = diff_weight_map(latents, ref, self.diff_weight_floor,
                                     self.diff_weight_slope_hi, self.diff_weight_max)
        else:
            weight = torch.ones(bs, 1, 1, h, w, device=device)

        ref_t = torch.zeros_like(t)
        if is_train and self.ref_noise_prob > 0:
            use = torch.rand(bs, device=device) < self.ref_noise_prob
            s = torch.exp(torch.randn(bs, device=device) * self.ref_noise_logstd + self.ref_noise_logmean).clamp(max=0.3)
            s = torch.where(use, s, torch.zeros_like(s))
            ref = (1 - s.view(-1, 1, 1, 1, 1)) * ref + s.view(-1, 1, 1, 1, 1) * torch.randn_like(ref)
            ref_t = s

        if is_train and self.ref_dropout > 0:
            drop = torch.rand(bs, device=device) < self.ref_dropout
            if drop.any():
                ref = torch.where(drop.view(-1, 1, 1, 1, 1), torch.zeros_like(ref), ref)
                ref_t = torch.where(drop, torch.zeros_like(ref_t), ref_t)
                # The model cannot see a blanked ref: fall back to uniform weights.
                weight = torch.where(drop.view(-1, 1, 1, 1, 1), torch.ones_like(weight), weight)

        x = torch.cat([noisy, ref], dim=2)            # target first
        t_frames = torch.stack([t, ref_t], dim=1)     # (B, 2)
        return (x, t_frames, *text), (target, mask, weight)

    def get_loss_fn(self):
        def loss_fn(output, label):
            target, mask, weight = label
            with torch.autocast('cuda', enabled=False):
                output = output.to(torch.float32)[:, :, :target.shape[2]]  # target frame only
                target = target.to(output.device, torch.float32)
                loss = F.mse_loss(output, target, reduction='none')
                w = weight.to(output.device, torch.float32)
                if mask is not None and mask.numel() > 0:
                    w = w * mask.to(output.device, torch.float32)
                return (loss * w).mean()
        return loss_fn
