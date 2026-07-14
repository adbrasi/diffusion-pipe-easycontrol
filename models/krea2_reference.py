"""Shared clean-reference training infrastructure for Krea 2.

Sequence contract::

    [text | noisy target | clean reference]

Krea2 accepts token-wise timestep modulation, so target/text use the sampled
flow timestep while reference tokens use timestep zero (clean in Krea's
training convention). Only the target span is decoded and supervised.
"""

from pathlib import Path

import safetensors.torch
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import peft

from models.base import make_contiguous
from models.krea2 import Krea2Pipeline, TransformerLayer
from comfy.ldm.flux.layers import timestep_embedding
import comfy.ldm.common_dit
from utils.common import AUTOCAST_DTYPE, get_git_commit, is_main_process


class Krea2ReferencePipeline(Krea2Pipeline):
    name = 'krea2_reference'
    config_section = 'krea2_reference'

    def __init__(self, config):
        super().__init__(config)
        reference = config.get(self.config_section, {})
        self.condition_dropout = float(reference.get('condition_dropout', 0.1))
        self.position_mode = reference.get('position_mode', 'subject')
        self.reference_position_offset = float(reference.get('reference_position_offset', 1.0))
        self.reference_position_scale = float(reference.get('reference_position_scale', 1.0))
        self.independent_condition = bool(reference.get('independent_condition', False))
        self.condition_token_stride = int(reference.get('condition_token_stride', 1))
        if not 0 <= self.condition_dropout <= 1:
            raise ValueError('condition_dropout must be between 0 and 1')
        if self.position_mode not in ('spatial', 'subject'):
            raise ValueError("position_mode must be 'spatial' or 'subject'")
        if self.condition_token_stride < 1:
            raise ValueError('condition_token_stride must be >= 1')

    def model_specific_dataset_config_validation(self, dataset_config):
        missing = [
            index for index, directory in enumerate(dataset_config.get('directory', []))
            if not directory.get('control_path')
        ]
        if missing:
            raise ValueError(
                f'{self.name} requires control_path on every dataset directory; missing entries: {missing}'
            )

    def get_call_vae_fn(self, vae):
        parent_vae_fn = super().get_call_vae_fn(vae)

        def fn(*args):
            if len(args) == 1:
                return parent_vae_fn(args[0])
            if len(args) == 2:
                target, reference = args
                result = parent_vae_fn(target)
                result['control_latents'] = parent_vae_fn(
                    self.prepare_reference_media(reference)
                )['latents']
                return result
            raise RuntimeError(f'Unexpected number of VAE inputs: {len(args)}')

        return fn

    def prepare_reference_media(self, reference):
        if self.condition_token_stride == 1:
            return reference
        if reference.ndim == 4:
            height, width = reference.shape[-2:]
            return F.interpolate(
                reference,
                size=(height // self.condition_token_stride, width // self.condition_token_stride),
                mode='bilinear', align_corners=False, antialias=True,
            )
        if reference.ndim == 5:
            frames, height, width = reference.shape[-3:]
            return F.interpolate(
                reference,
                size=(frames, height // self.condition_token_stride, width // self.condition_token_stride),
                mode='trilinear', align_corners=False,
            )
        raise ValueError(f'Krea2 reference pixels must be BCHW/BCTHW, got {tuple(reference.shape)}')

    def prepare_inputs(self, inputs, timestep_quantile=None):
        if 'control_latents' not in inputs:
            raise ValueError(f'{self.name} requires cached control_latents')
        model_inputs, labels = super().prepare_inputs(inputs, timestep_quantile=timestep_quantile)
        noisy_target = model_inputs[0]
        reference = self.prepare_reference_latents(
            inputs['control_latents'].float(),
            noisy_target,
            timestep_quantile=timestep_quantile,
        )
        return (*model_inputs, reference), labels

    def prepare_reference_latents(self, reference, noisy_target, timestep_quantile=None):
        """Apply the exact training-time reference transform for inference too."""
        if reference.shape[:3] != noisy_target.shape[:3]:
            raise ValueError(
                f'Krea2 reference and target batch/channel/frame shapes must match: '
                f'{tuple(reference.shape[:3])} != {tuple(noisy_target.shape[:3])}'
            )
        expected = tuple(size // self.condition_token_stride for size in noisy_target.shape[-2:])
        if reference.shape[-2:] != expected:
            raise ValueError(
                f'Krea2 reference latent shape {tuple(reference.shape[-2:])} does not match '
                f'expected compact shape {expected}; regenerate the VAE cache.'
            )
        if timestep_quantile is None and self.condition_dropout > 0:
            drop = torch.rand(reference.shape[0], device=reference.device) < self.condition_dropout
            if drop.any():
                reference = reference.clone()
                reference[drop] = 0
        return reference

    def configure_adapter(self, adapter_config):
        """Train only DiT block linears; text fusion never sees reference tokens."""
        target_model = self.diffusion_model
        target_linear_modules = set()
        for name, module in target_model.named_modules():
            if module.__class__.__name__ != 'SingleStreamBlock':
                continue
            for full_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, nn.Linear):
                    target_linear_modules.add(full_name)
        targets = sorted(target_linear_modules)
        if not targets:
            raise RuntimeError('No Krea2 DiT linear modules found for the reference adapter')

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
            print(f'[{self.name}] reference-aware LoRA targets: {len(targets)} linears in DiT blocks')
            self.lora_model.print_trainable_parameters()
        for name, parameter in target_model.named_parameters():
            parameter.original_name = name
            if parameter.requires_grad:
                parameter.data = parameter.data.to(adapter_config['dtype'])

    def to_layers(self):
        model = self.diffusion_model
        layers = [
            Krea2ReferenceInitialLayer(
                model,
                position_mode=self.position_mode,
                reference_position_offset=self.reference_position_offset,
                reference_position_scale=self.reference_position_scale,
                independent_condition=self.independent_condition,
            )
        ]
        layers.extend(TransformerLayer(block, index, self.offloader) for index, block in enumerate(model.blocks))
        layers.append(Krea2ReferenceFinalLayer(model))
        return layers

    def save_adapter(self, save_dir, state_dict):
        save_dir = Path(save_dir)
        self.peft_config.save_pretrained(save_dir)
        state_dict = {f'diffusion_model.{key}': value for key, value in state_dict.items()}
        metadata = {
            'format': 'pt',
            'diffusion_pipe_commit': str(get_git_commit()),
            'model_type': self.name,
            'reference_contract': 'krea2_clean_reference_v1',
            'sequence_layout': 'text,target,reference',
            'position_mode': self.position_mode,
            'reference_position_offset': str(self.reference_position_offset),
            'reference_position_scale': str(self.reference_position_scale),
            'condition_dropout': str(self.condition_dropout),
            'independent_condition': str(self.independent_condition).lower(),
            'condition_token_stride': str(self.condition_token_stride),
            'reference_model_timestep': '0.0',
            'condition_encode': 'pixel_bilinear' if self.condition_token_stride > 1 else 'native',
        }
        metadata.update(self.get_reference_metadata())
        safetensors.torch.save_file(state_dict, save_dir / 'adapter_model.safetensors', metadata=metadata)

    def get_reference_metadata(self):
        return {}


class Krea2ReferenceInitialLayer(nn.Module):
    def __init__(
        self,
        model,
        position_mode='subject',
        reference_position_offset=1.0,
        reference_position_scale=1.0,
        independent_condition=False,
    ):
        super().__init__()
        self.first = model.first
        self.tmlp = model.tmlp
        self.tproj = model.tproj
        self.txtfusion = model.txtfusion
        self.txtmlp = model.txtmlp
        self.position_mode = position_mode
        self.reference_position_offset = reference_position_offset
        self.reference_position_scale = reference_position_scale
        self.independent_condition = independent_condition
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        target, timesteps, context, text_attention_mask, reference = inputs
        if target.shape[2] != 1 or reference.shape[2] != 1:
            raise ValueError('Krea2 reference training currently supports one target and one reference frame')
        target = target[:, :, 0]
        reference = reference[:, :, 0]
        batch, channels, target_h_orig, target_w_orig = target.shape

        patch = self.patch
        target = comfy.ldm.common_dit.pad_to_patch_size(target, (patch, patch))
        reference = comfy.ldm.common_dit.pad_to_patch_size(reference, (patch, patch))
        target_h, target_w = target.shape[-2:]
        reference_h, reference_w = reference.shape[-2:]
        target_grid_h, target_grid_w = target_h // patch, target_w // patch
        reference_grid_h, reference_grid_w = reference_h // patch, reference_w // patch

        context = self._unpack_context(context)
        target_tokens = rearrange(
            target, 'b c (h ph) (w pw) -> b (h w) (c ph pw)', ph=patch, pw=patch
        )
        reference_tokens = rearrange(
            reference, 'b c (h ph) (w pw) -> b (h w) (c ph pw)', ph=patch, pw=patch
        )
        target_tokens = self.first(target_tokens)
        reference_tokens = self.first(reference_tokens)

        context = self.txtfusion(context, mask=None)
        context = self.txtmlp(context)
        text_length = context.shape[1]
        target_length = target_tokens.shape[1]
        reference_length = reference_tokens.shape[1]
        combined = torch.cat([context, target_tokens, reference_tokens], dim=1)

        target_timestep_features = self.tmlp(
            timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(combined.dtype)
        )
        target_t = timesteps[:, None].expand(batch, text_length + target_length)
        clean_reference_t = timesteps.new_zeros(batch, reference_length)
        per_token_timestep = torch.cat([target_t, clean_reference_t], dim=1)
        embedded_timesteps = timestep_embedding(per_token_timestep.reshape(-1), self.tdim).reshape(
            batch, combined.shape[1], self.tdim
        )
        per_token_timestep_features = self.tmlp(
            embedded_timesteps.to(combined.dtype)
        )
        tvec = self.tproj(per_token_timestep_features)

        target_pos = self._grid_positions(batch, target_grid_h, target_grid_w, combined.device)
        reference_pos = self._grid_positions(batch, reference_grid_h, reference_grid_w, combined.device)
        reference_pos = reference_pos.clone()
        scale_bias = (self.reference_position_scale - 1.0) / 2.0
        reference_pos[..., 1:] = (
            reference_pos[..., 1:] * self.reference_position_scale + scale_bias
        )
        if self.position_mode == 'subject':
            reference_pos[..., 0] = self.reference_position_offset
        text_pos = combined.new_zeros(batch, text_length, 3)
        positions = torch.cat([text_pos, target_pos, reference_pos], dim=1)
        freqs = self.pe_embedder(positions)

        image_mask = torch.ones(
            batch, target_length + reference_length,
            dtype=torch.bool, device=text_attention_mask.device,
        )
        valid_keys = torch.cat([text_attention_mask, image_mask], dim=1)
        if self.independent_condition:
            attention_mask = valid_keys[:, None, None, :].expand(
                batch, 1, combined.shape[1], combined.shape[1]
            ).clone()
            reference_start = text_length + target_length
            attention_mask[:, :, reference_start:, :] = False
            attention_mask[:, :, reference_start:, reference_start:] = True
            attention_mask = torch.zeros_like(attention_mask, dtype=combined.dtype).masked_fill_(
                ~attention_mask, -torch.finfo(combined.dtype).max
            )
        else:
            attention_mask = valid_keys[:, None, None, :]

        sizes = torch.tensor(
            [text_length, target_length, target_grid_h, target_grid_w, target_h_orig, target_w_orig],
            device=combined.device,
        )
        outputs = make_contiguous(combined, target_timestep_features, tvec, freqs, attention_mask, sizes)
        for item in outputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)
        return outputs

    @staticmethod
    def _grid_positions(batch, height, width, device):
        positions = torch.zeros(height, width, 3, device=device, dtype=torch.float32)
        positions[..., 1] = torch.arange(height, device=device, dtype=torch.float32)[:, None]
        positions[..., 2] = torch.arange(width, device=device, dtype=torch.float32)[None, :]
        return positions.reshape(1, height * width, 3).expand(batch, -1, -1)


class Krea2ReferenceFinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.last = model.last
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    @torch.compiler.disable
    def forward(self, inputs):
        combined, timestep_features, tvec, freqs, attention_mask, sizes = inputs
        text_length, target_length, grid_h, grid_w, original_h, original_w = (int(value) for value in sizes)
        output = self.last(combined, timestep_features)
        output = output[:, text_length:text_length + target_length]
        patch = self.patch
        output = rearrange(
            output,
            'b (h w) (c ph pw) -> b c (h ph) (w pw)',
            h=grid_h,
            w=grid_w,
            ph=patch,
            pw=patch,
            c=self.channels,
        )
        output = output[:, :, :original_h, :original_w]
        return output.unsqueeze(2)
