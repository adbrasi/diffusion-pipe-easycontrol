"""Reference-conditioned IC-LoRA training for Ideogram 4.

The packed transformer sequence is::

    [text | noisy target | clean reference]

Only the target tokens are decoded and supervised. The reference uses the same
VAE/token projection as the target, a clean model timestep, and a distinct
temporal MRoPE coordinate. This is an additive pipeline: the upstream
``ideogram4`` text-to-image implementation remains unchanged.
"""

from pathlib import Path

import safetensors.torch
import torch
from torch import nn
import torch.nn.functional as F

from models.ideogram4 import (
    IMAGE_POSITION_OFFSET,
    Ideogram4Pipeline,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
    TransformerLayer,
)
from comfy.text_encoders.llama import precompute_freqs_cis
from models.base import make_contiguous
from models.ideogram4_reference_contract import (
    LORA_FORBIDDEN_MODULE_PATTERNS,
    REFERENCE_CONTRACT_VERSION,
    REFERENCE_IMAGE_INDICATOR,
    apply_reference_dropout,
    build_model_timesteps,
    offset_reference_positions,
    split_lora_target_modules,
)
from utils.common import get_git_commit


class Ideogram4ICLoRAPipeline(Ideogram4Pipeline):
    """Train Ideogram 4 to generate a target from an arbitrary reference image.

    Configuration under ``[ideogram4_ic_lora]``:

    ``condition_dropout`` (default ``0.1``)
        Probability of replacing all reference latents for a sample with zero.
        Tokens stay in the sequence so training and inference packing match.
    ``reference_position_offset`` (default ``1``)
        Offset applied only to the temporal MRoPE coordinate of the reference.
    ``reference_model_timestep`` (default ``1.0``)
        Timestep in Ideogram's internal convention, where ``1.0`` is clean.
    ``train_adaln_modulation`` (default ``false``)
        When false, per-block ``adaln_modulation`` linears are excluded from the
        LoRA. Their input is the timestep embedding alone, so they cannot learn
        the ref->target relation, and the reference's fixed clean timestep makes
        them the most distribution-shifted pathway in this pipeline.
    """

    name = 'ideogram4_ic_lora'

    def __init__(self, config):
        super().__init__(config)
        reference_config = config.get('ideogram4_ic_lora', {})
        self.condition_dropout = float(reference_config.get('condition_dropout', 0.1))
        self.reference_position_offset = int(reference_config.get('reference_position_offset', 1))
        self.reference_model_timestep = float(reference_config.get('reference_model_timestep', 1.0))
        self.train_adaln_modulation = bool(reference_config.get('train_adaln_modulation', False))
        # Anti-atalho-de-caption (InstructPix2Pix assimétrico): fração de fetches
        # com caption VAZIA e referência MANTIDA — força identidade vir do latent.
        # Usa a infra grounded_uncond do dataset (cache per-sample de uncond).
        self.caption_dropout = float(reference_config.get('caption_dropout', 0.0))
        if not 0.0 <= self.caption_dropout <= 0.5:
            raise ValueError('caption_dropout must be between 0.0 and 0.5')

        if not 0.0 <= self.condition_dropout <= 1.0:
            raise ValueError('condition_dropout must be between 0.0 and 1.0')
        if not 0.0 <= self.reference_model_timestep <= 1.0:
            raise ValueError('reference_model_timestep must be between 0.0 and 1.0')

    def model_specific_dataset_config_validation(self, dataset_config):
        missing_reference = [
            index
            for index, directory in enumerate(dataset_config.get('directory', []))
            if not directory.get('control_path')
        ]
        if missing_reference:
            indices = ', '.join(str(index) for index in missing_reference)
            raise ValueError(
                'Ideogram4 IC-LoRA requires control_path on every dataset directory. '
                f'Missing on directory entries: {indices}'
            )

    def get_call_vae_fn(self, vae):
        parent_vae_fn = super().get_call_vae_fn(vae)

        def fn(*args):
            if len(args) == 1:
                return parent_vae_fn(args[0])
            if len(args) == 2:
                target, reference = args
                result = parent_vae_fn(target)
                reference_result = parent_vae_fn(self.prepare_reference_media(reference))
                result['control_latents'] = reference_result['latents']
                return result
            raise RuntimeError(f'Unexpected number of VAE inputs: {len(args)}')

        return fn

    def prepare_reference_media(self, reference):
        """Transform reference pixels before VAE encoding (identity for IC-LoRA/v1)."""
        return reference

    def prepare_inputs(self, inputs, timestep_quantile=None):
        if 'control_latents' not in inputs:
            raise ValueError(
                'Ideogram4 IC-LoRA requires cached control_latents. '
                'Add control_path to every dataset directory and regenerate the cache.'
            )

        model_inputs, labels = super().prepare_inputs(inputs, timestep_quantile=timestep_quantile)
        noisy_target = model_inputs[0]
        reference_latents = self.prepare_reference_latents(
            inputs['control_latents'].float(),
            noisy_target,
            timestep_quantile=timestep_quantile,
        )
        return (*model_inputs, reference_latents), labels

    def prepare_reference_latents(self, reference_latents, noisy_target, timestep_quantile=None):
        """Validate/transform reference latents before sequence packing.

        OminiControl2 overrides this hook to create a compact reference token
        grid. The default IC-LoRA contract intentionally remains same-size.
        """
        if reference_latents.shape != noisy_target.shape:
            raise ValueError(
                'Ideogram4 reference latents must match target latent shape, got '
                f'{tuple(reference_latents.shape)} and {tuple(noisy_target.shape)}'
            )

        # Evaluation uses explicit timestep quantiles and must stay deterministic.
        reference_latents = apply_reference_dropout(
            reference_latents,
            self.condition_dropout,
            enabled=timestep_quantile is None,
        )
        return reference_latents

    def configure_adapter(self, adapter_config):
        """CommonPipeline.configure_adapter with adaln_modulation excluded.

        The copy is deliberate: the base implementation offers no hook to filter
        target modules, and control adapters in this fork must never silently
        regain forbidden targets through an upstream default.
        """
        if self.train_adaln_modulation:
            super().configure_adapter(adapter_config)
            return

        import peft
        from utils.common import is_main_process

        target_model = self.diffusion_model
        target_linear_modules = set()
        for name, module in target_model.named_modules():
            if module.__class__.__name__ not in self.adapter_target_modules:
                continue
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, nn.Linear):
                    target_linear_modules.add(full_submodule_name)
        target_linear_modules, excluded = split_lora_target_modules(
            sorted(target_linear_modules),
            LORA_FORBIDDEN_MODULE_PATTERNS,
        )
        if not target_linear_modules:
            raise RuntimeError('No LoRA target modules remain after exclusion')
        if is_main_process():
            print(
                f'[{self.name}] LoRA targets: {len(target_linear_modules)} linears, '
                f'excluded {len(excluded)} matching {list(LORA_FORBIDDEN_MODULE_PATTERNS)} '
                '(set train_adaln_modulation = true to include them)'
            )

        adapter_type = adapter_config['type']
        if adapter_type == 'lora':
            peft_config = peft.LoraConfig(
                r=adapter_config['rank'],
                lora_alpha=adapter_config['alpha'],
                lora_dropout=adapter_config['dropout'],
                bias='none',
                target_modules=target_linear_modules,
            )
        elif adapter_type == 'lokr':
            peft_config = peft.LoKrConfig(
                r=adapter_config['rank'],
                decompose_factor=adapter_config['decompose_factor'],
                alpha=adapter_config['alpha'],
                rank_dropout=adapter_config['rank_dropout'],
                target_modules=target_linear_modules,
            )
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
        self.peft_config = peft_config
        self.lora_model = peft.get_peft_model(target_model, peft_config)
        if is_main_process():
            self.lora_model.print_trainable_parameters()
        for name, p in target_model.named_parameters():
            p.original_name = name
            if p.requires_grad:
                p.data = p.data.to(adapter_config['dtype'])

    def to_layers(self):
        diffusion_model = self.diffusion_model
        layers = [
            Ideogram4ReferenceInitialLayer(
                diffusion_model,
                reference_position_offset=self.reference_position_offset,
                reference_model_timestep=self.reference_model_timestep,
            )
        ]
        for index, block in enumerate(diffusion_model.layers):
            layers.append(TransformerLayer(block, index, self.offloader))
        layers.append(Ideogram4ReferenceFinalLayer(diffusion_model))
        return layers

    def save_adapter(self, save_dir, state_dict):
        """Save a ComfyUI LoRA with a machine-readable packing contract."""
        save_dir = Path(save_dir)
        self.peft_config.save_pretrained(save_dir)
        state_dict = {f'diffusion_model.{key}': value for key, value in state_dict.items()}
        self._audit_adapter_keys(state_dict.keys(), save_dir)

        base_model_path = Path(self.model_config['diffusion_model'])
        metadata = {
            'format': 'pt',
            'diffusion_pipe_commit': str(get_git_commit()),
            'model_type': self.name,
            'reference_contract': REFERENCE_CONTRACT_VERSION,
            'sequence_layout': 'text,target,reference',
            'reference_indicator': str(REFERENCE_IMAGE_INDICATOR),
            'reference_position_offset': str(self.reference_position_offset),
            'reference_model_timestep': str(self.reference_model_timestep),
            'condition_dropout': str(self.condition_dropout),
            'lora_train_adaln_modulation': str(self.train_adaln_modulation).lower(),
            # Identity of the frozen base and the training-time flow shift, so
            # inference can warn about cross-application and reproduce the
            # schedule the adapter was trained against (ComfyUI's canonical
            # Ideogram 4 sampling is shift=1.0; the fork's example configs
            # train with shift=3).
            'base_model_file': base_model_path.name,
            'base_model_size': str(base_model_path.stat().st_size if base_model_path.exists() else 0),
            'training_shift': str(self.model_config.get('shift', 'none')),
        }
        metadata.update(self.get_reference_metadata())
        safetensors.torch.save_file(
            state_dict,
            save_dir / 'adapter_model.safetensors',
            metadata=metadata,
        )

    def get_reference_metadata(self):
        return {}

    def _audit_adapter_keys(self, keys, save_dir):
        """Post-save audit in the fork's standard style: warn loudly and drop a
        marker file instead of raising, so an audit failure never destroys a
        finished training run — but it cannot be missed either."""
        keys = list(keys)
        problems = []
        if not keys:
            problems.append(('checkpoint has no trainable adapter keys', []))
        outside_layers = sorted(key for key in keys if '.layers.' not in key)
        if outside_layers:
            problems.append(('keys outside transformer layers', outside_layers))
        non_lora = sorted(
            key for key in keys if '.lora_A.' not in key and '.lora_B.' not in key
        )
        if non_lora:
            problems.append(('trainable keys that are not LoRA A/B tensors', non_lora))
        if not self.train_adaln_modulation:
            forbidden = sorted(
                key for key in keys
                if any(pattern in key for pattern in LORA_FORBIDDEN_MODULE_PATTERNS)
            )
            if forbidden:
                problems.append(
                    (f'keys matching forbidden patterns {list(LORA_FORBIDDEN_MODULE_PATTERNS)}', forbidden)
                )
        if not problems:
            print(f'[{self.name}] adapter audit OK: {len(keys)} keys')
            return
        lines = ['ADAPTER AUDIT FAILED: checkpoint contains unexpected LoRA keys.']
        for description, bad_keys in problems:
            lines.append(f'{description} ({len(bad_keys)}), first 10:')
            lines.extend(f'  {key}' for key in bad_keys[:10])
        lines.append('Fix configure_adapter before training further.')
        text = '\n'.join(lines)
        print('\n' + '!' * 80 + f'\n{text}\n' + '!' * 80 + '\n')
        with open(save_dir / 'ADAPTER_AUDIT_FAILED.txt', 'w') as marker:
            marker.write(text + '\n')


class Ideogram4ReferenceInitialLayer(nn.Module):
    """Pack text, target, and reference tokens for the Ideogram transformer."""

    def __init__(
        self,
        model,
        reference_position_offset=1,
        reference_model_timestep=1.0,
        reference_position_scale=1.0,
        require_matching_shape=True,
    ):
        super().__init__()
        self.input_proj = model.input_proj
        self.t_embedding = model.t_embedding
        self.adaln_proj = model.adaln_proj
        self.llm_cond_norm = model.llm_cond_norm
        self.llm_cond_proj = model.llm_cond_proj
        self.embed_image_indicator = model.embed_image_indicator
        self.reference_position_offset = reference_position_offset
        self.reference_model_timestep = reference_model_timestep
        self.reference_position_scale = reference_position_scale
        self.require_matching_shape = require_matching_shape
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    # Autocast here noticeably degrades Ideogram's output, matching upstream.
    @torch.compiler.disable
    def forward(self, inputs):
        for item in inputs:
            if torch.is_floating_point(item):
                item.requires_grad_(True)

        target, timesteps, context, text_attention_mask, reference = inputs
        if self.require_matching_shape and reference.shape != target.shape:
            raise ValueError(
                'Ideogram4 reference latents must match target latent shape, got '
                f'{tuple(reference.shape)} and {tuple(target.shape)}'
            )

        if reference.shape[0] != target.shape[0] or reference.shape[1] != target.shape[1]:
            raise ValueError(
                'Ideogram4 reference batch/channels must match target, got '
                f'{tuple(reference.shape)} and {tuple(target.shape)}'
            )

        batch_size, _, grid_h, grid_w = target.shape
        _, _, reference_grid_h, reference_grid_w = reference.shape
        device = target.device
        target_tokens = self._img_to_tokens(target)
        reference_tokens = self._img_to_tokens(reference)
        text_length = context.shape[1]
        target_length = target_tokens.shape[1]
        reference_length = reference_tokens.shape[1]
        sequence_length = text_length + target_length + reference_length
        latent_dim = target_tokens.shape[-1]

        packed = torch.zeros(
            batch_size,
            sequence_length,
            latent_dim,
            dtype=target_tokens.dtype,
            device=device,
        )
        target_start = text_length
        target_end = target_start + target_length
        packed[:, target_start:target_end] = target_tokens
        packed[:, target_end:] = reference_tokens

        text_positions = torch.arange(text_length, device=device).view(-1, 1).expand(text_length, 3)
        target_positions = self._image_position_ids(grid_h, grid_w, device)
        reference_positions = self._image_position_ids(reference_grid_h, reference_grid_w, device)
        if self.reference_position_scale != 1.0:
            reference_positions = reference_positions.to(torch.float32)
            scale_bias = (self.reference_position_scale - 1.0) / 2.0
            reference_positions[:, 1:] = IMAGE_POSITION_OFFSET + (
                reference_positions[:, 1:] - IMAGE_POSITION_OFFSET
            ) * self.reference_position_scale + scale_bias
        reference_positions = offset_reference_positions(
            reference_positions,
            self.reference_position_offset,
        )
        position_ids = torch.cat(
            [text_positions, target_positions, reference_positions],
            dim=0,
        ).unsqueeze(0).expand(batch_size, sequence_length, 3)

        indicator = torch.empty(batch_size, sequence_length, dtype=torch.long, device=device)
        indicator[:, :text_length] = LLM_TOKEN_INDICATOR
        indicator[:, target_start:target_end] = OUTPUT_IMAGE_INDICATOR
        indicator[:, target_end:] = REFERENCE_IMAGE_INDICATOR

        segment_ids = torch.ones(batch_size, sequence_length, dtype=torch.long, device=device)
        padding = text_attention_mask == 0
        segment_ids[:, :text_length][padding] = SEQUENCE_PADDING_INDICATOR
        indicator[:, :text_length][padding] = 0
        attention_mask = (
            segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)
        ).unsqueeze(1)

        output_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(packed.dtype).unsqueeze(-1)
        reference_mask = (indicator == REFERENCE_IMAGE_INDICATOR).to(packed.dtype).unsqueeze(-1)
        image_mask = output_mask + reference_mask

        packed = packed * image_mask
        hidden_states = self.input_proj(packed) * image_mask

        model_timesteps = build_model_timesteps(
            timesteps,
            sequence_length,
            target_end,
            self.reference_model_timestep,
        )
        timestep_condition = self.t_embedding(model_timesteps, dtype=packed.dtype)
        adaln_input = F.silu(self.adaln_proj(timestep_condition))

        text_mask = (
            indicator[:, :text_length] == LLM_TOKEN_INDICATOR
        ).to(packed.dtype).unsqueeze(-1)
        llm = self.llm_cond_norm(context * text_mask)
        llm = self.llm_cond_proj(llm) * text_mask
        hidden_states[:, :text_length] = hidden_states[:, :text_length] + llm

        hidden_states += self.embed_image_indicator(
            image_mask.squeeze(-1).to(torch.long)
        )

        freqs_cis = precompute_freqs_cis(
            self.head_dim,
            position_ids[0].transpose(0, 1),
            self.rope_theta,
            rope_dims=self.mrope_section,
            interleaved_mrope=True,
            device=position_ids.device,
        )

        attention_mask = torch.zeros_like(
            attention_mask,
            dtype=hidden_states.dtype,
        ).masked_fill_(~attention_mask, -torch.finfo(hidden_states.dtype).max)

        sizes = torch.tensor([text_length, grid_h, grid_w], device=hidden_states.device)
        return make_contiguous(
            hidden_states,
            attention_mask,
            adaln_input,
            sizes,
            *freqs_cis,
        )


class Ideogram4ReferenceFinalLayer(nn.Module):
    """Decode only the noisy target span; reference tokens never enter the loss."""

    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    def forward(self, inputs):
        hidden_states, attention_mask, adaln_input, sizes, *freqs_cis = inputs
        output = self.final_layer(hidden_states, adaln_input)
        text_length, grid_h, grid_w = sizes
        target_length = grid_h * grid_w
        target_output = output[:, text_length:text_length + target_length]
        return -self._tokens_to_img(target_output, grid_h, grid_w)
