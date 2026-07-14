"""OminiControl2 training for Ideogram 4.

Adds compact reference tokens and asymmetric attention to OminiControl v1.
Reference queries can only attend to reference keys, making their hidden states
independent from noisy target/text tokens and suitable for inference KV reuse.
"""

import torch
import torch.nn.functional as F

from models.base import make_contiguous
from models.ideogram4 import TransformerLayer
from models.ideogram4_ic_lora import Ideogram4ReferenceFinalLayer, Ideogram4ReferenceInitialLayer
from models.ideogram4_ominicontrol import (
    Ideogram4OminiControlPipeline,
    Ideogram4OminiTransformerLayer,
)
from models.ideogram4_reference_contract import apply_reference_dropout


class Ideogram4OminiControl2Pipeline(Ideogram4OminiControlPipeline):
    name = 'ideogram4_ominicontrol2'

    def __init__(self, config):
        super().__init__(config)
        control = config.get('ominicontrol', {})
        self.independent_condition = bool(control.get('independent_condition', True))
        self.condition_token_stride = int(control.get('condition_token_stride', 2))
        self.reference_position_scale = float(
            control.get('reference_position_scale', self.condition_token_stride)
        )
        if self.condition_token_stride < 1:
            raise ValueError('condition_token_stride must be >= 1')
        if self.reference_position_scale <= 0:
            raise ValueError('reference_position_scale must be > 0')

    def prepare_reference_media(self, reference):
        if self.condition_token_stride == 1:
            return reference
        if reference.ndim != 4:
            raise ValueError(f'Ideogram4 reference pixels must be BCHW, got {tuple(reference.shape)}')
        height, width = reference.shape[-2:]
        return F.interpolate(
            reference,
            size=(height // self.condition_token_stride, width // self.condition_token_stride),
            mode='bilinear',
            align_corners=False,
            antialias=True,
        )

    def prepare_reference_latents(self, reference_latents, noisy_target, timestep_quantile=None):
        if reference_latents.shape[:2] != noisy_target.shape[:2]:
            raise ValueError('Ideogram4 compact reference batch/channels must match target')
        expected = tuple(size // self.condition_token_stride for size in noisy_target.shape[-2:])
        if reference_latents.shape[-2:] != expected:
            raise ValueError(
                f'Ideogram4 compact reference latent shape {tuple(reference_latents.shape[-2:])} '
                f'does not match expected {expected}; regenerate the VAE cache.'
            )
        return apply_reference_dropout(
            reference_latents,
            self.condition_dropout,
            enabled=timestep_quantile is None,
        )

    def to_layers(self):
        model = self.diffusion_model
        layers = [
            Ideogram4OminiControl2InitialLayer(
                model,
                reference_position_offset=self.reference_position_offset,
                reference_model_timestep=self.reference_model_timestep,
                reference_position_scale=self.reference_position_scale,
                independent_condition=self.independent_condition,
            )
        ]
        layers.extend(
            Ideogram4OminiTransformerLayer(block, index, self.offloader, self.condition_lora_router)
            for index, block in enumerate(model.layers)
        )
        layers.append(Ideogram4ReferenceFinalLayer(model))
        return layers

    def get_reference_metadata(self):
        metadata = super().get_reference_metadata()
        metadata.update({
            'control_family': 'ominicontrol_v2',
            'independent_condition': str(self.independent_condition).lower(),
            'condition_token_stride': str(self.condition_token_stride),
            'reference_position_scale': str(self.reference_position_scale),
            'condition_encode': 'pixel_bilinear',
        })
        return metadata


class Ideogram4OminiControl2InitialLayer(Ideogram4ReferenceInitialLayer):
    def __init__(self, model, independent_condition=True, **kwargs):
        super().__init__(model, require_matching_shape=False, **kwargs)
        self.independent_condition = independent_condition

    def forward(self, inputs):
        outputs = list(super().forward(inputs))
        if not self.independent_condition:
            return tuple(outputs)

        hidden_states, attention_mask, adaln_input, sizes, *freqs_cis = outputs
        text_length, target_grid_h, target_grid_w = (int(value) for value in sizes)
        target_end = text_length + target_grid_h * target_grid_w
        reference_start = target_end

        attention_mask = attention_mask.clone()
        blocked = -torch.finfo(attention_mask.dtype).max
        attention_mask[:, :, reference_start:, :] = blocked
        attention_mask[:, :, reference_start:, reference_start:] = 0
        return make_contiguous(hidden_states, attention_mask, adaln_input, sizes, *freqs_cis)
