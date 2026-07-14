"""OminiControl v1 training for Ideogram 4.

Uses the proven Ideogram reference-token contract while exposing OminiControl's
spatial and subject position modes. Spatial controls share target MRoPE image
coordinates; subject references use a separate temporal plane.
"""

from models.base import make_contiguous
from models.condition_lora import ConditionOnlyLoRARouter
from models.ideogram4 import TransformerLayer
from models.ideogram4_ic_lora import (
    Ideogram4ICLoRAPipeline,
    Ideogram4ReferenceFinalLayer,
    Ideogram4ReferenceInitialLayer,
)


class Ideogram4OminiControlPipeline(Ideogram4ICLoRAPipeline):
    name = 'ideogram4_ominicontrol'

    def __init__(self, config):
        super().__init__(config)
        control = config.get('ominicontrol', {})
        self.position_mode = control.get('position_mode', 'spatial')
        self.condition_only_lora = bool(control.get('condition_only_lora', True))
        self.condition_lora_router = ConditionOnlyLoRARouter(self.condition_only_lora)
        self.condition_dropout = float(control.get('condition_dropout', self.condition_dropout))
        self.reference_model_timestep = float(
            control.get('reference_model_timestep', self.reference_model_timestep)
        )
        if self.position_mode not in ('spatial', 'subject'):
            raise ValueError("position_mode must be 'spatial' or 'subject'")
        self.reference_position_offset = 0 if self.position_mode == 'spatial' else int(
            control.get('reference_position_offset', 1)
        )

    def configure_adapter(self, adapter_config):
        super().configure_adapter(adapter_config)
        installed = self.condition_lora_router.install(self.diffusion_model)
        if installed:
            print(f'[{self.name}] condition-only LoRA routing installed on {installed} PEFT linears')

    def to_layers(self):
        model = self.diffusion_model
        layers = [
            Ideogram4ReferenceInitialLayer(
                model,
                reference_position_offset=self.reference_position_offset,
                reference_model_timestep=self.reference_model_timestep,
            )
        ]
        layers.extend(
            Ideogram4OminiTransformerLayer(block, index, self.offloader, self.condition_lora_router)
            for index, block in enumerate(model.layers)
        )
        layers.append(Ideogram4ReferenceFinalLayer(model))
        return layers

    def get_reference_metadata(self):
        return {
            'control_family': 'ominicontrol_v1',
            'position_mode': self.position_mode,
            'condition_only_lora': str(self.condition_only_lora).lower(),
        }


class Ideogram4OminiTransformerLayer(TransformerLayer):
    def __init__(self, layer, block_idx, offloader, router):
        super().__init__(layer, block_idx, offloader)
        self.router = router

    def forward(self, inputs):
        hidden_states, attention_mask, adaln_input, sizes, *freqs_cis = inputs
        text_length, target_grid_h, target_grid_w = (int(value) for value in sizes)
        reference_start = text_length + target_grid_h * target_grid_w
        self.router.set_reference_span(reference_start, hidden_states.shape[1])
        return super().forward(
            make_contiguous(hidden_states, attention_mask, adaln_input, sizes, *freqs_cis)
        )
