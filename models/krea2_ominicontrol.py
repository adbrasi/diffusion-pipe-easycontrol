"""OminiControl v1 for Krea 2."""

from models.base import make_contiguous
from models.condition_lora import ConditionOnlyLoRARouter
from models.krea2 import TransformerLayer
from models.krea2_reference import (
    Krea2ReferenceFinalLayer,
    Krea2ReferenceInitialLayer,
    Krea2ReferencePipeline,
)


class Krea2OminiControlPipeline(Krea2ReferencePipeline):
    name = 'krea2_ominicontrol'
    config_section = 'ominicontrol'
    # to_layers() wraps blocks in Krea2OminiTransformerLayer; DeepSpeed matches
    # checkpointable layers by exact class name, so without this entry
    # activation checkpointing is silently disabled for every block (same bug
    # class fixed for Ideogram4OminiTransformerLayer).
    checkpointable_layers = [
        'Krea2ReferenceInitialLayer', 'TransformerLayer', 'Krea2OminiTransformerLayer'
    ]

    def __init__(self, config):
        super().__init__(config)
        control = config.get('ominicontrol', {})
        self.condition_only_lora = bool(control.get('condition_only_lora', True))
        self.condition_lora_router = ConditionOnlyLoRARouter(self.condition_only_lora)
        if self.position_mode == 'spatial':
            self.reference_position_offset = 0.0

    def configure_adapter(self, adapter_config):
        super().configure_adapter(adapter_config)
        installed = self.condition_lora_router.install(self.diffusion_model)
        if installed:
            print(f'[{self.name}] condition-only LoRA routing installed on {installed} PEFT linears')

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
        layers.extend(
            Krea2OminiTransformerLayer(block, index, self.offloader, self.condition_lora_router)
            for index, block in enumerate(model.blocks)
        )
        layers.append(Krea2ReferenceFinalLayer(model))
        return layers

    def get_reference_metadata(self):
        return {
            'control_family': 'ominicontrol_v1',
            'condition_only_lora': str(self.condition_only_lora).lower(),
        }


class Krea2OminiTransformerLayer(TransformerLayer):
    def __init__(self, layer, block_idx, offloader, router):
        super().__init__(layer, block_idx, offloader)
        self.router = router

    def forward(self, inputs):
        combined, target_timestep, tvec, freqs, attention_mask, sizes = inputs
        text_length, target_length = int(sizes[0]), int(sizes[1])
        self.router.set_reference_span(text_length + target_length, combined.shape[1])
        return super().forward(
            make_contiguous(combined, target_timestep, tvec, freqs, attention_mask, sizes)
        )
