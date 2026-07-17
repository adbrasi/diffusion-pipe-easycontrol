"""Omini-Grounded: OminiControl's winning core + Qwen3-VL visual grounding.

Composes (via imports only — the proven pipelines stay untouched):

- from krea2_ominicontrol: condition-only LoRA routing on the DiT blocks
  (the delta only touches reference rows; target/text rows run the frozen
  base) and the per-block router span plumbing;
- from krea2_edit: dual conditioning — the reference also reaches the
  Qwen3-VL vision tower together with the caption, and the
  TextFusionTransformer is trainable (globally, not routed) so the grounding
  can adapt.

Geometry/timestep follow the empirically validated omini recipe: width-shift
reference positions and reference tokens modulated at t=0 (the contract the
working omini adapter actually trained with).
"""

from models.condition_lora import ConditionOnlyLoRARouter
from models.krea2_edit import Krea2EditPipeline
from models.krea2_ominicontrol import Krea2OminiTransformerLayer
from models.krea2_reference import (
    Krea2ReferenceFinalLayer,
    Krea2ReferenceInitialLayer,
)
from utils.common import is_main_process


class Krea2OminiGroundedPipeline(Krea2EditPipeline):
    name = 'krea2_omini_grounded'
    config_section = 'krea2_omini_grounded'
    checkpointable_layers = [
        'Krea2ReferenceInitialLayer', 'TransformerLayer', 'Krea2OminiTransformerLayer'
    ]
    adapter_allowed_key_substrings = ('.blocks.', '.txtfusion.')

    def __init__(self, config):
        super().__init__(config)
        section = config.get(self.config_section, {})
        self.condition_only_lora = bool(section.get('condition_only_lora', True))
        self.condition_lora_router = ConditionOnlyLoRARouter(self.condition_only_lora)

    def configure_adapter(self, adapter_config):
        # krea2_edit coverage: every linear in the SingleStreamBlocks plus the
        # TextFusionTransformer (projector excluded).
        super().configure_adapter(adapter_config)
        if self.condition_only_lora:
            # Route ONLY the block deltas to the reference span. The txtfusion
            # LoRA stays global: it runs on the text stream (different
            # sequence), where masking by image span would be meaningless.
            installed = self.condition_lora_router.install(self.diffusion_model.blocks)
            if is_main_process():
                print(f'[{self.name}] condition-only routing on {installed} block linears '
                      f'(txtfusion LoRA stays global)')

    def to_layers(self):
        model = self.diffusion_model
        layers = [
            Krea2ReferenceInitialLayer(
                model,
                position_mode=self.position_mode,
                reference_position_offset=self.reference_position_offset,
                reference_position_scale=self.reference_position_scale,
                independent_condition=self.independent_condition,
                reference_timestep_mode=self.reference_timestep_mode,
            )
        ]
        layers.extend(
            Krea2OminiTransformerLayer(block, index, self.offloader, self.condition_lora_router)
            for index, block in enumerate(model.blocks)
        )
        layers.append(Krea2ReferenceFinalLayer(model))
        return layers

    def get_reference_metadata(self):
        metadata = super().get_reference_metadata()
        metadata['control_family'] = 'krea2_omini_grounded'
        metadata['condition_only_lora'] = str(self.condition_only_lora).lower()
        return metadata
