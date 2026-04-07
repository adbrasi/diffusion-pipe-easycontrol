"""
IC-LoRA V2 pipeline for Anima.

Improved version of IC-LoRA with targeted LoRA application:
- Excludes adaln_modulation from LoRA targets (Anima already has internal adaln_lora)
- Excludes cross_attn from LoRA targets (cross-attention handles text, not visual consistency)
- LoRA applied ONLY to self_attn + mlp (~69% of block parameters, focused on visual features)

Based on analysis of:
- LTX-2 IC-LoRA trainer (does NOT train adaln)
- Opus reviewer findings: adaln LoRA causes "double-LoRA" amplification with Anima's internal adaln_lora
- Empirical evidence: skip_adaln=True at inference improves quality dramatically

Everything else (temporal concat, per-token timestep, loss masking) is identical to V1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import peft

from models.ic_lora import ICLoraPipeline
from utils import is_main_process


class ICLoraV2Pipeline(ICLoraPipeline):
    """IC-LoRA V2: focused LoRA targets for higher quality.

    Inherits all training logic from V1 (prepare_inputs, get_loss_fn).
    Only difference: configure_adapter excludes adaln_modulation and cross_attn.
    """

    def configure_adapter(self, adapter_config):
        """Apply LoRA ONLY to self_attn and mlp modules inside Block/TransformerBlock.

        Excluded:
        - adaln_modulation: Anima has internal adaln_lora (bottleneck dim=256).
          Adding PEFT LoRA on top causes double-LoRA amplification.
        - cross_attn: handles text conditioning, not visual consistency.
          Training it wastes LoRA capacity on text alignment that the base model already does well.
        """
        excluded_submodules = {'adaln_modulation', 'cross_attn'}

        target_linear_modules = set()
        for name, module in self.transformer.named_modules():
            if module.__class__.__name__ not in self.adapter_target_modules:
                continue
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, nn.Linear):
                    # Check if this linear belongs to an excluded submodule
                    parts = full_submodule_name.split('.')
                    if any(part in excluded_submodules for part in parts):
                        continue
                    target_linear_modules.add(full_submodule_name)
        target_linear_modules = list(target_linear_modules)

        if is_main_process():
            print(f'[IC-LoRA V2] LoRA targets: {len(target_linear_modules)} linear modules (excluding adaln_modulation, cross_attn)')

        adapter_type = adapter_config['type']
        if adapter_type == 'lora':
            peft_config = peft.LoraConfig(
                r=adapter_config['rank'],
                lora_alpha=adapter_config['alpha'],
                lora_dropout=adapter_config['dropout'],
                bias='none',
                target_modules=target_linear_modules
            )
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
        self.peft_config = peft_config
        self.lora_model = peft.get_peft_model(self.transformer, peft_config)
        if is_main_process():
            self.lora_model.print_trainable_parameters()
        for name, p in self.transformer.named_parameters():
            p.original_name = name
            if p.requires_grad:
                p.data = p.data.to(adapter_config['dtype'])
