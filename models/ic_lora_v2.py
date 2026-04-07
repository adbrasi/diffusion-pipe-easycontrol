"""
IC-LoRA V2 pipeline for Anima.

Best of both worlds: inherits the corrected training pipeline from ic_lora_full
(proper loss normalization, condition dropout for CFG, ref_first LTX-2 order,
fixed multiscale loss) with focused LoRA targets (self_attn + mlp only).

Key improvements over ic_lora (V1):
- Excludes adaln_modulation from LoRA (Anima has internal adaln_lora — double-LoRA causes amplification)
- Excludes cross_attn from LoRA (text conditioning, not visual consistency)
- Condition dropout (enables classifier-free guidance at inference)
- Corrected loss normalization (no gradient dilution from T=2)
- Fixed multiscale loss for T>1 outputs
- ref_first=True by default (LTX-2 convention: [ref T=0, target T=1])

References:
- LTX-2 IC-LoRA trainer: https://github.com/Lightricks/ltx-video-trainer
- Opus reviewer analysis of adaln_modulation double-LoRA issue
- Empirical evidence: skip_adaln=True at inference improves quality dramatically
"""

import torch.nn as nn

import peft

from models.ic_lora_full import ICLoraFullPipeline
from utils import is_main_process


class ICLoraV2Pipeline(ICLoraFullPipeline):
    """IC-LoRA V2: focused LoRA targets + corrected training pipeline.

    Inherits from ic_lora_full (prepare_inputs, get_loss_fn, condition dropout, ref_first).
    Only override: configure_adapter excludes adaln_modulation and cross_attn from LoRA.
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
