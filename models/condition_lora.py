"""Branch-routed PEFT LoRA support for packed visual conditions.

OminiControl applies its adapter to condition branches only. A normal PEFT
forward applies the same LoRA delta to every row of a packed sequence, which is
not equivalent. This router replaces PEFT Linear forwards with the standard
base-plus-LoRA computation and masks each LoRA delta to the active reference
span. Base weights and checkpoint keys remain unchanged.
"""

import types

import torch


class ConditionOnlyLoRARouter:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.reference_start = None
        self.reference_end = None
        self.sequence_length = None
        self._installed = []

    def set_reference_span(self, start, end):
        self.reference_start = int(start)
        self.reference_end = int(end)
        self.sequence_length = int(end)

    def install(self, model):
        if not self.enabled:
            return 0
        for module in model.modules():
            if not all(hasattr(module, name) for name in ('base_layer', 'lora_A', 'lora_B', 'lora_dropout', 'scaling')):
                continue
            if getattr(module, '_condition_only_lora_router', None) is self:
                continue
            module._condition_only_lora_router = self
            module.forward = types.MethodType(_condition_only_lora_forward, module)
            self._installed.append(module)
        if not self._installed:
            raise RuntimeError('Condition-only LoRA routing found no PEFT Linear modules')
        return len(self._installed)

    def mask_for(self, value):
        if self.reference_start is None:
            raise RuntimeError('Condition LoRA reference span was not set before a transformer block')
        if value.ndim < 3 or value.shape[-2] != self.sequence_length:
            raise RuntimeError(
                'Condition-only LoRA expected a packed sequence with length '
                f'{self.sequence_length}, got {tuple(value.shape)}'
            )
        mask = value.new_zeros(*value.shape[:-1], 1)
        mask[..., self.reference_start:self.reference_end, :] = 1
        return mask


def _condition_only_lora_forward(module, x, *args, **kwargs):
    router = module._condition_only_lora_router
    if not router.enabled or getattr(module, 'disable_adapters', False) or getattr(module, 'merged', False):
        return module.base_layer(x, *args, **kwargs)

    result = module.base_layer(x, *args, **kwargs)
    result_dtype = result.dtype
    mask = router.mask_for(result)
    active_adapters = getattr(module, 'active_adapters', ['default'])
    if isinstance(active_adapters, str):
        active_adapters = [active_adapters]

    for adapter in active_adapters:
        if adapter not in module.lora_A:
            continue
        lora_a = module.lora_A[adapter]
        lora_b = module.lora_B[adapter]
        dropout = module.lora_dropout[adapter]
        scaling = module.scaling[adapter]
        adapter_input = x
        if hasattr(module, '_cast_input_dtype'):
            adapter_input = module._cast_input_dtype(adapter_input, lora_a.weight.dtype)
        else:
            adapter_input = adapter_input.to(lora_a.weight.dtype)
        delta = lora_b(lora_a(dropout(adapter_input))) * scaling
        result = result + delta.to(result.dtype) * mask
    return result.to(result_dtype)
