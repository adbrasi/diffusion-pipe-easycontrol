"""Role-aware LoRA routing for the FIRA next-scene contract.

Sequence layout: [identity capsule | delta text | noisy target | compact VAE ref].
Each row role receives a different LoRA delta policy per module kind:

module            capsule  text  target       reference
llm_cond_proj       full   zero  (n/a)        (n/a)
attention.qkv       zero   zero  Q chunk only full
attention.o         zero   zero  full         full
feed_forward.*      zero   zero  zero         full
adaln_modulation    zero   zero  zero         full

The target thus learns to QUERY (Q) and to INTEGRATE what it retrieved (O),
while the reference rows learn to present themselves as a memory. Text keeps
the stock Ideogram pathway; the capsule adapts only at the entry projection.
"""

import types

import torch


class RoleAwareLoRARouter:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.capsule_end = None
        self.text_end = None
        self.target_end = None
        self.sequence_length = None
        self._installed = []

    def set_spans(self, capsule_end, text_end, target_end, sequence_length):
        self.capsule_end = int(capsule_end)
        self.text_end = int(text_end)
        self.target_end = int(target_end)
        self.sequence_length = int(sequence_length)

    def install(self, named_modules):
        """named_modules: iterable of (qualified_name, module). Wraps PEFT linears."""
        if not self.enabled:
            return 0
        for name, module in named_modules:
            if not all(hasattr(module, attr) for attr in ('base_layer', 'lora_A', 'lora_B', 'lora_dropout', 'scaling')):
                continue
            if getattr(module, '_role_router', None) is self:
                continue
            kind = _module_kind(name)
            if kind is None:
                continue
            module._role_router = self
            module._role_kind = kind
            module.forward = types.MethodType(_role_routed_forward, module)
            self._installed.append(module)
        if not self._installed:
            raise RuntimeError('Role-aware LoRA routing found no PEFT Linear modules')
        return len(self._installed)

    def delta_mask(self, kind, delta):
        """Row/chunk mask for the LoRA delta of a packed-sequence linear."""
        if self.sequence_length is None:
            raise RuntimeError('Role spans were not set before a routed forward')
        if delta.ndim < 3 or delta.shape[-2] != self.sequence_length:
            raise RuntimeError(
                f'Role-routed LoRA expected packed length {self.sequence_length}, '
                f'got {tuple(delta.shape)} for kind {kind}'
            )
        mask = delta.new_zeros(*delta.shape[:-1], 1)
        c, t, g = self.capsule_end, self.text_end, self.target_end
        if kind == 'qkv':
            out = delta.shape[-1]
            if out % 3:
                raise RuntimeError(f'qkv output dim {out} is not divisible by 3')
            h = out // 3
            mask = mask.expand(*delta.shape[:-1], out).clone()
            mask[..., g:, :] = 1                # reference: Q, K and V
            mask[..., t:g, :h] = 1              # target: Q chunk only
            return mask
        if kind == 'o':
            mask[..., t:g, :] = 1               # target integrates
            mask[..., g:, :] = 1                # reference forms the memory
            return mask
        if kind in ('mlp', 'adaln'):
            mask[..., g:, :] = 1                # reference rows only
            return mask
        raise RuntimeError(f'Unknown routed module kind: {kind}')


def _module_kind(name):
    leaf = name.split('.')
    if 'llm_cond_proj' in name:
        return None  # handled by LlmProjCapsuleRouter below (different seq length)
    if 'adaln_modulation' in name:
        return 'adaln'
    if 'attention' in leaf:
        if leaf[-1] == 'qkv':
            return 'qkv'
        if leaf[-1] == 'o':
            return 'o'
        return None
    if 'feed_forward' in leaf:
        return 'mlp'
    return None


def _role_routed_forward(module, x, *args, **kwargs):
    router = module._role_router
    if not router.enabled or getattr(module, 'disable_adapters', False) or getattr(module, 'merged', False):
        return module.base_layer(x, *args, **kwargs)

    result = module.base_layer(x, *args, **kwargs)
    result_dtype = result.dtype
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
        mask = router.delta_mask(module._role_kind, delta)
        result = result + delta.to(result.dtype) * mask
    return result.to(result_dtype)


class LlmProjCapsuleRouter:
    """llm_cond_proj runs on the [capsule | text] sequence (before DiT packing):
    the LoRA delta applies ONLY to the capsule rows; text keeps the base path."""

    def __init__(self, capsule_tokens):
        self.capsule_tokens = int(capsule_tokens)
        self._installed = []

    def install(self, named_modules):
        for name, module in named_modules:
            if 'llm_cond_proj' not in name:
                continue
            if not all(hasattr(module, attr) for attr in ('base_layer', 'lora_A', 'lora_B', 'lora_dropout', 'scaling')):
                continue
            module._capsule_router = self
            module.forward = types.MethodType(_capsule_only_forward, module)
            self._installed.append(module)
        return len(self._installed)


def _capsule_only_forward(module, x, *args, **kwargs):
    router = module._capsule_router
    result = module.base_layer(x, *args, **kwargs)
    if getattr(module, 'disable_adapters', False) or getattr(module, 'merged', False):
        return result
    result_dtype = result.dtype
    n = router.capsule_tokens
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
        mask = delta.new_zeros(*delta.shape[:-1], 1)
        if delta.ndim >= 3 and delta.shape[-2] >= n:
            mask[..., :n, :] = 1
        result = result + delta.to(result.dtype) * mask
    return result.to(result_dtype)
