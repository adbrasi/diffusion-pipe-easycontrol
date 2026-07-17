"""IC-LoRA Routed for Anima — condition-only LoRA routing ported from the
Krea 2 omini/omini-grounded method.

Identical training contract to ic_lora_v2 (ref_first [ref t=0 | target t=sigma],
loss only on the target frame, adaln/cross_attn/llm_adapter excluded, shifted
logit-normal sampler), plus ONE architectural change: the LoRA delta of every
routed linear applies ONLY to the reference-frame rows. Target rows always run
the frozen base -> zero drift of the base model on the generated frame, the
adapter only learns "how the reference should present itself" through
attention K/V (the OminiControl routing insight validated on Krea 2).

Token layouts inside Anima blocks (cosmos_predict2_modeling.Block):
- self_attn q/k/v/proj: (B, T*H*W, D) with frames flattened "t h w"
  -> ref_first means the reference occupies the FIRST H*W rows.
- mlp layers: (B, T, H, W, D) 5D
  -> mask over the T axis (frame 0 when ref_first).
cross_attn never carries LoRA (forbidden pattern), so those are the only two
shapes the router must handle.

Config: [ic_lora_full] section (inherited) + condition_only_lora (default true).
"""

import types

import torch

from models.ic_lora_v2 import ICLoraV2Pipeline
from models.cosmos_predict2 import (
    InitialLayer,
    TransformerLayer,
    FinalLayer,
    LLMAdapterLayer,
)


class AnimaConditionRouter:
    """Masks PEFT LoRA deltas to the reference frame of the temporal concat.

    Frame geometry (T, H, W in token units) is set per forward pass by
    RoutedTransformerLayer before each block executes.
    """

    def __init__(self, enabled=True, ref_first=True):
        self.enabled = enabled
        self.ref_first = ref_first
        self.t = None
        self.hw = None
        self._installed = []

    def set_frame_geometry(self, t, h, w):
        self.t = int(t)
        self.hw = int(h) * int(w)

    def install(self, blocks):
        if not self.enabled:
            return 0
        for module in blocks.modules():
            if not all(hasattr(module, name) for name in ('base_layer', 'lora_A', 'lora_B', 'lora_dropout', 'scaling')):
                continue
            if getattr(module, '_anima_condition_router', None) is self:
                continue
            module._anima_condition_router = self
            module.forward = types.MethodType(_routed_lora_forward, module)
            self._installed.append(module)
        if not self._installed:
            raise RuntimeError('Anima condition routing found no PEFT Linear modules in the blocks')
        return len(self._installed)

    def mask_for(self, value):
        if self.t is None:
            raise RuntimeError('Frame geometry was not set before a routed linear executed')
        if self.t == 1:
            # No reference frame present (plain single-frame fallback): no delta.
            return value.new_zeros(*([1] * (value.ndim - 1)), 1)
        if value.ndim == 3 and value.shape[1] == self.t * self.hw:
            # self_attn path: (B, T*H*W, D), frames flattened "t h w"
            mask = value.new_zeros(1, self.t * self.hw, 1)
            if self.ref_first:
                mask[:, : self.hw] = 1
            else:
                mask[:, -self.hw :] = 1
            return mask
        if value.ndim == 5 and value.shape[1] == self.t:
            # mlp path: (B, T, H, W, D)
            mask = value.new_zeros(1, self.t, 1, 1, 1)
            mask[:, 0 if self.ref_first else -1] = 1
            return mask
        raise RuntimeError(
            f'Routed linear got unexpected shape {tuple(value.shape)} for T={self.t}, HW={self.hw}'
        )


def _routed_lora_forward(module, x, *args, **kwargs):
    router = module._anima_condition_router
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


class RoutedTransformerLayer(TransformerLayer):
    """Sets the router frame geometry from the live tensor before each block."""

    def __init__(self, block, block_idx, offloader, router):
        super().__init__(block, block_idx, offloader)
        self.router = router

    def forward(self, inputs):
        x_B_T_H_W_D = inputs[0]
        _, t, h, w, _ = x_B_T_H_W_D.shape
        self.router.set_frame_geometry(t, h, w)
        return super().forward(inputs)


class ICLoraRoutedPipeline(ICLoraV2Pipeline):
    adapter_log_tag = 'IC-LoRA Routed'
    # A lição do krea2 omini: subclasse de layer PRECISA estar aqui, senão o
    # activation checkpointing silenciosamente não se aplica a ela.
    checkpointable_layers = ['TransformerLayer', 'RoutedTransformerLayer']

    def __init__(self, config):
        super().__init__(config)
        oc_config = config.get('ic_lora_full', {})
        self.condition_only_lora = bool(oc_config.get('condition_only_lora', True))
        self.condition_router = AnimaConditionRouter(
            enabled=self.condition_only_lora, ref_first=self.ref_first,
        )

    def configure_adapter(self, adapter_config):
        super().configure_adapter(adapter_config)
        if self.condition_only_lora:
            installed = self.condition_router.install(self.transformer.blocks)
            print(f'[{self.adapter_log_tag}] condition-only routing on {installed} PEFT linears '
                  f'(ref_first={self.ref_first})')

    def to_layers(self):
        transformer = self.transformer
        text_encoder = None if self.cache_text_embeddings else self.text_encoder
        layers = [
            InitialLayer(transformer, text_encoder, self.is_generic_llm),
            LLMAdapterLayer(transformer.llm_adapter if self.use_llm_adapter else None),
        ]
        for i, block in enumerate(transformer.blocks):
            layers.append(RoutedTransformerLayer(block, i, self.offloader, self.condition_router))
        layers.append(FinalLayer(transformer))
        return layers
