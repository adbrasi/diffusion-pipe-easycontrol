#!/usr/bin/env python3
"""Small CPU test for condition-only PEFT routing (no base model required)."""

from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.condition_lora import ConditionOnlyLoRARouter


class FakePeftLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_layer = nn.Linear(4, 3, bias=False)
        self.lora_A = nn.ModuleDict({'default': nn.Linear(4, 2, bias=False)})
        self.lora_B = nn.ModuleDict({'default': nn.Linear(2, 3, bias=False)})
        self.lora_dropout = nn.ModuleDict({'default': nn.Identity()})
        self.scaling = {'default': 1.0}
        self.active_adapters = ['default']
        self.disable_adapters = False
        self.merged = False

    def forward(self, value):
        return self.base_layer(value) + self.lora_B['default'](self.lora_A['default'](value))


def main():
    torch.manual_seed(0)
    module = FakePeftLinear()
    router = ConditionOnlyLoRARouter()
    assert router.install(module) == 1
    router.set_reference_span(3, 5)

    value = torch.randn(2, 5, 4)
    result = module(value)
    base = module.base_layer(value)
    delta = module.lora_B['default'](module.lora_A['default'](value))
    assert torch.equal(result[:, :3], base[:, :3])
    assert torch.allclose(result[:, 3:], base[:, 3:] + delta[:, 3:])

    module.zero_grad(set_to_none=True)
    result[:, :3].sum().backward()
    assert module.lora_A['default'].weight.grad is not None
    assert module.lora_A['default'].weight.grad.abs().max() == 0
    print('condition-only LoRA routing: OK')


if __name__ == '__main__':
    main()
