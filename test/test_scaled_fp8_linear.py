"""ScaledFP8Linear keeps a ComfyUI fp8_scaled weight bit-exact and trains LoRA on top."""

import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
ck_base = pytest.importorskip('comfy_kitchen.tensor.base')


def _import_base():
    if not torch.cuda.is_available() and 'comfy' not in sys.modules:
        sys.path.insert(0, os.path.join(ROOT, 'submodules', 'ComfyUI'))
        saved, sys.argv = sys.argv, ['pytest', '--cpu']
        try:
            import comfy.options
            comfy.options.enable_args_parsing()
            import comfy.model_management  # noqa: F401
        finally:
            sys.argv = saved
            sys.path.pop(0)
    import utils.common  # noqa: F401
    import models.base as base
    return base


def _comfy_fp8_linear(out_f=48, in_f=32, std=0.02):
    lin = torch.nn.Linear(in_f, out_f)
    q = ck_base.QuantizedTensor.from_float(torch.randn(out_f, in_f) * std, 'TensorCoreFP8Layout')
    lin.weight = torch.nn.Parameter(q, requires_grad=False)
    return lin, q


def test_bit_exact_with_comfy_dequant_and_no_zeroing():
    base = _import_base()
    lin, q = _comfy_fp8_linear()
    assert base.ScaledFP8Linear.accepts(lin)
    m = base.ScaledFP8Linear.from_comfy(lin)
    ref = q.dequantize()
    assert torch.equal(m.dequantized_weight(ref.dtype), ref)
    x = torch.randn(3, 32)
    assert torch.allclose(m(x), torch.nn.functional.linear(x, ref, lin.bias), atol=1e-6)


def test_peft_wraps_it_and_only_lora_gets_grad():
    import peft
    base = _import_base()
    lin, _ = _comfy_fp8_linear()
    net = torch.nn.Sequential()
    net.add_module('proj', base.ScaledFP8Linear.from_comfy(lin))
    model = peft.get_peft_model(net, peft.LoraConfig(r=4, lora_alpha=4, target_modules=['proj']))
    for n, p in model.named_parameters():
        if 'lora_B' in n:
            torch.nn.init.normal_(p, std=0.01)
    x = torch.randn(5, 32, requires_grad=True)
    model(x).pow(2).mean().backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.requires_grad}
    assert grads and all('lora_' in n for n in grads) and all(g is not None for g in grads.values())
    assert x.grad is not None
