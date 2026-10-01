"""GPU regression against ComfyUI's actual FP8 forward/backward implementation."""
import json
import os
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import utils.common
sys.path.insert(0, os.environ.get('KREA2_STOCK_COMFY', str(ROOT / 'submodules/ComfyUI')))
import comfy.ops
from models.base import ScaledFP8Linear


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Actual FP8 Tensor Core regression needs CUDA')
@pytest.mark.parametrize('full_precision', [False, True])
@pytest.mark.parametrize('checkpointed', [False, True])
def test_fp8_policy_matches_stock_with_peft_gradients(full_precision, checkpointed):
    from comfy_kitchen.tensor.base import QuantizedTensor
    from peft import get_peft_model, LoraConfig
    if not hasattr(comfy.ops, 'QuantLinearFunc'):
        pytest.skip('Set KREA2_STOCK_COMFY to the inference checkout')
    torch.manual_seed(76)
    dtype = torch.bfloat16
    q = QuantizedTensor.from_float(torch.randn(128, 128, device='cuda', dtype=dtype), 'TensorCoreFP8E4M3Layout')
    ops = comfy.ops.mixed_precision_ops(compute_dtype=dtype)
    stock = ops.Linear(128, 128, bias=True, device='cuda', dtype=dtype)
    conf = dict(format='float8_e4m3fn', full_precision_matrix_mult=full_precision)
    stock.load_state_dict(dict(weight=q._qdata, weight_scale=q._params.scale,
                               bias=torch.randn(128, device='cuda', dtype=dtype),
                               comfy_quant=torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)))
    stock.requires_grad_(False)
    native = ScaledFP8Linear.from_comfy(stock, matmul='comfy')
    x = torch.randn(2, 32, 128, device='cuda', dtype=dtype)
    xs, xn = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    ys = stock(xs)
    yn = checkpoint(native, xn, use_reentrant=False) if checkpointed else native(xn)
    torch.testing.assert_close(yn, ys, rtol=0, atol=0)
    ys.float().square().mean().backward()
    yn.float().square().mean().backward()
    torch.testing.assert_close(xn.grad, xs.grad, rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(native(x), stock(x), rtol=0, atol=0)

    wrapped = get_peft_model(torch.nn.Sequential(native),
                            LoraConfig(r=4, lora_alpha=4, target_modules=['0'], bias='none'))
    adapter = wrapped.base_model.model[0]
    for bank in [adapter.lora_A, adapter.lora_B]:
        for parameter in bank.parameters():
            parameter.data = parameter.data.to(dtype)
    # Fresh PEFT must not alter the FP8 base; then exercise A and B backward.
    with torch.no_grad():
        torch.testing.assert_close(adapter(x), stock(x), rtol=0, atol=0)
        adapter.lora_B['default'].weight.normal_(std=.01)
    out = checkpoint(adapter, x.requires_grad_(True), use_reentrant=False) if checkpointed else adapter(x.requires_grad_(True))
    out.float().square().mean().backward()
    for bank in [adapter.lora_A, adapter.lora_B]:
        for parameter in bank.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().max() > 0
    assert native.weight.dtype == torch.float8_e4m3fn
    assert native.weight.grad is None
