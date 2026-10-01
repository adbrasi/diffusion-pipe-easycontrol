#!/usr/bin/env python3
"""Compare actual FP8 checkpoint linears in stock ComfyUI and the training wrapper."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-repo', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--comfy', type=Path, help='ComfyUI actually used for inference')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--matmul', choices=['bf16', 'comfy'], default='bf16')
    args = parser.parse_args()
    sys.path.insert(0, str(args.train_repo))
    import utils.common
    sys.path.insert(0, str(args.comfy or args.train_repo / 'submodules/ComfyUI'))
    import torch
    import comfy.ops
    from models.base import ScaledFP8Linear
    from safetensors import safe_open
    torch.manual_seed(76)
    torch.set_num_threads(8)
    dtype = torch.bfloat16
    ops = comfy.ops.mixed_precision_ops(compute_dtype=dtype)
    results = []
    with safe_open(args.checkpoint, framework='pt', device='cpu') as source:
        configs = json.loads(source.metadata()['_quantization_metadata'])['layers']
        for key in ['blocks.0.attn.wq', 'blocks.0.attn.gate', 'txtfusion.refiner_blocks.0.attn.wq']:
            weight = source.get_tensor(key + '.weight')
            out_features, in_features = weight.shape
            has_bias = key + '.bias' in source.keys()
            stock = ops.Linear(in_features, out_features, bias=has_bias, device='cuda', dtype=dtype)
            state = dict(weight=weight, weight_scale=source.get_tensor(key + '.weight_scale'),
                         comfy_quant=torch.tensor(list(json.dumps(configs[key]).encode()), dtype=torch.uint8))
            if has_bias:
                state['bias'] = source.get_tensor(key + '.bias')
            stock.load_state_dict(state, strict=True)
            native = ScaledFP8Linear.from_comfy(stock, matmul=args.matmul).to('cuda')
            x = torch.randn(1, 32, in_features, device='cuda', dtype=dtype)
            xs = x.detach().clone().requires_grad_(True)
            xn = x.detach().clone().requires_grad_(True)
            ys = stock(xs)
            yn = native(xn)
            ys.float().square().mean().backward()
            yn.float().square().mean().backward()
            with torch.no_grad():
                yi = stock(x)
            def relative(a, b):
                return float((a.detach().float()-b.detach().float()).norm()/b.detach().float().norm().clamp_min(1e-30))
            result = dict(module=key, config=configs[key], stock_full_precision_mm=stock._full_precision_mm,
                          storage=str(native.weight.dtype), stock_train_vs_infer_relative_l2=relative(ys,yi),
                          native_vs_stock_output_relative_l2=relative(yn,ys),
                          native_vs_stock_input_gradient_relative_l2=relative(xn.grad,xs.grad))
            results.append(result)
            print(json.dumps(result), flush=True)
            del stock, native, x, xs, xn, ys, yn, yi, state, weight
            torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(checkpoint=str(args.checkpoint), matmul=args.matmul, results=results), indent=2))


if __name__ == '__main__':
    main()
