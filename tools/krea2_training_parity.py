#!/usr/bin/env python3
"""Audit real scaled-FP8/PEFT forwards and gradients with unequal text lengths."""
import argparse
import io
import json
from pathlib import Path
import sqlite3
import sys


def read_cache(path, index):
    with sqlite3.connect(f'file:{path}/metadata.db?mode=ro', uri=True) as con:
        shard, item = con.execute('SELECT shard,shard_index FROM items LIMIT 1 OFFSET ?', (index,)).fetchone()
        offset, size = con.execute(f'SELECT offset,size FROM shard_{shard} LIMIT 1 OFFSET ?', (item,)).fetchone()
    with (path / f'shard_{shard}.bin').open('rb') as stream:
        stream.seek(offset)
        data = stream.read(size)
    import torch
    return torch.load(io.BytesIO(data), map_location='cpu')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-repo', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--individual-padded', action='store_true',
                        help='Control: keep the same padded sequence shape for individual forwards')
    parser.add_argument('--compute-fp32', action='store_true', help='Numerical control; keep base storage FP8')
    parser.add_argument('--latent-height', type=int, default=50)
    parser.add_argument('--latent-width', type=int, default=80)
    args = parser.parse_args()
    sys.path.insert(0, str(args.train_repo))
    import torch
    import toml
    import utils.common
    compute_dtype = torch.float32 if args.compute_fp32 else torch.bfloat16
    utils.common.AUTOCAST_DTYPE = compute_dtype
    sys.path.insert(0, str(args.train_repo / 'submodules/ComfyUI'))
    from models.krea2_native import Krea2NativePipeline
    from models.base import ScaledFP8Linear
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(76)
    config = toml.load(args.config)
    config['model']['dtype'] = compute_dtype
    adapter = dict(config['adapter'], alpha=config['adapter']['rank'], dropout=0., dtype=compute_dtype)
    pipe = Krea2NativePipeline(config)
    pipe.load_diffusion_model()
    pipe.configure_adapter(adapter)
    net = pipe.diffusion_model.to('cuda')
    if args.compute_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # A precision control must exclude reduced-precision fused SDPA kernels.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        import comfy.ops
        # ComfyUI's wrapper re-enables its own kernel priority; use plain SDPA
        # only inside this diagnostic process to make the control truly FP32.
        comfy.ops.scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
        # Never cast FP8 codes: only the arithmetic/ordinary parameters change.
        for p in net.parameters():
            if p.is_floating_point() and p.dtype != torch.float8_e4m3fn:
                p.data = p.data.float()
    net.eval()  # dropout disabled; autograd remains enabled
    params = {n: p for n,p in net.named_parameters() if p.requires_grad}
    assert params and all('lora_' in n for n in params)
    assert all(not m.weight.requires_grad for m in net.modules() if isinstance(m, ScaledFP8Linear))
    # Nonzero B exposes gradients of both PEFT matrices. Frozen base is untouched.
    with torch.no_grad():
        for name, p in params.items():
            if 'lora_B' in name:
                p.normal_(std=.001)
    text_path = args.cache / 'ar_frames_1.587_1/text_embeddings_1'
    texts = [read_cache(text_path,i) for i in range(16)]
    first = texts[0]
    other = next(x for x in texts[1:] if x['text_embeds_0'].shape != first['text_embeds_0'].shape)
    text = [first['text_embeds_0'].to('cuda',compute_dtype),other['text_embeds_0'].to('cuda',compute_dtype)]
    masks = [first['attention_mask_0'].to('cuda'),other['attention_mask_0'].to('cuda')]
    context, mask = pipe.get_conds(dict(text_embeds_0=text, attention_mask_0=masks))
    # 512-pixel bucket; synthetic target/reference isolate batching from sample selection.
    target = torch.randn(2,16,1,args.latent_height,args.latent_width,device='cuda')
    reference = torch.randn_like(target)
    t = torch.full((2,),.63,device='cuda')
    layers = pipe.to_layers()
    dtype_trace = {}
    def trace(module, inputs, outputs):
        dtype_trace.setdefault('initial_outputs', [str(x.dtype) for x in outputs])
        dtype_trace.setdefault('autocast_enabled', torch.is_autocast_enabled('cuda'))
        dtype_trace.setdefault('autocast_dtype', str(torch.get_autocast_dtype('cuda')))
    layers[0].register_forward_hook(trace)

    def forward(x,ts,ctx,am,ref):
        h = checkpoint(layers[0],(x,ts,ctx,am,ref),use_reentrant=False)
        for layer in layers[1:-1]:
            h = checkpoint(layer,h,use_reentrant=False)
        return layers[-1](h)

    torch.cuda.reset_peak_memory_stats()
    batched = forward(target,t,context,mask,reference)
    expected_velocity = torch.randn_like(batched)
    (batched.float()-expected_velocity.float()).square().mean().backward()
    batched_out = batched.detach().float().cpu()
    batched_grad = {n:p.grad.detach().cpu().float() for n,p in params.items()}
    del batched
    net.zero_grad(set_to_none=True)
    outputs = []
    for i in range(2):
        ctx, am = ((context[i:i+1],mask[i:i+1]) if args.individual_padded
                   else (text[i][None],masks[i][None]))
        individual = forward(target[i:i+1],t[i:i+1],ctx,am,reference[i:i+1])
        outputs.append(individual.detach().float().cpu())
        ((individual.float()-expected_velocity[i:i+1].float()).square().mean()/2).backward()
        del individual
    individual_out = torch.cat(outputs)
    output_rel = float((batched_out-individual_out).norm()/individual_out.norm())
    err, denom, finite = 0., 0., True
    for name,p in params.items():
        grad = p.grad.detach().cpu().float()
        finite = finite and bool(torch.isfinite(grad).all())
        err += float((grad-batched_grad[name]).square().sum())
        denom += float(grad.square().sum())
    grad_rel = (err/denom)**.5
    result = dict(text_lengths=[len(x) for x in text], output_relative_l2=output_rel,
                  gradient_relative_l2=grad_rel, gradients_finite=finite,
                  trainable_tensors=len(params), peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                  fp8_scaled=True, peft=True, activation_checkpointing=True,
                  merge_adapters=config['model'].get('merge_adapters',[]),
                  individual_padded=args.individual_padded,
                  compute_dtype=str(compute_dtype), latent_shape=list(target.shape),
                  dtype_trace=dtype_trace,
                  precision='BF16 rounding is measured separately from FP32 structural parity',
                  passed=finite and (output_rel<1e-4 and grad_rel<1e-3 if args.compute_fp32
                                    else output_rel<.03 and grad_rel<.10))
    args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:
        raise RuntimeError('Real-path batching parity failed')


if __name__ == '__main__':
    main()
