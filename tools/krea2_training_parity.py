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
    args = parser.parse_args()
    sys.path.insert(0, str(args.train_repo))
    sys.path.insert(0, str(args.train_repo / 'submodules/ComfyUI'))
    import torch
    import toml
    import utils.common
    utils.common.AUTOCAST_DTYPE = torch.bfloat16
    from models.krea2_native import Krea2NativePipeline
    from models.base import ScaledFP8Linear
    from torch.utils.checkpoint import checkpoint
    torch.manual_seed(76)
    config = toml.load(args.config)
    config['model']['dtype'] = torch.bfloat16
    adapter = dict(config['adapter'], alpha=config['adapter']['rank'], dropout=0., dtype=torch.bfloat16)
    pipe = Krea2NativePipeline(config)
    pipe.load_diffusion_model()
    pipe.configure_adapter(adapter)
    net = pipe.diffusion_model.to('cuda')
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
    text = [first['text_embeds_0'].to('cuda'),other['text_embeds_0'].to('cuda')]
    masks = [first['attention_mask_0'].to('cuda'),other['attention_mask_0'].to('cuda')]
    context, mask = pipe.get_conds(dict(text_embeds_0=text, attention_mask_0=masks))
    # 512-pixel bucket; synthetic target/reference isolate batching from sample selection.
    target = torch.randn(2,16,1,50,80,device='cuda')
    reference = torch.randn_like(target)
    t = torch.full((2,),.63,device='cuda')
    layers = pipe.to_layers()

    def forward(x,ts,ctx,am,ref):
        h = checkpoint(layers[0],(x,ts,ctx,am,ref),use_reentrant=False)
        for layer in layers[1:-1]:
            h = checkpoint(layer,h,use_reentrant=False)
        return layers[-1](h)

    torch.cuda.reset_peak_memory_stats()
    batched = forward(target,t,context,mask,reference)
    batched.float().square().mean().backward()
    batched_out = batched.detach().float().cpu()
    batched_grad = {n:p.grad.detach().cpu().float() for n,p in params.items()}
    del batched
    net.zero_grad(set_to_none=True)
    outputs = []
    for i in range(2):
        individual = forward(target[i:i+1],t[i:i+1],text[i][None],masks[i][None],reference[i:i+1])
        outputs.append(individual.detach().float().cpu())
        (individual.float().square().mean()/2).backward()
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
                  precision='BF16: batching may change rounding; FP32 regression is tested separately',
                  passed=finite and output_rel<.03 and grad_rel<.10)
    args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    if not result['passed']:
        raise RuntimeError('Real-path batching parity failed')


if __name__ == '__main__':
    main()
