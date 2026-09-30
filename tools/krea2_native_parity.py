#!/usr/bin/env python3
"""Forward parity: fork training path (krea2_native contract) vs STOCK ComfyUI Krea 2.

The stock side runs comfy.ldm.krea2.model.SingleStreamDiT._forward with
ref_latents=[ref] and ref_latents_method='index_timestep_zero' (or 'index'),
i.e. exactly what TextEncodeQwenImageEditPlus + FluxKontextMultiReferenceLatentMethod
drive in an unmodified ComfyUI. The fork side runs Krea2ReferenceInitialLayer
(position_mode='subject', offset 1) + blocks + Krea2ReferenceFinalLayer, i.e.
what train.py optimises. They live in different ComfyUI versions, so the test is
two processes sharing a dump:

  # 1) in a checkout of the ComfyUI you actually run (>= c9602625):
  python tools/krea2_native_parity.py stock --comfy /path/to/ComfyUI --out /tmp/k2p.pt [--method index_timestep_zero]
  # 2) in this repo (uses submodules/ComfyUI):
  python tools/krea2_native_parity.py fork --dump /tmp/k2p.pt

Text-encoder parity (GPU box, real Qwen3-VL weights, 2-3 PNG refs):
  python tools/krea2_native_parity.py te-stock --comfy /path/to/ComfyUI --te qwen3vl_4b_bf16.safetensors \
      --image a.png --prompt "the same girl now sits" --image b.png --prompt "..." --out /tmp/te.pt
  python tools/krea2_native_parity.py te-fork --te qwen3vl_4b_bf16.safetensors --dump /tmp/te.pt
  (must pass BEFORE caching; if it fails, bump submodules/ComfyUI to the ComfyUI you run)

Default is a tiny random Krea 2 (runs on CPU in seconds). With --weights you can
point both sides at the real checkpoint on the GPU box (bf16, same file) to
measure numerics end to end; expect relL2 ~1e-3 in bf16, ~1e-6 in fp32 tiny.
"""

import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TINY = dict(features=128, tdim=32, txtdim=32, heads=4, kvheads=2, multiplier=2, layers=2,
            txtlayers=3, txtheads=2, txtkvheads=2)


def _comfy_cpu_import(comfy_root):
    sys.path.insert(0, comfy_root)
    argv, sys.argv = sys.argv, [sys.argv[0]] + (['--cpu'] if not torch.cuda.is_available() else [])
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.model_management  # noqa: F401
    sys.argv = argv


def make_inputs(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 16, 12, 16, generator=g)             # target latent (odd-free)
    ref = torch.randn(1, 16, 18, 14, generator=g)           # reference on its OWN grid, different AR
    ctx = torch.randn(1, 7, cfg['txtlayers'] * cfg['txtdim'], generator=g)
    t = torch.tensor([0.63])
    return x, ref, ctx, t


def run_stock(args):
    _comfy_cpu_import(args.comfy)
    import comfy.ops
    from comfy.ldm.krea2.model import SingleStreamDiT
    torch.manual_seed(0)
    net = SingleStreamDiT(**TINY, operations=comfy.ops.disable_weight_init).float().eval()
    with torch.no_grad():
        for p in net.parameters():
            p.normal_(0, 0.05)
    x, ref, ctx, t = make_inputs(TINY)
    with torch.no_grad():
        out = net._forward(x, t, ctx, ref_latents=[ref], ref_latents_method=args.method, transformer_options={})
        out_noref = net._forward(x, t, ctx, transformer_options={})
    torch.save({'state_dict': net.state_dict(), 'x': x, 'ref': ref, 'ctx': ctx, 't': t,
                'out': out, 'out_noref': out_noref, 'method': args.method, 'cfg': TINY}, args.out)
    print(f'stock: out {tuple(out.shape)}, |out-out_noref| mean {float((out - out_noref).abs().mean()):.4f} '
          f'(reference is used) -> {args.out}')


def run_fork(args):
    sys.path.insert(0, ROOT)
    _comfy_cpu_import(os.path.join(ROOT, 'submodules', 'ComfyUI'))
    sys.path.pop(0)
    import utils.common  # noqa: F401
    import comfy.ops
    from comfy.ldm.krea2.model import SingleStreamDiT
    from models.krea2_reference import Krea2ReferenceInitialLayer, Krea2ReferenceFinalLayer

    dump = torch.load(args.dump)
    net = SingleStreamDiT(**dump['cfg'], operations=comfy.ops.disable_weight_init).float().eval()
    missing, unexpected = net.load_state_dict(dump['state_dict'], strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    class _NoOffload:
        def wait_for_block(self, i): pass
        def submit_move_blocks_forward(self, i): pass

    ref_mode = 'zero' if dump['method'] == 'index_timestep_zero' else 'target'
    init = Krea2ReferenceInitialLayer(net, position_mode='subject', reference_position_offset=1.0,
                                      reference_timestep_mode=ref_mode)
    final = Krea2ReferenceFinalLayer(net)
    x, ref, ctx, t = dump['x'], dump['ref'], dump['ctx'], dump['t']
    mask = torch.ones(1, ctx.shape[1], dtype=torch.long)
    with torch.no_grad():
        h = init((x.unsqueeze(2), t, ctx, mask, ref.unsqueeze(2)))
        combined, tf, tvec, freqs, amask, sizes = h
        for block in net.blocks:
            combined = block(combined, tvec, freqs, amask)
        out = final((combined, tf, tvec, freqs, amask, sizes))[:, :, 0]
    stock = dump['out']
    rel = float((out - stock).norm() / stock.norm())
    print(f'fork vs stock ({dump["method"]}): relL2 = {rel:.2e}')
    if rel > args.tol:
        raise SystemExit(f'PARITY FAILED (> {args.tol}): the training contract does not match stock ComfyUI')
    print('PARITY OK')


def _load_krea2_clip(te_path):
    import comfy.sd
    return comfy.sd.load_clip(ckpt_paths=[te_path], clip_type=comfy.sd.CLIPType.KREA2)


def run_te_stock(args):
    """Conditioning from the REAL node code of the ComfyUI you run."""
    _comfy_cpu_import(args.comfy)
    sys.path.insert(0, ROOT)
    from comfy_extras.nodes_qwen import TextEncodeQwenImageEditPlus
    from models.krea2_native import load_comfy_image  # PIL decode, same as the fork (avoid PyAV/JPEG drift)
    clip = _load_krea2_clip(args.te)
    out = []
    for image, prompt in zip(args.image, args.prompt):
        cond = TextEncodeQwenImageEditPlus.execute(clip, prompt, None, load_comfy_image(image)).args[0]
        out.append(cond[0][0].float().cpu())
    torch.save({'cond': out, 'image': args.image, 'prompt': args.prompt}, args.out)
    print(f'te-stock: {[tuple(c.shape) for c in out]} -> {args.out}')


def run_te_fork(args):
    sys.path.insert(0, ROOT)
    _comfy_cpu_import(os.path.join(ROOT, 'submodules', 'ComfyUI'))
    sys.path.pop(0)
    import utils.common  # noqa: F401
    from models.krea2_native import native_tokenize
    dump = torch.load(args.dump)
    clip = _load_krea2_clip(args.te)
    worst = 0.0
    for image, prompt, stock in zip(dump['image'], dump['prompt'], dump['cond']):
        o = clip.encode_from_tokens_scheduled(native_tokenize(clip, prompt, image, grounded=True))
        mine = o[0][0].float().cpu()
        if mine.shape != stock.shape:
            raise SystemExit(f'TE PARITY FAILED: shape {tuple(mine.shape)} != stock {tuple(stock.shape)}')
        rel = float((mine - stock).norm() / stock.norm())
        worst = max(worst, rel)
        print(f'{os.path.basename(image)}: relL2 {rel:.2e}')
    if worst > args.tol:
        raise SystemExit(f'TE PARITY FAILED (worst {worst:.2e} > {args.tol}): update submodules/ComfyUI '
                         'to the ComfyUI you run (Qwen3-VL DeepStack/MRoPE changed after 2026-06-23)')
    print('TE PARITY OK')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('stock')
    s.add_argument('--comfy', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--method', default='index_timestep_zero', choices=['index_timestep_zero', 'index'])
    f = sub.add_parser('fork')
    f.add_argument('--dump', required=True)
    f.add_argument('--tol', type=float, default=1e-4)
    ts = sub.add_parser('te-stock', help='real Qwen3-VL weights: conditioning from the stock node')
    ts.add_argument('--comfy', required=True)
    ts.add_argument('--te', required=True, help='qwen3vl_4b_bf16.safetensors')
    ts.add_argument('--image', action='append', required=True)
    ts.add_argument('--prompt', action='append', required=True)
    ts.add_argument('--out', required=True)
    tf = sub.add_parser('te-fork', help='same inputs through the training cache tokenization')
    tf.add_argument('--te', required=True)
    tf.add_argument('--dump', required=True)
    tf.add_argument('--tol', type=float, default=2e-2)
    args = ap.parse_args()
    {'stock': run_stock, 'fork': run_fork, 'te-stock': run_te_stock, 'te-fork': run_te_fork}[args.cmd](args)


if __name__ == '__main__':
    main()
